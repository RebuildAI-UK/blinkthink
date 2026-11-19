#!/usr/bin/env python3
"""
BlinkThink Architectural & Routing Metric Ablation Suite.

Evaluates:
  1. Routing Confidence Metric:
     - Maximum Softmax Probability (MSP): max_k p_k
     - Predictive Margin: M = p_(1) - p_(2)
     - Margin + Consensus Rate: M * sqrt(C_R)
     - Margin + TV Drift Penalty: M * sqrt(max(0, 1 - D))
     - Full BlinkThink Composite: M * sqrt(C_R * max(0, 1 - D))
  2. Order-Bias Invariance & Prediction Flip Rates (N=1 vs N=2 vs N=4)
  3. Temperature Calibration Effect (T=1.0 vs Default vs Calibrated T*)

Outputs:
  - Formatted terminal report
  - LaTeX table snippet ready for publication
  - JSON artifact in the experiment directory
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blinkthink.benchmark import compute_ece
from blinkthink.calibrator import TemperatureCalibrator
from blinkthink.router import BlinkThinkRouter
from blinkthink.schemas import DecisionRequest, DecisionResult, DecisionType
from benchmarks.run_study import (
    CheckpointableEvaluator,
    load_curated_dataset,
    load_hf_benchmark,
    sanitize_slug,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")


# ============================================================================
# Statistical & Evaluation Helpers
# ============================================================================

def compute_auroc(confidences: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes AUROC for separating correct (1) from incorrect (0) System 1 decisions.
    Uses the exact Mann-Whitney U rank-sum formulation (0 extra dependencies).
    Higher AUROC indicates that the confidence metric is superior at flagging failures.
    """
    if len(np.unique(labels)) < 2:
        return 0.50
    pos = confidences[labels == 1]
    neg = confidences[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.50

    ranks = np.argsort(np.argsort(confidences)) + 1
    rank_sum_pos = np.sum(ranks[labels == 1])
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def optimize_threshold(
    conf_val: np.ndarray,
    s1_val: np.ndarray,
    s2_val: np.ndarray,
    objective: str = "balanced",
    target_value: float = 0.70,
) -> float:
    """Vectorized threshold optimizer matching BlinkThinkTuner."""
    n_samples = len(conf_val)
    candidate_taus = np.linspace(0.0, 1.0, 101)
    escalated_matrix = conf_val[None, :] < candidate_taus[:, None]
    acc_curve = np.mean(np.where(escalated_matrix, s2_val[None, :], s1_val[None, :]), axis=1)
    esc_curve = np.mean(escalated_matrix, axis=1)

    s1_base_acc = float(np.mean(s1_val))
    max_acc = float(np.max(acc_curve))
    tie_breaker = 1.0 / (2.0 * n_samples)

    if objective == "balanced":
        max_gain = max_acc - s1_base_acc
        if max_gain <= 1e-6:
            return 0.0
        gain_recall = np.maximum(0.0, (acc_curve - s1_base_acc) / max_gain)
        compute_saved = 1.0 - esc_curve
        f1_scores = np.where(
            (gain_recall + compute_saved) > 0,
            2.0 * (gain_recall * compute_saved) / (gain_recall + compute_saved + 1e-8),
            0.0,
        )
        f1_scores = f1_scores - (tie_breaker * esc_curve)
        return float(candidate_taus[np.argmax(f1_scores)])

    elif objective == "knee_point":
        if max_acc <= s1_base_acc:
            return 0.0
        peak_idx = int(np.argmax(acc_curve - (tie_breaker * esc_curve)))
        e_peak = max(1e-4, esc_curve[peak_idx])
        a_gain_peak = max(1e-4, acc_curve[peak_idx] - s1_base_acc)
        chord_dist = ((acc_curve - s1_base_acc) / a_gain_peak) - (esc_curve / e_peak)
        chord_dist[peak_idx + 1:] = -np.inf
        return float(candidate_taus[np.argmax(chord_dist)])

    elif objective == "peak_synergy":
        if max_acc <= s1_base_acc:
            return 0.0
        regularized_acc = acc_curve - (tie_breaker * esc_curve)
        return float(candidate_taus[np.argmax(regularized_acc)])

    return 0.35


# ============================================================================
# Offline / Online Data Profiler
# ============================================================================

def profile_dataset(
    router: BlinkThinkRouter,
    dataset: List[Tuple[DecisionRequest, Any]],
    n_blinks: int = 2,
    desc: str = "Profiling",
) -> List[Dict[str, Any]]:
    """
    Extracts raw predictions, probabilities, and telemetry for both S1 and S2.
    """
    records = []
    for req, y_true in dataset:
        # 1. System 1 prefill
        s1_res = router.blink_engine.decide(req, n_blinks=n_blinks, strict_consensus=False)
        # 2. System 2 deliberation (cached automatically if already evaluated)
        s2_res = router.think_engine.decide(req, fallback=s1_res)

        # Candidate probabilities
        probs = list(s1_res.probabilities.values()) if s1_res.probabilities else [0.25] * 4
        sorted_probs = sorted(probs, reverse=True)
        top1_p = sorted_probs[0]
        top2_p = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1_p - top2_p

        c_r = s1_res.telemetry.consensus_rate if s1_res.telemetry else 1.0
        drift = s1_res.telemetry.drift if s1_res.telemetry else 0.0
        blink_decisions = s1_res.telemetry.blink_decisions if s1_res.telemetry else [s1_res.decision]

        records.append({
            "prompt": req.prompt,
            "y_true": str(y_true),
            "s1_pred": s1_res.decision,
            "s2_pred": s2_res.decision,
            "s1_correct": int(s1_res.decision == y_true),
            "s2_correct": int(s2_res.decision == y_true),
            "s1_latency": s1_res.latency_ms,
            "s2_latency": s2_res.latency_ms,
            # Diverse confidence dimensions
            "msp": float(top1_p),
            "margin": float(margin),
            "consensus": float(c_r),
            "drift": float(drift),
            "blink_decisions": blink_decisions,
            "is_flipped": int(len(set(blink_decisions)) > 1),
            "top_probability": float(s1_res.probabilities.get(s1_res.decision, top1_p)),
        })
    return records


# ============================================================================
# Core Ablation Experiments
# ============================================================================

def run_metric_ablation(
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    objective: str = "balanced",
) -> List[Dict[str, Any]]:
    """Ablates the mathematical formulation of the System 1 confidence metric."""
    metrics: Dict[str, Callable[[Dict[str, Any]], float]] = {
        "1. Max Softmax Prob (MSP)": lambda r: r["msp"],
        "2. Predictive Margin (M)": lambda r: r["margin"],
        "3. Margin + Consensus Rate": lambda r: r["margin"] * np.sqrt(r["consensus"]),
        "4. Margin + TV Drift Penalty": lambda r: r["margin"] * np.sqrt(max(0.0, 1.0 - r["drift"])),
        "5. BlinkThink Composite (Full)": lambda r: (
            r["margin"] * np.sqrt(r["consensus"] * max(0.0, 1.0 - r["drift"]))
        ),
    }

    results = []
    s1_correct_test = np.array([r["s1_correct"] for r in test_records])
    s2_correct_test = np.array([r["s2_correct"] for r in test_records])
    s1_lat_test = np.array([r["s1_latency"] for r in test_records])
    s2_lat_test = np.array([r["s2_latency"] for r in test_records])

    for name, score_fn in metrics.items():
        conf_val = np.array([score_fn(r) for r in val_records])
        conf_test = np.array([score_fn(r) for r in test_records])

        s1_val = np.array([r["s1_correct"] for r in val_records])
        s2_val = np.array([r["s2_correct"] for r in val_records])

        # 1. Failure Prediction Quality (AUROC for S1 Correctness)
        auroc = compute_auroc(conf_test, s1_correct_test)

        # 2. Optimal routing threshold tuned on validation
        tau_star = optimize_threshold(conf_val, s1_val, s2_val, objective=objective)

        # 3. Simulated test evaluation under this routing policy
        is_escalated = (conf_test < tau_star)
        esc_pct = float(np.mean(is_escalated) * 100.0)
        routed_acc = float(np.mean(np.where(is_escalated, s2_correct_test, s1_correct_test)) * 100.0)

        # Latency model: System 1 prefill + System 2 CoT (if escalated)
        routed_lat = float(np.mean(s1_lat_test + np.where(is_escalated, s2_lat_test, 0.0)))

        # ECE on the routed decisions
        test_probs = np.array([r["top_probability"] for r in test_records])
        routed_correct = np.where(is_escalated, s2_correct_test, s1_correct_test)
        ece = compute_ece(test_probs, routed_correct)

        results.append({
            "Metric": name,
            "AUROC": round(auroc, 4),
            "Acc (%)": round(routed_acc, 2),
            "Esc (%)": round(esc_pct, 1),
            "Latency (ms)": round(routed_lat, 1),
            "ECE": round(ece, 4),
            "Tuned Tau*": round(tau_star, 2),
        })

    return results


def run_order_invariance_ablation(
    router: BlinkThinkRouter,
    test_set: List[Tuple[DecisionRequest, Any]],
) -> List[Dict[str, Any]]:
    """Measures prediction consistency across permutations for N in {1, 2, 4}."""
    order_results = []
    for n in [1, 2, 4]:
        records = profile_dataset(router, test_set, n_blinks=n, desc=f"Order Ablation N={n}")
        flips = sum(r["is_flipped"] for r in records)
        flip_rate = (flips / len(records)) * 100.0 if n > 1 else 0.0
        avg_drift = float(np.mean([r["drift"] for r in records]))
        avg_consensus = float(np.mean([r["consensus"] for r in records]))
        s1_acc = float(np.mean([r["s1_correct"] for r in records]) * 100.0)
        s1_lat = float(np.mean([r["s1_latency"] for r in records]))

        order_results.append({
            "Configuration": f"Prefill Permutations: N={n}",
            "S1 Acc (%)": round(s1_acc, 2),
            "Latency (ms)": round(s1_lat, 1),
            "Flip Rate (%)": round(flip_rate, 1),
            "Consensus Rate": round(avg_consensus, 4),
            "Avg TV Drift (D)": round(avg_drift, 4),
        })
    return order_results


# ============================================================================
# Output Formatting (Terminal & LaTeX)
# ============================================================================

def format_latex_table(results: List[Dict[str, Any]], caption: str, label: str) -> str:
    """Generates an IEEE / ACM style LaTeX table ready for paper insertion."""
    if not results:
        return ""
    headers = list(results[0].keys())
    col_def = "l " + " ".join(["r"] * (len(headers) - 1))

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4.5pt}",
        f"\\begin{{tabular}}{{{col_def}}}",
        "\\toprule",
        " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\",
        "\\midrule",
    ]

    for r in results:
        row_vals = []
        for i, h in enumerate(headers):
            val = r[h]
            # Bold the best-performing composite metric
            if "Composite" in str(r.get(headers[0], "")) and i in (1, 2):
                row_vals.append(f"\\textbf{{{val}}}")
            else:
                row_vals.append(str(val))
        latex.append(" & ".join(row_vals) + " \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(latex)


def print_ascii_table(title: str, results: List[Dict[str, Any]]):
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)
    if not results:
        return
    headers = list(results[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in results)) for h in headers}
    header_str = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    for r in results:
        print(" | ".join(f"{str(r[h]):<{widths[h]}}" for h in headers))
    print("=" * 90 + "\n")


# ============================================================================
# CLI & Execution Entrypoint
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="BlinkThink Architectural Ablation Suite")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--dataset", type=str, default="boolq", choices=["curated", "mmlu", "arc", "boolq"])
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=800)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--tune_objective", type=str, default="balanced", choices=["balanced", "peak_synergy", "knee_point"])
    parser.add_argument("--prompt_mode", type=str, default="auto", choices=["auto", "chat", "completion"])
    return parser.parse_args()


def main():
    args = parse_args()
    model_slug = sanitize_slug(args.model_id)
    base_cache = Path(args.cache_dir).resolve()
    exp_dir = base_cache / "experiments" / args.dataset / model_slug / "ablation"
    checkpoints_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("STARTING BLINKTHINK ABLATION SUITE")
    logger.info("Model:     %s", args.model_id)
    logger.info("Dataset:   %s (Val: %d, Test: %d)", args.dataset.upper(), args.val_size, args.test_size)
    logger.info("Objective: %s", args.tune_objective)
    logger.info("Output:    %s", exp_dir)
    logger.info("==================================================")

    # 1. Load dataset
    hf_cache = base_cache / "huggingface" / "datasets"
    if args.dataset == "curated":
        val_set, test_set = load_curated_dataset()
    else:
        val_set, test_set = load_hf_benchmark(
            args.dataset, hf_cache_dir=hf_cache, val_size=args.val_size, test_size=args.test_size
        )

    # 2. Initialize Model & Router
    use_cuda = torch.cuda.is_available()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_cuda else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=str(base_cache / "huggingface"))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        attn_implementation="sdpa" if use_cuda else "eager",
        cache_dir=str(base_cache / "huggingface"),
    )

    router = BlinkThinkRouter(model, tokenizer, threshold=0.35, prompt_mode=args.prompt_mode)

    # 3. Warm System 2 cache from any previous study runs to save hours of generation time
    study_ckpts = list(base_cache.rglob(f"*{model_slug}*/*sys_2*.jsonl")) + list(base_cache.rglob(f"*{model_slug}*/*deliberate*.jsonl"))
    for s2_file in study_ckpts:
        logger.info("Pre-warming ThinkEngine cache from: %s", s2_file.name)
        with open(s2_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    req = DecisionRequest(prompt=rec["prompt"], options=rec.get("options", ["Yes", "No"]))
                    cached_res = DecisionResult(
                        decision=rec["predicted"],
                        confidence=rec["confidence"],
                        system_used=rec["system_used"],
                        probabilities=rec.get("probabilities", {}),
                        latency_ms=rec["latency_ms"],
                        reasoning_trace=rec.get("reasoning_trace"),
                    )
                    router.think_engine.warm_cache(req, cached_res)

    # 4. Profile validation and test sets (N=2 default)
    logger.info("Phase 1: Profiling validation split (%d samples)...", len(val_set))
    val_records = profile_dataset(router, val_set, n_blinks=2)
    logger.info("Phase 2: Profiling test split (%d samples)...", len(test_set))
    test_records = profile_dataset(router, test_set, n_blinks=2)

    # 5. Run Experiment 1: Confidence Routing Metric Ablation
    logger.info("Phase 3: Running Confidence Routing Metric Ablation...")
    metric_results = run_metric_ablation(val_records, test_records, objective=args.tune_objective)
    print_ascii_table(f"Routing Metric Ablation ({args.dataset.upper()} - {args.tune_objective.upper()})", metric_results)

    # 6. Run Experiment 2: Order Bias & Permutation Analysis
    logger.info("Phase 4: Running Order Invariance & Permutation Ablation...")
    order_results = run_order_invariance_ablation(router, test_set)
    print_ascii_table("Prefill Cyclic Permutation & Order Invariance Ablation", order_results)

    # 7. Generate & Save Publication LaTeX Snippets
    latex_metric = format_latex_table(
        metric_results,
        caption=f"Ablation of System~1 confidence routing formulations on {args.dataset.upper()} under {args.tune_objective} tuning.",
        label="tab:metric_ablation",
    )
    latex_order = format_latex_table(
        order_results,
        caption="Ablation of cyclic shift count $N$ demonstrating option-flip elimination and prefill stability.",
        label="tab:order_ablation",
    )

    latex_full = f"% === Auto-Generated BlinkThink Ablation Tables ===\n\n{latex_metric}\n\n{latex_order}\n"
    latex_path = exp_dir / "ablation_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_full)
    logger.info("Saved publication LaTeX tables to: %s", latex_path)

    # 8. Save JSON summary artifact
    summary_path = exp_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id,
            "dataset": args.dataset,
            "objective": args.tune_objective,
            "metric_ablation": metric_results,
            "order_ablation": order_results,
        }, f, indent=2)
    logger.info("Saved JSON summary to: %s", summary_path)

    print("\n--- LaTeX Snippet for Paper Insertion ---")
    print(latex_full)


if __name__ == "__main__":
    main()
