#!/usr/bin/env python3
"""
BlinkThink Dual-Process Benchmark Study Runner.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from blinkthink.benchmark import BenchmarkReport, compute_ece
from blinkthink.router import BlinkThinkRouter
from blinkthink.tuner import BlinkThinkTuner
from blinkthink.schemas import DecisionRequest, DecisionResult, DecisionType

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_study")


def sanitize_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.lower()).strip("_")


def load_curated_dataset() -> Tuple[List[Tuple[DecisionRequest, Any]], List[Tuple[DecisionRequest, Any]]]:
    val_set = [
        (DecisionRequest(prompt="What is the capital of France?", decision_type=DecisionType.CHOICE, options=["London", "Berlin", "Paris", "Rome"]), "Paris"),
        (DecisionRequest(prompt="Is 17 a prime number?", decision_type=DecisionType.CHOICE, options=["Yes", "No"]), "Yes"),
        (DecisionRequest(prompt="Which planet is known as the Red Planet?", decision_type=DecisionType.CHOICE, options=["Venus", "Mars", "Jupiter", "Saturn"]), "Mars"),
    ]
    test_set = [
        (DecisionRequest(prompt="What is the chemical symbol for Gold?", decision_type=DecisionType.CHOICE, options=["Ag", "Fe", "Au", "Cu"]), "Au"),
        (DecisionRequest(prompt="Can sound travel through an absolute vacuum?", decision_type=DecisionType.CHOICE, options=["Yes", "No"]), "No"),
        (DecisionRequest(prompt="What is the largest ocean on Earth?", decision_type=DecisionType.CHOICE, options=["Atlantic", "Indian", "Arctic", "Pacific"]), "Pacific"),
    ]
    return val_set, test_set


def load_hf_benchmark(
    name: str,
    hf_cache_dir: Path,
    val_size: int = 50,
    test_size: int = 200,
    seed: int = 42,
) -> Tuple[List[Tuple[DecisionRequest, Any]], List[Tuple[DecisionRequest, Any]]]:
    from datasets import load_dataset

    samples: List[Tuple[DecisionRequest, Any]] = []
    logger.info("Loading '%s' benchmark into cache (%s)...", name, hf_cache_dir)
    total_needed = val_size + test_size

    if name.lower() == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=str(hf_cache_dir)).shuffle(seed=seed)
        ds = ds.select(range(min(len(ds), total_needed)))
        for row in ds:
            opts = [str(opt) for opt in row["choices"]]
            samples.append((
                DecisionRequest(prompt=row["question"], decision_type=DecisionType.CHOICE, options=opts),
                opts[row["answer"]]
            ))
    elif name.lower() in ("arc", "arc_challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=str(hf_cache_dir)).shuffle(seed=seed)
        ds = ds.select(range(min(len(ds), total_needed * 2)))
        for row in ds:
            choices = row["choices"]
            labels, texts = choices["label"], choices["text"]
            if row["answerKey"] in labels:
                samples.append((
                    DecisionRequest(prompt=row["question"], decision_type=DecisionType.CHOICE, options=texts),
                    texts[labels.index(row["answerKey"])]
                ))
            if len(samples) >= total_needed:
                break
    elif name.lower() == "boolq":
        ds = load_dataset("google/boolq", split="validation", cache_dir=str(hf_cache_dir)).shuffle(seed=seed)
        ds = ds.select(range(min(len(ds), total_needed)))
        for row in ds:
            samples.append((
                DecisionRequest(
                    prompt=f"Context: {row['passage']}\n\nQuestion: {row['question']}?",
                    decision_type=DecisionType.CHOICE,
                    options=["Yes", "No"],
                ),
                "Yes" if row["answer"] else "No"
            ))
    else:
        raise ValueError(f"Unknown benchmark: '{name}'. Choose from ['mmlu', 'arc', 'boolq', 'curated'].")

    samples = samples[:total_needed]
    return samples[:val_size], samples[val_size:]


class CheckpointableEvaluator:
    def __init__(self, router: BlinkThinkRouter, checkpoint_dir: Path):
        self.router = router
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_configuration(
        self,
        dataset: List[Tuple[DecisionRequest, Any]],
        config_name: str,
        n_blinks: int = 2,
        threshold: float = 0.35,
        force_mode: Optional[str] = None,
    ) -> BenchmarkReport:
        slug = sanitize_slug(config_name)
        jsonl_path = self.checkpoint_dir / f"{slug}_samples.jsonl"

        cached_records: List[Dict[str, Any]] = []
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cached_records.append(json.loads(line))
            logger.info("[%s] Resuming from checkpoint: %d/%d samples loaded.", config_name[:32], len(cached_records), len(dataset))

        start_idx = len(cached_records)
        all_records = list(cached_records)

        if start_idx >= len(dataset):
            return self._build_report_from_records(all_records, config_name, n_blinks, threshold)

        jsonl_file = open(jsonl_path, "a", encoding="utf-8")
        pbar = tqdm(total=len(dataset), initial=start_idx, desc=f"Eval {config_name[:26]}", unit="sample", dynamic_ncols=True)

        running_correct = sum(1 for r in cached_records if r["is_correct"])
        running_esc = sum(1 for r in cached_records if r["system_used"] in ("system_2", "system_1_fallback"))

        try:
            for idx in range(start_idx, len(dataset)):
                req, y_true = dataset[idx]
                res = self.router.decide(req, n_blinks=n_blinks, threshold=threshold, force_mode=force_mode)

                is_correct = (res.decision == y_true)
                is_escalated = res.system_used in ("system_2", "system_1_fallback")
                running_correct += (1 if is_correct else 0)
                running_esc += (1 if is_escalated else 0)

                # Statistical fix: Top probability corresponds to the chosen decision
                top_prob = (
                    res.probabilities.get(res.decision, res.confidence)
                    if res.probabilities
                    else res.confidence
                )

                rec = {
                    "index": idx,
                    "prompt": req.prompt,
                    "decision_type": req.decision_type.value,
                    "options": req.options,
                    "ground_truth": y_true,
                    "predicted": res.decision,
                    "is_correct": bool(is_correct),
                    "system_used": res.system_used,
                    "confidence": float(res.confidence),
                    "probabilities": res.probabilities,
                    "top_probability": float(top_prob),
                    "latency_ms": float(res.latency_ms),
                    "drift": res.telemetry.drift if res.telemetry else 0.0,
                    "consensus_rate": res.telemetry.consensus_rate if res.telemetry else 1.0,
                    "reasoning_trace": res.reasoning_trace,
                }
                jsonl_file.write(json.dumps(rec) + "\n")
                jsonl_file.flush()
                all_records.append(rec)

                pbar.set_postfix({
                    "Acc": f"{(running_correct / len(all_records)) * 100:.1f}%",
                    "Esc": f"{(running_esc / len(all_records)) * 100:.1f}%",
                    "Lat": f"{res.latency_ms:.0f}ms",
                })
                pbar.update(1)
        finally:
            jsonl_file.close()
            pbar.close()

        return self._build_report_from_records(all_records, config_name, n_blinks, threshold)

    def _build_report_from_records(
        self, records: List[Dict[str, Any]], config_name: str, n_blinks: int, threshold: float
    ) -> BenchmarkReport:
        if not records:
            return BenchmarkReport(config_name, n_blinks, threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        correct = np.array([1 if r["is_correct"] else 0 for r in records])
        latencies = np.array([r["latency_ms"] for r in records])
        escalated = np.array([1 if r["system_used"] in ("system_2", "system_1_fallback") else 0 for r in records])
        calib_probs = np.array([r.get("top_probability", r["confidence"]) for r in records])
        drifts = np.array([r.get("drift", 0.0) for r in records])
        consensuses = np.array([r.get("consensus_rate", 1.0) for r in records])

        return BenchmarkReport(
            config_name=config_name,
            n_blinks=n_blinks,
            threshold=threshold,
            accuracy=float(np.mean(correct)),
            avg_latency_ms=float(np.mean(latencies)),
            system2_escalation_pct=float(np.mean(escalated) * 100.0),
            expected_calibration_error=compute_ece(calib_probs, correct),
            avg_drift=float(np.mean(drifts)),
            avg_consensus=float(np.mean(consensuses)),
        )


def print_results_table(reports: List[BenchmarkReport]):
    sep_len = 110
    print("\n" + "=" * sep_len)
    print(f"{'Configuration':<42} | {'Acc (%)':<8} | {'Lat (ms)':<10} | {'Sys2 Esc (%)':<12} | {'ECE':<8} | {'Drift':<8}")
    print("-" * sep_len)
    for r in reports:
        print(
            f"{r.config_name:<42} | "
            f"{r.accuracy * 100:<8.1f} | "
            f"{r.avg_latency_ms:<10.1f} | "
            f"{r.system2_escalation_pct:<12.1f} | "
            f"{r.expected_calibration_error:<8.4f} | "
            f"{r.avg_drift:<8.4f}"
        )
    print("=" * sep_len + "\n")


def save_study_artifacts(reports: List[BenchmarkReport], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dict = [asdict(r) for r in reports]
    with open(output_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(reports_dict, f, indent=2)
    with open(output_dir / "summary_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reports_dict[0].keys()))
        writer.writeheader()
        writer.writerows(reports_dict)
    logger.info("Artifacts saved to: %s", output_dir.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="BlinkThink Dual-Process Study Runner")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["curated", "mmlu", "arc", "boolq"])
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tune_objective",
        type=str,
        default="balanced",
        choices=["balanced", "peak_synergy", "knee_point", "target_accuracy", "max_escalation"],
    )
    parser.add_argument("--target_value", type=float, default=0.70)
    parser.add_argument("--clean_checkpoints", action="store_true")
    parser.add_argument("--force_retune", action="store_true")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="auto",
        choices=["auto", "chat", "completion"],
        help="Prompt formatting mode for System 1: auto-detect, chat template, or raw completion.",
    )
    parser.add_argument(
        "--confidence_metric",
        type=str,
        default="composite",
        choices=["composite", "margin", "msp"],
        help="System 1 confidence metric formulation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_slug = sanitize_slug(args.model_id)
    base_cache_dir = Path(args.cache_dir).resolve()
    hf_cache_dir = base_cache_dir / "huggingface"

    obj_slug = f"{args.tune_objective}_{args.target_value:.2f}" if args.tune_objective in ("target_accuracy", "max_escalation") else args.tune_objective
    if args.confidence_metric != "composite":
        obj_slug = f"{obj_slug}_{args.confidence_metric}"
    experiments_dir = base_cache_dir / "experiments" / args.dataset / model_slug / obj_slug
    checkpoints_dir = experiments_dir / "checkpoints"

    if args.clean_checkpoints and checkpoints_dir.exists():
        logger.info("Purging previous checkpoints in: %s", checkpoints_dir)
        shutil.rmtree(checkpoints_dir)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")

    logger.info("=== Starting BlinkThink Evaluation ===")
    logger.info("Model:     %s", args.model_id)
    logger.info("Dataset:   %s (Val: %d, Test: %d)", args.dataset.upper(), args.val_size, args.test_size)
    logger.info("Objective: %s (Target Value: %.2f)", args.tune_objective, args.target_value)
    logger.info("Artifacts: %s", experiments_dir)

    if args.dataset == "curated":
        val_set, test_set = load_curated_dataset()
    else:
        val_set, test_set = load_hf_benchmark(
            args.dataset,
            hf_cache_dir=hf_cache_dir / "datasets",
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )

    use_cuda = torch.cuda.is_available()
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if use_cuda
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=str(hf_cache_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        attn_implementation="sdpa" if use_cuda else "eager",
        cache_dir=str(hf_cache_dir),
    )

    router = BlinkThinkRouter(
        model,
        tokenizer,
        threshold=0.35,
        prompt_mode=args.prompt_mode,
        confidence_metric=args.confidence_metric,
    )
    evaluator = CheckpointableEvaluator(router, checkpoint_dir=checkpoints_dir)
    tuner = BlinkThinkTuner(router)

    s2_checkpoint_files = list(checkpoints_dir.glob("*sys_2*.jsonl")) + list(checkpoints_dir.glob("*deliberate*.jsonl"))
    for s2_file in s2_checkpoint_files:
        logger.info("Pre-warming System 2 cache from: %s", s2_file.name)
        with open(s2_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    req = DecisionRequest(
                        prompt=rec["prompt"],
                        decision_type=DecisionType.CHOICE,
                        options=rec.get("options", ["Yes", "No"]),
                    )
                    cached_res = DecisionResult(
                        decision=rec["predicted"],
                        confidence=rec["confidence"],
                        system_used=rec["system_used"],
                        probabilities=rec.get("probabilities", {}),
                        latency_ms=rec["latency_ms"],
                        reasoning_trace=rec.get("reasoning_trace"),
                    )
                    router.think_engine.warm_cache(req, cached_res)

    default_temps = router.calibrator.to_dict()

    logger.info("PHASE 0: Parameter fitting on validation split")
    tuned_config_file = experiments_dir / "tuned_config.json"
    optimal_taus: Dict[int, float] = {}

    if tuned_config_file.exists() and not args.force_retune:
        logger.info("Loading cached tuning parameters from: %s", tuned_config_file)
        with open(tuned_config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "temperatures" in cfg:
            router.calibrator.load_dict(cfg["temperatures"])
        if "optimal_thresholds" in cfg:
            optimal_taus = {int(k): float(v) for k, v in cfg["optimal_thresholds"].items()}
        else:
            legacy_tau = cfg.get("threshold", 0.35)
            optimal_taus = {1: legacy_tau, 2: legacy_tau, 4: legacy_tau}
    else:
        logger.info("Fitting temperature and optimizing tau* for N=1 (Objective: %s)...", args.tune_objective)
        rep1 = tuner.fit(
            val_set,
            n_blinks=1,
            tune_temperatures=True,
            objective=args.tune_objective,
            target_value=args.target_value,
        )
        calibrated_temps = router.calibrator.to_dict()
        optimal_taus[1] = rep1.optimal_threshold

        for n in [2, 4]:
            logger.info("Optimizing threshold tau* for N=%d (Objective: %s)...", n, args.tune_objective)
            rep = tuner.fit(
                val_set,
                n_blinks=n,
                tune_temperatures=False,
                objective=args.tune_objective,
                target_value=args.target_value,
            )
            optimal_taus[n] = rep.optimal_threshold

        save_payload = {
            "threshold": optimal_taus.get(2, 0.35),
            "n_blinks": 2,
            "strict_consensus": router.strict_consensus,
            "temperatures": calibrated_temps,
            "optimal_thresholds": optimal_taus,
            "model_id": args.model_id,
            "dataset": args.dataset,
            "objective": args.tune_objective,
            "target_value": args.target_value,
            "confidence_metric": args.confidence_metric,
        }
        with open(tuned_config_file, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=2)

    reports: List[BenchmarkReport] = []

    logger.info("PHASE 1: Evaluating boundary baselines")
    router.calibrator.load_dict(default_temps)
    reports.append(evaluator.evaluate_configuration(
        test_set, config_name="1. Baseline: Sys 1 Only (N=1, Default T)", n_blinks=1, force_mode="blink_only"
    ))
    reports.append(evaluator.evaluate_configuration(
        test_set, config_name="2. Baseline: Deliberate (S1 Anchor + CoT)", force_mode="deliberate"
    ))

    logger.info("PHASE 2: Evaluating S1 multi-blink ablation with calibrated T*")
    router.load_config(str(tuned_config_file))
    for n in [1, 2, 4]:
        reports.append(evaluator.evaluate_configuration(
            test_set, config_name=f"3. S1 Pure Ablation: N={n} (Calibrated T*)", n_blinks=n, force_mode="blink_only"
        ))

    logger.info("PHASE 3: Evaluating fixed-threshold routing (tau=0.35)")
    for n in [1, 2, 4]:
        reports.append(evaluator.evaluate_configuration(
            test_set, config_name=f"4. BlinkThink: N={n} (Fixed tau=0.35)", n_blinks=n, threshold=0.35
        ))

    logger.info("PHASE 4: Evaluating calibrated dual-process routing with tuned tau*_N")
    for n in [1, 2, 4]:
        tau_n = optimal_taus.get(n, 0.35)
        reports.append(evaluator.evaluate_configuration(
            test_set, config_name=f"5. BlinkThink: N={n} (Tuned tau={tau_n:.2f})", n_blinks=n, threshold=tau_n
        ))

    print_results_table(reports)
    save_study_artifacts(reports, experiments_dir)


if __name__ == "__main__":
    main()
