#!/usr/bin/env python3
"""
BlinkThink Multi-Model & Multi-Objective Matrix Runner & Aggregator.

Usage Examples:
  # 1. Run full matrix for default models (google/gemma-4-12B & google/gemma-4-12B-it):
  python benchmarks/run_study_matrix.py --run --dataset curated

  # 2. Run for a custom list of models:
  python benchmarks/run_study_matrix.py --run --models google/gemma-4-12B google/gemma-4-12B-it --dataset mmlu

  # 3. View / compare all existing results in cache:
  python benchmarks/run_study_matrix.py
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

# Ensure project root is resolved properly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("study_matrix")

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-4B",
    "google/gemma-4-12B-it",
]


def parse_args():
    parser = argparse.ArgumentParser(description="BlinkThink Multi-Model Study Matrix")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the benchmark matrix across models and tuning objectives.",
    )
    parser.add_argument(
        "--models",
        "--model_ids",
        dest="models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of Hugging Face model IDs to evaluate (default: google/gemma-4-12B google/gemma-4-12B-it)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="Dataset to inspect/run ('curated', 'mmlu', 'arc', 'boolq', or 'auto')",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["balanced", "peak_synergy", "knee_point"],
        help="Tuning objectives to execute when --run is enabled",
    )
    parser.add_argument("--val_size", type=int, default=50, help="Validation split size for tuning")
    parser.add_argument("--test_size", type=int, default=200, help="Test split size for evaluation")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(PROJECT_ROOT / "cache"),
        help="Base cache directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Display all configurations (not just key dual-process configs)",
    )
    parser.add_argument(
        "--clean_checkpoints",
        action="store_true",
        help="Wipe previous checkpoints before running",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="auto",
        choices=["auto", "chat", "completion"],
        help="System 1 prompt template mode",
    )
    parser.add_argument(
        "--confidence_metric",
        type=str,
        default="composite",
        choices=["composite", "margin", "msp"],
        help="System 1 confidence metric formulation",
    )
    return parser.parse_args()


def execute_matrix(args, cache_dir: Path):
    """Executes benchmarks/run_study.py across all combinations of (models x objectives)."""
    dataset = "mmlu" if args.dataset == "auto" else args.dataset
    run_study_script = PROJECT_ROOT / "benchmarks" / "run_study.py"

    total_runs = len(args.models) * len(args.objectives)
    current_run = 0

    logger.info("==================================================")
    logger.info("STARTING STUDY MATRIX")
    logger.info("Models (%d):   %s", len(args.models), ", ".join(args.models))
    logger.info("Objectives:   %s", ", ".join(args.objectives))
    logger.info("Dataset:      %s", dataset.upper())
    logger.info("Total Runs:   %d", total_runs)
    logger.info("==================================================")

    for model_id in args.models:
        for obj in args.objectives:
            current_run += 1
            logger.info(
                "\n>>> [%d/%d] Model: '%s' | Objective: '%s' <<<",
                current_run,
                total_runs,
                model_id,
                obj,
            )

            cmd = [
                sys.executable,
                str(run_study_script),
                "--model_id", model_id,
                "--dataset", dataset,
                "--val_size", str(args.val_size),
                "--test_size", str(args.test_size),
                "--cache_dir", str(cache_dir),
                "--tune_objective", obj,
                "--prompt_mode", args.prompt_mode,
                "--confidence_metric", args.confidence_metric,
            ]
            if args.clean_checkpoints and current_run == 1:
                cmd.append("--clean_checkpoints")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(
                    "Run failed for model '%s' with objective '%s' (exit code %d)",
                    model_id,
                    obj,
                    result.returncode,
                )
                sys.exit(result.returncode)


def collect_reports(target_dir: Path, show_all: bool = False) -> List[Dict[str, Any]]:
    records = []
    if not target_dir.exists():
        return records

    for summary_path in target_dir.rglob("summary_report.json"):
        parent_dir = summary_path.parent  # obj_slug
        config_path = parent_dir / "tuned_config.json"
        metadata = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                reports = json.load(f)
        except Exception:
            continue

        for r in reports:
            is_key_config = "5. BlinkThink: N=2" in r["config_name"] or "Baseline" in r["config_name"]
            if show_all or is_key_config:
                records.append({
                    "Dataset": metadata.get("dataset", summary_path.parents[2].name),
                    "Model": metadata.get("model_id", parent_dir.parent.name),
                    "Objective": metadata.get("objective", parent_dir.name),
                    "Config": r["config_name"],
                    "Acc (%)": round(r["accuracy"] * 100, 2),
                    "Latency (ms)": round(r["avg_latency_ms"], 1),
                    "Sys2 Esc (%)": round(r["system2_escalation_pct"], 1),
                    "ECE": round(r["expected_calibration_error"], 4),
                    "Tau*": r["threshold"],
                })
    return records


def print_table(records: List[Dict[str, Any]]):
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df = df.sort_values(by=["Dataset", "Model", "Acc (%)"], ascending=[True, True, False])
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.to_string(index=False))
    except ImportError:
        if not records:
            return
        keys = list(records[0].keys())
        col_widths = {k: max(len(k), max(len(str(r[k])) for r in records)) for k in keys}
        header = " | ".join(f"{k:<{col_widths[k]}}" for k in keys)
        sep = "-" * len(header)
        print(header)
        print(sep)
        for row in sorted(records, key=lambda x: (x.get("Dataset", ""), x.get("Model", ""), -x.get("Acc (%)", 0))):
            print(" | ".join(f"{str(row[k]):<{col_widths[k]}}" for k in keys))


def main():
    args = parse_args()
    base_cache = Path(args.cache_dir).resolve()
    experiments_root = base_cache / "experiments"

    # Step 1: Run matrix if --run is specified
    if args.run:
        execute_matrix(args, base_cache)

    # Step 2: Auto-detect dataset directories
    target_dir = experiments_root
    dataset_filter = args.dataset.lower()

    if dataset_filter == "auto":
        if experiments_root.exists():
            available = [p.name for p in experiments_root.iterdir() if p.is_dir()]
            if len(available) == 1:
                target_dir = experiments_root / available[0]
                dataset_filter = available[0]
            elif len(available) > 1:
                target_dir = experiments_root
                dataset_filter = "all"
        else:
            available = []
    elif dataset_filter != "all":
        target_dir = experiments_root / dataset_filter

    records = collect_reports(target_dir, show_all=args.all)

    # Step 3: Handle empty state with helpful instructions
    if not records:
        print("\n" + "=" * 70)
        print("No BlinkThink study reports found!")
        print(f"Checked directory: {target_dir}")
        print("=" * 70)
        print("\nTo run the matrix across default models (gemma4-12b and gemma4-12b-it):\n")
        print("  python benchmarks/run_study_matrix.py --run --dataset curated\n")
        print("Or with custom models:\n")
        print("  python benchmarks/run_study_matrix.py --run --models google/gemma-4-12B google/gemma-4-12B-it --dataset mmlu\n")
        return

    title_dataset = dataset_filter.upper() if dataset_filter != "auto" else "ALL"
    print(f"\n=== BlinkThink {title_dataset} Multi-Model Comparative Matrix ===")
    print_table(records)
    print()


if __name__ == "__main__":
    main()
