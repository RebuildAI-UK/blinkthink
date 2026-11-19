import argparse
import json
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BlinkThink Study Reports")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Base cache directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        help="Dataset directory to inspect (e.g. 'mmlu', 'arc', 'boolq', or 'all')",
    )
    return parser.parse_args()


def collect_reports(base_dir: Path) -> pd.DataFrame:
    records = []
    if not base_dir.exists():
        return pd.DataFrame(records)

    for summary_path in base_dir.rglob("summary_report.json"):
        parent_dir = summary_path.parent
        config_path = parent_dir / "tuned_config.json"
        metadata = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        with open(summary_path, "r", encoding="utf-8") as f:
            reports = json.load(f)

        for r in reports:
            if "5. BlinkThink: N=" in r["config_name"] or "Baseline" in r["config_name"]:
                records.append({
                    "Model": metadata.get("model_id", parent_dir.parent.name),
                    "Objective": metadata.get("objective", parent_dir.name),
                    "Config": r["config_name"],
                    "Acc (%)": round(r["accuracy"] * 100, 2),
                    "Latency (ms)": round(r["avg_latency_ms"], 1),
                    "Sys2 Esc (%)": round(r["system2_escalation_pct"], 1),
                    "ECE": round(r["expected_calibration_error"], 4),
                    "Optimal Tau": r["threshold"],
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    args = parse_args()
    base_cache = Path(args.cache_dir)
    target_dir = (
        base_cache / "experiments"
        if args.dataset.lower() == "all"
        else base_cache / "experiments" / args.dataset
    )

    df = collect_reports(target_dir)
    if df.empty:
        print(f"No summary reports found in: {target_dir}")
        print("Run a study first (e.g., 'python benchmarks/run_study.py --dataset "
              f"{args.dataset}')!")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        title = f"{args.dataset.upper()} " if args.dataset.lower() != "all" else ""
        print(f"\n=== BlinkThink {title}Cross-Model & Objective Comparison ===")
        print(df.sort_values(by=["Model", "Acc (%)"], ascending=[True, False]).to_string(index=False))
