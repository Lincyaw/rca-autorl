"""Generate comparison tables across experiment runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# Metrics to show in comparison (order matters for display)
DISPLAY_METRICS = [
    "reward_mean",
    "eval_reward_mean",
    "success_rate",
    "eval_success_rate",
    "avg_trajectory_turns",
    "truncation_rate",
    "training_time_hours",
]


def load_run(run_dir: Path) -> dict:
    """Load run metadata and metrics."""
    meta = {}
    if yaml is not None and (run_dir / "meta.yaml").exists():
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}

    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = {k: v for k, v in raw.items() if v is not None and not k.startswith("_")}

    return {
        "run_id": meta.get("run_id", run_dir.name),
        "status": meta.get("status", "unknown"),
        "meta": meta,
        "metrics": metrics,
    }


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) >= 1000:
            return f"{value:.4g}"
        return f"{value:.4f}"
    return str(value)


def compare(run_dirs: list[Path], metrics_to_show: list[str] | None = None) -> str:
    """Generate markdown comparison table."""
    if metrics_to_show is None:
        metrics_to_show = DISPLAY_METRICS

    runs = [load_run(d) for d in run_dirs]

    # Collect all available metrics across runs
    available = set()
    for run in runs:
        available.update(run["metrics"].keys())
    display = [m for m in metrics_to_show if m in available]
    if not display:
        display = sorted(available)[:8]  # fallback: first 8 alphabetically

    # Header
    header = "| Run | Status | " + " | ".join(display) + " |"
    separator = "|-----|--------|" + "|".join("-" * (len(m) + 2) for m in display) + "|"

    rows = []
    for run in runs:
        values = [format_value(run["metrics"].get(m)) for m in display]
        row = f"| {run['run_id']} | {run['status']} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join([header, separator] + rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    parser.add_argument("--metrics", nargs="*", help="Specific metrics to show")
    parser.add_argument("--output", help="Write to file instead of stdout")
    args = parser.parse_args()

    dirs = [Path(d).resolve() for d in args.run_dirs]
    table = compare(dirs, args.metrics)

    if args.output:
        Path(args.output).write_text(table + "\n", encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
