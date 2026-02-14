"""Compare experiment manifests side-by-side.

Usage:
    python -m benchmarks.compare \
        experiments/run_a/experiment_manifest.json \
        experiments/run_b/experiment_manifest.json
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.utils.experiment import load_experiment_manifest


def _get(d: Dict[str, Any], *keys, default: Any = "N/A"):
    """Nested dict lookup with fallback."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _fmt(val: Any) -> str:
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def build_comparison_table(manifests: List[Dict[str, Any]], labels: List[str]) -> str:
    """Build a markdown table comparing experiment manifests."""
    rows = [
        ("Avg step time (ms)", lambda m: _get(m, "summary", "avg_step_time_ms")),
        ("Median step time (ms)", lambda m: _get(m, "summary", "median_step_time_ms")),
        ("P95 step time (ms)", lambda m: _get(m, "summary", "p95_step_time_ms")),
        ("Peak GPU memory (MB)", lambda m: _get(m, "summary", "peak_gpu_memory_mb")),
        ("Avg samples/sec", lambda m: _get(m, "summary", "avg_samples_per_sec")),
        ("Avg tokens/sec", lambda m: _get(m, "summary", "avg_tokens_per_sec")),
        ("Total time (sec)", lambda m: _get(m, "summary", "total_training_time_sec")),
        ("GPU", lambda m: (
            _get(m, "system_info", "gpus", default=[{}])[0].get("name", "N/A")
            if _get(m, "system_info", "gpus", default=[]) else "N/A"
        )),
        ("DeepSpeed", lambda m: (
            _get(m, "config", "deepspeed", "enabled", default=False)
        )),
    ]

    # Header
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join("---" for _ in labels) + "|"
    lines = [header, sep]

    for label, extractor in rows:
        vals = [_fmt(extractor(m)) for m in manifests]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare experiment manifests")
    parser.add_argument(
        "manifests", nargs="+",
        help="Paths to experiment_manifest.json files",
    )
    args = parser.parse_args()

    if len(args.manifests) < 2:
        print("Need at least 2 manifests to compare.", file=sys.stderr)
        sys.exit(1)

    manifests = []
    labels = []
    for p in args.manifests:
        path = Path(p)
        if not path.exists():
            print(f"WARNING: not found, skipping: {p}", file=sys.stderr)
            continue
        manifests.append(load_experiment_manifest(path))
        labels.append(path.parent.name)

    if len(manifests) < 2:
        print("Not enough valid manifests to compare.", file=sys.stderr)
        sys.exit(1)

    table = build_comparison_table(manifests, labels)
    print(table)


if __name__ == "__main__":
    main()
