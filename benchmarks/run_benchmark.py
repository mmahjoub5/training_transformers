"""Benchmark runner: execute multiple training configs and collect profiling manifests.

Usage:
    python -m benchmarks.run_benchmark \
        --configs configs/a.yaml configs/b.yaml \
        --max_steps 50

For each config, a temporary copy is created with training.max_steps overridden,
and profiling force-enabled.  The original config files are never modified.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def _patch_config(config_path: str, max_steps: int) -> dict:
    """Load a YAML config and override max_steps + enable profiling."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("training", {})["max_steps"] = max_steps

    cfg.setdefault("profiling", {})["enabled"] = True

    return cfg


def run_one(config_path: str, max_steps: int) -> int:
    """Run a single training job as a subprocess, returning its exit code."""
    print(f"\n{'='*60}")
    print(f"  Running: {config_path}  (max_steps={max_steps})")
    print(f"{'='*60}\n")

    patched = _patch_config(config_path, max_steps)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="bench_", delete=False
    ) as tmp:
        yaml.dump(patched, tmp)
        tmp_path = tmp.name

    cmd = [
        sys.executable, "-m", "scripts.lora_train",
        "--config", tmp_path,
    ]
    result = subprocess.run(cmd)

    Path(tmp_path).unlink(missing_ok=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run benchmark across configs")
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Paths to YAML training configs",
    )
    parser.add_argument(
        "--max_steps", type=int, default=50,
        help="Override max training steps for each run (default: 50)",
    )
    args = parser.parse_args()

    results = {}
    for cfg in args.configs:
        if not Path(cfg).exists():
            print(f"WARNING: config not found, skipping: {cfg}")
            continue
        rc = run_one(cfg, args.max_steps)
        results[cfg] = rc
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  -> {cfg}: {status}")

    print(f"\n{'='*60}")
    print("  Benchmark Summary")
    print(f"{'='*60}")
    for cfg, rc in results.items():
        status = "PASS" if rc == 0 else "FAIL"
        print(f"  [{status}] {cfg}")

    failed = sum(1 for rc in results.values() if rc != 0)
    if failed:
        print(f"\n{failed}/{len(results)} configs failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} configs passed.")


if __name__ == "__main__":
    main()
