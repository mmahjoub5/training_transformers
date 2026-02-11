import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.profiling.system_info import capture_system_info

logger = logging.getLogger(__name__)


def _compute_summary(profiling_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregate statistics from raw profiling data.

    Returns avg/median/p95 step time, peak GPU memory, and avg throughput.
    """
    summary: Dict[str, Any] = {}

    step_timings: List[Dict[str, Any]] = profiling_results.get("step_timings", [])
    if step_timings:
        wall_times = [t["wall_clock_ms"] for t in step_timings]
        summary["avg_step_time_ms"] = round(statistics.mean(wall_times), 1)
        summary["median_step_time_ms"] = round(statistics.median(wall_times), 1)

        sorted_times = sorted(wall_times)
        p95_idx = max(0, int(len(sorted_times) * 0.95) - 1)
        summary["p95_step_time_ms"] = round(sorted_times[p95_idx], 1)

        samples_vals = [t["samples_per_sec"] for t in step_timings]
        summary["avg_samples_per_sec"] = round(statistics.mean(samples_vals), 1)

        tokens_vals = [t["tokens_per_sec"] for t in step_timings if "tokens_per_sec" in t]
        if tokens_vals:
            summary["avg_tokens_per_sec"] = round(statistics.mean(tokens_vals), 1)

    memory_snapshots: List[Dict[str, Any]] = profiling_results.get("memory_snapshots", [])
    if memory_snapshots:
        peak_vals = [m["peak_allocated_mb"] for m in memory_snapshots]
        summary["peak_gpu_memory_mb"] = round(max(peak_vals), 1)

    total_time = profiling_results.get("total_training_time_sec")
    if total_time is not None:
        summary["total_training_time_sec"] = round(total_time, 2)

    return summary


def save_experiment_manifest(
    output_dir: str,
    config: dict,
    profiling_results: Optional[Dict[str, Any]] = None,
    train_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an experiment manifest JSON combining config, metrics, and profiling data.

    The manifest includes a timestamp, full system hardware/software info,
    the training config, HF Trainer metrics, raw profiling data, and a
    computed summary with aggregate statistics.
    """
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": capture_system_info().to_dict(),
        "config": config,
    }

    if train_metrics is not None:
        manifest["train_metrics"] = train_metrics

    if profiling_results is not None:
        manifest["profiling"] = profiling_results
        manifest["summary"] = _compute_summary(profiling_results)

    out_path = Path(output_dir) / "experiment_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Saved experiment manifest to %s", out_path)
    return out_path


def load_experiment_manifest(path: str | Path) -> Dict[str, Any]:
    """Load an experiment manifest JSON from disk."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)
