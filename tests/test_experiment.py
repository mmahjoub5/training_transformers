"""Tests for experiment manifest save/load and summary computation."""

import json

from src.utils.experiment import (
    _compute_summary,
    load_experiment_manifest,
    save_experiment_manifest,
)


class TestComputeSummary:
    def test_correct_avg_median_p95(self, sample_profiling_results):
        summary = _compute_summary(sample_profiling_results)
        wall_times = [250.0, 240.0, 300.0, 260.0]

        assert summary["avg_step_time_ms"] == round(sum(wall_times) / len(wall_times), 1)
        # median of [240, 250, 260, 300] = (250+260)/2 = 255
        assert summary["median_step_time_ms"] == 255.0

        sorted_times = sorted(wall_times)
        p95_idx = max(0, int(len(sorted_times) * 0.95) - 1)
        assert summary["p95_step_time_ms"] == round(sorted_times[p95_idx], 1)

    def test_peak_memory(self, sample_profiling_results):
        summary = _compute_summary(sample_profiling_results)
        assert summary["peak_gpu_memory_mb"] == 1500.0

    def test_total_training_time(self, sample_profiling_results):
        summary = _compute_summary(sample_profiling_results)
        assert summary["total_training_time_sec"] == 120.5

    def test_throughput_averages(self, sample_profiling_results):
        summary = _compute_summary(sample_profiling_results)
        samples = [8.0, 8.3, 6.7, 7.7]
        assert summary["avg_samples_per_sec"] == round(sum(samples) / len(samples), 1)
        assert "avg_tokens_per_sec" in summary

    def test_empty_results(self):
        summary = _compute_summary({})
        assert summary == {}

    def test_no_memory_snapshots(self):
        results = {
            "step_timings": [
                {"step": 1, "wall_clock_ms": 100.0, "cuda_elapsed_ms": 95.0, "samples_per_sec": 10.0},
            ],
        }
        summary = _compute_summary(results)
        assert "peak_gpu_memory_mb" not in summary
        assert summary["avg_step_time_ms"] == 100.0


class TestSaveLoadManifest:
    def test_save_creates_file(self, tmp_path, sample_profiling_results):
        out = save_experiment_manifest(
            output_dir=str(tmp_path),
            config={"model": {"name": "test"}},
            profiling_results=sample_profiling_results,
        )
        assert out.exists()
        assert out.name == "experiment_manifest.json"

    def test_round_trip(self, tmp_path, sample_profiling_results):
        config = {"model": {"name": "test"}, "training": {"lr": 1e-4}}
        save_experiment_manifest(
            output_dir=str(tmp_path),
            config=config,
            profiling_results=sample_profiling_results,
            train_metrics={"loss": 0.5},
        )
        loaded = load_experiment_manifest(tmp_path / "experiment_manifest.json")
        assert loaded["config"] == config
        assert loaded["train_metrics"]["loss"] == 0.5
        assert "summary" in loaded
        assert "profiling" in loaded
        assert loaded["profiling"]["total_training_time_sec"] == 120.5

    def test_no_profiling(self, tmp_path):
        out = save_experiment_manifest(
            output_dir=str(tmp_path),
            config={"model": {"name": "test"}},
        )
        loaded = load_experiment_manifest(out)
        assert "profiling" not in loaded
        assert "summary" not in loaded

    def test_valid_json(self, tmp_path, sample_profiling_results):
        out = save_experiment_manifest(
            output_dir=str(tmp_path),
            config={"model": {"name": "test"}},
            profiling_results=sample_profiling_results,
        )
        with open(out) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "system_info" in data
