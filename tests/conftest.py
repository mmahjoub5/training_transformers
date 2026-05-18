"""Shared fixtures for profiling / config / experiment tests."""

import pytest


@pytest.fixture
def sample_config_with_profiling():
    """Full config dict with profiling section enabled."""
    return {
        "model": {"name": "test/model", "kind": "clm"},
        "data": {"name": "json", "data_file": "data.json", "preprocessor": "SocraticPreprocessor"},
        "tokenizer": {"max_length": 512},
        "training": {"batch_size": 2, "lr": 1e-4, "precision": "bf16"},
        "profiling": {
            "enabled": True,
            "log_every_n_steps": 5,
            "trace_dir": "/tmp/traces",
            "enable_torch_profiler": False,
        },
        "deepspeed": {
            "enabled": True,
            "zero_stage": 2,
        },
    }


@pytest.fixture
def sample_config_minimal():
    """Config dict with no profiling or deepspeed sections."""
    return {
        "model": {"name": "test/model", "kind": "clm"},
        "training": {"batch_size": 2},
    }


@pytest.fixture
def sample_profiling_results():
    """Profiling results dict matching ProfilingCallback.get_results() output."""
    return {
        "total_training_time_sec": 120.5,
        "step_timings": [
            {"step": 10, "wall_clock_ms": 250.0, "cuda_elapsed_ms": 245.0, "samples_per_sec": 8.0, "tokens_per_sec": 4096.0},
            {"step": 20, "wall_clock_ms": 240.0, "cuda_elapsed_ms": 235.0, "samples_per_sec": 8.3, "tokens_per_sec": 4249.6},
            {"step": 30, "wall_clock_ms": 300.0, "cuda_elapsed_ms": 295.0, "samples_per_sec": 6.7, "tokens_per_sec": 3430.4},
            {"step": 40, "wall_clock_ms": 260.0, "cuda_elapsed_ms": 255.0, "samples_per_sec": 7.7, "tokens_per_sec": 3942.4},
        ],
        "memory_snapshots": [
            {"step": 10, "allocated_mb": 1200.0, "reserved_mb": 1500.0, "peak_allocated_mb": 1400.0, "peak_reserved_mb": 1600.0, "free_mb": 6000.0},
            {"step": 20, "allocated_mb": 1250.0, "reserved_mb": 1500.0, "peak_allocated_mb": 1450.0, "peak_reserved_mb": 1600.0, "free_mb": 5900.0},
            {"step": 30, "allocated_mb": 1300.0, "reserved_mb": 1550.0, "peak_allocated_mb": 1500.0, "peak_reserved_mb": 1650.0, "free_mb": 5800.0},
            {"step": 40, "allocated_mb": 1280.0, "reserved_mb": 1520.0, "peak_allocated_mb": 1500.0, "peak_reserved_mb": 1650.0, "free_mb": 5850.0},
        ],
    }
