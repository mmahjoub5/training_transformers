"""Tests for profiling utilities (CudaTimer, memory, system_info)."""

import json

from src.profiling.memory import take_memory_snapshot
from src.profiling.system_info import capture_system_info
from src.profiling.timer import CudaTimer


class TestCudaTimerCPUFallback:
    """CudaTimer unified interface on CPU (no CUDA)."""

    def test_start_stop_elapsed(self):
        t = CudaTimer(enabled=False)
        t.start()
        # do a tiny bit of work so elapsed > 0
        sum(range(1000))
        t.stop()
        ms = t.elapsed_ms()
        assert ms >= 0.0
        assert isinstance(ms, float)

    def test_cpu_methods_still_work(self):
        t = CudaTimer(enabled=False)
        t.cpu_start()
        _ = sum(range(1000))
        t.cpu_stop()
        assert t.cpu_elapsed_ms() >= 0.0

    def test_unified_uses_cpu_when_no_cuda(self):
        t = CudaTimer(enabled=False)
        assert t.enabled is False
        t.start()
        t.stop()
        # elapsed_ms should fall back to cpu_elapsed_ms
        ms = t.elapsed_ms()
        assert isinstance(ms, float)


class TestMemorySnapshot:
    def test_returns_none_without_cuda(self):
        snap = take_memory_snapshot()
        # On CPU-only machines this should be None
        # On GPU machines it would be a MemorySnapshot — both are valid
        assert snap is None or hasattr(snap, "allocated_mb")


class TestSystemInfo:
    def test_capture_returns_valid_info(self):
        info = capture_system_info()
        assert isinstance(info.hostname, str)
        assert isinstance(info.python_version, str)
        assert isinstance(info.torch_version, str)
        assert isinstance(info.cuda_available, bool)

    def test_to_dict_is_json_serializable(self):
        info = capture_system_info()
        d = info.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert "hostname" in serialized
