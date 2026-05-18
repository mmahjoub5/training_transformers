import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.config.profiling_config import ProfilingConfig
from src.profiling.memory import reset_peak_stats, take_memory_snapshot
from src.profiling.timer import CudaTimer

logger = logging.getLogger(__name__)


class ProfilingCallback(TrainerCallback):
    """Collects per-step timing, memory snapshots, and throughput metrics.

    Hooks into the Trainer lifecycle to record:
    - CUDA/wall-clock step timings via CudaTimer
    - GPU memory snapshots every N steps
    - Throughput (samples/sec, tokens/sec)
    - Optional torch.profiler Chrome/Perfetto traces (bounded window)

    Logs perf metrics under the 'perf/' prefix.
    """

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.step_timings: List[Dict[str, Any]] = []
        self.memory_snapshots: List[Dict[str, Any]] = []
        self._timer = CudaTimer()
        self._train_start_time: Optional[float] = None
        self._total_training_time_sec: Optional[float] = None
        self._profiler: Optional[torch.profiler.profile] = None
        self._cpu_step_start: Optional[float] = None
        self._nvtx = config.enable_nvtx and torch.cuda.is_available()
        logger.info("Initialized ProfilingCallback with config: %s", self.config.to_dict())

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):

        logger.info("Training started with ProfilingCallback. Config: %s", self.config.to_dict())
        """Record start time, reset peak memory stats, optionally start torch.profiler."""
        self._train_start_time = time.perf_counter()
        reset_peak_stats()
        if self._nvtx:
            torch.cuda.nvtx.range_push("training")
        if self.config.enable_torch_profiler and self.config.trace_dir:
            trace_dir = Path(self.config.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)

            def _trace_handler(prof: torch.profiler.profile):
                """Export profiler trace to TensorBoard (includes Chrome/Perfetto trace)."""
                torch.profiler.tensorboard_trace_handler(str(trace_dir))(prof)
                logger.info("Saved TensorBoard profiler trace to %s", trace_dir)

            self._profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=self.config.torch_profiler_active_steps,
                    repeat=1,
                ),
                on_trace_ready=_trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                activities=self.config.get_torch_activities(),
            )
            self._profiler.start()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs):
        """Start the CUDA timer and NVTX range for this step."""
        if self._nvtx:
            torch.cuda.nvtx.range_push(f"step_{state.global_step}")
        self._timer.cpu_start()
        self._timer.cuda_start()

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Stop timer, close NVTX range. Every log_every_n_steps: record timing + memory, compute throughput, log to TB."""
        if self._nvtx:
            torch.cuda.nvtx.range_pop()
        self._timer.cuda_stop()
        self._timer.cpu_stop()
        cuda_elapsed_ms = self._timer.cuda_elapsed_ms()
        wall_clock_ms = self._timer.cpu_elapsed_ms()

        step = state.global_step
        if step % self.config.log_every_n_steps == 0 and cuda_elapsed_ms is not None:
            elapsed_sec = cuda_elapsed_ms / 1000.0
            samples_per_sec = (
                args.per_device_train_batch_size / elapsed_sec if elapsed_sec > 0 else 0.0
            )
            max_seq_length = getattr(args, "max_seq_length", None)
            tokens_per_sec = (
                samples_per_sec * max_seq_length if max_seq_length else None
            )

            timing_entry = {
                "step": step,
                "wall_clock_ms": round(wall_clock_ms, 3),
                "cuda_elapsed_ms": round(cuda_elapsed_ms, 3),
                "samples_per_sec": round(samples_per_sec, 3),
            }
            if tokens_per_sec is not None:
                timing_entry["tokens_per_sec"] = round(tokens_per_sec, 3)
            self.step_timings.append(timing_entry)

            mem = take_memory_snapshot()
            if mem is not None:
                mem_entry = {
                    "step": step,
                    "allocated_mb": mem.allocated_mb,
                    "reserved_mb": mem.reserved_mb,
                    "peak_allocated_mb": mem.peak_allocated_mb,
                    "peak_reserved_mb": mem.peak_reserved_mb,
                    "free_mb": mem.free_mb,
                }
                self.memory_snapshots.append(mem_entry)

            log_entry = {
                "step": step,
                "perf/wall_clock_ms": timing_entry["wall_clock_ms"],
                "perf/cuda_elapsed_ms": timing_entry["cuda_elapsed_ms"],
                "perf/samples_per_sec": timing_entry["samples_per_sec"],
            }
            if tokens_per_sec is not None:
                log_entry["perf/tokens_per_sec"] = timing_entry["tokens_per_sec"]
            if mem is not None:
                log_entry["perf/allocated_mb"] = mem.allocated_mb
                log_entry["perf/reserved_mb"] = mem.reserved_mb
                log_entry["perf/peak_allocated_mb"] = mem.peak_allocated_mb
                log_entry["perf/peak_reserved_mb"] = mem.peak_reserved_mb
                log_entry["perf/free_mb"] = mem.free_mb
            state.log_history.append(log_entry)

        if step % self.config.save_every_n_steps == 0:
            self._save_results()

        if self._profiler is not None:
            self._profiler.step()

    def _save_results(self):
        """Write current profiling data to disk."""
        if not self.config.trace_dir:
            return
        out_dir = Path(self.config.trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = out_dir / "profiling_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "profiling_results.json"
        out_path.write_text(json.dumps(self.get_results(), indent=2))
        logger.info("Saved profiling results to %s (%d timings, %d memory snapshots)",
                     out_path, len(self.step_timings), len(self.memory_snapshots))

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """Record total training time, close NVTX range, close torch.profiler if active."""
        if self._nvtx:
            torch.cuda.nvtx.range_pop()
        if self._train_start_time is not None:
            self._total_training_time_sec = time.perf_counter() - self._train_start_time
        self._save_results()
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None

    def get_results(self) -> Dict[str, Any]:
        """Return all collected profiling data for the experiment manifest.

        Expected structure:
        {
            "total_training_time_sec": float,
            "step_timings": [{"step": int, "wall_clock_ms": float, "cuda_elapsed_ms": float, "samples_per_sec": float}, ...],
            "memory_snapshots": [{"step": int, "allocated_mb": float, "peak_allocated_mb": float}, ...],
        }
        """

        return {
            "total_training_time_sec": self._total_training_time_sec,
            "step_timings": self.step_timings,
            "memory_snapshots": self.memory_snapshots,
        }
