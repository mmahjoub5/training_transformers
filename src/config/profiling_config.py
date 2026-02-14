from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import warnings
import torch


@dataclass
class ProfilingConfig:
    """Configuration for profiling."""

    enabled: bool = False
    log_every_n_steps: int = 10
    trace_dir: Optional[str] = None
    enable_torch_profiler: bool = True
    profiler_start_step: int = 20
    profiler_end_step: int = 40
    torch_profiler_active_steps: int = 10
    save_every_n_steps: int = 100
    enable_nvtx: bool = False
    activities: List[str] = field(default_factory=lambda: ["cpu", "cuda"])

    def __post_init__(self):
        """Validation logic."""

        if self.profiler_start_step >= self.profiler_end_step:
            raise ValueError(
                "profiler_start_step must be < profiler_end_step"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict."""
        return {
            "enabled": self.enabled,
            "log_every_n_steps": self.log_every_n_steps,
            "trace_dir": self.trace_dir,
            "enable_torch_profiler": self.enable_torch_profiler,
            "profiler_start_step": self.profiler_start_step,
            "profiler_end_step": self.profiler_end_step,
            "torch_profiler_active_steps": self.torch_profiler_active_steps,
            "save_every_n_steps": self.save_every_n_steps,
            "enable_nvtx": self.enable_nvtx,
            "activities": self.activities,
        }

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ProfilingConfig":
        prof_cfg = cfg.get("profiling", {})
        if not prof_cfg:
            return cls()
        return cls(
            enabled=prof_cfg.get("enabled", False),
            log_every_n_steps=prof_cfg.get("log_every_n_steps", 10),
            trace_dir=prof_cfg.get("trace_dir", None),
            enable_torch_profiler=prof_cfg.get("enable_torch_profiler", False),
            profiler_start_step=prof_cfg.get("profiler_start_step", 20),
            profiler_end_step=prof_cfg.get("profiler_end_step", 40),
            torch_profiler_active_steps=prof_cfg.get("torch_profiler_active_steps", 10),
            save_every_n_steps=prof_cfg.get("save_every_n_steps", 100),
            enable_nvtx=prof_cfg.get("enable_nvtx", False),
            activities=prof_cfg.get("activities", ["cpu", "cuda"]),
        )

    def get_torch_activities(self):
        import torch.profiler as profiler

        valid_activities = []

        mapping = {
            "cpu": profiler.ProfilerActivity.CPU,
            "cuda": profiler.ProfilerActivity.CUDA,
        }

        for activity in self.activities:
            activity_lower = activity.lower()

            if activity_lower == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA profiling requested but CUDA unavailable"
                )
                continue

            if activity_lower not in mapping:
                raise ValueError(f"Unknown profiling activity: {activity}")

            valid_activities.append(mapping[activity_lower])

        return valid_activities
