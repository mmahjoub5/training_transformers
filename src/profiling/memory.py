import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    free_mb: float


def take_memory_snapshot() -> Optional[MemorySnapshot]:
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    free_bytes, _ = torch.cuda.mem_get_info()
    free = free_bytes / (1024 * 1024)

    return MemorySnapshot(
        allocated_mb=round(allocated, 1),
        reserved_mb=round(reserved, 1),
        peak_allocated_mb=round(peak_allocated, 1),
        peak_reserved_mb=round(peak_reserved, 1),
        free_mb=round(free, 1),
    )


def reset_peak_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()