import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GPUInfo:
    name: str
    memory_total_mb: int
    compute_capability: str


@dataclass
class SystemInfo:
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    driver_version: Optional[str]
    library_versions: Dict[str, Optional[str]]
    gpus: List[GPUInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_library_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _get_driver_version() -> Optional[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_gpu_info() -> List[GPUInfo]:
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(GPUInfo(
            name=props.name,
            memory_total_mb=props.total_mem // (1024 * 1024),
            compute_capability=f"{props.major}.{props.minor}",
        ))
    return gpus


def capture_system_info() -> SystemInfo:
    cuda_available = torch.cuda.is_available()
    return SystemInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda if cuda_available else None,
        cudnn_version=str(torch.backends.cudnn.version()) if cuda_available and torch.backends.cudnn.is_available() else None,
        driver_version=_get_driver_version() if cuda_available else None,
        library_versions={
            "transformers": _get_library_version("transformers"),
            "peft": _get_library_version("peft"),
            "trl": _get_library_version("trl"),
            "deepspeed": _get_library_version("deepspeed"),
        },
        gpus=_get_gpu_info(),
    )
