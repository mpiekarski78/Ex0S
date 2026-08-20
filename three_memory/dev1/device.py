"""Canonical device policy for EX0S-DEV1."""

from __future__ import annotations

import os

import torch


def dev1_device(*, require_cuda: bool = False) -> torch.device:
    """
    Return the compute device for dev1 organisms and batched evaluators.

    Scored Reference Birth runs require CUDA unless prereg records fallback.
    """
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA required but not available")
    return torch.device("cpu")


def dev1_device_name() -> str:
    return str(dev1_device())


def assert_cuda_tensors(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.device.type != "cuda":
            raise RuntimeError(f"expected CUDA tensor, got {t.device}")


def cuda_utilization_sample() -> dict:
    """Best-effort diagnostic; not a pass/fail gate."""
    out: dict = {"cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return out
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            out["gpu_util_percent"] = float(parts[0].strip())
            out["memory_used_mb"] = float(parts[1].strip())
    except Exception as exc:
        out["nvidia_smi_error"] = str(exc)
    out["torch_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
    return out
