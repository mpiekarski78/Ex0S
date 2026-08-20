"""
CUDA throughput and stability benchmark for Reference Birth.

Unscored diagnostic run; records wall time, device utilization, and finiteness.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.dev1.conventional_ac_ceiling import evaluate_ceiling_life
from experiments.dev1.reference_birth_life import evaluate_reference_birth_life, run_batched_lives_cuda
from three_memory.dev1.device import cuda_utilization_sample, dev1_device


BENCHMARK_SEED = "reference_birth_cuda_benchmark_20260820"
DEFAULT_BATCH = 4
DEFAULT_EPISODES = 8


@dataclass
class CudaBenchmarkResult:
    passed: bool
    device: str
    cuda_available: bool
    cpu_wall_seconds: float
    cuda_wall_seconds: float
    speedup_ratio: float
    cpu_gpu_accuracy_delta: float
    finite_forward: bool
    utilization: dict
    details: dict


def run_cuda_benchmark(
    seed: str = BENCHMARK_SEED,
    batch_size: int = DEFAULT_BATCH,
    n_episodes: int = DEFAULT_EPISODES,
) -> CudaBenchmarkResult:
    if not torch.cuda.is_available():
        return CudaBenchmarkResult(
            passed=False,
            device="cpu",
            cuda_available=False,
            cpu_wall_seconds=0.0,
            cuda_wall_seconds=0.0,
            speedup_ratio=0.0,
            cpu_gpu_accuracy_delta=0.0,
            finite_forward=False,
            utilization={"cuda_available": False},
            details={"reason": "cuda_not_available"},
        )

    cpu_dev = torch.device("cpu")
    cuda_dev = torch.device("cuda")
    seeds = [f"{seed}_row_{i}" for i in range(batch_size)]

    t0 = time.time()
    cpu_lives = [
        evaluate_reference_birth_life(
            "reward_eprop_rate_adaptation",
            s,
            "stochastic",
            device=cpu_dev,
            n_episodes=n_episodes,
        )
        for s in seeds
    ]
    cpu_wall = time.time() - t0

    torch.cuda.synchronize()
    t1 = time.time()
    cuda_lives = run_batched_lives_cuda(
        "reward_eprop_rate_adaptation",
        seeds,
        batch_size=batch_size,
        policy_mode="stochastic",
        device=cuda_dev,
        n_episodes=n_episodes,
    )
    torch.cuda.synchronize()
    cuda_wall = time.time() - t1

    ceiling_cuda = evaluate_ceiling_life(
        seed + "_ceiling",
        "stochastic",
        device=cuda_dev,
        n_episodes=n_episodes,
    )

    cpu_acc = sum(l.treatment_accuracy for l in cpu_lives) / len(cpu_lives)
    cuda_acc = sum(l.treatment_accuracy for l in cuda_lives) / len(cuda_lives)
    acc_delta = abs(cpu_acc - cuda_acc)
    finite = all(
        torch.isfinite(torch.tensor([l.treatment_accuracy, l.cumulative_reward])).all().item()
        for l in cuda_lives
    ) and ceiling_cuda.treatment_accuracy >= 0.0

    speedup = cpu_wall / max(cuda_wall, 1e-6)
    util = cuda_utilization_sample()

    return CudaBenchmarkResult(
        passed=finite and acc_delta <= 0.5,
        device=str(cuda_dev),
        cuda_available=True,
        cpu_wall_seconds=cpu_wall,
        cuda_wall_seconds=cuda_wall,
        speedup_ratio=speedup,
        cpu_gpu_accuracy_delta=acc_delta,
        finite_forward=finite,
        utilization=util,
        details={
            "cpu_accuracy_mean": cpu_acc,
            "cuda_accuracy_mean": cuda_acc,
            "ceiling_cuda_accuracy": ceiling_cuda.treatment_accuracy,
            "batch_size": batch_size,
            "n_episodes": n_episodes,
        },
    )


def write_benchmark_report(
    output_dir: str = "runs/exos_dev1/stage_a_reference_birth",
    **kwargs,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run_cuda_benchmark(**kwargs)
    payload = asdict(result)
    path = out / "cuda_throughput_benchmark.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(write_benchmark_report(), indent=2))
