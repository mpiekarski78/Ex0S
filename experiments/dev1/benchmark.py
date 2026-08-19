"""
Unscored Stage A R1 CPU/GPU throughput benchmark.

This is a measurement script only. It is separate from correctness tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.interfaces import OrganismObservation


def benchmark(device: torch.device, n_steps: int = 256) -> dict:
    genome = DevGenome.default()
    org = ModularOrganism.birth(genome, device=device, h_disabled=True, consolidation_disabled=True)
    rng = np.random.RandomState(123)
    t0 = time.time()
    for i in range(n_steps):
        v = rng.randn(genome.sensory_dim).astype(np.float32)
        org.observe(OrganismObservation(sensory_vector=v, reward=float(i % 2)))
        org.act()
        org.rest()
    elapsed = time.time() - t0
    return {
        "device": str(device),
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "steps_per_s": n_steps / max(elapsed, 1e-9),
    }


if __name__ == "__main__":
    out = [benchmark(torch.device("cpu"))]
    if torch.cuda.is_available():
        out.append(benchmark(torch.device("cuda:0")))
    Path("runs/exos_dev1/stage_a_r1").mkdir(parents=True, exist_ok=True)
    with open("runs/exos_dev1/stage_a_r1/throughput_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
