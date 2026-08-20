"""
Developmental Birth R4 scored search harness (not authorized until prereg + explicit go).

Matched outer budgets across fixed-credit and LSG columns. Does not execute a
scored run unless invoked with --scored after prereg authorization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.dev1.developmental_birth_r4_ceiling import evaluate_ceiling_gate_bundle
from experiments.dev1.developmental_birth_r4_life import (
    FACTORIAL_CELLS,
    evaluate_matched_factorial,
    evaluate_r4_life,
)
from experiments.dev1.developmental_birth_r4_outer import MatchedOuterBudget, run_matched_es_smoke
from three_memory.dev1.device import cuda_utilization_sample, dev1_device
from three_memory.dev1.development.generative_genome import GenerativeGenome


def run_unscored_benchmark(world_seed: str, device: torch.device | None = None) -> dict:
    dev = device or dev1_device()
    t0 = time.perf_counter()
    cells = evaluate_matched_factorial(
        world_seed, n_episodes=4, episode_ticks=8, embryonic_seed=0, device=dev
    )
    es = run_matched_es_smoke(
        world_seed + ":es",
        MatchedOuterBudget(population=4, generations=2, n_episodes=2, episode_ticks=4),
        device=dev,
    )
    ceiling = evaluate_ceiling_gate_bundle(
        GenerativeGenome.small(), world_seed + ":ceiling", n_episodes=4, episode_ticks=8, device=dev
    )
    return {
        "wall_s": time.perf_counter() - t0,
        "device": str(dev),
        "cuda_utilization": cuda_utilization_sample(),
        "cells": {k: v.treatment_accuracy for k, v in cells.items()},
        "es_matched": es["matched"],
        "es_lives": es["fixed_lives"],
        "ceiling_accuracy": ceiling["final_comfort_rate"],
        "ceiling_kind": ceiling["ceiling_kind"],
        "ceiling_margin_over_random": ceiling["comfort_margin_over_random"],
        "factorial_cells": list(FACTORIAL_CELLS),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", action="store_true", help="Forbidden until explicit authorization")
    p.add_argument("--world-seed", default="developmental_birth_r4_unscored_bench")
    p.add_argument("--out", default="runs/exos_dev1/stage_a_developmental_birth_r4/unscored_benchmark.json")
    args = p.parse_args()
    if args.scored:
        raise SystemExit(
            "Scored R4 run is not authorized. Freeze prereg and wait for explicit authorization."
        )
    report = run_unscored_benchmark(args.world_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"ok": True, "path": str(out), "wall_s": report["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
