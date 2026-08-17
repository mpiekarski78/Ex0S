"""TM.0.23.CORTEX.D3.R1.GATE — historical score_d3; lives from TM023.D3.R1. seeds."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_develop_life import LifeSeeds, bind_life_actuators, curriculum_tokens
from experiments.cortex_develop_scorers import score_d0, score_d3
from experiments.cortex_d3_r1_worlds import sealed_pair_seeds
from experiments.cortex_v2_gate import THRESHOLDS
from experiments.run_tm023cortex import make_cortex


def run_gate_life(seeds: LifeSeeds, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"d3r1_{seeds.role}_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=device)
        toks = curriculum_tokens(seeds)
        d0 = score_d0(ag, seeds, toks)
        bind_life_actuators(ag, toks, seeds)
        d3 = score_d3(ag, seeds, toks)
        return {
            "role": seeds.role,
            "pair_id": seeds.pair_id,
            "d0_ok": bool(d0.get("ok")),
            "d3_ok": bool(d3.get("ok")),
            "stages": {"D0": d0, "D3": d3},
            "eligible_pair_member": bool(d0.get("ok") and d3.get("ok")),
        }


def run_gate_pair(pair_id: int, seed_hex: str, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = sealed_pair_seeds(pair_id, seed_hex)
    main = run_gate_life(main_s, device=device)
    twin = run_gate_life(twin_s, device=device)
    pair_clear = bool(main["eligible_pair_member"] and twin["eligible_pair_member"])
    return {"pair_id": pair_id, "main": main, "twin": twin, "pair_clear": pair_clear}


def run_d3_r1_gate_battery(*, seed_hex: str, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    pairs = [run_gate_pair(i, seed_hex, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    stage_counts = {"D0": 0, "D3": 0}
    for p in pairs:
        for role in ("main", "twin"):
            if p[role]["d0_ok"]:
                stage_counts["D0"] += 1
            if p[role]["d3_ok"]:
                stage_counts["D3"] += 1
    gate_clear = n_clear >= THRESHOLDS["gate_clear_min_pairs"]
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "stage_ok_counts": stage_counts,
        "thresholds": THRESHOLDS,
        "worlds": "sealed_eval_seed",
        "domain": "TM023.D3.R1.",
        "stage": "D3",
        "relation_gate_clear": gate_clear,
        "pairs": pairs,
    }
