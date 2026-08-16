"""TM.0.23.CORTEX.V6.GATE — D0–D2 with exchangeable opaque motor slots."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_develop_life import (
    LifeSeeds,
    bind_life_actuators,
    curriculum_tokens,
    motor_latent,
    pair_seeds,
)
from experiments.cortex_develop_scorers import score_d0, score_d1, score_d2
from experiments.cortex_v2_gate import THRESHOLDS
from experiments.cortex_develop_life import teach_loop
from experiments.run_tm023cortex import make_cortex

STAGES = ["D0", "D1", "D2"]


def life_clears_d1_d2(stages: dict[str, Any]) -> bool:
    return bool(stages.get("D1", {}).get("ok")) and bool(stages.get("D2", {}).get("ok"))


def run_gate_life(seeds: LifeSeeds, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"v6gate_{seeds.role}_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=device)
        toks = curriculum_tokens(seeds)
        stages: dict[str, Any] = {}
        stages["D0"] = score_d0(ag, seeds, toks)
        bind_life_actuators(ag, toks, seeds)
        teach_loop(
            ag,
            seeds,
            n=30,
            symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
            latent=motor_latent(toks),
        )
        stages["D1"] = score_d1(ag, seeds, toks)
        stages["D2"] = score_d2(ag, seeds, toks)
        d0_ok = bool(stages["D0"].get("ok"))
        d12_ok = life_clears_d1_d2(stages)
        return {
            "role": seeds.role,
            "pair_id": seeds.pair_id,
            "tokens": {k: toks[k] for k in ("press", "harm", "get", "drop", "a", "b", "c")},
            "stages": stages,
            "d0_ok": d0_ok,
            "d1_d2_ok": d12_ok,
            "eligible_pair_member": d0_ok and d12_ok,
        }


def run_gate_pair(pair_id: int, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = pair_seeds(pair_id)
    main = run_gate_life(main_s, device=device)
    twin = run_gate_life(twin_s, device=device)
    pair_clear = bool(
        main["d0_ok"] and twin["d0_ok"] and main["d1_d2_ok"] and twin["d1_d2_ok"]
    )
    return {"pair_id": pair_id, "main": main, "twin": twin, "pair_clear": pair_clear}


def run_v6_gate_battery(*, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    pairs = [run_gate_pair(i, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    d0_fails = 0
    d0_total = 0
    stage_counts = {"D0": 0, "D1": 0, "D2": 0}
    for p in pairs:
        for role in ("main", "twin"):
            life = p[role]
            d0_total += 1
            if not life["d0_ok"]:
                d0_fails += 1
            for sid in STAGES:
                if life["stages"].get(sid, {}).get("ok"):
                    stage_counts[sid] += 1
    systematic_d0_fail = d0_fails > (d0_total // 2)
    gate_clear = n_clear >= THRESHOLDS["gate_clear_min_pairs"] and not systematic_d0_fail
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "stage_ok_counts": stage_counts,
        "d0_fails": d0_fails,
        "d0_total": d0_total,
        "systematic_d0_fail": systematic_d0_fail,
        "thresholds": THRESHOLDS,
        "sensorimotor_association_gate_clear": gate_clear,
        "pairs": pairs,
    }
