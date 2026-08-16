"""TM.0.23.CORTEX.V11.GATE — D2 HOLD under swapped physics; v10 D1/extras retained."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_develop_life import LifeSeeds, curriculum_tokens
from experiments.cortex_v2_gate import THRESHOLDS
from experiments.cortex_v11_scorers import population_d1, population_d2, score_d0_v11, score_d1_v11, score_d2_v11
from experiments.cortex_v11_worlds import sealed_pair_seeds
from experiments.run_tm023cortex import make_cortex

STAGES = ["D0", "D1", "D2"]


def life_clears_d1_d2(stages: dict[str, Any]) -> bool:
    return bool(stages.get("D1", {}).get("ok")) and bool(stages.get("D2", {}).get("ok"))


def run_gate_life(seeds: LifeSeeds, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"v11gate_{seeds.role}_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=device)
        toks = curriculum_tokens(seeds)
        stages: dict[str, Any] = {}
        stages["D0"] = score_d0_v11(ag, seeds, toks)
        ag.bind_actuators([toks["press"], toks["harm"]])
        birth_ckpt = ag.checkpoint()
        stages["D1"] = score_d1_v11(ag, seeds, toks, birth_ckpt=birth_ckpt)
        ag.bind_actuators([toks["press"], toks["harm"], toks["get"], toks["drop"]])
        stages["D2"] = score_d2_v11(ag, seeds, toks, birth_ckpt=birth_ckpt)
        d0_ok = bool(stages["D0"].get("ok"))
        d12_ok = life_clears_d1_d2(stages)
        return {
            "role": seeds.role,
            "pair_id": seeds.pair_id,
            "tokens": {k: toks[k] for k in ("press", "harm", "get", "drop", "a", "b", "c")},
            "d1_bind": ["press", "harm"],
            "d2_conflict": "swapped_press_harm",
            "stages": stages,
            "d0_ok": d0_ok,
            "d1_d2_ok": d12_ok,
            "eligible_pair_member": d0_ok and d12_ok,
        }


def run_gate_pair(pair_id: int, seed_hex: str, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = sealed_pair_seeds(pair_id, seed_hex)
    main = run_gate_life(main_s, device=device)
    twin = run_gate_life(twin_s, device=device)
    pair_clear = bool(main["d0_ok"] and twin["d0_ok"] and main["d1_d2_ok"] and twin["d1_d2_ok"])
    return {"pair_id": pair_id, "main": main, "twin": twin, "pair_clear": pair_clear}


def run_v11_gate_battery(*, seed_hex: str, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    pairs = [run_gate_pair(i, seed_hex, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    d0_fails = sum(0 if p[r]["d0_ok"] else 1 for p in pairs for r in ("main", "twin"))
    stage_counts = {"D0": 0, "D1": 0, "D2": 0}
    d1_lives: list[dict[str, Any]] = []
    d2_lives: list[dict[str, Any]] = []
    for p in pairs:
        for role in ("main", "twin"):
            for sid in STAGES:
                if p[role]["stages"].get(sid, {}).get("ok"):
                    stage_counts[sid] += 1
            d1_lives.append(p[role]["stages"]["D1"])
            d2_lives.append(p[role]["stages"]["D2"])
    pop_d1 = population_d1(d1_lives)
    pop_d2 = population_d2(d2_lives)
    systematic_d0_fail = d0_fails > 16
    gate_clear = (
        n_clear >= THRESHOLDS["gate_clear_min_pairs"]
        and not systematic_d0_fail
        and bool(pop_d1["ok"])
        and bool(pop_d2["ok"])
    )
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "stage_ok_counts": stage_counts,
        "d0_fails": d0_fails,
        "d0_total": 32,
        "systematic_d0_fail": systematic_d0_fail,
        "thresholds": THRESHOLDS,
        "worlds": "sealed_eval_seed",
        "d1_bind": ["press", "harm"],
        "d2_conflict": "swapped_press_harm",
        "extras": "population",
        "population_d1": pop_d1,
        "population_d2": pop_d2,
        "sensorimotor_association_gate_clear": gate_clear,
        "pairs": pairs,
    }
