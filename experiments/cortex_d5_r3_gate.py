"""TM.0.23.CORTEX.D5.R3.GATE — historical score_d5 after D1–D4 prefix."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_d5_r3_worlds import sealed_pair_seeds
from experiments.cortex_develop_life import bind_life_actuators, curriculum_tokens, motor_latent, teach_loop
from experiments.cortex_develop_scorers import score_d0, score_d1, score_d2, score_d3, score_d4, score_d5
from experiments.cortex_v2_gate import THRESHOLDS
from experiments.run_tm023cortex import make_cortex


def run_gate_life(seeds, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"d5r3_{seeds.role}_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=device)
        toks = curriculum_tokens(seeds)
        d0 = score_d0(ag, seeds, toks)
        bind_life_actuators(ag, toks, seeds)
        teach_loop(
            ag,
            seeds,
            n=30,
            symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
            latent=motor_latent(toks),
        )
        d1 = score_d1(ag, seeds, toks)
        d2 = score_d2(ag, seeds, toks)
        d3 = score_d3(ag, seeds, toks)
        d4 = score_d4(ag, seeds, toks)
        d5 = score_d5(ag, seeds, toks)
        return {
            "role": seeds.role,
            "pair_id": seeds.pair_id,
            "d0_ok": bool(d0.get("ok")),
            "d1_ok": bool(d1.get("ok")),
            "d2_ok": bool(d2.get("ok")),
            "d3_ok": bool(d3.get("ok")),
            "d4_ok": bool(d4.get("ok")),
            "d5_ok": bool(d5.get("ok")),
            "stages": {"D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5},
            "eligible_pair_member": bool(d0.get("ok") and d5.get("ok")),
        }


def run_gate_pair(pair_id: int, seed_hex: str, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = sealed_pair_seeds(pair_id, seed_hex)
    main = run_gate_life(main_s, device=device)
    twin = run_gate_life(twin_s, device=device)
    pair_clear = bool(main["eligible_pair_member"] and twin["eligible_pair_member"])
    return {"pair_id": pair_id, "main": main, "twin": twin, "pair_clear": pair_clear}


def run_d5_r3_gate_battery(*, seed_hex: str, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    pairs = [run_gate_pair(i, seed_hex, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    stage_counts = {s: 0 for s in ("D0", "D1", "D2", "D3", "D4", "D5")}
    for p in pairs:
        for role in ("main", "twin"):
            for s in stage_counts:
                if p[role][f"{s.lower()}_ok"]:
                    stage_counts[s] += 1
    gate_clear = n_clear >= THRESHOLDS["gate_clear_min_pairs"]
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "stage_ok_counts": stage_counts,
        "thresholds": THRESHOLDS,
        "worlds": "sealed_eval_seed",
        "domain": "TM023.D5.R3.",
        "stage": "D5",
        "relation_gate_clear": gate_clear,
        "pairs": pairs,
    }
