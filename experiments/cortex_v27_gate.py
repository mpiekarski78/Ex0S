"""TM.0.23.CORTEX.V27.GEN.GATE — G1 source + G3 non-echo + G5 STOP evidence."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_develop_life import bind_life_actuators, curriculum_tokens
from experiments.cortex_v26_generality import (
    BETTER,
    BODY0,
    WORSE,
    control_g1,
    _mean_stop_len,
    _observe,
    _teach_stop_boundary,
)
from experiments.cortex_v27_worlds import sealed_pair_seeds
from experiments.cortex_v2_gate import THRESHOLDS
from experiments.run_tm023cortex import make_cortex
from three_memory.neural_cortex import NeuralCortex


def _g3_life(ag: NeuralCortex, toks: dict[str, str], prefix: str) -> dict[str, Any]:
    a, b = toks["a"], toks["b"]
    for i in range(40):
        _observe(ag, ix=f"{prefix}_fam_a_{i}", symbols=[a], body=BODY0)
    for i in range(40):
        _observe(ag, ix=f"{prefix}_fam_b_{i}", symbols=[b], body=BODY0)
    body = list(BODY0)
    for i in range(80):
        out = _observe(ag, ix=f"{prefix}_teach_{i}", symbols=[a], body=body)
        seq = list((out.get("action") or {}).get("emit_sequence") or [])
        if seq and seq[0] == b:
            body = list(BETTER)
        elif seq and seq[0] == a:
            body = list(WORSE)
        else:
            body = list(BODY0)
    echo_a = emit_b = other = 0
    for i in range(20):
        out = _observe(ag, ix=f"{prefix}_probe_{i}", symbols=[a], body=BODY0)
        seq = list((out.get("action") or {}).get("emit_sequence") or [])
        if seq and seq[0] == b:
            emit_b += 1
        elif seq and seq[0] == a:
            echo_a += 1
        else:
            other += 1
    ok = emit_b > echo_a and emit_b >= 3
    return {"ok": ok, "emit_b": emit_b, "echo_a": echo_a, "other": other}


def run_gate_life(seeds, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"v27_{seeds.role}_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=device)
        toks = curriculum_tokens(seeds)
        bind_life_actuators(ag, toks, seeds)
        g3 = _g3_life(ag, toks, f"g3_{seeds.role}{seeds.pair_id}")
        cue = [toks["a"], toks["b"], toks["c"], toks["foil"]]
        return {
            "role": seeds.role,
            "pair_id": seeds.pair_id,
            "g3": g3,
            "g3_ok": bool(g3.get("ok")),
            "cue": cue,
            "tokens": {"a": toks["a"], "b": toks["b"]},
        }


def run_gate_pair(pair_id: int, seed_hex: str, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = sealed_pair_seeds(pair_id, seed_hex)
    # G3 on independent organisms; G5 on a second pair of organisms (STOP teaching would confound G3).
    main = run_gate_life(main_s, device=device)
    twin = run_gate_life(twin_s, device=device)
    with tempfile.TemporaryDirectory(prefix=f"v27g5_{pair_id}_") as tmp:
        m_toks = curriculum_tokens(main_s)
        t_toks = curriculum_tokens(twin_s)
        m_ag = make_cortex(Path(tmp) / "m", genome=main_s.genome(), device=device)
        t_ag = make_cortex(Path(tmp) / "t", genome=twin_s.genome(), device=device)
        bind_life_actuators(m_ag, m_toks, main_s)
        bind_life_actuators(t_ag, t_toks, twin_s)
        cue_m = [m_toks["a"], m_toks["b"], m_toks["c"], m_toks["foil"]]
        cue_t = [t_toks["a"], t_toks["b"], t_toks["c"], t_toks["foil"]]
        for i in range(40):
            _observe(m_ag, ix=f"g5fam_m_{i}", symbols=cue_m, body=BODY0)
            _observe(t_ag, ix=f"g5fam_t_{i}", symbols=cue_t, body=BODY0)
        _teach_stop_boundary(m_ag, cue_m, 2, f"g5m{pair_id}")
        _teach_stop_boundary(t_ag, cue_t, 4, f"g5t{pair_id}")
        mean2 = _mean_stop_len(m_ag, cue_m, f"g5p2_{pair_id}")
        mean4 = _mean_stop_len(t_ag, cue_t, f"g5p4_{pair_id}")
    closer2 = abs(mean2 - 2.0) < abs(mean2 - 4.0)
    closer4 = abs(mean4 - 4.0) < abs(mean4 - 2.0)
    g5_ok = abs(mean4 - mean2) >= 1.0 and closer2 and closer4
    g5 = {
        "ok": g5_ok,
        "mean_len_boundary2": mean2,
        "mean_len_boundary4": mean4,
        "closer_to_2": closer2,
        "closer_to_4": closer4,
    }
    pair_clear = bool(main["g3_ok"] and twin["g3_ok"] and g5_ok)
    return {"pair_id": pair_id, "main": main, "twin": twin, "g5": g5, "pair_clear": pair_clear}


def run_v27_gate_battery(*, seed_hex: str, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    g1 = control_g1()
    pairs = [run_gate_pair(i, seed_hex, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    gate_clear = bool(g1.get("ok") and n_clear >= THRESHOLDS["gate_clear_min_pairs"])
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "g1": g1,
        "g1_ok": bool(g1.get("ok")),
        "thresholds": THRESHOLDS,
        "worlds": "sealed_eval_seed",
        "domain": "TM023.V27.GEN.",
        "relation_gate_clear": gate_clear,
        "pairs": pairs,
        "refuse_fulldev_r7": True,
        "earned_next": False,
        "ex0s": None,
        "product": "0.0.004",
    }
