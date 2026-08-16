"""Shared v7 population motor-learning measurements (no neural edits)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from experiments.cortex_develop_life import (
    BODY0,
    LifeSeeds,
    apply_event,
    bind_life_actuators,
    curriculum_tokens,
    motor_latent,
    pair_seeds,
    teach_loop,
)
from experiments.cortex_mact_boundary import SWAP_REVISE_EPISODES, _frozen_pref_counts, _freeze_plasticity
from experiments.run_tm023cortex import make_cortex
from three_memory.neural_cortex import NeuralCortex

N_PAIRS = 32
N_PROBES = 40
TEACH_N = 80
MAJORITY_MIN = 24
MEAN_DELTA_MIN = 0.10
MAX_NUISANCE_ABS = 0.15
PERM_N = 9999
PERM_ALPHA = 0.05


def freeze_credit(ag: NeuralCortex) -> None:
    _freeze_plasticity(ag)


def p_handle(counts: dict[str, int], hid: str, n: int = N_PROBES) -> float:
    return float(counts.get(hid, 0)) / float(n)


def exact_perm_two_sided(labels: list[int], values: list[float], *, n_perm: int = PERM_N, seed: int = 7) -> float:
    """Permutation test: mean(values|label=1) - mean(values|label=0) vs 0."""
    lab = np.asarray(labels, dtype=np.int64)
    val = np.asarray(values, dtype=np.float64)
    if lab.size == 0:
        return 1.0
    obs = float(val[lab == 1].mean() - val[lab == 0].mean()) if (lab == 0).any() and (lab == 1).any() else 0.0
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        shuf = rng.permutation(lab)
        d = float(val[shuf == 1].mean() - val[shuf == 0].mean()) if (shuf == 0).any() and (shuf == 1).any() else 0.0
        if abs(d) >= abs(obs) - 1e-15:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def _make(seeds: LifeSeeds, tmp: str) -> NeuralCortex:
    from pathlib import Path

    return make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")


def paired_c5_one(pair_id: int, *, beneficial_slot: int, swap_vectors: bool) -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    main_s, _ = pair_seeds(pair_id)
    toks = curriculum_tokens(main_s)
    a_h, b_h = toks["press"], toks["harm"]
    order = [a_h, b_h] if beneficial_slot == 0 else [b_h, a_h]
    if swap_vectors:
        order = list(reversed(order))
    lat = {
        "act_effects": {
            a_h: {"state": ["st_pressed"], "delta": [0.25, -0.1, 0.15, 0.0]},
            b_h: {"state": ["st_hurt"], "delta": [-0.35, 0.45, -0.15, 0.0]},
        }
    }
    with tempfile.TemporaryDirectory(prefix=f"v7c5_{pair_id}_") as tmp:
        ag = make_cortex(Path(tmp) / "t", genome=main_s.genome(), device="cpu")
        ag.bind_actuators(order)
        birth = _frozen_pref_counts(ag, toks, N_PROBES, [toks["a"]])
        ckpt = ag.checkpoint()
        teach_loop(ag, main_s, n=TEACH_N, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=lat)
        trained = _frozen_pref_counts(ag, toks, N_PROBES, [toks["a"]])
        ag.load_checkpoint(ckpt)
        freeze_credit(ag)
        teach_loop(ag, main_s, n=TEACH_N, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=lat)
        frozen = _frozen_pref_counts(ag, toks, N_PROBES, [toks["a"]])
    p_t = p_handle(trained, a_h)
    p_f = p_handle(frozen, a_h)
    return {
        "pair_id": pair_id,
        "beneficial_slot": beneficial_slot,
        "swap_vectors": swap_vectors,
        "order": order,
        "press": a_h,
        "harm": b_h,
        "birth": birth,
        "trained": trained,
        "frozen": frozen,
        "p_trained": p_t,
        "p_frozen": p_f,
        "delta": p_t - p_f,
        "trained_beats_frozen": p_t > p_f,
    }


def run_c5_population(*, n_pairs: int = N_PAIRS) -> dict[str, Any]:
    rows = []
    for i in range(n_pairs):
        rows.append(paired_c5_one(i, beneficial_slot=i % 2, swap_vectors=(i // 2) % 2 == 1))
    deltas = [r["delta"] for r in rows]
    n_beat = sum(1 for r in rows if r["trained_beats_frozen"])
    def _among(counts: dict[str, int], hid: str) -> float:
        n = sum(counts.values()) or 1
        return float(counts.get(hid, 0)) / float(n)

    fr_press = [_among(r["frozen"], r["press"]) for r in rows]
    fr_harm = [_among(r["frozen"], r["harm"]) for r in rows]
    obs = float(np.mean(fr_press) - np.mean(fr_harm))
    rng = np.random.default_rng(11)
    ge = 0
    for _ in range(PERM_N):
        pick = rng.integers(0, 2, size=len(rows))
        d = float(np.mean(np.where(pick == 0, fr_press, fr_harm)) - np.mean(np.where(pick == 0, fr_harm, fr_press)))
        if abs(d) >= abs(obs) - 1e-15:
            ge += 1
    perm_p = (ge + 1) / (PERM_N + 1)
    mean_delta = float(np.mean(deltas))
    mean_f_ben = float(np.mean(fr_press))
    ok = (
        n_beat >= MAJORITY_MIN
        and mean_delta >= MEAN_DELTA_MIN
        and perm_p >= PERM_ALPHA
        and abs(mean_f_ben - 0.5) <= MAX_NUISANCE_ABS
    )
    return {
        "id": "C5_plasticity_necessity",
        "ok": ok,
        "n_pairs": n_pairs,
        "n_trained_beats_frozen": n_beat,
        "majority_min": MAJORITY_MIN,
        "mean_delta": mean_delta,
        "mean_delta_min": MEAN_DELTA_MIN,
        "frozen_mean_p_press": mean_f_ben,
        "frozen_perm_p": perm_p,
        "rows": rows,
        "why": None if ok else "trained_does_not_beat_frozen_at_population_level",
    }


def run_c6_population(*, n_pairs: int = N_PAIRS) -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    slot0_rates = []
    label_rates = []
    rows = []
    for i in range(n_pairs):
        main_s, _ = pair_seeds(i)
        toks = curriculum_tokens(main_s)
        handles = [toks["press"], toks["harm"]]
        rng = np.random.default_rng(main_s.seed_motor ^ (0xC600 + i))
        order = list(handles)
        rng.shuffle(order)
        # label AFTER bind
        label_ben = str(rng.choice(order))
        label_other = order[0] if order[1] == label_ben else order[1]
        neutral = {
            "act_effects": {
                order[0]: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
                order[1]: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
            }
        }
        with tempfile.TemporaryDirectory(prefix=f"v7c6_{i}_") as tmp:
            ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
            ag.bind_actuators(order)
            teach_loop(
                ag,
                main_s,
                n=TEACH_N,
                symbols_fn=lambda j, r: [toks["a"], toks["b"]],
                latent=neutral,
            )
            counts = _frozen_pref_counts(ag, toks, N_PROBES, [toks["a"]])
        n_act = sum(counts.values()) or 1
        p_slot0 = counts.get(order[0], 0) / n_act
        p_lab = counts.get(label_ben, 0) / n_act
        slot0_rates.append(p_slot0)
        label_rates.append(p_lab)
        rows.append(
            {
                "pair_id": i,
                "order": order,
                "label_ben": label_ben,
                "counts": counts,
                "p_slot0": p_slot0,
                "p_label": p_lab,
            }
        )
    slot_eff = float(np.mean(slot0_rates) - 0.5)
    lab_eff = float(np.mean(label_rates) - 0.5)
    # permutation: shuffle rates against a dummy half-split
    rng = np.random.default_rng(13)
    ge_s = ge_l = 0
    sarr = np.asarray(slot0_rates)
    larr = np.asarray(label_rates)
    for _ in range(PERM_N):
        # reflect around 0.5 by random sign
        sign = rng.choice([-1.0, 1.0], size=sarr.size)
        if abs(float(np.mean((sarr - 0.5) * sign))) >= abs(slot_eff) - 1e-15:
            ge_s += 1
        sign = rng.choice([-1.0, 1.0], size=larr.size)
        if abs(float(np.mean((larr - 0.5) * sign))) >= abs(lab_eff) - 1e-15:
            ge_l += 1
    p_slot = (ge_s + 1) / (PERM_N + 1)
    p_lab = (ge_l + 1) / (PERM_N + 1)
    ok = (
        abs(slot_eff) <= MAX_NUISANCE_ABS
        and abs(lab_eff) <= MAX_NUISANCE_ABS
        and p_slot >= PERM_ALPHA
        and p_lab >= PERM_ALPHA
    )
    return {
        "id": "C6_no_consequence_population",
        "ok": ok,
        "n_pairs": n_pairs,
        "slot_effect": slot_eff,
        "label_effect": lab_eff,
        "slot_perm_p": p_slot,
        "label_perm_p": p_lab,
        "max_nuisance_abs": MAX_NUISANCE_ABS,
        "rows": rows,
        "why": None if ok else "slot_or_label_effect_exceeds_preregistered_max",
    }


def d1_floors(press: int, harm: int, cf_differs: bool) -> bool:
    return press >= 3 and press > harm and cf_differs


def summarize_v6_gate_failures(gate: dict[str, Any]) -> dict[str, Any]:
    d1_zero = 0
    d2_holds = 0
    n_life = 0
    for p in (gate.get("battery") or {}).get("pairs") or []:
        for role in ("main", "twin"):
            n_life += 1
            d1 = p[role]["stages"]["D1"]
            d2 = p[role]["stages"]["D2"]
            if int(d1.get("press") or 0) == 0 and int(d1.get("harm") or 0) == 0:
                d1_zero += 1
            if not d2.get("ok") and int(d2.get("holds_during_conflict") or 0) < 5:
                d2_holds += 1
    return {
        "n_lives": n_life,
        "d1_press0_harm0": d1_zero,
        "d2_holds_below_5": d2_holds,
        "n_pair_clear": (gate.get("battery") or {}).get("n_pair_clear"),
    }
