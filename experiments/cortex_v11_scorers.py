"""v11 D2: HOLD counted under swapped physics. D1 and extras remain v10."""

from __future__ import annotations

import copy
from typing import Any

from experiments.cortex_develop_life import LifeSeeds
from experiments.cortex_develop_scorers import BODY0, _act_token_counts, apply_event, motor_latent, teach_loop
from experiments.cortex_v8_scorers import LIFE_DELTA_MIN, _assoc_contrast, _clone, _frozen_pref_counts, _p, N_PROBES, TEACH_N
from experiments.cortex_v10_scorers import population_d1, population_d2, score_d0_v10, score_d1_v10
from experiments.cortex_mact_boundary import _freeze_plasticity
from three_memory.neural_cortex import NeuralCortex


def score_d0_v11(ag: NeuralCortex, seeds: LifeSeeds, toks: dict[str, str]) -> dict[str, Any]:
    return score_d0_v10(ag, seeds, toks)


def score_d1_v11(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    birth_ckpt: dict[str, Any],
) -> dict[str, Any]:
    return score_d1_v10(ag, seeds, toks, birth_ckpt=birth_ckpt)


def _swapped_latent(toks: dict[str, str]) -> dict[str, Any]:
    lat = motor_latent(toks)
    lat_swap = copy.deepcopy(lat)
    a_h, b_h = toks["press"], toks["harm"]
    lat_swap["act_effects"][a_h], lat_swap["act_effects"][b_h] = (
        lat_swap["act_effects"][b_h],
        lat_swap["act_effects"][a_h],
    )
    return lat, lat_swap


def score_d2_v11(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    birth_ckpt: dict[str, Any],
) -> dict[str, Any]:
    lat, lat_swap = _swapped_latent(toks)
    teach_loop(ag, seeds, toks=toks, n=60, symbols_fn=lambda i, rng: [toks["c"]], latent=lat)
    body = list(BODY0)
    state = ["st_idle"]
    holds = 0
    for i in range(30):
        out, state, body = apply_event(
            ag,
            ix=f"d2c_{i}",
            source="src_d2",
            symbols=[toks["c"]],
            state=state,
            body=body,
            latent=lat_swap,
        )
        if (out.get("action") or {}).get("op") == "HOLD":
            holds += 1
    w_before = ag.weight_hash()
    ag.reset_rho()
    w_after = ag.weight_hash()
    teach_loop(ag, seeds, toks=toks, n=40, symbols_fn=lambda i, rng: [toks["c"]], latent=lat)
    counts = _act_token_counts(ag, toks, 30, [toks["c"]], latent=lat)
    beneficial = counts.get(toks["press"], 0) + counts.get(toks["get"], 0)
    rho_ok = w_before == w_after
    floors_ok = rho_ok and beneficial >= 3 and holds >= 5

    press_h, get_h = toks["press"], toks["get"]
    trained_rate = _clone(seeds, birth_ckpt, str(ag.device))
    frozen_rate = _clone(seeds, birth_ckpt, str(ag.device))
    _freeze_plasticity(frozen_rate)
    teach_loop(trained_rate, seeds, n=TEACH_N, symbols_fn=lambda i, rng: [toks["c"]], latent=motor_latent(toks))
    teach_loop(frozen_rate, seeds, n=TEACH_N, symbols_fn=lambda i, rng: [toks["c"]], latent=motor_latent(toks))
    ct = _frozen_pref_counts(trained_rate, toks, N_PROBES, [toks["c"]])
    cf = _frozen_pref_counts(frozen_rate, toks, N_PROBES, [toks["c"]])
    p_t = _p(ct, press_h) + _p(ct, get_h)
    p_f = _p(cf, press_h) + _p(cf, get_h)
    trained_ag = _clone(seeds, birth_ckpt, str(ag.device))
    frozen_ag = _clone(seeds, birth_ckpt, str(ag.device))
    _freeze_plasticity(frozen_ag)
    contrast_t = _assoc_contrast(trained_ag, seeds, toks)
    contrast_f = _assoc_contrast(frozen_ag, seeds, toks)
    extras_ok = (p_t - p_f) >= LIFE_DELTA_MIN and contrast_t > contrast_f
    return {
        "stage": "D2",
        "ok": bool(floors_ok),
        "floors_ok": floors_ok,
        "holds_during_conflict": holds,
        "conflict_latent": "swapped_press_harm",
        "rho_reset_preserves_weights": rho_ok,
        "beneficial_act": beneficial,
        "counts": counts,
        "extras_ok": extras_ok,
        "extras_veto_life": False,
        "p_ben_trained": p_t,
        "p_ben_frozen": p_f,
        "trained_gt_frozen": (p_t - p_f) >= LIFE_DELTA_MIN,
        "assoc_trained": contrast_t,
        "assoc_frozen": contrast_f,
        "assoc_ok": contrast_t > contrast_f,
        "life_delta_min": LIFE_DELTA_MIN,
        "probe": "frozen",
    }


__all__ = [
    "score_d0_v11",
    "score_d1_v11",
    "score_d2_v11",
    "population_d1",
    "population_d2",
]
