"""v8 D1/D2: floors plus birth-weight frozen-probe extras with life_delta_min."""

from __future__ import annotations

import copy
from typing import Any

from experiments.cortex_develop_life import LifeSeeds, motor_latent, teach_loop
from experiments.cortex_develop_scorers import score_d0, score_d2
from experiments.cortex_mact_boundary import _frozen_pref_counts, _freeze_plasticity
from experiments.run_tm023cortex import make_cortex, physics
from experiments.cortex_develop_scorers import BODY0
from three_memory.neural_cortex import NeuralCortex

N_PROBES = 40
TEACH_N = 80
LIFE_DELTA_MIN = 0.10
ASSOC_TEACH = 40
ASSOC_PROBES = 20


def _clone(seeds: LifeSeeds, ckpt: dict[str, Any], device: str) -> NeuralCortex:
    ag = make_cortex(None, genome=seeds.genome(), device=device)
    ag.load_checkpoint(ckpt)
    return ag


def _p(counts: dict[str, int], hid: str, n: int = N_PROBES) -> float:
    return float(counts.get(hid, 0)) / float(n)


def score_d1_v8(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    birth_ckpt: dict[str, Any],
) -> dict[str, Any]:
    press_h, harm_h = toks["press"], toks["harm"]
    birth_ag = _clone(seeds, birth_ckpt, str(ag.device))
    birth = _frozen_pref_counts(birth_ag, toks, N_PROBES, [toks["a"]])
    trained_ag = _clone(seeds, birth_ckpt, str(ag.device))
    teach_loop(trained_ag, seeds, n=TEACH_N, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=motor_latent(toks))
    trained = _frozen_pref_counts(trained_ag, toks, N_PROBES, [toks["a"]])
    frozen_ag = _clone(seeds, birth_ckpt, str(ag.device))
    _freeze_plasticity(frozen_ag)
    teach_loop(frozen_ag, seeds, n=TEACH_N, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=motor_latent(toks))
    frozen = _frozen_pref_counts(frozen_ag, toks, N_PROBES, [toks["a"]])
    p_b, p_t, p_f = _p(birth, press_h), _p(trained, press_h), _p(frozen, press_h)
    press = int(trained.get(press_h, 0))
    harm = int(trained.get(harm_h, 0))
    lat = motor_latent(toks)
    s1, b1 = physics(BODY0, press_h, lat)
    s2, b2 = physics(BODY0, harm_h, lat)
    cf_differs = (s1 != s2) or (b1 != b2)
    floors_ok = press >= 3 and press > harm and cf_differs
    extras_ok = (p_t - p_b) >= LIFE_DELTA_MIN and (p_t - p_f) >= LIFE_DELTA_MIN
    ag.load_checkpoint(trained_ag.checkpoint())
    return {
        "stage": "D1",
        "ok": bool(floors_ok and extras_ok),
        "floors_ok": floors_ok,
        "press": press,
        "harm": harm,
        "cf_differs": cf_differs,
        "p_birth": p_b,
        "p_trained": p_t,
        "p_frozen": p_f,
        "trained_gt_birth": (p_t - p_b) >= LIFE_DELTA_MIN,
        "trained_gt_frozen": (p_t - p_f) >= LIFE_DELTA_MIN,
        "life_delta_min": LIFE_DELTA_MIN,
        "probe": "frozen",
    }


def _assoc_contrast(ag: NeuralCortex, seeds: LifeSeeds, toks: dict[str, str]) -> float:
    a_h, b_h = toks["press"], toks["harm"]
    lat = motor_latent(toks)
    lat_swap = copy.deepcopy(lat)
    lat_swap["act_effects"][a_h], lat_swap["act_effects"][b_h] = (
        lat_swap["act_effects"][b_h],
        lat_swap["act_effects"][a_h],
    )
    ckpt = ag.checkpoint()
    teach_loop(ag, seeds, n=ASSOC_TEACH, symbols_fn=lambda i, rng: [toks["c"]], latent=lat)
    p_ben = _p(_frozen_pref_counts(ag, toks, ASSOC_PROBES, [toks["c"]]), a_h, ASSOC_PROBES)
    ag.load_checkpoint(ckpt)
    teach_loop(ag, seeds, n=ASSOC_TEACH, symbols_fn=lambda i, rng: [toks["c"]], latent=lat_swap)
    p_harm = _p(_frozen_pref_counts(ag, toks, ASSOC_PROBES, [toks["c"]]), a_h, ASSOC_PROBES)
    ag.load_checkpoint(ckpt)
    return float(p_ben - p_harm)


def score_d2_v8(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    birth_ckpt: dict[str, Any],
) -> dict[str, Any]:
    d2 = score_d2(ag, seeds, toks)
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
    ok = bool(d2.get("ok")) and extras_ok
    return {
        **d2,
        "ok": ok,
        "floors_ok": bool(d2.get("ok")),
        "p_ben_trained": p_t,
        "p_ben_frozen": p_f,
        "trained_gt_frozen": (p_t - p_f) >= LIFE_DELTA_MIN,
        "assoc_trained": contrast_t,
        "assoc_frozen": contrast_f,
        "assoc_ok": contrast_t > contrast_f,
        "life_delta_min": LIFE_DELTA_MIN,
        "probe": "frozen",
    }


def score_d0_v8(ag: NeuralCortex, seeds: LifeSeeds, toks: dict[str, str]) -> dict[str, Any]:
    return score_d0(ag, seeds, toks)
