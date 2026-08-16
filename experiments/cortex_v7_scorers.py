"""v7 D1/D2: existing floors plus paired birth/frozen baselines and D2 association."""

from __future__ import annotations

import copy
from typing import Any

from experiments.cortex_develop_life import LifeSeeds, motor_latent, teach_loop
from experiments.cortex_develop_scorers import score_d0, score_d1, score_d2
from experiments.cortex_mact_boundary import _frozen_pref_counts, _freeze_plasticity
from experiments.run_tm023cortex import make_cortex
from three_memory.neural_cortex import NeuralCortex

ASSOC_TEACH = 40
ASSOC_PROBES = 20


def _clone(seeds: LifeSeeds, ckpt: dict[str, Any], device: str) -> NeuralCortex:
    ag = make_cortex(None, genome=seeds.genome(), device=device)
    ag.load_checkpoint(ckpt)
    return ag


def score_d1_v7(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    press_birth: int,
) -> dict[str, Any]:
    """Floors from score_d1; extras require trained > post-bind birth and > paired frozen.

    `press_birth` must be a frozen-probe press count taken after bind and before teach.
    """
    ckpt = ag.checkpoint()
    d1 = score_d1(ag, seeds, toks)
    frozen_ag = _clone(seeds, ckpt, str(ag.device))
    _freeze_plasticity(frozen_ag)
    d1_f = score_d1(frozen_ag, seeds, toks)
    press_t = int(d1.get("press") or 0)
    press_f = int(d1_f.get("press") or 0)
    extras_ok = press_t > int(press_birth) and press_t > press_f
    ok = bool(d1.get("ok")) and extras_ok
    return {
        **d1,
        "ok": ok,
        "floors_ok": bool(d1.get("ok")),
        "press_birth": int(press_birth),
        "press_frozen": press_f,
        "trained_gt_birth": press_t > int(press_birth),
        "trained_gt_frozen": press_t > press_f,
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
    p_ben = _frozen_pref_counts(ag, toks, ASSOC_PROBES, [toks["c"]]).get(a_h, 0)
    ag.load_checkpoint(ckpt)
    teach_loop(ag, seeds, n=ASSOC_TEACH, symbols_fn=lambda i, rng: [toks["c"]], latent=lat_swap)
    p_harm = _frozen_pref_counts(ag, toks, ASSOC_PROBES, [toks["c"]]).get(a_h, 0)
    ag.load_checkpoint(ckpt)
    return float(p_ben - p_harm)


def score_d2_v7(ag: NeuralCortex, seeds: LifeSeeds, toks: dict[str, str]) -> dict[str, Any]:
    ckpt = ag.checkpoint()
    d2 = score_d2(ag, seeds, toks)
    frozen_ag = _clone(seeds, ckpt, str(ag.device))
    _freeze_plasticity(frozen_ag)
    d2_f = score_d2(frozen_ag, seeds, toks)
    trained_ag = _clone(seeds, ckpt, str(ag.device))
    contrast_t = _assoc_contrast(trained_ag, seeds, toks)
    contrast_f = _assoc_contrast(frozen_ag, seeds, toks)
    extras_ok = int(d2.get("beneficial_act") or 0) > int(d2_f.get("beneficial_act") or 0) and contrast_t > contrast_f
    ok = bool(d2.get("ok")) and extras_ok
    return {
        **d2,
        "ok": ok,
        "floors_ok": bool(d2.get("ok")),
        "beneficial_frozen": d2_f.get("beneficial_act"),
        "trained_gt_frozen": int(d2.get("beneficial_act") or 0) > int(d2_f.get("beneficial_act") or 0),
        "assoc_trained": contrast_t,
        "assoc_frozen": contrast_f,
        "assoc_ok": contrast_t > contrast_f,
    }


def score_d0_v7(ag: NeuralCortex, seeds: LifeSeeds, toks: dict[str, str]) -> dict[str, Any]:
    return score_d0(ag, seeds, toks)
