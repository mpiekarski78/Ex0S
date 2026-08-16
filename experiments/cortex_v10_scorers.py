"""v10 D1/D2: per-life floors only; extras recorded for population aggregation."""

from __future__ import annotations

from typing import Any

from experiments.cortex_develop_life import LifeSeeds
from experiments.cortex_v7_stats import MAJORITY_MIN, MEAN_DELTA_MIN
from experiments.cortex_v8_scorers import LIFE_DELTA_MIN, score_d0_v8, score_d1_v8, score_d2_v8
from three_memory.neural_cortex import NeuralCortex

MAJORITY = MAJORITY_MIN
MEAN_MIN = MEAN_DELTA_MIN
DELTA = LIFE_DELTA_MIN


def score_d0_v10(ag: NeuralCortex, seeds: LifeSeeds, toks: dict[str, str]) -> dict[str, Any]:
    return score_d0_v8(ag, seeds, toks)


def score_d1_v10(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    birth_ckpt: dict[str, Any],
) -> dict[str, Any]:
    r = score_d1_v8(ag, seeds, toks, birth_ckpt=birth_ckpt)
    extras_ok = bool(r.get("trained_gt_birth") and r.get("trained_gt_frozen"))
    return {**r, "ok": bool(r.get("floors_ok")), "extras_ok": extras_ok, "extras_veto_life": False}


def score_d2_v10(
    ag: NeuralCortex,
    seeds: LifeSeeds,
    toks: dict[str, str],
    *,
    birth_ckpt: dict[str, Any],
) -> dict[str, Any]:
    r = score_d2_v8(ag, seeds, toks, birth_ckpt=birth_ckpt)
    extras_ok = bool(r.get("trained_gt_frozen") and r.get("assoc_ok"))
    return {**r, "ok": bool(r.get("floors_ok")), "extras_ok": extras_ok, "extras_veto_life": False}


def population_d1(lives: list[dict[str, Any]]) -> dict[str, Any]:
    db = [float(x["p_trained"]) - float(x["p_birth"]) for x in lives]
    df = [float(x["p_trained"]) - float(x["p_frozen"]) for x in lives]
    n_b = sum(1 for d in db if d >= DELTA)
    n_f = sum(1 for d in df if d >= DELTA)
    mean_b = sum(db) / len(db)
    mean_f = sum(df) / len(df)
    ok = n_b >= MAJORITY and n_f >= MAJORITY and mean_b >= MEAN_MIN and mean_f >= MEAN_MIN
    return {
        "n": len(lives),
        "n_trained_beats_birth": n_b,
        "n_trained_beats_frozen": n_f,
        "majority_min": MAJORITY,
        "mean_delta_birth": mean_b,
        "mean_delta_frozen": mean_f,
        "mean_delta_min": MEAN_MIN,
        "life_delta_min": DELTA,
        "ok": ok,
    }


def population_d2(lives: list[dict[str, Any]]) -> dict[str, Any]:
    df = [float(x["p_ben_trained"]) - float(x["p_ben_frozen"]) for x in lives]
    n_f = sum(1 for d in df if d >= DELTA)
    n_a = sum(1 for x in lives if x.get("assoc_ok"))
    mean_f = sum(df) / len(df)
    ok = n_f >= MAJORITY and n_a >= MAJORITY and mean_f >= MEAN_MIN
    return {
        "n": len(lives),
        "n_trained_beats_frozen": n_f,
        "n_assoc_ok": n_a,
        "majority_min": MAJORITY,
        "mean_delta_frozen": mean_f,
        "mean_delta_min": MEAN_MIN,
        "life_delta_min": DELTA,
        "ok": ok,
    }
