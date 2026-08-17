"""TM.0.23.CORTEX.FULLDEV.R4 — D0–D12 on sealed TM023.FULL.R4. worlds."""

from __future__ import annotations

from typing import Any

from experiments.cortex_develop_life import STAGES, run_one_life
from experiments.cortex_develop_scorers import score_d11_twin_follows
from experiments.cortex_fulldev_r4_worlds import sealed_pair_seeds


def _cross_lexicon_probe(twin: dict[str, Any], main: dict[str, Any], *, device: str) -> dict[str, Any]:
    from experiments.cortex_develop_life import _cross_lexicon_probe as _cross

    return _cross(twin, main, device=device)


def run_fulldev_pair(pair_id: int, seed_hex: str, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = sealed_pair_seeds(pair_id, seed_hex)
    main = run_one_life(main_s, device=device)
    twin = run_one_life(twin_s, device=device)
    cross = _cross_lexicon_probe(twin, main, device=device)
    d11 = score_d11_twin_follows(main, twin, cross_lexicon=cross)
    main["stages"]["D11"] = d11
    twin["stages"]["D11"] = d11
    for life in (main, twin):
        life.pop("mature_checkpoint", None)
        if isinstance(life.get("tokens"), dict):
            life["tokens"].pop("all", None)

    def full_clear(life: dict[str, Any]) -> bool:
        st = life["stages"]
        return all(st[s].get("ok") for s in STAGES)

    main_clear = full_clear(main)
    twin_clear = full_clear(twin)
    return {
        "pair_id": pair_id,
        "main": main,
        "twin": twin,
        "pair_clear": bool(main_clear and twin_clear),
        "maturation_main": main.get("d10_adult_gt_child"),
        "maturation_twin": twin.get("d10_adult_gt_child"),
        "maturation_pair": bool(main.get("d10_adult_gt_child") and twin.get("d10_adult_gt_child")),
        "d11_cross": cross,
        "worlds": "sealed_eval_seed",
        "domain": "TM023.FULL.R4.",
    }


def run_fulldev_r4_battery(*, seed_hex: str, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    pairs = [run_fulldev_pair(i, seed_hex, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    n_mat = sum(1 for p in pairs if p["maturation_pair"])
    development_gate_clear = n_clear >= 13
    maturation_ok = n_mat >= 14
    dist: dict[str, int] = {s: 0 for s in STAGES}
    first_fail: dict[str, int] = {}
    for p in pairs:
        for role in ("main", "twin"):
            for s in STAGES:
                if p[role]["stages"].get(s, {}).get("ok"):
                    dist[s] += 1
            ff = p[role].get("first_fail")
            if ff:
                first_fail[ff] = first_fail.get(ff, 0) + 1
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "n_maturation": n_mat,
        "development_gate_clear": development_gate_clear,
        "eligible_for_000005": False,
        "earned_next": False,
        "ex0s": None,
        "product": "0.0.004",
        "stage_pass_counts_main_and_twin": dist,
        "first_fail_histogram": first_fail,
        "worlds": "sealed_eval_seed",
        "domain": "TM023.FULL.R4.",
        "pairs": pairs,
        "device": device,
        "maturation_ok": maturation_ok,
    }
