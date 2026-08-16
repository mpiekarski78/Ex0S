"""TM.0.23.CORTEX.V2.GATE — D0–D2 only on make_cortex interface (no D3–D12).

Frozen before candidate v2 exists. Pins interface, not candidate SHA.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_develop_life import (
    LifeSeeds,
    curriculum_tokens,
    development_seed_table,
    pair_seeds,
    teach_loop,
)
from experiments.cortex_develop_scorers import score_d0, score_d1, score_d2
from experiments.run_tm023cortex import make_cortex
from three_memory.neural_cortex import GenomeConfig, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["D0", "D1", "D2"]

# Frozen thresholds (must match scorers / gate contract)
THRESHOLDS = {
    "D1": {"press_min": 3, "press_gt_harm": True, "cf_differs": True},
    "D2": {"holds_min": 5, "beneficial_min": 3, "rho_reset_preserves_weights": True},
    "D0": {"n_probes": 64, "p0": 0.5, "alpha": 0.01},
    "pair_clear_requires": "main_D1_and_D2_and_twin_D1_and_D2",
    "gate_clear_min_pairs": 13,
    "n_pairs": 16,
}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def life_clears_d1_d2(stages: dict[str, Any]) -> bool:
    return bool(stages.get("D1", {}).get("ok")) and bool(stages.get("D2", {}).get("ok"))


def run_gate_life(seeds: LifeSeeds, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"v2gate_{seeds.role}_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=device)
        toks = curriculum_tokens(seeds)
        stages: dict[str, Any] = {}
        stages["D0"] = score_d0(ag, seeds, toks)
        teach_loop(ag, seeds, n=30, symbols_fn=lambda i, rng: [toks["a"], toks["b"]])
        stages["D1"] = score_d1(ag, seeds, toks)
        stages["D2"] = score_d2(ag, seeds, toks)
        d0_ok = bool(stages["D0"].get("ok"))
        d12_ok = life_clears_d1_d2(stages)
        return {
            "role": seeds.role,
            "pair_id": seeds.pair_id,
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
        main["d0_ok"]
        and twin["d0_ok"]
        and main["d1_d2_ok"]
        and twin["d1_d2_ok"]
    )
    return {
        "pair_id": pair_id,
        "main": main,
        "twin": twin,
        "pair_clear": pair_clear,
    }


def run_v2_gate_battery(*, n_pairs: int = 16, device: str = "cpu") -> dict[str, Any]:
    pairs = [run_gate_pair(i, device=device) for i in range(n_pairs)]
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    # Systematic D0 failure: majority of lives fail D0
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
    gate_clear = (n_clear >= THRESHOLDS["gate_clear_min_pairs"]) and (not systematic_d0_fail)
    return {
        "n_pairs": n_pairs,
        "n_pair_clear": n_clear,
        "sensorimotor_association_gate_clear": gate_clear,
        "systematic_d0_birth_leakage_failure": systematic_d0_fail,
        "d0_fail_lives": d0_fails,
        "stage_pass_counts_main_and_twin": stage_counts,
        "thresholds": THRESHOLDS,
        "pairs": pairs,
        "device": device,
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "seed_table_fingerprint": hashlib.sha256(
            json.dumps(development_seed_table(n_pairs), sort_keys=True).encode()
        ).hexdigest(),
    }
