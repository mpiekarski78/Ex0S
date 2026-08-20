"""
Developmental Birth R4-R2 preflight + excluded-seed battery.

Engineering/certification nursery seeds permanently excluded from scored partitions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.developmental_birth_r4_r2_ceiling import evaluate_ceiling_gate_bundle
from experiments.dev1.developmental_birth_r4_r2_life import (
    evaluate_matched_factorial,
    evaluate_r4_r2_life,
)
from experiments.dev1.developmental_birth_r4_r2_outer import MatchedOuterBudget, run_matched_es_smoke
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import GestationMode
from three_memory.dev1.nursery_v2.construction import construct_nursery_organism
from three_memory.dev1.nursery_v2.gestation import run_nursery_gestation
from three_memory.dev1.nursery_v2.metrics import (
    AC_RETRAIN_INITIALIZATION_SEEDS,
    AC_RETRAIN_N_INITIALIZATIONS,
)
from three_memory.dev1.nursery_v2.world import analytic_reachability_report

NURSERY_ENG_SEEDS = (
    "nursery_v2_eng_seed_001",
    "nursery_v2_eng_seed_002",
    "nursery_v2_eng_seed_003",
)

EXCLUDED_LINEAGE_SEEDS = (
    "exos_dev1_developmental_birth_r4_world_001",
    "exos_dev1_developmental_birth_r4_world_002",
    "exos_dev1_developmental_birth_r4_world_003",
    "exos_dev1_developmental_birth_r4_world_004",
    "exos_dev1_developmental_birth_r4_world_005",
    "exos_dev1_developmental_birth_r4_world_006",
    "exos_dev1_developmental_birth_r4_conf_001",
    "exos_dev1_developmental_birth_r4_conf_002",
    "exos_dev1_developmental_birth_r4_conf_003",
    "exos_dev1_developmental_birth_r4_conf_004",
    "exos_dev1_developmental_birth_r4_r1_world_001",
    "exos_dev1_developmental_birth_r4_r1_world_002",
    "exos_dev1_developmental_birth_r4_r1_world_003",
    "exos_dev1_developmental_birth_r4_r1_world_004",
    "exos_dev1_developmental_birth_r4_r1_world_005",
    "exos_dev1_developmental_birth_r4_r1_world_006",
    "exos_dev1_developmental_birth_r4_r1_conf_001",
    "exos_dev1_developmental_birth_r4_r1_conf_002",
    "exos_dev1_developmental_birth_r4_r1_conf_003",
    "exos_dev1_developmental_birth_r4_r1_conf_004",
    "developmental_birth_r4_excluded_preflight_20260820",
    "developmental_birth_r4_r1_excluded_ceiling_preflight_20260820",
)

PREFLIGHT_SEED = "developmental_birth_r4_r2_excluded_preflight_20260820"


def ownership_leakage_check(tag: str = "r4_r2_leak") -> dict[str, Any]:
    g = GenerativeGenome.small(embryonic_seed=1)
    blob = json.dumps(g.to_dict())
    forbidden = ["cue_", "fixture", "expected_action", "validation_seed", "confirm_"]
    hits = [f for f in forbidden if f in blob]
    org, receipt = construct_nursery_organism(g, device=torch.device("cpu"))
    assert org.r4_use_organism_valence is True
    return {
        "tag": tag,
        "forbidden_hits": hits,
        "ok": len(hits) == 0,
        "genome_hash": receipt.generative_genome_hash,
        "pre_gestation_checkpoint_hash": receipt.pre_gestation_checkpoint_hash,
        "synergy_projection": receipt.metadata.get("synergy_projection"),
    }


def sham_vs_active_compute_match_check() -> dict[str, Any]:
    g = GenerativeGenome.small(embryonic_seed=2)
    g.gestation_ticks = 16
    org, _ = construct_nursery_organism(g, device=torch.device("cpu"))
    _, sham_r = run_nursery_gestation(org, g, GestationMode.SHAM, body_seed=7)
    _, act_r = run_nursery_gestation(org, g, GestationMode.ACTIVE, body_seed=7)
    return {
        "same_ticks": sham_r.ticks == act_r.ticks,
        "sham_plasticity_updates": sham_r.plasticity_updates,
        "active_plasticity_updates": act_r.plasticity_updates,
        "sham_no_credit": sham_r.plasticity_updates == 0,
        "active_has_credit": act_r.plasticity_updates > 0,
        "ok": (
            sham_r.ticks == act_r.ticks == g.gestation_ticks
            and sham_r.plasticity_updates == 0
            and act_r.plasticity_updates > 0
        ),
        "same_pre_gestation_via_clone": True,
    }


def partitions_exclude_engineering_seeds(
    discovery: list[str],
    validation: list[str],
    confirmation: list[str],
) -> dict[str, Any]:
    scored = set(discovery) | set(validation) | set(confirmation)
    eng_hits = sorted(scored & set(NURSERY_ENG_SEEDS))
    lineage_hits = sorted(scored & set(EXCLUDED_LINEAGE_SEEDS))
    return {
        "ok": len(eng_hits) == 0 and len(lineage_hits) == 0 and PREFLIGHT_SEED not in scored,
        "engineering_seed_hits": eng_hits,
        "lineage_seed_hits": lineage_hits,
        "preflight_seed_excluded": PREFLIGHT_SEED not in scored,
    }


def run_preflight(device: torch.device | None = None) -> dict[str, Any]:
    dev = device or torch.device("cpu")
    out: dict[str, Any] = {}
    out["ownership"] = ownership_leakage_check()
    out["sham_active"] = sham_vs_active_compute_match_check()
    out["reachability"] = analytic_reachability_report(
        PREFLIGHT_SEED,
        n_episodes=16,
        episode_ticks=16,
        safety_margin=0.85,
        device=dev,
    )
    out["factorial"] = {
        k: {
            "end_in_zone": v.end_in_zone_rate,
            "ever_reached": v.ever_reached_rate,
            "distance_reduction": v.distance_reduction,
            "pre_gestation": v.pre_gestation_checkpoint_hash,
            "credit": v.credit,
            "development": v.development,
        }
        for k, v in evaluate_matched_factorial(
            PREFLIGHT_SEED + ":factorial",
            n_episodes=2,
            episode_ticks=4,
            embryonic_seed=0,
            device=dev,
        ).items()
    }
    # Same pre-gestation checkpoint across sham/active within each credit column
    pre_ok = True
    for credit in ("r2_fixed_eprop_baseline", "inherited_learning_signal_generator"):
        sham = out["factorial"][f"sham_gestation__{credit}"]["pre_gestation"]
        active = out["factorial"][f"active_gestation__{credit}"]["pre_gestation"]
        pre_ok = pre_ok and sham == active
    out["same_pre_gestation_checkpoint"] = {"ok": pre_ok}

    out["gestational_plasticity_off"] = evaluate_r4_r2_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        PREFLIGHT_SEED + ":gpo",
        n_episodes=1,
        episode_ticks=4,
        gestational_plasticity_off=True,
        use_teacher=False,
        device=dev,
    ).life_record["gestation_plasticity_updates"]
    out["lifetime_plasticity_off"] = evaluate_r4_r2_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        PREFLIGHT_SEED + ":lpo",
        n_episodes=2,
        episode_ticks=4,
        lifetime_plasticity_off=True,
        use_teacher=False,
        device=dev,
    ).plasticity_updates
    out["lsg_off"] = evaluate_r4_r2_life(
        "active_gestation",
        "inherited_learning_signal_generator",
        PREFLIGHT_SEED + ":lsg_off",
        n_episodes=2,
        episode_ticks=4,
        lsg_off=True,
        use_teacher=False,
        device=dev,
    ).intervention
    out["lsg_permuted"] = evaluate_r4_r2_life(
        "active_gestation",
        "inherited_learning_signal_generator",
        PREFLIGHT_SEED + ":lsg_perm",
        n_episodes=1,
        episode_ticks=4,
        lsg_permuted=True,
        use_teacher=False,
        device=dev,
    ).intervention
    out["open_loop"] = evaluate_r4_r2_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        PREFLIGHT_SEED + ":ol",
        n_episodes=1,
        episode_ticks=4,
        open_loop=True,
        use_teacher=False,
        device=dev,
    ).intervention
    out["reward_valence_off"] = evaluate_r4_r2_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        PREFLIGHT_SEED + ":rv",
        n_episodes=1,
        episode_ticks=4,
        reward_valence_off=True,
        use_teacher=False,
        device=dev,
    ).intervention

    out["ceiling"] = evaluate_ceiling_gate_bundle(
        GenerativeGenome.small(),
        PREFLIGHT_SEED + ":ceiling",
        n_episodes=4,
        episode_ticks=8,
        device=dev,
        train_episodes=16,
    )
    out["ac_battery"] = {
        "n_initializations": AC_RETRAIN_N_INITIALIZATIONS,
        "seeds": list(AC_RETRAIN_INITIALIZATION_SEEDS),
        "ok": (
            out["ceiling"]["n_initializations_run"] == 3
            and out["ceiling"]["ac_retrain_tolerance"]["not_retry_until_pass"]
        ),
    }
    out["matched_es"] = run_matched_es_smoke(
        PREFLIGHT_SEED + ":es",
        MatchedOuterBudget(population=2, generations=1, n_episodes=2, episode_ticks=4),
        device=dev,
    )
    discovery = [f"exos_dev1_developmental_birth_r4_r2_world_{i:03d}" for i in range(1, 7)]
    confirmation = [f"exos_dev1_developmental_birth_r4_r2_conf_{i:03d}" for i in range(1, 5)]
    out["partitions"] = partitions_exclude_engineering_seeds(
        discovery[:2], discovery[2:], confirmation
    )
    out["ok"] = (
        out["ownership"]["ok"]
        and out["sham_active"]["ok"]
        and out["reachability"]["pass"]
        and out["same_pre_gestation_checkpoint"]["ok"]
        and out["lifetime_plasticity_off"] == 0
        and out["gestational_plasticity_off"] == 0
        and out["matched_es"]["matched"]
        and out["ac_battery"]["ok"]
        and out["partitions"]["ok"]
    )
    return out


def run_excluded_preflight_and_benchmark(device: torch.device | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    report = run_preflight(device=device)
    wall = time.perf_counter() - t0
    return {
        "preflight_seed": PREFLIGHT_SEED,
        "passed": bool(report["ok"]),
        "decision_code": (
            "developmental_birth_r4_r2_excluded_preflight_pass"
            if report["ok"]
            else "developmental_birth_r4_r2_excluded_preflight_fail"
        ),
        "wall_s": wall,
        "checks": {
            "ownership": report["ownership"]["ok"],
            "sham_active": report["sham_active"]["ok"],
            "reachability_chi_le_0_85": report["reachability"]["pass"],
            "same_pre_gestation": report["same_pre_gestation_checkpoint"]["ok"],
            "matched_outer": report["matched_es"]["matched"],
            "ac_fixed_battery": report["ac_battery"]["ok"],
            "partitions_exclude_eng": report["partitions"]["ok"],
            "lifetime_plasticity_off_zero_updates": report["lifetime_plasticity_off"] == 0,
            "gestational_plasticity_off_zero_updates": report["gestational_plasticity_off"] == 0,
        },
        "metrics": {
            "ceiling_end_in_zone": report["ceiling"]["end_in_zone_rate"],
            "ceiling_ever_reached": report["ceiling"]["ever_reached_rate"],
            "ceiling_distance_reduction": report["ceiling"]["distance_reduction"],
            "fraction_reachable": report["reachability"]["fraction_reachable"],
            "safety_rule": report["reachability"]["safety_rule"],
        },
        "report": report,
        "engineering_seeds_permanently_excluded": list(NURSERY_ENG_SEEDS),
        "scored_run_authorized": False,
    }


def main() -> None:
    report = run_excluded_preflight_and_benchmark()
    out_dir = Path("runs/exos_dev1/stage_a_developmental_birth_r4_r2")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "excluded_preflight_and_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"ok": report["passed"], "path": str(path), "wall_s": report["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
