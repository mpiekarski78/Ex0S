"""Developmental Birth R4-R2 unit / ownership / behavioral-gate / partition tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.developmental_birth_r4_r2_life import (
    evaluate_matched_factorial,
    evaluate_r4_r2_life,
)
from experiments.dev1.developmental_birth_r4_r2_outer import MatchedOuterBudget, run_matched_es_smoke
from experiments.dev1.developmental_birth_r4_r2_preflight import (
    NURSERY_ENG_SEEDS,
    PREFLIGHT_SEED,
    ownership_leakage_check,
    partitions_exclude_engineering_seeds,
    sham_vs_active_compute_match_check,
)
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import GestationMode
from three_memory.dev1.nursery_v2.construction import construct_nursery_organism
from three_memory.dev1.nursery_v2.gestation import run_nursery_gestation
from three_memory.dev1.nursery_v2.metrics import (
    AC_RETRAIN_INITIALIZATION_SEEDS,
    AC_RETRAIN_N_INITIALIZATIONS,
    BehavioralEpisodeGates,
    aggregate_behavioral_gates,
    best_of_ac_initializations,
)
from three_memory.dev1.nursery_v2.world import analytic_reachability_report, reachability_chi
from three_memory.dev1.nursery_v2.physics import BodyConfig


def test_reachability_chi_conventions():
    cfg = BodyConfig()
    chi = reachability_chi(1.05, cfg, 16)
    assert chi <= 1.0
    assert chi <= 0.85
    conv = json.loads(Path("docs/exos_dev1.stage_a_nursery_body_v2.conventions.lock").read_text())
    assert conv["reachability"]["analytically_reachable"] == "chi <= 1"
    assert conv["reachability"]["safety_rule"] == "chi <= 0.85"
    report = analytic_reachability_report("r4_r2_chi_test", n_episodes=8, episode_ticks=16)
    assert report["safety_rule"] == "chi <= 0.85"
    assert report["analytical_reachable_rule"] == "chi <= 1"


def test_behavioral_gates_independent():
    eps = [
        BehavioralEpisodeGates(True, False, 1.0, 0.5),
        BehavioralEpisodeGates(True, True, 1.2, 0.2),
        BehavioralEpisodeGates(False, False, 0.9, 0.8),
    ]
    agg = aggregate_behavioral_gates(eps)
    assert abs(agg["ever_reached_rate"] - 2 / 3) < 1e-9
    assert abs(agg["end_in_zone_rate"] - 1 / 3) < 1e-9
    assert agg["distance_reduction"] > 0
    conv = json.loads(Path("docs/exos_dev1.stage_a_nursery_body_v2.conventions.lock").read_text())
    assert conv["behavioral_gates"]["tick_fraction_comfort_retired"] is True
    assert conv["behavioral_gates"]["primary_scored_episode_success_for_r4_r2"] == "end_in_zone"


def test_ac_retrain_fixed_battery_not_retry_until_pass():
    rows = [
        {"init_seed": 0, "end_in_zone_rate": 0.1, "distance_reduction": 0.05},
        {"init_seed": 1, "end_in_zone_rate": 0.4, "distance_reduction": 0.10},
        {"init_seed": 2, "end_in_zone_rate": 0.4, "distance_reduction": 0.20},
    ]
    best = best_of_ac_initializations(rows)
    assert best["init_seed"] == 2
    assert best["n_initializations_run"] == 3
    assert AC_RETRAIN_N_INITIALIZATIONS == 3
    assert AC_RETRAIN_INITIALIZATION_SEEDS == (0, 1, 2)
    conv = json.loads(Path("docs/exos_dev1.stage_a_nursery_body_v2.conventions.lock").read_text())
    assert "retry_until_ceiling_passes" in conv["ac_retrain_tolerance"]["forbidden"]


def test_same_pre_gestation_checkpoint():
    cells = evaluate_matched_factorial(
        "r4_r2_test_factorial",
        n_episodes=1,
        episode_ticks=2,
        embryonic_seed=0,
        device=torch.device("cpu"),
    )
    assert len(cells) == 4
    for credit in ("r2_fixed_eprop_baseline", "inherited_learning_signal_generator"):
        sham = cells[f"sham_gestation__{credit}"]
        active = cells[f"active_gestation__{credit}"]
        assert sham.pre_gestation_checkpoint_hash == active.pre_gestation_checkpoint_hash


def test_primary_metric_is_end_in_zone():
    m = evaluate_r4_r2_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "r4_r2_test_metric",
        n_episodes=2,
        episode_ticks=4,
        use_teacher=False,
        device=torch.device("cpu"),
    )
    assert m.treatment_accuracy == m.end_in_zone_rate
    assert "ever_reached_rate" in m.life_record
    assert m.life_record["tick_fraction_comfort_retired"] is True


def test_gestational_and_lifetime_plasticity_off_separated():
    g = GenerativeGenome.small(embryonic_seed=5)
    g.gestation_ticks = 12
    org, _ = construct_nursery_organism(g, device=torch.device("cpu"))
    _, receipt = run_nursery_gestation(
        org, g, GestationMode.ACTIVE, body_seed=1, gestational_plasticity_off=True
    )
    assert receipt.plasticity_updates == 0
    m = evaluate_r4_r2_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "r4_r2_test_lpo",
        n_episodes=2,
        episode_ticks=4,
        lifetime_plasticity_off=True,
        use_teacher=False,
        device=torch.device("cpu"),
    )
    assert m.plasticity_updates == 0
    assert m.intervention == "lifetime_plasticity_off"


def test_lsg_and_open_loop_controls():
    assert (
        evaluate_r4_r2_life(
            "active_gestation",
            "inherited_learning_signal_generator",
            "r4_r2_lsg_off",
            n_episodes=1,
            episode_ticks=2,
            lsg_off=True,
            use_teacher=False,
            device=torch.device("cpu"),
        ).intervention
        == "lsg_off"
    )
    assert (
        evaluate_r4_r2_life(
            "active_gestation",
            "inherited_learning_signal_generator",
            "r4_r2_lsg_perm",
            n_episodes=1,
            episode_ticks=2,
            lsg_permuted=True,
            use_teacher=False,
            device=torch.device("cpu"),
        ).intervention
        == "lsg_permuted"
    )
    assert (
        evaluate_r4_r2_life(
            "active_gestation",
            "r2_fixed_eprop_baseline",
            "r4_r2_ol",
            n_episodes=1,
            episode_ticks=2,
            open_loop=True,
            use_teacher=False,
            device=torch.device("cpu"),
        ).intervention
        == "open_loop"
    )


def test_matched_outer_budgets():
    out = run_matched_es_smoke(
        "r4_r2_es",
        MatchedOuterBudget(population=2, generations=1, n_episodes=1, episode_ticks=2),
        device=torch.device("cpu"),
    )
    assert out["matched"]
    assert out["same_nursery_v2_trajectories_and_budgets"]


def test_ownership_and_sham_active():
    assert ownership_leakage_check()["ok"]
    assert sham_vs_active_compute_match_check()["ok"]


def test_engineering_seeds_excluded_from_partitions():
    discovery = [f"exos_dev1_developmental_birth_r4_r2_world_{i:03d}" for i in range(1, 7)]
    conf = [f"exos_dev1_developmental_birth_r4_r2_conf_{i:03d}" for i in range(1, 5)]
    out = partitions_exclude_engineering_seeds(discovery[:2], discovery[2:], conf)
    assert out["ok"]
    for s in NURSERY_ENG_SEEDS:
        assert s not in discovery and s not in conf
    assert PREFLIGHT_SEED not in discovery


def test_prereg_scored_authorized():
    prereg = json.loads(
        Path("docs/exos_dev1.stage_a_developmental_birth_r4_r2.prereg.lock").read_text()
    )
    assert prereg["scored_run_authorized"] is True
    assert prereg["body_world"]["version"] == "NurseryBodyV2"
