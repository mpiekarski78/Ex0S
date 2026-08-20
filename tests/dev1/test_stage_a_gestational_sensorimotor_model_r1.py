"""GSM-R1 protocol-only revision tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.gsm_life import evaluate_gsm_life
from experiments.dev1.search_gestational_sensorimotor_model_r1 import apply_decision_ladder


MECHANISM_SHA = "8129ccff11177263159f3d76342317288886c481"


def test_mechanism_package_matches_pin():
    import hashlib
    import subprocess

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD:three_memory/dev1/gsm"], text=True
    ).strip()
    # Tree may include __pycache__; compare file contents to pin commit instead.
    diff = subprocess.check_output(
        ["git", "diff", "--stat", "8129ccf", "--", "three_memory/dev1/gsm/"], text=True
    ).strip()
    assert diff == "", f"mechanism package drifted from 8129ccf: {diff}"
    prereg = json.loads(
        Path("docs/exos_dev1.stage_a_gestational_sensorimotor_model_r1.prereg.lock").read_text()
    )
    assert prereg["provenance"]["mechanism_sha"] == MECHANISM_SHA


def test_runner_retires_ac_hard_gate():
    src = Path("experiments/dev1/search_gestational_sensorimotor_model_r1.py").read_text()
    assert "ac_is_diagnostic_not_hard_gate" in src or "ac_hard_gate_retired" in src
    assert "setup_reference_fail" in src
    assert "gsm_not_causal" in src
    assert "valence_off" in src
    assert "optimization_ceiling_fail" not in src or "diagnostic" in src
    # Must not use AC conjunct as hard ladder return for setup
    assert 'return "optimization_ceiling_fail"' not in src


def test_ladder_order_and_setup_before_cert():
    setup = {"setup_reference_pass": False}
    cert = {"all_certified": True}
    arms = {
        "predictive_gestation": {
            "train_end_in_zone_mean": 1.0,
            "train_distance_reduction_mean": 1.0,
            "fraction_model_actions_mean": 1.0,
            "systematic_misprediction_risk": False,
            "validation_end_in_zone_mean": 1.0,
        },
        "sham_gestation": {"train_end_in_zone_mean": 0.0},
        "predictive_gestation_shuffled_consequences": {"train_end_in_zone_mean": 0.0},
        "learned_model_off_at_action_selection": {"train_end_in_zone_mean": 0.0},
    }
    interventions = {
        "open_loop": {"end_in_zone_rate": 0.0},
        "valence_off": {"end_in_zone_rate": 0.0},
    }
    thresh = {
        "predictive_min_train_end_in_zone": 0.25,
        "predictive_min_distance_reduction": 0.0,
        "active_beats_control_epsilon": 0.05,
        "fresh_world_min_validation_end_in_zone": 0.2,
        "min_fraction_model_actions": 0.5,
    }
    assert (
        apply_decision_ladder(
            setup=setup,
            cert=cert,
            arms=arms,
            interventions=interventions,
            thresh=thresh,
            evidence_complete=True,
        )
        == "setup_reference_fail"
    )
    setup["setup_reference_pass"] = True
    cert["all_certified"] = False
    assert (
        apply_decision_ladder(
            setup=setup,
            cert=cert,
            arms=arms,
            interventions=interventions,
            thresh=thresh,
            evidence_complete=True,
        )
        == "model_certification_fail"
    )


def test_valence_off_intervention_path():
    m = evaluate_gsm_life(
        "predictive_gestation",
        "gsm_r1_eng_life_valence_off",
        n_episodes=1,
        episode_ticks=4,
        gestation_ticks=8,
        valence_off=True,
        embryonic_seed=0,
        body_seed=2,
    )
    assert m.life_record["valence_off"] is True
    assert all(s == "valence_off" for s in m.life_record["action_sources"])


def test_fresh_partitions_disjoint_from_prior_gsm():
    prereg = json.loads(
        Path("docs/exos_dev1.stage_a_gestational_sensorimotor_model_r1.prereg.lock").read_text()
    )
    parts = prereg["seed_partitions"]
    scored = set(parts["discovery_world_seeds"]) | set(parts["validation_world_seeds"]) | set(
        parts["confirmation_seeds"]
    )
    excl = set(parts["excluded_seeds"])
    assert not (scored & excl)
    assert all("_r1_" in s for s in scored)
    assert "exos_dev1_gestational_sensorimotor_model_world_001" in excl
    assert "exos_dev1_gestational_sensorimotor_model_conf_001" in excl
    assert prereg["scored_run_authorized"] is True
    assert prereg["single_scored_attempt"] is True
