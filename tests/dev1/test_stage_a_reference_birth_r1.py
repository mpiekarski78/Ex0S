from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.reference_birth_r1_life import evaluate_r1_life, run_matched_interventions
from experiments.dev1.reference_birth_r1_outer import (
    SURFACE_KEYS,
    default_surface,
    extremes_for_sensitivity,
    run_batched_es,
    SurfaceIndividual,
)
from experiments.dev1.reference_birth_r1_preflight import (
    run_search_surface_sensitivity,
    run_teacher_permutation_proof,
)
from three_memory.dev1.device import dev1_device
from three_memory.dev1.genome import DevGenome


def test_r1_inherited_surface_roundtrip():
    g = DevGenome.default()
    surf = g.r1_inherited_surface_dict()
    assert set(surf) == set(SURFACE_KEYS)
    surf["log_actor_learning_rate"] = -2.0
    surf["learning_signal_projection_scale"] = 12.0
    g.apply_r1_inherited_surface(surf)
    assert abs(g.plasticity.learning_rate - __import__("math").exp(-2.0)) < 1e-9
    assert g.plasticity.projection_scale == 12.0


def test_r1_es_records_outer_updates():
    def evaluate(surface):
        return SurfaceIndividual(surface=surface, fitness=sum(surface.values()), phenotype_hash="x")

    out = run_batched_es(evaluate, generations=3, population_size=4, seed=1)
    assert out["outer_updates_executed"] == 3
    assert len(out["history"]) == 3
    assert "population_diversity" in out["history"][0]
    assert "parameter_movement" in out["history"][0]
    assert out["best_surface"]


def test_r1_update_effect_telemetry_present():
    life = evaluate_r1_life(
        "reward_eprop_rate_adaptation",
        "rb_r1_telemetry",
        "stochastic",
        n_episodes=2,
        life_rng_seed=0,
        device=dev1_device(),
    )
    assert life.life_record["n_credit_events"] > 0
    ev = life.credit_events[0]
    for k in [
        "pre_rewarded_action_probability",
        "post_rewarded_action_probability",
        "pre_rewarded_action_margin",
        "post_rewarded_action_margin",
        "margin_change_sign",
        "eligibility_norm",
        "learning_signal_norm",
        "update_norm",
        "td_error",
        "action_entropy",
    ]:
        assert k in ev


def test_r1_matched_interventions_all_run():
    surf = default_surface()
    out = run_matched_interventions(
        "reward_eprop_rate_adaptation",
        "rb_r1_interv",
        surf,
        n_episodes=2,
        life_rng_seed=5,
        device=dev1_device(),
    )
    assert set(out["results"]) == {
        "none",
        "reward_off",
        "eligibility_zero",
        "eligibility_permuted",
        "motor_feedback_permuted",
    }


def test_r1_search_surface_sensitivity_can_leave_1e8():
    high = dict(default_surface())
    high["log_actor_learning_rate"] = -1.5
    high["learning_signal_projection_scale"] = 50.0
    life = evaluate_r1_life(
        "reward_eprop_rate_adaptation",
        "rb_r1_sens_unit",
        "stochastic",
        surface=high,
        n_episodes=2,
        life_rng_seed=9,
        device=dev1_device(),
    )
    assert life.update_norm_mean > 1e-6


def test_r1_teacher_permutation_changes_behavior():
    result = run_teacher_permutation_proof("rb_r1_teacher_unit")
    assert result["normal_demo_count"] > 0
    assert result["passed"] is True


def test_r1_search_surface_sensitivity_section():
    sens = run_search_surface_sensitivity("rb_r1_sens_section")
    assert "max_update_norm" in sens
    assert sens["leaves_1e8_regime"] is True
    assert len(extremes_for_sensitivity()) >= 3
