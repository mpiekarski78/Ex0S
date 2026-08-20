"""Required Reference Birth R2 selector and teacher-credit unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.reference_birth_r2_life import evaluate_r2_life
from experiments.dev1.reference_birth_r2_outer import (
    LexicographicFitness,
    SurfaceIndividual,
    compare_fitness,
    run_batched_es_lexicographic,
    select_best,
    signed_margin_improvement,
)
from experiments.dev1.reference_birth_r2_preflight import run_teacher_separation_proof
from three_memory.dev1.device import dev1_device
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


def test_signed_margin_formula_rewards_correct_direction():
    # Positive outcome + raised margin → positive g
    assert abs(signed_margin_improvement(0.1, 0.3, +1) - 0.2) < 1e-12
    # Negative outcome + lowered margin → positive g (correct direction)
    assert abs(signed_margin_improvement(0.4, 0.1, -1) - 0.3) < 1e-12
    # Positive outcome + lowered margin → negative g
    assert abs(signed_margin_improvement(0.4, 0.1, +1) - (-0.3)) < 1e-12


def test_equal_accuracy_better_signed_margin_mutant_wins():
    parent = SurfaceIndividual(
        surface={"x": 0.0},
        fitness_key=LexicographicFitness(0.10, 0.00, 0.05, 0.99),
        metrics={"update_norm_mean": 1e-8, "is_parent_slot": True},
    )
    mutant = SurfaceIndividual(
        surface={"x": 1.0},
        fitness_key=LexicographicFitness(0.10, 0.20, 0.05, 0.01),
        metrics={"update_norm_mean": 1e-6, "is_parent_slot": False},
    )
    assert select_best([parent, mutant]) is mutant
    assert compare_fitness(mutant.fitness_key, parent.fitness_key) > 0


def test_larger_update_worse_signed_margin_mutant_loses():
    parent = SurfaceIndividual(
        surface={"x": 0.0},
        fitness_key=LexicographicFitness(0.10, 0.15, 0.05, 0.1),
        metrics={"update_norm_mean": 1e-8},
    )
    mutant = SurfaceIndividual(
        surface={"x": 1.0},
        fitness_key=LexicographicFitness(0.10, 0.01, 0.05, 0.9),
        metrics={"update_norm_mean": 1e-3},  # larger update must not win
    )
    assert select_best([parent, mutant]) is parent


def test_better_accuracy_always_beats_margin_only_improvement():
    low_acc = SurfaceIndividual(
        surface={"x": 0.0},
        fitness_key=LexicographicFitness(0.05, 0.90, 0.90, 0.9),
    )
    high_acc = SurfaceIndividual(
        surface={"x": 1.0},
        fitness_key=LexicographicFitness(0.20, 0.00, 0.00, 0.1),
    )
    assert select_best([low_acc, high_acc]) is high_acc


def test_complete_ties_use_seeded_neutral_tie_break_not_parent_priority():
    # Same primary keys; only tie_break differs. Parent slot must not win permanently.
    parent = SurfaceIndividual(
        surface={"x": 0.0},
        fitness_key=LexicographicFitness(0.1, 0.1, 0.1, 0.1),
        metrics={"is_parent_slot": True},
    )
    mutant = SurfaceIndividual(
        surface={"x": 1.0},
        fitness_key=LexicographicFitness(0.1, 0.1, 0.1, 0.9),
        metrics={"is_parent_slot": False},
    )
    assert select_best([parent, mutant]) is mutant

    # ES: with deterministic evaluate returning identical primaries, selection must
    # sometimes leave parent when tie_break favors a mutant (seeded, not parent-first).
    def evaluate(surface):
        return SurfaceIndividual(
            surface=surface,
            fitness_key=LexicographicFitness(0.5, 0.0, 0.0, 0.0),
            phenotype_hash="same",
        )

    out = run_batched_es_lexicographic(evaluate, generations=6, population_size=4, seed=7)
    left_parent = [h for h in out["history"] if not h["parent_slot_selected"]]
    assert len(left_parent) >= 1
    assert out["tie_break"] == "seeded_neutral"
    assert out["fitness_forbids_raw_update_norm"] is True


def test_selection_fitness_and_behavioral_pass_gates_remain_separate():
    life = evaluate_r2_life(
        "reward_eprop_rate_adaptation",
        "rb_r2_gate_sep",
        "stochastic",
        n_episodes=2,
        life_rng_seed=0,
        device=dev1_device(),
    )
    assert life.life_record["selection_fitness_separate_from_behavioral_gates"] is True
    assert "fitness_key" in life.life_record
    # Behavioral causal predicate is not folded into fitness_key.
    assert life.first_failing_causal_predicate is not None
    assert life.fitness_key.accuracy == life.treatment_accuracy


def test_teacher_credit_off_state_injection_alone_does_not_teach():
    genome = DevGenome.default()
    genome.plasticity_family = "reward_eprop_rate_adaptation"
    org = ModularOrganism.birth(genome, device=torch.device("cpu"), h_disabled=True, consolidation_disabled=True)
    org.set_teacher_credit_enabled(False)
    sensory = torch.zeros(genome.sensory_dim)
    org.observe(OrganismObservation(sensory_vector=sensory, observed_motor_event=3))
    org.observe(OrganismObservation(sensory_vector=sensory, teaching_signal=1.0))
    credit = org.apply_teacher_demonstration_credit()
    assert credit["applied"] is False
    assert credit["reason"] == "teacher_credit_disabled"


def test_self_action_reward_never_updates_teacher_target():
    genome = DevGenome.default()
    genome.plasticity_family = "reward_eprop_rate_adaptation"
    genome.plasticity.learning_rate = 0.05
    genome.plasticity.projection_scale = 10.0
    org = ModularOrganism.birth(genome, device=torch.device("cpu"), h_disabled=True, consolidation_disabled=True)
    sensory = torch.zeros(genome.sensory_dim)
    org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
    action = org.act(policy_mode="hard")
    sampled = action.motor_channel
    teacher_channel = (sampled + 1) % genome.n_motor_channels
    org.observe(OrganismObservation(sensory_vector=sensory, reward=-1.0))
    self_credit = org.apply_outcome_credit()
    assert self_credit["applied"]
    assert self_credit["credit_source"] == "self_action"
    assert self_credit["actor_target"] == sampled
    assert self_credit["actor_target"] != teacher_channel


def test_teacher_outcome_never_rewrites_sampled_action_eligibility():
    genome = DevGenome.default()
    genome.plasticity_family = "reward_eprop_rate_adaptation"
    org = ModularOrganism.birth(genome, device=torch.device("cpu"), h_disabled=True, consolidation_disabled=True)
    sensory = torch.zeros(genome.sensory_dim)
    org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=sensory, reward=1.0))
    org.apply_outcome_credit()
    elig_after_self = org.eligibility.trace.detach().clone()
    org.observe(OrganismObservation(sensory_vector=sensory, observed_motor_event=2))
    org.observe(OrganismObservation(sensory_vector=sensory, teaching_signal=1.0))
    teacher_credit = org.apply_teacher_demonstration_credit()
    assert teacher_credit["applied"]
    assert teacher_credit["credit_source"] == "teacher_demonstration"
    assert teacher_credit["actor_target"] == 2
    assert teacher_credit["self_eligibility_unchanged"] is True
    assert torch.allclose(org.eligibility.trace, elig_after_self)


def test_conflicting_self_and_teacher_events_separately_attributable():
    genome = DevGenome.default()
    genome.plasticity_family = "reward_eprop_rate_adaptation"
    genome.plasticity.learning_rate = 0.05
    genome.plasticity.projection_scale = 10.0
    org = ModularOrganism.birth(genome, device=torch.device("cpu"), h_disabled=True, consolidation_disabled=True)
    sensory = torch.zeros(genome.sensory_dim)
    org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
    action = org.act(policy_mode="stochastic")
    sampled = action.motor_channel
    demo = (sampled + 3) % genome.n_motor_channels
    org.observe(OrganismObservation(sensory_vector=sensory, reward=-1.0))
    self_credit = org.apply_outcome_credit()
    org.observe(OrganismObservation(sensory_vector=sensory, observed_motor_event=demo))
    org.observe(OrganismObservation(sensory_vector=sensory, teaching_signal=1.0))
    teacher_credit = org.apply_teacher_demonstration_credit()
    assert self_credit["credit_source"] == "self_action"
    assert teacher_credit["credit_source"] == "teacher_demonstration"
    assert self_credit["actor_target"] == sampled
    assert teacher_credit["actor_target"] == demo
    assert self_credit["actor_target"] != teacher_credit["actor_target"]


def test_permuted_demonstration_learning_follows_observed_demonstration():
    result = run_teacher_separation_proof("rb_r2_teacher_unit")
    assert result["normal_demo_count"] > 0
    assert result["passed"] is True
    assert result["teacher_credit_off_no_teacher_updates"] is True
