"""Reference Birth R3 unit tests: LSG, inheritance, controls, teacher separation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.reference_birth_r3_life import (
    evaluate_r3_life,
    inheritance_leakage_check,
)
from experiments.dev1.reference_birth_r3_outer import (
    default_lsg_surface,
    lsg_param_count,
    mutate_lsg_vector,
    run_batched_es_lsg,
)
from experiments.dev1.reference_birth_r2_outer import LexicographicFitness, SurfaceIndividual
from three_memory.dev1.device import dev1_device
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.plasticity.eprop.signal_generator import InheritedLearningSignalGenerator


def test_lsg_param_roundtrip():
    g = DevGenome.default()
    n = lsg_param_count(g.n_motor_channels, g.action_ctx.n_units)
    vec = default_lsg_surface(seed=1)
    assert len(vec) == n
    gen = InheritedLearningSignalGenerator(
        g.relational_ctx.n_units, g.n_motor_channels, g.action_ctx.n_units, param_vector=vec
    )
    assert gen.param_vector() == vec


def test_lsg_off_and_permute_controls():
    g = DevGenome.default()
    vec = default_lsg_surface(seed=2)
    gen = InheritedLearningSignalGenerator(
        g.relational_ctx.n_units, g.n_motor_channels, g.action_ctx.n_units, param_vector=vec
    )
    rel = torch.randn(g.relational_ctx.n_units)
    pol = torch.randn(g.n_motor_channels)
    on = gen.learning_signal(rel, pol, 0.5)
    off = gen.learning_signal(rel, pol, 0.5, generator_off=True)
    perm = gen.learning_signal(rel, pol, 0.5, generator_permuted=True)
    assert float(off.norm()) == 0.0
    assert float(on.norm()) > 0.0
    assert not torch.allclose(on, perm)


def test_within_life_update_is_local_eta_L_e():
    vec = default_lsg_surface(seed=3)
    genome = DevGenome.default()
    genome.plasticity_family = "inherited_learning_signal_generator"
    genome.lsg_param_vector = vec
    genome.plasticity.learning_rate = 0.05
    genome.plasticity.projection_scale = 5.0
    org = ModularOrganism.birth(genome, device=torch.device("cpu"), h_disabled=True, consolidation_disabled=True)
    lsg_before = org.plasticity_rule.signal_generator.param_vector()
    sensory = torch.zeros(genome.sensory_dim)
    org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=sensory, reward=1.0))
    credit = org.apply_outcome_credit()
    assert credit["applied"]
    assert credit["credit_source"] == "self_action"
    assert org.plasticity_rule.signal_generator.param_vector() == lsg_before


def test_separated_teacher_credit_preserved():
    vec = default_lsg_surface(seed=4)
    genome = DevGenome.default()
    genome.plasticity_family = "inherited_learning_signal_generator"
    genome.lsg_param_vector = vec
    genome.plasticity.learning_rate = 0.05
    org = ModularOrganism.birth(genome, device=torch.device("cpu"), h_disabled=True, consolidation_disabled=True)
    sensory = torch.zeros(genome.sensory_dim)
    org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
    action = org.act(policy_mode="stochastic")
    sampled = action.motor_channel
    demo = (sampled + 5) % genome.n_motor_channels
    org.observe(OrganismObservation(sensory_vector=sensory, reward=-1.0))
    self_c = org.apply_outcome_credit()
    org.observe(OrganismObservation(sensory_vector=sensory, observed_motor_event=demo))
    org.observe(OrganismObservation(sensory_vector=sensory, teaching_signal=1.0))
    teach_c = org.apply_teacher_demonstration_credit()
    assert self_c["actor_target"] == sampled
    assert teach_c["actor_target"] == demo
    assert teach_c["self_eligibility_unchanged"] is True


def test_inheritance_leakage_check():
    out = inheritance_leakage_check("unit_leak")
    assert out["passed"] is True


def test_es_lsg_outer_runs():
    def evaluate(vec):
        # Cheap proxy: prefer vectors with larger mean abs — exercises selection only.
        score = sum(abs(v) for v in vec) / len(vec)
        return SurfaceIndividual(
            surface={"lsg_norm": score},
            fitness_key=LexicographicFitness(score, 0.0, 0.0, 0.0),
            phenotype_hash="x",
        )

    out = run_batched_es_lsg(evaluate, generations=2, population_size=4, seed=5)
    assert out["outer_updates_executed"] == 2
    assert len(out["best_lsg_vector"]) == lsg_param_count(
        DevGenome.default().n_motor_channels, DevGenome.default().action_ctx.n_units
    )


def test_r3_candidate_life_short():
    life = evaluate_r3_life(
        "inherited_learning_signal_generator",
        "rb_r3_short",
        "stochastic",
        lsg_vector=default_lsg_surface(seed=6),
        n_episodes=2,
        life_rng_seed=0,
        device=dev1_device(),
    )
    assert life.life_record["n_credit_events"] > 0
    assert life.self_credit_event_count > 0
    assert life.teacher_credit_event_count > 0
    assert life.life_record["lsg_params_unchanged_within_life"] is True


def test_r3_baseline_life_short():
    life = evaluate_r3_life(
        "r2_fixed_eprop_baseline",
        "rb_r3_base",
        "stochastic",
        n_episodes=2,
        life_rng_seed=0,
        device=dev1_device(),
    )
    assert life.plasticity_family_name == "reward_eprop_rate_adaptation"
    assert life.life_record["n_credit_events"] > 0


def test_mutate_lsg_changes_vector():
    import random
    parent = default_lsg_surface(seed=8)
    child = mutate_lsg_vector(parent, 0.2, random.Random(9))
    assert child != parent
    assert len(child) == len(parent)
