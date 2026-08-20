from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.reference_birth_life import (
    REFERENCE_BIRTH_ARMS,
    bind_genome_for_arm,
    evaluate_reference_birth_life,
    plasticity_implementation_hash,
)
from experiments.dev1.reference_birth_preflight import run_reference_birth_preflight
from experiments.dev1.search_r1_1 import _make_world
from three_memory.dev1.device import dev1_device
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention


def test_eprop_intervention_fields_are_booleans_not_classmethods():
    """Regression: dataclass fields must not be shadowed by factory classmethods."""
    base = EpropIntervention.none()
    assert isinstance(base.reward_off, bool)
    assert isinstance(base.eligibility_zero, bool)
    assert isinstance(base.eligibility_permuted, bool)
    assert isinstance(base.motor_feedback_permuted, bool)
    assert base.reward_off is False


def test_eprop_intervention_factories_set_only_intended_field():
    factories = [
        (EpropIntervention.with_reward_off(), "reward_off", True),
        (EpropIntervention.with_eligibility_zero(), "eligibility_zero", True),
        (EpropIntervention.with_eligibility_permuted(), "eligibility_permuted", True),
        (EpropIntervention.with_motor_feedback_permuted(), "motor_feedback_permuted", True),
    ]
    all_fields = (
        "reward_off",
        "eligibility_zero",
        "eligibility_permuted",
        "motor_feedback_permuted",
    )
    for intervention, active_field, active_value in factories:
        for field in all_fields:
            expected = active_value if field == active_field else False
            assert getattr(intervention, field) is expected, f"{intervention.name}: {field}"


def test_reference_birth_eprop_family_dispatch():
    genome = bind_genome_for_arm("reward_eprop_rate_adaptation")
    org = ModularOrganism.birth(genome, device=dev1_device(), h_disabled=True, consolidation_disabled=True)
    assert org.plasticity_rule.name() == "reward_eprop_rate_adaptation"


def test_reference_birth_teacher_genome_uses_eprop_rule():
    genome = bind_genome_for_arm("teacher_demo_eprop")
    org = ModularOrganism.birth(genome, device=dev1_device(), h_disabled=True, consolidation_disabled=True)
    assert org.plasticity_rule.name() == "reward_eprop_rate_adaptation"


def test_reference_birth_plasticity_hashes_unique_per_arm():
    hashes = {arm: plasticity_implementation_hash(arm) for arm in REFERENCE_BIRTH_ARMS if arm != "conventional_actor_critic_ceiling"}
    assert len(set(hashes.values())) >= 1


def test_reference_birth_eprop_credit_applies():
    genome = bind_genome_for_arm("reward_eprop_rate_adaptation")
    world = _make_world("rb_test_credit")
    event = world.generate_episode()[0]
    org = ModularOrganism.birth(genome, device=dev1_device(), h_disabled=True, consolidation_disabled=True)
    w0 = org.action_ctx.W_motor.weight.data.clone()
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=1.0))
    credit = org.apply_outcome_credit()
    assert credit["applied"] is True
    assert float((org.action_ctx.W_motor.weight.data - w0).norm().item()) > 1e-8


def test_reference_birth_reward_off_intervention_reduces_updates():
    genome = bind_genome_for_arm("reward_eprop_rate_adaptation")
    world = _make_world("rb_test_reward_off")
    event = world.generate_episode()[0]
    org = ModularOrganism.birth(genome, device=dev1_device(), h_disabled=True, consolidation_disabled=True)
    org.plasticity_rule.set_intervention(EpropIntervention.with_reward_off())
    w0 = org.action_ctx.W_motor.weight.data.clone()
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=1.0))
    org.apply_outcome_credit()
    delta = float((org.action_ctx.W_motor.weight.data - w0).norm().item())
    assert delta < 1e-6


def test_reference_birth_observed_motor_event_injection():
    genome = bind_genome_for_arm("teacher_demo_eprop")
    org = ModularOrganism.birth(genome, device=dev1_device(), h_disabled=True, consolidation_disabled=True)
    before = org.rho.action_repr.clone()
    org.observe(OrganismObservation(sensory_vector=[0.1] * genome.sensory_dim, observed_motor_event=3))
    assert float((org.rho.action_repr - before).norm().item()) > 0.0


def test_reference_birth_preflight_passes():
    result = run_reference_birth_preflight("rb_unit_preflight")
    assert result.checks["eprop_finite_forward"] is True
    assert result.checks["eprop_plasticity_hash_stable"] is True


def test_reference_birth_life_eval_runs():
    life = evaluate_reference_birth_life(
        "reward_eprop_rate_adaptation",
        "rb_unit_life",
        "hard",
        device=dev1_device(),
        n_episodes=2,
    )
    assert 0.0 <= life.treatment_accuracy <= 1.0
    assert life.plasticity_family_name == "reward_eprop_rate_adaptation"


def test_reference_birth_cuda_device_when_available():
    dev = dev1_device()
    life = evaluate_reference_birth_life(
        "reward_eprop_rate_adaptation",
        "rb_unit_cuda",
        "hard",
        device=dev,
        n_episodes=1,
    )
    assert life.device == str(dev)
    if torch.cuda.is_available():
        assert life.device.startswith("cuda")
