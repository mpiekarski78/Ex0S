"""
Stage A R2.1 credit lifecycle helpers and excluded-seed preflight.

Repairs the R2 execution defects:
- eligibility cleared before plasticity
- plasticity_family not bound per candidate
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass

from pathlib import Path

import torch

from experiments.dev1.scaffold_r2 import (
    ContinuousScaffoldPhenotype,
    TopologyScaffoldPhenotype,
    apply_scaffold_to_organism,
    run_scaffold_sensitivity_preflight,
    scaffold_extremes,
    scaffold_hash,
)
from experiments.dev1.search_r1_1 import _make_world
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism, _build_cortical_plasticity


R2_1_CREDIT_FAMILIES = [
    "reward_baseline_three_factor",
    "action_contingent_actor_critic",
    "consequence_prediction_credit",
]

PLASTICITY_SOURCE_FILES = {
    "reward_baseline_three_factor": "three_memory/dev1/plasticity/cortical_plasticity/three_factor.py",
    "action_contingent_actor_critic": "three_memory/dev1/plasticity/cortical_plasticity/meta_learned.py",
    "consequence_prediction_credit": "three_memory/dev1/plasticity/cortical_plasticity/evolved.py",
}


@dataclass
class CreditLifecyclePreflightResult:
    passed: bool
    decision_code: str
    checks: dict[str, bool]
    metrics: dict[str, float | dict]
    details: dict


def plasticity_implementation_hash(family: str) -> str:
    genome = DevGenome.default()
    genome.plasticity_family = family
    rule = _build_cortical_plasticity(
        family,
        genome,
        genome.relational_ctx.n_units,
        genome.action_ctx.n_units,
    )
    src = inspect.getsource(type(rule))
    return hashlib.sha256(src.encode()).hexdigest()


def bind_genome_for_family(family: str, seed: int | None = None) -> DevGenome:
    if family not in R2_1_CREDIT_FAMILIES:
        raise ValueError(f"unknown plasticity family: {family}")
    genome = DevGenome.default()
    if seed is not None:
        genome.seed = seed
    genome.plasticity_family = family
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    bound = org.plasticity_rule.name()
    assert bound == family, f"family dispatch failed: {bound} != {family}"
    return genome


def run_r2_1_interaction_step(
    org: ModularOrganism,
    sensory_vector,
    world,
    world_event,
    policy_mode: str = "hard",
    action_generator: torch.Generator | None = None,
) -> tuple[int, float, dict]:
    """observe → act → consequence → apply eligible plasticity."""
    org.observe(OrganismObservation(sensory_vector=sensory_vector, reward=0.0))
    action = org.act(policy_mode=policy_mode, action_generator=action_generator)
    reward = world.reward_for_action(world_event, action.motor_channel)
    org.observe(OrganismObservation(sensory_vector=sensory_vector, reward=reward))
    credit = org.apply_outcome_credit()
    return action.motor_channel, reward, credit


def _birth_with_scaffold(family: str, continuous: ContinuousScaffoldPhenotype, topology: TopologyScaffoldPhenotype) -> ModularOrganism:
    genome = bind_genome_for_family(family)
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    apply_scaffold_to_organism(org, continuous, topology)
    return org


def _reward_off_update_norm(org: ModularOrganism, world, event) -> float:
    org2 = ModularOrganism.birth(org.genome, h_disabled=True, consolidation_disabled=True)
    apply_scaffold_to_organism(org2, ContinuousScaffoldPhenotype(), TopologyScaffoldPhenotype())
    w0 = org2.action_ctx.W_motor.weight.data.clone()
    org2.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org2.act(policy_mode="hard")
    org2.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org2.apply_outcome_credit()
    return float((org2.action_ctx.W_motor.weight.data - w0).norm().item())


def _single_outcome_delta(family: str, reward: float, event) -> torch.Tensor:
    org = _birth_with_scaffold(family, ContinuousScaffoldPhenotype(), TopologyScaffoldPhenotype())
    w0 = org.action_ctx.W_motor.weight.data.clone()
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=reward))
    org.apply_outcome_credit()
    return (org.action_ctx.W_motor.weight.data - w0).flatten()


def _static_no_expected_action_proof() -> bool:
    repo = Path(__file__).resolve().parents[2]
    forbidden = ("correct_channel", "_correct_channel", "expected_action", "oracle")
    for relpath in PLASTICITY_SOURCE_FILES.values():
        text = (repo / relpath).read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            return False
    body = inspect.getsource(ModularOrganism._apply_local_plasticity)
    return not any(token in body for token in forbidden)


def run_credit_lifecycle_preflight(
    seed: str = "r2_1_credit_lifecycle_preflight",
    family: str = "reward_baseline_three_factor",
) -> CreditLifecyclePreflightResult:
    world = _make_world(seed)
    event = world.generate_episode()[0]
    continuous = ContinuousScaffoldPhenotype()
    topology = TopologyScaffoldPhenotype()
    org = _birth_with_scaffold(family, continuous, topology)

    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org.act(policy_mode="hard")
    elig_before = float(org.eligibility.trace.norm().item())
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=1.0))
    credit = org.apply_outcome_credit()
    elig_after_credit = float(org.eligibility.trace.norm().item())
    org.episode_reset()
    elig_after_reset = float(org.eligibility.trace.norm().item())

    reward_off_norm = _reward_off_update_norm(org, world, event)

    pos_vec = _single_outcome_delta(family, reward=1.0, event=event)
    neg_vec = _single_outcome_delta(family, reward=-0.1, event=event)
    pos_neg_dot = float(torch.dot(pos_vec, neg_vec).item())
    pos_norm = float(pos_vec.norm().item())
    neg_norm = float(neg_vec.norm().item())

    family_vectors = {
        fam: _single_outcome_delta(fam, reward=1.0, event=event)
        for fam in R2_1_CREDIT_FAMILIES
    }
    family_pairwise = {}
    fam_list = R2_1_CREDIT_FAMILIES
    for i, a in enumerate(fam_list):
        for b in fam_list[i + 1:]:
            va = family_vectors[a]
            vb = family_vectors[b]
            cos = float(torch.dot(va, vb).item() / (va.norm() * vb.norm() + 1e-12))
            family_pairwise[f"{a}__{b}"] = cos

    scaffold = run_scaffold_sensitivity_preflight(bind_genome_for_family(family), family)

    checks = {
        "eligibility_before_credit": bool(elig_before > 1e-8),
        "rewarded_update": bool(credit["rewarded_update_norm"] > 1e-8),
        "reward_off_zero": bool(reward_off_norm < 1e-8),
        "three_families_distinct": bool(
            all(float(family_vectors[f].norm().item()) > 1e-10 for f in R2_1_CREDIT_FAMILIES[:2])
            and float(family_vectors["consequence_prediction_credit"].norm().item()) < float(family_vectors["reward_baseline_three_factor"].norm().item()) / 10.0
            and all(abs(cos) < 0.99 for cos in family_pairwise.values())
        ),
        "positive_vs_negative_outcome": bool(
            pos_norm > 1e-10 and neg_norm > 1e-10 and pos_neg_dot <= 0.0
        ),
        "reset_after_credit": bool(elig_after_credit > 1e-8 and elig_after_reset < 1e-8),
        "genome_scaffold_motion": bool(scaffold.passed),
        "no_expected_action_in_update": bool(_static_no_expected_action_proof()),
    }

    passed = all(checks.values())
    return CreditLifecyclePreflightResult(
        passed=passed,
        decision_code="credit_lifecycle_preflight_pass" if passed else "credit_lifecycle_preflight_fail",
        checks=checks,
        metrics={
            "eligibility_norm_before_credit": elig_before,
            "rewarded_update_norm": credit["rewarded_update_norm"],
            "reward_off_update_norm": reward_off_norm,
            "positive_update_norm": pos_norm,
            "negative_update_norm": neg_norm,
            "positive_negative_dot": pos_neg_dot,
            "eligibility_after_credit": elig_after_credit,
            "eligibility_after_reset": elig_after_reset,
            "family_pairwise_cosine": family_pairwise,
        },
        details={
            "plasticity_implementation_hash": plasticity_implementation_hash(family),
            "scaffold_sensitivity": scaffold.metrics,
        },
    )
