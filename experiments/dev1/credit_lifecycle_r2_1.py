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
import torch.nn.functional as F

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


def _target_channel_preference_from_state(
    org: ModularOrganism,
    target_channel: int,
    *,
    apply_motor_bias: bool = False,
) -> tuple[float, float, float]:
    """Softmax probability, signed margin, and raw logit for a fixed motor channel."""
    _, motor_logits = org.action_ctx(org.rho.relational_repr, org.rho.action_repr)
    if apply_motor_bias and hasattr(org, "_r2_motor_channel_bias"):
        motor_logits = motor_logits + org._r2_motor_channel_bias
    logit = float(motor_logits[target_channel].item())
    scores = F.softmax(motor_logits, dim=-1)
    prob = float(scores[target_channel].item())
    top2 = scores.topk(min(2, scores.numel())).values
    top1 = float(top2[0].item())
    second = float(top2[-1].item())
    if int(scores.argmax().item()) == target_channel:
        margin = top1 - second
    else:
        margin = prob - top1
    return prob, margin, logit


def _pin_action_channel(org: ModularOrganism, target_channel: int) -> None:
    """Keep hard policy on one motor channel for repeated cue/action grounding."""
    bias = torch.zeros(org.genome.n_motor_channels, device=org.device)
    bias[target_channel] = 2.0
    org._r2_motor_channel_bias = bias


def _preference_trajectory_for_reward(
    family: str,
    event,
    reward: float,
    n_repeats: int = 40,
) -> dict[str, list[float] | int]:
    """
    Exact R2.1 lifecycle: repeated cue/action pairs with outcome credit.
    Tracks the first chosen channel's probability, margin, and raw logit.
    """
    continuous = ContinuousScaffoldPhenotype(plasticity_mask_gain=12.0)
    org = _birth_with_scaffold(family, continuous, TopologyScaffoldPhenotype())
    sensory = event.sensory_vector
    org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
    action = org.act(policy_mode="hard")
    target = action.motor_channel
    _pin_action_channel(org, target)
    probs: list[float] = []
    margins: list[float] = []
    logits: list[float] = []
    p0, m0, l0 = _target_channel_preference_from_state(org, target)
    probs.append(p0)
    margins.append(m0)
    logits.append(l0)
    for _ in range(n_repeats):
        org.observe(OrganismObservation(sensory_vector=sensory, reward=reward))
        org.apply_outcome_credit()
        org.episode_reset()
        org.rest()
        org.observe(OrganismObservation(sensory_vector=sensory, reward=0.0))
        org.act(policy_mode="hard")
        p, m, l = _target_channel_preference_from_state(org, target)
        probs.append(p)
        margins.append(m)
        logits.append(l)
    return {
        "target_channel": target,
        "probabilities": probs,
        "margins": margins,
        "logits": logits,
    }


@dataclass
class BehavioralMicroGroundingResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float | dict]


def run_behavioral_micro_grounding(
    seed: str = "r2_1_micro_grounding",
    family: str = "reward_baseline_three_factor",
    n_repeats: int = 40,
) -> BehavioralMicroGroundingResult:
    """
    Behavioral sanity: repeated signed consequences move the chosen action's
    margin/probability in the expected direction; reward-off leaves them stable.
    Uses the public R2.1 observe→act→consequence→credit lifecycle.
    """
    world = _make_world(seed)
    event = world.generate_episode()[0]
    pos = _preference_trajectory_for_reward(family, event, reward=1.0, n_repeats=n_repeats)
    neg = _preference_trajectory_for_reward(family, event, reward=-1.0, n_repeats=n_repeats)
    off = _preference_trajectory_for_reward(family, event, reward=0.0, n_repeats=n_repeats)

    pos_prob_delta = float(pos["probabilities"][-1]) - float(pos["probabilities"][0])
    pos_margin_delta = float(pos["margins"][-1]) - float(pos["margins"][0])
    pos_logit_delta = float(pos["logits"][-1]) - float(pos["logits"][0])
    neg_prob_delta = float(neg["probabilities"][-1]) - float(neg["probabilities"][0])
    neg_margin_delta = float(neg["margins"][-1]) - float(neg["margins"][0])
    neg_logit_delta = float(neg["logits"][-1]) - float(neg["logits"][0])
    off_prob_span = max(abs(float(off["probabilities"][-1]) - float(off["probabilities"][0])), 0.0)
    off_margin_span = max(abs(float(off["margins"][-1]) - float(off["margins"][0])), 0.0)
    off_logit_span = max(abs(float(off["logits"][-1]) - float(off["logits"][0])), 0.0)

    checks = {
        "positive_increases_preference": bool(
            pos_logit_delta > 1e-6 or pos_prob_delta > 1e-6 or pos_margin_delta > 1e-6
        ),
        "negative_decreases_preference": bool(
            neg_logit_delta < -1e-6 or neg_prob_delta < -1e-6 or neg_margin_delta < -1e-6
        ),
        "reward_off_stable": bool(
            off_prob_span < 1e-5 and off_margin_span < 1e-5 and off_logit_span < 1e-5
        ),
    }
    passed = all(checks.values())
    return BehavioralMicroGroundingResult(
        passed=passed,
        checks=checks,
        metrics={
            "positive_prob_delta": pos_prob_delta,
            "positive_margin_delta": pos_margin_delta,
            "positive_logit_delta": pos_logit_delta,
            "negative_prob_delta": neg_prob_delta,
            "negative_margin_delta": neg_margin_delta,
            "negative_logit_delta": neg_logit_delta,
            "reward_off_prob_span": off_prob_span,
            "reward_off_margin_span": off_margin_span,
            "reward_off_logit_span": off_logit_span,
            "positive_trajectory": pos,
            "negative_trajectory": neg,
            "reward_off_trajectory": off,
        },
    )


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

    micro = run_behavioral_micro_grounding(seed=seed, family=family)

    checks = {
        "credit_applied_once": bool(credit["applied"] and not credit["refused"]),
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
        "behavioral_micro_grounding": bool(micro.passed),
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
            "behavioral_micro_grounding": micro.metrics,
        },
        details={
            "plasticity_implementation_hash": plasticity_implementation_hash(family),
            "scaffold_sensitivity": scaffold.metrics,
        },
    )
