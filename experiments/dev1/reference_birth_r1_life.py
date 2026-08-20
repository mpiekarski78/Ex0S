"""
Reference Birth R1 life evaluation with update-effect telemetry and interventions.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from experiments.dev1.conventional_ac_ceiling import evaluate_ceiling_life
from experiments.dev1.probes import probe_permuted_feedback, probe_reward_off, run_causal_decision_ladder
from experiments.dev1.reference_birth_r1_outer import default_surface, genome_from_surface
from experiments.dev1.scaffold_r2 import (
    ContinuousScaffoldPhenotype,
    TopologyScaffoldPhenotype,
    apply_scaffold_to_organism,
    normalize_r2_state,
    scaffold_hash,
)
from experiments.dev1.search_r1_1 import _make_world
from three_memory.dev1.device import dev1_device
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention


@dataclass
class R1LifeMetrics:
    treatment_accuracy: float
    cumulative_reward: float
    learning_fitness: float
    first_failing_causal_predicate: str
    phenotype_hash: str
    genome_hash: str
    scaffold_hash: str
    plasticity_family_name: str
    device: str
    update_norm_mean: float
    margin_increase_fraction: float
    teacher_demo_count: int
    teacher_follow_rate: float
    life_record: dict[str, Any] = field(default_factory=dict)
    credit_events: list[dict[str, Any]] = field(default_factory=list)


def _intervention_from_name(name: str) -> EpropIntervention:
    return {
        "none": EpropIntervention.none(),
        "reward_off": EpropIntervention.with_reward_off(),
        "eligibility_zero": EpropIntervention.with_eligibility_zero(),
        "eligibility_permuted": EpropIntervention.with_eligibility_permuted(),
        "motor_feedback_permuted": EpropIntervention.with_motor_feedback_permuted(),
    }[name]


def evaluate_r1_life(
    arm: str,
    world_seed: str,
    policy_mode: str,
    *,
    surface: dict[str, float] | None = None,
    intervention_name: str = "none",
    device: torch.device | None = None,
    n_episodes: int = 32,
    h_disabled: bool = True,
    life_rng_seed: int = 0,
    permute_teacher_demos: bool = False,
) -> R1LifeMetrics:
    if arm == "conventional_actor_critic_ceiling":
        cm = evaluate_ceiling_life(
            world_seed,
            policy_mode,
            device=device or dev1_device(),
            n_episodes=n_episodes,
            train_with_autograd=(policy_mode == "stochastic"),
        )
        return R1LifeMetrics(
            treatment_accuracy=cm.treatment_accuracy,
            cumulative_reward=cm.cumulative_reward,
            learning_fitness=cm.learning_fitness,
            first_failing_causal_predicate="measurement_only_ceiling",
            phenotype_hash=cm.plasticity_implementation_hash,
            genome_hash="",
            scaffold_hash="",
            plasticity_family_name=cm.plasticity_family_name,
            device=cm.device,
            update_norm_mean=cm.update_norm_mean,
            margin_increase_fraction=0.0,
            teacher_demo_count=0,
            teacher_follow_rate=0.0,
            life_record=cm.life_record,
        )

    family = "reward_eprop_rate_adaptation" if arm in (
        "reward_eprop_rate_adaptation", "teacher_demo_eprop"
    ) else "reward_baseline_three_factor"
    if arm == "teacher_demo_eprop":
        family = "teacher_demo_eprop"

    surf = surface if surface is not None else default_surface()
    bind_family = "reward_eprop_rate_adaptation" if arm in (
        "reward_eprop_rate_adaptation", "teacher_demo_eprop"
    ) else "reward_baseline_three_factor"
    genome = genome_from_surface(bind_family, surf, seed=life_rng_seed)
    if arm == "teacher_demo_eprop":
        genome.plasticity_family = "teacher_demo_eprop"

    dev = device or dev1_device()
    world = _make_world(world_seed)
    org = ModularOrganism.birth(genome, device=dev, h_disabled=h_disabled, consolidation_disabled=True)
    cont = ContinuousScaffoldPhenotype()
    topo = TopologyScaffoldPhenotype()
    apply_scaffold_to_organism(org, cont, topo)
    if hasattr(org.plasticity_rule, "set_intervention"):
        org.plasticity_rule.set_intervention(_intervention_from_name(intervention_name))
    # Sync eligibility decay if surface changed it
    org.eligibility.decay = genome.plasticity.eligibility_decay

    use_teacher = arm == "teacher_demo_eprop"
    pending_teacher: int | None = None
    teacher_demo_count = 0
    teacher_follow_hits = 0
    teacher_follow_trials = 0

    credit_events: list[dict[str, Any]] = []
    action_hist = torch.zeros(genome.n_motor_channels)
    entropies: list[float] = []
    correct = total = 0
    rewards: list[float] = []
    margin_positive = 0
    margin_total = 0

    torch.manual_seed(life_rng_seed + 101)

    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            observed = pending_teacher if use_teacher else None
            if observed is not None:
                teacher_demo_count += 1
            org.observe(OrganismObservation(
                sensory_vector=we.sensory_vector,
                reward=0.0,
                observed_motor_event=observed,
            ))
            pending_teacher = None
            normalize_r2_state(org)
            temp = genome.plasticity.temperature
            # temperature-aware act: temporarily scale logits via competition on scaled logits
            action = org.act(policy_mode=policy_mode)
            if use_teacher and observed is not None:
                teacher_follow_trials += 1
                if action.motor_channel == observed:
                    teacher_follow_hits += 1
            reward = world.reward_for_action(we, action.motor_channel)
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=reward))
            normalize_r2_state(org)
            credit = org.apply_outcome_credit()
            if use_teacher:
                demo = we._correct_channel
                if permute_teacher_demos:
                    demo = (demo + genome.n_motor_channels // 2) % genome.n_motor_channels
                pending_teacher = demo

            if credit.get("applied"):
                credit_events.append({
                    "pre_rewarded_action_probability": credit.get("pre_rewarded_action_probability"),
                    "post_rewarded_action_probability": credit.get("post_rewarded_action_probability"),
                    "pre_rewarded_action_margin": credit.get("pre_rewarded_action_margin"),
                    "post_rewarded_action_margin": credit.get("post_rewarded_action_margin"),
                    "margin_change_sign": credit.get("margin_change_sign"),
                    "eligibility_norm": credit.get("eligibility_norm_before_credit"),
                    "learning_signal_norm": credit.get("learning_signal_norm"),
                    "update_norm": credit.get("rewarded_update_norm"),
                    "critic_value": credit.get("critic_value"),
                    "td_error": credit.get("td_error"),
                    "action_entropy": credit.get("action_entropy"),
                    "rewarded_channel": action.motor_channel,
                    "reward": reward,
                })
                if credit.get("margin_change_sign") is not None and reward > 0:
                    margin_total += 1
                    if credit["margin_change_sign"] > 0:
                        margin_positive += 1

            scores = torch.tensor(action.motor_scores, dtype=torch.float32)
            entropies.append(float(-(scores * torch.log(scores + 1e-12)).sum().item()))
            action_hist[action.motor_channel] += 1
            rewards.append(reward)
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
        org.rest()

    accuracy = correct / max(1, total)
    update_norms = [e["update_norm"] or 0.0 for e in credit_events]
    margin_frac = margin_positive / max(1, margin_total)
    follow_rate = teacher_follow_hits / max(1, teacher_follow_trials)

    causal = run_causal_decision_ladder(org, world, n_test_episodes=4)
    first_fail = causal[0].causal_label if causal else "unknown"
    ph = hashlib.sha256(org.action_ctx.W_motor.weight.data.detach().cpu().numpy().tobytes()).hexdigest()

    life_record = {
        "arm": arm,
        "world_seed": world_seed,
        "policy_mode": policy_mode,
        "accuracy": accuracy,
        "cumulative_reward": sum(rewards),
        "action_entropy_mean": sum(entropies) / max(1, len(entropies)),
        "action_histogram": (action_hist / action_hist.sum().clamp(min=1)).tolist(),
        "update_norm_mean": sum(update_norms) / max(1, len(update_norms)),
        "margin_increase_fraction": margin_frac,
        "teacher_demo_count": teacher_demo_count,
        "teacher_follow_rate": follow_rate,
        "intervention": intervention_name,
        "surface": surf,
        "device": str(dev),
        "genome_hash": genome.genome_hash(),
        "scaffold_hash": scaffold_hash(cont, topo),
        "life_rng_seed": life_rng_seed,
        "permute_teacher_demos": permute_teacher_demos,
        "n_credit_events": len(credit_events),
    }

    return R1LifeMetrics(
        treatment_accuracy=accuracy,
        cumulative_reward=float(sum(rewards)),
        learning_fitness=accuracy - 0.5 * (1.0 - accuracy),
        first_failing_causal_predicate=first_fail,
        phenotype_hash=ph,
        genome_hash=genome.genome_hash(),
        scaffold_hash=life_record["scaffold_hash"],
        plasticity_family_name=org.plasticity_rule.name(),
        device=str(dev),
        update_norm_mean=life_record["update_norm_mean"],
        margin_increase_fraction=margin_frac,
        teacher_demo_count=teacher_demo_count,
        teacher_follow_rate=follow_rate,
        life_record=life_record,
        credit_events=credit_events,
    )


def run_matched_interventions(
    arm: str,
    world_seed: str,
    surface: dict[str, float],
    *,
    device: torch.device | None = None,
    n_episodes: int = 16,
    life_rng_seed: int = 0,
) -> dict[str, Any]:
    """Identical life RNG; interventions from matched newborn setup."""
    names = ["none", "reward_off", "eligibility_zero", "eligibility_permuted", "motor_feedback_permuted"]
    results = {}
    for name in names:
        life = evaluate_r1_life(
            arm,
            world_seed,
            "stochastic",
            surface=surface,
            intervention_name=name,
            device=device,
            n_episodes=n_episodes,
            life_rng_seed=life_rng_seed,
        )
        results[name] = {
            "accuracy": life.treatment_accuracy,
            "update_norm_mean": life.update_norm_mean,
            "margin_increase_fraction": life.margin_increase_fraction,
            "cumulative_reward": life.cumulative_reward,
        }
    treatment = results["none"]
    beats = {
        k: treatment["margin_increase_fraction"] > results[k]["margin_increase_fraction"]
        or treatment["accuracy"] > results[k]["accuracy"]
        for k in names if k != "none"
    }
    return {
        "results": results,
        "treatment_outperforms_interventions": all(beats.values()),
        "beats": beats,
    }
