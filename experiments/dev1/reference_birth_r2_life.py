"""
Reference Birth R2 life evaluation: lexicographic learning metrics + separated teacher credit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.dev1.conventional_ac_ceiling import evaluate_ceiling_life
from experiments.dev1.probes import run_causal_decision_ladder
from experiments.dev1.reference_birth_r1_outer import default_surface, genome_from_surface
from experiments.dev1.reference_birth_r2_outer import LexicographicFitness, signed_margin_improvement
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
class R2LifeMetrics:
    treatment_accuracy: float
    cumulative_reward: float
    signed_margin_improvement: float
    retention_after_reset: float
    fitness_key: LexicographicFitness
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
    teacher_credit_event_count: int
    self_credit_event_count: int
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


def _credit_telemetry(credit: dict[str, Any], *, reward: float, channel: int) -> dict[str, Any]:
    pre_m = credit.get("pre_rewarded_action_margin")
    post_m = credit.get("post_rewarded_action_margin")
    outcome_sign = credit.get("outcome_sign")
    if outcome_sign is None:
        outcome_sign = int((reward > 0) - (reward < 0))
    g = 0.0
    if pre_m is not None and post_m is not None:
        g = signed_margin_improvement(pre_m, post_m, outcome_sign)
    return {
        "credit_source": credit.get("credit_source"),
        "actor_target": credit.get("actor_target", channel),
        "pre_rewarded_action_probability": credit.get("pre_rewarded_action_probability"),
        "post_rewarded_action_probability": credit.get("post_rewarded_action_probability"),
        "pre_rewarded_action_margin": pre_m,
        "post_rewarded_action_margin": post_m,
        "margin_change_sign": credit.get("margin_change_sign"),
        "signed_margin_g": g,
        "outcome_sign": outcome_sign,
        "eligibility_norm": credit.get("eligibility_norm_before_credit"),
        "learning_signal_norm": credit.get("learning_signal_norm"),
        "update_norm": credit.get("rewarded_update_norm"),
        "critic_value": credit.get("critic_value"),
        "td_error": credit.get("td_error"),
        "action_entropy": credit.get("action_entropy"),
        "reward": reward,
        "self_eligibility_unchanged": credit.get("self_eligibility_unchanged"),
        "teaching_signal": credit.get("teaching_signal"),
    }


def evaluate_r2_life(
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
    teacher_credit_enabled: bool = True,
    retention_probe_episodes: int = 2,
) -> R2LifeMetrics:
    if arm == "conventional_actor_critic_ceiling":
        cm = evaluate_ceiling_life(
            world_seed,
            policy_mode,
            device=device or dev1_device(),
            n_episodes=n_episodes,
            train_with_autograd=(policy_mode == "stochastic"),
        )
        fk = LexicographicFitness(cm.treatment_accuracy, 0.0, cm.treatment_accuracy, 0.0)
        return R2LifeMetrics(
            treatment_accuracy=cm.treatment_accuracy,
            cumulative_reward=cm.cumulative_reward,
            signed_margin_improvement=0.0,
            retention_after_reset=cm.treatment_accuracy,
            fitness_key=fk,
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
            teacher_credit_event_count=0,
            self_credit_event_count=0,
            life_record=cm.life_record,
        )

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
    org.eligibility.decay = genome.plasticity.eligibility_decay
    org.set_teacher_credit_enabled(teacher_credit_enabled)

    use_teacher = arm == "teacher_demo_eprop"
    last_demo: int | None = None
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
    g_values: list[float] = []

    torch.manual_seed(life_rng_seed + 101)

    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=0.0))
            normalize_r2_state(org)
            action = org.act(policy_mode=policy_mode)

            if use_teacher and last_demo is not None:
                teacher_follow_trials += 1
                if action.motor_channel == last_demo:
                    teacher_follow_hits += 1

            reward = world.reward_for_action(we, action.motor_channel)
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=reward))
            normalize_r2_state(org)
            self_credit = org.apply_outcome_credit()
            if self_credit.get("applied"):
                tel = _credit_telemetry(self_credit, reward=reward, channel=action.motor_channel)
                credit_events.append(tel)
                g_values.append(float(tel["signed_margin_g"]))
                if tel["outcome_sign"] and tel["outcome_sign"] > 0 and tel["margin_change_sign"] is not None:
                    margin_total += 1
                    if tel["margin_change_sign"] > 0:
                        margin_positive += 1

            if use_teacher:
                demo = we._correct_channel
                if permute_teacher_demos:
                    demo = (demo + genome.n_motor_channels // 2) % genome.n_motor_channels
                teacher_demo_count += 1
                last_demo = demo
                # Separate teacher event: observed motor + teaching consequence.
                teaching = float(world.reward_for_action(we, demo))
                org.observe(OrganismObservation(
                    sensory_vector=we.sensory_vector,
                    reward=0.0,
                    observed_motor_event=demo,
                ))
                org.observe(OrganismObservation(
                    sensory_vector=we.sensory_vector,
                    reward=0.0,
                    teaching_signal=teaching,
                ))
                teacher_credit = org.apply_teacher_demonstration_credit()
                if teacher_credit.get("applied"):
                    tel = _credit_telemetry(teacher_credit, reward=teaching, channel=demo)
                    credit_events.append(tel)
                    g_values.append(float(tel["signed_margin_g"]))

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
    signed_margin_mean = sum(g_values) / max(1, len(g_values))
    update_norms = [e["update_norm"] or 0.0 for e in credit_events]
    margin_frac = margin_positive / max(1, margin_total)
    follow_rate = teacher_follow_hits / max(1, teacher_follow_trials)
    self_n = sum(1 for e in credit_events if e.get("credit_source") == "self_action")
    teach_n = sum(1 for e in credit_events if e.get("credit_source") == "teacher_demonstration")

    # Retention after transient reset: frozen probe (no further credit).
    retention = _retention_probe(
        org, world, policy_mode="hard", n_episodes=retention_probe_episodes
    )

    causal = run_causal_decision_ladder(org, world, n_test_episodes=4)
    first_fail = causal[0].causal_label if causal else "unknown"
    ph = hashlib.sha256(org.action_ctx.W_motor.weight.data.detach().cpu().numpy().tobytes()).hexdigest()

    # Selection fitness key (tie_break filled by outer loop). Behavioral gates stay separate.
    fitness_key = LexicographicFitness(
        accuracy=accuracy,
        signed_margin_improvement=signed_margin_mean,
        retention=retention,
        tie_break=0.0,
    )

    life_record = {
        "arm": arm,
        "world_seed": world_seed,
        "policy_mode": policy_mode,
        "accuracy": accuracy,
        "cumulative_reward": sum(rewards),
        "signed_margin_improvement": signed_margin_mean,
        "retention_after_reset": retention,
        "action_entropy_mean": sum(entropies) / max(1, len(entropies)),
        "action_histogram": (action_hist / action_hist.sum().clamp(min=1)).tolist(),
        "update_norm_mean": sum(update_norms) / max(1, len(update_norms)),
        "margin_increase_fraction": margin_frac,
        "teacher_demo_count": teacher_demo_count,
        "teacher_follow_rate": follow_rate,
        "teacher_credit_enabled": teacher_credit_enabled,
        "self_credit_event_count": self_n,
        "teacher_credit_event_count": teach_n,
        "intervention": intervention_name,
        "surface": surf,
        "device": str(dev),
        "genome_hash": genome.genome_hash(),
        "scaffold_hash": scaffold_hash(cont, topo),
        "life_rng_seed": life_rng_seed,
        "permute_teacher_demos": permute_teacher_demos,
        "n_credit_events": len(credit_events),
        "fitness_key": fitness_key.as_tuple(),
        "selection_fitness_separate_from_behavioral_gates": True,
    }

    return R2LifeMetrics(
        treatment_accuracy=accuracy,
        cumulative_reward=float(sum(rewards)),
        signed_margin_improvement=signed_margin_mean,
        retention_after_reset=retention,
        fitness_key=fitness_key,
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
        teacher_credit_event_count=teach_n,
        self_credit_event_count=self_n,
        life_record=life_record,
        credit_events=credit_events,
    )


def _retention_probe(
    org: ModularOrganism,
    world: Any,
    *,
    policy_mode: str,
    n_episodes: int,
) -> float:
    """Accuracy after episode reset with no further credit (weights persist)."""
    org.episode_reset()
    correct = total = 0
    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=0.0))
            normalize_r2_state(org)
            action = org.act(policy_mode=policy_mode)
            # Behavioral probe only — drop consequence without credit update.
            org._awaiting_consequence = False
            org._outcome_credit_pending = False
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
    return correct / max(1, total)


def run_matched_interventions(
    arm: str,
    world_seed: str,
    surface: dict[str, float],
    *,
    device: torch.device | None = None,
    n_episodes: int = 16,
    life_rng_seed: int = 0,
) -> dict[str, Any]:
    names = ["none", "reward_off", "eligibility_zero", "eligibility_permuted", "motor_feedback_permuted"]
    results = {}
    for name in names:
        life = evaluate_r2_life(
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
            "signed_margin_improvement": life.signed_margin_improvement,
            "retention_after_reset": life.retention_after_reset,
            "cumulative_reward": life.cumulative_reward,
        }
    treatment = results["none"]
    beats = {
        k: (
            treatment["signed_margin_improvement"] > results[k]["signed_margin_improvement"]
            or treatment["accuracy"] > results[k]["accuracy"]
        )
        for k in names if k != "none"
    }
    return {
        "results": results,
        "treatment_outperforms_interventions": all(beats.values()),
        "beats": beats,
    }
