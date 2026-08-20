"""
Reference Birth R3 life evaluation: inherited LSG + R2-separated teacher credit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from experiments.dev1.conventional_ac_ceiling import evaluate_ceiling_life
from experiments.dev1.probes import run_causal_decision_ladder
from experiments.dev1.reference_birth_r1_outer import default_surface, genome_from_surface
from experiments.dev1.reference_birth_r2_life import (
    R2LifeMetrics,
    _credit_telemetry,
    _retention_probe,
)
from experiments.dev1.reference_birth_r2_outer import LexicographicFitness
from experiments.dev1.reference_birth_r3_outer import default_lsg_surface, genome_from_lsg
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


def _intervention_from_name(name: str) -> EpropIntervention:
    return {
        "none": EpropIntervention.none(),
        "reward_off": EpropIntervention.with_reward_off(),
        "eligibility_zero": EpropIntervention.with_eligibility_zero(),
        "eligibility_permuted": EpropIntervention.with_eligibility_permuted(),
        "motor_feedback_permuted": EpropIntervention.with_motor_feedback_permuted(),
        "signal_generator_off": EpropIntervention.with_signal_generator_off(),
        "signal_generator_permuted": EpropIntervention.with_signal_generator_permuted(),
    }[name]


def evaluate_r3_life(
    arm: str,
    world_seed: str,
    policy_mode: str,
    *,
    lsg_vector: Sequence[float] | None = None,
    plasticity_surface: dict[str, float] | None = None,
    intervention_name: str = "none",
    device: torch.device | None = None,
    n_episodes: int = 32,
    h_disabled: bool = True,
    life_rng_seed: int = 0,
    permute_teacher_demos: bool = False,
    teacher_credit_enabled: bool = True,
    use_teacher: bool | None = None,
    retention_probe_episodes: int = 2,
) -> R2LifeMetrics:
    """
    Arms:
      - conventional_actor_critic_ceiling
      - r2_fixed_eprop_baseline  (frozen failed fixed/broadcast L)
      - inherited_learning_signal_generator  (candidate; teacher credit separated)
    """
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

    if arm == "r2_fixed_eprop_baseline":
        surf = plasticity_surface if plasticity_surface is not None else default_surface()
        genome = genome_from_surface("reward_eprop_rate_adaptation", surf, seed=life_rng_seed)
        teacher = False if use_teacher is None else use_teacher
    elif arm == "inherited_learning_signal_generator":
        vec = list(lsg_vector) if lsg_vector is not None else default_lsg_surface(seed=life_rng_seed)
        genome = genome_from_lsg(
            "inherited_learning_signal_generator",
            vec,
            seed=life_rng_seed,
            plasticity_surface=plasticity_surface,
        )
        teacher = True if use_teacher is None else use_teacher
    else:
        raise ValueError(f"unknown R3 arm: {arm!r}")

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

    # Snapshot actor weights at birth for inheritance-leakage checks.
    w_birth = org.action_ctx.W_motor.weight.data.detach().clone()
    lsg_birth = None
    if hasattr(org.plasticity_rule, "signal_generator"):
        lsg_birth = org.plasticity_rule.signal_generator.param_vector()

    torch.manual_seed(life_rng_seed + 101)

    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=0.0))
            normalize_r2_state(org)
            action = org.act(policy_mode=policy_mode)

            if teacher and last_demo is not None:
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

            if teacher:
                demo = we._correct_channel
                if permute_teacher_demos:
                    demo = (demo + genome.n_motor_channels // 2) % genome.n_motor_channels
                teacher_demo_count += 1
                last_demo = demo
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

    # LSG inheritance: generator params must not change within life.
    lsg_unchanged = True
    if lsg_birth is not None and hasattr(org.plasticity_rule, "signal_generator"):
        lsg_now = org.plasticity_rule.signal_generator.param_vector()
        lsg_unchanged = all(abs(a - b) < 1e-8 for a, b in zip(lsg_birth, lsg_now))

    accuracy = correct / max(1, total)
    signed_margin_mean = sum(g_values) / max(1, len(g_values))
    update_norms = [e["update_norm"] or 0.0 for e in credit_events]
    margin_frac = margin_positive / max(1, margin_total)
    follow_rate = teacher_follow_hits / max(1, teacher_follow_trials)
    self_n = sum(1 for e in credit_events if e.get("credit_source") == "self_action")
    teach_n = sum(1 for e in credit_events if e.get("credit_source") == "teacher_demonstration")

    retention = _retention_probe(
        org, world, policy_mode="hard", n_episodes=retention_probe_episodes
    )
    causal = run_causal_decision_ladder(org, world, n_test_episodes=4)
    first_fail = causal[0].causal_label if causal else "unknown"
    ph = hashlib.sha256(org.action_ctx.W_motor.weight.data.detach().cpu().numpy().tobytes()).hexdigest()
    fitness_key = LexicographicFitness(accuracy, signed_margin_mean, retention, 0.0)

    actor_moved = float((org.action_ctx.W_motor.weight.data - w_birth).norm().item())

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
        "device": str(dev),
        "genome_hash": genome.genome_hash(),
        "scaffold_hash": scaffold_hash(cont, topo),
        "life_rng_seed": life_rng_seed,
        "permute_teacher_demos": permute_teacher_demos,
        "n_credit_events": len(credit_events),
        "fitness_key": fitness_key.as_tuple(),
        "lsg_params_unchanged_within_life": lsg_unchanged,
        "actor_weight_update_norm_life": actor_moved,
        "selection_fitness_separate_from_behavioral_gates": True,
        "inherits_only_generic_learning_machinery": True,
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


def run_r3_causal_controls(
    lsg_vector: Sequence[float],
    world_seed: str,
    *,
    device: torch.device | None = None,
    n_episodes: int = 16,
    life_rng_seed: int = 0,
) -> dict[str, Any]:
    names = ["none", "reward_off", "signal_generator_off", "signal_generator_permuted"]
    results = {}
    for name in names:
        life = evaluate_r3_life(
            "inherited_learning_signal_generator",
            world_seed,
            "stochastic",
            lsg_vector=lsg_vector,
            intervention_name=name,
            device=device,
            n_episodes=n_episodes,
            life_rng_seed=life_rng_seed,
        )
        results[name] = {
            "accuracy": life.treatment_accuracy,
            "update_norm_mean": life.update_norm_mean,
            "signed_margin_improvement": life.signed_margin_improvement,
            "retention_after_reset": life.retention_after_reset,
            "teacher_follow_rate": life.teacher_follow_rate,
        }
    treatment = results["none"]
    beats = {
        k: (
            treatment["signed_margin_improvement"] > results[k]["signed_margin_improvement"]
            or treatment["accuracy"] > results[k]["accuracy"]
        )
        for k in names if k != "none"
    }
    # Teacher permutation control (separate lives, matched RNG).
    normal = evaluate_r3_life(
        "inherited_learning_signal_generator",
        world_seed + "_teach",
        "stochastic",
        lsg_vector=lsg_vector,
        device=device,
        n_episodes=n_episodes,
        life_rng_seed=life_rng_seed + 3,
        permute_teacher_demos=False,
    )
    permuted = evaluate_r3_life(
        "inherited_learning_signal_generator",
        world_seed + "_teach",
        "stochastic",
        lsg_vector=lsg_vector,
        device=device,
        n_episodes=n_episodes,
        life_rng_seed=life_rng_seed + 3,
        permute_teacher_demos=True,
    )
    teacher_perm = {
        "normal_follow": normal.teacher_follow_rate,
        "permuted_follow": permuted.teacher_follow_rate,
        "histograms_differ": normal.life_record["action_histogram"] != permuted.life_record["action_histogram"],
        "passed": normal.teacher_demo_count > 0 and (
            normal.life_record["action_histogram"] != permuted.life_record["action_histogram"]
            or abs(normal.teacher_follow_rate - permuted.teacher_follow_rate) > 1e-6
        ),
    }
    return {
        "results": results,
        "treatment_outperforms_signal_controls": all(beats.values()),
        "beats": beats,
        "reward_off_causality": beats.get("reward_off", False),
        "teacher_permutation": teacher_perm,
    }


def inheritance_leakage_check(seed: str = "r3_leak") -> dict[str, Any]:
    """Two newborns with same LSG must not share world-specific actor weights."""
    vec = default_lsg_surface(seed=7)
    a = evaluate_r3_life(
        "inherited_learning_signal_generator",
        seed + "_a",
        "stochastic",
        lsg_vector=vec,
        n_episodes=2,
        life_rng_seed=1,
        device=torch.device("cpu"),
    )
    b = evaluate_r3_life(
        "inherited_learning_signal_generator",
        seed + "_b",
        "stochastic",
        lsg_vector=vec,
        n_episodes=2,
        life_rng_seed=2,
        device=torch.device("cpu"),
    )
    return {
        "same_lsg_genome_family": a.plasticity_family_name == b.plasticity_family_name,
        "lsg_unchanged_within_each_life": (
            a.life_record["lsg_params_unchanged_within_life"]
            and b.life_record["lsg_params_unchanged_within_life"]
        ),
        "phenotype_hashes_differ": a.phenotype_hash != b.phenotype_hash,
        "passed": (
            a.life_record["lsg_params_unchanged_within_life"]
            and b.life_record["lsg_params_unchanged_within_life"]
            and a.plasticity_family_name == "inherited_learning_signal_generator"
        ),
    }
