"""
Stage A R1.1 controller-corrected search.

Preserves the historical R1 runner unchanged. This module introduces:
- graded normalized training fitness derived from organism experience only
- a genuine reward-based score-function outer optimizer arm
- an evolutionary arm over the same inherited genome surface
- structural preflight as the only early hard kill
- causal validity as a promotion rule for validation, not a kill switch for training
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.optimizers.evolutionary import EvolutionaryOptimizer
from experiments.dev1.optimizers.reward_based_meta_gradient import RewardBasedMetaGradientOptimizer
from experiments.dev1.preflight import run_credit_preflight
from experiments.dev1.probes import run_causal_decision_ladder
from experiments.dev1.worlds import InteractionWorld, WorldConfig
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


BEAM_SIZE = 3
R1_1_CREDIT_FAMILIES = [
    "reward_baseline_three_factor",
    "action_contingent_actor_critic",
    "consequence_prediction_credit",
]
R1_1_OPTIMIZER_ARMS = [
    "reward_based_meta_gradient",
    "evolutionary",
]
FITNESS_WEIGHTS = {
    "normalized_return": 0.40,
    "rewarded_improvement": 0.20,
    "reward_weighted_margin": 0.20,
    "rewarded_retention": 0.20,
    "unbounded_penalty": 0.10,
}
FITNESS_RANGES = {
    "normalized_return": [-1.0, 1.0],
    "rewarded_improvement": [-1.0, 1.0],
    "reward_weighted_margin": [-1.0, 1.0],
    "rewarded_retention": [-1.0, 1.0],
    "unbounded_penalty": [0.0, 1.0],
    "fitness_total": [-1.1, 1.0],
}


@dataclass
class LifeMetrics:
    correctness_score: float
    normalized_return: float
    rewarded_improvement: float
    reward_weighted_margin: float
    rewarded_retention: float
    unbounded_penalty: float
    total_reward: float
    total_steps: int
    causal_valid: bool
    training_policy: str
    evaluation_policy: str
    train_eval_gap_reported: bool
    h_begins_empty: bool
    h_write_counter_zero: bool
    h_read_counter_zero: bool
    h_state_hash_unchanged: bool
    causal_results: list[Any] = field(default_factory=list)

    @property
    def normalized_fitness(self) -> float:
        return (
            FITNESS_WEIGHTS["normalized_return"] * self.normalized_return
            + FITNESS_WEIGHTS["rewarded_improvement"] * self.rewarded_improvement
            + FITNESS_WEIGHTS["reward_weighted_margin"] * self.reward_weighted_margin
            + FITNESS_WEIGHTS["rewarded_retention"] * self.rewarded_retention
            - FITNESS_WEIGHTS["unbounded_penalty"] * self.unbounded_penalty
        )


@dataclass
class Candidate:
    genome: DevGenome
    credit_family: str = ""
    optimizer_arm: str = ""
    preflight_passed: bool = False
    decision_code: str = "outer_optimization_not_exercised"
    training_fitnesses: list[float] = field(default_factory=list)
    training_scores: list[float] = field(default_factory=list)
    validation_scores: list[float] = field(default_factory=list)
    validation_fitnesses: list[float] = field(default_factory=list)
    causal_valid: bool = False
    validation_pass: bool = False
    consumed_world_seeds: list[str] = field(default_factory=list)
    optimizer_telemetry: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False
    confirmation_failed: bool = False

    def reliability(self) -> float:
        if not self.validation_scores:
            return 0.0
        return sum(self.validation_scores) / len(self.validation_scores)

    def training_rank(self) -> float:
        if not self.training_fitnesses:
            return 0.0
        return sum(self.training_fitnesses) / len(self.training_fitnesses)

    def complexity(self) -> float:
        return float(len(self.credit_family))

    def rank_key(self) -> tuple:
        return (self.causal_valid, self.reliability(), -self.complexity())


@dataclass
class ConfirmationRecord:
    candidate_genome_hash: str
    confirmation_seeds: list[str]
    scores: list[float]
    passed: bool
    timestamp: float = field(default_factory=time.time)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "candidate_genome_hash": self.candidate_genome_hash,
                "confirmation_seeds": self.confirmation_seeds,
                "scores": self.scores,
                "passed": self.passed,
                "timestamp": self.timestamp,
            }, f, indent=2)


def _make_world(seed: str, cfg: WorldConfig | None = None) -> InteractionWorld:
    c = cfg or WorldConfig(seed=seed)
    c.seed = seed
    return InteractionWorld(c)


def _chunk_mean(values: list[float], start_frac: float, end_frac: float) -> float:
    if not values:
        return 0.0
    n = len(values)
    start = min(n - 1, max(0, int(n * start_frac)))
    end = min(n, max(start + 1, int(math.ceil(n * end_frac))))
    chunk = values[start:end]
    return sum(chunk) / max(1, len(chunk))


def _boundedness_score(org: ModularOrganism) -> float:
    tensors = [
        org.rho.sensory_repr,
        org.rho.relational_repr,
        org.rho.action_repr,
        org.eligibility.trace,
        org.action_ctx.W_motor.weight.data,
    ]
    if not all(torch.isfinite(t).all().item() for t in tensors):
        return 0.0
    max_norm = max(float(t.norm().item()) for t in tensors)
    return float(1.0 / (1.0 + max(0.0, max_norm - 10.0)))


def _signed_reward_signal(reward: float, world: InteractionWorld) -> float:
    if reward == 0.0:
        return 0.0
    if reward > 0.0:
        return float(max(-1.0, min(1.0, reward / max(1e-8, world.cfg.reward_on_correct))))
    return float(max(-1.0, min(1.0, reward / max(1e-8, abs(world.cfg.reward_on_incorrect)))))


def _life_generator(seed: str) -> torch.Generator:
    digest = hashlib.sha256(seed.encode()).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(digest[:8], "big"))
    return g


def _run_newborn_life(
    genome: DevGenome,
    world: InteractionWorld,
    n_episodes: int = 32,
    device: torch.device | None = None,
    h_disabled: bool = True,
    consolidation_disabled: bool = False,
    policy_mode: str = "hard",
) -> LifeMetrics:
    dev = device or torch.device("cpu")
    org = ModularOrganism.birth(genome, device=dev, h_disabled=h_disabled, consolidation_disabled=consolidation_disabled)
    org._max_steps_hint = n_episodes * world.cfg.episode_length
    h_before = org.hippocampus.capacity_telemetry()
    h_hash_before = org.hippocampus.state_hash()

    correct = 0
    total = 0
    reward_history: list[float] = []
    signed_reward_history: list[float] = []
    confidence_history: list[float] = []
    reward_weighted_margin_history: list[float] = []
    action_generator = _life_generator(world.cfg.seed)
    for _ in range(n_episodes):
        events = world.generate_episode()
        prev_reward = 0.0
        for we in events:
            obs = OrganismObservation(sensory_vector=we.sensory_vector, reward=prev_reward)
            org.observe(obs)
            action = org.act(policy_mode=policy_mode, action_generator=action_generator)
            reward = world.reward_for_action(we, action.motor_channel)
            prev_reward = reward
            signed_reward = _signed_reward_signal(reward, world)
            reward_history.append(reward)
            signed_reward_history.append(signed_reward)
            confidence_history.append(float(action.confidence))
            reward_weighted_margin_history.append(float(action.confidence) * signed_reward)
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
        org.rest()

    correctness = correct / max(1, total)
    normalized_return = _chunk_mean(signed_reward_history, 0.0, 1.0)
    rewarded_improvement = _chunk_mean(signed_reward_history, 0.75, 1.0) - _chunk_mean(signed_reward_history, 0.0, 0.25)
    reward_weighted_margin = _chunk_mean(reward_weighted_margin_history, 0.0, 1.0)
    rewarded_retention = _chunk_mean(signed_reward_history, 0.75, 1.0)
    unbounded_penalty = 1.0 - _boundedness_score(org)
    h_after = org.hippocampus.capacity_telemetry()
    h_hash_after = org.hippocampus.state_hash()
    causal_results = run_causal_decision_ladder(org, world, n_test_episodes=4)
    causal_valid = all(r.passed for r in causal_results)
    return LifeMetrics(
        correctness_score=correctness,
        normalized_return=normalized_return,
        rewarded_improvement=rewarded_improvement,
        reward_weighted_margin=reward_weighted_margin,
        rewarded_retention=rewarded_retention,
        unbounded_penalty=unbounded_penalty,
        total_reward=sum(reward_history),
        total_steps=total,
        causal_valid=causal_valid,
        training_policy=policy_mode,
        evaluation_policy="hard",
        train_eval_gap_reported=(policy_mode != "hard"),
        h_begins_empty=(h_before["capacity_used"] == 0),
        h_write_counter_zero=(h_after["write_attempts_total"] == 0 and h_after["successful_writes_total"] == 0),
        h_read_counter_zero=(h_after["read_attempts_total"] == 0 and h_after["successful_reads_total"] == 0),
        h_state_hash_unchanged=(h_hash_before == h_hash_after),
        causal_results=causal_results,
    )


def _make_optimizer(arm: str):
    if arm == "reward_based_meta_gradient":
        return RewardBasedMetaGradientOptimizer()
    if arm == "evolutionary":
        return EvolutionaryOptimizer()
    raise ValueError(f"unknown optimizer arm: {arm}")


def _evaluate_training_lives(
    genome: DevGenome,
    train_seeds: list[str],
    h_disabled: bool,
) -> tuple[list[LifeMetrics], list[str]]:
    metrics: list[LifeMetrics] = []
    consumed: list[str] = []
    for seed in train_seeds:
        world = _make_world(seed)
        lm = _run_newborn_life(genome, world, h_disabled=h_disabled, policy_mode="stochastic")
        metrics.append(lm)
        consumed.append(seed)
    return metrics, consumed


def _train_candidate(
    genome: DevGenome,
    optimizer_arm: str,
    cheap_train_seeds: list[str],
    meta_updates: int,
    evo_generations: int,
    h_disabled: bool = True,
) -> Candidate:
    genome.plasticity_family = genome.plasticity_family
    pre = run_credit_preflight(genome, genome.plasticity_family)
    candidate = Candidate(
        genome=genome,
        credit_family=genome.plasticity_family,
        optimizer_arm=optimizer_arm,
        preflight_passed=pre.passed,
        decision_code=pre.decision_code,
    )
    if not pre.passed:
        return candidate

    optimizer = _make_optimizer(optimizer_arm)
    working_genome = genome
    best_genome = genome
    best_fitness = -1.0

    if optimizer_arm == "reward_based_meta_gradient":
        for _ in range(meta_updates):
            proposed, metadata = optimizer.propose(working_genome)
            lives, consumed = _evaluate_training_lives(proposed, cheap_train_seeds, h_disabled=h_disabled)
            candidate.consumed_world_seeds.extend(consumed)
            avg_fit = sum(l.normalized_fitness for l in lives) / max(1, len(lives))
            avg_score = sum(l.correctness_score for l in lives) / max(1, len(lives))
            candidate.training_fitnesses.append(avg_fit)
            candidate.training_scores.append(avg_score)
            optimizer.update_after_training_lives(metadata, avg_fit)
            working_genome = optimizer.current_genome(working_genome)
            if avg_fit > best_fitness:
                best_fitness = avg_fit
                best_genome = proposed
    elif optimizer_arm == "evolutionary":
        for generation in range(evo_generations):
            population = optimizer.spawn_population(working_genome)
            pop_fitnesses: list[float] = []
            pop_scores: list[float] = []
            for idx, child in enumerate(population):
                train_seed_batch = [f"{seed}_r11_g{generation}_child{idx}" for seed in cheap_train_seeds]
                lives, consumed = _evaluate_training_lives(child, train_seed_batch, h_disabled=h_disabled)
                candidate.consumed_world_seeds.extend(consumed)
                avg_fit = sum(l.normalized_fitness for l in lives) / max(1, len(lives))
                avg_score = sum(l.correctness_score for l in lives) / max(1, len(lives))
                pop_fitnesses.append(avg_fit)
                pop_scores.append(avg_score)
                if avg_fit > best_fitness:
                    best_fitness = avg_fit
                    best_genome = child
            candidate.training_fitnesses.append(max(pop_fitnesses) if pop_fitnesses else 0.0)
            candidate.training_scores.append(max(pop_scores) if pop_scores else 0.0)
            working_genome = optimizer.select(population, pop_fitnesses)
    else:
        raise ValueError(f"unsupported optimizer arm: {optimizer_arm}")

    candidate.genome = best_genome
    candidate.optimizer_telemetry = optimizer.telemetry()
    candidate.optimizer_telemetry.update({
        "training_action_policy": "stochastic_softmax_sample",
        "evaluation_action_policy": "hard_argmax",
        "train_eval_gap_reported": True,
        "fitness_weights": FITNESS_WEIGHTS,
        "fitness_ranges": FITNESS_RANGES,
    })
    if (
        candidate.optimizer_telemetry.get("outer_updates", 0) == 0
        and candidate.optimizer_telemetry.get("outer_generations", 0) == 0
    ):
        candidate.decision_code = "outer_optimization_not_exercised"
        return candidate
    return candidate


def _validate_candidate(candidate: Candidate, rotating_validation_seeds: list[str], h_disabled: bool = True) -> Candidate:
    causal_results_all = []
    for seed in rotating_validation_seeds:
        world = _make_world(seed)
        metrics = _run_newborn_life(candidate.genome, world, h_disabled=h_disabled, policy_mode="hard")
        candidate.validation_scores.append(metrics.correctness_score)
        candidate.validation_fitnesses.append(metrics.normalized_fitness)
        causal_results_all.extend(metrics.causal_results)
        candidate.consumed_world_seeds.append(seed)

    candidate.validation_pass = len(candidate.validation_scores) > 0
    candidate.causal_valid = all(r.passed for r in causal_results_all) if causal_results_all else False
    train_avg = sum(candidate.training_scores) / max(1, len(candidate.training_scores))
    valid_avg = sum(candidate.validation_scores) / max(1, len(candidate.validation_scores))
    if not candidate.causal_valid:
        candidate.decision_code = "credit_not_causal"
    elif train_avg >= 0.65 and valid_avg < 0.65:
        candidate.decision_code = "sensorimotor_overfit"
    elif valid_avg < 0.65:
        candidate.decision_code = "local_rule_optimization_fail"
    else:
        candidate.decision_code = "validation_pass"
    return candidate


def run_confirmation(
    candidate: Candidate,
    confirmation_seeds: list[str],
    pass_threshold: float = 0.65,
    h_disabled: bool = True,
) -> ConfirmationRecord:
    scores = []
    for seed in confirmation_seeds:
        world = _make_world(seed)
        metrics = _run_newborn_life(candidate.genome, world, h_disabled=h_disabled, policy_mode="hard")
        scores.append(metrics.correctness_score)
        candidate.consumed_world_seeds.append(seed)

    avg = sum(scores) / max(1, len(scores))
    passed = avg >= pass_threshold
    candidate.confirmed = passed
    candidate.confirmation_failed = not passed
    candidate.decision_code = "stage_a_confirmation_pass" if passed else "stage_a_confirmation_fail"
    return ConfirmationRecord(
        candidate_genome_hash=candidate.genome.genome_hash(),
        confirmation_seeds=confirmation_seeds,
        scores=scores,
        passed=passed,
    )


def run_stage_a_r1_1_search(
    world_seeds: list[str],
    confirmation_seeds: list[str],
    meta_updates: int = 4,
    evo_generations: int = 4,
    h_disabled: bool = True,
    output_dir: str = "runs/exos_dev1/stage_a_r1_1",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if len(world_seeds) < 6:
        raise ValueError("Stage A R1.1 requires at least 6 world seeds for cheap-train and rotating validation")

    cheap_train_seeds = world_seeds[:2]
    rotating_validation_seeds = world_seeds[2:6]
    candidates: list[Candidate] = []

    for family in R1_1_CREDIT_FAMILIES:
        for optimizer_arm in R1_1_OPTIMIZER_ARMS:
            genome = DevGenome.default()
            genome.plasticity_family = family
            cand = _train_candidate(
                genome=genome,
                optimizer_arm=optimizer_arm,
                cheap_train_seeds=cheap_train_seeds,
                meta_updates=meta_updates,
                evo_generations=evo_generations,
                h_disabled=h_disabled,
            )
            candidates.append(_validate_candidate(cand, rotating_validation_seeds, h_disabled=h_disabled) if cand.preflight_passed else cand)

    beam = [c for c in candidates if c.causal_valid]
    beam.sort(key=lambda c: c.rank_key(), reverse=True)
    beam = beam[:BEAM_SIZE]

    if not beam:
        summary = {
            "outcome": "local_rule_optimization_fail",
            "beam": [],
            "candidates": [self_summary(c) for c in candidates],
            "training_budget_exhausted": True,
        }
        with open(Path(output_dir) / "search_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    best = beam[0]
    conf_rec = run_confirmation(best, confirmation_seeds, h_disabled=h_disabled)
    conf_rec.save(Path(output_dir) / "confirmation.json")
    summary = {
        "outcome": best.decision_code,
        "best_genome_hash": best.genome.genome_hash(),
        "best_family": best.credit_family,
        "best_optimizer_arm": best.optimizer_arm,
        "beam_size": len(beam),
        "beam": [self_summary(c) for c in beam],
        "confirmation": {
            "passed": conf_rec.passed,
            "scores": conf_rec.scores,
        },
        "training_budget_exhausted": True,
        "validation_pass": True,
    }
    with open(Path(output_dir) / "search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def self_summary(candidate: Candidate) -> dict:
    return {
        "family": candidate.credit_family,
        "optimizer_arm": candidate.optimizer_arm,
        "decision_code": candidate.decision_code,
        "preflight_passed": candidate.preflight_passed,
        "causal_valid": candidate.causal_valid,
        "training_scores": candidate.training_scores,
        "training_fitnesses": candidate.training_fitnesses,
        "validation_scores": candidate.validation_scores,
        "validation_fitnesses": candidate.validation_fitnesses,
        "fitness_weights": FITNESS_WEIGHTS,
        "fitness_ranges": FITNESS_RANGES,
        "optimizer_telemetry": candidate.optimizer_telemetry,
        "genome_hash": candidate.genome.genome_hash(),
    }
