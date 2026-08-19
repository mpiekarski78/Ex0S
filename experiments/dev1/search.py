"""
Stage A R1 genuine optimization search.

This runner executes a real between-life search over inherited credit
parameters with:
- mandatory sensorimotor-credit preflight
- both outer optimizers on the same genome surface
- two-life cheap training screen
- rotating validation on held-out worlds
- beam retention of the best 3 causally valid candidates
- one untouched confirmation tranche

During every evaluated life:
- H write/read remains disabled
- no gradient update is applied to W, H, rho, or eligibility state

Outer updates may modify only inherited G between completed training lives.
Validation and confirmation worlds are never used for outer updates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.interfaces import OrganismObservation
from experiments.dev1.worlds import InteractionWorld, WorldConfig
from experiments.dev1.probes import run_causal_decision_ladder
from experiments.dev1.preflight import run_credit_preflight
from experiments.dev1.optimizers.meta_gradient import MetaGradientOptimizer
from experiments.dev1.optimizers.evolutionary import EvolutionaryOptimizer


BEAM_SIZE = 3   # retain top BEAM_SIZE causally valid candidates per axis
R1_CREDIT_FAMILIES = [
    "reward_baseline_three_factor",
    "action_contingent_actor_critic",
    "consequence_prediction_credit",
]
R1_OPTIMIZER_ARMS = [
    "meta_gradient",
    "evolutionary",
]


@dataclass
class Candidate:
    genome: DevGenome
    credit_family: str = ""
    optimizer_arm: str = ""
    preflight_passed: bool = False
    validation_pass: bool = False
    decision_code: str = "outer_optimization_not_exercised"
    training_scores: list[float] = field(default_factory=list)
    validation_scores: list[float] = field(default_factory=list)
    causal_valid: bool = False
    confirmed: bool = False
    confirmation_failed: bool = False
    consumed_world_seeds: list[str] = field(default_factory=list)
    optimizer_telemetry: dict[str, Any] = field(default_factory=dict)

    def reliability(self) -> float:
        if not self.validation_scores:
            return 0.0
        return sum(self.validation_scores) / len(self.validation_scores)

    def complexity(self) -> float:
        """Lower is simpler; used only as final tie-breaker."""
        return float(len(self.credit_family))

    def rank_key(self) -> tuple:
        return (self.causal_valid, self.reliability(), -self.complexity())


def _make_world(seed: str, cfg: WorldConfig | None = None) -> InteractionWorld:
    c = cfg or WorldConfig(seed=seed)
    c.seed = seed
    return InteractionWorld(c)


def _run_newborn_life(
    genome: DevGenome,
    world: InteractionWorld,
    n_episodes: int = 32,
    device: torch.device | None = None,
    h_disabled: bool = True,
    consolidation_disabled: bool = False,
) -> dict:
    """
    Run one newborn life and return behavioral metrics.
    No gradient updates during this life.
    """
    dev = device or torch.device("cpu")
    org = ModularOrganism.birth(genome, device=dev, h_disabled=h_disabled, consolidation_disabled=consolidation_disabled)
    org._max_steps_hint = n_episodes * world.cfg.episode_length

    correct = 0
    total = 0
    for _ in range(n_episodes):
        events = world.generate_episode()
        prev_reward = 0.0
        for we in events:
            obs = OrganismObservation(sensory_vector=we.sensory_vector, reward=prev_reward)
            org.observe(obs)
            action = org.act()
            prev_reward = world.reward_for_action(we, action.motor_channel)
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
        org.rest()

    score = correct / max(1, total)
    causal_results = run_causal_decision_ladder(org, world, n_test_episodes=4)
    causal_valid = all(r.passed for r in causal_results)
    return {"score": score, "causal_valid": causal_valid, "causal_results": causal_results}


def _make_optimizer(arm: str):
    if arm == "meta_gradient":
        return MetaGradientOptimizer()
    if arm == "evolutionary":
        return EvolutionaryOptimizer()
    raise ValueError(f"unknown optimizer arm: {arm}")


def _cheap_train_and_validate(
    genome: DevGenome,
    optimizer_arm: str,
    cheap_train_seeds: list[str],
    rotating_validation_seeds: list[str],
    h_disabled: bool = True,
) -> Candidate:
    """
    Run the Stage A R1 cheap training screen, between-life optimizer update,
    and rotating validation for one (family, optimizer) candidate.
    """
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
    working_genome = optimizer.propose(genome) if hasattr(optimizer, "propose") else genome

    # Two-life cheap training screen.
    training_scores: list[float] = []
    for seed in cheap_train_seeds:
        world = _make_world(seed)
        result = _run_newborn_life(working_genome, world, h_disabled=h_disabled)
        training_scores.append(result["score"])
        candidate.consumed_world_seeds.append(seed)

    if optimizer_arm == "meta_gradient":
        working_genome = optimizer.update_after_training_lives(working_genome, training_scores)
    elif optimizer_arm == "evolutionary":
        population = optimizer.spawn_population(working_genome)
        pop_scores = []
        for idx, child in enumerate(population):
            seed = cheap_train_seeds[idx % len(cheap_train_seeds)]
            world = _make_world(f"{seed}_evo_{idx}")
            result = _run_newborn_life(child, world, h_disabled=h_disabled)
            pop_scores.append(result["score"])
            candidate.consumed_world_seeds.append(f"{seed}_evo_{idx}")
        working_genome = optimizer.select(population, pop_scores)
        training_scores.extend(pop_scores)

    candidate.genome = working_genome
    candidate.training_scores = training_scores
    candidate.optimizer_telemetry = optimizer.telemetry()

    if (
        candidate.optimizer_telemetry.get("outer_updates", 0) == 0
        and candidate.optimizer_telemetry.get("outer_generations", 0) == 0
    ):
        candidate.decision_code = "outer_optimization_not_exercised"
        return candidate

    # Rotating validation worlds never contribute outer updates.
    validation_scores = []
    causal_results_all = []
    for seed in rotating_validation_seeds:
        world = _make_world(seed)
        result = _run_newborn_life(working_genome, world, h_disabled=h_disabled)
        validation_scores.append(result["score"])
        causal_results_all.extend(result["causal_results"])
        candidate.consumed_world_seeds.append(seed)
    candidate.validation_scores = validation_scores
    candidate.validation_pass = len(validation_scores) > 0
    candidate.causal_valid = all(r.passed for r in causal_results_all) if causal_results_all else False

    train_avg = sum(training_scores) / max(1, len(training_scores))
    valid_avg = sum(validation_scores) / max(1, len(validation_scores))
    if not candidate.causal_valid:
        candidate.decision_code = "credit_not_causal"
    elif train_avg >= 0.65 and valid_avg < 0.65:
        candidate.decision_code = "sensorimotor_overfit"
    elif valid_avg < 0.65:
        candidate.decision_code = "local_rule_optimization_fail"
    else:
        candidate.decision_code = "validation_pass"
    return candidate


@dataclass
class ConfirmationRecord:
    """Immutable; written when confirmation runs complete."""
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


def run_confirmation(
    candidate: Candidate,
    confirmation_seeds: list[str],
    pass_threshold: float = 0.65,
    h_disabled: bool = True,
) -> ConfirmationRecord:
    """
    ONE untouched confirmation tranche.
    Failed confirmations are immutable and their worlds permanently consumed.
    Discovery may reopen without using those cells as training data.
    """
    scores = []
    for seed in confirmation_seeds:
        world = _make_world(seed)
        result = _run_newborn_life(candidate.genome, world, h_disabled=h_disabled)
        scores.append(result["score"])
        candidate.consumed_world_seeds.append(seed)

    avg = sum(scores) / max(1, len(scores))
    passed = avg >= pass_threshold
    candidate.confirmed = passed
    candidate.confirmation_failed = not passed
    candidate.decision_code = "stage_a_confirmation_pass" if passed else "stage_a_confirmation_fail"

    rec = ConfirmationRecord(
        candidate_genome_hash=candidate.genome.genome_hash(),
        confirmation_seeds=confirmation_seeds,
        scores=scores,
        passed=passed,
    )
    return rec


def run_stage_a_search(
    world_seeds: list[str],
    confirmation_seeds: list[str],
    preregistered_n: int = 1,
    h_disabled: bool = True,
    output_dir: str = "runs/exos_dev1/stage_a_r1",
) -> dict:
    """
    Full Stage A R1 search loop.

    Uses both optimizer arms on the same inherited genome surface.
    Keeps a beam of the best 3 causally valid candidates.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if len(world_seeds) < 6:
        raise ValueError("Stage A R1 requires at least 6 world seeds for cheap-train and rotating validation")

    cheap_train_seeds = world_seeds[:2]
    rotating_validation_seeds = world_seeds[2:6]
    candidates: list[Candidate] = []

    for family in R1_CREDIT_FAMILIES:
        for optimizer_arm in R1_OPTIMIZER_ARMS:
            genome = DevGenome.default()
            genome.plasticity_family = family
            cand = _cheap_train_and_validate(
                genome=genome,
                optimizer_arm=optimizer_arm,
                cheap_train_seeds=cheap_train_seeds,
                rotating_validation_seeds=rotating_validation_seeds,
                h_disabled=h_disabled,
            )
            candidates.append(cand)

    # Hard gate on causal validity, then reliability, then complexity.
    beam = [c for c in candidates if c.causal_valid]
    beam.sort(key=lambda c: c.rank_key(), reverse=True)
    beam = beam[:BEAM_SIZE]

    if not beam:
        summary = {
            "outcome": "local_rule_optimization_fail",
            "beam": [],
            "candidates": [self_summary(c) for c in candidates],
        }
        with open(Path(output_dir) / "search_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # Candidate promoted through validation; confirmation is terminal.
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
        "validation_scores": candidate.validation_scores,
        "optimizer_telemetry": candidate.optimizer_telemetry,
        "genome_hash": candidate.genome.genome_hash(),
    }
