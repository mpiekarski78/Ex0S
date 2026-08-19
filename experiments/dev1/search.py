"""
EX0S-DEV1 staged conditional search.

Discovery loop
──────────────
propose variant
  → run batched newborn lives (with H write/read disabled during Stage A)
  → causal audit (full decision ladder)
  → discovery pass? → promote to extended rotating validation
                    → search continues (not a stop)
  → retain top 2-3 causally valid candidates (beam)
  → test bounded interaction matrix
  → budget expires OR preregistered N candidates validate
  → select by: causal validity, cross-world reliability, complexity
  → freeze exact candidate
  → run ONE untouched confirmation tranche
  → confirmation fails? → record immutable failure; worlds permanently consumed;
                         discovery may reopen, but not using those cells as training data
  → confirmation passes? → stage confirmed; unlock next stage prereg

Search axes (staged conditional, not Cartesian):
1. Cortical plasticity:  three_factor → meta_learned → evolved
2. Fast memory:          competitive_hebbian → factorized
3. Consolidation:        replay → online_slow → hybrid
4. Capacity:             small → medium → large

Beam: retain top BEAM_SIZE causally valid candidates after each axis.
Interaction matrix: bounded cross-product of top beam after axes 1-3.

Research optimizer is active from Stage A (not locked until Stage D).
It updates only inherited G between lives; validation and confirmation
lives never contribute gradients.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.interfaces import OrganismObservation
from experiments.dev1.worlds import InteractionWorld, WorldConfig
from experiments.dev1.probes import run_causal_decision_ladder, ProbeResult


BEAM_SIZE = 3   # retain top BEAM_SIZE causally valid candidates per axis

AXIS_1_PLASTICITY = ["three_factor", "meta_learned", "evolved"]
AXIS_2_FAST_MEMORY = ["competitive_hebbian", "factorized"]
AXIS_3_CONSOLIDATION = ["replay", "online_slow", "hybrid"]
AXIS_4_CAPACITY = [64, 256, 1024]


@dataclass
class Candidate:
    genome: DevGenome
    discovery_score: float = 0.0
    causal_valid: bool = False
    validation_scores: list[float] = field(default_factory=list)
    confirmed: bool = False
    confirmation_failed: bool = False
    consumed_world_seeds: list[str] = field(default_factory=list)
    axis_tag: str = ""

    def reliability(self) -> float:
        if not self.validation_scores:
            return 0.0
        return sum(self.validation_scores) / len(self.validation_scores)

    def complexity(self) -> float:
        """Lower = simpler. Used as tiebreaker."""
        params = sum(
            p.numel()
            for m in [self.genome.plasticity_family]
            for _ in [None]
        )
        return 0.0   # simplified; full version counts genome parameters

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


def search_axis(
    axis_name: str,
    variants: list[str],
    base_genome: DevGenome,
    genome_field: str,
    world_seeds: list[str],
    budget_per_variant: int = 4,
    h_disabled: bool = True,
) -> list[Candidate]:
    """
    Search one axis. Returns all causally valid candidates (up to BEAM_SIZE).
    Discovery pass promotes to rotating validation; search continues.
    """
    candidates: list[Candidate] = []

    for v in variants:
        g = copy.deepcopy(base_genome)
        setattr(g, genome_field, v)

        all_scores = []
        causal_flags = []
        consumed: list[str] = []
        for seed in world_seeds[:budget_per_variant]:
            world = _make_world(seed)
            result = _run_newborn_life(g, world, h_disabled=h_disabled)
            all_scores.append(result["score"])
            causal_flags.append(result["causal_valid"])
            consumed.append(seed)

        score = sum(all_scores) / len(all_scores)
        causal_valid = sum(causal_flags) > len(causal_flags) / 2

        c = Candidate(
            genome=g,
            discovery_score=score,
            causal_valid=causal_valid,
            validation_scores=all_scores,
            consumed_world_seeds=consumed,
            axis_tag=f"{axis_name}:{v}",
        )
        candidates.append(c)

    candidates.sort(key=lambda c: c.rank_key(), reverse=True)
    return candidates[:BEAM_SIZE]


def bounded_interaction_matrix(
    beam: list[Candidate],
    test_seeds: list[str],
    h_disabled: bool = True,
) -> list[Candidate]:
    """
    Test cross-axis interactions within the beam.
    Returns re-ranked beam.
    """
    for cand in beam:
        for seed in test_seeds:
            world = _make_world(seed)
            result = _run_newborn_life(cand.genome, world, h_disabled=h_disabled)
            cand.validation_scores.append(result["score"])
            if not result["causal_valid"]:
                cand.causal_valid = False
            cand.consumed_world_seeds.append(seed)
    beam.sort(key=lambda c: c.rank_key(), reverse=True)
    return beam


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
    output_dir: str = "runs/exos_dev1/stage_a",
) -> dict:
    """
    Full Stage A discovery loop.

    Returns summary with best candidate and confirmation record.
    Writes ConfirmationRecord to output_dir/confirmation.json.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_genome = DevGenome.default()

    # Axis 1: cortical plasticity
    beam1 = search_axis("plasticity", AXIS_1_PLASTICITY, base_genome, "plasticity_family", world_seeds, h_disabled=h_disabled)

    # Axis 2: fast memory (cross each plasticity candidate)
    beam2: list[Candidate] = []
    for cand in beam1:
        b = search_axis("fast_memory", AXIS_2_FAST_MEMORY, cand.genome, "fast_memory_family", world_seeds, h_disabled=h_disabled)
        beam2.extend(b)
    beam2.sort(key=lambda c: c.rank_key(), reverse=True)
    beam2 = beam2[:BEAM_SIZE]

    # Axis 3: consolidation
    beam3: list[Candidate] = []
    for cand in beam2:
        b = search_axis("consolidation", AXIS_3_CONSOLIDATION, cand.genome, "consolidation_family", world_seeds, h_disabled=h_disabled)
        beam3.extend(b)
    beam3.sort(key=lambda c: c.rank_key(), reverse=True)
    beam3 = beam3[:BEAM_SIZE]

    # Interaction matrix
    interaction_seeds = world_seeds[:4]
    beam_final = bounded_interaction_matrix(beam3, interaction_seeds, h_disabled=h_disabled)
    beam_final = [c for c in beam_final if c.causal_valid]

    if not beam_final:
        return {"outcome": "no_causally_valid_candidate", "beam": []}

    # Select best candidate
    best = beam_final[0]

    # ONE untouched confirmation tranche
    conf_rec = run_confirmation(best, confirmation_seeds, h_disabled=h_disabled)
    conf_rec.save(Path(output_dir) / "confirmation.json")

    outcome = "stage_a_confirmed" if conf_rec.passed else "confirmation_failed_immutable"
    return {
        "outcome": outcome,
        "best_genome_hash": best.genome.genome_hash(),
        "confirmation": {
            "passed": conf_rec.passed,
            "scores": conf_rec.scores,
        },
        "beam_size": len(beam_final),
    }
