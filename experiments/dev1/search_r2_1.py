"""
Stage A R2.1 scaffold-search runner.

Execution correction for R2: repaired credit lifecycle and plasticity family
dispatch. Preserves R2 scaffold surfaces unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.dev1.credit_lifecycle_r2_1 import (
    R2_1_CREDIT_FAMILIES,
    bind_genome_for_family,
    plasticity_implementation_hash,
    run_credit_lifecycle_preflight,
)
from experiments.dev1.probes import probe_permuted_feedback, probe_reward_off, run_causal_decision_ladder
from experiments.dev1.scaffold_r2 import (
    ContinuousScaffoldPhenotype,
    TopologyScaffoldPhenotype,
    apply_scaffold_to_organism,
    normalize_r2_state,
    run_scaffold_sensitivity_preflight,
    scaffold_extremes,
    scaffold_hash,
)
from experiments.dev1.search_r1_1 import FITNESS_WEIGHTS, _make_world, _signed_reward_signal
from experiments.dev1.search_r2 import (
    _atomic_write_json,
    _append_jsonl,
    _boundedness,
    _chunk,
    _continuous_step,
    _rng_state_snapshot,
    _topology_mutation,
)
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


R2_1_CONTINUOUS_ARMS = ["meta_gradient_continuous", "evolution_continuous"]
R2_1_TOPOLOGY_ARMS = ["evolution_topology"]
INFEASIBLE_BOUNDEDNESS_P = 0.95


@dataclass
class R2_1LifeMetrics:
    total_fitness: float
    learning_fitness: float
    components: dict[str, float]
    cumulative_reward: float
    treatment_accuracy: float
    reward_off_score: float
    feedback_off_score: float
    permuted_feedback_score: float
    first_failing_causal_predicate: str
    phenotype_hash: str
    scaffold_hash: str
    plasticity_family_name: str
    plasticity_implementation_hash: str
    life_record: dict[str, Any]


@dataclass
class R2_1Candidate:
    credit_family: str
    optimizer_arm: str
    continuous_scaffold: ContinuousScaffoldPhenotype
    topology_scaffold: TopologyScaffoldPhenotype
    training_curves: list[dict[str, Any]] = field(default_factory=list)
    validation_curves: list[dict[str, Any]] = field(default_factory=list)
    life_records: list[dict[str, Any]] = field(default_factory=list)
    causal_valid: bool = False
    decision_code: str = "outer_optimization_not_exercised"


def _learning_fitness(components: dict[str, float]) -> float:
    """Reward-linked fitness only; P is a feasibility constraint, not learning evidence."""
    return (
        FITNESS_WEIGHTS["normalized_return"] * components["R"]
        + FITNESS_WEIGHTS["rewarded_improvement"] * components["A"]
        + FITNESS_WEIGHTS["reward_weighted_margin"] * components["M"]
        + FITNESS_WEIGHTS["rewarded_retention"] * components["T"]
    )


def _total_fitness(components: dict[str, float], org: ModularOrganism) -> float:
    p_penalty = components["P"]
    if p_penalty >= INFEASIBLE_BOUNDEDNESS_P:
        return -1.0
    return _learning_fitness(components) - FITNESS_WEIGHTS["unbounded_penalty"] * p_penalty


def _reward_off_update_norm(org: ModularOrganism, world, event) -> float:
    probe_org = ModularOrganism.birth(org.genome, h_disabled=True, consolidation_disabled=True)
    if hasattr(org, "_r2_scaffold_hash"):
        apply_scaffold_to_organism(
            probe_org,
            ContinuousScaffoldPhenotype(),
            TopologyScaffoldPhenotype(),
        )
    w0 = probe_org.action_ctx.W_motor.weight.data.clone()
    probe_org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    probe_org.act(policy_mode="hard")
    probe_org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    probe_org.apply_outcome_credit()
    return float((probe_org.action_ctx.W_motor.weight.data - w0).norm().item())


def _evaluate_r2_1_life(
    plasticity_family: str,
    continuous: ContinuousScaffoldPhenotype,
    topology: TopologyScaffoldPhenotype,
    world_seed: str,
    policy_mode: str,
    h_disabled: bool = True,
) -> R2_1LifeMetrics:
    genome = bind_genome_for_family(plasticity_family)
    world = _make_world(world_seed)
    org = ModularOrganism.birth(genome, h_disabled=h_disabled, consolidation_disabled=True)
    apply_scaffold_to_organism(org, continuous, topology)

    reward_history: list[float] = []
    confidence_history: list[float] = []
    elig_norms: list[float] = []
    update_norms: list[float] = []
    signed_projs: list[float] = []
    correct = 0
    total = 0

    for _ in range(32):
        events = world.generate_episode()
        for we in events:
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=0.0))
            normalize_r2_state(org)
            action = org.act(policy_mode=policy_mode)
            reward = world.reward_for_action(we, action.motor_channel)
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=reward))
            normalize_r2_state(org)
            credit = org.apply_outcome_credit()
            elig_norms.append(credit["eligibility_norm_before_credit"])
            update_norms.append(credit["rewarded_update_norm"])
            signed_projs.append(credit["signed_reward_projection"])
            reward_history.append(reward)
            confidence_history.append(float(action.confidence))
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
        org.rest()

    signed_rewards = [_signed_reward_signal(r, world) for r in reward_history]
    reward_weighted_margin = [c * r for c, r in zip(confidence_history, signed_rewards)]
    components = {
        "R": float(sum(signed_rewards) / max(1, len(signed_rewards))),
        "A": float(_chunk(signed_rewards, 0.75, 1.0) - _chunk(signed_rewards, 0.0, 0.25)),
        "M": float(sum(reward_weighted_margin) / max(1, len(reward_weighted_margin))),
        "T": float(_chunk(signed_rewards, 0.75, 1.0)),
        "P": float(max(0.0, 1.0 - _boundedness(org))),
    }
    impl_hash = plasticity_implementation_hash(plasticity_family)
    sc_hash = scaffold_hash(continuous, topology)
    life_record = {
        "plasticity_family_name": plasticity_family,
        "plasticity_implementation_hash": impl_hash,
        "scaffold_hash": sc_hash,
        "eligibility_norm_before_credit": float(sum(elig_norms) / max(1, len(elig_norms))),
        "rewarded_update_norm": float(sum(update_norms) / max(1, len(update_norms))),
        "signed_reward_projection": float(sum(signed_projs) / max(1, len(signed_projs))),
        "reward_off_update_norm": _reward_off_update_norm(org, world, world.generate_episode()[0]),
    }

    rr = probe_reward_off(org, world)
    pf = probe_permuted_feedback(org, world)
    causal = run_causal_decision_ladder(org, world, n_test_episodes=4)
    first_fail = causal[0].causal_label if causal else "unknown"

    return R2_1LifeMetrics(
        total_fitness=_total_fitness(components, org),
        learning_fitness=_learning_fitness(components),
        components=components,
        cumulative_reward=float(sum(reward_history)),
        treatment_accuracy=correct / max(1, total),
        reward_off_score=rr.score,
        feedback_off_score=rr.score,
        permuted_feedback_score=pf.score,
        first_failing_causal_predicate=first_fail,
        phenotype_hash=hashlib.sha256(json.dumps(genome.credit_parameter_dict(), sort_keys=True).encode()).hexdigest(),
        scaffold_hash=sc_hash,
        plasticity_family_name=plasticity_family,
        plasticity_implementation_hash=impl_hash,
        life_record=life_record,
    )


def run_stage_a_r2_1_search(
    run_id: str,
    world_seeds: list[str],
    confirmation_seeds: list[str],
    output_dir: str = "runs/exos_dev1/stage_a_r2_1",
    h_disabled: bool = True,
    meta_updates: int = 8,
    evo_generations: int = 8,
    population_size: int = 8,
    preflight_seed: str = "r2_1_credit_lifecycle_preflight",
) -> dict[str, Any]:
    out = Path(output_dir)
    _atomic_write_json(out / "run_started.json", {
        "run_id": run_id,
        "world_seeds": world_seeds,
        "confirmation_seeds": confirmation_seeds,
        "started_at": time.time(),
        "rng": _rng_state_snapshot(),
    })

    lifecycle = run_credit_lifecycle_preflight(seed=preflight_seed)
    _atomic_write_json(out / "credit_lifecycle_preflight.json", {
        "passed": lifecycle.passed,
        "decision_code": lifecycle.decision_code,
        "checks": lifecycle.checks,
        "metrics": lifecycle.metrics,
    })
    if not lifecycle.passed:
        summary = {
            "outcome": "credit_lifecycle_preflight_fail",
            "preflight": asdict(lifecycle),
            "candidates": [],
        }
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary

    candidates: list[R2_1Candidate] = []
    cheap = world_seeds[:2]
    val = world_seeds[2:6]
    ledger_path = out / "candidate_life_records.jsonl"

    try:
        for family in R2_1_CREDIT_FAMILIES:
            sens = run_scaffold_sensitivity_preflight(bind_genome_for_family(family), family)
            if not sens.passed:
                candidates.append(R2_1Candidate(
                    credit_family=family,
                    optimizer_arm="scaffold_sensitivity_preflight",
                    continuous_scaffold=scaffold_extremes()[0],
                    topology_scaffold=TopologyScaffoldPhenotype(),
                    decision_code="scaffold_sensitivity_fail",
                ))
                continue

            base = ContinuousScaffoldPhenotype()
            for arm in R2_1_CONTINUOUS_ARMS:
                cand = R2_1Candidate(
                    credit_family=family,
                    optimizer_arm=arm,
                    continuous_scaffold=base,
                    topology_scaffold=TopologyScaffoldPhenotype(),
                )
                scaffold = base
                if arm == "meta_gradient_continuous":
                    for step in range(meta_updates):
                        lives = [
                            _evaluate_r2_1_life(family, scaffold, cand.topology_scaffold, s, "stochastic", h_disabled)
                            for s in cheap
                        ]
                        for life in lives:
                            cand.life_records.append(life.life_record)
                            _append_jsonl(ledger_path, {"run_id": run_id, **life.life_record})
                        avg_learning = sum(m.learning_fitness for m in lives) / len(lives)
                        avg_total = sum(m.total_fitness for m in lives) / len(lives)
                        last = lives[-1]
                        cand.training_curves.append({
                            "update": step,
                            "fitness": avg_total,
                            "learning_fitness": avg_learning,
                            "components": last.components,
                            "cumulative_reward": last.cumulative_reward,
                            "treatment_accuracy": last.treatment_accuracy,
                            "plasticity_family_name": family,
                            "plasticity_implementation_hash": last.plasticity_implementation_hash,
                            "first_failing_causal_predicate": last.first_failing_causal_predicate,
                        })
                        scaffold = _continuous_step(scaffold, 0.05, step)
                else:
                    for generation in range(evo_generations):
                        pop_rows = []
                        pop_learning = []
                        for child in range(population_size):
                            child_scaffold = _continuous_step(scaffold, 0.2, generation * 100 + child)
                            lives = [
                                _evaluate_r2_1_life(family, child_scaffold, cand.topology_scaffold, s, "stochastic", h_disabled)
                                for s in cheap
                            ]
                            avg_learning = sum(m.learning_fitness for m in lives) / len(lives)
                            pop_learning.append(avg_learning)
                            pop_rows.append((child_scaffold, lives[-1], avg_learning))
                        best_idx = max(range(len(pop_rows)), key=lambda i: pop_rows[i][2])
                        scaffold = pop_rows[best_idx][0]
                        best = pop_rows[best_idx][1]
                        cand.training_curves.append({
                            "generation": generation,
                            "fitness": pop_rows[best_idx][2],
                            "learning_fitness": pop_rows[best_idx][2],
                            "fitness_variance": float(np.var(pop_learning)),
                            "components": best.components,
                            "cumulative_reward": best.cumulative_reward,
                            "treatment_accuracy": best.treatment_accuracy,
                            "plasticity_family_name": family,
                            "plasticity_implementation_hash": best.plasticity_implementation_hash,
                            "first_failing_causal_predicate": best.first_failing_causal_predicate,
                        })
                cand.continuous_scaffold = scaffold
                vm = [_evaluate_r2_1_life(family, cand.continuous_scaffold, cand.topology_scaffold, s, "hard", h_disabled) for s in val]
                cand.validation_curves = [asdict(v) for v in vm]
                cand.decision_code = "credit_not_causal"
                candidates.append(cand)

            cand = R2_1Candidate(
                credit_family=family,
                optimizer_arm="evolution_topology",
                continuous_scaffold=ContinuousScaffoldPhenotype(),
                topology_scaffold=TopologyScaffoldPhenotype(),
            )
            topo = cand.topology_scaffold
            for generation in range(evo_generations):
                pop_rows = []
                pop_learning = []
                for child in range(population_size):
                    child_topo = _topology_mutation(generation * 100 + child)
                    lives = [
                        _evaluate_r2_1_life(family, cand.continuous_scaffold, child_topo, s, "stochastic", h_disabled)
                        for s in cheap
                    ]
                    avg_learning = sum(m.learning_fitness for m in lives) / len(lives)
                    pop_learning.append(avg_learning)
                    pop_rows.append((child_topo, lives[-1], avg_learning))
                best_idx = max(range(len(pop_rows)), key=lambda i: pop_rows[i][2])
                topo = pop_rows[best_idx][0]
                best = pop_rows[best_idx][1]
                cand.training_curves.append({
                    "generation": generation,
                    "fitness": pop_rows[best_idx][2],
                    "learning_fitness": pop_rows[best_idx][2],
                    "fitness_variance": float(np.var(pop_learning)),
                    "components": best.components,
                    "cumulative_reward": best.cumulative_reward,
                    "treatment_accuracy": best.treatment_accuracy,
                    "plasticity_family_name": family,
                    "plasticity_implementation_hash": best.plasticity_implementation_hash,
                    "first_failing_causal_predicate": best.first_failing_causal_predicate,
                })
            cand.topology_scaffold = topo
            vm = [_evaluate_r2_1_life(family, cand.continuous_scaffold, cand.topology_scaffold, s, "hard", h_disabled) for s in val]
            cand.validation_curves = [asdict(v) for v in vm]
            cand.decision_code = "credit_not_causal"
            candidates.append(cand)

        summary = {
            "outcome": "scaffold_optimization_fail",
            "credit_lifecycle_preflight": lifecycle.decision_code,
            "candidates": [
                {
                    "family": c.credit_family,
                    "optimizer_arm": c.optimizer_arm,
                    "decision_code": c.decision_code,
                    "scaffold_hash": scaffold_hash(c.continuous_scaffold, c.topology_scaffold),
                    "training_curves": c.training_curves,
                    "validation_curves": c.validation_curves,
                    "life_records_sample": c.life_records[:2],
                }
                for c in candidates
            ],
        }
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary
    except Exception as e:
        _atomic_write_json(out / "run_failed.json", {
            "run_id": run_id,
            "exception": type(e).__name__,
            "message": str(e),
            "traceback_hash": hashlib.sha256(f"{type(e).__name__}:{e}".encode()).hexdigest(),
            "rng": _rng_state_snapshot(),
        })
        raise
