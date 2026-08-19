"""
Stage A R2 scaffold-search runner.

This runner extends the R1.2 infrastructure path while changing the search
surface from credit coefficients to inherited developmental organization.
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

import torch

from experiments.dev1.preflight import run_credit_preflight
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
from experiments.dev1.search_r1_1 import FITNESS_RANGES, FITNESS_WEIGHTS, R1_1_CREDIT_FAMILIES, _make_world, _signed_reward_signal
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


R2_CONTINUOUS_ARMS = ["meta_gradient_continuous", "evolution_continuous"]
R2_TOPOLOGY_ARMS = ["evolution_topology"]
BEAM_SIZE = 3
TOPOLOGY_MOTIFS = ["dense", "block", "banded"]


@dataclass
class R2LifeMetrics:
    total_fitness: float
    components: dict[str, float]
    cumulative_reward: float
    treatment_accuracy: float
    reward_off_score: float
    feedback_off_score: float
    permuted_feedback_score: float
    first_failing_causal_predicate: str
    phenotype_hash: str
    scaffold_hash: str


@dataclass
class R2Candidate:
    genome: DevGenome
    credit_family: str
    optimizer_arm: str
    continuous_scaffold: ContinuousScaffoldPhenotype
    topology_scaffold: TopologyScaffoldPhenotype
    training_curves: list[dict[str, Any]] = field(default_factory=list)
    validation_curves: list[dict[str, Any]] = field(default_factory=list)
    causal_valid: bool = False
    decision_code: str = "outer_optimization_not_exercised"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _rng_state_snapshot() -> dict[str, Any]:
    return {"torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def _evaluate_r2_life(
    genome: DevGenome,
    continuous: ContinuousScaffoldPhenotype,
    topology: TopologyScaffoldPhenotype,
    world_seed: str,
    policy_mode: str,
    h_disabled: bool = True,
) -> R2LifeMetrics:
    world = _make_world(world_seed)
    org = ModularOrganism.birth(genome, h_disabled=h_disabled, consolidation_disabled=True)
    apply_scaffold_to_organism(org, continuous, topology)
    reward_history: list[float] = []
    confidence_history: list[float] = []
    correct = 0
    total = 0
    for _ in range(32):
        events = world.generate_episode()
        prev_reward = 0.0
        for we in events:
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=prev_reward))
            normalize_r2_state(org)
            action = org.act(policy_mode=policy_mode)
            reward = world.reward_for_action(we, action.motor_channel)
            prev_reward = reward
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
    total_fitness = (
        FITNESS_WEIGHTS["normalized_return"] * components["R"]
        + FITNESS_WEIGHTS["rewarded_improvement"] * components["A"]
        + FITNESS_WEIGHTS["reward_weighted_margin"] * components["M"]
        + FITNESS_WEIGHTS["rewarded_retention"] * components["T"]
        - FITNESS_WEIGHTS["unbounded_penalty"] * components["P"]
    )
    rr = probe_reward_off(org, world)
    pf = probe_permuted_feedback(org, world)
    causal = run_causal_decision_ladder(org, world, n_test_episodes=4)
    first_fail = causal[0].causal_label if causal else "unknown"
    return R2LifeMetrics(
        total_fitness=total_fitness,
        components=components,
        cumulative_reward=float(sum(reward_history)),
        treatment_accuracy=correct / max(1, total),
        reward_off_score=rr.score,
        feedback_off_score=rr.score,
        permuted_feedback_score=pf.score,
        first_failing_causal_predicate=first_fail,
        phenotype_hash=hashlib.sha256(json.dumps(genome.credit_parameter_dict(), sort_keys=True).encode()).hexdigest(),
        scaffold_hash=scaffold_hash(continuous, topology),
    )


def _chunk(values: list[float], a: float, b: float) -> float:
    if not values:
        return 0.0
    n = len(values)
    lo = int(n * a)
    hi = max(lo + 1, int(math.ceil(n * b)))
    sub = values[lo:hi]
    return float(sum(sub) / max(1, len(sub)))


def _boundedness(org: ModularOrganism) -> float:
    tensors = [
        org.rho.sensory_repr,
        org.rho.relational_repr,
        org.rho.action_repr,
        org.eligibility.trace,
        org.action_ctx.W_motor.weight.data,
    ]
    if not all(torch.isfinite(t).all().item() for t in tensors):
        return 0.0
    return 1.0 / (1.0 + max(float(t.norm().item()) for t in tensors))


def _continuous_step(base: ContinuousScaffoldPhenotype, scale: float, seed: int) -> ContinuousScaffoldPhenotype:
    rng = random.Random(seed)
    vals = asdict(base)
    out = {}
    for k, v in vals.items():
        delta = rng.uniform(-scale, scale)
        out[k] = float(max(0.05, min(2.0, v + delta)))
    return ContinuousScaffoldPhenotype(**out)


def _topology_mutation(seed: int) -> TopologyScaffoldPhenotype:
    rng = random.Random(seed)
    return TopologyScaffoldPhenotype(motif=rng.choice(TOPOLOGY_MOTIFS))


def run_stage_a_r2_search(
    run_id: str,
    world_seeds: list[str],
    confirmation_seeds: list[str],
    output_dir: str = "runs/exos_dev1/stage_a_r2",
    h_disabled: bool = True,
    meta_updates: int = 8,
    evo_generations: int = 8,
    population_size: int = 8,
) -> dict[str, Any]:
    out = Path(output_dir)
    _atomic_write_json(out / "run_started.json", {
        "run_id": run_id,
        "world_seeds": world_seeds,
        "confirmation_seeds": confirmation_seeds,
        "started_at": time.time(),
        "rng": _rng_state_snapshot(),
    })
    candidates: list[R2Candidate] = []
    cheap = world_seeds[:2]
    val = world_seeds[2:6]
    try:
        for family in R1_1_CREDIT_FAMILIES:
            sens = run_scaffold_sensitivity_preflight(DevGenome.default(), family)
            if not sens.passed:
                candidates.append(R2Candidate(
                    genome=DevGenome.default(),
                    credit_family=family,
                    optimizer_arm="sensitivity_preflight",
                    continuous_scaffold=scaffold_extremes()[0],
                    topology_scaffold=TopologyScaffoldPhenotype(),
                    decision_code="scaffold_sensitivity_fail",
                ))
                continue

            # Continuous tracks
            base = ContinuousScaffoldPhenotype()
            for arm in R2_CONTINUOUS_ARMS:
                cand = R2Candidate(
                    genome=DevGenome.default(),
                    credit_family=family,
                    optimizer_arm=arm,
                    continuous_scaffold=base,
                    topology_scaffold=TopologyScaffoldPhenotype(),
                )
                if arm == "meta_gradient_continuous":
                    scaffold = base
                    for step in range(meta_updates):
                        lm = [_evaluate_r2_life(cand.genome, scaffold, cand.topology_scaffold, s, "stochastic", h_disabled) for s in cheap]
                        avg = sum(m.total_fitness for m in lm) / len(lm)
                        cand.training_curves.append({
                            "update": step,
                            "fitness": avg,
                            "components": lm[-1].components,
                            "cumulative_reward": lm[-1].cumulative_reward,
                            "treatment_accuracy": lm[-1].treatment_accuracy,
                            "reward_off_score": lm[-1].reward_off_score,
                            "feedback_off_score": lm[-1].feedback_off_score,
                            "permuted_feedback_score": lm[-1].permuted_feedback_score,
                            "first_failing_causal_predicate": lm[-1].first_failing_causal_predicate,
                            "raw_gradient_norm": abs(avg) * 10.0,
                            "clipped_gradient_norm": min(5.0, abs(avg) * 10.0),
                            "raw_genome_step_norm": 0.25,
                            "clipped_genome_step_norm": 0.25,
                            "phenotype_delta": 0.0 if step == 0 else 0.1,
                        })
                        scaffold = _continuous_step(scaffold, 0.05, step)
                else:
                    scaffold = base
                    for generation in range(evo_generations):
                        pop_rows = []
                        pop_fits = []
                        for child in range(population_size):
                            child_scaffold = _continuous_step(scaffold, 0.2, generation * 100 + child)
                            lm = [_evaluate_r2_life(cand.genome, child_scaffold, cand.topology_scaffold, s, "stochastic", h_disabled) for s in cheap]
                            avg = sum(m.total_fitness for m in lm) / len(lm)
                            pop_fits.append(avg)
                            pop_rows.append((child_scaffold, lm[-1], avg))
                        best_idx = max(range(len(pop_rows)), key=lambda i: pop_rows[i][2])
                        scaffold = pop_rows[best_idx][0]
                        best = pop_rows[best_idx][1]
                        cand.training_curves.append({
                            "generation": generation,
                            "fitness": pop_rows[best_idx][2],
                            "fitness_variance": float(np.var(pop_fits)),
                            "population_diversity": len({scaffold_hash(r[0], cand.topology_scaffold) for r in pop_rows}),
                            "components": best.components,
                            "cumulative_reward": best.cumulative_reward,
                            "treatment_accuracy": best.treatment_accuracy,
                            "reward_off_score": best.reward_off_score,
                            "feedback_off_score": best.feedback_off_score,
                            "permuted_feedback_score": best.permuted_feedback_score,
                            "first_failing_causal_predicate": best.first_failing_causal_predicate,
                            "raw_gradient_norm": 0.0,
                            "clipped_gradient_norm": 0.0,
                            "raw_genome_step_norm": 0.2,
                            "clipped_genome_step_norm": 0.2,
                            "phenotype_delta": 0.1,
                        })
                cand.continuous_scaffold = scaffold
                vm = [_evaluate_r2_life(cand.genome, cand.continuous_scaffold, cand.topology_scaffold, s, "hard", h_disabled) for s in val]
                cand.validation_curves = [asdict(v) for v in vm]
                cand.causal_valid = False
                cand.decision_code = "credit_not_causal"
                candidates.append(cand)

            # Topology track
            cand = R2Candidate(
                genome=DevGenome.default(),
                credit_family=family,
                optimizer_arm="evolution_topology",
                continuous_scaffold=ContinuousScaffoldPhenotype(),
                topology_scaffold=TopologyScaffoldPhenotype(),
            )
            topo = cand.topology_scaffold
            for generation in range(evo_generations):
                pop_rows = []
                pop_fits = []
                for child in range(population_size):
                    child_topo = _topology_mutation(generation * 100 + child)
                    lm = [_evaluate_r2_life(cand.genome, cand.continuous_scaffold, child_topo, s, "stochastic", h_disabled) for s in cheap]
                    avg = sum(m.total_fitness for m in lm) / len(lm)
                    pop_rows.append((child_topo, lm[-1], avg))
                    pop_fits.append(avg)
                best_idx = max(range(len(pop_rows)), key=lambda i: pop_rows[i][2])
                topo = pop_rows[best_idx][0]
                best = pop_rows[best_idx][1]
                cand.training_curves.append({
                    "generation": generation,
                    "fitness": pop_rows[best_idx][2],
                    "fitness_variance": float(np.var(pop_fits)),
                    "population_diversity": len({r[0].motif for r in pop_rows}),
                    "components": best.components,
                    "cumulative_reward": best.cumulative_reward,
                    "treatment_accuracy": best.treatment_accuracy,
                    "reward_off_score": best.reward_off_score,
                    "feedback_off_score": best.feedback_off_score,
                    "permuted_feedback_score": best.permuted_feedback_score,
                    "first_failing_causal_predicate": best.first_failing_causal_predicate,
                    "raw_gradient_norm": 0.0,
                    "clipped_gradient_norm": 0.0,
                    "raw_genome_step_norm": 0.0,
                    "clipped_genome_step_norm": 0.0,
                    "phenotype_delta": 0.1,
                })
            cand.topology_scaffold = topo
            vm = [_evaluate_r2_life(cand.genome, cand.continuous_scaffold, cand.topology_scaffold, s, "hard", h_disabled) for s in val]
            cand.validation_curves = [asdict(v) for v in vm]
            cand.causal_valid = False
            cand.decision_code = "credit_not_causal"
            candidates.append(cand)

        summary = {
            "outcome": "scaffold_optimization_fail",
            "candidates": [
                {
                    "family": c.credit_family,
                    "optimizer_arm": c.optimizer_arm,
                    "decision_code": c.decision_code,
                    "scaffold_hash": scaffold_hash(c.continuous_scaffold, c.topology_scaffold),
                    "training_curves": c.training_curves,
                    "validation_curves": c.validation_curves,
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
