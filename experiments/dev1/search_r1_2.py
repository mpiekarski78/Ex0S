"""
Stage A R1.2 numerically robust runner.

This runner preserves the R1.1 learning objective and causal gates, but adds:
- latent z -> phenotype parameter transforms
- fail-closed invalid-candidate rejection
- stable stochastic action sampling
- crash-safe run ledger
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.optimizers.latent_r1_2 import (
    LATENT_PARAM_ORDER,
    WORST_FITNESS,
    EvolutionaryR12,
    RewardBasedMetaGradientR12,
    latent_from_genome,
    validate_phenotype,
)
from experiments.dev1.preflight import run_credit_preflight
from experiments.dev1.search_r1_1 import (
    FITNESS_RANGES,
    FITNESS_WEIGHTS,
    R1_1_CREDIT_FAMILIES,
    _make_world,
    _run_newborn_life,
    run_confirmation,
    self_summary as r11_self_summary,
)
from three_memory.dev1.genome import DevGenome


R1_2_OPTIMIZER_ARMS = ["reward_based_meta_gradient_r1_2", "evolutionary_r1_2"]
INVALID_FRACTION_THRESHOLD = 0.75
INVALID_STREAK_LIMIT = 2
BEAM_SIZE = 3


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
    invalid_candidate_count: int = 0
    consumed_world_seeds: list[str] = field(default_factory=list)
    optimizer_telemetry: dict[str, Any] = field(default_factory=dict)

    def reliability(self) -> float:
        return sum(self.validation_scores) / max(1, len(self.validation_scores))

    def complexity(self) -> float:
        return float(len(self.credit_family))

    def rank_key(self) -> tuple:
        return (self.causal_valid, self.reliability(), -self.complexity())


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


def _last_finite_hash(candidate: Candidate | None) -> dict[str, Any]:
    if candidate is None:
        return {"genome_hash": None, "telemetry_hash": None}
    telem_json = json.dumps(candidate.optimizer_telemetry, sort_keys=True, default=str)
    return {
        "genome_hash": candidate.genome.genome_hash(),
        "telemetry_hash": hashlib.sha256(telem_json.encode()).hexdigest(),
    }


def _stable_training_eval(genome: DevGenome, train_seeds: list[str], h_disabled: bool) -> tuple[list[Any], list[str]]:
    lives = []
    consumed = []
    for seed in train_seeds:
        world = _make_world(seed)
        lm = _run_newborn_life(genome, world, h_disabled=h_disabled, policy_mode="stochastic")
        lives.append(lm)
        consumed.append(seed)
    return lives, consumed


def _validate_candidate(candidate: Candidate, seeds: list[str], h_disabled: bool) -> Candidate:
    causal_results_all = []
    for seed in seeds:
        world = _make_world(seed)
        lm = _run_newborn_life(candidate.genome, world, h_disabled=h_disabled, policy_mode="hard")
        candidate.validation_scores.append(lm.correctness_score)
        candidate.validation_fitnesses.append(lm.normalized_fitness)
        causal_results_all.extend(lm.causal_results)
        candidate.consumed_world_seeds.append(seed)
    candidate.validation_pass = bool(candidate.validation_scores)
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


def _make_meta_optimizer() -> RewardBasedMetaGradientR12:
    return RewardBasedMetaGradientR12()


def _make_evo_optimizer() -> EvolutionaryR12:
    return EvolutionaryR12()


def _train_meta_candidate(candidate: Candidate, train_seeds: list[str], meta_updates: int, h_disabled: bool, ledger: Path) -> Candidate:
    optimizer = _make_meta_optimizer()
    invalid_streak = 0
    working_template = candidate.genome
    best_genome = candidate.genome
    best_fit = WORST_FITNESS
    for update_idx in range(meta_updates):
        proposed, meta = optimizer.propose(working_template)
        if proposed is None:
            candidate.invalid_candidate_count += 1
            candidate.training_fitnesses.append(WORST_FITNESS)
            _append_jsonl(ledger / "candidate_records.jsonl", {
                "phase": "training",
                "optimizer_arm": candidate.optimizer_arm,
                "family": candidate.credit_family,
                "update_idx": update_idx,
                "status": "invalid_candidate_rejected",
                "invalid_reason": meta["invalid_reason"],
                "rng": _rng_state_snapshot(),
                **_last_finite_hash(candidate),
            })
            invalid_streak += 1
            if invalid_streak >= INVALID_STREAK_LIMIT:
                candidate.decision_code = "optimizer_level_failure"
                break
            continue
        invalid_streak = 0
        lives, consumed = _stable_training_eval(proposed, train_seeds, h_disabled)
        candidate.consumed_world_seeds.extend(consumed)
        avg_fit = sum(l.normalized_fitness for l in lives) / max(1, len(lives))
        avg_score = sum(l.correctness_score for l in lives) / max(1, len(lives))
        candidate.training_fitnesses.append(avg_fit)
        candidate.training_scores.append(avg_score)
        ok = optimizer.update_after_training_lives(meta, avg_fit)
        candidate.optimizer_telemetry = optimizer.telemetry()
        if not ok:
            candidate.invalid_candidate_count += 1
            candidate.training_fitnesses[-1] = WORST_FITNESS
            _append_jsonl(ledger / "candidate_records.jsonl", {
                "phase": "training",
                "optimizer_arm": candidate.optimizer_arm,
                "family": candidate.credit_family,
                "update_idx": update_idx,
                "status": "invalid_candidate_rejected",
                "invalid_reason": optimizer.telemetry().get("invalid_reason"),
                "rng": _rng_state_snapshot(),
                **_last_finite_hash(candidate),
            })
            continue
        current, current_meta = optimizer.current_genome(working_template)
        if current is None:
            candidate.invalid_candidate_count += 1
            continue
        working_template = current
        if avg_fit > best_fit:
            best_fit = avg_fit
            best_genome = proposed
        _append_jsonl(ledger / "candidate_records.jsonl", {
            "phase": "training",
            "optimizer_arm": candidate.optimizer_arm,
            "family": candidate.credit_family,
            "update_idx": update_idx,
            "status": "completed",
            "fitness": avg_fit,
            "score": avg_score,
            "latent_hash": meta["latent_hash"],
            "phenotype_hash": meta["phenotype_hash"],
            "rng": _rng_state_snapshot(),
        })
    candidate.genome = best_genome
    candidate.optimizer_telemetry = optimizer.telemetry()
    return candidate


def _train_evo_candidate(candidate: Candidate, train_seeds: list[str], generations: int, h_disabled: bool, ledger: Path) -> Candidate:
    optimizer = _make_evo_optimizer()
    parent_z = latent_from_genome(candidate.genome)
    best_genome = candidate.genome
    best_fit = WORST_FITNESS
    invalid_streak = 0
    for generation in range(generations):
        population = optimizer.spawn_population(candidate.genome, parent_z)
        invalid_count = 0
        fits = []
        for child_idx, (child, meta) in enumerate(population):
            if child is None:
                invalid_count += 1
                fits.append(WORST_FITNESS)
                _append_jsonl(ledger / "candidate_records.jsonl", {
                    "phase": "training",
                    "optimizer_arm": candidate.optimizer_arm,
                    "family": candidate.credit_family,
                    "generation": generation,
                    "candidate_idx": child_idx,
                    "status": "invalid_candidate_rejected",
                    "invalid_reason": meta["invalid_reason"],
                    "rng": _rng_state_snapshot(),
                    **_last_finite_hash(candidate),
                })
                continue
            eval_seeds = [f"{seed}_r12_g{generation}_c{child_idx}" for seed in train_seeds]
            lives, consumed = _stable_training_eval(child, eval_seeds, h_disabled)
            candidate.consumed_world_seeds.extend(consumed)
            avg_fit = sum(l.normalized_fitness for l in lives) / max(1, len(lives))
            avg_score = sum(l.correctness_score for l in lives) / max(1, len(lives))
            fits.append(avg_fit)
            if avg_fit > best_fit:
                best_fit = avg_fit
                best_genome = child
            _append_jsonl(ledger / "candidate_records.jsonl", {
                "phase": "training",
                "optimizer_arm": candidate.optimizer_arm,
                "family": candidate.credit_family,
                "generation": generation,
                "candidate_idx": child_idx,
                "status": "completed",
                "fitness": avg_fit,
                "score": avg_score,
                "latent_hash": meta["latent_hash"],
                "phenotype_hash": meta["phenotype_hash"],
                "rng": _rng_state_snapshot(),
            })
        candidate.training_fitnesses.append(max(fits) if fits else WORST_FITNESS)
        candidate.training_scores.append(candidate.training_fitnesses[-1])
        invalid_fraction = invalid_count / max(1, len(population))
        if invalid_fraction >= INVALID_FRACTION_THRESHOLD:
            invalid_streak += 1
        else:
            invalid_streak = 0
        if invalid_streak >= INVALID_STREAK_LIMIT:
            candidate.decision_code = "optimizer_level_failure"
            break
        parent_z, _ = optimizer.select(population, fits)
        if parent_z is None:
            candidate.decision_code = "optimizer_level_failure"
            break
    candidate.genome = best_genome
    candidate.optimizer_telemetry = optimizer.telemetry()
    return candidate


def self_summary(candidate: Candidate) -> dict[str, Any]:
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
        "invalid_candidate_count": candidate.invalid_candidate_count,
        "fitness_weights": FITNESS_WEIGHTS,
        "fitness_ranges": FITNESS_RANGES,
        "optimizer_telemetry": candidate.optimizer_telemetry,
        "genome_hash": candidate.genome.genome_hash(),
    }


def run_stage_a_r1_2_search(
    run_id: str,
    world_seeds: list[str],
    confirmation_seeds: list[str],
    meta_updates: int = 4,
    evo_generations: int = 4,
    h_disabled: bool = True,
    output_dir: str = "runs/exos_dev1/stage_a_r1_2",
    inject_exception_after_started: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out / "run_started.json", {
        "run_id": run_id,
        "started_at": time.time(),
        "world_seeds": world_seeds,
        "confirmation_seeds": confirmation_seeds,
        "rng": _rng_state_snapshot(),
    })
    if inject_exception_after_started:
        try:
            raise RuntimeError("injected_r1_2_exception")
        except Exception as e:
            _atomic_write_json(out / "run_failed.json", {
                "run_id": run_id,
                "exception": type(e).__name__,
                "message": str(e),
                "traceback_hash": hashlib.sha256(str(e).encode()).hexdigest(),
                "last_finite_boundary": None,
            })
            raise
    cheap_train_seeds = world_seeds[:2]
    rotating_validation_seeds = world_seeds[2:6]
    candidates: list[Candidate] = []
    last_finite_candidate: Candidate | None = None
    try:
        for family in R1_1_CREDIT_FAMILIES:
            for arm in R1_2_OPTIMIZER_ARMS:
                genome = DevGenome.default()
                genome.plasticity_family = family
                pre = run_credit_preflight(genome, family)
                candidate = Candidate(genome=genome, credit_family=family, optimizer_arm=arm, preflight_passed=pre.passed, decision_code=pre.decision_code)
                if not pre.passed:
                    candidates.append(candidate)
                    continue
                if arm == "reward_based_meta_gradient_r1_2":
                    candidate = _train_meta_candidate(candidate, cheap_train_seeds, meta_updates, h_disabled, out)
                else:
                    candidate = _train_evo_candidate(candidate, cheap_train_seeds, evo_generations, h_disabled, out)
                candidate = _validate_candidate(candidate, rotating_validation_seeds, h_disabled)
                candidates.append(candidate)
                last_finite_candidate = candidate

        beam = [c for c in candidates if c.causal_valid]
        beam.sort(key=lambda c: c.rank_key(), reverse=True)
        beam = beam[:BEAM_SIZE]
        if not beam:
            summary = {
                "outcome": "local_rule_optimization_fail",
                "beam": [],
                "candidates": [self_summary(c) for c in candidates],
            }
            _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
            _atomic_write_json(out / "search_summary.json", summary)
            return summary
        best = beam[0]
        conf = run_confirmation(best, confirmation_seeds, h_disabled=h_disabled)
        conf.save(out / "confirmation.json")
        summary = {
            "outcome": best.decision_code,
            "beam": [self_summary(c) for c in beam],
            "confirmation": {"passed": conf.passed, "scores": conf.scores},
        }
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary
    except Exception as e:
        _atomic_write_json(out / "run_failed.json", {
            "run_id": run_id,
            "stage": "A",
            "exception": type(e).__name__,
            "message": str(e),
            "traceback_hash": hashlib.sha256(f"{type(e).__name__}:{e}".encode()).hexdigest(),
            "last_finite_boundary": _last_finite_hash(last_finite_candidate),
            "rng": _rng_state_snapshot(),
        })
        raise
