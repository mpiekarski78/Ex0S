"""
Stage A Reference Birth R1 search runner with batched ES outer optimization.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.reference_birth_r1_life import evaluate_r1_life, run_matched_interventions
from experiments.dev1.reference_birth_r1_outer import SurfaceIndividual, run_batched_es
from experiments.dev1.reference_birth_r1_preflight import run_reference_birth_r1_preflight
from experiments.dev1.search_r2 import _atomic_write_json, _append_jsonl, _rng_state_snapshot
from three_memory.dev1.device import dev1_device


TREATMENT_ARMS = [
    "reward_eprop_rate_adaptation",
    "teacher_demo_eprop",
]


def run_stage_a_reference_birth_r1_search(
    run_id: str,
    world_seeds: list[str],
    confirmation_seeds: list[str],
    output_dir: str = "runs/exos_dev1/stage_a_reference_birth_r1",
    preflight_seed: str = "reference_birth_r1_excluded_preflight_20260820",
    require_cuda: bool = True,
    generations: int = 8,
    population_size: int = 8,
    n_episodes: int = 32,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = dev1_device(require_cuda=require_cuda)

    _atomic_write_json(out / "run_started.json", {
        "run_id": run_id,
        "world_seeds": world_seeds,
        "confirmation_seeds": confirmation_seeds,
        "started_at": time.time(),
        "device": str(dev),
        "rng": _rng_state_snapshot(),
        "outer_optimizer": "batched_evolution_strategies",
        "generations": generations,
        "population_size": population_size,
    })

    preflight = run_reference_birth_r1_preflight(seed=preflight_seed)
    _atomic_write_json(out / "reference_birth_r1_preflight.json", asdict(preflight))
    if not preflight.passed:
        if not preflight.checks.get("search_surface_sensitivity", True):
            outcome = "search_surface_sensitivity_fail"
        else:
            outcome = preflight.decision_code
        summary = {"outcome": outcome, "preflight": asdict(preflight), "candidates": []}
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary

    cheap = world_seeds[:2]
    val = world_seeds[2:6]
    ledger = out / "candidate_life_records.jsonl"

    ceiling = evaluate_r1_life(
        "conventional_actor_critic_ceiling",
        cheap[0],
        "stochastic",
        device=dev,
        n_episodes=n_episodes,
    )
    _append_jsonl(ledger, {"run_id": run_id, **ceiling.life_record})
    if ceiling.treatment_accuracy < 0.1:
        summary = {"outcome": "setup_ceiling_fail", "ceiling_accuracy": ceiling.treatment_accuracy, "candidates": []}
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary

    candidates: list[dict[str, Any]] = []

    for arm in TREATMENT_ARMS:
        def _eval(surface, _arm=arm):
            lives = [
                evaluate_r1_life(
                    _arm,
                    s,
                    "stochastic",
                    surface=surface,
                    device=dev,
                    n_episodes=n_episodes,
                    life_rng_seed=hash(s) % 10_000,
                )
                for s in cheap
            ]
            for life in lives:
                _append_jsonl(ledger, {"run_id": run_id, "phase": "outer_train", **life.life_record})
            fit = sum(l.learning_fitness for l in lives) / len(lives)
            return SurfaceIndividual(
                surface=surface,
                fitness=fit,
                phenotype_hash=lives[-1].phenotype_hash,
                metrics={
                    "train_accuracy_mean": sum(l.treatment_accuracy for l in lives) / len(lives),
                    "update_norm_mean": sum(l.update_norm_mean for l in lives) / len(lives),
                    "margin_increase_fraction": sum(l.margin_increase_fraction for l in lives) / len(lives),
                },
            )

        es = run_batched_es(
            _eval,
            generations=generations,
            population_size=population_size,
            seed=hash(arm) % 1_000_000,
        )
        _atomic_write_json(out / f"outer_es_{arm}.json", es)

        if es["outer_updates_executed"] <= 0:
            summary = {"outcome": "outer_optimization_not_exercised", "arm": arm, "candidates": candidates}
            _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
            _atomic_write_json(out / "search_summary.json", summary)
            return summary

        best_surface = es["best_surface"]
        val_lives = [
            evaluate_r1_life(
                arm,
                s,
                "hard",
                surface=best_surface,
                device=dev,
                n_episodes=n_episodes,
                life_rng_seed=hash(s) % 10_000,
            )
            for s in val
        ]
        for life in val_lives:
            _append_jsonl(ledger, {"run_id": run_id, "phase": "validation", **life.life_record})

        interv = run_matched_interventions(
            arm,
            cheap[0],
            best_surface,
            device=dev,
            n_episodes=min(16, n_episodes),
            life_rng_seed=42,
        )
        _atomic_write_json(out / f"interventions_{arm}.json", interv)

        val_acc = sum(l.treatment_accuracy for l in val_lives) / max(1, len(val_lives))
        first_fail = val_lives[0].first_failing_causal_predicate if val_lives else "unknown"
        if not interv["treatment_outperforms_interventions"]:
            decision = "credit_not_causal"
            first_fail = "credit_not_causal"
        elif first_fail != "integrated_development_pass":
            decision = "reference_birth_r1_validation_fail"
        else:
            decision = "validation_pass"

        candidates.append({
            "arm": arm,
            "validation_accuracy_mean": val_acc,
            "train_accuracy_last": es["history"][-1]["metrics"].get("train_accuracy_mean") if es["history"] else None,
            "outer_updates_executed": es["outer_updates_executed"],
            "best_surface": best_surface,
            "best_fitness": es["best_fitness"],
            "interventions": interv,
            "first_failing_causal_predicate": first_fail,
            "decision_code": decision,
        })

    summary = {
        "outcome": "reference_birth_r1_validation_complete",
        "candidates": candidates,
        "confirmation_seeds_registered": confirmation_seeds,
        "confirmation_consumed": False,
        "outer_optimization_exercised": True,
    }
    _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
    _atomic_write_json(out / "search_summary.json", summary)
    return summary
