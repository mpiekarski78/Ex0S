"""
Stage A Reference Birth R3 search: inherited learning-signal generator.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.dev1.reference_birth_r2_outer import LexicographicFitness, SurfaceIndividual
from experiments.dev1.reference_birth_r3_life import evaluate_r3_life, run_r3_causal_controls
from experiments.dev1.reference_birth_r3_outer import default_lsg_surface, run_batched_es_lsg
from experiments.dev1.reference_birth_r3_preflight import run_reference_birth_r3_preflight
from experiments.dev1.search_r2 import _atomic_write_json, _append_jsonl, _rng_state_snapshot
from three_memory.dev1.device import dev1_device


def run_stage_a_reference_birth_r3_search(
    run_id: str,
    world_seeds: list[str],
    confirmation_seeds: list[str],
    output_dir: str = "runs/exos_dev1/stage_a_reference_birth_r3",
    preflight_seed: str = "reference_birth_r3_excluded_preflight_20260820",
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
        "outer_optimizer": "batched_evolution_strategies_lsg",
        "generations": generations,
        "population_size": population_size,
        "r1_r2_confirmation_sealed": True,
        "comparison_arms": [
            "r2_fixed_eprop_baseline",
            "inherited_learning_signal_generator",
            "conventional_actor_critic_ceiling",
            "signal_generator_off_or_permuted",
        ],
    })

    preflight = run_reference_birth_r3_preflight(seed=preflight_seed)
    _atomic_write_json(out / "reference_birth_r3_preflight.json", asdict(preflight))
    if not preflight.passed:
        if not preflight.checks.get("signal_generator_unit", True):
            outcome = "signal_generator_unit_fail"
        elif not preflight.checks.get("inheritance_leakage", True):
            outcome = "inheritance_leakage_fail"
        elif not preflight.checks.get("search_surface_sensitivity", True):
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

    ceiling = evaluate_r3_life(
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

    # Frozen R2 fixed-e-prop baseline (contrast only; not promoted).
    baseline_lives = [
        evaluate_r3_life(
            "r2_fixed_eprop_baseline",
            s,
            "stochastic",
            device=dev,
            n_episodes=n_episodes,
            life_rng_seed=hash(s) % 10_000,
        )
        for s in cheap
    ]
    for life in baseline_lives:
        _append_jsonl(ledger, {"run_id": run_id, "phase": "baseline_train", **life.life_record})
    baseline_val = [
        evaluate_r3_life(
            "r2_fixed_eprop_baseline",
            s,
            "hard",
            device=dev,
            n_episodes=n_episodes,
            life_rng_seed=hash(s) % 10_000,
        )
        for s in val
    ]
    for life in baseline_val:
        _append_jsonl(ledger, {"run_id": run_id, "phase": "baseline_validation", **life.life_record})
    baseline_record = {
        "arm": "r2_fixed_eprop_baseline",
        "purpose": "frozen_failed_baseline",
        "organism_candidate": False,
        "train_accuracy_mean": sum(l.treatment_accuracy for l in baseline_lives) / len(baseline_lives),
        "validation_accuracy_mean": sum(l.treatment_accuracy for l in baseline_val) / max(1, len(baseline_val)),
    }
    _atomic_write_json(out / "baseline_r2_fixed_eprop.json", baseline_record)

    def _eval(lsg_vec):
        lives = [
            evaluate_r3_life(
                "inherited_learning_signal_generator",
                s,
                "stochastic",
                lsg_vector=lsg_vec,
                device=dev,
                n_episodes=n_episodes,
                life_rng_seed=hash(s) % 10_000,
            )
            for s in cheap
        ]
        for life in lives:
            _append_jsonl(ledger, {"run_id": run_id, "phase": "outer_train", **life.life_record})
        acc = sum(l.treatment_accuracy for l in lives) / len(lives)
        margin = sum(l.signed_margin_improvement for l in lives) / len(lives)
        retention = sum(l.retention_after_reset for l in lives) / len(lives)
        return SurfaceIndividual(
            surface={"lsg_norm": 0.0},
            fitness_key=LexicographicFitness(acc, margin, retention, 0.0),
            phenotype_hash=lives[-1].phenotype_hash,
            metrics={
                "train_accuracy_mean": acc,
                "signed_margin_improvement": margin,
                "retention_after_reset": retention,
                "update_norm_mean": sum(l.update_norm_mean for l in lives) / len(lives),
            },
        )

    es = run_batched_es_lsg(
        _eval,
        generations=generations,
        population_size=population_size,
        seed=hash("inherited_learning_signal_generator") % 1_000_000,
    )
    # Drop huge vectors from history file already omitted; store best vector separately.
    es_public = {k: v for k, v in es.items() if k != "best_lsg_vector"}
    es_public["best_lsg_vector_sha256"] = __import__("hashlib").sha256(
        repr(es["best_lsg_vector"]).encode()
    ).hexdigest() if es.get("best_lsg_vector") else None
    _atomic_write_json(out / "outer_es_inherited_learning_signal_generator.json", es_public)
    _atomic_write_json(out / "best_lsg_vector.json", {"vector": es["best_lsg_vector"]})

    if es["outer_updates_executed"] <= 0:
        summary = {"outcome": "outer_optimization_not_exercised", "candidates": [], "baseline": baseline_record}
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary

    best_vec = es["best_lsg_vector"]
    val_lives = [
        evaluate_r3_life(
            "inherited_learning_signal_generator",
            s,
            "hard",
            lsg_vector=best_vec,
            device=dev,
            n_episodes=n_episodes,
            life_rng_seed=hash(s) % 10_000,
        )
        for s in val
    ]
    for life in val_lives:
        _append_jsonl(ledger, {"run_id": run_id, "phase": "validation", **life.life_record})

    controls = run_r3_causal_controls(
        best_vec,
        cheap[0],
        device=dev,
        n_episodes=min(16, n_episodes),
        life_rng_seed=42,
    )
    _atomic_write_json(out / "controls_signal_generator.json", controls)

    val_acc = sum(l.treatment_accuracy for l in val_lives) / max(1, len(val_lives))
    train_last = es["history"][-1]["metrics"].get("train_accuracy_mean") if es["history"] else None
    first_fail = val_lives[0].first_failing_causal_predicate if val_lives else "unknown"

    acquisition = (train_last or 0.0) > (baseline_record["train_accuracy_mean"] + 0.01) or (
        train_last or 0.0
    ) > 0.05
    if not acquisition:
        decision = "training_acquisition_fail"
        first_fail = "training_acquisition_fail"
    elif not controls.get("reward_off_causality") or not controls.get("teacher_permutation", {}).get("passed"):
        decision = "credit_not_causal"
        first_fail = "credit_not_causal"
    elif not controls.get("treatment_outperforms_signal_controls"):
        decision = "credit_not_causal"
        first_fail = "credit_not_causal"
    elif val_acc <= 0.0 and (train_last or 0.0) < 0.05:
        decision = "reference_birth_r3_validation_fail"
    elif first_fail != "integrated_development_pass":
        decision = "reference_birth_r3_validation_fail"
    else:
        decision = "validation_pass"

    candidate = {
        "arm": "inherited_learning_signal_generator",
        "organism_candidate": True,
        "validation_accuracy_mean": val_acc,
        "train_accuracy_last": train_last,
        "outer_updates_executed": es["outer_updates_executed"],
        "best_fitness_key": es["best_fitness_key"],
        "controls": controls,
        "first_failing_causal_predicate": first_fail,
        "decision_code": decision,
        "best_lsg_vector_sha256": es_public["best_lsg_vector_sha256"],
    }

    summary = {
        "outcome": "reference_birth_r3_validation_complete",
        "baseline": baseline_record,
        "ceiling_accuracy": ceiling.treatment_accuracy,
        "candidates": [candidate],
        "confirmation_seeds_registered": confirmation_seeds,
        "confirmation_consumed": False,
        "outer_optimization_exercised": True,
        "r1_r2_confirmation_sealed": True,
    }
    _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
    _atomic_write_json(out / "search_summary.json", summary)
    return summary
