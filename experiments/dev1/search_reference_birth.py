"""
Stage A Reference Birth search runner.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.reference_birth_life import (
    REFERENCE_BIRTH_ARMS,
    evaluate_reference_birth_life,
    run_batched_lives_cuda,
)
from experiments.dev1.reference_birth_preflight import run_reference_birth_preflight
from experiments.dev1.search_r2 import _atomic_write_json, _append_jsonl, _rng_state_snapshot
from three_memory.dev1.device import dev1_device


TREATMENT_ARMS = [
    "reward_eprop_rate_adaptation",
    "teacher_demo_eprop",
    "r2_1_local_plasticity_control",
]


def run_stage_a_reference_birth_search(
    run_id: str,
    world_seeds: list[str],
    confirmation_seeds: list[str],
    output_dir: str = "runs/exos_dev1/stage_a_reference_birth",
    h_disabled: bool = True,
    preflight_seed: str = "reference_birth_excluded_preflight_20260820",
    require_cuda: bool = False,
    batch_size: int = 4,
) -> dict[str, Any]:
    out = Path(output_dir)
    dev = dev1_device(require_cuda=require_cuda and torch.cuda.is_available())

    _atomic_write_json(out / "run_started.json", {
        "run_id": run_id,
        "world_seeds": world_seeds,
        "confirmation_seeds": confirmation_seeds,
        "started_at": time.time(),
        "device": str(dev),
        "rng": _rng_state_snapshot(),
    })

    preflight = run_reference_birth_preflight(seed=preflight_seed)
    _atomic_write_json(out / "reference_birth_preflight.json", asdict(preflight))
    if not preflight.passed:
        summary = {"outcome": preflight.decision_code, "preflight": asdict(preflight), "candidates": []}
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary

    cheap = world_seeds[:2]
    val = world_seeds[2:6]
    ledger_path = out / "candidate_life_records.jsonl"
    candidates: list[dict[str, Any]] = []

    ceiling = evaluate_reference_birth_life(
        "conventional_actor_critic_ceiling",
        cheap[0],
        "stochastic",
        device=dev,
        h_disabled=h_disabled,
        n_episodes=8,
    )
    _append_jsonl(ledger_path, {"run_id": run_id, "arm": "ceiling", **ceiling.life_record})
    if ceiling.treatment_accuracy < 0.1:
        summary = {
            "outcome": "setup_ceiling_fail",
            "ceiling_accuracy": ceiling.treatment_accuracy,
            "candidates": [],
        }
        _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
        _atomic_write_json(out / "search_summary.json", summary)
        return summary

    for arm in TREATMENT_ARMS:
        train_lives = run_batched_lives_cuda(
            arm,
            cheap,
            batch_size=batch_size,
            policy_mode="stochastic",
            h_disabled=h_disabled,
            device=dev,
        )
        val_lives = run_batched_lives_cuda(
            arm,
            val,
            batch_size=batch_size,
            policy_mode="hard",
            h_disabled=h_disabled,
            device=dev,
        )
        for life in train_lives + val_lives:
            _append_jsonl(ledger_path, {"run_id": run_id, **life.life_record})
        train_acc = sum(l.treatment_accuracy for l in train_lives) / max(1, len(train_lives))
        val_acc = sum(l.treatment_accuracy for l in val_lives) / max(1, len(val_lives))
        first_fail = val_lives[0].first_failing_causal_predicate if val_lives else "unknown"
        candidates.append({
            "arm": arm,
            "training_accuracy_mean": train_acc,
            "validation_accuracy_mean": val_acc,
            "first_failing_causal_predicate": first_fail,
            "decision_code": "reference_birth_validation_fail" if first_fail != "integrated_development_pass" else "validation_pass",
        })

    summary = {
        "outcome": "reference_birth_validation_complete",
        "candidates": candidates,
        "confirmation_seeds_registered": confirmation_seeds,
        "confirmation_consumed": False,
    }
    _atomic_write_json(out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()})
    _atomic_write_json(out / "search_summary.json", summary)
    return summary
