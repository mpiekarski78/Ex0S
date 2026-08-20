"""
Developmental Birth R4-R2 scored search harness on Nursery Body v2.

Full evidence before decision ladder. Reference ceiling required on the scored
setup before ladder interpretation. Scored execution requires prereg
scored_run_authorized=true.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.developmental_birth_r4_r2_ceiling import (
    ceiling_body_behavior_passes,
    evaluate_ceiling_gate_bundle,
)
from experiments.dev1.developmental_birth_r4_r2_life import FACTORIAL_CELLS, evaluate_r4_r2_life
from experiments.dev1.developmental_birth_r4_r2_outer import (
    MatchedOuterBudget,
    default_fixed_credit_surface,
    default_lsg_surface_for_r4_r2,
    mutate_vector,
    run_matched_es_smoke,
)
from experiments.dev1.search_r2 import _append_jsonl, _atomic_write_json, _rng_state_snapshot
from three_memory.dev1.device import cuda_utilization_sample, dev1_device
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.plasticity.eprop.signal_generator import lsg_param_count

PREREG_PATH = Path("docs/exos_dev1.stage_a_developmental_birth_r4_r2.prereg.lock")
RUN_ID = "exos_dev1_developmental_birth_r4_r2_scored_20260820"
OUT = Path("runs/exos_dev1/stage_a_developmental_birth_r4_r2") / RUN_ID
NURSERY_FREEZE = "docs/exos_dev1.stage_a_nursery_body_v2.freeze.lock"

WORLD_SEEDS = [
    "exos_dev1_developmental_birth_r4_r2_world_001",
    "exos_dev1_developmental_birth_r4_r2_world_002",
    "exos_dev1_developmental_birth_r4_r2_world_003",
    "exos_dev1_developmental_birth_r4_r2_world_004",
    "exos_dev1_developmental_birth_r4_r2_world_005",
    "exos_dev1_developmental_birth_r4_r2_world_006",
]
CONFIRMATION_SEEDS = [
    "exos_dev1_developmental_birth_r4_r2_conf_001",
    "exos_dev1_developmental_birth_r4_r2_conf_002",
    "exos_dev1_developmental_birth_r4_r2_conf_003",
    "exos_dev1_developmental_birth_r4_r2_conf_004",
]

GENS = 8
POP = 8
N_EP = 32
EP_TICKS = 16
GEST_TICKS = 128
MUT = 0.15
EMBRYONIC = 0


def _load_prereg() -> dict[str, Any]:
    return json.loads(PREREG_PATH.read_text())


def _load_thresholds() -> dict[str, float]:
    t = _load_prereg()["thresholds"]
    return {
        "optimization_ceiling_min_end_in_zone_rate": float(
            t["optimization_ceiling_min_end_in_zone_rate"]
        ),
        "optimization_ceiling_min_distance_reduction": float(
            t["optimization_ceiling_min_distance_reduction"]
        ),
        "optimization_ceiling_min_margin_over_random": float(
            t["optimization_ceiling_min_margin_over_random"]
        ),
        "grounded_acquisition_min_train_end_in_zone": float(
            t["grounded_acquisition_min_train_end_in_zone"]
        ),
        "grounded_acquisition_min_signed_margin_proxy": float(
            t["grounded_acquisition_min_signed_margin_proxy"]
        ),
        "development_not_causal_if_sham_within_epsilon_of_active": float(
            t["development_not_causal_if_sham_within_epsilon_of_active"]
        ),
        "fresh_world_min_validation_end_in_zone": float(
            t["fresh_world_min_validation_end_in_zone"]
        ),
    }


def _life_dict(m) -> dict[str, Any]:
    return {
        "development": m.development,
        "credit": m.credit,
        "treatment_accuracy": m.treatment_accuracy,
        "end_in_zone_rate": m.end_in_zone_rate,
        "ever_reached_rate": m.ever_reached_rate,
        "distance_reduction": m.distance_reduction,
        "mean_behavioral_score": m.mean_behavioral_score,
        "mean_organism_valence": m.mean_organism_valence,
        "signed_margin_proxy": m.signed_margin_proxy,
        "phenotype_hash": m.phenotype_hash,
        "generative_genome_hash": m.generative_genome_hash,
        "construction_algorithm_hash": m.construction_algorithm_hash,
        "embryonic_seed": m.embryonic_seed,
        "pre_gestation_checkpoint_hash": m.pre_gestation_checkpoint_hash,
        "gestation_transcript_hash": m.gestation_transcript_hash,
        "post_gestation_checkpoint_hash": m.post_gestation_checkpoint_hash,
        "body_physics_hash": m.body_physics_hash,
        "credit_implementation": m.credit_implementation,
        "plasticity_updates": m.plasticity_updates,
        "teacher_demo_count": m.teacher_demo_count,
        "device": m.device,
        "intervention": m.intervention,
        "life_record": m.life_record,
    }


def _genome_fixed(surf: dict[str, float]) -> GenerativeGenome:
    g = GenerativeGenome.small(embryonic_seed=EMBRYONIC)
    g.gestation_ticks = GEST_TICKS
    g = g.with_credit_family("r2_fixed_eprop_baseline")
    g.learning_rate = float(math.exp(surf["log_actor_learning_rate"]))
    g.critic_learning_rate = float(math.exp(surf["log_critic_learning_rate"]))
    g.eligibility_decay = float(surf["eligibility_decay"])
    return g


def _genome_lsg(vec: list[float]) -> GenerativeGenome:
    g = GenerativeGenome.small(embryonic_seed=EMBRYONIC)
    g.gestation_ticks = GEST_TICKS
    g = g.with_credit_family("inherited_learning_signal_generator")
    g.lsg_param_vector = list(vec)
    return g


def apply_decision_ladder(
    ceiling: dict,
    cell_results: dict,
    interventions: dict,
    thresh: dict[str, float],
) -> str:
    """Ladder only after reference ceiling on the scored setup and full evidence."""
    beam = ceiling.get("exact_dynamics_beam_planner_reference") or {}
    if float(beam.get("end_in_zone_rate", 0.0)) < 0.50 or float(
        beam.get("distance_reduction", 0.0)
    ) < 0.10:
        return "body_controllability_fail"

    if not ceiling_body_behavior_passes(
        ceiling,
        {
            "min_end_in_zone_rate": thresh["optimization_ceiling_min_end_in_zone_rate"],
            "min_distance_reduction": thresh["optimization_ceiling_min_distance_reduction"],
            "min_margin_over_random": thresh["optimization_ceiling_min_margin_over_random"],
        },
    ):
        return "optimization_ceiling_fail"

    if interventions["lifetime_plasticity_off"]["end_in_zone_rate"] >= thresh[
        "grounded_acquisition_min_train_end_in_zone"
    ]:
        return "genome_overencoded_behavior"

    active_fixed = cell_results["active_gestation__r2_fixed_eprop_baseline"]
    active_lsg = cell_results["active_gestation__inherited_learning_signal_generator"]
    best_active_train = max(
        active_fixed["train_end_in_zone_mean"], active_lsg["train_end_in_zone_mean"]
    )
    best_active_margin = max(active_fixed["train_margin_mean"], active_lsg["train_margin_mean"])
    best_active_val = max(
        active_fixed["validation_end_in_zone_mean"], active_lsg["validation_end_in_zone_mean"]
    )

    if (
        best_active_train < thresh["grounded_acquisition_min_train_end_in_zone"]
        or best_active_margin < thresh["grounded_acquisition_min_signed_margin_proxy"]
    ):
        return "grounded_acquisition_fail"

    eps = thresh["development_not_causal_if_sham_within_epsilon_of_active"]
    sham_close = []
    for credit_key in ("r2_fixed_eprop_baseline", "inherited_learning_signal_generator"):
        a = cell_results[f"active_gestation__{credit_key}"]["train_end_in_zone_mean"]
        s = cell_results[f"sham_gestation__{credit_key}"]["train_end_in_zone_mean"]
        sham_close.append(abs(a - s) <= eps)
    if all(sham_close):
        return "development_not_causal"

    if interventions["reward_valence_off"]["end_in_zone_rate"] >= best_active_train - 1e-12:
        return "self_credit_not_causal"

    teacher_on = interventions.get("teacher_baseline")
    teacher_perm = interventions["teacher_demonstration_permutation"]
    if teacher_on is not None:
        if teacher_on["end_in_zone_rate"] >= thresh["grounded_acquisition_min_train_end_in_zone"]:
            if abs(teacher_perm["end_in_zone_rate"] - teacher_on["end_in_zone_rate"]) <= eps:
                return "teacher_credit_not_causal"

    if active_lsg["train_end_in_zone_mean"] >= active_fixed["train_end_in_zone_mean"] + 0.01:
        if (
            interventions["lsg_off"]["end_in_zone_rate"] >= active_lsg["train_end_in_zone_mean"] - eps
            and interventions["lsg_permuted"]["end_in_zone_rate"]
            >= active_lsg["train_end_in_zone_mean"] - eps
        ):
            return "lsg_not_causal"

    if best_active_val < thresh["fresh_world_min_validation_end_in_zone"]:
        return "fresh_world_fail"

    return "stage_a_r4_r2_validation_pass"


def run_scored() -> dict[str, Any]:
    prereg = _load_prereg()
    if not prereg.get("scored_run_authorized"):
        raise SystemExit(
            "R4-R2 scored run not authorized. Prereg scored_run_authorized=false."
        )
    thresh = _load_thresholds()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = OUT / "candidate_life_records.jsonl"
    cheap = WORLD_SEEDS[:2]
    val = WORLD_SEEDS[2:6]
    assert len(cheap) == 2 and len(val) == 4

    dev = dev1_device(require_cuda=True)
    import subprocess

    executing_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    _atomic_write_json(
        OUT / "run_started.json",
        {
            "run_id": RUN_ID,
            "revision": "DevelopmentalBirthR4-R2",
            "attempt": "001",
            "executing_head": executing_head,
            "nursery_freeze": NURSERY_FREEZE,
            "body": "NurseryBodyV2",
            "world_seeds": WORLD_SEEDS,
            "confirmation_seeds": CONFIRMATION_SEEDS,
            "started_at": time.time(),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(dev),
            "device_name": torch.cuda.get_device_name(0),
            "rng": _rng_state_snapshot(),
            "cuda_utilization": cuda_utilization_sample(),
            "protocol": "full_evidence_before_decision_ladder",
            "primary_scored_episode_success": "end_in_zone",
            "tick_fraction_comfort_retired": True,
            "budget": {
                "generations": GENS,
                "population_size": POP,
                "n_episodes": N_EP,
                "episode_ticks": EP_TICKS,
                "gestation_ticks": GEST_TICKS,
                "mutation_scale": MUT,
            },
            "factorial_cells": [list(c) for c in FACTORIAL_CELLS],
            "confirmation_sealed_until_validation_pass": True,
            "engineering_seeds_excluded": True,
        },
    )

    # Phase 1: reference ceiling on scored discovery setup (required before ladder)
    ceiling = evaluate_ceiling_gate_bundle(
        GenerativeGenome.small(EMBRYONIC),
        cheap[0] + ":ceiling",
        n_episodes=N_EP,
        episode_ticks=EP_TICKS,
        device=dev,
    )
    _atomic_write_json(OUT / "ceiling.json", ceiling)
    _append_jsonl(
        ledger,
        {
            "run_id": RUN_ID,
            "phase": "ceiling",
            "end_in_zone_rate": ceiling["end_in_zone_rate"],
            "ever_reached_rate": ceiling["ever_reached_rate"],
            "distance_reduction": ceiling["distance_reduction"],
            "comfort_margin_over_random": ceiling["comfort_margin_over_random"],
            "n_initializations_run": ceiling["n_initializations_run"],
        },
    )
    print(
        "PHASE_CEILING_DONE",
        ceiling["end_in_zone_rate"],
        ceiling["distance_reduction"],
        flush=True,
    )

    # Phase 2: matched outer ES
    surf = default_fixed_credit_surface(seed=EMBRYONIC)
    best_surf = dict(surf)
    best_fixed_acc = -1.0
    fixed_history = []
    fixed_lives = 0
    for gen_i in range(GENS):
        gen_accs = []
        for ind in range(POP):
            g = _genome_fixed(surf)
            accs = []
            for s in cheap:
                m = evaluate_r4_r2_life(
                    "active_gestation",
                    "r2_fixed_eprop_baseline",
                    f"{s}:outer_fixed:{gen_i}:{ind}",
                    generative=g,
                    n_episodes=N_EP,
                    episode_ticks=EP_TICKS,
                    embryonic_seed=EMBRYONIC,
                    life_rng_seed=gen_i * 1000 + ind,
                    device=dev,
                )
                fixed_lives += 1
                _append_jsonl(
                    ledger,
                    {
                        "run_id": RUN_ID,
                        "phase": "outer_fixed",
                        "gen": gen_i,
                        "ind": ind,
                        **_life_dict(m),
                    },
                )
                accs.append(m.end_in_zone_rate)
            mean_acc = sum(accs) / len(accs)
            gen_accs.append(mean_acc)
            if mean_acc > best_fixed_acc:
                best_fixed_acc = mean_acc
                best_surf = dict(surf)
            surf = {
                "log_actor_learning_rate": surf["log_actor_learning_rate"]
                + MUT * (0.5 - (ind % 3) / 2.0),
                "log_critic_learning_rate": surf["log_critic_learning_rate"]
                + MUT * (0.25 - (ind % 2) / 2.0),
                "eligibility_decay": min(
                    0.99, max(0.5, surf["eligibility_decay"] + MUT * 0.01 * ((ind % 5) - 2))
                ),
            }
        fixed_history.append(
            {
                "generation": gen_i,
                "mean_end_in_zone": sum(gen_accs) / len(gen_accs),
                "best_so_far": best_fixed_acc,
            }
        )
        print("OUTER_FIXED_GEN", gen_i, flush=True)

    base_g = GenerativeGenome.small(EMBRYONIC)
    vec = default_lsg_surface_for_r4_r2(base_g, seed=EMBRYONIC)
    assert len(vec) == lsg_param_count(base_g.n_motor_channels, base_g.action_units)
    best_vec = list(vec)
    best_lsg_acc = -1.0
    lsg_history = []
    lsg_lives = 0
    for gen_i in range(GENS):
        gen_accs = []
        for ind in range(POP):
            g = _genome_lsg(vec)
            accs = []
            for s in cheap:
                m = evaluate_r4_r2_life(
                    "active_gestation",
                    "inherited_learning_signal_generator",
                    f"{s}:outer_lsg:{gen_i}:{ind}",
                    generative=g,
                    n_episodes=N_EP,
                    episode_ticks=EP_TICKS,
                    embryonic_seed=EMBRYONIC,
                    life_rng_seed=gen_i * 1000 + ind,
                    device=dev,
                )
                lsg_lives += 1
                _append_jsonl(
                    ledger,
                    {
                        "run_id": RUN_ID,
                        "phase": "outer_lsg",
                        "gen": gen_i,
                        "ind": ind,
                        **_life_dict(m),
                    },
                )
                accs.append(m.end_in_zone_rate)
            mean_acc = sum(accs) / len(accs)
            gen_accs.append(mean_acc)
            if mean_acc > best_lsg_acc:
                best_lsg_acc = mean_acc
                best_vec = list(vec)
            vec = mutate_vector(vec, MUT, seed=gen_i * 10000 + ind)
        lsg_history.append(
            {
                "generation": gen_i,
                "mean_end_in_zone": sum(gen_accs) / len(gen_accs),
                "best_so_far": best_lsg_acc,
            }
        )
        print("OUTER_LSG_GEN", gen_i, flush=True)

    assert fixed_lives == lsg_lives
    _atomic_write_json(
        OUT / "outer_matched_es.json",
        {
            "fixed_lives": fixed_lives,
            "lsg_lives": lsg_lives,
            "matched": fixed_lives == lsg_lives,
            "best_fixed_end_in_zone": best_fixed_acc,
            "best_lsg_end_in_zone": best_lsg_acc,
            "fixed_history": fixed_history,
            "lsg_history": lsg_history,
            "best_fixed_surface": best_surf,
            "best_lsg_vector_len": len(best_vec),
            "same_nursery_v2_trajectories_and_budgets": True,
        },
    )
    print("PHASE_OUTER_DONE", fixed_lives, lsg_lives, flush=True)

    # Phase 3: factorial
    cell_results: dict[str, Any] = {}
    for development, credit in FACTORIAL_CELLS:
        g = _genome_fixed(best_surf) if credit == "r2_fixed_eprop_baseline" else _genome_lsg(best_vec)
        key = f"{development}__{credit}"
        train_lives = []
        for s in cheap:
            m = evaluate_r4_r2_life(
                development,
                credit,
                f"{s}:factorial_train:{key}",
                generative=g,
                n_episodes=N_EP,
                episode_ticks=EP_TICKS,
                embryonic_seed=EMBRYONIC,
                life_rng_seed=hash(s + key) % 10_000,
                device=dev,
            )
            train_lives.append(m)
            _append_jsonl(
                ledger, {"run_id": RUN_ID, "phase": "factorial_train", "cell": key, **_life_dict(m)}
            )
        val_lives = []
        for s in val:
            m = evaluate_r4_r2_life(
                development,
                credit,
                f"{s}:factorial_val:{key}",
                generative=g,
                n_episodes=N_EP,
                episode_ticks=EP_TICKS,
                embryonic_seed=EMBRYONIC,
                life_rng_seed=hash(s + key) % 10_000,
                device=dev,
            )
            val_lives.append(m)
            _append_jsonl(
                ledger, {"run_id": RUN_ID, "phase": "factorial_val", "cell": key, **_life_dict(m)}
            )
        cell_results[key] = {
            "development": development,
            "credit": credit,
            "train_end_in_zone_mean": sum(x.end_in_zone_rate for x in train_lives) / len(train_lives),
            "train_ever_reached_mean": sum(x.ever_reached_rate for x in train_lives)
            / len(train_lives),
            "train_distance_reduction_mean": sum(x.distance_reduction for x in train_lives)
            / len(train_lives),
            "train_margin_mean": sum(x.signed_margin_proxy for x in train_lives) / len(train_lives),
            "validation_end_in_zone_mean": sum(x.end_in_zone_rate for x in val_lives)
            / len(val_lives),
            "validation_ever_reached_mean": sum(x.ever_reached_rate for x in val_lives)
            / len(val_lives),
            "validation_distance_reduction_mean": sum(x.distance_reduction for x in val_lives)
            / len(val_lives),
            "pre_gestation_hashes": sorted(
                {x.pre_gestation_checkpoint_hash for x in train_lives + val_lives}
            ),
        }
        print("FACTORIAL_CELL_DONE", key, flush=True)
    _atomic_write_json(OUT / "factorial_cells.json", cell_results)

    # Phase 4: interventions
    interventions: dict[str, Any] = {}
    intervention_specs = [
        ("zero_tick_skip", dict(development="zero_tick_skip", credit="r2_fixed_eprop_baseline")),
        ("gestational_plasticity_off", dict(gestational_plasticity_off=True)),
        ("lifetime_plasticity_off", dict(lifetime_plasticity_off=True, use_teacher=False)),
        ("reward_valence_off", dict(reward_valence_off=True, use_teacher=False)),
        ("motor_to_body_permutation", dict(motor_permutation=True, use_teacher=False)),
        ("proprioceptive_feedback_permutation", dict(proprio_permutation=True, use_teacher=False)),
        ("teacher_baseline", dict()),
        ("teacher_demonstration_permutation", dict(permute_teacher=True)),
        ("open_loop_or_shuffled_action_consequences", dict(open_loop=True, use_teacher=False)),
        (
            "lsg_off",
            dict(credit="inherited_learning_signal_generator", lsg_off=True, use_teacher=False),
        ),
        (
            "lsg_permuted",
            dict(
                credit="inherited_learning_signal_generator",
                lsg_permuted=True,
                use_teacher=False,
            ),
        ),
        ("teacher_credit_off", dict(use_teacher=False)),
    ]
    for name, kwargs in intervention_specs:
        development = kwargs.pop("development", "active_gestation")
        credit = kwargs.pop("credit", "r2_fixed_eprop_baseline")
        g = _genome_fixed(best_surf) if credit == "r2_fixed_eprop_baseline" else _genome_lsg(best_vec)
        m = evaluate_r4_r2_life(
            development,
            credit,
            f"{cheap[0]}:interv:{name}",
            generative=g,
            n_episodes=N_EP,
            episode_ticks=EP_TICKS,
            embryonic_seed=EMBRYONIC,
            life_rng_seed=hash(name) % 10_000,
            device=dev,
            **kwargs,
        )
        interventions[name] = _life_dict(m)
        _append_jsonl(
            ledger, {"run_id": RUN_ID, "phase": "intervention", "name": name, **_life_dict(m)}
        )
        print("INTERVENTION_DONE", name, flush=True)
    _atomic_write_json(
        OUT / "interventions.json",
        {
            k: {
                "end_in_zone": v["end_in_zone_rate"],
                "ever_reached": v["ever_reached_rate"],
                "distance_reduction": v["distance_reduction"],
                "intervention": v["intervention"],
                "plast": v["plasticity_updates"],
            }
            for k, v in interventions.items()
        },
    )

    # Phase 5: decision ladder (ceiling already on scored setup)
    decision = apply_decision_ladder(ceiling, cell_results, interventions, thresh)
    confirmation_consumed = False
    confirmation_results = None
    if decision == "stage_a_r4_r2_validation_pass":
        confirmation_results = {}
        active_fixed = cell_results["active_gestation__r2_fixed_eprop_baseline"]
        active_lsg = cell_results["active_gestation__inherited_learning_signal_generator"]
        if active_lsg["validation_end_in_zone_mean"] >= active_fixed["validation_end_in_zone_mean"]:
            conf_credit = "inherited_learning_signal_generator"
            g = _genome_lsg(best_vec)
        else:
            conf_credit = "r2_fixed_eprop_baseline"
            g = _genome_fixed(best_surf)
        # Ceiling also on confirmation setup before consuming confirmation
        conf_ceiling = evaluate_ceiling_gate_bundle(
            GenerativeGenome.small(EMBRYONIC),
            CONFIRMATION_SEEDS[0] + ":ceiling",
            n_episodes=N_EP,
            episode_ticks=EP_TICKS,
            device=dev,
        )
        _atomic_write_json(OUT / "confirmation_ceiling.json", conf_ceiling)
        for s in CONFIRMATION_SEEDS:
            m = evaluate_r4_r2_life(
                "active_gestation",
                conf_credit,
                f"{s}:confirmation",
                generative=g,
                n_episodes=N_EP,
                episode_ticks=EP_TICKS,
                embryonic_seed=EMBRYONIC,
                life_rng_seed=hash(s) % 10_000,
                device=dev,
            )
            confirmation_results[s] = _life_dict(m)
            _append_jsonl(ledger, {"run_id": RUN_ID, "phase": "confirmation", **_life_dict(m)})
        confirmation_consumed = True
        conf_acc = sum(v["end_in_zone_rate"] for v in confirmation_results.values()) / len(
            confirmation_results
        )
        if conf_acc >= thresh["fresh_world_min_validation_end_in_zone"]:
            decision = "stage_a_r4_r2_confirmation_pass"
        else:
            decision = "fresh_world_fail"

    summary = {
        "outcome": "developmental_birth_r4_r2_scored_complete",
        "revision": "DevelopmentalBirthR4-R2",
        "attempt": "001",
        "decision_code": decision,
        "body": "NurseryBodyV2",
        "ceiling": {
            "end_in_zone_rate": ceiling["end_in_zone_rate"],
            "ever_reached_rate": ceiling["ever_reached_rate"],
            "distance_reduction": ceiling["distance_reduction"],
            "comfort_margin_over_random": ceiling["comfort_margin_over_random"],
            "ceiling_kind": ceiling["ceiling_kind"],
            "n_initializations_run": ceiling["n_initializations_run"],
            "ac_retrain_not_retry_until_pass": True,
        },
        "factorial_cells": cell_results,
        "outer_matched": {"fixed_lives": fixed_lives, "lsg_lives": lsg_lives, "matched": True},
        "interventions_executed": sorted(interventions.keys()),
        "confirmation_seeds_registered": CONFIRMATION_SEEDS,
        "confirmation_consumed": confirmation_consumed,
        "confirmation_results_present": confirmation_results is not None,
        "executing_head": executing_head,
        "nursery_freeze": NURSERY_FREEZE,
        "thresholds": thresh,
        "protocol": "full_evidence_before_decision_ladder",
        "primary_scored_episode_success": "end_in_zone",
        "cuda_utilization_end": cuda_utilization_sample(),
    }
    _atomic_write_json(
        OUT / "run_completed.json", {"run_id": RUN_ID, "summary": summary, "completed_at": time.time()}
    )
    _atomic_write_json(OUT / "search_summary.json", summary)
    return summary


def run_unscored_smoke(device: torch.device | None = None) -> dict[str, Any]:
    """Implementation smoke only — not a scored ladder run."""
    dev = device or torch.device("cpu")
    ceiling = evaluate_ceiling_gate_bundle(
        GenerativeGenome.small(EMBRYONIC),
        WORLD_SEEDS[0] + ":smoke_ceiling",
        n_episodes=2,
        episode_ticks=4,
        device=dev,
        train_episodes=8,
    )
    cells = {}
    for development, credit in FACTORIAL_CELLS:
        m = evaluate_r4_r2_life(
            development,
            credit,
            f"{WORLD_SEEDS[0]}:smoke:{development}:{credit}",
            n_episodes=1,
            episode_ticks=4,
            embryonic_seed=EMBRYONIC,
            device=dev,
        )
        cells[f"{development}__{credit}"] = {
            "end_in_zone_rate": m.end_in_zone_rate,
            "ever_reached_rate": m.ever_reached_rate,
            "distance_reduction": m.distance_reduction,
            "pre_gestation_checkpoint_hash": m.pre_gestation_checkpoint_hash,
        }
    es = run_matched_es_smoke(
        WORLD_SEEDS[0] + ":smoke_es",
        MatchedOuterBudget(population=2, generations=1, n_episodes=1, episode_ticks=4),
        device=dev,
    )
    return {
        "ceiling_end_in_zone": ceiling["end_in_zone_rate"],
        "ceiling_inits": ceiling["n_initializations_run"],
        "cells": cells,
        "es_matched": es["matched"],
        "factorial_cells": [list(c) for c in FACTORIAL_CELLS],
        "world_seeds": WORLD_SEEDS,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "device": str(dev),
        "cuda_utilization": cuda_utilization_sample(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", action="store_true")
    p.add_argument(
        "--out",
        default="runs/exos_dev1/stage_a_developmental_birth_r4_r2/unscored_smoke.json",
    )
    args = p.parse_args()
    if args.scored:
        try:
            summary = run_scored()
            print("SCORED_DONE", summary.get("decision_code"), flush=True)
        except Exception as exc:
            OUT.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": RUN_ID,
                "revision": "DevelopmentalBirthR4-R2",
                "attempt": "001",
                "failed_at": time.time(),
                "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "nursery_freeze": NURSERY_FREEZE,
                "infrastructure_failure": True,
                "silent_replay_forbidden": True,
            }
            _atomic_write_json(OUT / "run_failed.json", payload)
            print("SCORED_FAILED", exc, flush=True)
            raise
        return
    report = run_unscored_smoke()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"ok": True, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
