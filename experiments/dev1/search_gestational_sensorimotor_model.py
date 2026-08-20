"""
Gestational Sensorimotor Model scored search harness on Nursery Body v2.

Model certification before behavioral scoring. Full evidence before the
decision ladder. Confirmation sealed unless validation passes. Scored
execution requires prereg scored_run_authorized=true.

Rehearsal (--rehearsal) is infrastructure-only on excluded seeds with a
reduced budget; it must not retune behavioral thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.developmental_birth_r4_r2_ceiling import (
    ceiling_body_behavior_passes,
    evaluate_ceiling_gate_bundle,
)
from experiments.dev1.gsm_life import (
    DEFAULT_UNCERTAINTY_MAX,
    FALLBACK_POLICY,
    evaluate_gsm_life,
)
from experiments.dev1.gsm_model_certification import run_model_certification
from experiments.dev1.search_r2 import _append_jsonl, _atomic_write_json, _rng_state_snapshot
from three_memory.dev1.device import cuda_utilization_sample, dev1_device
from three_memory.dev1.development.generative_genome import GenerativeGenome

PREREG_PATH = Path("docs/exos_dev1.stage_a_gestational_sensorimotor_model.prereg.lock")
CONTRACT_PATH = Path("docs/exos_dev1.stage_a_gestational_sensorimotor_model.contract.lock")
RUNNER_FILE = "experiments/dev1/search_gestational_sensorimotor_model.py"
NURSERY_FREEZE = "docs/exos_dev1.stage_a_nursery_body_v2.freeze.lock"

REQUIRED_ARMS = (
    "sham_gestation",
    "existing_homeostatic_gestation",
    "predictive_gestation",
    "predictive_gestation_shuffled_consequences",
    "learned_model_off_at_action_selection",
)

SEALED_R4R2_PREFIX = "exos_dev1_developmental_birth_r4_r2_"

REHEARSAL_SEED = "gestational_sensorimotor_model_excluded_rehearsal_20260820"
REHEARSAL_RUN_ID = "exos_dev1_gestational_sensorimotor_model_excluded_rehearsal_20260820"


def _load_prereg() -> dict[str, Any]:
    return json.loads(PREREG_PATH.read_text())


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _life_dict(m) -> dict[str, Any]:
    return {
        "arm": m.arm,
        "end_in_zone_rate": m.end_in_zone_rate,
        "ever_reached_rate": m.ever_reached_rate,
        "distance_reduction": m.distance_reduction,
        "mean_model_uncertainty": m.mean_model_uncertainty,
        "fraction_model_actions": m.fraction_model_actions,
        "fraction_fallback_actions": m.fraction_fallback_actions,
        "mean_abs_calibration_error": m.mean_abs_calibration_error,
        "plasticity_updates": m.plasticity_updates,
        "model_updates": m.model_updates,
        "pre_gestation_checkpoint_hash": m.pre_gestation_checkpoint_hash,
        "post_gestation_checkpoint_hash": m.post_gestation_checkpoint_hash,
        "all_finite": m.all_finite,
        "life_record": m.life_record,
    }


def _mean_field(lives: list[dict[str, Any]], key: str) -> float:
    return sum(float(x[key]) for x in lives) / max(1, len(lives))


def apply_decision_ladder(
    *,
    cert: dict[str, Any],
    ceiling: dict[str, Any],
    arms: dict[str, Any],
    interventions: dict[str, Any],
    thresh: dict[str, float],
    evidence_complete: bool,
) -> str:
    """Apply ladder only after model certification and full behavioral evidence."""
    if not evidence_complete:
        return "evidence_truncated_forbidden"
    if not cert.get("all_certified"):
        return "model_certification_fail"

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

    pred = arms["predictive_gestation"]
    sham = arms["sham_gestation"]
    shuffled = arms["predictive_gestation_shuffled_consequences"]
    model_off = arms["learned_model_off_at_action_selection"]
    open_loop = interventions["open_loop"]

    if float(pred["train_end_in_zone_mean"]) < thresh["predictive_min_train_end_in_zone"]:
        return "predictive_acquisition_fail"
    if float(pred["train_distance_reduction_mean"]) < thresh["predictive_min_distance_reduction"]:
        return "predictive_acquisition_fail"

    if float(pred["fraction_model_actions_mean"]) < thresh["min_fraction_model_actions"]:
        return "treatment_primarily_fallback"
    if bool(pred.get("systematic_misprediction_risk")):
        return "model_exploitation_risk"

    eps = thresh["active_beats_control_epsilon"]
    if float(pred["train_end_in_zone_mean"]) <= float(sham["train_end_in_zone_mean"]) + eps:
        return "active_not_better_than_sham"
    if float(pred["train_end_in_zone_mean"]) <= float(shuffled["train_end_in_zone_mean"]) + eps:
        return "shuffled_consequence_not_causal"
    if float(pred["train_end_in_zone_mean"]) <= float(model_off["train_end_in_zone_mean"]) + eps:
        return "model_off_does_not_remove_gain"
    if float(pred["train_end_in_zone_mean"]) <= float(open_loop["end_in_zone_rate"]) + eps:
        return "closed_loop_not_better_than_open_loop"

    if float(pred["validation_end_in_zone_mean"]) < thresh["fresh_world_min_validation_end_in_zone"]:
        return "fresh_world_fail"

    return "stage_a_gsm_validation_pass"


def _budget_from_prereg(prereg: dict[str, Any], *, rehearsal: bool) -> dict[str, int]:
    b = prereg["budget"]
    if rehearsal:
        return {
            "gestation_ticks": 24,
            "n_episodes_per_life": 4,
            "episode_ticks": 8,
            "cert_episodes": 8,
            "cert_epochs": 15,
            "discovery_seeds": 1,
            "validation_seeds": 1,
        }
    return {
        "gestation_ticks": int(b["gestation_ticks"]),
        "n_episodes_per_life": int(b["n_episodes_per_life"]),
        "episode_ticks": int(b["episode_ticks"]),
        "cert_episodes": 24,
        "cert_epochs": 40,
        "discovery_seeds": int(b["discovery_seeds"]),
        "validation_seeds": int(b["validation_seeds"]),
    }


def _thresholds(prereg: dict[str, Any]) -> dict[str, float]:
    t = prereg["thresholds"]
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
        "predictive_min_train_end_in_zone": float(t["predictive_min_train_end_in_zone"]),
        "predictive_min_distance_reduction": float(t["predictive_min_distance_reduction"]),
        "active_beats_control_epsilon": float(t["active_beats_control_epsilon"]),
        "fresh_world_min_validation_end_in_zone": float(
            t["fresh_world_min_validation_end_in_zone"]
        ),
        "min_fraction_model_actions": float(t["min_fraction_model_actions"]),
        "uncertainty_max": float(t["uncertainty_max"]),
    }


def _assert_seed_policy(prereg: dict[str, Any], seeds: list[str], *, allow_excluded: bool) -> None:
    excl = set(prereg["seed_partitions"]["excluded_seeds"])
    for s in seeds:
        if s.startswith(SEALED_R4R2_PREFIX):
            raise RuntimeError(f"sealed R4-R2 seed forbidden: {s}")
        if (s in excl) and (not allow_excluded):
            raise RuntimeError(f"excluded seed used in scored path: {s}")
        if (s not in excl) and allow_excluded and s.startswith("exos_dev1_gestational_sensorimotor_model_"):
            raise RuntimeError(f"scored partition seed used in rehearsal: {s}")


def run_protocol(*, rehearsal: bool = False) -> dict[str, Any]:
    prereg = _load_prereg()
    if (not rehearsal) and (not prereg.get("scored_run_authorized")):
        raise SystemExit(
            "GSM scored run not authorized. Prereg scored_run_authorized=false."
        )

    uncertainty = prereg["uncertainty_fallback"]
    assert float(uncertainty["uncertainty_max"]) == float(DEFAULT_UNCERTAINTY_MAX)
    assert uncertainty["fallback_policy"] == FALLBACK_POLICY
    assert uncertainty["no_planner"] is True
    assert uncertainty["no_expected_action"] is True
    assert uncertainty["no_runner_selected_synergy"] is True

    thresh = _thresholds(prereg)
    budget = _budget_from_prereg(prereg, rehearsal=rehearsal)

    if rehearsal:
        run_id = REHEARSAL_RUN_ID
        out = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_model") / run_id
        discovery = [f"{REHEARSAL_SEED}:disc"]
        validation = [f"{REHEARSAL_SEED}:val"]
        confirmation = [
            f"{REHEARSAL_SEED}:conf_001",
            f"{REHEARSAL_SEED}:conf_002",
            f"{REHEARSAL_SEED}:conf_003",
            f"{REHEARSAL_SEED}:conf_004",
        ]
        cert_seeds = [f"{REHEARSAL_SEED}:cert"]
        allow_excluded = True
        require_cuda = True  # rehearsal uses GPU when protocol is exercised on this host
    else:
        run_id = prereg["run_identity"]["run_id"]
        out = Path(prereg["run_identity"]["output_dir"])
        discovery = list(prereg["seed_partitions"]["discovery_world_seeds"])
        validation = list(prereg["seed_partitions"]["validation_world_seeds"])
        confirmation = list(prereg["seed_partitions"]["confirmation_seeds"])
        cert_seeds = list(prereg["seed_partitions"].get("engineering_cert_seeds") or []) or [
            "gsm_eng_cert_seed_001",
            "gsm_eng_cert_seed_002",
            "gsm_eng_cert_seed_003",
        ]
        # Engineering cert seeds are excluded from scored behavioral partitions but
        # permitted for the mandatory model-certification phase.
        allow_excluded = False
        require_cuda = bool(prereg["budget"].get("scored_run_cuda_required", True))
        assert len(discovery) == budget["discovery_seeds"]
        assert len(validation) == budget["validation_seeds"]

    _assert_seed_policy(prereg, discovery + validation + confirmation, allow_excluded=allow_excluded)
    for s in discovery + validation + confirmation:
        if s.startswith(SEALED_R4R2_PREFIX):
            raise RuntimeError("R4-R2 sealed seed leaked into GSM protocol")

    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "candidate_life_records.jsonl"
    if ledger.exists():
        ledger.unlink()

    executing_head = _git_head()
    runner_sha = _file_sha(RUNNER_FILE)
    impl_sha = prereg["provenance"]["implementation_sha"]
    dev = dev1_device(require_cuda=require_cuda)

    _atomic_write_json(
        out / "run_started.json",
        {
            "run_id": run_id,
            "revision": "GestationalSensorimotorModel",
            "attempt": prereg["run_identity"].get("attempt", "001"),
            "rehearsal": bool(rehearsal),
            "infrastructure_only": bool(rehearsal),
            "do_not_tune_thresholds_from_rehearsal": bool(rehearsal),
            "executing_head": executing_head,
            "implementation_sha": impl_sha,
            "runner_file": RUNNER_FILE,
            "runner_sha256": runner_sha,
            "nursery_freeze": NURSERY_FREEZE,
            "body": "NurseryBodyV2",
            "discovery_seeds": discovery,
            "validation_seeds": validation,
            "confirmation_seeds": confirmation,
            "confirmation_sealed_until_validation_pass": True,
            "started_at": time.time(),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(dev),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "rng": _rng_state_snapshot(),
            "cuda_utilization": cuda_utilization_sample(),
            "protocol": "full_evidence_before_decision_ladder",
            "uncertainty_max": thresh["uncertainty_max"],
            "fallback_policy": FALLBACK_POLICY,
            "budget": budget,
            "required_arms": list(REQUIRED_ARMS),
        },
    )

    # Phase 1: model certification (required before behavioral scoring)
    cert_rows = []
    for seed in cert_seeds:
        if seed.startswith(SEALED_R4R2_PREFIX):
            raise RuntimeError("sealed R4-R2 seed in certification")
        c = run_model_certification(
            seed,
            device=dev,
            n_episodes=budget["cert_episodes"],
            episode_ticks=budget["episode_ticks"],
            epochs=budget["cert_epochs"],
        )
        cert_rows.append(c)
        _append_jsonl(
            ledger,
            {
                "run_id": run_id,
                "phase": "model_certification",
                "world_seed": seed,
                "certified": c["certified"],
                "checks": c["certification_checks"],
            },
        )
        print("CERT_DONE", seed, c["certified"], flush=True)
    cert_report = {
        "seeds": [c["world_seed"] for c in cert_rows],
        "all_certified": all(c["certified"] for c in cert_rows),
        "rows": [
            {"world_seed": c["world_seed"], "certified": c["certified"], "checks": c["certification_checks"]}
            for c in cert_rows
        ],
    }
    _atomic_write_json(out / "model_certification.json", cert_report)

    # Phase 2: reference ceiling on discovery setup
    ceiling = evaluate_ceiling_gate_bundle(
        GenerativeGenome.small(0),
        discovery[0] + ":ceiling",
        n_episodes=budget["n_episodes_per_life"],
        episode_ticks=budget["episode_ticks"],
        device=dev,
        train_episodes=max(8, budget["n_episodes_per_life"] * 2) if rehearsal else None,
    )
    _atomic_write_json(out / "ceiling.json", ceiling)
    _append_jsonl(
        ledger,
        {
            "run_id": run_id,
            "phase": "ceiling",
            "end_in_zone_rate": ceiling["end_in_zone_rate"],
            "distance_reduction": ceiling["distance_reduction"],
            "comfort_margin_over_random": ceiling["comfort_margin_over_random"],
            "exact_dynamics_beam_not_organism_candidate": True,
        },
    )
    print("PHASE_CEILING_DONE", ceiling["end_in_zone_rate"], flush=True)

    # Phase 3: required behavioral arms (discovery + validation) — no early ladder
    arm_results: dict[str, Any] = {}
    for arm in REQUIRED_ARMS:
        train_lives = []
        for s in discovery:
            m = evaluate_gsm_life(
                arm,
                f"{s}:arm:{arm}",
                n_episodes=budget["n_episodes_per_life"],
                episode_ticks=budget["episode_ticks"],
                gestation_ticks=budget["gestation_ticks"],
                uncertainty_max=thresh["uncertainty_max"],
                embryonic_seed=0,
                body_seed=1,
                life_rng_seed=hash(arm + s) % 10_000,
                device=dev,
            )
            if not m.all_finite:
                raise RuntimeError(f"non-finite GSM values on arm {arm}")
            d = _life_dict(m)
            train_lives.append(d)
            _append_jsonl(ledger, {"run_id": run_id, "phase": "arm_train", **d})
        val_lives = []
        for s in validation:
            m = evaluate_gsm_life(
                arm,
                f"{s}:arm:{arm}",
                n_episodes=budget["n_episodes_per_life"],
                episode_ticks=budget["episode_ticks"],
                gestation_ticks=budget["gestation_ticks"],
                uncertainty_max=thresh["uncertainty_max"],
                embryonic_seed=0,
                body_seed=1,
                life_rng_seed=hash(arm + s + "val") % 10_000,
                device=dev,
            )
            if not m.all_finite:
                raise RuntimeError(f"non-finite GSM values on arm {arm} validation")
            d = _life_dict(m)
            val_lives.append(d)
            _append_jsonl(ledger, {"run_id": run_id, "phase": "arm_val", **d})
        arm_results[arm] = {
            "train_end_in_zone_mean": _mean_field(train_lives, "end_in_zone_rate"),
            "train_ever_reached_mean": _mean_field(train_lives, "ever_reached_rate"),
            "train_distance_reduction_mean": _mean_field(train_lives, "distance_reduction"),
            "validation_end_in_zone_mean": _mean_field(val_lives, "end_in_zone_rate"),
            "validation_distance_reduction_mean": _mean_field(val_lives, "distance_reduction"),
            "fraction_model_actions_mean": _mean_field(train_lives, "fraction_model_actions"),
            "fraction_fallback_actions_mean": _mean_field(train_lives, "fraction_fallback_actions"),
            "systematic_misprediction_risk": any(
                bool(x["life_record"].get("systematic_misprediction_risk")) for x in train_lives
            ),
            "action_source_counts": train_lives[0]["life_record"].get("action_source_counts", {}),
            "pre_gestation_hashes": sorted(
                {x["pre_gestation_checkpoint_hash"] for x in train_lives + val_lives}
            ),
        }
        print("ARM_DONE", arm, flush=True)
    _atomic_write_json(out / "arms.json", arm_results)

    # Phase 4: interventions (model-off already an arm; add open-loop + uncertainty fallback)
    interventions: dict[str, Any] = {}
    open_m = evaluate_gsm_life(
        "predictive_gestation",
        f"{discovery[0]}:interv:open_loop",
        n_episodes=budget["n_episodes_per_life"],
        episode_ticks=budget["episode_ticks"],
        gestation_ticks=budget["gestation_ticks"],
        uncertainty_max=thresh["uncertainty_max"],
        open_loop=True,
        embryonic_seed=0,
        body_seed=1,
        life_rng_seed=17,
        device=dev,
    )
    interventions["open_loop"] = _life_dict(open_m)
    _append_jsonl(
        ledger, {"run_id": run_id, "phase": "intervention", "name": "open_loop", **_life_dict(open_m)}
    )

    fb_m = evaluate_gsm_life(
        "predictive_gestation",
        f"{discovery[0]}:interv:uncertainty_fallback",
        n_episodes=budget["n_episodes_per_life"],
        episode_ticks=budget["episode_ticks"],
        gestation_ticks=budget["gestation_ticks"],
        uncertainty_max=thresh["uncertainty_max"],
        force_uncertainty_fallback=True,
        embryonic_seed=0,
        body_seed=1,
        life_rng_seed=19,
        device=dev,
    )
    interventions["uncertainty_fallback"] = _life_dict(fb_m)
    _append_jsonl(
        ledger,
        {
            "run_id": run_id,
            "phase": "intervention",
            "name": "uncertainty_fallback",
            **_life_dict(fb_m),
        },
    )
    if fb_m.fraction_model_actions != 0.0:
        raise RuntimeError("uncertainty fallback path still used model actions")
    if fb_m.fraction_fallback_actions <= 0.0:
        raise RuntimeError("uncertainty fallback path produced no fallback actions")

    # Exact body-dynamics measurement ceiling only (never organism candidate)
    interventions["exact_body_dynamics_model_measurement_ceiling_only"] = {
        "end_in_zone_rate": float(
            ceiling["exact_dynamics_beam_planner_reference"]["end_in_zone_rate"]
        ),
        "distance_reduction": float(
            ceiling["exact_dynamics_beam_planner_reference"]["distance_reduction"]
        ),
        "not_organism_candidate": True,
        "from_ceiling_bundle": True,
    }
    _atomic_write_json(out / "interventions.json", interventions)
    print("INTERVENTIONS_DONE", flush=True)

    # Phase 5: decision ladder only after full evidence
    evidence_complete = True
    decision = apply_decision_ladder(
        cert=cert_report,
        ceiling=ceiling,
        arms=arm_results,
        interventions=interventions,
        thresh=thresh,
        evidence_complete=evidence_complete,
    )

    confirmation_consumed = False
    confirmation_results = None
    confirmation_refusal_reason = None
    if rehearsal:
        confirmation_refusal_reason = "rehearsal_confirmation_sealed"
    elif decision != "stage_a_gsm_validation_pass":
        confirmation_refusal_reason = f"validation_did_not_pass:{decision}"
    else:
        confirmation_results = {}
        for s in confirmation:
            m = evaluate_gsm_life(
                "predictive_gestation",
                f"{s}:confirmation",
                n_episodes=budget["n_episodes_per_life"],
                episode_ticks=budget["episode_ticks"],
                gestation_ticks=budget["gestation_ticks"],
                uncertainty_max=thresh["uncertainty_max"],
                embryonic_seed=0,
                body_seed=1,
                life_rng_seed=hash(s) % 10_000,
                device=dev,
            )
            confirmation_results[s] = _life_dict(m)
            _append_jsonl(ledger, {"run_id": run_id, "phase": "confirmation", **_life_dict(m)})
        confirmation_consumed = True
        conf_acc = sum(v["end_in_zone_rate"] for v in confirmation_results.values()) / len(
            confirmation_results
        )
        if conf_acc >= thresh["fresh_world_min_validation_end_in_zone"]:
            decision = "stage_a_gsm_confirmation_pass"
        else:
            decision = "fresh_world_fail"

    if confirmation_refusal_reason is not None:
        _atomic_write_json(
            out / "confirmation_refusal.json",
            {
                "refused": True,
                "reason": confirmation_refusal_reason,
                "confirmation_seeds_registered": confirmation,
                "confirmation_consumed": False,
            },
        )

    summary = {
        "outcome": (
            "gestational_sensorimotor_model_rehearsal_complete"
            if rehearsal
            else "gestational_sensorimotor_model_scored_complete"
        ),
        "revision": "GestationalSensorimotorModel",
        "rehearsal": bool(rehearsal),
        "infrastructure_only": bool(rehearsal),
        "do_not_tune_thresholds_from_rehearsal": bool(rehearsal),
        "decision_code": decision,
        "body": "NurseryBodyV2",
        "model_certification": cert_report,
        "ceiling": {
            "end_in_zone_rate": ceiling["end_in_zone_rate"],
            "distance_reduction": ceiling["distance_reduction"],
            "comfort_margin_over_random": ceiling["comfort_margin_over_random"],
            "exact_dynamics_not_organism_candidate": True,
        },
        "arms": arm_results,
        "interventions_executed": sorted(interventions.keys()),
        "uncertainty_fallback_exercised": interventions["uncertainty_fallback"][
            "fraction_fallback_actions"
        ]
        > 0.0,
        "model_off_fraction_model_actions": arm_results["learned_model_off_at_action_selection"][
            "fraction_model_actions_mean"
        ],
        "confirmation_seeds_registered": confirmation,
        "confirmation_consumed": confirmation_consumed,
        "confirmation_refusal_reason": confirmation_refusal_reason,
        "confirmation_results_present": confirmation_results is not None,
        "executing_head": executing_head,
        "implementation_sha": impl_sha,
        "runner_sha256": runner_sha,
        "thresholds": thresh,
        "uncertainty_max": thresh["uncertainty_max"],
        "fallback_policy": FALLBACK_POLICY,
        "protocol": "full_evidence_before_decision_ladder",
        "evidence_complete_before_ladder": evidence_complete,
        "all_lives_finite": True,
        "cuda_utilization_end": cuda_utilization_sample(),
        "ledger": str(ledger),
    }
    _atomic_write_json(
        out / "run_completed.json", {"run_id": run_id, "summary": summary, "completed_at": time.time()}
    )
    _atomic_write_json(out / "search_summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", action="store_true")
    p.add_argument("--rehearsal", action="store_true")
    args = p.parse_args()
    if args.scored and args.rehearsal:
        raise SystemExit("Choose either --scored or --rehearsal, not both")
    if not args.scored and not args.rehearsal:
        raise SystemExit("Specify --scored or --rehearsal")

    out_root = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_model")
    try:
        summary = run_protocol(rehearsal=bool(args.rehearsal))
        print(
            "PROTOCOL_DONE",
            summary.get("decision_code"),
            "rehearsal=" + str(summary.get("rehearsal")),
            flush=True,
        )
    except Exception as exc:
        out_root.mkdir(parents=True, exist_ok=True)
        fail_dir = out_root / (
            REHEARSAL_RUN_ID if args.rehearsal else "exos_dev1_gestational_sensorimotor_model_scored_20260820"
        )
        fail_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "revision": "GestationalSensorimotorModel",
            "rehearsal": bool(args.rehearsal),
            "failed_at": time.time(),
            "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "infrastructure_failure": True,
            "silent_replay_forbidden": True,
        }
        _atomic_write_json(fail_dir / "run_failed.json", payload)
        print("PROTOCOL_FAILED", exc, flush=True)
        raise


if __name__ == "__main__":
    main()
