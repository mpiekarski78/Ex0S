"""
GSM-R1 scored harness — protocol-only revision.

Hard setup gates are deterministic/reference (reachability, beam planner,
same-observation supervised, alias absence) plus GSM model certification.
Recurrent AC is diagnostic only. Mechanism package SHAs unchanged from GSM-0.
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

from experiments.dev1.gsm_life import (
    DEFAULT_UNCERTAINTY_MAX,
    FALLBACK_POLICY,
    evaluate_gsm_life,
)
from experiments.dev1.gsm_r1_setup_gates import (
    evaluate_model_certification_battery,
    evaluate_setup_reference,
)
from experiments.dev1.search_r2 import _append_jsonl, _atomic_write_json, _rng_state_snapshot
from three_memory.dev1.device import cuda_utilization_sample, dev1_device

PREREG_PATH = Path("docs/exos_dev1.stage_a_gestational_sensorimotor_model_r1.prereg.lock")
RUNNER_FILE = "experiments/dev1/search_gestational_sensorimotor_model_r1.py"
NURSERY_FREEZE = "docs/exos_dev1.stage_a_nursery_body_v2.freeze.lock"
MECHANISM_SHA = "8129ccff11177263159f3d76342317288886c481"

REQUIRED_ARMS = (
    "sham_gestation",
    "existing_homeostatic_gestation",
    "predictive_gestation",
    "predictive_gestation_shuffled_consequences",
    "learned_model_off_at_action_selection",
)

SEALED_R4R2_PREFIX = "exos_dev1_developmental_birth_r4_r2_"

REHEARSAL_SEED = "gestational_sensorimotor_model_r1_excluded_rehearsal_20260821"
REHEARSAL_RUN_ID = "exos_dev1_gestational_sensorimotor_model_r1_excluded_rehearsal_20260821"


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
    setup: dict[str, Any],
    cert: dict[str, Any],
    arms: dict[str, Any],
    interventions: dict[str, Any],
    thresh: dict[str, float],
    evidence_complete: bool,
) -> str:
    if not evidence_complete:
        return "evidence_truncated_forbidden"
    if not setup.get("setup_reference_pass"):
        return "setup_reference_fail"
    if not cert.get("all_certified"):
        return "model_certification_fail"

    pred = arms["predictive_gestation"]
    sham = arms["sham_gestation"]
    shuffled = arms["predictive_gestation_shuffled_consequences"]
    model_off = arms["learned_model_off_at_action_selection"]
    open_loop = interventions["open_loop"]
    valence_off = interventions["valence_off"]

    if float(pred["train_end_in_zone_mean"]) < thresh["predictive_min_train_end_in_zone"]:
        return "grounded_acquisition_fail"
    if float(pred["train_distance_reduction_mean"]) < thresh["predictive_min_distance_reduction"]:
        return "grounded_acquisition_fail"
    if float(pred["fraction_model_actions_mean"]) < thresh["min_fraction_model_actions"]:
        return "grounded_acquisition_fail"
    if bool(pred.get("systematic_misprediction_risk")):
        return "grounded_acquisition_fail"

    eps = thresh["active_beats_control_epsilon"]
    if float(pred["train_end_in_zone_mean"]) <= float(sham["train_end_in_zone_mean"]) + eps:
        return "gsm_not_causal"
    if float(pred["train_end_in_zone_mean"]) <= float(shuffled["train_end_in_zone_mean"]) + eps:
        return "gsm_not_causal"
    if float(pred["train_end_in_zone_mean"]) <= float(model_off["train_end_in_zone_mean"]) + eps:
        return "gsm_not_causal"
    if float(pred["train_end_in_zone_mean"]) <= float(valence_off["end_in_zone_rate"]) + eps:
        return "gsm_not_causal"

    if float(pred["train_end_in_zone_mean"]) <= float(open_loop["end_in_zone_rate"]) + eps:
        return "closed_loop_not_causal"

    if float(pred["validation_end_in_zone_mean"]) < thresh["fresh_world_min_validation_end_in_zone"]:
        return "fresh_world_fail"

    return "gsm_validation_pass"


def _budget_from_prereg(prereg: dict[str, Any], *, rehearsal: bool) -> dict[str, int]:
    b = prereg["budget"]
    if rehearsal:
        return {
            "gestation_ticks": 24,
            "n_episodes_per_life": 4,
            "episode_ticks": 8,
            "cert_episodes": 8,
            "cert_epochs": 15,
            "setup_episodes": 8,
            "discovery_seeds": 1,
            "validation_seeds": 1,
        }
    return {
        "gestation_ticks": int(b["gestation_ticks"]),
        "n_episodes_per_life": int(b["n_episodes_per_life"]),
        "episode_ticks": int(b["episode_ticks"]),
        "cert_episodes": 24,
        "cert_epochs": 40,
        "setup_episodes": int(b["n_episodes_per_life"]),
        "discovery_seeds": int(b["discovery_seeds"]),
        "validation_seeds": int(b["validation_seeds"]),
    }


def _thresholds(prereg: dict[str, Any]) -> dict[str, float]:
    t = prereg["thresholds"]
    return {
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
        base = s.split(":", 1)[0]
        if base.startswith("exos_dev1_developmental_birth_r4_r2_"):
            raise RuntimeError(f"sealed R4-R2 seed forbidden: {base}")
        if base.startswith("exos_dev1_gestational_sensorimotor_model_world_"):
            raise RuntimeError(f"prior GSM scored world sealed: {base}")
        if base.startswith("exos_dev1_gestational_sensorimotor_model_conf_"):
            raise RuntimeError(f"prior GSM confirmation sealed: {base}")
        if (base in excl) and (not allow_excluded):
            raise RuntimeError(f"excluded seed used in scored path: {base}")
        if allow_excluded and base.startswith("exos_dev1_gestational_sensorimotor_model_r1_"):
            raise RuntimeError(f"scored R1 partition used in rehearsal: {base}")


def run_protocol(*, rehearsal: bool = False) -> dict[str, Any]:
    prereg = _load_prereg()
    if (not rehearsal) and (not prereg.get("scored_run_authorized")):
        raise SystemExit(
            "GSM-R1 scored run not authorized. Prereg scored_run_authorized=false."
        )

    uncertainty = prereg["uncertainty_fallback"]
    assert float(uncertainty["uncertainty_max"]) == float(DEFAULT_UNCERTAINTY_MAX)
    assert uncertainty["fallback_policy"] == FALLBACK_POLICY
    assert prereg["provenance"]["mechanism_sha"] == MECHANISM_SHA

    thresh = _thresholds(prereg)
    budget = _budget_from_prereg(prereg, rehearsal=rehearsal)

    if rehearsal:
        run_id = REHEARSAL_RUN_ID
        out = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_model_r1") / run_id
        discovery = [f"{REHEARSAL_SEED}:disc"]
        validation = [f"{REHEARSAL_SEED}:val"]
        confirmation = [f"{REHEARSAL_SEED}:conf_{i:03d}" for i in range(1, 5)]
        cert_seeds = [f"{REHEARSAL_SEED}:cert"]
        allow_excluded = True
    else:
        run_id = prereg["run_identity"]["run_id"]
        out = Path(prereg["run_identity"]["output_dir"])
        discovery = list(prereg["seed_partitions"]["discovery_world_seeds"])
        validation = list(prereg["seed_partitions"]["validation_world_seeds"])
        confirmation = list(prereg["seed_partitions"]["confirmation_seeds"])
        cert_seeds = list(prereg["seed_partitions"]["engineering_cert_seeds"])
        allow_excluded = False
        assert len(discovery) == budget["discovery_seeds"]
        assert len(validation) == budget["validation_seeds"]

    _assert_seed_policy(prereg, discovery + validation + confirmation, allow_excluded=allow_excluded)

    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "candidate_life_records.jsonl"
    if ledger.exists():
        ledger.unlink()

    executing_head = _git_head()
    runner_sha = _file_sha(RUNNER_FILE)
    require_cuda = bool(prereg["budget"].get("scored_run_cuda_required", True)) and (not rehearsal)
    if rehearsal:
        require_cuda = True
    dev = dev1_device(require_cuda=require_cuda)

    _atomic_write_json(
        out / "run_started.json",
        {
            "run_id": run_id,
            "revision": "GestationalSensorimotorModelR1",
            "protocol_only_revision": True,
            "mechanism_unchanged": True,
            "mechanism_sha": MECHANISM_SHA,
            "prior_gsm_attempt_descriptive_only": True,
            "rehearsal": bool(rehearsal),
            "executing_head": executing_head,
            "runner_file": RUNNER_FILE,
            "runner_sha256": runner_sha,
            "nursery_freeze": NURSERY_FREEZE,
            "body": "NurseryBodyV2",
            "discovery_seeds": discovery,
            "validation_seeds": validation,
            "confirmation_seeds": confirmation,
            "confirmation_sealed_until_validation_pass": True,
            "ac_is_diagnostic_not_hard_gate": True,
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

    # Phase 1: setup reference (no AC hard gate)
    setup = evaluate_setup_reference(
        discovery[0],
        device=dev,
        n_episodes=budget["setup_episodes"],
        episode_ticks=budget["episode_ticks"],
    )
    _atomic_write_json(out / "setup_reference.json", setup)
    _append_jsonl(
        ledger,
        {
            "run_id": run_id,
            "phase": "setup_reference",
            "setup_reference_pass": setup["setup_reference_pass"],
            "hard_checks": setup["hard_checks"],
            "ac_diagnostic_end_in_zone": setup["recurrent_ac_diagnostic_only"]["end_in_zone_rate"],
        },
    )
    print("SETUP_DONE", setup["setup_reference_pass"], flush=True)

    # Phase 2: GSM model certification
    cert = evaluate_model_certification_battery(
        cert_seeds,
        device=dev,
        n_episodes=budget["cert_episodes"],
        episode_ticks=budget["episode_ticks"],
        epochs=budget["cert_epochs"],
    )
    _atomic_write_json(out / "model_certification.json", cert)
    for row in cert["rows"]:
        _append_jsonl(
            ledger,
            {
                "run_id": run_id,
                "phase": "model_certification",
                "world_seed": row["world_seed"],
                "certified": row["certified"],
                "checks": row["checks"],
            },
        )
        print("CERT_DONE", row["world_seed"], row["certified"], flush=True)

    # Phase 3: behavioral arms (full evidence; no early ladder)
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

    # Phase 4: required causal interventions
    interventions: dict[str, Any] = {}
    for name, kwargs in (
        ("open_loop", {"open_loop": True}),
        ("uncertainty_fallback", {"force_uncertainty_fallback": True}),
        ("valence_off", {"valence_off": True}),
    ):
        m = evaluate_gsm_life(
            "predictive_gestation",
            f"{discovery[0]}:interv:{name}",
            n_episodes=budget["n_episodes_per_life"],
            episode_ticks=budget["episode_ticks"],
            gestation_ticks=budget["gestation_ticks"],
            uncertainty_max=thresh["uncertainty_max"],
            embryonic_seed=0,
            body_seed=1,
            life_rng_seed=hash(name) % 10_000,
            device=dev,
            **kwargs,
        )
        interventions[name] = _life_dict(m)
        _append_jsonl(
            ledger, {"run_id": run_id, "phase": "intervention", "name": name, **_life_dict(m)}
        )
    if interventions["uncertainty_fallback"]["fraction_model_actions"] != 0.0:
        raise RuntimeError("uncertainty fallback still used model actions")
    if interventions["uncertainty_fallback"]["fraction_fallback_actions"] <= 0.0:
        raise RuntimeError("uncertainty fallback produced no fallback actions")
    interventions["exact_body_dynamics_model_measurement_ceiling_only"] = {
        "end_in_zone_rate": float(setup["exact_dynamics_beam_planner"]["end_in_zone_rate"]),
        "distance_reduction": float(setup["exact_dynamics_beam_planner"]["distance_reduction"]),
        "not_organism_candidate": True,
        "from_setup_reference": True,
    }
    _atomic_write_json(out / "interventions.json", interventions)
    print("INTERVENTIONS_DONE", flush=True)

    decision = apply_decision_ladder(
        setup=setup,
        cert=cert,
        arms=arm_results,
        interventions=interventions,
        thresh=thresh,
        evidence_complete=True,
    )

    confirmation_consumed = False
    confirmation_results = None
    confirmation_refusal_reason = None
    if rehearsal:
        confirmation_refusal_reason = "rehearsal_confirmation_sealed"
    elif decision != "gsm_validation_pass":
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
            decision = "gsm_confirmation_pass"
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
            "gestational_sensorimotor_model_r1_rehearsal_complete"
            if rehearsal
            else "gestational_sensorimotor_model_r1_scored_complete"
        ),
        "revision": "GestationalSensorimotorModelR1",
        "protocol_only_revision": True,
        "mechanism_sha": MECHANISM_SHA,
        "prior_gsm_attempt_descriptive_only": True,
        "rehearsal": bool(rehearsal),
        "decision_code": decision,
        "body": "NurseryBodyV2",
        "setup_reference": {
            "pass": setup["setup_reference_pass"],
            "hard_checks": setup["hard_checks"],
            "ac_diagnostic_only": setup["recurrent_ac_diagnostic_only"],
        },
        "model_certification": cert,
        "arms": arm_results,
        "interventions_executed": sorted(interventions.keys()),
        "uncertainty_fallback_exercised": interventions["uncertainty_fallback"][
            "fraction_fallback_actions"
        ]
        > 0.0,
        "valence_off_executed": True,
        "confirmation_seeds_registered": confirmation,
        "confirmation_consumed": confirmation_consumed,
        "confirmation_refusal_reason": confirmation_refusal_reason,
        "executing_head": executing_head,
        "runner_sha256": runner_sha,
        "thresholds": thresh,
        "protocol": "full_evidence_before_decision_ladder",
        "evidence_complete_before_ladder": True,
        "ac_hard_gate_retired": True,
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
    if args.scored == args.rehearsal:
        raise SystemExit("Specify exactly one of --scored or --rehearsal")
    out_root = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_model_r1")
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
            REHEARSAL_RUN_ID if args.rehearsal else "exos_dev1_gestational_sensorimotor_model_r1_scored_20260821"
        )
        fail_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            fail_dir / "run_failed.json",
            {
                "revision": "GestationalSensorimotorModelR1",
                "rehearsal": bool(args.rehearsal),
                "failed_at": time.time(),
                "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "infrastructure_failure": True,
                "silent_replay_forbidden": True,
            },
        )
        print("PROTOCOL_FAILED", exc, flush=True)
        raise


if __name__ == "__main__":
    main()
