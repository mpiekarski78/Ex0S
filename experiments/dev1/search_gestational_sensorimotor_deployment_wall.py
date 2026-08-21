"""
GSM deployment-wall scored harness.

Decisive online interface wall after GSM-R1 close.
Prereg remains scored_run_authorized=false; scored execution requires a separate
scored_authorization.lock. Runner SHA is pinned in a separate runner_pin.lock.
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

from experiments.dev1.gsm_deployment_wall_controllers import (
    HORIZON_TICKS,
    UNCERTAINTY_MAX_PIN,
    controller_invariants,
)
from experiments.dev1.gsm_deployment_wall_life import (
    REQUIRED_ARMS,
    TELEMETRY_FIELDS,
    evaluate_deployment_wall_life,
    paired_life_rng_seed,
)
from experiments.dev1.gsm_life import DEFAULT_UNCERTAINTY_MAX, FALLBACK_POLICY
from experiments.dev1.gsm_r1_setup_gates import (
    evaluate_model_certification_battery,
    evaluate_setup_reference,
)
from experiments.dev1.search_r2 import _append_jsonl, _atomic_write_json, _rng_state_snapshot
from three_memory.dev1.device import cuda_utilization_sample, dev1_device

PREREG_PATH = Path("docs/exos_dev1.stage_a_gestational_sensorimotor_deployment_wall.prereg.lock")
AUTH_PATH = Path(
    "docs/exos_dev1.stage_a_gestational_sensorimotor_deployment_wall.scored_authorization.lock"
)
RUNNER_PIN_PATH = Path(
    "docs/exos_dev1.stage_a_gestational_sensorimotor_deployment_wall.runner_pin.lock"
)
RUNNER_FILE = "experiments/dev1/search_gestational_sensorimotor_deployment_wall.py"
NURSERY_FREEZE = "docs/exos_dev1.stage_a_nursery_body_v2.freeze.lock"
MECHANISM_SHA = "8129ccff11177263159f3d76342317288886c481"
R1_CLOSE = "docs/exos_dev1.stage_a_gestational_sensorimotor_model_r1.close.lock"

REHEARSAL_SEED = "gestational_sensorimotor_deployment_wall_excluded_rehearsal_20260821"
REHEARSAL_RUN_ID = "exos_dev1_gestational_sensorimotor_deployment_wall_excluded_rehearsal_20260821"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _assert_mechanism_immutable() -> None:
    diff = subprocess.check_output(
        ["git", "diff", "--stat", "8129ccf", "--", "three_memory/dev1/gsm/"], text=True
    ).strip()
    if diff:
        raise RuntimeError(f"GSM mechanism package drifted from 8129ccf: {diff}")


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
        "organism_candidate": m.organism_candidate,
        "measurement_only": m.measurement_only,
        "life_record": m.life_record,
    }


def _mean_field(lives: list[dict[str, Any]], key: str) -> float:
    return sum(float(x[key]) for x in lives) / max(1, len(lives))


def _behavioral_pass(row: dict[str, Any], thresh: dict[str, float], *, require_model_frac: bool) -> bool:
    if float(row["train_end_in_zone_mean"]) < thresh["predictive_min_train_end_in_zone"]:
        return False
    if float(row["train_distance_reduction_mean"]) < thresh["predictive_min_distance_reduction"]:
        return False
    if require_model_frac:
        if float(row["fraction_model_actions_mean"]) < thresh["min_fraction_model_actions"]:
            return False
    if bool(row.get("systematic_misprediction_risk")):
        return False
    return True


def apply_decision_ladder(
    *,
    setup: dict[str, Any],
    cert: dict[str, Any],
    arms: dict[str, Any],
    thresh: dict[str, float],
    evidence_complete: bool,
) -> str:
    if not evidence_complete:
        return "evidence_truncated_forbidden"
    if not setup.get("setup_reference_pass"):
        return "setup_reference_fail"
    if not cert.get("all_certified"):
        return "model_certification_fail"

    exact1 = arms["exact_one_step_valence"]
    exact_h = arms["exact_receding_horizon"]
    forced = arms["learned_forced"]
    gated = arms["learned_gated"]
    random_fb = arms["random_fallback"]

    exact1_ok = _behavioral_pass(exact1, thresh, require_model_frac=False)
    exact_h_ok = _behavioral_pass(exact_h, thresh, require_model_frac=False)
    forced_ok = _behavioral_pass(forced, thresh, require_model_frac=False)
    # Forced should nearly always use model actions; still record but do not require 0.5
    # if require_trusted=false always selects.
    gated_ok = _behavioral_pass(gated, thresh, require_model_frac=True)

    if (not exact1_ok) and exact_h_ok:
        return "missing_value_or_horizon_machinery"
    if exact1_ok and (not forced_ok):
        return "learned_representation_does_not_transfer_online"
    if forced_ok and (not gated_ok):
        return "uncertainty_calibration_or_coverage_failure"

    eps = thresh["active_beats_control_epsilon"]
    if gated_ok:
        if float(gated["train_end_in_zone_mean"]) <= float(random_fb["train_end_in_zone_mean"]) + eps:
            return "model_not_causally_useful"
        if float(gated["validation_end_in_zone_mean"]) < thresh["fresh_world_min_validation_end_in_zone"]:
            return "fresh_world_fail"
        return "deployment_wall_pass"

    return "deployment_wall_fail"


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
    assert abs(float(t["uncertainty_max"]) - float(UNCERTAINTY_MAX_PIN)) < 1e-12
    assert int(t["receding_horizon_ticks"]) == int(HORIZON_TICKS)
    return {
        "predictive_min_train_end_in_zone": float(t["predictive_min_train_end_in_zone"]),
        "predictive_min_distance_reduction": float(t["predictive_min_distance_reduction"]),
        "active_beats_control_epsilon": float(t["active_beats_control_epsilon"]),
        "fresh_world_min_validation_end_in_zone": float(
            t["fresh_world_min_validation_end_in_zone"]
        ),
        "min_fraction_model_actions": float(t["min_fraction_model_actions"]),
        "uncertainty_max": float(t["uncertainty_max"]),
        "receding_horizon_ticks": float(t["receding_horizon_ticks"]),
    }


def _assert_seed_policy(prereg: dict[str, Any], seeds: list[str], *, allow_excluded: bool) -> None:
    excl = set(prereg["seed_partitions"]["excluded_seeds"])
    for s in seeds:
        base = s.split(":", 1)[0]
        if base.startswith("exos_dev1_developmental_birth_r4_r2_"):
            raise RuntimeError(f"sealed R4-R2 seed forbidden: {base}")
        if base.startswith("exos_dev1_gestational_sensorimotor_model_r1_"):
            raise RuntimeError(f"sealed GSM-R1 partition forbidden: {base}")
        if base.startswith("exos_dev1_gestational_sensorimotor_model_world_"):
            raise RuntimeError(f"prior GSM scored world sealed: {base}")
        if base.startswith("exos_dev1_gestational_sensorimotor_model_conf_"):
            raise RuntimeError(f"prior GSM confirmation sealed: {base}")
        if (base in excl) and (not allow_excluded):
            raise RuntimeError(f"excluded seed used in scored path: {base}")
        if allow_excluded and base.startswith("exos_dev1_gestational_sensorimotor_deployment_wall_"):
            raise RuntimeError(f"scored DW partition used in rehearsal: {base}")


def _authorize_scored(prereg: dict[str, Any], runner_sha: str) -> dict[str, Any]:
    """Scored path: prereg stays unauthorized; separate auth lock required."""
    if prereg.get("scored_run_authorized"):
        raise RuntimeError("frozen prereg must remain scored_run_authorized=false; use auth lock")
    if not AUTH_PATH.exists():
        raise SystemExit(
            "GSM deployment-wall scored run not authorized. "
            "Missing scored_authorization.lock (prereg stays false)."
        )
    auth = _load_json(AUTH_PATH)
    if not auth.get("authorized"):
        raise SystemExit("scored_authorization.lock present but authorized=false")
    if auth.get("prereg_remains_scored_run_authorized_false") is not True:
        raise RuntimeError("auth lock must keep prereg scored_run_authorized=false")
    if auth.get("runner_sha256") != runner_sha:
        raise RuntimeError(
            f"auth runner_sha256 mismatch: lock={auth.get('runner_sha256')} live={runner_sha}"
        )
    if RUNNER_PIN_PATH.exists():
        pin = _load_json(RUNNER_PIN_PATH)
        if pin.get("runner_sha256") != runner_sha:
            raise RuntimeError("runner_pin.lock SHA does not match live runner")
    return auth


def run_protocol(*, rehearsal: bool = False) -> dict[str, Any]:
    prereg = _load_json(PREREG_PATH)
    assert prereg["provenance"]["mechanism_sha"] == MECHANISM_SHA
    assert float(prereg["uncertainty_fallback"]["uncertainty_max"]) == float(DEFAULT_UNCERTAINTY_MAX)
    assert prereg["uncertainty_fallback"]["fallback_policy"] == FALLBACK_POLICY
    _assert_mechanism_immutable()

    thresh = _thresholds(prereg)
    budget = _budget_from_prereg(prereg, rehearsal=rehearsal)
    executing_head = _git_head()
    runner_sha = _file_sha(RUNNER_FILE)

    auth = None
    if rehearsal:
        run_id = REHEARSAL_RUN_ID
        out = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_deployment_wall") / run_id
        discovery = [f"{REHEARSAL_SEED}:disc"]
        validation = [f"{REHEARSAL_SEED}:val"]
        confirmation = [f"{REHEARSAL_SEED}:conf_{i:03d}" for i in range(1, 5)]
        cert_seeds = [f"{REHEARSAL_SEED}:cert"]
        allow_excluded = True
    else:
        auth = _authorize_scored(prereg, runner_sha)
        run_id = auth["run_id"]
        out = Path(auth["output_dir"])
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

    require_cuda = bool(prereg["budget"].get("scored_run_cuda_required", True))
    if rehearsal:
        require_cuda = True
    dev = dev1_device(require_cuda=require_cuda)

    _atomic_write_json(
        out / "run_started.json",
        {
            "run_id": run_id,
            "revision": "GestationalSensorimotorDeploymentWall",
            "mechanism_sha": MECHANISM_SHA,
            "r1_close": R1_CLOSE,
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
            "horizon_ticks": HORIZON_TICKS,
            "uncertainty_max": UNCERTAINTY_MAX_PIN,
            "paired_randomness": True,
            "controller_invariants": controller_invariants(),
            "telemetry_fields": list(TELEMETRY_FIELDS),
            "required_arms": list(REQUIRED_ARMS),
            "started_at": time.time(),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(dev),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "rng": _rng_state_snapshot(),
            "cuda_utilization": cuda_utilization_sample(),
            "protocol": "full_evidence_before_decision_ladder",
            "budget": budget,
            "authorization_lock": str(AUTH_PATH) if auth else None,
            "prereg_scored_run_authorized": False,
        },
    )

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
        },
    )
    print("SETUP_DONE", setup["setup_reference_pass"], flush=True)

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

    arm_results: dict[str, Any] = {}
    matched_pre_hashes: dict[str, set[str]] = {}
    matched_world_hashes: dict[str, set[str]] = {}
    for arm in REQUIRED_ARMS:
        train_lives = []
        for s in discovery:
            m = evaluate_deployment_wall_life(
                arm,
                f"{s}:arm:{arm}",
                n_episodes=budget["n_episodes_per_life"],
                episode_ticks=budget["episode_ticks"],
                gestation_ticks=budget["gestation_ticks"],
                uncertainty_max=thresh["uncertainty_max"],
                horizon_ticks=int(thresh["receding_horizon_ticks"]),
                embryonic_seed=0,
                body_seed=1,
                life_rng_seed=paired_life_rng_seed(s),
                device=dev,
            )
            d = _life_dict(m)
            train_lives.append(d)
            _append_jsonl(ledger, {"run_id": run_id, "phase": "arm_train", **d})
            matched_pre_hashes.setdefault(s, set()).add(d["pre_gestation_checkpoint_hash"])
            matched_world_hashes.setdefault(s, set()).add(d["life_record"]["world_hash"])
        val_lives = []
        for s in validation:
            m = evaluate_deployment_wall_life(
                arm,
                f"{s}:arm:{arm}",
                n_episodes=budget["n_episodes_per_life"],
                episode_ticks=budget["episode_ticks"],
                gestation_ticks=budget["gestation_ticks"],
                uncertainty_max=thresh["uncertainty_max"],
                horizon_ticks=int(thresh["receding_horizon_ticks"]),
                embryonic_seed=0,
                body_seed=1,
                life_rng_seed=paired_life_rng_seed(s + ":val"),
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
            "organism_candidate": train_lives[0]["organism_candidate"],
            "measurement_only": train_lives[0]["measurement_only"],
            "action_source_counts": train_lives[0]["life_record"].get("action_source_counts", {}),
            "pre_gestation_hashes": sorted(
                {x["pre_gestation_checkpoint_hash"] for x in train_lives + val_lives}
            ),
            "paired_life_rng_seeds": sorted(
                {x["life_record"]["paired_life_rng_seed"] for x in train_lives}
            ),
            "telemetry_sample": {
                k: train_lives[0]["life_record"].get(k) for k in TELEMETRY_FIELDS if k != "synergy_histograms"
            },
            "synergy_histogram": train_lives[0]["life_record"].get("synergy_histogram"),
        }
        print("ARM_DONE", arm, flush=True)
    _atomic_write_json(out / "arms.json", arm_results)

    # Matched checkpoints: learned arms share pre-hash; all arms share embryonic/body seeds.
    learned_pres = set(arm_results["learned_gated"]["pre_gestation_hashes"]) | set(
        arm_results["learned_forced"]["pre_gestation_hashes"]
    )
    if len(learned_pres) != 1:
        raise RuntimeError(f"learned arms must share matched pre-gestation hash; got {learned_pres}")
    if arm_results["learned_gated"]["paired_life_rng_seeds"] != arm_results["learned_forced"][
        "paired_life_rng_seeds"
    ]:
        raise RuntimeError("learned arms must share paired life RNG seeds")

    decision = apply_decision_ladder(
        setup=setup,
        cert=cert,
        arms=arm_results,
        thresh=thresh,
        evidence_complete=True,
    )

    confirmation_consumed = False
    confirmation_results = None
    confirmation_refusal_reason = None
    if rehearsal:
        confirmation_refusal_reason = "rehearsal_confirmation_sealed"
    elif decision != "deployment_wall_pass":
        confirmation_refusal_reason = f"validation_did_not_pass:{decision}"
    else:
        # Fresh-world already checked inside ladder; confirmation only after pass.
        confirmation_results = {}
        for s in confirmation:
            m = evaluate_deployment_wall_life(
                "learned_gated",
                f"{s}:confirmation",
                n_episodes=budget["n_episodes_per_life"],
                episode_ticks=budget["episode_ticks"],
                gestation_ticks=budget["gestation_ticks"],
                uncertainty_max=thresh["uncertainty_max"],
                embryonic_seed=0,
                body_seed=1,
                life_rng_seed=paired_life_rng_seed(s),
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
            "gestational_sensorimotor_deployment_wall_rehearsal_complete"
            if rehearsal
            else "gestational_sensorimotor_deployment_wall_scored_complete"
        ),
        "revision": "GestationalSensorimotorDeploymentWall",
        "mechanism_sha": MECHANISM_SHA,
        "r1_close": R1_CLOSE,
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
        "horizon_ticks": HORIZON_TICKS,
        "uncertainty_max": UNCERTAINTY_MAX_PIN,
        "controller_invariants": controller_invariants(),
        "confirmation_seeds_registered": confirmation,
        "confirmation_consumed": confirmation_consumed,
        "confirmation_refusal_reason": confirmation_refusal_reason,
        "executing_head": executing_head,
        "runner_sha256": runner_sha,
        "thresholds": thresh,
        "protocol": "full_evidence_before_decision_ladder",
        "evidence_complete_before_ladder": True,
        "prereg_scored_run_authorized": False,
        "cuda_utilization_end": cuda_utilization_sample(),
        "ledger": str(ledger),
        "no_threshold_adjustment_from_rehearsal": True,
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
    out_root = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_deployment_wall")
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
            REHEARSAL_RUN_ID
            if args.rehearsal
            else "exos_dev1_gestational_sensorimotor_deployment_wall_scored_fail"
        )
        fail_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            fail_dir / "run_failed.json",
            {
                "revision": "GestationalSensorimotorDeploymentWall",
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
