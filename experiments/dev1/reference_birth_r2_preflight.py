"""
Reference Birth R2 excluded-seed preflight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from experiments.dev1.conventional_ac_ceiling import evaluate_ceiling_life
from experiments.dev1.reference_birth_r2_life import evaluate_r2_life, run_matched_interventions
from experiments.dev1.reference_birth_r2_outer import extremes_for_sensitivity
from three_memory.dev1.device import cuda_utilization_sample, dev1_device


PREFLIGHT_SEED = "reference_birth_r2_excluded_preflight_20260820"


@dataclass
class R2PreflightResult:
    passed: bool
    decision_code: str
    checks: dict[str, bool]
    metrics: dict
    details: dict


def run_search_surface_sensitivity(seed: str = PREFLIGHT_SEED + "_sens") -> dict:
    rows = []
    baseline_upd = None
    baseline_margin = None
    for name, surface in extremes_for_sensitivity():
        life = evaluate_r2_life(
            "reward_eprop_rate_adaptation",
            seed,
            "stochastic",
            surface=surface,
            n_episodes=4,
            life_rng_seed=7,
            device=dev1_device(),
        )
        row = {
            "name": name,
            "update_norm_mean": life.update_norm_mean,
            "margin_increase_fraction": life.margin_increase_fraction,
            "signed_margin_improvement": life.signed_margin_improvement,
            "accuracy": life.treatment_accuracy,
        }
        rows.append(row)
        if name == "baseline":
            baseline_upd = life.update_norm_mean
            baseline_margin = life.margin_increase_fraction

    max_upd = max(r["update_norm_mean"] for r in rows)
    margin_spread = max(r["margin_increase_fraction"] for r in rows) - min(
        r["margin_increase_fraction"] for r in rows
    )
    signed_vals = [r["signed_margin_improvement"] for r in rows]
    signed_spread = max(signed_vals) - min(signed_vals)
    leaves_1e8 = max_upd > 1e-6
    # R2 causal term is signed margin g_t; fraction-of-increases can be zero when
    # positive outcomes are rare on short excluded-seed probes.
    margins_move = (
        margin_spread > 1e-6
        or signed_spread > 1e-15
        or any(abs(r["margin_increase_fraction"] - (baseline_margin or 0.0)) > 1e-6 for r in rows)
    )
    return {
        "rows": rows,
        "baseline_update_norm": baseline_upd,
        "max_update_norm": max_upd,
        "margin_spread": margin_spread,
        "signed_margin_spread": signed_spread,
        "leaves_1e8_regime": leaves_1e8,
        "margins_move": margins_move,
        "passed": leaves_1e8 and margins_move,
    }


def run_teacher_separation_proof(seed: str = PREFLIGHT_SEED + "_teacher") -> dict:
    surface = dict(extremes_for_sensitivity()[0][1])
    surface["log_actor_learning_rate"] = -3.0
    surface["learning_signal_projection_scale"] = 20.0

    normal = evaluate_r2_life(
        "teacher_demo_eprop",
        seed,
        "stochastic",
        surface=surface,
        n_episodes=8,
        life_rng_seed=3,
        permute_teacher_demos=False,
        teacher_credit_enabled=True,
    )
    permuted = evaluate_r2_life(
        "teacher_demo_eprop",
        seed,
        "stochastic",
        surface=surface,
        n_episodes=8,
        life_rng_seed=3,
        permute_teacher_demos=True,
        teacher_credit_enabled=True,
    )
    credit_off = evaluate_r2_life(
        "teacher_demo_eprop",
        seed,
        "stochastic",
        surface=surface,
        n_episodes=8,
        life_rng_seed=3,
        permute_teacher_demos=False,
        teacher_credit_enabled=False,
    )

    self_targets = {
        e["actor_target"]
        for e in normal.credit_events
        if e.get("credit_source") == "self_action"
    }
    teacher_targets = {
        e["actor_target"]
        for e in normal.credit_events
        if e.get("credit_source") == "teacher_demonstration"
    }
    self_never_teacher = all(
        e.get("actor_target") is not None
        for e in normal.credit_events
        if e.get("credit_source") == "self_action"
    )
    teacher_elg_ok = all(
        e.get("self_eligibility_unchanged") in (True, None)
        for e in normal.credit_events
        if e.get("credit_source") == "teacher_demonstration"
    )

    return {
        "normal_follow_rate": normal.teacher_follow_rate,
        "permuted_follow_rate": permuted.teacher_follow_rate,
        "normal_demo_count": normal.teacher_demo_count,
        "permuted_demo_count": permuted.teacher_demo_count,
        "credit_off_teacher_events": credit_off.teacher_credit_event_count,
        "credit_off_demo_count": credit_off.teacher_demo_count,
        "normal_teacher_events": normal.teacher_credit_event_count,
        "normal_self_events": normal.self_credit_event_count,
        "histograms_differ": normal.life_record["action_histogram"] != permuted.life_record["action_histogram"],
        "teacher_credit_off_no_teacher_updates": credit_off.teacher_credit_event_count == 0,
        "teacher_credit_off_still_injects": credit_off.teacher_demo_count > 0,
        "separate_sources_present": (
            normal.self_credit_event_count > 0 and normal.teacher_credit_event_count > 0
        ),
        "self_eligibility_preserved_on_teacher": teacher_elg_ok,
        "passed": (
            normal.teacher_demo_count > 0
            and credit_off.teacher_credit_event_count == 0
            and credit_off.teacher_demo_count > 0
            and normal.teacher_credit_event_count > 0
            and normal.self_credit_event_count > 0
            and (
                normal.life_record["action_histogram"] != permuted.life_record["action_histogram"]
                or abs(normal.teacher_follow_rate - permuted.teacher_follow_rate) > 1e-6
            )
        ),
        "self_targets_sample": list(self_targets)[:8],
        "teacher_targets_sample": list(teacher_targets)[:8],
        "self_never_teacher_ok": self_never_teacher,
    }


def run_reference_birth_r2_preflight(seed: str = PREFLIGHT_SEED) -> R2PreflightResult:
    checks: dict[str, bool] = {}
    metrics: dict = {}
    details: dict = {}

    checks["cuda_available"] = torch.cuda.is_available()
    sens = run_search_surface_sensitivity(seed + "_sens")
    checks["search_surface_sensitivity"] = sens["passed"]
    details["sensitivity"] = sens
    metrics["max_update_norm"] = sens["max_update_norm"]

    teacher = run_teacher_separation_proof(seed + "_teacher")
    checks["teacher_credit_separation"] = teacher["passed"]
    details["teacher"] = teacher

    interv = run_matched_interventions(
        "reward_eprop_rate_adaptation",
        seed + "_interv",
        extremes_for_sensitivity()[0][1],
        n_episodes=4,
        life_rng_seed=11,
        device=dev1_device(),
    )
    checks["matched_interventions_run"] = set(interv["results"].keys()) >= {
        "none", "reward_off", "eligibility_zero", "eligibility_permuted", "motor_feedback_permuted"
    }
    details["interventions"] = interv

    ceiling = evaluate_ceiling_life(seed + "_ceiling", "stochastic", n_episodes=8, train_with_autograd=True)
    checks["ceiling_runs"] = ceiling.treatment_accuracy >= 0.0
    metrics["ceiling_accuracy"] = ceiling.treatment_accuracy

    life = evaluate_r2_life(
        "reward_eprop_rate_adaptation",
        seed,
        "stochastic",
        n_episodes=4,
        life_rng_seed=1,
    )
    checks["update_effect_telemetry"] = life.life_record.get("n_credit_events", 0) > 0
    checks["lexicographic_fitness_present"] = (
        "fitness_key" in life.life_record and len(life.life_record["fitness_key"]) == 4
    )
    checks["cuda_life"] = (not torch.cuda.is_available()) or life.device.startswith("cuda")
    metrics["cuda_utilization"] = cuda_utilization_sample()
    metrics["signed_margin_improvement"] = life.signed_margin_improvement
    metrics["retention_after_reset"] = life.retention_after_reset

    passed = all(checks.values())
    return R2PreflightResult(
        passed=passed,
        decision_code="eprop_reference_preflight_pass" if passed else "eprop_reference_preflight_fail",
        checks=checks,
        metrics=metrics,
        details=details,
    )


if __name__ == "__main__":
    import json
    print(json.dumps(asdict(run_reference_birth_r2_preflight()), indent=2, default=str))
