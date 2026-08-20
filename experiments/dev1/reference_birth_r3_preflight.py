"""Reference Birth R3 excluded-seed preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from experiments.dev1.conventional_ac_ceiling import evaluate_ceiling_life
from experiments.dev1.reference_birth_r3_life import (
    evaluate_r3_life,
    inheritance_leakage_check,
    run_r3_causal_controls,
)
from experiments.dev1.reference_birth_r3_outer import default_lsg_surface, lsg_param_count, lsg_dims
from three_memory.dev1.device import cuda_utilization_sample, dev1_device
from three_memory.dev1.plasticity.eprop.signal_generator import InheritedLearningSignalGenerator


PREFLIGHT_SEED = "reference_birth_r3_excluded_preflight_20260820"


@dataclass
class R3PreflightResult:
    passed: bool
    decision_code: str
    checks: dict[str, bool]
    metrics: dict
    details: dict


def run_signal_generator_unit(seed: str = PREFLIGHT_SEED + "_lsg") -> dict:
    n_motor, n_post, _ = lsg_dims()
    vec = default_lsg_surface(seed=11)
    gen = InheritedLearningSignalGenerator(
        n_rel=128,
        n_motor=n_motor,
        n_post=n_post,
        param_vector=vec,
        seed=11,
        device=torch.device("cpu"),
    )
    rel = torch.randn(128)
    policy = torch.randn(n_motor)
    l1 = gen.learning_signal(rel, policy, 1.0)
    l_off = gen.learning_signal(rel, policy, 1.0, generator_off=True)
    l_perm = gen.learning_signal(rel, policy, 1.0, generator_permuted=True)
    return {
        "param_count": len(vec),
        "expected_param_count": lsg_param_count(n_motor, n_post),
        "signal_norm": float(l1.norm().item()),
        "off_norm": float(l_off.norm().item()),
        "perm_differs": not torch.allclose(l1, l_perm),
        "passed": (
            len(vec) == lsg_param_count(n_motor, n_post)
            and float(l1.norm().item()) > 0.0
            and float(l_off.norm().item()) == 0.0
            and not torch.allclose(l1, l_perm)
        ),
    }


def run_reference_birth_r3_preflight(seed: str = PREFLIGHT_SEED) -> R3PreflightResult:
    checks: dict[str, bool] = {}
    metrics: dict = {}
    details: dict = {}

    checks["cuda_available"] = torch.cuda.is_available()

    unit = run_signal_generator_unit(seed + "_unit")
    checks["signal_generator_unit"] = unit["passed"]
    details["signal_generator_unit"] = unit
    metrics["lsg_param_count"] = unit["param_count"]

    leak = inheritance_leakage_check(seed + "_leak")
    checks["inheritance_leakage"] = leak["passed"]
    details["inheritance_leakage"] = leak

    vec = default_lsg_surface(seed=3)
    # Sensitivity: nonzero updates under LSG
    life = evaluate_r3_life(
        "inherited_learning_signal_generator",
        seed,
        "stochastic",
        lsg_vector=vec,
        n_episodes=4,
        life_rng_seed=1,
        device=dev1_device(),
    )
    checks["update_effect_telemetry"] = life.life_record.get("n_credit_events", 0) > 0
    checks["lsg_frozen_within_life"] = bool(life.life_record.get("lsg_params_unchanged_within_life"))
    checks["separated_teacher_credit"] = (
        life.self_credit_event_count > 0 and life.teacher_credit_event_count > 0
    )
    checks["cuda_life"] = (not torch.cuda.is_available()) or life.device.startswith("cuda")
    metrics["update_norm_mean"] = life.update_norm_mean
    metrics["signed_margin_improvement"] = life.signed_margin_improvement
    details["candidate_life"] = {
        "accuracy": life.treatment_accuracy,
        "self_events": life.self_credit_event_count,
        "teacher_events": life.teacher_credit_event_count,
        "update_norm_mean": life.update_norm_mean,
    }
    checks["search_surface_sensitivity"] = life.update_norm_mean > 1e-8 or abs(
        life.signed_margin_improvement
    ) > 0.0

    controls = run_r3_causal_controls(
        vec,
        seed + "_ctrl",
        n_episodes=4,
        life_rng_seed=9,
        device=dev1_device(),
    )
    checks["signal_controls_run"] = set(controls["results"].keys()) >= {
        "none", "reward_off", "signal_generator_off", "signal_generator_permuted"
    }
    details["controls"] = controls

    baseline = evaluate_r3_life(
        "r2_fixed_eprop_baseline",
        seed + "_base",
        "stochastic",
        n_episodes=2,
        life_rng_seed=2,
        device=dev1_device(),
    )
    checks["baseline_runs"] = baseline.life_record.get("n_credit_events", 0) > 0
    details["baseline"] = {"accuracy": baseline.treatment_accuracy}

    ceiling = evaluate_ceiling_life(seed + "_ceiling", "stochastic", n_episodes=8, train_with_autograd=True)
    checks["ceiling_runs"] = ceiling.treatment_accuracy >= 0.0
    metrics["ceiling_accuracy"] = ceiling.treatment_accuracy
    metrics["cuda_utilization"] = cuda_utilization_sample()

    passed = all(checks.values())
    return R3PreflightResult(
        passed=passed,
        decision_code="eprop_reference_preflight_pass" if passed else "eprop_reference_preflight_fail",
        checks=checks,
        metrics=metrics,
        details=details,
    )


if __name__ == "__main__":
    import json
    print(json.dumps(asdict(run_reference_birth_r3_preflight()), indent=2, default=str))
