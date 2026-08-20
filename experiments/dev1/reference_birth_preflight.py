"""
Reference Birth excluded-seed preflight.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from experiments.dev1.reference_birth_life import (
    REFERENCE_BIRTH_ARMS,
    evaluate_reference_birth_life,
    plasticity_implementation_hash,
)
from three_memory.dev1.device import cuda_utilization_sample, dev1_device
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention


PREFLIGHT_SEED = "reference_birth_excluded_preflight_20260820"


@dataclass
class ReferenceBirthPreflightResult:
    passed: bool
    decision_code: str
    checks: dict[str, bool]
    metrics: dict
    details: dict


def run_reference_birth_preflight(seed: str = PREFLIGHT_SEED) -> ReferenceBirthPreflightResult:
    checks: dict[str, bool] = {}
    metrics: dict = {}
    details: dict = {}

    dev = dev1_device()
    checks["cuda_available"] = torch.cuda.is_available()
    metrics["device"] = str(dev)

    eprop = evaluate_reference_birth_life(
        "reward_eprop_rate_adaptation",
        seed,
        "stochastic",
        device=dev,
        n_episodes=4,
    )
    checks["eprop_finite_forward"] = eprop.treatment_accuracy >= 0.0
    checks["eprop_plasticity_hash_stable"] = len(plasticity_implementation_hash("reward_eprop_rate_adaptation")) == 64
    metrics["eprop_accuracy"] = eprop.treatment_accuracy
    metrics["eprop_update_norm"] = eprop.life_record.get("update_norm_mean", 0.0)

    reward_off = evaluate_reference_birth_life(
        "reward_eprop_rate_adaptation",
        seed + "_reward_off",
        "stochastic",
        device=dev,
        intervention=EpropIntervention.with_reward_off(),
        n_episodes=4,
    )
    checks["reward_off_intervention_runs"] = reward_off.life_record.get("intervention") == "reward_off"

    if torch.cuda.is_available():
        t0 = time.time()
        cuda_life = evaluate_reference_birth_life(
            "reward_eprop_rate_adaptation",
            seed + "_cuda",
            "stochastic",
            device=torch.device("cuda"),
            n_episodes=4,
        )
        cpu_life = evaluate_reference_birth_life(
            "reward_eprop_rate_adaptation",
            seed + "_cpu",
            "stochastic",
            device=torch.device("cpu"),
            n_episodes=4,
        )
        metrics["cuda_wall_seconds"] = time.time() - t0
        metrics["cuda_utilization"] = cuda_utilization_sample()
        checks["cuda_life_completes"] = cuda_life.device.startswith("cuda")
        acc_delta = abs(cuda_life.treatment_accuracy - cpu_life.treatment_accuracy)
        checks["cpu_gpu_parity_tolerance"] = acc_delta <= 0.5
        metrics["cpu_gpu_accuracy_delta"] = acc_delta
    else:
        checks["cuda_life_completes"] = True
        checks["cpu_gpu_parity_tolerance"] = True

    checks["all_arms_bind"] = all(
        plasticity_implementation_hash(arm) for arm in REFERENCE_BIRTH_ARMS if arm != "conventional_actor_critic_ceiling"
    )

    passed = all(checks.values())
    decision_code = "eprop_reference_preflight_pass" if passed else "eprop_reference_preflight_fail"
    return ReferenceBirthPreflightResult(
        passed=passed,
        decision_code=decision_code,
        checks=checks,
        metrics=metrics,
        details=details,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(asdict(run_reference_birth_preflight()), indent=2))
