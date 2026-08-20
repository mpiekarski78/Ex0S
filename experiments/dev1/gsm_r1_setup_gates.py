"""
GSM-R1 setup/reference gates (protocol revision).

Hard gates: analytic reachability, exact-dynamics beam planner, same-observation
supervised policy, observation-alias absence, and GSM model certification.
Recurrent AC is reported as a diagnostic only — never a first-match gate.
"""

from __future__ import annotations

from typing import Any

import torch

from experiments.dev1.developmental_birth_r4_r2_ceiling import evaluate_ceiling_gate_bundle
from experiments.dev1.gsm_model_certification import run_model_certification
from experiments.dev1.nursery_body_v2_certification import run_nursery_v2_certification
from three_memory.dev1.development.generative_genome import GenerativeGenome


SETUP_HARD_CHECKS = (
    "analytic_reachability",
    "exact_dynamics_beam_planner",
    "same_observation_supervised",
    "observation_aliases_absent_or_handled",
)


def evaluate_setup_reference(
    world_seed: str,
    *,
    device: torch.device | None = None,
    n_episodes: int = 32,
    episode_ticks: int = 16,
) -> dict[str, Any]:
    """Deterministic/reference setup gates on a scored discovery world."""
    nursery = run_nursery_v2_certification(
        world_seed + ":setup",
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=device,
    )
    hard = {k: bool(nursery["certification_checks"][k]) for k in SETUP_HARD_CHECKS}
    # AC diagnostic only (same frozen battery as R4-R2 / GSM-0, never a hard gate).
    ac_bundle = evaluate_ceiling_gate_bundle(
        GenerativeGenome.small(0),
        world_seed + ":ac_diagnostic",
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=device,
    )
    setup_pass = all(hard.values())
    return {
        "world_seed": world_seed,
        "setup_reference_pass": setup_pass,
        "hard_checks": hard,
        "nursery_certification_checks": nursery["certification_checks"],
        "reachability": nursery["reachability"],
        "exact_dynamics_beam_planner": nursery["references"]["exact_dynamics_beam_planner"],
        "same_observation_supervised": nursery["references"][
            "same_observation_supervised_controller"
        ],
        "observation_aliases": nursery["references"]["observation_collision_analysis"],
        "random_policy": nursery["references"]["random_policy"],
        "recurrent_ac_diagnostic_only": {
            "end_in_zone_rate": ac_bundle["end_in_zone_rate"],
            "distance_reduction": ac_bundle["distance_reduction"],
            "comfort_margin_over_random": ac_bundle["comfort_margin_over_random"],
            "n_initializations_run": ac_bundle["n_initializations_run"],
            "not_a_hard_gate": True,
            "first_match_forbidden": True,
        },
        "body": "NurseryBodyV2",
        "protocol": "gsm_r1_deterministic_reference_gates",
    }


def evaluate_model_certification_battery(
    seeds: list[str],
    *,
    device: torch.device | None = None,
    n_episodes: int = 24,
    episode_ticks: int = 16,
    epochs: int = 40,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        c = run_model_certification(
            seed,
            device=device,
            n_episodes=n_episodes,
            episode_ticks=episode_ticks,
            epochs=epochs,
        )
        rows.append(c)
    return {
        "seeds": [c["world_seed"] for c in rows],
        "all_certified": all(c["certified"] for c in rows),
        "rows": [
            {
                "world_seed": c["world_seed"],
                "certified": c["certified"],
                "checks": c["certification_checks"],
            }
            for c in rows
        ],
    }
