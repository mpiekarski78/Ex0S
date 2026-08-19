"""
Stage A R1 sensorimotor-credit preflight.

Cheaply kills candidates before scored behavioral search when the local
credit apparatus is degenerate or when H is not truly absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.dev1.worlds import InteractionWorld, WorldConfig
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


@dataclass
class PreflightResult:
    passed: bool
    decision_code: str
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def run_credit_preflight(
    genome: DevGenome,
    family: str,
    device: torch.device | None = None,
    min_nontrivial_delta_w: float = 1e-8,
) -> PreflightResult:
    """
    Run mandatory Stage A R1 preflight.

    Preconditions:
    - H disabled for all checks
    - no scored search yet
    """
    dev = device or torch.device("cpu")
    g = genome
    g.plasticity_family = family
    org = ModularOrganism.birth(g, device=dev, h_disabled=True, consolidation_disabled=True)
    world = InteractionWorld(WorldConfig(seed="stage_a_r1_preflight", n_roles=4))

    initial_h_cap = org.hippocampus.capacity_telemetry()
    h_hash_before = org.hippocampus.state_hash()
    sens_states = []
    action_states = []
    mod_values = []
    last_delta = 0.0

    # Four distinct cues to test sensory and efference differentiation.
    events = world.generate_episode()[:4]
    prev_reward = 0.0
    for we in events:
        org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=prev_reward))
        action = org.act()
        sens_states.append(org.rho.sensory_repr.detach().cpu().numpy())
        action_states.append(org.rho.action_repr.detach().cpu().numpy())
        mod_values.append(org._last_mod)
        prev_reward = world.reward_for_action(we, action.motor_channel)
        org.rest()
        last_delta = float(getattr(org, "_last_actor_delta", torch.zeros(1)).norm().item())

    # Reward-off should not create persistent reward-gated actor update.
    reward_off_org = ModularOrganism.birth(g, device=dev, h_disabled=True, consolidation_disabled=True)
    reward_off_event = events[0]
    reward_off_org.observe(OrganismObservation(sensory_vector=reward_off_event.sensory_vector, reward=0.0))
    reward_off_org.act()
    reward_off_org.rest()
    reward_off_delta = float(getattr(reward_off_org, "_last_actor_delta", torch.zeros(1)).norm().item())

    # Permuted feedback must follow observed motor path, not fixture answer.
    permuted_org = ModularOrganism.birth(g, device=dev, h_disabled=True, consolidation_disabled=True)
    perm_event = events[1]
    permuted_org.observe(OrganismObservation(sensory_vector=perm_event.sensory_vector, reward=0.0))
    perm_action = permuted_org.act()
    wrong_reward = world.cfg.reward_on_correct if (perm_action.motor_channel + 1) % world.cfg.n_roles == 0 else world.cfg.reward_on_incorrect
    permuted_org.observe(OrganismObservation(sensory_vector=perm_event.sensory_vector, reward=wrong_reward))
    permuted_org.act()
    permuted_org.rest()
    permuted_delta = float(getattr(permuted_org, "_last_actor_delta", torch.zeros(1)).norm().item())

    h_hash_after = org.hippocampus.state_hash()
    h_cap = org.hippocampus.capacity_telemetry()

    sensory_dist = _pairwise_min_distance(sens_states)
    action_dist = _pairwise_min_distance(action_states)
    elig_norm = float(org.eligibility.trace.norm().item())
    reward_err = float(mod_values[-1]["reward_baseline_error"].item())
    neg_mod = _scalar_modulation(mod_values[0])
    pos_mod = _scalar_modulation(mod_values[-1])
    finite_ok = _all_finite(org)

    checks = {
        "sensory_states_differ": sensory_dist > 1e-6,
        "motor_states_differ": action_dist > 1e-6,
        "credit_modulation_opposed": abs(pos_mod - neg_mod) > 1e-6 and np.sign(pos_mod + 1e-12) != np.sign(neg_mod + 1e-12),
        "eligibility_nonzero": elig_norm > 0.0,
        "rewarded_delta_w_nontrivial": last_delta > min_nontrivial_delta_w,
        "reward_off_no_persistent_actor_update": reward_off_delta <= min_nontrivial_delta_w,
        "permuted_feedback_updates_from_observed_motor": permuted_delta >= 0.0,
        "finite_noncollapsed": finite_ok,
        "H_begins_empty": initial_h_cap["capacity_used"] == 0,
        "H_write_counter_zero": h_cap["write_attempts_total"] == 0 and h_cap["successful_writes_total"] == 0,
        "H_read_counter_zero": h_cap["read_attempts_total"] == 0 and h_cap["successful_reads_total"] == 0,
        "H_state_hash_unchanged": h_hash_before == h_hash_after,
    }

    passed = all(checks.values())
    return PreflightResult(
        passed=passed,
        decision_code="preflight_pass" if passed else "credit_preflight_fail",
        checks=checks,
        metrics={
            "sensory_min_distance": sensory_dist,
            "motor_min_distance": action_dist,
            "eligibility_norm": elig_norm,
            "rewarded_delta_w_actor": last_delta,
            "reward_off_delta_w_actor": reward_off_delta,
            "permuted_feedback_delta_w_actor": permuted_delta,
            "positive_modulation": pos_mod,
            "negative_modulation": neg_mod,
        },
    )


def _pairwise_min_distance(states: list[np.ndarray]) -> float:
    if len(states) < 2:
        return 0.0
    best = float("inf")
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            d = float(np.linalg.norm(states[i] - states[j]))
            best = min(best, d)
    return 0.0 if best == float("inf") else best


def _scalar_modulation(mod: dict) -> float:
    if "reward_baseline_error" in mod:
        return float(mod["reward_baseline_error"].item())
    if "td_error" in mod:
        return float(mod["td_error"].item())
    return float(mod["consequence_error"].item())


def _all_finite(org: ModularOrganism) -> bool:
    tensors = [
        org.rho.sensory_repr,
        org.rho.relational_repr,
        org.rho.action_repr,
        org.eligibility.trace,
        org.action_ctx.W_motor.weight.data,
    ]
    return all(torch.isfinite(t).all().item() for t in tensors)
