"""Nursery Body v2 unit tests (engineering surface; R4 body untouched)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import (
    SYNERGY_REPORT_NAMES,
    channels_to_synergy_activations,
    expand_synergy_index_to_motor,
    synergy_projection_matrix,
)
from three_memory.dev1.nursery_v2.world import max_useful_travel, reachability_chi


def test_egocentric_names_not_goal_relative():
    for forbidden in ("approach", "withdraw", "orient", "wait"):
        assert forbidden not in SYNERGY_REPORT_NAMES
    assert SYNERGY_REPORT_NAMES == ("forward", "backward", "rotate_left", "rotate_right")


def test_mass_preserving_projection_full_strength():
    P = synergy_projection_matrix()
    onehot = expand_synergy_index_to_motor(0, encoding="onehot_in_block", channel_within_block=3)
    uniform = expand_synergy_index_to_motor(0, encoding="uniform_block")
    s1 = channels_to_synergy_activations(onehot, P)
    s2 = channels_to_synergy_activations(uniform, P)
    assert abs(float(s1[0].item()) - 1.0) < 1e-6
    assert abs(float(s2[0].item()) - 1.0) < 1e-6
    assert torch.allclose(s1, s2, atol=1e-6)


def test_exact_dynamics_match_body():
    from experiments.dev1.nursery_body_v2_certification import (
        exact_state_from_body,
        exact_synergy_transition,
    )

    body = NurseryBodyV2(BodyConfig(seed=4), device=torch.device("cpu"))
    for syn in range(4):
        body.reset(seed=20 + syn)
        before = exact_state_from_body(body)
        body.step(expand_synergy_index_to_motor(syn))
        after_body = exact_state_from_body(body)
        after_exact = exact_synergy_transition(before, syn, cfg=body.config)
        assert abs(after_body.x - after_exact.x) < 1e-5
        assert abs(after_body.y - after_exact.y) < 1e-5
        assert abs(after_body.orientation - after_exact.orientation) < 1e-5


def test_reachability_chi_definition():
    cfg = BodyConfig()
    travel = max_useful_travel(cfg, 16)
    assert abs(travel - 0.15 * 16) < 1e-9
    chi = reachability_chi(1.05, cfg, 16)
    assert abs(chi - (1.05 - 0.35) / travel) < 1e-9
    assert chi <= 1.0  # analytically reachable
    assert chi <= 0.85  # safety rule
    report = __import__(
        "three_memory.dev1.nursery_v2.world", fromlist=["analytic_reachability_report"]
    ).analytic_reachability_report("unit_chi", n_episodes=4, episode_ticks=16)
    assert report["analytical_reachable_rule"] == "chi <= 1"
    assert report["safety_rule"] == "chi <= 0.85"
