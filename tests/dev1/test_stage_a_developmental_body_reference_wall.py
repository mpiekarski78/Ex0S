"""Tests for Developmental Body Reference Wall (measurement only)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.developmental_birth_r4_ceiling import expand_synergy_index_to_motor
from experiments.dev1.developmental_body_reference_wall import (
    ExactState,
    apply_wall_routing,
    exact_synergy_transition,
    exact_state_from_body,
    model_based_choose_synergy,
    observation_key,
)
from three_memory.dev1.body.physics import BodyConfig, GenericBody
from three_memory.dev1.development.synergies import channels_to_synergy_activations


def test_exact_dynamics_match_generic_body():
    body = GenericBody(BodyConfig(seed=3), device=torch.device("cpu"))
    for syn in range(4):
        body.reset(seed=100 + syn)
        before = exact_state_from_body(body)
        motor = expand_synergy_index_to_motor(syn, encoding="uniform_block")
        body.step(motor)
        after_body = exact_state_from_body(body)
        after_exact = exact_synergy_transition(before, syn)
        assert abs(after_body.x - after_exact.x) < 1e-5
        assert abs(after_body.y - after_exact.y) < 1e-5
        assert abs(after_body.orientation - after_exact.orientation) < 1e-5


def test_model_based_not_heuristic_label():
    # Capacity controller must prefer reducing |x| via approach/withdraw when on x-axis.
    st = ExactState(x=0.8, y=0.0, orientation=0.0, energy=0.7)
    a = model_based_choose_synergy(st, horizon=8, beam=16)
    assert a in (1, 3)  # withdraw toward origin, or wait if already planning hold
    st2 = ExactState(x=-0.8, y=0.0, orientation=0.0, energy=0.7)
    b = model_based_choose_synergy(st2, horizon=8, beam=16)
    assert b in (0, 3)


def test_observation_key_stable():
    a = torch.zeros(48)
    a[0], a[1] = 0.5, -0.25
    i = torch.tensor([0.4, 0.7, 0.1, 0.4])
    k1 = observation_key(a, i)
    k2 = observation_key(a.clone(), i.clone())
    assert k1 == k2


def test_routing_model_based_fail():
    thr = {
        "min_final_comfort_rate": 0.05,
        "min_distance_reduction": 0.05,
        "min_margin_over_random": 0.02,
    }
    rnd = {"comfort_rate": 0.02, "distance_reduction": 0.0}
    bad = {"comfort_rate": 0.02, "distance_reduction": 0.0}
    good = {"comfort_rate": 0.2, "distance_reduction": 0.1}
    results = {
        "random_policy": rnd,
        "exact_model_based_controller": bad,
        "full_state_supervised_controller": good,
        "same_observation_supervised_controller": good,
        "feedforward_actor_critic": good,
        "recurrent_actor_critic": good,
    }
    assert apply_wall_routing(results, thr) == "body_physics_or_control_bug"


def test_uniform_block_synergy_activation():
    m = expand_synergy_index_to_motor(0, encoding="uniform_block")
    syn = channels_to_synergy_activations(m)
    assert abs(float(syn[0].item()) - 0.125) < 1e-6
