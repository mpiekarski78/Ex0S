"""R4-R1 ceiling measurement invariants (protocol repair; body/credit frozen)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.developmental_birth_r4_ceiling import (
    ceiling_body_behavior_passes,
    evaluate_ceiling_gate_bundle,
    evaluate_ceiling_on_body_world,
    expand_synergy_index_to_motor,
    expand_synergy_probs_to_motor,
    permute_channels_within_synergy,
)
from three_memory.dev1.body.physics import BodyConfig, GenericBody
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.synergies import channels_to_synergy_activations


def test_onehot_and_uniform_block_same_body_transition():
    body = GenericBody(BodyConfig(seed=42), device=torch.device("cpu"))
    for s in range(4):
        body.reset(seed=100 + s)
        onehot = expand_synergy_index_to_motor(s, encoding="onehot_in_block", channel_within_block=3)
        body.reset(seed=100 + s)
        a = body.step(onehot)
        body.reset(seed=100 + s)
        uniform = expand_synergy_index_to_motor(s, encoding="uniform_block")
        b = body.step(uniform)
        assert torch.allclose(a.synergy_activations, b.synergy_activations, atol=1e-6)
        assert torch.allclose(a.body_state.position, b.body_state.position, atol=1e-6)
        assert abs(a.body_state.orientation - b.body_state.orientation) < 1e-6


def test_distributed_synergy_probs_match_index_expansion():
    probs = torch.tensor([0.0, 1.0, 0.0, 0.0])
    from_probs = expand_synergy_probs_to_motor(probs)
    from_idx = expand_synergy_index_to_motor(1, encoding="uniform_block")
    assert torch.allclose(from_probs, from_idx, atol=1e-6)


def test_within_synergy_channel_permutation_leaves_body_unchanged():
    motor = expand_synergy_index_to_motor(2, encoding="uniform_block")
    permuted = permute_channels_within_synergy(motor, perm_seed=9)
    # Uniform block is invariant to within-block permutation by construction.
    assert torch.allclose(motor, permuted, atol=1e-6)

    # One-hot within block: any channel in the same synergy → identical synergy / body step.
    body = GenericBody(BodyConfig(seed=7), device=torch.device("cpu"))
    refs = []
    for ch in range(8):
        body.reset(seed=7)
        m = expand_synergy_index_to_motor(0, encoding="onehot_in_block", channel_within_block=ch)
        refs.append(body.step(m))
    for r in refs[1:]:
        assert torch.allclose(refs[0].synergy_activations, r.synergy_activations, atol=1e-6)
        assert torch.allclose(refs[0].body_state.position, r.body_state.position, atol=1e-6)


def test_shared_synergy_credit_not_32_way_actor():
    g = GenerativeGenome.small()
    # Actor head is 4-way; expansion uses shared block mass.
    out = evaluate_ceiling_on_body_world(
        g,
        "r4_r1_ceiling_invariant",
        n_episodes=2,
        episode_ticks=4,
        device=torch.device("cpu"),
    )
    assert out["n_synergies"] == 4
    assert out["n_motor_channels"] == 32
    assert out["ceiling_kind"] == "synergy_aware_r4_r1"
    assert out["no_expected_action"] is True
    assert len(out["synergy_histogram"]) == 4
    assert sum(out["synergy_histogram"]) == 8


def test_ceiling_gate_bundle_has_body_behavior_fields():
    g = GenerativeGenome.small()
    bundle = evaluate_ceiling_gate_bundle(
        g,
        "r4_r1_ceiling_gate_bundle",
        n_episodes=2,
        episode_ticks=4,
        device=torch.device("cpu"),
    )
    for key in (
        "final_comfort_rate",
        "comfort_improvement",
        "distance_reduction",
        "comfort_margin_over_random",
        "comfort_margin_over_untrained",
    ):
        assert key in bundle
    # Gate helper is strict conjunctive; tiny smoke budgets need not pass.
    thr = {
        "min_final_comfort_rate": 1.1,
        "min_comfort_improvement": 1.1,
        "min_distance_reduction": 1.1,
        "min_margin_over_random": 1.1,
    }
    assert ceiling_body_behavior_passes(bundle, thr) is False


def test_projection_shared_credit_mass():
    """Channels in one synergy contribute identically under frozen projection."""
    P_act = channels_to_synergy_activations
    m0 = expand_synergy_index_to_motor(0, encoding="onehot_in_block", channel_within_block=0)
    m7 = expand_synergy_index_to_motor(0, encoding="onehot_in_block", channel_within_block=7)
    assert torch.allclose(P_act(m0), P_act(m7), atol=1e-6)
