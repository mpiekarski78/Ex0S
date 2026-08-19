"""
Motor-permutation equivariance audit for Stage A R2 scaffold wiring.

The scaffold may organize a generic motor basis but must NOT encode fixed
action preferences or cue→action routes. Permuting motor channels must permute
behavior and learning correspondingly.

Three claims are tested:
1. Permuting the W_motor rows permutes action preferences correspondingly.
2. Permuting motor channels permutes the local actor update (dW) correspondingly.
3. The motor-channel bias introduced by the scaffold permutes under the same
   permutation — it must not break equivariance by assigning semantically
   meaningful positions to specific channels.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.scaffold_r2 import (
    ContinuousScaffoldPhenotype,
    TopologyScaffoldPhenotype,
    apply_scaffold_to_organism,
)
from experiments.dev1.worlds import InteractionWorld, WorldConfig
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


def _birth_with_scaffold(
    genome: DevGenome,
    continuous: ContinuousScaffoldPhenotype,
    topology: TopologyScaffoldPhenotype,
) -> ModularOrganism:
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    apply_scaffold_to_organism(org, continuous, topology)
    return org


def _permute_organism(org: ModularOrganism, perm: list[int]) -> ModularOrganism:
    """
    Return a deep-copy of org with W_motor rows permuted by `perm`,
    and the scaffold motor-channel bias permuted by the same permutation.
    """
    org2 = copy.deepcopy(org)
    perm_t = torch.tensor(perm, dtype=torch.long)
    with torch.no_grad():
        org2.action_ctx.W_motor.weight.data = org2.action_ctx.W_motor.weight.data[perm_t]
        if hasattr(org2, "_r2_motor_channel_bias"):
            org2._r2_motor_channel_bias = org2._r2_motor_channel_bias[perm_t]
        if hasattr(org2, "_r2_plasticity_channel_mask"):
            org2._r2_plasticity_channel_mask = org2._r2_plasticity_channel_mask[perm_t]
    return org2


def _run_one_step(org: ModularOrganism, world: InteractionWorld) -> tuple[int, torch.Tensor]:
    """One observe→act cycle; returns (channel, W_motor_snapshot_before_rest)."""
    event = world.generate_episode()[0]
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    action = org.act(policy_mode="hard")
    reward = world.reward_for_action(event, action.motor_channel)
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=reward))
    w_before = org.action_ctx.W_motor.weight.data.clone()
    org.rest()
    w_after = org.action_ctx.W_motor.weight.data.clone()
    dW = w_after - w_before
    return action.motor_channel, dW


def test_motor_permutation_permutes_action_channel():
    """
    Permuting W_motor rows permutes which motor channel is preferred.
    If channel k is preferred before permutation, channel perm[k] is preferred
    after, given the same sensory state.

    We verify this by:
    1. Recording motor logits before permutation → channel k chosen.
    2. Building permuted organism, running the same observe.
    3. Checking that the permuted organism chooses perm[k].
    """
    genome = DevGenome.default()
    n_ch = genome.n_motor_channels
    continuous = ContinuousScaffoldPhenotype(motor_basis_scale=1.4)
    topology = TopologyScaffoldPhenotype(motif="dense")
    world = WorldConfig(seed="perm_audit_seed_001", n_roles=n_ch)
    iw = InteractionWorld(world)

    perm = list(range(n_ch - 1, -1, -1))  # reverse permutation

    org_base = _birth_with_scaffold(genome, continuous, topology)
    event = iw.generate_episode()[0]
    org_base.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    action_base = org_base.act(policy_mode="hard")
    k = action_base.motor_channel

    # Clone state and apply permutation
    org_perm = _birth_with_scaffold(genome, continuous, topology)
    # Copy weights from base so they share the same learned state
    with torch.no_grad():
        for name in ["W_in", "W_rec"]:
            for ctx_name in ["sensory_ctx", "relational_ctx", "action_ctx"]:
                ctx = getattr(org_perm, ctx_name)
                if hasattr(ctx, "pop"):
                    w = getattr(ctx.pop, name, None)
                    wb = getattr(getattr(org_base, ctx_name).pop, name, None)
                    if w is not None and wb is not None:
                        w.weight.data.copy_(wb.weight.data)
    # Apply same permutation
    perm_t = torch.tensor(perm, dtype=torch.long)
    with torch.no_grad():
        org_perm.action_ctx.W_motor.weight.data.copy_(
            org_base.action_ctx.W_motor.weight.data[perm_t]
        )
        if hasattr(org_base, "_r2_motor_channel_bias"):
            org_perm._r2_motor_channel_bias = org_base._r2_motor_channel_bias[perm_t].clone()

    # Copy working state so they share the same sensory context
    with torch.no_grad():
        org_perm.rho.sensory_repr.copy_(org_base.rho.sensory_repr)
        org_perm.rho.relational_repr.copy_(org_base.rho.relational_repr)

    action_perm = org_perm.act(policy_mode="hard")
    expected_channel = perm[k]
    assert action_perm.motor_channel == expected_channel, (
        f"Expected permuted channel {expected_channel} (perm[{k}]) "
        f"but got {action_perm.motor_channel}"
    )


def test_motor_permutation_permutes_actor_update():
    """
    Permuting motor channels permutes the dW update correspondingly.

    After one rewarded interaction:
    - dW_base has a pattern across motor channels.
    - dW_perm (from permuted organism) should equal dW_base with rows permuted.

    This checks that the actor credit rule is equivariant: the scaffold
    may not break equivariance by coupling learning preferentially to
    specific channel indices.
    """
    genome = DevGenome.default()
    n_ch = genome.n_motor_channels
    continuous = ContinuousScaffoldPhenotype(motor_basis_scale=1.0, plasticity_mask_gain=1.0)
    topology = TopologyScaffoldPhenotype(motif="dense")
    world = WorldConfig(seed="perm_audit_seed_002", n_roles=n_ch)
    iw = InteractionWorld(world)
    perm = list(range(n_ch - 1, -1, -1))  # reverse

    org_base = _birth_with_scaffold(genome, continuous, topology)
    # Force same sensory event for both
    events = iw.generate_episode()
    event = events[0]

    org_base.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    action_base = org_base.act(policy_mode="hard")
    reward = iw.reward_for_action(event, action_base.motor_channel)
    org_base.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=reward))
    w_before_base = org_base.action_ctx.W_motor.weight.data.clone()
    org_base.rest()
    dW_base = org_base.action_ctx.W_motor.weight.data - w_before_base  # (n_ch, n_action)

    # Build permuted organism with same recurrent state
    org_perm = _birth_with_scaffold(genome, continuous, topology)
    perm_t = torch.tensor(perm, dtype=torch.long)
    with torch.no_grad():
        org_perm.action_ctx.W_motor.weight.data.copy_(
            org_base.action_ctx.W_motor.weight.data[perm_t]
        )
        if hasattr(org_base, "_r2_motor_channel_bias"):
            org_perm._r2_motor_channel_bias = org_base._r2_motor_channel_bias[perm_t].clone()
        if hasattr(org_base, "_r2_plasticity_channel_mask"):
            org_perm._r2_plasticity_channel_mask = org_base._r2_plasticity_channel_mask[perm_t].clone()
        # Copy recurrent state from base so the eligibility trace evolves the same way
        org_perm.rho.sensory_repr.copy_(org_base.rho.sensory_repr)
        org_perm.rho.relational_repr.copy_(org_base.rho.relational_repr)
        org_perm.rho.action_repr.copy_(org_base.rho.action_repr)
        org_perm.eligibility.trace.copy_(org_base.eligibility.trace)

    # Perm organism uses the channel at perm[action_base.motor_channel]
    permuted_channel = perm[action_base.motor_channel]
    org_perm._last_action_channel = permuted_channel
    org_perm._last_mod = org_base._last_mod
    w_before_perm = org_perm.action_ctx.W_motor.weight.data.clone()
    org_perm.rest()
    dW_perm = org_perm.action_ctx.W_motor.weight.data - w_before_perm

    # dW_perm should equal dW_base with rows permuted by perm
    dW_base_permuted = dW_base[perm_t]
    max_err = float((dW_perm - dW_base_permuted).abs().max().item())
    assert max_err < 1e-5, (
        f"Actor update not equivariant under motor-channel permutation: "
        f"max element-wise error = {max_err:.2e}"
    )


def test_scaffold_bias_does_not_encode_fixed_preference_magnitude():
    """
    The scaffold motor-channel bias must be small relative to the dynamic
    range of motor logits so it organizes the basis without fixing a winner.

    Specifically: for all motifs, the bias range (max - min) must be ≤ 0.15
    (5% scale × 3 for linspace extremes) and the bias values must sum to
    approximately zero (no net drift toward any channel index).
    """
    genome = DevGenome.default()
    n_ch = genome.n_motor_channels

    for motif in ["dense", "block", "banded"]:
        continuous = ContinuousScaffoldPhenotype(motor_basis_scale=1.0)
        topology = TopologyScaffoldPhenotype(motif=motif)
        org = _birth_with_scaffold(genome, continuous, topology)
        bias = org._r2_motor_channel_bias  # (n_ch,)
        bias_range = float((bias.max() - bias.min()).item())
        bias_sum = float(bias.sum().item())

        assert bias_range <= 0.15, (
            f"motif={motif}: bias range {bias_range:.4f} exceeds 0.15"
        )
        # banded uses sin so sum won't be exactly 0, but should not
        # strongly favor low or high channel indices
        if motif == "dense":
            assert abs(bias_sum) < 1e-4, (
                f"motif=dense: bias sum {bias_sum:.4f} is non-zero, "
                "indicating a fixed drift toward one channel"
            )


def test_permuted_scaffold_bias_permutes_channel_ordering():
    """
    After applying a row permutation to W_motor and the bias vector,
    the resulting action ordering is the permuted ordering of the original.

    This ensures the scaffold wiring preserves equivariance: a permutation
    applied to the motor basis results in an equivalent permuted behavior,
    with no 'canonical' channel ordering hard-coded in the bias.
    """
    genome = DevGenome.default()
    n_ch = genome.n_motor_channels
    perm = list(range(n_ch - 1, -1, -1))
    perm_t = torch.tensor(perm, dtype=torch.long)

    for motif in ["dense", "banded"]:
        continuous = ContinuousScaffoldPhenotype(motor_basis_scale=1.2)
        topology = TopologyScaffoldPhenotype(motif=motif)
        org = _birth_with_scaffold(genome, continuous, topology)

        # Logits before permutation
        with torch.no_grad():
            _, logits_base = org.action_ctx(org.rho.relational_repr, org.rho.action_repr)
            logits_base = logits_base + org._r2_motor_channel_bias

        # Build permuted copy
        org2 = copy.deepcopy(org)
        with torch.no_grad():
            org2.action_ctx.W_motor.weight.data = org2.action_ctx.W_motor.weight.data[perm_t]
            org2._r2_motor_channel_bias = org2._r2_motor_channel_bias[perm_t]

        with torch.no_grad():
            _, logits_perm = org2.action_ctx(org2.rho.relational_repr, org2.rho.action_repr)
            logits_perm = logits_perm + org2._r2_motor_channel_bias

        # The permuted logits should equal the original logits permuted
        expected = logits_base[perm_t]
        max_err = float((logits_perm - expected).abs().max().item())
        assert max_err < 1e-5, (
            f"motif={motif}: permuted motor logits do not match permuted original "
            f"(max err {max_err:.2e}); scaffold bias breaks equivariance"
        )
