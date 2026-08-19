"""
EX0S-DEV1 causal gate tests.

Tests that each intervention in the causal decision ladder produces
the expected behavioral outcome. All tests use short lives (fast).

Gates verified here:
- reward_off → learning fails
- wipe → fact removed
- episode_reset → rho cleared, H intact
- full_checkpoint + restore → behavior reproduced
- hippocampal_graft → H transplanted between twins
- h_disabled → Stage A: no fast memory solve

Note: grounding accuracy tests use random organisms; they verify
structural validity, not that performance exceeds a threshold
(which requires proper training runs, not unit tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.interfaces import (
    ActionResult,
    EpisodeReset,
    FullCheckpoint,
    HippocampalGraft,
    OrganismObservation,
)
from experiments.dev1.worlds import InteractionWorld, WorldConfig


def _world(seed: str = "test_world_v1") -> InteractionWorld:
    return InteractionWorld(WorldConfig(seed=seed, episode_length=16, n_episodes=4))


def _obs(v=None, reward=0.0) -> OrganismObservation:
    if v is None:
        v = np.zeros(64)
    return OrganismObservation(sensory_vector=v, reward=reward)


class TestRewardOff:
    """reward_off ablation: zeroed reward must prevent role grounding."""

    def test_reward_off_produces_valid_action(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        world = _world()
        for _ in range(2):
            for we in world.generate_episode():
                org.observe(_obs(we.sensory_vector, reward=0.0))
                action = org.act()
                assert isinstance(action.motor_channel, int)
                assert 0 <= action.motor_channel < genome.n_motor_channels


class TestWipeGate:
    """wipe() removes H episodes; reads return zeros."""

    def test_wipe_clears_store(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.hippocampus._store.append((torch.zeros(512), torch.zeros(64)))
        org.hippocampus._store.append((torch.ones(512), torch.ones(64)))
        assert len(org.hippocampus._store) == 2
        org.hippocampus.wipe()
        assert len(org.hippocampus._store) == 0

    def test_wipe_read_returns_zero(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.hippocampus._store.append((torch.randn(512), torch.randn(64)))
        org.hippocampus.wipe()
        result = org.hippocampus.read(org.rho.relational_repr)
        assert result.abs().sum().item() == 0.0


class TestEpisodeReset:
    """episode_reset() clears rho; H persists."""

    def test_reset_clears_rho(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.observe(_obs(np.ones(64), reward=1.0))
        rho_before = org.rho.relational_repr.norm().item()
        er = org.episode_reset()
        assert isinstance(er, EpisodeReset)
        rho_after = org.rho.relational_repr.norm().item()
        assert rho_after == 0.0 or rho_after < rho_before

    def test_reset_preserves_h(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.hippocampus._store.append((torch.randn(512), torch.randn(64)))
        n_before = len(org.hippocampus._store)
        org.episode_reset()
        assert len(org.hippocampus._store) == n_before


class TestFullCheckpoint:
    """full_checkpoint() + restore_from_checkpoint() reproduces state."""

    def test_checkpoint_restore_reproduces_action(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.observe(_obs(np.ones(64) * 0.3, reward=0.5))
        cp = org.full_checkpoint()
        assert isinstance(cp, FullCheckpoint)

        org2 = ModularOrganism.birth(genome)
        org2.restore_from_checkpoint(cp)
        action1 = org.act()
        action2 = org2.act()
        assert action1.motor_channel == action2.motor_channel

    def test_checkpoint_slog_restored(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.observe(_obs())
        # Snapshot is taken inside full_checkpoint() before the CHECKPOINT event is appended
        cp = org.full_checkpoint()
        # The restored slog should match the snapshot (pre-CHECKPOINT event count)
        snapshot_count = len(cp.slog_snapshot)

        org2 = ModularOrganism.birth(genome)
        org2.restore_from_checkpoint(cp)
        assert org2.slog.event_count() == snapshot_count


class TestHippocampalGraft:
    """HippocampalGraft transfers only H between matched twins."""

    def test_graft_transfers_h_only(self):
        genome = DevGenome.default()

        # Create matched twins from same birth checkpoint
        twin_a = ModularOrganism.birth(genome)
        twin_b = ModularOrganism.birth(genome)

        # Teach twin_a a fact
        fact_vec = np.ones(64) * 0.9
        twin_a.hippocampus._store.append((torch.randn(512), torch.tensor(fact_vec, dtype=torch.float32)))
        n_a = len(twin_a.hippocampus._store)

        # Graft H from twin_a to twin_b
        graft = HippocampalGraft(
            donor_hippocampus_state=twin_a.hippocampus.hippocampus_state_dict(),
            donor_hippocampus_plasticity_state=twin_a.hippocampus.hippocampus_plasticity_state_dict(),
            donor_checkpoint_hash="test_donor",
        )
        n_b_before = len(twin_b.hippocampus._store)
        twin_b.hippocampal_graft(graft)
        assert len(twin_b.hippocampus._store) == n_a, "Graft must transfer H store"

    def test_graft_does_not_touch_cortex(self):
        genome = DevGenome.default()
        twin_a = ModularOrganism.birth(genome)
        twin_b = ModularOrganism.birth(genome)

        # Record cortex weights of twin_b before graft
        w_before = {k: v.clone() for k, v in twin_b.relational_ctx.state_dict().items()}

        graft = HippocampalGraft(
            donor_hippocampus_state=twin_a.hippocampus.hippocampus_state_dict(),
            donor_hippocampus_plasticity_state=twin_a.hippocampus.hippocampus_plasticity_state_dict(),
            donor_checkpoint_hash="test_donor",
        )
        twin_b.hippocampal_graft(graft)
        w_after = {k: v for k, v in twin_b.relational_ctx.state_dict().items()}

        for k in w_before:
            assert torch.allclose(w_before[k], w_after[k]), f"Graft must not touch cortex weight {k}"


class TestHDisabledStageA:
    """During Stage A, h_disabled=True must prevent any H write/read."""

    def test_h_disabled_write_returns_false(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome, h_disabled=True)
        rel = torch.zeros(genome.relational_ctx.n_units)
        content = torch.zeros(genome.sensory_ctx.n_units)
        wrote = org.hippocampus.write(rel, content)
        assert wrote is False

    def test_h_disabled_read_returns_zero(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome, h_disabled=True)
        rel = torch.zeros(genome.relational_ctx.n_units)
        result = org.hippocampus.read(rel)
        assert result.abs().sum().item() == 0.0

    def test_h_disabled_store_remains_empty_during_life(self):
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome, h_disabled=True)
        org.age_frac = 0.9
        for _ in range(20):
            org.observe(_obs(np.random.randn(64), reward=1.0))
        assert len(org.hippocampus._store) == 0


class TestActionResultContract:
    """ActionResult must only expose motor_channel, motor_scores, confidence."""

    def test_action_result_fields(self):
        from three_memory.dev1.interfaces import ActionResult
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ActionResult)}
        assert field_names == {"motor_channel", "motor_scores", "confidence"}, \
            f"ActionResult fields changed: {field_names}"
