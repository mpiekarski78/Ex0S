"""v13: generic action= copy; untrained ignores the tag; no door in use features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import TagStore, write_tag_notes


def test_untrained_use_read_ignores_red_tag(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "action": 2})])
    policy = UsePolicy(seed=7)
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert action == Action.OPEN
    assert meta["policy"]["use"] is False


def test_forced_use_copies_use_key(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "action": 2})])
    policy = UsePolicy(seed=7)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["use"] is True
    assert action == Action.USE_KEY


def test_forced_use_copies_wait(tmp_path: Path):
    write_tag_notes(tmp_path, [("d2.tag", {"door": 2, "action": 0})])
    policy = UsePolicy(seed=7)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_green")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["use"] is True
    assert action == Action.WAIT


def test_use_features_are_s_hit_only():
    a = UsePolicy.features(True, False)
    assert a.tolist() == [1.0, 0.0]


def test_use_update_changes_hash():
    p = UsePolicy(seed=1)
    h0 = p.weight_hash()
    tr = p.decide_use(UsePolicy.features(True, False), epsilon=0.0)
    p.update([tr], 1.0)
    assert p.weight_hash() != h0


if __name__ == "__main__":
    import tempfile

    test_use_features_are_s_hit_only()
    test_use_update_changes_hash()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_use_read_ignores_red_tag(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_use_copies_use_key(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_use_copies_wait(Path(d))
    print("ok")
