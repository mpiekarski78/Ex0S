"""v12: retrieve features have no door id; untrained dumps; select ignores the other life."""

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


def test_retrieve_features_ignore_counts_only():
    a = UsePolicy.retrieve_features(2, 1)
    b = UsePolicy.retrieve_features(5, 1)
    assert a.tolist() == [1.0, 1.0]
    assert b.tolist() == [1.0, 1.0]


def test_untrained_dumps_two_notes(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [("d0.tag", {"door": 0, "action": 2}), ("d2.tag", {"door": 2, "action": 0})],
    )
    policy = UsePolicy(seed=7)
    a = ThreeMemoryAgent(
        native=True,
        retrieve_policy="policy",
        use_policy=policy,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["retrieve_mode"] == "dump"
    assert action != Action.USE_KEY


def test_forced_select_uses_red_note(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [("d0.tag", {"door": 0, "action": 2}), ("d2.tag", {"door": 2, "action": 0})],
    )
    policy = UsePolicy(seed=7)
    policy.b_retrieve = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        retrieve_policy="policy",
        use_policy=policy,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["retrieve_mode"] == "select"
    assert action == Action.USE_KEY


def test_retrieve_update_changes_hash():
    p = UsePolicy(seed=1)
    h0 = p.weight_hash()
    tr = p.decide_retrieve(UsePolicy.retrieve_features(2, 1), epsilon=0.0)
    p.update([tr], 1.0)
    assert p.weight_hash() != h0


if __name__ == "__main__":
    import tempfile

    test_retrieve_features_ignore_counts_only()
    test_retrieve_update_changes_hash()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_dumps_two_notes(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_select_uses_red_note(Path(d))
    print("ok")
