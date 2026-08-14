"""v14: pick one among matches vs write complete schema. Compare A and B."""

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


def _agent(tmp_path: Path, policy: UsePolicy, **kw) -> ThreeMemoryAgent:
    return ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
        **kw,
    )


def test_pick_features_ignore_counts_only():
    a = UsePolicy.pick_features(2)
    b = UsePolicy.pick_features(5)
    assert a.tolist() == [1.0, 1.0]
    assert b.tolist() == [1.0, 1.0]


def test_untrained_pick_mixes_two_red_notes(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    a = _agent(tmp_path, UsePolicy(seed=7), use_pick=True, force_use=True)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["one"] is False
    assert action != Action.USE_KEY


def test_forced_pick_newest_uses_red(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.b_pick = np.array(3.0, dtype=np.float64)
    a = _agent(tmp_path, policy, use_pick=True, force_use=True)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["one"] is True
    assert action == Action.USE_KEY


def test_forced_pick_newest_waits_green(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d2_t1.tag", {"door": 2, "action": 1, "when": 1}),
            ("d2_t10.tag", {"door": 2, "action": 0, "when": 10}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.b_pick = np.array(3.0, dtype=np.float64)
    a = _agent(tmp_path, policy, use_pick=True, force_use=True)
    obs = KeyDoorWorld(0).reset("probe_green")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["one"] is True
    assert action == Action.WAIT


def test_incomplete_note_does_not_copy_use_key(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0})])
    a = _agent(tmp_path, UsePolicy(seed=7), force_use=True)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.OPEN


def test_reload_preserves_when(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0_t7.tag", {"door": 0, "action": 2, "when": 7})])
    store = TagStore(tmp_path)
    recs = store.records()
    assert recs[0].when == 7
    assert recs[0].tags["when"] == 7


if __name__ == "__main__":
    import tempfile

    test_pick_features_ignore_counts_only()
    with tempfile.TemporaryDirectory() as d:
        test_reload_preserves_when(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_untrained_pick_mixes_two_red_notes(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_pick_newest_uses_red(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_pick_newest_waits_green(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_incomplete_note_does_not_copy_use_key(Path(d))
    print("ok")
