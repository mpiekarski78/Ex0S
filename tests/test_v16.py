"""v16: rank ok= vs newest; untrained recency prior."""

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
        use_rank=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
        **kw,
    )


def test_rank_features_are_newest_and_ok_only():
    a = UsePolicy.rank_features(True, False)
    b = UsePolicy.rank_features(False, True)
    assert a.tolist() == [1.0, 0.0]
    assert b.tolist() == [0.0, 1.0]


def test_untrained_rank_follows_newest(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_ok.tag", {"door": 0, "action": 2, "when": 1, "ok": 1}),
            ("d0_new.tag", {"door": 0, "action": 0, "when": 10}),
        ],
    )
    a = _agent(tmp_path, UsePolicy(seed=7), force_use=True)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["is_newest"] is True
    assert meta["policy"]["has_ok"] is False
    assert action != Action.USE_KEY
    assert action == Action.WAIT


def test_trained_rank_prefers_ok(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_ok.tag", {"door": 0, "action": 2, "when": 1, "ok": 1}),
            ("d0_new.tag", {"door": 0, "action": 0, "when": 10}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.w_rank = np.array([0.0, 3.0], dtype=np.float64)
    a = _agent(tmp_path, policy, force_use=True)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["has_ok"] is True
    assert action == Action.USE_KEY


def test_ok_prefers_wait_on_green(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d2_ok.tag", {"door": 2, "action": 0, "when": 1, "ok": 1}),
            ("d2_new.tag", {"door": 2, "action": 1, "when": 10}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.w_rank = np.array([0.0, 3.0], dtype=np.float64)
    a = _agent(tmp_path, policy, force_use=True)
    obs = KeyDoorWorld(0).reset("probe_green")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["has_ok"] is True
    assert action == Action.WAIT


if __name__ == "__main__":
    import tempfile

    test_rank_features_are_newest_and_ok_only()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_rank_follows_newest(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_trained_rank_prefers_ok(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_ok_prefers_wait_on_green(Path(d))
    print("ok")
