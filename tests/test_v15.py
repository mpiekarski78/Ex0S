"""v15: joint write/schema/use/pick. No force_use or force_write."""

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


def _joint(tmp_path: Path, policy: UsePolicy, **kw) -> ThreeMemoryAgent:
    return ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        use_pick=True,
        write_schema=True,
        unique_writes=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
        **kw,
    )


def test_untrained_joint_ignores_conflict(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    a = _joint(tmp_path, UsePolicy(seed=7))
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["use"] is False
    assert action != Action.USE_KEY


def test_forced_joint_picks_newest_and_uses(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.b_pick = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = _joint(tmp_path, policy)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["one"] is True
    assert meta["policy"]["use"] is True
    assert action == Action.USE_KEY


def test_v14_pick_with_force_use_still_mixes_untrained(tmp_path: Path):
    write_tag_notes(
        tmp_path,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        use_pick=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["one"] is False
    assert action != Action.USE_KEY


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_untrained_joint_ignores_conflict(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_joint_picks_newest_and_uses(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_v14_pick_with_force_use_still_mixes_untrained(Path(d))
    print("ok")
