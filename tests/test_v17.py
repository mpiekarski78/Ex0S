"""v17: read do= vs action=; match here= vs door=."""

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


def test_untrained_key_ignores_do(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "do": 2})])
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        use_key_head=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["key_alt"] is False
    assert action == Action.OPEN


def test_forced_key_copies_do(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "do": 2})])
    policy = UsePolicy(seed=7)
    policy.b_key = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        use_key_head=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["key_alt"] is True
    assert action == Action.USE_KEY


def test_untrained_match_ignores_here(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"here": 0, "action": 2})])
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        use_match_head=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["match_alt"] is False
    assert action == Action.OPEN


def test_forced_match_uses_here(tmp_path: Path):
    write_tag_notes(tmp_path, [("d2.tag", {"here": 2, "action": 0})])
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        use_match_head=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_green")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert meta["policy"]["match_alt"] is True
    assert action == Action.WAIT


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_untrained_key_ignores_do(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_key_copies_do(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_untrained_match_ignores_here(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_match_uses_here(Path(d))
    print("ok")
