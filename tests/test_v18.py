"""v18: write do= vs action=; write here= vs door=. Read/match frozen."""

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


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _write_red(folder: Path, policy: UsePolicy, **kwargs) -> ThreeMemoryAgent:
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        write_from_events=True,
        force_write=True,
        store=TagStore(folder),
        cortex_seed=1337,
        policy_epsilon=0.0,
        **kwargs,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    a.observe_outcome(obs, True, {"opened": True, "action": "use_key"})
    return a


def test_untrained_wkey_writes_action(tmp_path: Path):
    policy = UsePolicy(seed=7)
    _write_red(tmp_path, policy, use_wkey_head=True, value_key="do")
    text = _tags(tmp_path)
    assert "action=2" in text
    assert "do=" not in text


def test_forced_wkey_writes_do(tmp_path: Path):
    policy = UsePolicy(seed=7)
    policy.b_wkey = np.array(3.0, dtype=np.float64)
    _write_red(tmp_path, policy, use_wkey_head=True, value_key="do")
    text = _tags(tmp_path)
    assert "do=2" in text
    assert "action=" not in text


def test_frozen_read_needs_do(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "action": 2})])
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        value_key="do",
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.OPEN


def test_frozen_read_copies_do(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "do": 2})])
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        value_key="do",
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.USE_KEY


def test_untrained_wplace_writes_door(tmp_path: Path):
    policy = UsePolicy(seed=7)
    _write_red(tmp_path, policy, use_wplace_head=True, place_key="here")
    text = _tags(tmp_path)
    assert "door=0" in text
    assert "here=" not in text


def test_forced_wplace_writes_here(tmp_path: Path):
    policy = UsePolicy(seed=7)
    policy.b_wplace = np.array(3.0, dtype=np.float64)
    _write_red(tmp_path, policy, use_wplace_head=True, place_key="here")
    text = _tags(tmp_path)
    assert "here=0" in text
    assert "door=" not in text


def test_frozen_match_needs_here(tmp_path: Path):
    write_tag_notes(tmp_path, [("d0.tag", {"door": 0, "action": 2})])
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        place_key="here",
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.OPEN


def test_frozen_match_uses_here(tmp_path: Path):
    write_tag_notes(tmp_path, [("d2.tag", {"here": 2, "action": 0})])
    a = ThreeMemoryAgent(
        native=True,
        use_policy=UsePolicy(seed=7),
        use_read=True,
        place_key="here",
        force_use=True,
        write_from_events=False,
        store=TagStore(tmp_path),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_green")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.WAIT


def test_empty_green_prior_is_open():
    a = ThreeMemoryAgent(native=True, cortex_seed=1337, write_from_events=False)
    obs = KeyDoorWorld(0).reset("probe_green")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.OPEN


def test_clone_empty_keeps_write_name_flags(tmp_path: Path):
    policy = UsePolicy(seed=7)
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        value_key="do",
        place_key="here",
        use_wkey_head=True,
        use_wplace_head=True,
        store=TagStore(tmp_path),
        cortex_seed=1337,
    )
    b = a.clone_empty()
    assert b.native is True
    assert b.use_read is True
    assert b.value_key == "do"
    assert b.place_key == "here"
    assert b.use_wkey_head is True
    assert b.use_wplace_head is True
    assert b.use_policy is policy
    assert len(b.store) == 0


def test_unknown_trace_kind_does_not_train_collect():
    p = UsePolicy(seed=7)
    h0 = p.weight_hash()
    w0 = p.W_collect.copy()
    try:
        p.update([{"kind": "nope", "feat": [0.0, 1.0], "collect_idx": 0, "apply": False}], 1.0)
    except ValueError as e:
        assert "unknown policy trace kind" in str(e)
    else:
        raise AssertionError("expected ValueError")
    assert p.weight_hash() == h0
    assert np.allclose(p.W_collect, w0)


if __name__ == "__main__":
    import tempfile

    test_empty_green_prior_is_open()
    test_unknown_trace_kind_does_not_train_collect()
    with tempfile.TemporaryDirectory() as d:
        test_clone_empty_keeps_write_name_flags(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_untrained_wkey_writes_action(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_wkey_writes_do(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_frozen_read_needs_do(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_frozen_read_copies_do(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_untrained_wplace_writes_door(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_wplace_writes_here(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_frozen_match_needs_here(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_frozen_match_uses_here(Path(d))
    print("ok")
