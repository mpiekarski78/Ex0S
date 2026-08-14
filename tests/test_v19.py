"""v19: shared field name. Neither side frozen. Untrained write/read disagree."""

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


def _mismatch_value(policy: UsePolicy) -> None:
    """Untrained: write action=, read do=."""
    policy.b_wkey = np.array(-1.2, dtype=np.float64)
    policy.b_key = np.array(1.2, dtype=np.float64)


def _mismatch_place(policy: UsePolicy) -> None:
    """Untrained: write door=, match here=."""
    policy.b_wplace = np.array(-1.2, dtype=np.float64)
    policy.b_match = np.array(1.2, dtype=np.float64)


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


def _probe_red(folder: Path, policy: UsePolicy, **kwargs) -> tuple[int, dict]:
    a = ThreeMemoryAgent(
        native=True,
        use_policy=policy,
        use_read=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(folder),
        cortex_seed=1337,
        policy_epsilon=0.0,
        **kwargs,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    return a.act(obs, update_rho=False, explore=False)


def test_untrained_value_mismatch_stays_open(tmp_path: Path):
    policy = UsePolicy(seed=7)
    _mismatch_value(policy)
    _write_red(tmp_path, policy, use_wkey_head=True, use_key_head=True)
    text = _tags(tmp_path)
    assert "action=2" in text
    assert "do=" not in text
    action, meta = _probe_red(tmp_path, policy, use_key_head=True)
    assert meta["policy"]["key_alt"] is True
    assert action == Action.OPEN


def test_both_alt_copies_do(tmp_path: Path):
    policy = UsePolicy(seed=7)
    policy.b_wkey = np.array(3.0, dtype=np.float64)
    policy.b_key = np.array(3.0, dtype=np.float64)
    _write_red(tmp_path, policy, use_wkey_head=True, use_key_head=True)
    text = _tags(tmp_path)
    assert "do=2" in text
    assert "action=" not in text
    action, meta = _probe_red(tmp_path, policy, use_key_head=True)
    assert meta["policy"]["key_alt"] is True
    assert action == Action.USE_KEY


def test_both_old_name_copies_action(tmp_path: Path):
    policy = UsePolicy(seed=7)
    policy.b_wkey = np.array(-3.0, dtype=np.float64)
    policy.b_key = np.array(-3.0, dtype=np.float64)
    _write_red(tmp_path, policy, use_wkey_head=True, use_key_head=True)
    text = _tags(tmp_path)
    assert "action=2" in text
    assert "do=" not in text
    action, meta = _probe_red(tmp_path, policy, use_key_head=True)
    assert meta["policy"]["key_alt"] is False
    assert action == Action.USE_KEY


def test_untrained_place_mismatch_stays_open(tmp_path: Path):
    policy = UsePolicy(seed=7)
    _mismatch_place(policy)
    _write_red(tmp_path, policy, use_wplace_head=True, use_match_head=True)
    text = _tags(tmp_path)
    assert "door=0" in text
    assert "here=" not in text
    action, meta = _probe_red(tmp_path, policy, use_match_head=True)
    assert meta["policy"]["match_alt"] is True
    assert action == Action.OPEN


def test_both_place_alt_matches_here(tmp_path: Path):
    policy = UsePolicy(seed=7)
    policy.b_wplace = np.array(3.0, dtype=np.float64)
    policy.b_match = np.array(3.0, dtype=np.float64)
    _write_red(tmp_path, policy, use_wplace_head=True, use_match_head=True)
    text = _tags(tmp_path)
    assert "here=0" in text
    assert "door=" not in text
    action, meta = _probe_red(tmp_path, policy, use_match_head=True)
    assert meta["policy"]["match_alt"] is True
    assert action == Action.USE_KEY


def test_both_place_old_name_matches_door(tmp_path: Path):
    policy = UsePolicy(seed=7)
    policy.b_wplace = np.array(-3.0, dtype=np.float64)
    policy.b_match = np.array(-3.0, dtype=np.float64)
    _write_red(tmp_path, policy, use_wplace_head=True, use_match_head=True)
    text = _tags(tmp_path)
    assert "door=0" in text
    assert "here=" not in text
    action, meta = _probe_red(tmp_path, policy, use_match_head=True)
    assert meta["policy"]["match_alt"] is False
    assert action == Action.USE_KEY


def test_default_priors_still_prefer_old_names():
    p = UsePolicy(seed=7)
    assert float(p.b_key) < 0.0
    assert float(p.b_wkey) < 0.0
    assert float(p.b_match) < 0.0
    assert float(p.b_wplace) < 0.0


def test_wait_zero_still_copies(tmp_path: Path):
    write_tag_notes(tmp_path, [("d2.tag", {"door": 2, "action": 0})])
    policy = UsePolicy(seed=7)
    policy.b_key = np.array(-3.0, dtype=np.float64)
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
    obs = KeyDoorWorld(0).reset("probe_green")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.WAIT


if __name__ == "__main__":
    import tempfile

    fns = [
        test_untrained_value_mismatch_stays_open,
        test_both_alt_copies_do,
        test_both_old_name_copies_action,
        test_untrained_place_mismatch_stays_open,
        test_both_place_alt_matches_here,
        test_both_place_old_name_matches_door,
        test_default_priors_still_prefer_old_names,
        test_wait_zero_still_copies,
    ]
    for fn in fns:
        if fn.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
    print("ok")
