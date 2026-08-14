"""TM.0.5.1: correct a wrong commit — drop junk S, retry, keep after ρ reset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm050 import wiki_prose
from experiments.run_tm051 import live_free, make
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_untrained_does_not_revise(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = UsePolicy(seed=7)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=8)
    assert live["n_revised"] == 0
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.HOLD)


def test_forced_revise_walks_to_press(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = UsePolicy(seed=7)
    # Fail + no act-name → revise. Keep files that already name an act.
    policy.w_revise = np.array([3.0, -3.0], dtype=np.float64)
    policy.b_revise = np.array(-1.0, dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=20)
    assert live["n_revised"] >= 1
    assert live["found_press"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.PRESS)
    assert not meta.get("explored")


def test_revise_off_stays_on_clutter(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = UsePolicy(seed=7)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, use_revise_head=False, explore_epsilon=0.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=20)
    assert live["n_revised"] == 0
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action != int(DialAction.PRESS)


def test_no_door_in_tm051():
    src = Path(REPO_ROOT / "experiments" / "run_tm051.py").read_text(encoding="utf-8")
    assert "KeyDoorWorld" not in src
    assert "probe_red" not in src
    assert "use_revise_head" in src


def test_default_agent_does_not_revise():
    from three_memory.agent import ThreeMemoryAgent

    a = ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.use_revise_head is False


if __name__ == "__main__":
    import tempfile

    test_no_door_in_tm051()
    test_default_agent_does_not_revise()
    fns = [
        test_untrained_does_not_revise,
        test_forced_revise_walks_to_press,
        test_revise_off_stays_on_clutter,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
