"""TM.0.4.0: channel dial — leave the door world; prose free life."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm040 import live_free, make, wiki_prose
from three_memory import agent as agent_mod
from three_memory.dial_env import CH_A, CH_C, CORRECT, ChannelDialWorld, DialAction
from three_memory.env import KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import ProseLibrary, extract_prose_ints, prose_to_record, write_prose_notes


def test_no_door_world_in_tm040():
    src = Path(REPO_ROOT / "experiments" / "run_tm040.py").read_text(encoding="utf-8")
    assert "KeyDoorWorld" not in src
    assert "probe_red" not in src
    assert "ChannelDialWorld" in src
    # confound documentation may mention door_world
    assert "channel_dial" in src


def test_agent_still_defaults_door():
    p = UsePolicy(seed=1)
    a = agent_mod.ThreeMemoryAgent(use_policy=p, store_enabled=False, cortex_seed=1337)
    assert a.domain == "door"
    assert a.n_actions == 4


def test_dial_place_copy_trap():
    assert CH_C == int(DialAction.PRESS)
    assert CORRECT[CH_A] == DialAction.PRESS
    assert CORRECT[CH_C] == DialAction.TUNE
    assert CORRECT[CH_C] != CH_C


def test_wiki_is_pure_prose_dial():
    notes = wiki_prose(include_a=True, include_c=True)
    names = {n[0] for n in notes}
    assert "p99.md" in names and "p98.md" in names
    for name, body in notes:
        assert name.endswith(".md")
        assert "where=" not in body
        assert "action=" not in body
        assert "loc=" not in body
        assert "door=" not in body
        assert "key" not in body.lower()
        assert "\nhere=" not in body
    a_body = dict(notes)["p99.md"]
    assert CH_A in extract_prose_ints(a_body)
    assert int(DialAction.PRESS) in extract_prose_ints(a_body)


def test_empty_s_prior_is_hold_not_press(tmp_path: Path):
    policy = UsePolicy(seed=7)
    a = make(tmp_path / "s", None, policy, explore_epsilon=0.0)
    assert a.domain == "dial" and a.n_actions == 5
    world = ChannelDialWorld(seed=0)
    obs = world.reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.HOLD)
    assert action != int(DialAction.PRESS)


def test_prefer_rare_prose_dial_life(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_vname = np.array([-2.0, 0.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    assert a.use_prose_ints is True
    assert isinstance(a.world, ProseLibrary)
    live = live_free(a, "experience_channel_a", 3, max_steps=8)
    assert live["found_a_pair"]
    a.world = None
    a.reset_rho()
    world = ChannelDialWorld(seed=3)
    obs = world.reset("probe_channel_a")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.PRESS)
    assert not meta.get("explored")


def test_held_out_c_tune(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_c=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_vname = np.array([-2.0, 0.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(1))
    live = live_free(a, "experience_channel_c", 4, max_steps=8)
    tag_vals = set()
    for ln in live["tag"].splitlines():
        if "=" in ln and not ln.startswith("#"):
            k, _, v = ln.partition("=")
            if k.strip().startswith("n"):
                tag_vals.add(int(v.strip()))
    assert CH_C in tag_vals and int(DialAction.TUNE) in tag_vals
    a.world = None
    a.reset_rho()
    world = ChannelDialWorld(seed=4)
    obs = world.reset("probe_channel_c")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.TUNE)
    assert action != CH_C  # not place-copy


def test_door_world_untouched():
    w = KeyDoorWorld(0)
    obs = w.reset("probe_red_with_key")
    assert obs.at_red_door


def test_agent_source_keeps_door_and_dial():
    src = inspect.getsource(agent_mod)
    assert "domain" in src
    assert "DialAction" in src
    assert "Action.OPEN" in src


if __name__ == "__main__":
    import tempfile

    test_no_door_world_in_tm040()
    test_agent_still_defaults_door()
    test_dial_place_copy_trap()
    test_wiki_is_pure_prose_dial()
    test_door_world_untouched()
    test_agent_source_keeps_door_and_dial()
    fns = [
        test_empty_s_prior_is_hold_not_press,
        test_prefer_rare_prose_dial_life,
        test_held_out_c_tune,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
