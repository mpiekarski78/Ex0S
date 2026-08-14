"""TM.0.5.2: unread W does not name the motor; stamp did= from the event."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm052 import clutter_prose, live_free, make, wiki_prose
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import extract_prose_ints, prose_to_record, prose_tokens, write_prose_notes

_MOTOR = {a.name.lower() for a in DialAction}


def _forced_policy() -> UsePolicy:
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_write = np.array([3.0, 3.0], dtype=np.float64)
    policy.b_write = np.array(3.0, dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    return policy


def test_wiki_has_no_motor_names():
    notes = wiki_prose(include_a=True, include_c=True)
    for name, body in notes:
        assert name.endswith(".md")
        assert not extract_prose_ints(body)
        assert not (prose_tokens(body) & _MOTOR)
        assert "where=" not in body
        assert "action=" not in body
        assert "press" not in body.lower()
        assert "tune" not in body.lower()
    a_body = dict(notes)["p99.md"]
    c_body = dict(notes)["p98.md"]
    assert "krypton" in a_body.lower()
    assert "helium" in c_body.lower()


def test_prose_to_record_no_motor_token(tmp_path: Path):
    write_prose_notes(tmp_path, [wiki_prose(include_a=True)[-1]])
    rec = prose_to_record(tmp_path / "p99.md")
    assert rec is not None
    assert not any(k.startswith("n") for k in rec.tags)
    vals = {str(rec.tags[k]).lower() for k in rec.tags if str(k).startswith("w")}
    assert "krypton" in vals
    assert not (vals & _MOTOR)


def test_untrained_does_not_annotate(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = UsePolicy(seed=7)
    a = make(s, w, policy, explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=32)
    assert live["n_annotated"] == 0
    assert not live["found_press"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.HOLD)


def test_forced_annotate_stamps_press(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=40)
    assert live["opened"]
    assert live["n_annotated"] >= 1
    assert live["found_press"]
    assert live["found_krypton"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.PRESS)
    assert not meta.get("explored")


def test_clutter_only_does_not_stamp(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, clutter_prose())
    a = make(s, w, _forced_policy(), force_write=True, explore_epsilon=1.0, rng=np.random.default_rng(1))
    live = live_free(a, "experience_channel_a", 4, max_steps=40)
    assert live["n_annotated"] == 0
    assert not live["found_press"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=4).reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action != int(DialAction.PRESS)


def test_annotate_off_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(
        s,
        w,
        _forced_policy(),
        use_event_annotate=False,
        explore_epsilon=1.0,
        rng=np.random.default_rng(2),
    )
    live = live_free(a, "experience_channel_a", 5, max_steps=40)
    assert live["n_annotated"] == 0
    assert not live["found_press"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=5).reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action != int(DialAction.PRESS)


def test_held_out_tune(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_c=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(3))
    live = live_free(a, "experience_channel_c", 6, max_steps=40)
    assert live["found_tune"]
    assert live["found_helium"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=6).reset("probe_channel_c")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.TUNE)


def test_no_door_world_in_tm052():
    src = Path(REPO_ROOT / "experiments" / "run_tm052.py").read_text(encoding="utf-8")
    assert "KeyDoorWorld" not in src
    assert "probe_red" not in src
    assert "use_event_annotate" in src
    assert "Working motor was press" not in src
    assert "Working motor was tune" not in src


def test_agent_default_does_not_annotate():
    p = UsePolicy(seed=1)
    a = agent_mod.ThreeMemoryAgent(use_policy=p, store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_event_annotate and not a.use_prose_tokens


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src


if __name__ == "__main__":
    import tempfile

    test_wiki_has_no_motor_names()
    test_no_door_world_in_tm052()
    test_agent_default_does_not_annotate()
    test_agent_source_no_synonym_table()
    fns = [
        test_prose_to_record_no_motor_token,
        test_untrained_does_not_annotate,
        test_forced_annotate_stamps_press,
        test_clutter_only_does_not_stamp,
        test_annotate_off_holds,
        test_held_out_tune,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
