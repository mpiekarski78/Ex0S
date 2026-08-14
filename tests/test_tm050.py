"""TM.0.5.0: no answer integers — copy innate motor name token."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm050 import live_free, make, wiki_prose
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import extract_prose_ints, prose_to_record, write_prose_notes


def test_no_digits_in_wiki():
    notes = wiki_prose(include_a=True, include_c=True)
    for name, body in notes:
        assert name.endswith(".md")
        assert not extract_prose_ints(body)
        assert "where=" not in body
        assert "action=" not in body
    a_body = dict(notes)["p99.md"]
    c_body = dict(notes)["p98.md"]
    assert "press" in a_body and "tune" not in a_body
    assert "tune" in c_body and "press" not in c_body


def test_prose_to_record_tokens_without_ints(tmp_path: Path):
    write_prose_notes(tmp_path, [("p99.md", "Krypton scrap. Working motor was press.\n")])
    rec = prose_to_record(tmp_path / "p99.md")
    assert rec is not None
    assert not any(k.startswith("n") for k in rec.tags)
    vals = [rec.tags[k] for k in sorted(rec.tags) if k.startswith("w")]
    assert "press" in vals
    assert "krypton" in vals


def test_empty_s_is_hold(tmp_path: Path):
    policy = UsePolicy(seed=7)
    a = make(tmp_path / "s", None, policy, explore_epsilon=0.0)
    assert a.use_prose_tokens and not a.use_prose_ints
    obs = ChannelDialWorld(seed=0).reset("probe_channel_a")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.HOLD)


def test_prefer_rare_token_life(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)  # prefer is_act over is_common
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=8)
    assert live["found_press"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.PRESS)
    assert not meta.get("explored")


def test_held_out_tune(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_c=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(1))
    live = live_free(a, "experience_channel_c", 4, max_steps=8)
    assert live["found_tune"]
    a.world = None
    a.reset_rho()
    obs = ChannelDialWorld(seed=4).reset("probe_channel_c")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == int(DialAction.TUNE)


def test_no_door_world_in_tm050():
    src = Path(REPO_ROOT / "experiments" / "run_tm050.py").read_text(encoding="utf-8")
    assert "KeyDoorWorld" not in src
    assert "probe_red" not in src


def test_agent_default_still_door():
    p = UsePolicy(seed=1)
    a = agent_mod.ThreeMemoryAgent(use_policy=p, store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_prose_tokens


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src


if __name__ == "__main__":
    import tempfile

    test_no_digits_in_wiki()
    test_no_door_world_in_tm050()
    test_agent_default_still_door()
    test_agent_source_no_synonym_table()
    fns = [
        test_prose_to_record_tokens_without_ints,
        test_empty_s_is_hold,
        test_prefer_rare_token_life,
        test_held_out_tune,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
