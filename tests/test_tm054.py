"""TM.0.5.4: Open W — document-shaped unread pages, distinct clutter."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm052 import live_free
from experiments.run_tm054 import (
    _n_paragraphs,
    _rare_words,
    clutter_prose,
    make,
    wiki_prose,
)
from three_memory import agent as agent_mod
from three_memory.dial_env import STATION_NAMES, ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import extract_prose_ints, prose_tokens, write_prose_notes

_MOTOR = {a.name.lower() for a in DialAction}
_STATIONS = set(STATION_NAMES.values())


def _forced_policy() -> UsePolicy:
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_write = np.array([3.0, 3.0], dtype=np.float64)
    policy.b_write = np.array(3.0, dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    return policy


def test_wiki_is_open_documents():
    notes = wiki_prose(include_a=True, include_c=True)
    clutter = clutter_prose()
    bodies = [b for _, b in clutter]
    assert len(clutter) >= 11
    assert len(set(bodies)) == len(bodies)
    for _, body in notes:
        assert _n_paragraphs(body) >= 2
        toks = prose_tokens(body)
        assert not extract_prose_ints(body)
        assert not (toks & _MOTOR)
        assert not (toks & _STATIONS)
    rare = _rare_words(wiki_prose(include_a=True))
    assert not any(rare[n] for n in rare if n.startswith("c"))
    assert "krypton" in set(rare["p99.md"])
    headed = [(n, f"# {Path(n).stem}\n\n{b}") for n, b in wiki_prose(include_a=True)]
    headed_rare = _rare_words(headed)
    assert not any(headed_rare[n] for n in headed_rare if n.startswith("c"))
    assert headed_rare.get("p99.md")


def test_untrained_holds_on_a_and_c(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=32)
    assert live["n_annotated"] == 0
    a.world = None
    a.reset_rho()
    obs_a = ChannelDialWorld(seed=3).reset("probe_channel_a")
    act_a, _ = a.act(obs_a, update_rho=False, explore=False)
    a.reset_rho()
    obs_c = ChannelDialWorld(seed=3).reset("probe_channel_c")
    act_c, _ = a.act(obs_c, update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)
    assert act_c == int(DialAction.HOLD)


def test_forced_here_match_a_press_c_hold(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=40)
    assert live["opened"]
    tag = live["tag"]
    assert "press" in tag and "cha" in tag
    a.world = None
    a.reset_rho()
    act_a, meta = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_a == int(DialAction.PRESS)
    assert not meta.get("explored")
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_c == int(DialAction.HOLD)


def test_copy_only_presses_on_c(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(
        s, w, _forced_policy(), use_here_match=False, explore_epsilon=1.0, rng=np.random.default_rng(0)
    )
    live_free(a, "experience_channel_a", 3, max_steps=40)
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_a == int(DialAction.PRESS)
    assert act_c == int(DialAction.PRESS)


def test_c_life_tune_then_a_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_c=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(3))
    live = live_free(a, "experience_channel_c", 6, max_steps=40)
    assert "tune" in live["tag"] and "chc" in live["tag"]
    a.world = None
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=6).reset("probe_channel_c"), update_rho=False, explore=False)
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=6).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_c == int(DialAction.TUNE)
    assert act_a == int(DialAction.HOLD)


def test_clutter_only_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, clutter_prose())
    a = make(s, w, _forced_policy(), force_write=True, explore_epsilon=1.0, rng=np.random.default_rng(1))
    live_free(a, "experience_channel_a", 4, max_steps=40)
    a.world = None
    a.reset_rho()
    act, _ = a.act(ChannelDialWorld(seed=4).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act != int(DialAction.PRESS)


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm054.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "domain=\"dial\"" in src_exp or "domain='dial'" in src_exp
    assert "KeyDoorWorld" not in src_exp
    assert "Working motor was press" not in src_exp
    assert "shutil.rmtree(s_dir)" in src_exp
    assert 'flags["open_w"] = True' in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_here_match and not a.use_event_annotate


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src


if __name__ == "__main__":
    import tempfile

    test_wiki_is_open_documents()
    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_untrained_holds_on_a_and_c,
        test_forced_here_match_a_press_c_hold,
        test_copy_only_presses_on_c,
        test_c_life_tune_then_a_holds,
        test_clutter_only_holds,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
