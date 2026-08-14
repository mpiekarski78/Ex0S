"""TM.0.6.0: first English life — tiny corpus, bind a page word from the event."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm052 import live_free
from experiments.run_tm054 import make as make054
from experiments.run_tm060 import (
    _n_paragraphs,
    _rare_words,
    _s_has_did,
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


def test_wiki_is_english_tiny():
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
    a_toks = prose_tokens(wiki_prose(include_a=True)[-1][1])
    c_toks = prose_tokens(wiki_prose(include_c=True)[-1][1])
    assert "push" in a_toks and "adjust" in c_toks
    assert "krypton" not in a_toks and "helium" not in c_toks
    rare = _rare_words(wiki_prose(include_a=True))
    assert not any(rare[n] for n in rare if n.startswith("c"))
    assert "push" in set(rare["p99.md"])


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


def test_forced_bind_a_press_c_hold(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=40)
    assert live["opened"]
    tag = live["tag"]
    assert "push" in tag and "cha" in tag
    assert _s_has_did(tag, "press")
    assert "w" in tag
    from experiments.run_tm052 import _s_has_token

    assert _s_has_token(tag, "push")
    assert not _s_has_token(tag, "press")
    a.world = None
    a.reset_rho()
    act_a, meta = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_a == int(DialAction.PRESS)
    assert not meta.get("explored")
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_c == int(DialAction.HOLD)


def test_bind_off_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(
        s,
        w,
        _forced_policy(),
        explore_epsilon=1.0,
        rng=np.random.default_rng(0),
        use_alias_bind=False,
        use_did_stamp=True,
    )
    live = live_free(a, "experience_channel_a", 3, max_steps=40)
    assert live["opened"]
    assert _s_has_did(live["tag"], "press")
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)


def test_c_life_tune_then_a_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_c=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_c", 3, max_steps=40)
    assert live["opened"]
    tag = live["tag"]
    from experiments.run_tm052 import _s_has_token

    assert _s_has_token(tag, "adjust")
    assert _s_has_did(tag, "tune")
    assert not _s_has_token(tag, "tune")
    a.world = None
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_c == int(DialAction.TUNE)
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)


def test_copy_only_presses_on_c(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(
        s,
        w,
        _forced_policy(),
        explore_epsilon=1.0,
        rng=np.random.default_rng(0),
        use_here_match=False,
    )
    live_free(a, "experience_channel_a", 3, max_steps=40)
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_a == int(DialAction.PRESS)
    assert act_c == int(DialAction.PRESS)


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm060.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_alias_bind" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    assert "Working motor was press" not in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_here_match and not a.use_event_annotate
    assert not a.use_alias_bind and not a.use_did_stamp
    b = make054(Path("/tmp/tm060_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_alias_bind and not b.use_did_stamp
    assert not b.use_revise_head and not b.use_commit_here_only


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"adjust"' not in src and "'adjust'" not in src


if __name__ == "__main__":
    import tempfile

    test_wiki_is_english_tiny()
    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_untrained_holds_on_a_and_c,
        test_forced_bind_a_press_c_hold,
        test_bind_off_holds,
        test_c_life_tune_then_a_holds,
        test_copy_only_presses_on_c,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
