"""TM.0.6.1: one bind per note — distractor hapax must not fire the motor."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm052 import _s_has_token, live_free
from experiments.run_tm054 import make as make054
from experiments.run_tm060 import make as make060
from experiments.run_tm061 import (
    _n_paragraphs,
    _rare_words,
    _s_has_bind,
    _s_has_did,
    _write_nonce_s,
    clutter_prose,
    make,
    wiki_prose,
)
from three_memory import agent as agent_mod
from three_memory.dial_env import STATION_NAMES, ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import extract_prose_ints, prose_token_stream, prose_tokens, write_prose_notes

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


def test_wiki_has_two_rares_document_order():
    clutter = clutter_prose()
    assert len(set(b for _, b in clutter)) == len(clutter) >= 11
    a_body = wiki_prose(include_a=True)[-1][1]
    c_body = wiki_prose(include_c=True)[-1][1]
    for body in (a_body, c_body):
        assert _n_paragraphs(body) >= 2
        toks = prose_tokens(body)
        assert not extract_prose_ints(body)
        assert not (toks & _MOTOR)
        assert not (toks & _STATIONS)
    assert "push" in prose_tokens(a_body) and "argon" in prose_tokens(a_body)
    assert "adjust" in prose_tokens(c_body) and "alpha" in prose_tokens(c_body)
    stream = prose_token_stream(a_body)
    assert stream.index("push") < stream.index("argon")
    assert prose_token_stream(c_body).index("adjust") < prose_token_stream(c_body).index("alpha")
    rare = _rare_words(wiki_prose(include_a=True))
    assert not any(rare[n] for n in rare if n.startswith("c"))
    assert set(rare["p99.md"]) >= {"push", "argon"}
    assert len(rare["p99.md"]) >= 2


def test_untrained_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=32)
    assert live["n_annotated"] == 0
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)
    assert act_c == int(DialAction.HOLD)


def test_forced_one_bind_press_argon_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=40)
    assert live["opened"]
    tag = live["tag"]
    assert _s_has_token(tag, "push") and _s_has_token(tag, "argon") and _s_has_token(tag, "cha")
    assert _s_has_did(tag, "press")
    assert _s_has_bind(tag, "push")
    assert not _s_has_bind(tag, "argon")
    assert not _s_has_token(tag, "press")
    a.world = None
    a.reset_rho()
    act_a, meta = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_a == int(DialAction.PRESS) and not meta.get("explored")
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_c == int(DialAction.HOLD)
    nonce = tmp_path / "nonce"
    _write_nonce_s(s, nonce, nonce="argon", station="cha")
    n = make(nonce, None, _forced_policy(), explore_epsilon=0.0)
    n.reset_rho()
    act_n, _ = n.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_n == int(DialAction.HOLD)
    b = make(nonce, None, _forced_policy(), explore_epsilon=0.0, use_one_bind=False)
    b.reset_rho()
    act_b, _ = b.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_b == int(DialAction.PRESS)


def test_c_life_tune(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_c=True))
    a = make(s, w, _forced_policy(), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_c", 3, max_steps=40)
    tag = live["tag"]
    assert _s_has_token(tag, "adjust") and _s_has_token(tag, "alpha")
    assert _s_has_bind(tag, "adjust") and not _s_has_bind(tag, "alpha")
    a.world = None
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_c == int(DialAction.TUNE)
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm061.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_one_bind" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_one_bind and not a.use_alias_bind
    b = make054(Path("/tmp/tm061_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_one_bind and not b.use_alias_bind
    c = make060(Path("/tmp/tm061_skip_060"), None, UsePolicy(seed=1), enabled=False)
    assert not c.use_one_bind and c.use_alias_bind


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"argon"' not in src and "'argon'" not in src
    assert '"adjust"' not in src and "'adjust'" not in src
    assert '"alpha"' not in src and "'alpha'" not in src


if __name__ == "__main__":
    import tempfile

    test_wiki_has_two_rares_document_order()
    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_untrained_holds,
        test_forced_one_bind_press_argon_holds,
        test_c_life_tune,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
