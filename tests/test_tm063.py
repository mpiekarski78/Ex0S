"""TM.0.6.3: a new station gets an unmarked rare page."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm060 import make as make060
from experiments.run_tm061 import (
    _n_paragraphs,
    _rare_words,
    _s_has_bind,
    _s_has_did,
    _write_nonce_s,
    clutter_prose,
    make as make061,
    wiki_prose,
)
from experiments.run_tm062 import make as make062
from experiments.run_tm062 import _copy_s, _c_life_on_s, _probe_s, _s_snapshot, _train_keep
from experiments.run_tm063 import make
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
    stream = prose_token_stream(a_body)
    assert stream.index("push") < stream.index("argon")
    rare = _rare_words(wiki_prose(include_a=True))
    assert set(rare["p99.md"]) >= {"push", "argon"}


def test_untrained_holds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=1.0, rng=np.random.default_rng(0))
    from experiments.run_tm052 import live_free

    live = live_free(a, "experience_channel_a", 3, max_steps=32)
    assert live["n_annotated"] == 0
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)
    assert act_c == int(DialAction.HOLD)


def test_c_life_on_dirty_train_s(tmp_path: Path):
    w_a = tmp_path / "Wa"
    w_both = tmp_path / "Wb"
    work = tmp_path / "work"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    policy = _forced_policy()
    import experiments.run_tm062 as tm062

    saved = tm062.make
    tm062.make = make
    try:
        _train_keep(policy, w_a, work, n=2, seed=3, split=True, max_steps=40)
        acc = tmp_path / "acc"
        _copy_s(work / "ep", acc)
        before = _s_snapshot(acc)
        assert before["found_bind_push"] and before["found_cha"]
        p_a, p_c = _probe_s(acc, policy, 3)
        assert p_a["action_name"] == "press"
        assert p_c["action_name"] == "hold"
        _, c_live, both_a, both_c, tag = _c_life_on_s(
            acc,
            w_both,
            policy,
            6,
            max_steps=40,
            explore_epsilon=1.0,
            rng=np.random.default_rng(3),
        )
    finally:
        tm062.make = saved
    assert c_live["found_bind_adjust"] and c_live["found_chc"] and c_live["found_did_tune"]
    assert c_live["found_alpha"] and not c_live["found_bind_alpha"]
    assert c_live["found_bind_push"] and c_live["found_cha"]
    assert both_a["action_name"] == "press"
    assert both_c["action_name"] == "tune"
    assert _s_has_bind(tag, "adjust") and not _s_has_bind(tag, "alpha")
    assert _s_has_did(tag, "press") and _s_has_did(tag, "tune")
    nonce = tmp_path / "nonce"
    _write_nonce_s(work / "ep", nonce, nonce="argon", station="cha")
    n = make(nonce, None, policy, explore_epsilon=0.0)
    n.reset_rho()
    act_n, _ = n.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act_n == int(DialAction.HOLD)


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm063.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_stamp_new_here" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_stamp_new_here and not a.use_one_bind
    assert not a.use_commit_rare_only and not a.use_revise_head
    b = make054(Path("/tmp/tm063_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_stamp_new_here and not b.use_one_bind
    c = make060(Path("/tmp/tm063_skip_060"), None, UsePolicy(seed=1), enabled=False)
    assert not c.use_stamp_new_here and not c.use_one_bind
    d = make061(Path("/tmp/tm063_skip_061"), None, UsePolicy(seed=1), enabled=False)
    assert d.use_one_bind and not d.use_stamp_new_here
    e = make062(Path("/tmp/tm063_skip_062"), None, UsePolicy(seed=1), enabled=False)
    assert e.use_one_bind and e.use_commit_rare_only and not e.use_stamp_new_here
    f = make(Path("/tmp/tm063_skip_063"), None, UsePolicy(seed=1), enabled=False)
    assert f.use_stamp_new_here and f.use_one_bind and not f.use_revise_head


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
    fns = [test_untrained_holds, test_c_life_on_dirty_train_s]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
