"""TM.0.6.8: find the unread page that adds the most rare tokens S lacks."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm059 import make as make059
from experiments.run_tm060 import make as make060
from experiments.run_tm061 import make as make061
from experiments.run_tm062 import make as make062
from experiments.run_tm063 import make as make063
from experiments.run_tm064 import make as make064
from experiments.run_tm064 import wiki_prose
from experiments.run_tm065 import make as make065
from experiments.run_tm066 import make as make066
from experiments.run_tm067 import make as make067
from experiments.run_tm068 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_empty_s_keeps_two_rare_pages(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    pool = list(a.world.records())
    kept = {getattr(r, "fact_id", None) for r in a._filter_find_novel(pool)}
    assert "p99" in kept and "p98" in kept
    assert "c08" not in kept and "c09" not in kept and "c10" not in kept


def test_after_p99_keeps_p98_not_clutter(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["p99"])
    a._in_hand_id = "p99"
    obs_a = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs_a)
    owned = {r.fact_id for r in a.store.records()}
    pool = [r for r in a.world.records() if getattr(r, "fact_id", None) not in owned]
    kept = {getattr(r, "fact_id", None) for r in a._filter_find_novel(pool)}
    assert kept == {"p98"}


def test_c_life_in_hand_is_p98_not_neon(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["p99"])
    a._in_hand_id = "p99"
    obs_a = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs_a)
    obs_c = ChannelDialWorld(seed=3).reset("experience_channel_c")
    a.collect(obs_c, record=False)
    assert a._in_hand_id == "p98"
    a._maybe_annotate("tune", obs_c)
    notes = {r.fact_id: r for r in a.store.records()}
    assert "p98" in notes
    assert str(notes["p98"].tags.get("bind") or "").lower() == "adjust"
    binds = [str(r.tags.get("bind") or "").lower() for r in a.store.records() if r.tags.get("bind")]
    assert "neon" not in binds and "xenon" not in binds


def test_067_still_keeps_clutter_hapax(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make067(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert not a.use_find_novel
    pool = list(a.world.records())
    kept = {getattr(r, "fact_id", None) for r in a._filter_find_novel(pool)}
    assert "c08" in kept and "p99" in kept


def test_c_life_does_not_stamp_leftover_after_find(tmp_path: Path):
    import numpy as np
    from experiments.run_tm052 import live_free

    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    policy = UsePolicy(seed=7)
    a = make(s, w, policy, force_write=True, explore_epsilon=0.0)
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["p99"])
    a._in_hand_id = "p99"
    obs_a = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs_a)
    a2 = make(s, w, policy, force_write=True, explore_epsilon=0.5, rng=np.random.default_rng(3))
    live = live_free(a2, "experience_channel_c", 23, max_steps=32)
    binds = [str(r.tags.get("bind") or "").lower() for r in a2.store.records() if r.tags.get("bind")]
    assert "adjust" in binds
    assert "xenon" not in binds and "neon" not in binds and "krypton" not in binds
    assert live["n_annotated"] >= 1


def test_untrained_holds(tmp_path: Path):
    import numpy as np
    from experiments.run_tm052 import live_free

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


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm068.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_find_novel" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_find_novel
    b = make054(Path("/tmp/tm068_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_find_novel
    c = make060(Path("/tmp/tm068_skip_060"), None, UsePolicy(seed=1), enabled=False)
    assert not c.use_find_novel
    d = make061(Path("/tmp/tm068_skip_061"), None, UsePolicy(seed=1), enabled=False)
    assert d.use_one_bind and not d.use_find_novel
    e = make062(Path("/tmp/tm068_skip_062"), None, UsePolicy(seed=1), enabled=False)
    assert not e.use_find_novel
    f = make063(Path("/tmp/tm068_skip_063"), None, UsePolicy(seed=1), enabled=False)
    assert f.use_stamp_new_here and not f.use_find_novel
    g = make064(Path("/tmp/tm068_skip_064"), None, UsePolicy(seed=1), enabled=False)
    assert not g.use_find_novel
    h = make065(Path("/tmp/tm068_skip_065"), None, UsePolicy(seed=1), enabled=False)
    assert h.use_block_here and not h.use_find_novel
    i = make066(Path("/tmp/tm068_skip_066"), None, UsePolicy(seed=1), enabled=False)
    assert i.use_revise_head and not i.use_find_novel
    j = make067(Path("/tmp/tm068_skip_067"), None, UsePolicy(seed=1), enabled=False)
    assert j.use_in_hand_new_here and not j.use_find_novel
    k = make(Path("/tmp/tm068_skip_068"), None, UsePolicy(seed=1), enabled=False)
    assert k.use_find_novel and k.use_in_hand_new_here and k.use_block_here
    m = make059(Path("/tmp/tm068_skip_059"), None, UsePolicy(seed=1), enabled=False)
    assert m.use_revise_head and not m.use_find_novel
    copy = make(Path("/tmp/tm068_copy"), None, UsePolicy(seed=1), enabled=False, use_here_match=False)
    assert copy.use_find_novel and not copy.use_in_hand_new_here


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"xenon"' not in src and "'xenon'" not in src
    assert '"adjust"' not in src and "'adjust'" not in src
    assert '"p98"' not in src and "'p98'" not in src


if __name__ == "__main__":
    import tempfile

    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_empty_s_keeps_two_rare_pages,
        test_after_p99_keeps_p98_not_clutter,
        test_c_life_in_hand_is_p98_not_neon,
        test_c_life_does_not_stamp_leftover_after_find,
        test_067_still_keeps_clutter_hapax,
        test_untrained_holds,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
