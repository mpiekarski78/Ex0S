"""TM.0.7.0: keep looking through the novel tie."""

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
from experiments.run_tm065 import make as make065
from experiments.run_tm066 import make as make066
from experiments.run_tm067 import make as make067
from experiments.run_tm068 import make as make068
from experiments.run_tm069 import make as make069
from experiments.run_tm069 import wiki_prose
from experiments.run_tm070 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def _stamp_a(a) -> None:
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["c09"])
    a._in_hand_id = "c09"
    obs_a = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs_a)


def test_069_here_only_locks_after_one_stamp(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make069(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_find_novel and not a.use_retry_novel
    _stamp_a(a)
    owned = {r.fact_id for r in a.store.records()}
    obs_a = ChannelDialWorld(seed=3).reset("experience_channel_a")
    a.collect(obs_a, record=False)
    assert a._in_hand_id in owned or a._in_hand_id is None or a._in_hand_id == "c09"


def test_retry_attends_another_novel_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_retry_novel and a.use_find_novel
    _stamp_a(a)
    obs_a = ChannelDialWorld(seed=3).reset("experience_channel_a")
    a.collect(obs_a, record=False)
    assert a._in_hand_id is not None
    assert a._in_hand_id != "c09"
    owned = {r.fact_id for r in a.store.records()}
    assert a._in_hand_id not in owned
    a._maybe_annotate("press", obs_a)
    ids = {r.fact_id for r in a.store.records()}
    assert "c09" in ids
    assert a._in_hand_id in ids
    assert len(ids) >= 2


def test_retry_does_not_commit_common_leftover(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    by_id = {r.fact_id: r for r in a.world.records()}
    for fid in ("c08", "c09", "c10", "p99"):
        a._commit_w_record(by_id[fid])
        a._in_hand_id = fid
        a._maybe_annotate("press", ChannelDialWorld(seed=3).reset("probe_channel_a"))
    n = len(list(a.store.records()))
    obs_a = ChannelDialWorld(seed=3).reset("experience_channel_a")
    a.collect(obs_a, record=False)
    a._maybe_annotate("press", obs_a)
    ids = {r.fact_id for r in a.store.records()}
    assert "c00" not in ids and "c01" not in ids
    assert len(ids) == n


def test_069_confounds_retry_smuggle():
    import experiments.run_tm069 as tm069

    saved = tm069._classify_common068
    tm069._classify_common068 = lambda m: None
    try:
        label, why = tm069.classify_common(
            {
                "w_clutter_has_two_rare": True,
                "w_n_two_rare_clutter": 3,
                "find_without_unique_pair": True,
                "w_useful_only_two_rare": False,
                "use_retry_novel": True,
            }
        )
    finally:
        tm069._classify_common068 = saved
    assert label == "Confound"
    assert "retry" in why.lower()


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
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm070.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_retry_novel" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_retry_novel and not a.use_find_novel
    b = make054(Path("/tmp/tm070_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_retry_novel
    c = make060(Path("/tmp/tm070_skip_060"), None, UsePolicy(seed=1), enabled=False)
    assert not c.use_retry_novel
    d = make061(Path("/tmp/tm070_skip_061"), None, UsePolicy(seed=1), enabled=False)
    assert d.use_one_bind and not d.use_retry_novel
    e = make062(Path("/tmp/tm070_skip_062"), None, UsePolicy(seed=1), enabled=False)
    assert not e.use_retry_novel
    f = make063(Path("/tmp/tm070_skip_063"), None, UsePolicy(seed=1), enabled=False)
    assert f.use_stamp_new_here and not f.use_retry_novel
    g = make064(Path("/tmp/tm070_skip_064"), None, UsePolicy(seed=1), enabled=False)
    assert not g.use_retry_novel
    h = make065(Path("/tmp/tm070_skip_065"), None, UsePolicy(seed=1), enabled=False)
    assert h.use_block_here and not h.use_retry_novel
    i = make066(Path("/tmp/tm070_skip_066"), None, UsePolicy(seed=1), enabled=False)
    assert i.use_revise_head and not i.use_retry_novel
    j = make067(Path("/tmp/tm070_skip_067"), None, UsePolicy(seed=1), enabled=False)
    assert j.use_in_hand_new_here and not j.use_retry_novel
    k = make068(Path("/tmp/tm070_skip_068"), None, UsePolicy(seed=1), enabled=False)
    assert k.use_find_novel and not k.use_retry_novel
    n = make069(Path("/tmp/tm070_skip_069"), None, UsePolicy(seed=1), enabled=False)
    assert n.use_find_novel and not n.use_retry_novel
    p = make(Path("/tmp/tm070_skip_070"), None, UsePolicy(seed=1), enabled=False)
    assert p.use_retry_novel and p.use_find_novel
    m = make059(Path("/tmp/tm070_skip_059"), None, UsePolicy(seed=1), enabled=False)
    assert m.use_revise_head and not m.use_retry_novel
    copy = make(Path("/tmp/tm070_copy"), None, UsePolicy(seed=1), enabled=False, use_find_novel=False)
    assert not copy.use_retry_novel


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"xenon"' not in src and "'xenon'" not in src
    assert '"radon"' not in src and "'radon'" not in src
    assert '"adjust"' not in src and "'adjust'" not in src
    assert '"p98"' not in src and "'p98'" not in src


if __name__ == "__main__":
    import tempfile

    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    test_069_confounds_retry_smuggle()
    fns = [
        test_069_here_only_locks_after_one_stamp,
        test_retry_attends_another_novel_page,
        test_retry_does_not_commit_common_leftover,
        test_untrained_holds,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
