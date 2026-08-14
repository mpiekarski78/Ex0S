"""TM.0.6.6: correct the dirty English store."""

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
from experiments.run_tm066 import MAX_TRAIN_S_FILES, make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_sweep_drops_unstamped_after_bind(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["c08"])
    a._commit_w_record(by_id["p99"])
    a._in_hand_id = "p99"
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs)
    ids = {r.fact_id for r in a.store.records()}
    assert "p99" in ids
    assert "c08" not in ids
    assert a.n_revised >= 1
    binds = [str(r.tags.get("bind") or "").lower() for r in a.store.records() if r.tags.get("bind")]
    assert binds == ["push"]


def test_new_here_does_not_stamp_a_common_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["p99"])
    a._in_hand_id = "p99"
    obs_a = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs_a)
    a._commit_w_record(by_id["c00"])
    a._in_hand_id = "c00"
    obs_c = ChannelDialWorld(seed=3).reset("probe_channel_c")
    a._maybe_annotate("tune", obs_c)
    ids = {r.fact_id for r in a.store.records()}
    assert "p99" in ids
    assert "c00" not in ids or not any(
        str(r.tags.get("did") or "").lower() == "tune" and r.fact_id == "c00" for r in a.store.records()
    )
    c_notes = [r for r in a.store.records() if str(r.tags.get("did") or "").lower() == "tune"]
    assert c_notes
    assert all(a._is_rare_in_world(r) for r in c_notes)


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
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm066.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_revise_head" in src_agent
    assert "use_commit_here_only" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    assert MAX_TRAIN_S_FILES == 4
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_revise_head and not a.use_commit_here_only
    assert not a.use_block_here and not a.use_in_hand_new_here
    b = make054(Path("/tmp/tm066_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_revise_head and not b.use_commit_here_only
    c = make060(Path("/tmp/tm066_skip_060"), None, UsePolicy(seed=1), enabled=False)
    assert not c.use_revise_head
    d = make061(Path("/tmp/tm066_skip_061"), None, UsePolicy(seed=1), enabled=False)
    assert d.use_one_bind and not d.use_revise_head
    e = make062(Path("/tmp/tm066_skip_062"), None, UsePolicy(seed=1), enabled=False)
    assert not e.use_revise_head and not e.use_commit_here_only and not e.use_block_here
    f = make063(Path("/tmp/tm066_skip_063"), None, UsePolicy(seed=1), enabled=False)
    assert f.use_stamp_new_here and not f.use_revise_head
    g = make064(Path("/tmp/tm066_skip_064"), None, UsePolicy(seed=1), enabled=False)
    assert not g.use_revise_head and not g.use_block_here
    h = make065(Path("/tmp/tm066_skip_065"), None, UsePolicy(seed=1), enabled=False)
    assert h.use_block_here and not h.use_revise_head and not h.use_commit_here_only
    i = make(Path("/tmp/tm066_skip_066"), None, UsePolicy(seed=1), enabled=False)
    assert i.use_revise_head and i.use_commit_here_only and i.use_block_here
    assert not i.use_in_hand_new_here
    j = make059(Path("/tmp/tm066_skip_059"), None, UsePolicy(seed=1), enabled=False)
    assert j.use_revise_head and j.use_commit_here_only and not j.use_block_here
    copy = make(Path("/tmp/tm066_copy"), None, UsePolicy(seed=1), enabled=False, use_here_match=False)
    assert not copy.use_revise_head and not copy.use_commit_here_only


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"xenon"' not in src and "'xenon'" not in src


if __name__ == "__main__":
    import tempfile

    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [test_sweep_drops_unstamped_after_bind, test_new_here_does_not_stamp_a_common_page, test_untrained_holds]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
