"""TM.0.7.1: file-local bind→did. No global hapax lexicon."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm060 import make as make060
from experiments.run_tm068 import make as make068
from experiments.run_tm069 import wiki_prose
from experiments.run_tm070 import make as make070
from experiments.run_tm071 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_local_alias_does_not_export_other_binds(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_local_alias
    a._in_hand_id = "c08"
    a._maybe_annotate("press", ChannelDialWorld(seed=3).reset("probe_channel_a"))
    a._in_hand_id = "p99"
    a._maybe_annotate("press", ChannelDialWorld(seed=3).reset("probe_channel_a"))
    notes = {r.fact_id: r for r in a.store.records()}
    global_map = a._act_map()
    assert "xenon" not in global_map and "push" not in global_map
    assert "xenon" in a._act_map([notes["c08"]])
    assert "push" in a._act_map([notes["p99"]])
    assert "push" not in a._act_map([notes["c08"]])


def test_070_keeps_global_alias(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make070(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_retry_novel and not a.use_local_alias
    by_id = {r.fact_id: r for r in a.world.records()}
    a._commit_w_record(by_id["c08"])
    a._in_hand_id = "c08"
    a._maybe_annotate("press", ChannelDialWorld(seed=3).reset("probe_channel_a"))
    assert "xenon" in a._act_map()


def test_070_confounds_local_alias_smuggle():
    import experiments.run_tm070 as tm070

    saved = tm070._classify_common069
    tm070._classify_common069 = lambda m: None
    try:
        label, why = tm070.classify_common(
            {"use_retry_novel": True, "retry_novel": True, "use_local_alias": True}
        )
    finally:
        tm070._classify_common069 = saved
    assert label == "Confound"
    assert "local" in why.lower()


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
    assert act_a == int(DialAction.HOLD)


def test_skips_stay_skipped():
    src = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    assert "use_local_alias" in src
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_local_alias
    assert not make054(Path("/tmp/tm071_054"), None, UsePolicy(seed=1), enabled=False).use_local_alias
    assert not make060(Path("/tmp/tm071_060"), None, UsePolicy(seed=1), enabled=False).use_local_alias
    assert not make068(Path("/tmp/tm071_068"), None, UsePolicy(seed=1), enabled=False).use_local_alias
    assert not make070(Path("/tmp/tm071_070"), None, UsePolicy(seed=1), enabled=False).use_local_alias
    b = make(Path("/tmp/tm071_071"), None, UsePolicy(seed=1), enabled=False)
    assert b.use_local_alias and b.use_retry_novel


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"xenon"' not in src and "'xenon'" not in src


if __name__ == "__main__":
    import tempfile

    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    test_070_confounds_local_alias_smuggle()
    for fn in (test_local_alias_does_not_export_other_binds, test_070_keeps_global_alias, test_untrained_holds):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
