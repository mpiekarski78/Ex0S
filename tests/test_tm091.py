"""TM.0.9.1: keep untested competing hypotheses."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm080 import wiki_prose
from experiments.run_tm090 import make as make090
from experiments.run_tm091 import classify_a, make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_090_confounds_hyp_survive_smuggle():
    import experiments.run_tm090 as tm090

    saved = tm090._classify_common082
    tm090._classify_common082 = lambda m: None
    try:
        label, why = tm090.classify_common(
            {
                "use_count_search": True,
                "count_search": True,
                "use_hyp_survive": True,
            }
        )
    finally:
        tm090._classify_common082 = saved
    assert label == "Confound"
    assert "hyp" in why.lower()


def test_hyp_survive_on_091_off_before():
    assert not make090(Path("/tmp/tm091_090"), None, UsePolicy(seed=1), enabled=False).use_hyp_survive
    a = make(Path("/tmp/tm091_091"), None, UsePolicy(seed=1), enabled=False)
    assert a.use_hyp_survive and a.use_keep_steerer and a.use_count_search and a.n_actions == 5
    door = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert door.domain == "door" and not door.use_hyp_survive
    assert not make054(Path("/tmp/tm091_054"), None, UsePolicy(seed=1), enabled=False).use_hyp_survive
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src
    assert "def add(" not in src


def test_keep_steerer_spares_untried(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_hyp_survive and a.use_keep_steerer
    a._in_hand_id = "c08"
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs)
    a._in_hand_id = "p99"
    a._maybe_annotate("press", obs)
    ids = {r.fact_id for r in a.store.records()}
    assert {"c08", "p99"} <= ids
    a._last_chosen_ids = ["c08"]
    a._in_hand_id = "c08"
    a._update_chosen_hyp(success=True)
    a._keep_steerer(obs)
    recs = {r.fact_id: r for r in a.store.records()}
    assert "c08" in recs and "p99" in recs
    assert a._hyp_state(recs["c08"]) == "supported"
    assert a._hyp_state(recs["p99"]) == "untried"


def test_090_keep_steerer_still_drops(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make090(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_keep_steerer and not a.use_hyp_survive
    a._in_hand_id = "c08"
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs)
    a._in_hand_id = "p99"
    a._maybe_annotate("press", obs)
    a._last_chosen_ids = ["p99"]
    a._in_hand_id = "p99"
    a._keep_steerer(obs)
    ids = {r.fact_id for r in a.store.records()}
    assert "p99" in ids
    assert "c08" not in ids


def test_prefer_untried(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    a._in_hand_id = "c08"
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs)
    a._in_hand_id = "p99"
    a._maybe_annotate("press", obs)
    recs = {r.fact_id: r for r in a.store.records()}
    a._last_chosen_ids = ["c08"]
    a._update_chosen_hyp(success=True)
    recs = {r.fact_id: r for r in a.store.records()}
    pool = [recs["c08"], recs["p99"]]
    picked = a._prefer_untried(pool)
    assert [r.fact_id for r in picked] == ["p99"]


def test_classify_rewrites_lexical_when_hyps_survive():
    tag = (
        "# c08\nbind=xenon\ndid=press\nhyp=supported\ntrials=1\nwins=1\nlosses=0\n"
        "# p99\nbind=push\ndid=press\nhyp=untried\ntrials=0\nwins=0\nlosses=0\n"
    )
    import experiments.run_tm091 as tm091

    saved = tm091._classify_a090
    tm091._classify_a090 = lambda m: ("Fail", "Retrieve used clutter hapax xenon.")
    try:
        label, why = classify_a(
            {
                "use_hyp_survive": True,
                "hyp_survive": True,
                "train_s": {"n": 2, "tag": tag},
                "a_tag": tag,
            }
        )
    finally:
        tm091._classify_a090 = saved
    assert label == "Store-works"
    assert "hyp" in why.lower()


def test_classify_fails_when_rival_deleted():
    tag = "# c08\nbind=xenon\ndid=press\nhyp=supported\ntrials=1\nwins=1\nlosses=0\n"
    import experiments.run_tm091 as tm091

    saved = tm091._classify_a090
    tm091._classify_a090 = lambda m: ("Fail", "Retrieve used clutter hapax xenon.")
    try:
        label, why = classify_a(
            {
                "use_hyp_survive": True,
                "hyp_survive": True,
                "train_s": {"n": 1, "tag": tag},
                "a_tag": tag,
            }
        )
    finally:
        tm091._classify_a090 = saved
    assert label == "Fail"
    assert "deleted" in why.lower() or "rival" in why.lower()


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
    act, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    assert act == int(DialAction.HOLD)


if __name__ == "__main__":
    import tempfile

    test_090_confounds_hyp_survive_smuggle()
    test_hyp_survive_on_091_off_before()
    test_classify_rewrites_lexical_when_hyps_survive()
    test_classify_fails_when_rival_deleted()
    for fn in (
        test_keep_steerer_spares_untried,
        test_090_keep_steerer_still_drops,
        test_prefer_untried,
        test_untrained_holds,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
