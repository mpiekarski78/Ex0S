"""TM.0.7.2: keep the note that steered; drop other same-here stamps."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm071 import make as make071
from experiments.run_tm069 import wiki_prose
from experiments.run_tm072 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_keep_steerer_drops_other_same_here(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert a.use_keep_steerer and a.use_local_alias
    a._in_hand_id = "c08"
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs)
    a._in_hand_id = "p99"
    a._maybe_annotate("press", obs)
    assert {r.fact_id for r in a.store.records()} >= {"c08", "p99"}
    a._last_chosen_ids = ["p99"]
    a._in_hand_id = "p99"
    a._keep_steerer(obs)
    ids = {r.fact_id for r in a.store.records()}
    assert "p99" in ids
    assert "c08" not in ids


def test_071_keeps_both_notes(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    a = make071(s, w, UsePolicy(seed=7), force_write=True, explore_epsilon=0.0)
    assert not a.use_keep_steerer
    a._in_hand_id = "c08"
    obs = ChannelDialWorld(seed=3).reset("probe_channel_a")
    a._maybe_annotate("press", obs)
    a._in_hand_id = "p99"
    a._maybe_annotate("press", obs)
    a._last_chosen_ids = ["p99"]
    a._keep_steerer(obs)
    ids = {r.fact_id for r in a.store.records()}
    assert "c08" in ids and "p99" in ids


def test_072_confounds_scale_smuggle():
    import experiments.run_tm072 as tm072

    saved = tm072._classify_common071
    tm072._classify_common071 = lambda m: None
    try:
        label, why = tm072.classify_common(
            {
                "use_keep_steerer": True,
                "keep_steerer": True,
                "w_scale": True,
                "w_n_distinct_clutter": 64,
            }
        )
    finally:
        tm072._classify_common071 = saved
    assert label == "Confound"
    assert "scale" in why.lower()


def test_071_confounds_keep_steerer_smuggle():
    import experiments.run_tm071 as tm071

    saved = tm071._classify_common070
    tm071._classify_common070 = lambda m: None
    try:
        label, why = tm071.classify_common(
            {
                "use_local_alias": True,
                "local_alias": True,
                "use_keep_steerer": True,
            }
        )
    finally:
        tm071._classify_common070 = saved
    assert label == "Confound"
    assert "keep" in why.lower()


def test_skips_stay_skipped():
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_keep_steerer
    assert not make054(Path("/tmp/tm072_054"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
    assert not make071(Path("/tmp/tm072_071"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
    b = make(Path("/tmp/tm072_072"), None, UsePolicy(seed=1), enabled=False)
    assert b.use_keep_steerer and b.use_local_alias
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src


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

    test_skips_stay_skipped()
    test_071_confounds_keep_steerer_smuggle()
    test_072_confounds_scale_smuggle()
    for fn in (test_keep_steerer_drops_other_same_here, test_071_keeps_both_notes, test_untrained_holds):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
