"""TM.0.8.0: scale English Open W to a 64-page pile."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm058 import N_CLUTTER
from experiments.run_tm069 import wiki_prose as wiki069
from experiments.run_tm072 import make as make072
from experiments.run_tm080 import MIN_TWO_RARE, clutter_prose, make, wiki_prose
from experiments.run_tm054 import _rare_words
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_english_w_is_a_pile():
    clutter = clutter_prose()
    assert len(clutter) == N_CLUTTER
    assert len({b for _, b in clutter}) == N_CLUTTER
    both = wiki_prose(include_a=True, include_c=True)
    assert len(both) == N_CLUTTER + 2
    rare = _rare_words(both)
    two_clutter = [n for n, ws in rare.items() if n.startswith("c") and len(ws) >= 2]
    assert len(two_clutter) >= MIN_TWO_RARE
    assert len(rare.get("p99.md") or []) >= 2
    assert len(rare.get("p98.md") or []) >= 2
    tiny = wiki069(include_a=True, include_c=True)
    assert len([n for n, _ in tiny if n.startswith("c")]) < N_CLUTTER


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


def test_skips_stay_skipped():
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_keep_steerer
    assert not make054(Path("/tmp/tm080_054"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
    b = make072(Path("/tmp/tm080_072"), None, UsePolicy(seed=1), enabled=False)
    assert b.use_keep_steerer
    c = make(Path("/tmp/tm080_080"), None, UsePolicy(seed=1), enabled=False)
    assert c.use_keep_steerer and c.use_local_alias
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

    test_english_w_is_a_pile()
    test_072_confounds_scale_smuggle()
    test_skips_stay_skipped()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_holds(Path(d))
    print("ok")
