"""TM.0.8.1: one shared return is the A recipe."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm058 import N_CLUTTER
from experiments.run_tm080 import make as make080, wiki_prose
from experiments.run_tm081 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_080_confounds_one_return_smuggle():
    from experiments.run_tm080 import _require_scale

    req = _require_scale(
        {
            "w_n_distinct_clutter": 64,
            "w_n": 66,
            "w_scale": True,
            "scale_english_w": True,
            "w_clutter_has_two_rare": True,
            "w_useful_only_two_rare": False,
            "w_n_two_rare_clutter": 16,
            "one_return_recipe": True,
        }
    )
    assert req and req[0] == "Confound"
    assert "one-return" in req[1].lower() or "one return" in req[1].lower()


def test_081_b_hide_does_not_restore_split():
    import experiments.run_tm081 as tm081

    saved = tm081._classify_b080
    seen = {}

    def _spy(m):
        seen["trained_split"] = m.get("trained_split")
        return "Store-works", "ok"

    tm081._classify_b080 = _spy
    try:
        label, why = tm081.classify_b(
            {"one_return_recipe": True, "trained_split": False, "train_s": {"n": 2}}
        )
    finally:
        tm081._classify_b080 = saved
    assert seen.get("trained_split") is False
    assert label == "Store-works"


def test_081_confounds_split_restored():
    import experiments.run_tm081 as tm081

    saved = tm081._classify_a080
    tm081._classify_a080 = lambda m: ("Fail", "hidden")
    try:
        label, why = tm081.classify_a(
            {"one_return_recipe": True, "trained_split": True}
        )
    finally:
        tm081._classify_a080 = saved
    assert label == "Confound"
    assert "split" in why.lower()


def test_w_still_a_pile():
    assert len(wiki_prose(include_a=True, include_c=True)) == N_CLUTTER + 2


def test_skips_stay_skipped():
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_keep_steerer
    assert not make054(Path("/tmp/tm081_054"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
    assert make080(Path("/tmp/tm081_080"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
    b = make(Path("/tmp/tm081_081"), None, UsePolicy(seed=1), enabled=False)
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

    test_080_confounds_one_return_smuggle()
    test_081_b_hide_does_not_restore_split()
    test_081_confounds_split_restored()
    test_w_still_a_pile()
    test_skips_stay_skipped()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_holds(Path(d))
    print("ok")
