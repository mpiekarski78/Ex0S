"""TM.0.8.2: one machine — body from n_actions / percepts, not domain=."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm081 import make as make081
from experiments.run_tm080 import wiki_prose
from experiments.run_tm082 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_agent_has_no_domain_switch():
    src = inspect.getsource(agent_mod.ThreeMemoryAgent)
    assert "self.domain ==" not in src
    assert "self.domain !=" not in src
    assert "if domain ==" not in src


def test_081_confounds_no_domain_switch():
    import experiments.run_tm081 as tm081

    saved = tm081._classify_common080
    tm081._classify_common080 = lambda m: None
    try:
        label, why = tm081.classify_common(
            {
                "one_return_recipe": True,
                "trained_split": False,
                "no_domain_switch": True,
            }
        )
    finally:
        tm081._classify_common080 = saved
    assert label == "Confound"
    assert "domain" in why.lower()


def test_body_from_n_actions():
    door = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert door.n_actions == 4 and door.domain == "door"
    assert "open" in door._act_names() and "hold" not in door._act_names()
    dial = make(Path("/tmp/tm082_082"), None, UsePolicy(seed=1), enabled=False)
    assert dial.n_actions == 5
    assert "hold" in dial._act_names() and "open" not in dial._act_names()
    assert not make054(Path("/tmp/tm082_054"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
    assert make081(Path("/tmp/tm082_081"), None, UsePolicy(seed=1), enabled=False).use_keep_steerer
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

    test_agent_has_no_domain_switch()
    test_081_confounds_no_domain_switch()
    test_body_from_n_actions()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_holds(Path(d))
    print("ok")
