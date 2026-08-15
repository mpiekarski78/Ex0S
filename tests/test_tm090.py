"""TM.0.9.0: first math life — count unread rares, not + in cortex."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm082 import make as make082
from experiments.run_tm080 import wiki_prose
from experiments.run_tm090 import make
from three_memory import agent as agent_mod
from three_memory.dial_env import ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def test_082_confounds_count_search_smuggle():
    from experiments.run_tm082 import _require_one_machine

    req = _require_one_machine(
        {"domain_switch": False, "no_domain_switch": True, "use_count_search": True}
    )
    assert req and req[0] == "Confound"
    assert "count" in req[1].lower()


def test_count_search_on_090_off_before():
    assert not make082(Path("/tmp/tm090_082"), None, UsePolicy(seed=1), enabled=False).use_count_search
    a = make(Path("/tmp/tm090_090"), None, UsePolicy(seed=1), enabled=False)
    assert a.use_count_search and a.use_keep_steerer and a.n_actions == 5
    door = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert door.domain == "door" and not door.use_count_search
    assert not make054(Path("/tmp/tm090_054"), None, UsePolicy(seed=1), enabled=False).use_count_search
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src
    assert "def add(" not in src


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

    test_082_confounds_count_search_smuggle()
    test_count_search_on_090_off_before()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_holds(Path(d))
    print("ok")
