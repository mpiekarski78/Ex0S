"""TM.0.11: COMPOSE — unit tests."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm093 import make as make093
from experiments.run_tm094 import make as make094
from experiments.run_tm011compose import (
    classify_compose,
    make,
    permute_compose,
    run_compose_battery,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy


def test_compose_on_011_off_before():
    assert make(Path("/tmp/tm011compose_011"), None, UsePolicy(seed=1), enabled=False).use_compose
    assert not make094(Path("/tmp/tm011compose_094"), None, UsePolicy(seed=1), enabled=False).use_compose
    assert not make093(Path("/tmp/tm011compose_093"), None, UsePolicy(seed=1), enabled=False).use_compose
    src = inspect.getsource(agent_mod)
    for word in ("push", "flim", "zorg", "wibble"):
        assert f'"{word}"' not in src
    assert "use_two_hop" not in src
    assert "use_three_hop" not in src
    assert "MAX_HOPS" not in src
    assert UsePolicy.n_feat == 2


def test_compose_battery_permuted(tmp_path: Path):
    policy = UsePolicy(seed=7, lr=0.2)
    for seed in (11, 22, 33):
        spec = permute_compose(seed)
        bat = run_compose_battery(policy, spec, tmp_path / f"s{seed}")
        assert bat["classification"] == "Store-works", bat["rationale"]
        assert not bat["direct_before"] and not bat["direct_after"]
        assert bat["s_hash_before"] == bat["s_hash_after"]


def test_shortcut_is_fail():
    spec = {"x": "aa", "mid": "bb", "m1": "press", "m2": "tune"}
    cells = {
        "main": {"action_name": "press", "compose_hops": 2},
        "direct_before": True,
        "direct_after": False,
        "s_hash_before": "a",
        "s_hash_after": "a",
        "broken": {"action_name": "hold"},
        "wrong_second": {"action_name": "tune"},
        "wrong_first": {"action_name": "hold"},
        "irr": {"action_name": "press"},
        "donor_press": {"action_name": "press"},
        "donor_tune": {"action_name": "tune"},
        "upstream": {"action_name": "tune"},
        "residue": {"action_name": "hold"},
        "wiped": {"action_name": "hold"},
        "reset": {"action_name": "press"},
    }
    label, why = classify_compose(cells, spec)
    assert label == "Fail"
    assert "direct" in why.lower()


if __name__ == "__main__":
    test_compose_on_011_off_before()
    test_shortcut_is_fail()
    with tempfile.TemporaryDirectory() as d:
        test_compose_battery_permuted(Path(d))
    print("ok")
