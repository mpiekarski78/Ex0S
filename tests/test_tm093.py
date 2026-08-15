"""TM.0.9.3: EVIDENCE — unit tests."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm091 import make as make091
from experiments.run_tm092 import make as make092
from experiments.run_tm093 import (
    classify_evidence,
    make,
    permute_evidence,
    run_evidence_battery,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy


def test_evidence_on_093_off_before():
    assert make(Path("/tmp/tm093_093"), None, UsePolicy(seed=1), enabled=False).use_evidence
    assert not make092(Path("/tmp/tm093_092"), None, UsePolicy(seed=1), enabled=False).use_evidence
    assert not make091(Path("/tmp/tm093_091"), None, UsePolicy(seed=1), enabled=False).use_evidence
    assert not make054(Path("/tmp/tm093_054"), None, UsePolicy(seed=1), enabled=False).use_evidence
    src = inspect.getsource(agent_mod)
    for word in ("push", "flim", "zorg", "wibble"):
        assert f'"{word}"' not in src
    assert UsePolicy.n_feat == 2


def test_evidence_battery_permuted(tmp_path: Path):
    policy = UsePolicy(seed=7, lr=0.2)
    for seed in (11, 22, 33):
        spec = permute_evidence(seed)
        bat = run_evidence_battery(policy, spec, tmp_path / f"s{seed}")
        assert bat["classification"] == "Store-works", bat["rationale"]
        assert spec["fx"] != spec["fy"]


def test_equal_is_not_a_filename_win():
    spec = {"x": "aa", "y": "bb", "m1": "press", "m2": "tune"}
    cells = {
        "unequal": {"action_name": "press"},
        "equal": {"action_name": "press", "evidence_resolved": True},
        "swap_a": {"action_name": "press"},
        "swap_b": {"action_name": "tune"},
        "wiped": {"action_name": "hold"},
        "reset": {"action_name": "press"},
        "earned": {
            "ids": {"m1": "a", "m2": "b", "y": "c"},
            "counts": {
                "a": {"support": 2, "contradiction": 0},
                "b": {"support": 0, "contradiction": 1},
                "c": {"support": 100, "contradiction": 0},
            },
        },
    }
    label, why = classify_evidence(cells, spec)
    assert label == "Fail"
    assert "equal" in why.lower() or "winner" in why.lower()


if __name__ == "__main__":
    test_evidence_on_093_off_before()
    test_equal_is_not_a_filename_win()
    with tempfile.TemporaryDirectory() as d:
        test_evidence_battery_permuted(Path(d))
    print("ok")
