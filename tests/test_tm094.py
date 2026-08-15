"""TM.0.9.4: REVISION — unit tests. No new genome."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm093 import make as make093, permute_evidence
from experiments.run_tm094 import classify_revision, make, run_revision_battery
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy


def test_revision_is_093_genome():
    src094 = inspect.getsource(make)
    src_agent = inspect.getsource(agent_mod)
    assert "use_revision" not in src094
    assert "use_revision" not in src_agent
    assert "recency" not in src_agent.lower()
    ag = make(Path("/tmp/tm094_094"), None, UsePolicy(seed=1), enabled=False)
    old = make093(Path("/tmp/tm094_093"), None, UsePolicy(seed=1), enabled=False)
    assert ag.use_evidence and old.use_evidence
    assert ag.use_bind_match and ag.use_hyp_survive
    assert UsePolicy.n_feat == 2
    for word in ("push", "flim", "zorg", "wibble"):
        assert f'"{word}"' not in src_agent


def test_revision_battery_permuted(tmp_path: Path):
    policy = UsePolicy(seed=7, lr=0.2)
    for seed in (11, 22, 33):
        spec = permute_evidence(seed)
        bat = run_revision_battery(policy, spec, tmp_path / f"s{seed}")
        assert bat["classification"] == "Store-works", bat["rationale"]


def test_order_residue_is_fail():
    spec = {"x": "aa", "y": "bb", "m1": "press", "m2": "tune"}
    ids = {"m1": "a", "m2": "b"}
    counts = {
        "a": {"support": 2, "contradiction": 1},
        "b": {"support": 1, "contradiction": 1},
    }
    cells = {
        "walk": [
            {"action_name": "hold"},
            {"action_name": "press"},
            {"action_name": "hold"},
            {"action_name": "tune"},
        ],
        "early": {"action_name": "press"},
        "revised": {"action_name": "tune"},
        "revise_ids": ids,
        "early_counts": {
            "a": {"support": 2, "contradiction": 0},
            "b": {"support": 0, "contradiction": 1},
        },
        "revised_counts": {
            "a": {"support": 2, "contradiction": 2},
            "b": {"support": 3, "contradiction": 1},
        },
        "order_a": {"action_name": "press"},
        "order_b": {"action_name": "tune"},
        "order_a_ids": ids,
        "order_b_ids": ids,
        "order_a_counts": counts,
        "order_b_counts": counts,
    }
    label, why = classify_revision(cells, spec)
    assert label == "Fail"
    assert "order" in why.lower() or "motor" in why.lower()


if __name__ == "__main__":
    test_revision_is_093_genome()
    test_order_residue_is_fail()
    with tempfile.TemporaryDirectory() as d:
        test_revision_battery_permuted(Path(d))
    print("ok")
