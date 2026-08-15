"""TM.0.10.FAMILY: freeze and generator tests. No organism edits."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm010family import (
    BANNED,
    DEVELOP,
    HOLDOUT,
    expected_motor,
    generate_world,
    run_family,
    verify_freeze,
)
from experiments.run_tm094 import make as make094
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy


def test_frozen_094_genome():
    ok, why, _snap = verify_freeze()
    assert ok, why
    assert (REPO_ROOT / "docs" / "genome_094.lock").exists()
    ag = make094(Path("/tmp/tm010_094"), None, UsePolicy(seed=1), enabled=False)
    assert ag.use_evidence and ag.use_bind_match and ag.use_hyp_survive
    assert UsePolicy.n_feat == 2
    src = inspect.getsource(agent_mod)
    assert "use_family" not in src
    for word in ("push", "flim", "zorg", "wibble"):
        assert f'"{word}"' not in src


def test_holdout_split_and_nonces():
    assert DEVELOP == ("A", "B", "C", "D")
    assert HOLDOUT == ("E", "F", "G")
    for fam in DEVELOP + HOLDOUT:
        w = generate_world(fam, 12345, 0)
        assert w.family == fam
        assert w.holdout == (fam in HOLDOUT)
        for r in w.relations:
            assert r.bind not in BANNED
            assert r.did in ("press", "tune", "flip", "hold")


def test_expected_motor_tie_and_reverse():
    rels = [
        {"bind": "aa", "did": "press", "support": 1, "contradiction": 0},
        {"bind": "aa", "did": "tune", "support": 1, "contradiction": 0},
        {"bind": "bb", "did": "flip", "support": 1000, "contradiction": 0},
    ]
    assert expected_motor(rels, "aa") == "hold"
    assert expected_motor(rels, "bb") == "flip"
    assert expected_motor(rels, "zz") == "hold"
    rels[0]["support"] = 2
    assert expected_motor(rels, "aa") == "press"
    rels[1]["support"] = 3
    rels[1]["contradiction"] = 1
    rels[0]["contradiction"] = 2
    assert expected_motor(rels, "aa") == "tune"


def test_smoke_one_birth_per_family(tmp_path: Path):
    summary = run_family(seed=11, per_family=1, births=1, workers=1)
    assert summary["n_worlds"] == 7
    assert summary["genome_ok"]
    assert summary["intervention"]["required_genome_changes"] == 0


if __name__ == "__main__":
    test_frozen_094_genome()
    test_holdout_split_and_nonces()
    test_expected_motor_tie_and_reverse()
    with tempfile.TemporaryDirectory() as d:
        test_smoke_one_birth_per_family(Path(d))
    print("ok")
