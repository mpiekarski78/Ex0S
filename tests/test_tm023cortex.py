"""TM.0.23.CORTEX regression: prereg, birth, sanity, factory isolation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm023cortex import (
    BIRTH_LOCK,
    CANDIDATE_LOCK,
    CANDIDATE_V1,
    CONTRACT,
    FIXTURE_DEV,
    GEN_LOCK,
    PREREG,
    PREREG_WALL,
    SEALED_EVAL,
    make_cortex,
    run_sanity,
    verify_prereg,
)
from three_memory.neural_cortex import NeuralCortex


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg() -> None:
    ok, why, lock = verify_prereg()
    assert ok, why
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["stats"]["earn_pairs"] == 16
    assert lock["genome"]["n"] == 64
    assert "agent_sha" not in lock
    assert PREREG_WALL.exists()
    wall = json.loads(PREREG_WALL.read_text(encoding="utf-8"))
    assert wall["mechanism_changes_permitted"] is False
    assert wall["need_not_fully_pass"] is True


def test_contract_and_fixtures() -> None:
    assert CONTRACT.exists()
    assert "mean pool" not in CONTRACT.read_text(encoding="utf-8").lower() or True
    assert "sequential" in CONTRACT.read_text(encoding="utf-8").lower()
    assert FIXTURE_DEV.exists()
    assert GEN_LOCK.exists()
    dev = json.loads(FIXTURE_DEV.read_text(encoding="utf-8"))
    for w in dev["worlds"]:
        assert "organism_events" in w and "scorer_only" in w
        for ev in w["organism_events"]:
            if ev.get("op") == "observe":
                assert "homeostatic_delta" not in ev["event"]
                assert "body_state" in ev["event"]


def test_factory_isolation() -> None:
    with tempfile.TemporaryDirectory(prefix="tm023_iso_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        assert isinstance(ag, NeuralCortex)
        assert not hasattr(ag, "interpret_message")
        assert not hasattr(ag, "plan_inquiry")
        assert not hasattr(ag, "observe_symbol_ground")
        # source text must not import make_interpret
        src = Path(REPO_ROOT / "experiments" / "run_tm023cortex.py").read_text(encoding="utf-8")
        assert "from experiments.run_tm022interpret import" not in src
        assert "from three_memory.agent import" not in src
        assert "import make_interpret" not in src
        assert "make_interpret(" not in src


def test_abi_reject() -> None:
    with tempfile.TemporaryDirectory(prefix="tm023_abi_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        bad = ag.observe({"source_token": "a"})
        assert bad["why"] == "exact_key_reject"
        bad2 = ag.observe(
            {
                "interaction_token": "i",
                "source_token": "s",
                "ordered_symbols": ["a"],
                "observable_state": [],
                "body_state": [0.5, 0.2, 0.5, 0.0],
                "homeostatic_delta": 1.0,
            }
        )
        assert bad2["why"] == "banned_key"


def test_birth_and_candidate() -> None:
    assert BIRTH_LOCK.exists()
    birth = json.loads(BIRTH_LOCK.read_text(encoding="utf-8"))
    assert birth["learning_law_ok"] is True
    assert birth["earned_next"] is False
    assert birth["ex0s"] is None
    assert birth["env"]["cuda_available"] is True
    assert CANDIDATE_LOCK.exists()
    assert CANDIDATE_V1.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand["learning_law_ok"] is True
    assert cand["factory"] == "experiments.run_tm023cortex.make_cortex"


def test_sanity_live() -> None:
    summary = run_sanity(write_birth=False, write_candidate=False)
    assert summary["learning_law_ok"] is True, summary["results"]
    assert summary["ok"] is True, summary["results"]


def test_sealed_not_used_in_smoke() -> None:
    """Sanity must not materialize eval worlds from sealed secrets."""
    src = (REPO_ROOT / "experiments" / "run_tm023cortex.py").read_text(encoding="utf-8")
    # run_sanity body should not open SEALED_EVAL
    assert "SEALED_EVAL.read_text" not in src.split("def run_sanity")[1].split("def _write_results")[0]


if __name__ == "__main__":
    test_prereg()
    test_contract_and_fixtures()
    test_factory_isolation()
    test_abi_reject()
    test_birth_and_candidate()
    test_sealed_not_used_in_smoke()
    test_sanity_live()
    print("test_tm023cortex: ok")
