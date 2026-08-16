"""TM.0.18.SEQUENCE regression: baseline, sequence cells, expressive life, dialogue."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm018sequence import (
    AGENT_PY,
    BASELINE_LOCK,
    CANDIDATE_LOCK,
    DIALOGUE_LOCK,
    FIXTURE_JSON,
    MECH_LOCK,
    PREREG_BASELINE,
    PREREG_DIALOGUE,
    PREREG_MECH,
    SEQUENCE_LOCK,
    apply_ground,
    apply_seq_step,
    fresh,
    run_baseline,
    run_dialogue,
    run_life,
    run_unit_cells,
    verify_baseline_prereg,
    verify_dialogue_prereg,
    verify_mech_prereg,
)
from three_memory.policy import UsePolicy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg_and_fixture() -> None:
    ok, why, lock = verify_baseline_prereg()
    assert ok, why
    assert lock["fixture_sha"] == sha(FIXTURE_JSON)
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["phase_a"]["agent_edits_permitted"] is False
    ok_m, why_m, mlock = verify_mech_prereg()
    assert ok_m, why_m
    assert mlock["flag"] == "use_symbol_sequence"
    assert mlock["flag_default"] is False
    assert mlock["source"] == "experience_sequence"
    assert "agent_sha" not in mlock
    ok_d, why_d, dlock = verify_dialogue_prereg()
    assert ok_d, why_d
    assert dlock["mechanism_changes_permitted"] is False
    assert dlock["fixture_sha"] == sha(FIXTURE_JSON)


def test_baseline_hold() -> None:
    assert BASELINE_LOCK.exists()
    summary = run_baseline(seed=12345, write_lock=False)
    assert summary["ok"] is True
    assert summary["first_fail"] is None
    freeze = json.loads(BASELINE_LOCK.read_text(encoding="utf-8"))
    assert freeze["ok"] is True
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)


def test_unit_cells() -> None:
    assert CANDIDATE_LOCK.exists()
    cells = run_unit_cells(seed=12345)
    assert cells["ok"] is True
    assert cells["n_pass"] == cells["n_cells"] == 6
    assert MECH_LOCK.exists()
    freeze = json.loads(MECH_LOCK.read_text(encoding="utf-8"))
    assert freeze["ok"] is True


def test_expressive_life() -> None:
    assert SEQUENCE_LOCK.exists()
    summary = run_life(seed=12345, write_lock=False)
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["agent_sha"] == sha(AGENT_PY)
    freeze = json.loads(SEQUENCE_LOCK.read_text(encoding="utf-8"))
    assert freeze["last_stage_clear"] == summary["last_stage_clear"]
    assert freeze["first_fail_stage"] == summary["first_fail_stage"]
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)
    assert summary["ok"] is True
    assert summary["last_stage_clear"] == "E12"
    assert summary.get("life_last_stage_clear") == "E11"
    assert freeze.get("life_last_stage_clear") == "E11"
    assert summary["first_fail"] is None
    assert summary["main"]["ok"] and summary["twin"]["ok"]
    assert summary["capacity"]["ok"] is True
    vocab = summary["capacity"]["lanes"]["vocab_lane"]
    assert [r["rung"] for r in vocab["rungs"]] == [8, 32, 128, 512]


def test_dialogue_wall() -> None:
    assert DIALOGUE_LOCK.exists()
    summary = run_dialogue(seed=12345, write_lock=False)
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["need_not_fully_pass"] is True
    freeze = json.loads(DIALOGUE_LOCK.read_text(encoding="utf-8"))
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)
    assert summary["first_fail_dialogue"] is None


def test_atomic_hold_and_isolation_invariants() -> None:
    """Audit: no partial credit; grounding≠emit; sequence≠select; scene refuse."""
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm018_audit_") as tmp:
        root = Path(tmp)
        _, ag = fresh(root, "g", policy, sequenced=True)
        apply_ground(
            ag, {"symbol": "ball", "paired": "cat_ball", "trial_id": "a0", "result": "success"}
        )
        apply_ground(
            ag, {"symbol": "ball", "paired": "cat_ball", "trial_id": "a1", "result": "success"}
        )
        assert ag.emit_sequence(["cat_ball"], ["describe"]).get("sequence") is None

        _, ag = fresh(root, "s", policy, sequenced=True)
        for _ in range(2):
            apply_seq_step(
                ag,
                {
                    "context_atoms": ["cat_ball"],
                    "input_symbols": ["describe"],
                    "prefix": [],
                    "next_operation": "emit",
                    "next_symbol": "ball",
                    "result": "success",
                },
            )
            apply_seq_step(
                ag,
                {
                    "context_atoms": ["cat_ball"],
                    "input_symbols": ["describe"],
                    "prefix": ["ball"],
                    "next_operation": "stop",
                    "next_symbol": "",
                    "result": "success",
                },
            )
        assert (
            ag.select_grounded(["cat_ball"], ["ball", "cup"], expression=True).get("selected")
            is None
        )

        _, ag = fresh(root, "p", policy, sequenced=True)
        apply_ground(ag, {"symbol": "a", "paired": "c", "trial_id": "p0", "result": "success"})
        apply_ground(ag, {"symbol": "a", "paired": "c", "trial_id": "p1", "result": "success"})
        apply_ground(ag, {"symbol": "b", "paired": "c", "trial_id": "p2", "result": "success"})
        apply_ground(ag, {"symbol": "b", "paired": "c", "trial_id": "p3", "result": "success"})
        for _ in range(2):
            apply_seq_step(
                ag,
                {
                    "context_atoms": ["c"],
                    "input_symbols": ["q"],
                    "prefix": [],
                    "next_operation": "emit",
                    "next_symbol": "a",
                    "result": "success",
                },
            )
        apply_seq_step(
            ag,
            {
                "context_atoms": ["c"],
                "input_symbols": ["q"],
                "prefix": ["a"],
                "next_operation": "emit",
                "next_symbol": "b",
                "result": "success",
            },
        )
        apply_seq_step(
            ag,
            {
                "context_atoms": ["c"],
                "input_symbols": ["q"],
                "prefix": ["a"],
                "next_operation": "emit",
                "next_symbol": "x",
                "result": "success",
            },
        )
        mid = ag.emit_sequence(["c"], ["q"])
        assert mid.get("sequence") is None
        assert mid.get("first_internal_fail") is not None

        bad = ag.observe_sequence_step(
            {
                "context_atoms": ["scene_184"],
                "input_symbols": ["q"],
                "prefix": [],
                "next_operation": "emit",
                "next_symbol": "x",
                "result": "success",
            }
        )
        assert bad.get("ok") is False
        assert bad.get("why") == "scene_id_refuse"


def test_no_scene_id_or_grammar_slots() -> None:
    src = AGENT_PY.read_text(encoding="utf-8")
    assert "def observe_sequence_step" in src
    assert "def emit_sequence" in src
    assert "def learn_grammar" not in src
    assert "noun_slot=" not in src
    assert "verb_slot=" not in src


def test_e6_target_absent_from_teaching() -> None:
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    target = tuple(fix["e6_target_seq_main"])
    completed = []
    for o in fix["script_life"]:
        if o.get("id") == "E6_unseen":
            break
        if o.get("op") == "seq_step" and o.get("next_operation") == "stop":
            completed.append(tuple(o.get("prefix") or []))
    assert target not in completed


def test_capacity_branches_unconfounded() -> None:
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    cap = fix["capacity"]
    assert "vocab_lane" in cap and "length_lane" in cap and "age_lane" in cap
    assert cap["vocab_lane"]["rungs"] == [8, 32, 128, 512]
    for lane in ("vocab_lane", "length_lane", "age_lane"):
        assert cap[lane]["branches"]
        assert all("script" in b for b in cap[lane]["branches"])


def test_dialogue_prereg_frozen_before_candidate() -> None:
    assert PREREG_DIALOGUE.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand.get("sequence_dialogue_prereg_sha") == sha(PREREG_DIALOGUE)
    assert PREREG_BASELINE.exists() and PREREG_MECH.exists()


if __name__ == "__main__":
    test_prereg_and_fixture()
    test_baseline_hold()
    test_unit_cells()
    test_e6_target_absent_from_teaching()
    test_capacity_branches_unconfounded()
    test_dialogue_prereg_frozen_before_candidate()
    test_atomic_hold_and_isolation_invariants()
    test_expressive_life()
    test_dialogue_wall()
    test_no_scene_id_or_grammar_slots()
    print("ok")
