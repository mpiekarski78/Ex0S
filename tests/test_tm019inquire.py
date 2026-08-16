"""TM.0.19.INQUIRE regression: baseline, inquire cells, I-life, capacity, wall."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm019inquire import (
    AGENT_PY,
    BASELINE_LOCK,
    CANDIDATE_LOCK,
    FIXTURE_JSON,
    INQUIRE_LOCK,
    MECH_LOCK,
    PREREG_BASELINE,
    PREREG_MECH,
    PREREG_WALL,
    WALL_LOCK,
    apply_ground,
    fresh,
    run_baseline,
    run_life,
    run_smoke,
    run_unit_cells,
    run_wall,
    verify_baseline_prereg,
    verify_mech_prereg,
    verify_wall_prereg,
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
    assert mlock["flag"] == "use_inquire"
    assert mlock["flag_default"] is False
    assert mlock["source"] == "experience_inquire"
    assert mlock["inquire_budget"] == 8
    assert "agent_sha" not in mlock
    ok_w, why_w, wlock = verify_wall_prereg()
    assert ok_w, why_w
    assert wlock["mechanism_changes_permitted"] is False
    assert wlock["fixture_sha"] == sha(FIXTURE_JSON)
    assert wlock["need_not_fully_pass"] is True


def test_baseline_hold() -> None:
    assert BASELINE_LOCK.exists()
    summary = run_baseline(seed=12345, write_lock=False)
    assert summary["ok"] is True
    assert summary["first_fail"] is None
    freeze = json.loads(BASELINE_LOCK.read_text(encoding="utf-8"))
    assert freeze["ok"] is True
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)


def test_smoke_and_candidate() -> None:
    smoke = run_smoke(seed=12345)
    assert smoke["ok"] is True
    assert CANDIDATE_LOCK.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand["flag"] == "use_inquire"
    assert cand["earned_next"] is False
    assert cand["ex0s"] is None
    assert cand.get("inquire_wall_prereg_sha") == sha(PREREG_WALL)


def test_unit_cells() -> None:
    assert CANDIDATE_LOCK.exists()
    cells = run_unit_cells(seed=12345)
    assert cells["ok"] is True
    assert cells["n_pass"] == cells["n_cells"] == 6
    assert MECH_LOCK.exists()
    freeze = json.loads(MECH_LOCK.read_text(encoding="utf-8"))
    assert freeze["ok"] is True


def test_developmental_life() -> None:
    assert INQUIRE_LOCK.exists()
    summary = run_life(seed=12345, write_lock=False)
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["agent_sha"] == sha(AGENT_PY)
    freeze = json.loads(INQUIRE_LOCK.read_text(encoding="utf-8"))
    assert freeze["last_stage_clear"] == summary["last_stage_clear"]
    assert freeze["first_fail_stage"] == summary["first_fail_stage"]
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)
    assert summary["ok"] is True
    assert summary["last_stage_clear"] == "I12"
    assert summary.get("life_last_stage_clear") == "I12"
    assert freeze.get("life_last_stage_clear") == "I12"
    assert summary["first_fail"] is None
    assert summary["main"]["ok"] and summary["twin"]["ok"]
    assert summary["capacity"]["ok"] is True
    hyps = summary["capacity"]["lanes"]["hypotheses_lane"]
    assert [r["rung"] for r in hyps["rungs"]] == [2, 4, 8, 16]
    depth = summary["capacity"]["lanes"]["depth_lane"]
    assert [r["rung"] for r in depth["rungs"]] == [1, 2, 4]


def test_final_wall() -> None:
    assert WALL_LOCK.exists()
    summary = run_wall(seed=12345, write_lock=False)
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["need_not_fully_pass"] is True
    freeze = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)
    assert summary["scored_probes_ok"] is True
    assert summary["first_fail_wall"] is not None
    assert summary["first_fail_wall"]["id"] == "W_conflict_teachers"
    assert summary["next_primitive_hint"]


def test_no_teacher_callback_in_plan_inquiry() -> None:
    src = AGENT_PY.read_text(encoding="utf-8")
    assert "def plan_inquiry" in src
    assert "def observe_inquire_trace" in src
    assert "def _inquire_render_probe" in src
    start = src.index("def plan_inquiry")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "correct_question" not in body
    assert "prediction_table" not in body
    assert "self.teacher" not in body
    assert "call_teacher" not in body
    assert "scorer(" not in body
    assert "emit_sequence" in body or "_inquire_render_probe" in body


def test_dual_memory_isolation() -> None:
    """Consequences in grounding; inquire strip keeps answer; grounding strip → HOLD."""
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm019_audit_") as tmp:
        root = Path(tmp)
        s_dir, ag = fresh(root, "dm", policy, inquired=True)
        for row in (
            {"symbol": "obj_round", "paired": "feat_round", "trial_id": "a0", "result": "success"},
            {"symbol": "obj_round", "paired": "feat_round", "trial_id": "a1", "result": "success"},
            {"symbol": "dax", "paired": "obj_red", "trial_id": "a2", "result": "success"},
            {"symbol": "dax", "paired": "obj_red", "trial_id": "a3", "result": "success"},
            {"symbol": "dax", "paired": "obj_round", "trial_id": "a4", "result": "success"},
            {"symbol": "dax", "paired": "obj_round", "trial_id": "a5", "result": "success"},
        ):
            apply_ground(ag, row)
        plan = ag.plan_inquiry(
            {"context_atoms": ["world"], "input_symbols": ["what", "dax"]}
        )
        assert plan["status"] == "PROBE_ATOMS"
        assert "feat_round" in (plan.get("probe_atoms") or [])


def test_capacity_branches_unconfounded() -> None:
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    cap = fix["capacity"]
    assert set(cap) >= {
        "hypotheses_lane",
        "depth_lane",
        "age_lane",
        "source_count_lane",
    }
    assert cap["hypotheses_lane"]["rungs"] == [2, 4, 8, 16]
    assert cap["depth_lane"]["rungs"] == [1, 2, 4]
    assert cap["age_lane"]["rungs"] == [100, 1000, 10000]
    assert cap["source_count_lane"]["rungs"] == [1, 2, 4]
    assert fix.get("probe_renders")
    assert "world" in (fix.get("context_tokens") or [])
    forks = [o for o in fix["script_life"] if o.get("op") == "fork"]
    kinds = {f["kind"] for f in forks}
    assert "strip_consequence" in kinds
    assert "strip_inquire" in kinds
    assert "donor" in kinds


def test_candidate_v1_preserved_after_audit() -> None:
    from experiments.run_tm019inquire import CANDIDATE_V1_LOCK

    assert CANDIDATE_V1_LOCK.exists()
    v1 = json.loads(CANDIDATE_V1_LOCK.read_text(encoding="utf-8"))
    assert v1["flag"] == "use_inquire"
    assert v1.get("agent_sha")
    # Tip candidate may differ after audit rewrite; v1 must remain.
    tip = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert tip["agent_sha"] != v1["agent_sha"] or tip["plan_inquiry_sha"] != v1.get(
        "plan_inquiry_sha"
    )


def test_wall_prereg_frozen_before_candidate() -> None:
    assert PREREG_WALL.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand.get("inquire_wall_prereg_sha") == sha(PREREG_WALL)
    assert PREREG_BASELINE.exists() and PREREG_MECH.exists()


if __name__ == "__main__":
    test_prereg_and_fixture()
    test_baseline_hold()
    test_smoke_and_candidate()
    test_unit_cells()
    test_capacity_branches_unconfounded()
    test_wall_prereg_frozen_before_candidate()
    test_candidate_v1_preserved_after_audit()
    test_no_teacher_callback_in_plan_inquiry()
    test_dual_memory_isolation()
    test_developmental_life()
    test_final_wall()
    print("ok")
