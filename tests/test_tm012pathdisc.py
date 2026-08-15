"""TM.0.12.PATHDISC: rules — same-S C8, no route discovery, no 0.0.004."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import CONTEXT_LOCK
from experiments.run_tm012minimap import (
    MINIMAP_LOCK,
    OUTCOME_COLLISION,
    OUTCOME_DISTINGUISHES,
    OUTCOME_INADMISSIBLE,
    score_contrast,
)
from experiments.run_tm012pathdisc import (
    CELL_ID,
    PATHDISC_LOCK,
    ROLE_ORIGIN_VS_PATH,
    extract_states_from_trace,
    find_edge,
    gen_c8,
    refuse_answer_derived_fid,
    relations_content_hash,
    run_pathdisc,
    validate_c8_pair,
    verify_pathdisc_lock,
)


def test_locks_fail_closed_no_rewrite():
    ok, why, snap = verify_pathdisc_lock()
    assert ok, why
    assert snap["earned_next"] is False
    assert snap["cell_id"] == CELL_ID
    assert snap["role"] == ROLE_ORIGIN_VS_PATH
    assert PATHDISC_LOCK.exists()
    lock = json.loads(PATHDISC_LOCK.read_text(encoding="utf-8"))
    assert lock["context_012_lock_sha"] == hashlib.sha256(CONTEXT_LOCK.read_bytes()).hexdigest()
    assert lock["minimap_012_lock_sha"] == hashlib.sha256(MINIMAP_LOCK.read_bytes()).hexdigest()
    # Tests must not rewrite CONTEXT / MINIMAP / PATHDISC locks.
    import experiments.run_tm012pathdisc as pd

    assert hasattr(pd, "write_pathdisc_lock")
    # Presence of write helper is fine; tests never call it.


def test_same_s_hash_one_store():
    cell = gen_c8(12345)
    h = relations_content_hash(cell.relations)
    assert h == relations_content_hash(cell.relations)
    assert len(cell.traces) == 2
    assert cell.traces[0].nodes[0] == cell.origin
    assert cell.traces[1].nodes[0] == cell.origin
    assert cell.traces[0].nodes[-1] == cell.frontier
    assert cell.traces[1].nodes[-1] == cell.frontier
    # Both traces' edges live in the same S (true same-store lower bound).
    store = {r.fid for r in cell.relations}
    for t in cell.traces:
        for a, b in zip(t.nodes, t.nodes[1:]):
            e = find_edge(cell.relations, a, b)
            assert e is not None and e.fid in store
        m = find_edge(cell.relations, cell.frontier, t.required_motor)
        assert m is not None and m.fid in store


def test_validate_requires_y_motor_edges():
    cell = gen_c8(12345)
    a, b = cell.traces
    no_motor = [r for r in cell.relations if r.bind.lower() != cell.frontier.lower()]
    errs = validate_c8_pair(
        no_motor, a, b, origin=cell.origin, frontier=cell.frontier
    )
    assert any("missing outgoing" in e for e in errs)
    from experiments.run_tm012pathdisc import TraceSpec

    bad = TraceSpec("route_a", a.nodes, "notamotor")
    errs2 = validate_c8_pair(
        cell.relations, bad, b, origin=cell.origin, frontier=cell.frontier
    )
    assert any("MOTORS" in e for e in errs2)


def test_trace_validation_and_motors_differ():
    cell = gen_c8(12345)
    a, b = cell.traces
    errs = validate_c8_pair(
        cell.relations, a, b, origin=cell.origin, frontier=cell.frontier
    )
    assert errs == []
    assert a.required_motor.lower() != b.required_motor.lower()
    assert [n.lower() for n in a.nodes[1:-1]] != [n.lower() for n in b.nodes[1:-1]]


def test_extract_states_no_path_and_frontier():
    src = inspect.getsource(extract_states_from_trace)
    assert "path_and_frontier" not in src
    cell = gen_c8(12345)
    sa = extract_states_from_trace(cell.relations, cell.traces[0])
    sb = extract_states_from_trace(cell.relations, cell.traces[1])
    assert sa["H3a"] == sb["H3a"]
    assert sa["H3b"] != sb["H3b"]
    assert sa["H0"] == sb["H0"]
    assert sa["H1"] == sb["H1"]
    assert sa["H2"] != sb["H2"]
    assert sa["H4"] != sb["H4"]


def test_h3a_collides_h3b_distinguishes():
    cell = gen_c8(12345)
    a, b = cell.traces
    sa = extract_states_from_trace(cell.relations, a)
    sb = extract_states_from_trace(cell.relations, b)
    assert (
        score_contrast(
            contrast_id="C8",
            role=ROLE_ORIGIN_VS_PATH,
            left_states=sa,
            right_states=sb,
            left_motor=a.required_motor,
            right_motor=b.required_motor,
            candidate="H3a",
        )
        == OUTCOME_COLLISION
    )
    assert (
        score_contrast(
            contrast_id="C8",
            role=ROLE_ORIGIN_VS_PATH,
            left_states=sa,
            right_states=sb,
            left_motor=a.required_motor,
            right_motor=b.required_motor,
            candidate="H3b",
        )
        == OUTCOME_DISTINGUISHES
    )


def test_outgoing_y_motor_fid_inadmissible():
    cell = gen_c8(12345)
    a = cell.traces[0]
    out = find_edge(cell.relations, cell.frontier, a.required_motor)
    inc = find_edge(cell.relations, a.nodes[-2], a.nodes[-1])
    assert out is not None and inc is not None
    assert out.fid != inc.fid
    assert refuse_answer_derived_fid(inc.fid, out.fid) == OUTCOME_INADMISSIBLE


def test_computed_table_origin_insufficient():
    summary = run_pathdisc(seed=12345)
    assert summary["ok"], summary.get("why")
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert summary["origin_insufficient"] is True
    assert summary["h2_survives"] is True
    table = summary["table"]
    assert table["H0"]["C8"] == OUTCOME_COLLISION
    assert table["H1"]["C8"] == OUTCOME_COLLISION
    assert table["H2"]["C8"] == OUTCOME_DISTINGUISHES
    assert table["H3a"]["C8"] == OUTCOME_COLLISION
    assert table["H3b"]["C8"] == OUTCOME_DISTINGUISHES
    assert table["H4"]["C8"] == OUTCOME_DISTINGUISHES
    claim = summary["claim"]
    assert "Origin alone is insufficient" in claim
    assert "store-the-full-path" in claim or "full path" in claim.lower()
    assert "H3c" in claim


def test_never_stamps_004():
    summary = run_pathdisc(seed=12345)
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert summary["version"] == "TM.0.12.PATHDISC"
    # Refuse list may mention 0.0.004; the run must not earn it.
    assert summary.get("earned_next") is False
    lock = summary.get("lock") or {}
    assert lock.get("earned_next") is False
    assert "stamp Ex0S 0.0.004" in (lock.get("refuse") or [])


def test_no_write_lock_in_tests_module():
    """This test file must not invoke lock writers."""
    src = Path(__file__).read_text(encoding="utf-8")
    banned = ["write_" + "pathdisc_lock(", "write_" + "context_lock", "write_" + "minimap_lock"]
    for phrase in banned:
        assert phrase not in src


def test_scorer_sha_pinned():
    ok, why, snap = verify_pathdisc_lock()
    assert ok, why
    assert "scorer_sha" in snap
    lock = json.loads(PATHDISC_LOCK.read_text(encoding="utf-8"))
    assert lock["scorer_sha"] == snap["scorer_sha"]


if __name__ == "__main__":
    test_locks_fail_closed_no_rewrite()
    test_same_s_hash_one_store()
    test_validate_requires_y_motor_edges()
    test_trace_validation_and_motors_differ()
    test_extract_states_no_path_and_frontier()
    test_h3a_collides_h3b_distinguishes()
    test_outgoing_y_motor_fid_inadmissible()
    test_computed_table_origin_insufficient()
    test_never_stamps_004()
    test_scorer_sha_pinned()
    test_no_write_lock_in_tests_module()
    print("ok")
