"""TM.0.12.MIDPATH: rules — C9 geometry, projected-states-only, no 0.0.004."""

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
from experiments.run_tm012pathdisc import PATHDISC_LOCK, find_edge, refuse_answer_derived_fid
from experiments.run_tm012midpath import (
    CELL_ID,
    MIDPATH_CANDIDATES,
    MIDPATH_LOCK,
    ROLE_MIDPATH_VS_ENDPOINT,
    TRACE_LEN,
    extract_states_midpath,
    gen_c9,
    run_midpath,
    validate_c9_pair,
    verify_midpath_lock,
)


def test_locks_fail_closed_no_rewrite():
    ok, why, snap = verify_midpath_lock()
    assert ok, why
    assert snap["earned_next"] is False
    assert snap["cell_id"] == CELL_ID
    assert snap["role"] == ROLE_MIDPATH_VS_ENDPOINT
    assert MIDPATH_LOCK.exists()
    lock = json.loads(MIDPATH_LOCK.read_text(encoding="utf-8"))
    assert lock["context_012_lock_sha"] == hashlib.sha256(CONTEXT_LOCK.read_bytes()).hexdigest()
    assert lock["minimap_012_lock_sha"] == hashlib.sha256(MINIMAP_LOCK.read_bytes()).hexdigest()
    assert lock["pathdisc_012_lock_sha"] == hashlib.sha256(PATHDISC_LOCK.read_bytes()).hexdigest()
    for key in (
        "score_contrast_sha",
        "find_edge_sha",
        "relations_content_hash_sha",
        "refuse_answer_derived_fid_sha",
        "trace_spec_sha",
        "gen_c9_sha",
        "validate_c9_sha",
        "midpath_extractor_sha",
    ):
        assert key in lock
    assert "raw apparatus trace is extractor-only ground truth" in " ".join(lock["validation"])
    assert "raw TraceSpec / nodes as selector input" in lock["refuse"]
    assert "exactly one P→Y" in " ".join(lock["validation"])


def test_geometry_lock_length4_one_interior():
    cell = gen_c9(12345)
    a, b = cell.traces
    assert len(a.nodes) == TRACE_LEN == len(b.nodes)
    na = [n.lower() for n in a.nodes]
    nb = [n.lower() for n in b.nodes]
    assert na[0] == nb[0] == cell.origin.lower()
    assert na[1] != nb[1]
    assert na[2] == nb[2] == cell.predecessor.lower()
    assert na[3] == nb[3] == cell.frontier.lower()
    errs = validate_c9_pair(
        cell.relations,
        a,
        b,
        origin=cell.origin,
        predecessor=cell.predecessor,
        frontier=cell.frontier,
    )
    assert errs == []
    for r in cell.relations:
        assert r.here == "chb"
        assert r.init == (1, 0)


def test_extract_no_path_and_frontier_and_h3c_local():
    src = inspect.getsource(extract_states_midpath)
    assert "path_and_frontier" not in src
    assert MIDPATH_CANDIDATES == ("H0", "H1", "H2", "H3a", "H3c", "H3b", "H4")
    # MINIMAP candidate order must not gain H3c.
    from experiments.run_tm012minimap import CANDIDATE_ORDER

    assert "H3c" not in CANDIDATE_ORDER
    cell = gen_c9(12345)
    sa = extract_states_midpath(cell.relations, cell.traces[0])
    sb = extract_states_midpath(cell.relations, cell.traces[1])
    assert sa["H3c"] == sb["H3c"]
    assert sa["H3b"] != sb["H3b"]
    assert sa["H2"] == sb["H2"]
    assert sa["H3a"] == sb["H3a"]
    assert sa["H4"] == sb["H4"]


def test_h3c_collides_h3b_distinguishes():
    cell = gen_c9(12345)
    a, b = cell.traces
    sa = extract_states_midpath(cell.relations, a)
    sb = extract_states_midpath(cell.relations, b)
    assert (
        score_contrast(
            contrast_id="C9",
            role=ROLE_MIDPATH_VS_ENDPOINT,
            left_states=sa,
            right_states=sb,
            left_motor=a.required_motor,
            right_motor=b.required_motor,
            candidate="H3c",
        )
        == OUTCOME_COLLISION
    )
    assert (
        score_contrast(
            contrast_id="C9",
            role=ROLE_MIDPATH_VS_ENDPOINT,
            left_states=sa,
            right_states=sb,
            left_motor=a.required_motor,
            right_motor=b.required_motor,
            candidate="H3b",
        )
        == OUTCOME_DISTINGUISHES
    )


def test_unique_shared_py_edge_required():
    cell = gen_c9(12345)
    a, b = cell.traces
    from experiments.run_tm012context import Rel

    dup = list(cell.relations)
    py = find_edge(cell.relations, cell.predecessor, cell.frontier)
    assert py is not None
    dup.append(Rel("n99999", py.bind, py.did, "py_dup", py.init, here=py.here))
    errs = validate_c9_pair(
        dup,
        a,
        b,
        origin=cell.origin,
        predecessor=cell.predecessor,
        frontier=cell.frontier,
    )
    assert any("exactly one P→Y" in e for e in errs)


def test_scorer_receives_projected_states_only():
    """score_contrast signature has no trace/nodes parameter."""
    sig = inspect.signature(score_contrast)
    assert "trace" not in sig.parameters
    assert "nodes" not in sig.parameters
    assert set(sig.parameters) >= {
        "contrast_id",
        "role",
        "left_states",
        "right_states",
        "left_motor",
        "right_motor",
        "candidate",
    }
    src = inspect.getsource(run_midpath)
    # Only one score_contrast call; kwargs must be projected states + motors.
    assert src.count("score_contrast(") == 1
    call = src.split("score_contrast(")[1].split(")", 1)[0]
    assert "left_states=sa" in call
    assert "right_states=sb" in call
    assert "nodes" not in call
    assert "trace" not in call.lower()


def test_outgoing_fid_inadmissible_shared_incoming():
    cell = gen_c9(12345)
    a = cell.traces[0]
    out = find_edge(cell.relations, cell.frontier, a.required_motor)
    inc = find_edge(cell.relations, cell.predecessor, cell.frontier)
    assert out is not None and inc is not None
    assert refuse_answer_derived_fid(inc.fid, out.fid) == OUTCOME_INADMISSIBLE


def test_runtime_seed_must_match_lock():
    bad = run_midpath(seed=99999)
    assert bad["ok"] is False
    assert "runtime seed" in bad["why"]
    assert bad["earned_next"] is False


def test_computed_table_endpoint_provenance_insufficient():
    summary = run_midpath(seed=12345)
    assert summary["ok"], summary.get("why")
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert summary["endpoint_provenance_insufficient"] is True
    assert summary["h3b_survives"] is True
    assert summary["scorer_inputs"] == "projected_states_and_motors_only"
    table = summary["table"]
    for cand in ("H0", "H1", "H2", "H3a", "H3c", "H4"):
        assert table[cand]["C9"] == OUTCOME_COLLISION, cand
    assert table["H3b"]["C9"] == OUTCOME_DISTINGUISHES
    claim = summary["claim"]
    assert "Endpoint provenance is insufficient" in claim
    assert "store-the-full-path" in claim or "full path" in claim.lower()
    assert "route-signature" in claim


def test_never_stamps_004():
    summary = run_midpath(seed=12345)
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    lock = summary.get("lock") or {}
    assert "stamp Ex0S 0.0.004" in (lock.get("refuse") or [])


def test_no_write_lock_in_tests_module():
    src = Path(__file__).read_text(encoding="utf-8")
    banned = [
        "write_" + "midpath_lock(",
        "write_" + "pathdisc_lock(",
        "write_" + "context_lock",
        "write_" + "minimap_lock",
    ]
    for phrase in banned:
        assert phrase not in src


if __name__ == "__main__":
    test_locks_fail_closed_no_rewrite()
    test_geometry_lock_length4_one_interior()
    test_extract_no_path_and_frontier_and_h3c_local()
    test_h3c_collides_h3b_distinguishes()
    test_scorer_receives_projected_states_only()
    test_outgoing_fid_inadmissible_shared_incoming()
    test_unique_shared_py_edge_required()
    test_runtime_seed_must_match_lock()
    test_computed_table_endpoint_provenance_insufficient()
    test_never_stamps_004()
    test_no_write_lock_in_tests_module()
    print("ok")
