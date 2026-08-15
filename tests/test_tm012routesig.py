"""TM.0.12.ROUTESIG phase 1: C10 order contract — rules, not the result table."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import CONTEXT_LOCK, MOTORS
from experiments.run_tm012minimap import (
    MINIMAP_LOCK,
    OUTCOME_COLLISION,
    OUTCOME_DISTINGUISHES,
    score_contrast,
)
from experiments.run_tm012pathdisc import PATHDISC_LOCK
from experiments.run_tm012midpath import MIDPATH_LOCK
from experiments.run_tm012routesig import (
    CELL_ID,
    ROLE_ORDER,
    ROUTESIG_CANDIDATES,
    ROUTESIG_LOCK,
    SUFFIX_K,
    TRACE_LEN,
    edge_fid,
    extract_states_routesig,
    gen_c10,
    kappa_seed,
    kappa_step,
    path_edge_fids,
    route_kappa,
    run_routesig,
    validate_c10_pair,
    verify_routesig_lock,
)


def test_locks_fail_closed():
    ok, why, snap = verify_routesig_lock()
    assert ok, why
    assert snap["earned_next"] is False
    assert snap["phase"] == "order"
    assert snap["suffix_k"] == SUFFIX_K == 2
    assert snap["cell_id"] == CELL_ID
    lock = json.loads(ROUTESIG_LOCK.read_text(encoding="utf-8"))
    assert lock["context_012_lock_sha"] == hashlib.sha256(CONTEXT_LOCK.read_bytes()).hexdigest()
    assert lock["minimap_012_lock_sha"] == hashlib.sha256(MINIMAP_LOCK.read_bytes()).hexdigest()
    assert lock["pathdisc_012_lock_sha"] == hashlib.sha256(PATHDISC_LOCK.read_bytes()).hexdigest()
    assert lock["midpath_012_lock_sha"] == hashlib.sha256(MIDPATH_LOCK.read_bytes()).hexdigest()
    assert lock["candidates"] == list(ROUTESIG_CANDIDATES)
    assert "route_kappa(origin, ordered_path_fids)" in lock["kappa_api"]
    assert "trace_spec_sha" in lock
    assert "same path-edge fid multiset" in lock["validation"]
    assert any("exactly one relation per directed" in v for v in lock["validation"])


def test_same_edge_fid_set_different_order():
    cell = gen_c10(12345)
    a, b = cell.traces
    assert len(a.nodes) == TRACE_LEN == len(b.nodes)
    fa = path_edge_fids(cell.relations, a.nodes)
    fb = path_edge_fids(cell.relations, b.nodes)
    assert frozenset(fa) == frozenset(fb)
    assert fa != fb
    assert fa[-2:] == fb[-2:]
    assert sorted(n.lower() for n in a.nodes) == sorted(n.lower() for n in b.nodes)
    errs = validate_c10_pair(
        cell.relations,
        a,
        b,
        origin=cell.origin,
        hub=cell.hub,
        predecessor=cell.predecessor,
        frontier=cell.frontier,
    )
    assert errs == []


def test_edge_fid_requires_unique():
    cell = gen_c10(12345)
    from experiments.run_tm012context import Rel

    dup = list(cell.relations)
    e = cell.relations[0]
    dup.append(Rel("n99999", e.bind, e.did, "dup", e.init, here=e.here))
    try:
        edge_fid(dup, e.bind, e.did)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "exactly one" in str(exc)


def test_r1_collides_r3_r4_distinguish():
    cell = gen_c10(12345)
    a, b = cell.traces
    sa = extract_states_routesig(cell.relations, a.nodes)
    sb = extract_states_routesig(cell.relations, b.nodes)
    assert sa["R1"] == sb["R1"]
    assert sa["R2"] == sb["R2"]
    assert sa["R0"] == sb["R0"]
    assert sa["R3"] != sb["R3"]
    assert sa["R4"] != sb["R4"]
    for cand, want in (
        ("R0", OUTCOME_COLLISION),
        ("R1", OUTCOME_COLLISION),
        ("R2", OUTCOME_COLLISION),
        ("R3", OUTCOME_DISTINGUISHES),
        ("R4", OUTCOME_DISTINGUISHES),
    ):
        assert (
            score_contrast(
                contrast_id="C10",
                role=ROLE_ORDER,
                left_states=sa,
                right_states=sb,
                left_motor=a.required_motor,
                right_motor=b.required_motor,
                candidate=cand,
            )
            == want
        ), cand


def test_global_unique_directed_edges():
    cell = gen_c10(12345)
    from experiments.run_tm012context import Rel

    # Duplicate a non-path-critical? Any duplicate (bind,did) must fail.
    # Use hub→predecessor (on path) — already covered. Also unused-style:
    # duplicate Y→motor.
    dup = list(cell.relations)
    ym = [r for r in cell.relations if r.bind.lower() == cell.frontier.lower()][0]
    dup.append(Rel("n77777", ym.bind, ym.did, "dup_m", ym.init, here=ym.here))
    errs = validate_c10_pair(
        dup,
        cell.traces[0],
        cell.traces[1],
        origin=cell.origin,
        hub=cell.hub,
        predecessor=cell.predecessor,
        frontier=cell.frontier,
    )
    assert any("duplicate directed edge" in e for e in errs)


def test_kappa_output_blind_behavioral():
    """R4 depends on path nodes/fids only — TraceSpec.motor cannot change κ."""
    cell = gen_c10(12345)
    nodes = cell.traces[0].nodes
    from experiments.run_tm012pathdisc import TraceSpec

    t_press = TraceSpec("x", nodes, "press")
    t_tune = TraceSpec("x", nodes, "tune")
    assert t_press.required_motor != t_tune.required_motor
    # Extractor takes nodes only (not TraceSpec) — R4 identical.
    r4_press = extract_states_routesig(cell.relations, t_press.nodes)["R4"]
    r4_tune = extract_states_routesig(cell.relations, t_tune.nodes)["R4"]
    assert r4_press == r4_tune
    fids = path_edge_fids(cell.relations, nodes)
    assert route_kappa(cell.origin, fids) == r4_press[1]


def test_kappa_api_signature_no_tracespec():
    for fn in (kappa_seed, kappa_step, route_kappa):
        params = set(inspect.signature(fn).parameters)
        assert "trace" not in params
        assert "motor" not in params
        assert "required_motor" not in params
        assert "context_expect" not in params
        src = inspect.getsource(fn)
        assert "TraceSpec" not in src
        assert "required_motor" not in src


def test_kappa_shuffle_invariant():
    cell = gen_c10(12345)
    nodes = cell.traces[0].nodes
    fa = path_edge_fids(cell.relations, nodes)
    shuffled = list(cell.relations)
    random.Random(0).shuffle(shuffled)
    reversed_rels = list(reversed(cell.relations))
    assert route_kappa(cell.origin, fa) == route_kappa(
        cell.origin, path_edge_fids(shuffled, nodes)
    )
    assert route_kappa(cell.origin, fa) == route_kappa(
        cell.origin, path_edge_fids(reversed_rels, nodes)
    )


def test_runtime_seed_must_match_lock():
    bad = run_routesig(seed=99999)
    assert bad["ok"] is False
    assert "runtime seed" in bad["why"]


def test_computed_table_order_necessary():
    summary = run_routesig(seed=12345)
    assert summary["ok"], summary.get("why")
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert summary["order_necessary"] is True
    assert summary["membership_insufficient"] is True
    assert summary["kappa_shuffle_invariant"] is True
    table = summary["table"]
    assert table["R0"]["C10"] == OUTCOME_COLLISION
    assert table["R1"]["C10"] == OUTCOME_COLLISION
    assert table["R2"]["C10"] == OUTCOME_COLLISION
    assert table["R3"]["C10"] == OUTCOME_DISTINGUISHES
    assert table["R4"]["C10"] == OUTCOME_DISTINGUISHES
    claim = summary["claim"]
    assert "order is necessary" in claim
    assert "genome primitive" in claim or "SHA-256" in claim


def test_never_stamps_004():
    summary = run_routesig(seed=12345)
    assert summary["earned_next"] is False
    lock = summary.get("lock") or {}
    assert "stamp Ex0S 0.0.004" in (lock.get("refuse") or [])


def test_no_write_lock_in_tests():
    src = Path(__file__).read_text(encoding="utf-8")
    banned = [
        "write_" + "routesig_lock(",
        "write_" + "midpath_lock(",
        "write_" + "pathdisc_lock(",
        "write_" + "context_lock",
        "write_" + "minimap_lock",
    ]
    for phrase in banned:
        assert phrase not in src


if __name__ == "__main__":
    test_locks_fail_closed()
    test_same_edge_fid_set_different_order()
    test_edge_fid_requires_unique()
    test_global_unique_directed_edges()
    test_r1_collides_r3_r4_distinguish()
    test_kappa_output_blind_behavioral()
    test_kappa_api_signature_no_tracespec()
    test_kappa_shuffle_invariant()
    test_runtime_seed_must_match_lock()
    test_computed_table_order_necessary()
    test_never_stamps_004()
    test_no_write_lock_in_tests()
    print("ok")
