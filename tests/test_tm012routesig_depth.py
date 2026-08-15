"""TM.0.12.ROUTESIG.DEPTH: rules — geometric R2/R2x, k=1..8, no 0.0.004."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import CONTEXT_LOCK, Rel
from experiments.run_tm012minimap import (
    MINIMAP_LOCK,
    OUTCOME_COLLISION,
    OUTCOME_DISTINGUISHES,
)
from experiments.run_tm012pathdisc import PATHDISC_LOCK, TraceSpec
from experiments.run_tm012midpath import MIDPATH_LOCK
from experiments.run_tm012routesig import (
    ROUTESIG_LOCK,
    kappa_seed,
    kappa_step,
    path_edge_fids,
    route_kappa,
)
from experiments.run_tm012routesig_depth import (
    DEPTH_LOCK,
    K_FAMILY,
    MOTOR_A,
    MOTOR_B,
    assert_geometry_k,
    extract_states_depth,
    gen_c11_store,
    run_depth,
    traces_for_k,
    validate_store,
    verify_depth_lock,
)


def test_locks_fail_closed():
    ok, why, snap = verify_depth_lock()
    assert ok, why
    assert snap["earned_next"] is False
    assert snap["phase"] == "depth"
    assert snap["k_family"] == list(K_FAMILY)
    lock = json.loads(DEPTH_LOCK.read_text(encoding="utf-8"))
    assert lock["routesig_012_lock_sha"] == hashlib.sha256(ROUTESIG_LOCK.read_bytes()).hexdigest()
    assert lock["context_012_lock_sha"] == hashlib.sha256(CONTEXT_LOCK.read_bytes()).hexdigest()
    assert lock["minimap_012_lock_sha"] == hashlib.sha256(MINIMAP_LOCK.read_bytes()).hexdigest()
    assert lock["pathdisc_012_lock_sha"] == hashlib.sha256(PATHDISC_LOCK.read_bytes()).hexdigest()
    assert lock["midpath_012_lock_sha"] == hashlib.sha256(MIDPATH_LOCK.read_bytes()).hexdigest()
    assert lock["motors"] == {"route_a": MOTOR_A, "route_b": MOTOR_B}
    assert "trace_spec_sha" in lock
    assert lock["trace_spec_sha"] == hashlib.sha256(
        inspect.getsource(TraceSpec).encode()
    ).hexdigest()
    assert any("path-edge fids subset" in v for v in lock["validation"])


def test_geometry_each_k():
    store = gen_c11_store(12345)
    assert validate_store(store) == []
    for k in K_FAMILY:
        ta, tb, fr = traces_for_k(store, k)
        assert ta.nodes[-1].lower() == fr.lower() == store.chain[k - 1].lower()
        assert ta.required_motor == MOTOR_A
        assert tb.required_motor == MOTOR_B
        errs = assert_geometry_k(store.relations, ta.nodes, tb.nodes, k)
        assert errs == [], (k, errs)
        fa = path_edge_fids(store.relations, ta.nodes)
        fb = path_edge_fids(store.relations, tb.nodes)
        assert fa[-k:] == fb[-k:]
        assert fa[-(k + 1)] != fb[-(k + 1)]
        assert fa[-(k + 1) + 1 :] == fb[-(k + 1) + 1 :]


def test_distinct_frontiers():
    store = gen_c11_store(12345)
    fronts = [traces_for_k(store, k)[2].lower() for k in K_FAMILY]
    assert len(set(fronts)) == 8


def test_global_unique_directed_edges():
    store = gen_c11_store(12345)
    dup = list(store.relations)
    ym = [r for r in store.relations if r.bind.lower() == store.chain[0].lower()][0]
    dup.append(Rel("n77777", ym.bind, ym.did, "dup_m", ym.init, here=ym.here))
    bad = type(store)(
        relations=dup,
        origin=store.origin,
        hub=store.hub,
        a_node=store.a_node,
        b_node=store.b_node,
        chain=store.chain,
    )
    errs = validate_store(bad)
    assert any("duplicate directed edge" in e for e in errs)


def test_computed_table_all_k():
    summary = run_depth(seed=12345)
    assert summary["ok"], summary.get("why")
    assert summary["earned_next"] is False
    assert summary["depth_windows_insufficient"] is True
    assert summary["kappa_preserves_tested_depths"] is True
    assert summary["c11_2_matches_c10_suffix2_pattern"] is True
    table = summary["table"]
    for k in K_FAMILY:
        col = f"C11[{k}]"
        assert table["R1"][col] == OUTCOME_COLLISION
        assert table["R2"][col] == OUTCOME_COLLISION
        assert table["R2x"][col] == OUTCOME_DISTINGUISHES
        assert table["R3"][col] == OUTCOME_DISTINGUISHES
        assert table["R4"][col] == OUTCOME_DISTINGUISHES
    sizes = {row["k"]: row for row in summary["size_vs_k"]}
    assert sizes[1]["R2_fid_count"] == 1
    assert sizes[8]["R2_fid_count"] == 8
    assert sizes[1]["R3_fid_count"] < sizes[8]["R3_fid_count"]
    assert all(row["R4_bits"] == 256 for row in summary["size_vs_k"])


def test_kappa_output_blind_behavioral():
    """R4 depends on path nodes/fids only — TraceSpec.motor cannot change κ."""
    store = gen_c11_store(12345)
    nodes = traces_for_k(store, 3)[0].nodes
    t_press = TraceSpec("x", nodes, "press")
    t_tune = TraceSpec("x", nodes, "tune")
    assert t_press.required_motor != t_tune.required_motor
    r4_press = extract_states_depth(store.relations, t_press.nodes, k=3)["R4"]
    r4_tune = extract_states_depth(store.relations, t_tune.nodes, k=3)["R4"]
    assert r4_press == r4_tune
    fids = path_edge_fids(store.relations, nodes)
    assert route_kappa(store.origin, fids) == r4_press[1]
    bad = run_depth(seed=99999)
    assert bad["ok"] is False
    assert "runtime seed" in bad["why"]


def test_kappa_api_and_extract_guards():
    for fn in (kappa_seed, kappa_step, route_kappa):
        src = inspect.getsource(fn)
        assert "TraceSpec" not in src
        assert "required_motor" not in src
    assert "path_and_frontier" not in inspect.getsource(extract_states_depth)


def test_kappa_shuffle_invariant():
    store = gen_c11_store(12345)
    nodes = traces_for_k(store, 8)[0].nodes
    fa = path_edge_fids(store.relations, nodes)
    shuffled = list(store.relations)
    random.Random(0).shuffle(shuffled)
    assert route_kappa(store.origin, fa) == route_kappa(
        store.origin, path_edge_fids(shuffled, nodes)
    )


def test_never_stamps_004():
    summary = run_depth(seed=12345)
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert "stamp Ex0S 0.0.004" in (summary.get("lock") or {}).get("refuse", [])


def test_no_write_lock_in_tests():
    src = Path(__file__).read_text(encoding="utf-8")
    banned = ["write_" + "depth_lock(", "write_" + "routesig_lock("]
    for phrase in banned:
        assert phrase not in src


if __name__ == "__main__":
    test_locks_fail_closed()
    test_geometry_each_k()
    test_distinct_frontiers()
    test_global_unique_directed_edges()
    test_computed_table_all_k()
    test_kappa_output_blind_behavioral()
    test_kappa_api_and_extract_guards()
    test_kappa_shuffle_invariant()
    test_never_stamps_004()
    test_no_write_lock_in_tests()
    print("ok")
