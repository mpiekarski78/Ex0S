"""TM.0.12.ROUTESIG.IDENTITY: C12A/C12B rules — no alpha, no 0.0.004."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import CONTEXT_LOCK
from experiments.run_tm012minimap import MINIMAP_LOCK
from experiments.run_tm012pathdisc import PATHDISC_LOCK, TraceSpec
from experiments.run_tm012midpath import MIDPATH_LOCK
from experiments.run_tm012routesig import ROUTESIG_LOCK, path_edge_fids, route_kappa
from experiments.run_tm012routesig_depth import DEPTH_LOCK
from experiments.run_tm012routesig_identity import (
    IDENTITY_LOCK,
    OUTCOME_DIFFERS,
    OUTCOME_SAME,
    assert_c12a_geometry,
    assert_c12b_geometry,
    edge_sem,
    extract_identity,
    gen_c12a,
    gen_c12b,
    path_edge_sems,
    run_identity,
    validate_pair,
    verify_identity_lock,
)


def test_locks_fail_closed():
    ok, why, snap = verify_identity_lock()
    assert ok, why
    assert snap["earned_next"] is False
    assert snap["phase"] == "identity"
    assert snap["ksem_fields"] == ["bind", "did"]
    lock = json.loads(IDENTITY_LOCK.read_text(encoding="utf-8"))
    assert lock["routesig_012_lock_sha"] == hashlib.sha256(ROUTESIG_LOCK.read_bytes()).hexdigest()
    assert lock["routesig_depth_012_lock_sha"] == hashlib.sha256(DEPTH_LOCK.read_bytes()).hexdigest()
    assert lock["context_012_lock_sha"] == hashlib.sha256(CONTEXT_LOCK.read_bytes()).hexdigest()
    assert lock["minimap_012_lock_sha"] == hashlib.sha256(MINIMAP_LOCK.read_bytes()).hexdigest()
    assert lock["pathdisc_012_lock_sha"] == hashlib.sha256(PATHDISC_LOCK.read_bytes()).hexdigest()
    assert lock["midpath_012_lock_sha"] == hashlib.sha256(MIDPATH_LOCK.read_bytes()).hexdigest()
    order = json.loads(ROUTESIG_LOCK.read_text(encoding="utf-8"))
    assert lock["gen_c10_sha"] == order["gen_c10_sha"]
    assert lock["trace_spec_sha"] == hashlib.sha256(inspect.getsource(TraceSpec).encode()).hexdigest()


def test_c12a_geometry():
    pair = gen_c12a(12345)
    assert validate_pair(pair) == []
    assert assert_c12a_geometry(pair) == []
    left, right = pair.left, pair.right
    assert {r.fid for r in left.relations}.isdisjoint({r.fid for r in right.relations})
    for tl, tr in zip(left.traces, right.traces):
        assert [n.lower() for n in tl.nodes] == [n.lower() for n in tr.nodes]
        fa = path_edge_fids(left.relations, tl.nodes)
        fb = path_edge_fids(right.relations, tr.nodes)
        assert all(a != b for a, b in zip(fa, fb))
        assert path_edge_sems(left.relations, tl.nodes) == path_edge_sems(right.relations, tr.nodes)


def test_c12b_geometry():
    pair = gen_c12b(12345)
    assert validate_pair(pair) == []
    assert assert_c12b_geometry(pair) == []
    left_fids = {r.fid for r in pair.left.relations}
    inherited = [r for r in pair.right.relations if r.fid in left_fids]
    assert len(inherited) == len(pair.left.relations)
    left_by = {r.fid: r for r in pair.left.relations}
    changed = [r for r in inherited if (left_by[r.fid].bind.lower(), left_by[r.fid].did.lower()) != (r.bind.lower(), r.did.lower())]
    assert len(changed) == 1
    for tl, tr in zip(pair.left.traces, pair.right.traces):
        na = [n.lower() for n in tl.nodes]
        nb = [n.lower() for n in tr.nodes]
        assert na[:-1] == nb[:-1]
        assert na[-1] != nb[-1]
        assert path_edge_fids(pair.left.relations, tl.nodes) == path_edge_fids(
            pair.right.relations, tr.nodes
        )
        sa = path_edge_sems(pair.left.relations, tl.nodes)
        sb = path_edge_sems(pair.right.relations, tr.nodes)
        assert sa[:-1] == sb[:-1]
        assert sa[-1] != sb[-1]


def test_computed_table():
    summary = run_identity(seed=12345)
    assert summary["ok"], summary.get("why")
    assert summary["earned_next"] is False
    table = summary["table"]
    assert table["Kfid"]["C12A"] == OUTCOME_DIFFERS
    assert table["Ksem"]["C12A"] == OUTCOME_SAME
    assert table["Kfid"]["C12B"] == OUTCOME_SAME
    assert table["Ksem"]["C12B"] == OUTCOME_DIFFERS
    assert summary["fid_is_storage_identity"] is True
    assert summary["sem_is_relation_identity"] is True


def test_ksem_payload_is_endpoints_only():
    src = inspect.getsource(edge_sem)
    for banned in ("fid", "here", "support", "role", "init"):
        assert banned not in src
    a = edge_sem("Dog", "Mammal")
    b = edge_sem("dog", "mammal")
    assert a == b == "dog\0mammal"


def test_order_f_cross_pinned():
    lock = json.loads(IDENTITY_LOCK.read_text(encoding="utf-8"))
    order = json.loads(ROUTESIG_LOCK.read_text(encoding="utf-8"))
    for k in (
        "kappa_seed_sha",
        "kappa_step_sha",
        "route_kappa_sha",
        "edge_fid_sha",
        "path_edge_fids_sha",
        "gen_c10_sha",
    ):
        assert lock[k] == order[k], k


def test_shuffle_invariant():
    pair = gen_c12a(12345)
    nodes = pair.left.traces[0].nodes
    sl = extract_identity(pair.left.relations, nodes)
    shuffled = list(pair.left.relations)
    random.Random(0).shuffle(shuffled)
    assert extract_identity(shuffled, nodes) == sl


def test_same_f_different_payload():
    pair = gen_c12a(12345)
    nodes = pair.left.traces[0].nodes
    sl = extract_identity(pair.left.relations, nodes)
    assert sl["Kfid"] == route_kappa(pair.left.origin, path_edge_fids(pair.left.relations, nodes))
    assert sl["Ksem"] == route_kappa(pair.left.origin, path_edge_sems(pair.left.relations, nodes))
    assert sl["Kfid"] != sl["Ksem"]


def test_runtime_seed_and_no_004():
    bad = run_identity(seed=99999)
    assert bad["ok"] is False
    assert "runtime seed" in bad["why"]
    summary = run_identity(seed=12345)
    assert summary["ex0s_under_test"] == "0.0.003"
    assert "stamp Ex0S 0.0.004" in (summary.get("lock") or {}).get("refuse", [])
    assert "alpha-rename" in " ".join((summary.get("lock") or {}).get("refuse", []))


def test_no_write_lock_in_tests():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "write_" + "identity_lock(" not in src


if __name__ == "__main__":
    test_locks_fail_closed()
    test_c12a_geometry()
    test_c12b_geometry()
    test_computed_table()
    test_ksem_payload_is_endpoints_only()
    test_order_f_cross_pinned()
    test_shuffle_invariant()
    test_same_f_different_payload()
    test_runtime_seed_and_no_004()
    test_no_write_lock_in_tests()
    print("ok")
