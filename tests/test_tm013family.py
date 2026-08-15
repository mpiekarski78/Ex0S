"""TM.0.13.FAMILY: A–D behavioral smoke; E–H sealed (no organism answers)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm013family import (
    DEVELOP,
    EXPECTED_N,
    FAMILY_LOCK,
    HOLDOUT,
    MANDATORY_MEASURES,
    generate_world,
    holdout_manifests,
    lexical_edge_order,
    order_changing_fid_remap,
    run_family,
    run_one,
    seed_jobs_sha,
    verify_family_lock,
    verify_holdout_sealed,
    world_manifest_sha,
)
from experiments.run_tm013context import verify_genome_013, verify_kappa_vectors


def test_locks_and_kappa():
    ok, why, _ = verify_kappa_vectors()
    assert ok, why
    ok, why, _ = verify_genome_013()
    assert ok, why
    ok, why, snap = verify_family_lock()
    assert ok, why
    assert snap["expected_n"] == EXPECTED_N
    assert snap["earned_next"] is False
    lock = json.loads(FAMILY_LOCK.read_text(encoding="utf-8"))
    assert lock["ex0s_if_earned"] == "0.0.004"
    assert lock["claim_name_if_earned"] == "Contextual Composition"
    assert "behavioral contact with E-H before canonical run" in lock["refuse"]


def test_holdout_sealed_no_organism():
    ok, why, snap = verify_holdout_sealed()
    assert ok, why
    assert snap.get("holdout_sealed_ok") is True
    live = holdout_manifests()
    lock = json.loads(FAMILY_LOCK.read_text(encoding="utf-8"))
    assert live["holdout_manifest_sha"] == lock["holdout_manifest_sha"]
    assert live["n_holdout_worlds"] == 144


def test_seed_jobs_canonical():
    assert seed_jobs_sha() == json.loads(FAMILY_LOCK.read_text())["seed_jobs_sha"]


def test_order_changing_fid_remap():
    w = generate_world("D", 12345, 0)
    remapped = order_changing_fid_remap(w.relations_primary + w.clutter, 12345)
    assert lexical_edge_order(remapped) != lexical_edge_order(w.relations_primary + w.clutter)
    # Semantics preserved
    assert {(r.bind, r.did, r.ctx) for r in remapped} == {
        (r.bind, r.did, r.ctx) for r in (w.relations_primary + w.clutter)
    }


def test_manifest_roundtrip():
    for fam in HOLDOUT:
        w = generate_world(fam, 12345, 0)
        assert w.holdout
        w2 = type(w).from_manifest(w.to_manifest())
        assert world_manifest_sha(w) == world_manifest_sha(w2)


def test_develop_smoke_behavioral():
    """A–D only — never probes E–H."""
    summary = run_family(
        seed=11,
        per_family=1,
        births=1,
        workers=1,
        families=DEVELOP,
        allow_holdout_behavior=False,
    )
    assert summary["n_worlds"] == 4
    assert summary["solved_frac"] == 1.0, summary["families"]
    assert summary["ex0s"] is None
    assert summary["earned_next"] is False
    assert summary["allow_holdout_behavior"] is False
    for fam in DEVELOP:
        assert summary["families"][fam]["solved"] == 1


def test_holdout_sealed_from_run_one():
    """Even if called, E–H without allow flag must not answer."""
    with tempfile.TemporaryDirectory() as d:
        row = run_one(
            {
                "family": "E",
                "seed": 12345,
                "birth": 0,
                "dest": str(Path(d) / "e"),
                "genome_ok": True,
                "allow_holdout_behavior": False,
            }
        )
    assert row.get("sealed") is True
    assert row["solved"] is False
    assert "holdout_behavior_sealed" in row["errors"]


def test_refuse_partial_holdout_peek():
    try:
        run_family(
            seed=11,
            per_family=1,
            births=1,
            workers=1,
            families=("E",),
            allow_holdout_behavior=True,
        )
    except ValueError as exc:
        assert "refuse" in str(exc)
        return
    raise AssertionError("expected refuse ValueError for partial E–H peek")


def test_abcd_structurally_distinct():
    a = generate_world("A", 12345, 0)
    b = generate_world("B", 12345, 0)
    c = generate_world("C", 12345, 0)
    assert a.hops_primary != b.hops_primary
    assert c.feature_off_expect == "flip"
    assert any(r.role == "trap0" for r in c.relations_primary)


def test_mandatory_measures_list():
    assert "ctx_no_fallback" in MANDATORY_MEASURES
    assert "rho_reset_same_agent" in MANDATORY_MEASURES
    assert "newborn_reload" in MANDATORY_MEASURES
    assert "storage_identity_order_invariance" in MANDATORY_MEASURES


if __name__ == "__main__":
    test_locks_and_kappa()
    test_holdout_sealed_no_organism()
    test_seed_jobs_canonical()
    test_order_changing_fid_remap()
    test_manifest_roundtrip()
    test_holdout_sealed_from_run_one()
    test_mandatory_measures_list()
    test_abcd_structurally_distinct()
    test_refuse_partial_holdout_peek()
    test_develop_smoke_behavioral()
    print("ok")
