"""TM.0.14.FAMILY: A–D behavioral smoke; E–H sealed (no organism answers)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import verify_genome_014
from experiments.run_tm014family import (
    DEVELOP,
    EXPECTED_N,
    FAMILY_LOCK,
    HOLDOUT,
    PREREGISTERED_CLAIM,
    generate_world,
    holdout_manifests,
    run_family,
    run_one,
    seed_jobs_sha,
    verify_family_lock,
    verify_holdout_sealed,
    world_manifest_sha,
)


def test_locks_and_genome():
    ok, why, _ = verify_genome_014()
    assert ok, why
    ok, why, snap = verify_family_lock()
    assert ok, why
    assert snap["expected_n"] == EXPECTED_N
    lock = json.loads(FAMILY_LOCK.read_text(encoding="utf-8"))
    assert lock["ex0s_if_earned"] is None
    assert lock["preregistered_claim"] == PREREGISTERED_CLAIM
    assert "behavioral contact with E-H before canonical run" in lock["refuse"]
    assert "pre-name Ex0S product version" in lock["refuse"]


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


def test_manifest_roundtrip_no_ctx():
    for fam in HOLDOUT:
        w = generate_world(fam, 12345 + 1000 * ("ABCDEFGH".index(fam)), 0)
        assert w.holdout
        assert all("ctx" not in e for e in w.birth_edges)
        w2 = type(w).from_manifest(w.to_manifest())
        assert world_manifest_sha(w) == world_manifest_sha(w2)
        assert w.primary_life
        assert w.interventions


def test_develop_smoke_behavioral():
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
    for fam in DEVELOP:
        assert summary["families"][fam]["solved"] == 1


def test_holdout_sealed_from_run_one():
    with tempfile.TemporaryDirectory() as d:
        row = run_one(
            {
                "family": "E",
                "seed": 12345 + 4000,
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
        raise AssertionError("expected refuse")
    except ValueError as e:
        assert "canonical" in str(e).lower() or "holdout" in str(e).lower()


if __name__ == "__main__":
    test_locks_and_genome()
    test_holdout_sealed_no_organism()
    test_seed_jobs_canonical()
    test_manifest_roundtrip_no_ctx()
    test_develop_smoke_behavioral()
    test_holdout_sealed_from_run_one()
    test_refuse_partial_holdout_peek()
    print("ok")
