"""TM027/TM027.R2 tests — classifier, compat, historical immutability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

V1_DEV = REPO / "docs" / "lineage_gatedrehearsal.dev.lock"
V1_DEC = REPO / "docs" / "lineage_gatedrehearsal.decision.lock"
V1_ADD = REPO / "docs" / "lineage_gatedrehearsal.v1.addendum.lock"
COMPAT = REPO / "docs" / "lineage_gatedrehearsal.compat.lock"
R2_PREREG = REPO / "docs" / "lineage_gatedrehearsal.r2.prereg.lock"
R2_DEV = REPO / "docs" / "lineage_gatedrehearsal.r2.dev.lock"
R2_DEC = REPO / "docs" / "lineage_gatedrehearsal.r2.decision.lock"

HISTORICAL_V1_DEV_SHA = "2d784be0d5b80f416aeb88114b43905fc72d3544ea6255d9f9c4339948ad603a"
HISTORICAL_V1_DEC_SHA = "9f447bc7f1ecdd909daa3b4dab55776498003c35c79ef816f650c9ff7604a8a0"
HISTORICAL_COMPAT_SHA = "93981fe3838258d25061c8963d45fdebc396f9d590b6c848f37603a4a038bd09"
HISTORICAL_R2_DEV_SHA = "de1c3e831ffdaf795bef615d5bbf01d36aa50fde0c43882e834dadaddb55e081"
HISTORICAL_R2_DEC_SHA = "98d51873109f98ccaf10eafb6aa88e26ddce3090b5b83096d4f25cfeaa295899"
R2_FROZEN_RUNNER_SHA = "7f487792a38970c6429469ee9781b942d8b606de434f91c915f7ab06441cc8cd"
R2_MANIFEST = "89f144c6ce13faa5093cbaaeb8f9599289725da6e9a0bfdad958b3d17ff9da10"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_v1_historical_dev_immutable():
    assert V1_DEV.exists()
    assert _sha(V1_DEV) == HISTORICAL_V1_DEV_SHA


def test_v1_historical_decision_immutable():
    assert V1_DEC.exists()
    assert _sha(V1_DEC) == HISTORICAL_V1_DEC_SHA


def test_compat_lock_immutable():
    assert _sha(COMPAT) == HISTORICAL_COMPAT_SHA


def test_v1_addendum_exists():
    assert V1_ADD.exists()
    add = json.loads(V1_ADD.read_text())
    assert add["rewrite_historical_dev"] is False
    assert "perturbation_instability" in add["failure_classifier_correction"]["classes"]


def test_compat_informative_preflight():
    assert COMPAT.exists()
    c = json.loads(COMPAT.read_text())
    assert c["compatible"] is True
    assert c["informative_preflight_only"] is True
    assert c["not_confirmatory_r2"] is True


def test_r2_prereg_manifest():
    p = json.loads(R2_PREREG.read_text())
    assert p["manifest_sha"] == R2_MANIFEST
    assert p["preregistered_decision_outcome"] is False
    assert p["frozen_runner_sha"] == R2_FROZEN_RUNNER_SHA


def test_r2_historical_boundary_immutable():
    assert _sha(R2_DEV) == HISTORICAL_R2_DEV_SHA
    assert _sha(R2_DEC) == HISTORICAL_R2_DEC_SHA
    dev = json.loads(R2_DEV.read_text())
    dec = json.loads(R2_DEC.read_text())
    assert dev["frozen_runner_sha"] == R2_FROZEN_RUNNER_SHA
    assert dec["frozen_runner_sha"] == R2_FROZEN_RUNNER_SHA
    assert dec["dev_lock_sha"] == HISTORICAL_R2_DEV_SHA


def test_r2_dev_and_decision_exist():
    assert R2_DEV.exists()
    assert R2_DEC.exists()
    dec = json.loads(R2_DEC.read_text())
    assert dec["confirmatory_r2"] is True
    assert dec["decision"]["code"] == "gated_rehearsal_r2_core_stability_fail"


def test_classify_failure_perturbation_instability():
    from experiments.run_tm027gatedrehearsal import classify_failure

    fc = classify_failure(
        stored_pre_mix={"all_margin_ok": True},
        stored_post_mix={"all_margin_ok": True},
        live_ranking_ok=True,
        live_geometric_ok=True,
        live_perturbation_ok=False,
    )
    assert fc == "perturbation_instability"


def test_classify_failure_reinstatement_wall_ranking():
    from experiments.run_tm027gatedrehearsal import classify_failure

    fc = classify_failure(
        stored_pre_mix={"all_margin_ok": True},
        stored_post_mix={"all_margin_ok": True},
        live_ranking_ok=False,
        live_geometric_ok=True,
        live_perturbation_ok=True,
    )
    assert fc == "reinstatement_wall"


def test_r2_smoke():
    from experiments.run_tm027gatedrehearsal_r2 import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["r2_manifest_sha"] == R2_MANIFEST


def test_compat_replay_refused_after_record():
    from experiments.run_tm027gatedrehearsal import refuse_compat_replay

    with pytest.raises(RuntimeError):
        refuse_compat_replay()
