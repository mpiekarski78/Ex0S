"""TM.0.13.CONTEXT: κ vectors, historical freeze, causal family cells."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011family import (
    ORGANISM_BASELINE_COMMIT,
    verify_011_compatibility,
    verify_historical_freeze,
)
from experiments.run_tm013context import (
    CONTEXT_LOCK,
    GENOME_011_LOCK,
    GENOME_013_LOCK,
    KAPPA_LOCK,
    reference_route_kappa,
    run_context,
    verify_context_lock,
    verify_genome_013,
    verify_kappa_vectors,
)
from three_memory.kappa import CTX_ENCODING


def test_kappa_vectors_and_reference():
    ok, why, snap = verify_kappa_vectors()
    assert ok, why
    assert snap["n_vectors"] == 3
    lock = json.loads(KAPPA_LOCK.read_text(encoding="utf-8"))
    assert lock["ctx_encoding"] == CTX_ENCODING
    assert lock["earned_next"] is False
    vec = lock["known_answer_vectors"]["A_order_a_then_b"]
    hops = [tuple(h) for h in vec["hops"]]
    assert reference_route_kappa(vec["origin"], hops) == vec["steps"][-1]["kappa"]


def test_genome_011_immutable_historical():
    ok, why, snap = verify_historical_freeze()
    assert ok, why
    assert snap["baseline_commit"] == ORGANISM_BASELINE_COMMIT
    assert snap["head_matches_lock_agent"] is False  # CONTEXT candidate changed agent.py
    lock = json.loads(GENOME_011_LOCK.read_text(encoding="utf-8"))
    assert lock["agent_sha"] == snap["agent_sha"]
    kappa = json.loads(KAPPA_LOCK.read_text(encoding="utf-8"))
    assert kappa["genome_011_lock_sha"] == hashlib.sha256(
        GENOME_011_LOCK.read_bytes()
    ).hexdigest()


def test_011_compatibility_feature_off():
    ok, why, snap = verify_011_compatibility()
    assert ok, why
    assert snap["use_context_kappa"] is False
    assert snap["use_compose"] is True


def test_genome_013_and_context_locks():
    ok, why, _ = verify_genome_013()
    assert ok, why
    ok2, why2, _snap = verify_context_lock()
    assert ok2, why2
    lock = json.loads(CONTEXT_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert lock["ex0s_under_test"] == "0.0.003"
    assert "stamp Ex0S 0.0.004" in lock["refuse"]
    g013 = json.loads(GENOME_013_LOCK.read_text(encoding="utf-8"))
    assert g013["use_context_kappa"] is True
    assert g013["ctx_encoding"] == CTX_ENCODING


def test_full_context_battery():
    summary = run_context(seed=12345, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 14
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["C13_route_order"]["ab_motor"] == "press"
    assert by["C13_route_order"]["ba_motor"] == "tune"
    assert by["C13_c7_tie_hold"]["motor"] == "hold"
    assert by["C13_donor_revise"]["before_motor"] == "press"
    assert by["C13_donor_revise"]["after_motor"] == "tune"
    assert by["C13_ctx_beats_untagged"]["motor"] == "press"
    assert by["C13_no_fallback_untagged"]["motor"] == "hold"
    assert by["C13_no_fallback_untagged"]["kappa"] == by["C13_no_fallback_untagged"]["expect_kappa"]
    assert by["C13_hop1_motor_no_kappa"]["kappa"] is None
    assert by["C13_visited_ctx_no_poison"]["motor"] == "press"
    assert by["C13_depth_holdout"]["ok"]
    assert by["C13_new_nonce_order"]["ok"]
    assert by["C13_feature_off_untagged_wins"]["on_motor"] == "press"
    assert by["C13_feature_off_untagged_wins"]["off_motor"] == "tune"


def test_verify_context_lock_fail_closed_on_cells():
    ok, why, _ = verify_context_lock(rows=[{"cell": "bogus", "ok": False}])
    assert not ok
    assert "cell_ids" in why or "n_ok" in why or "n_cells" in why


if __name__ == "__main__":
    test_kappa_vectors_and_reference()
    test_genome_011_immutable_historical()
    test_011_compatibility_feature_off()
    test_genome_013_and_context_locks()
    test_full_context_battery()
    print("ok")
