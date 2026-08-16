"""TM.0.20.RELIABILITY regression: baseline, cells, R-life, capacity, wall."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016relate import empty_birth
from experiments.run_tm020reliability import (
    AGENT_PY,
    BASELINE_LOCK,
    CANDIDATE_LOCK,
    CANDIDATE_V1_LOCK,
    FIXTURE_JSON,
    MECH_LOCK,
    PREREG_BASELINE,
    PREREG_MECH,
    PREREG_WALL,
    RELIABILITY_LOCK,
    WALL_LOCK,
    make_reliability,
    run_baseline,
    run_life,
    run_smoke,
    run_unit_cells,
    run_wall,
    verify_baseline_prereg,
    verify_mech_prereg,
    verify_wall_prereg,
)
from experiments.run_tm019inquire import make_inquire
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
    assert mlock["lambda"] == 4 and mlock["n_min"] == 2
    assert mlock["flag"] == "use_source_reliability"
    ok_w, why_w, wlock = verify_wall_prereg()
    assert ok_w, why_w
    assert "honesty" in wlock["social_dimensions"]
    assert wlock["mechanism_changes_permitted"] is False


def test_baseline_lock() -> None:
    assert BASELINE_LOCK.exists()
    lock = json.loads(BASELINE_LOCK.read_text(encoding="utf-8"))
    assert lock["ok"] is True
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["fixture_sha"] == sha(FIXTURE_JSON)
    summary = run_baseline(write_lock=False)
    assert summary["ok"] is True


def test_smoke_and_candidate() -> None:
    smoke = run_smoke()
    assert smoke["ok"], smoke
    assert CANDIDATE_LOCK.exists()
    assert CANDIDATE_V1_LOCK.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand["flag"] == "use_source_reliability"
    assert cand["margin_name"] == "source_evidence_margin"
    assert "trust" not in cand.get("margin_name", "").lower()


def test_unit_cells_and_mech() -> None:
    cells = run_unit_cells()
    assert cells["ok"], cells
    assert cells["n_pass"] == cells["n_cells"] == 6
    assert MECH_LOCK.exists()
    mech = json.loads(MECH_LOCK.read_text(encoding="utf-8"))
    assert mech["ok"] is True


def test_life_lock() -> None:
    assert RELIABILITY_LOCK.exists()
    lock = json.loads(RELIABILITY_LOCK.read_text(encoding="utf-8"))
    assert lock["ok"] is True
    assert lock["life_last_stage_clear"] == "R12"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["twin_ok"] is True
    assert lock["capacity"]["ok"] is True
    assert "trust_score" in (lock.get("refuse") or [])


def test_wall_diagnostic() -> None:
    assert WALL_LOCK.exists()
    lock = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    assert lock["need_not_fully_pass"] is True
    assert lock["scored_probes_ok"] is True
    assert lock["first_fail_wall"]["id"] == "W_competent_liar"
    assert lock["first_fail_wall"]["actual"] == "diagnostic_fail"
    assert lock["next_primitive_hint"] == "honesty"
    # Re-run wall without rewrite
    summary = run_wall(write_lock=False)
    assert summary["scored_probes_ok"] is True
    assert summary["first_fail_wall"]["id"] == "W_competent_liar"
    assert summary["first_fail_wall"]["actual"] == "diagnostic_fail"
    # not_run probes must not be first_fail
    not_run = [p for p in summary["probes"] if p["actual"] == "not_run"]
    assert not_run
    assert all(p["id"] != summary["first_fail_wall"]["id"] for p in not_run)


def test_inquire_unchanged_default() -> None:
    """make_inquire must leave reliability off (INQUIRE freeze)."""
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm020_inq_") as tmp:
        from experiments.run_tm016relate import empty_birth

        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_inquire(s, policy)
        assert getattr(ag, "use_source_reliability", False) is False
        assert getattr(ag, "use_inquire_liveness", False) is False
        out = ag.observe_testimony(
            {
                "speaker_token": "spk_a",
                "context_atoms": ["scene", "fac_lab"],
                "claim_atoms": ["c", "h"],
                "event_token": "e",
            }
        )
        assert out.get("why") == "reliability_off"


def test_make_reliability_opt_in() -> None:
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm020_rel_") as tmp:
        from experiments.run_tm016relate import empty_birth

        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_reliability(s, policy)
        assert ag.use_source_reliability is True
        assert ag.use_inquire_liveness is True
        assert ag.reliability_lambda == 4
        assert ag.reliability_n_min == 2


def test_no_host_confirm_fields() -> None:
    text = AGENT_PY.read_text(encoding="utf-8")
    assert "observe_reliability_verify" not in text


def test_live_supersede_persists() -> None:
    """Replacement claim must clear prior live bit on disk (TagStore reload)."""
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm020_super_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_reliability(s, policy)
        ag.reset_rho()
        CTX = ["scene", "fac_lab"]
        ag.observe_testimony(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "claim_atoms": ["cue", "hyp_old"],
                "event_token": "e1",
            }
        )
        ag.observe_testimony(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "claim_atoms": ["cue", "hyp_new"],
                "event_token": "e2",
            }
        )
        live = [
            r
            for r in ag._testimony_rows()
            if int(r.tags.get("live") or 0) == 1
        ]
        hyps = sorted(str(r.tags.get("hypothesis")) for r in live)
        assert hyps == ["hyp_new"], hyps


def test_testimony_derived_not_inquire_support() -> None:
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm020_td_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_reliability(s, policy)
        ag.reset_rho()
        for i in range(2):
            ag.observe_symbol_ground(
                {
                    "symbol": "curr_td",
                    "paired": "hyp_red",
                    "trial_id": f"td_{i}",
                    "result": "success",
                    "provenance": "testimony_derived",
                }
            )
        hyps = ag._inquire_hypotheses("curr_td", min_support=2)
        assert hyps == []


if __name__ == "__main__":
    test_prereg_and_fixture()
    test_baseline_lock()
    test_smoke_and_candidate()
    test_unit_cells_and_mech()
    test_life_lock()
    test_wall_diagnostic()
    test_inquire_unchanged_default()
    test_make_reliability_opt_in()
    test_no_host_confirm_fields()
    test_live_supersede_persists()
    test_testimony_derived_not_inquire_support()
    print("test_tm020reliability: ok")
