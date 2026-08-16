"""TM.0.21.PERSPECTIVE regression: baseline, cells, P-life, capacity, wall."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016relate import empty_birth
from experiments.run_tm020reliability import make_reliability
from experiments.run_tm021perspective import (
    AGENT_PY,
    BASELINE_LOCK,
    CANDIDATE_LOCK,
    CANDIDATE_V1_LOCK,
    FIXTURE_JSON,
    MECH_LOCK,
    PERSPECTIVE_LOCK,
    PREREG_BASELINE,
    PREREG_MECH,
    PREREG_WALL,
    WALL_LOCK,
    make_perspective,
    run_baseline,
    run_life,
    run_smoke,
    run_unit_cells,
    run_wall,
    verify_baseline_prereg,
    verify_mech_prereg,
    verify_wall_prereg,
)
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
    assert mlock["flag"] == "use_source_perspective"
    ok_w, why_w, wlock = verify_wall_prereg()
    assert ok_w, why_w
    assert wlock["mechanism_changes_permitted"] is False
    assert "W_misunderstood" in wlock["probe_ids"]


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
    assert cand["flag"] == "use_source_perspective"
    assert "report_alignment" in cand.get("margin_name", "")
    assert "honesty" not in cand.get("margin_name", "").lower()


def test_unit_cells_and_mech() -> None:
    cells = run_unit_cells()
    assert cells["ok"], cells
    assert cells["n_pass"] == cells["n_cells"] == 9
    assert MECH_LOCK.exists()
    mech = json.loads(MECH_LOCK.read_text(encoding="utf-8"))
    assert mech["ok"] is True


def test_life_lock() -> None:
    assert PERSPECTIVE_LOCK.exists()
    lock = json.loads(PERSPECTIVE_LOCK.read_text(encoding="utf-8"))
    assert lock["ok"] is True
    assert lock["life_last_stage_clear"] == "P12"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["twin_ok"] is True
    assert lock["capacity"]["ok"] is True
    refuse = lock.get("refuse") or []
    assert any("honesty" in str(x).lower() for x in refuse)


def test_wall_diagnostic() -> None:
    assert WALL_LOCK.exists()
    lock = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    assert lock["need_not_fully_pass"] is True
    assert lock["scored_probes_ok"] is True
    assert lock["first_fail_wall"]["id"] == "W_misunderstood"
    assert lock["first_fail_wall"]["actual"] == "diagnostic_fail"
    assert lock["next_primitive_hint"] == "comprehension"
    summary = run_wall(write_lock=False)
    assert summary["scored_probes_ok"] is True
    assert summary["first_fail_wall"]["id"] == "W_misunderstood"
    assert summary["first_fail_wall"]["actual"] == "diagnostic_fail"
    not_run = [p for p in summary["probes"] if p["actual"] == "not_run"]
    assert not_run
    assert all(p["id"] != summary["first_fail_wall"]["id"] for p in not_run)


def test_reliability_unchanged_default() -> None:
    """make_reliability must leave perspective off (RELIABILITY freeze)."""
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm021_rel_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_reliability(s, policy)
        assert getattr(ag, "use_source_perspective", False) is False
        out = ag.observe_exposure(
            {
                "speaker_token": "spk_a",
                "context_atoms": ["scene", "fac_lab"],
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "e",
            }
        )
        assert out.get("why") == "perspective_off"


def test_make_perspective_opt_in() -> None:
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm021_persp_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_perspective(s, policy)
        assert ag.use_source_reliability is True
        assert ag.use_source_perspective is True
        assert ag.perspective_lambda == 4
        assert ag.perspective_n_min == 2


def test_presence_does_not_attach() -> None:
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm021_pres_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_perspective(s, policy)
        ag.reset_rho()
        CTX = ["scene", "fac_lab"]
        ag.observe_exposure(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_present"],
                "event_token": "ep",
            }
        )
        ag.observe_symbol_ground(
            {
                "symbol": "box",
                "paired": "hyp_red",
                "trial_id": "evt_ep__v0",
                "result": "success",
                "provenance": "direct",
            }
        )
        assert ag.evidenced_perspective_hyp("spk_a", "box") is None
        assert (
            ag.report_alignment_status("spk_a", ["box", "hyp_red"], CTX) == "UNKNOWN"
        )


def test_no_honesty_score_fields() -> None:
    text = AGENT_PY.read_text(encoding="utf-8")
    assert "honesty_score" not in text
    assert "def believes" not in text


def test_world_unique_beats_testimony() -> None:
    """Frozen influence #1: unique world grounding answers before predictive margin."""
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm021_world_") as tmp:
        from experiments.run_tm019inquire import ensure_context_grounded

        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_perspective(s, policy)
        ag.reset_rho()
        ensure_context_grounded(ag, ["scene", "fac_lab"], tag="w")
        CTX = ["scene", "fac_lab"]
        for i in range(2):
            ag.observe_testimony(
                {
                    "speaker_token": "spk_a",
                    "context_atoms": CTX,
                    "claim_atoms": [f"past_w{i}", "hyp_ok"],
                    "event_token": f"epw{i}",
                }
            )
            ag.observe_symbol_ground(
                {
                    "symbol": f"past_w{i}",
                    "paired": "hyp_ok",
                    "trial_id": f"evt_epw{i}__v0",
                    "result": "success",
                    "provenance": "direct",
                }
            )
        for i in range(2):
            ag.observe_symbol_ground(
                {
                    "symbol": "box",
                    "paired": "hyp_red",
                    "trial_id": f"wuniq_{i}",
                    "result": "success",
                    "provenance": "direct",
                }
            )
        ag.observe_exposure(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "eblue",
            }
        )
        ag.observe_symbol_ground(
            {
                "symbol": "box",
                "paired": "hyp_blue",
                "trial_id": "evt_eblue__v0",
                "result": "success",
                "provenance": "direct",
            }
        )
        ag.observe_testimony(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "claim_atoms": ["box", "hyp_blue"],
                "event_token": "etb",
            }
        )
        plan = ag.plan_inquiry(
            {"context_atoms": CTX, "input_symbols": ["what", "box"]}
        )
        assert plan.get("status") == "ANSWER", plan
        assert plan.get("answer_symbols") == ["hyp_red"], plan
        assert plan.get("why") == "unique_hypothesis", plan


def test_donor_exposure_changes_alignment() -> None:
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm021_donor_") as tmp:
        from experiments.run_tm019inquire import ensure_context_grounded
        from experiments.run_tm021perspective import run_fork

        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_perspective(s, policy)
        ag.reset_rho()
        ensure_context_grounded(ag, ["scene", "fac_lab"], tag="d")
        CTX = ["scene", "fac_lab"]
        ag.observe_exposure(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "ed0",
            }
        )
        ag.observe_symbol_ground(
            {
                "symbol": "box",
                "paired": "hyp_red",
                "trial_id": "evt_ed0__v0",
                "result": "success",
                "provenance": "direct",
            }
        )
        assert ag.report_alignment_status("spk_a", ["box", "hyp_red"], CTX) == "ALIGNED"
        fail = run_fork(
            Path(tmp),
            s,
            policy,
            {
                "kind": "donor_exposure",
                "speaker": "spk_a",
                "claim_atoms": ["box", "hyp_red"],
                "context_atoms": CTX,
                "expect_alignment": "MISALIGNED",
                "donor_ops": [
                    {
                        "op": "exposure",
                        "speaker_token": "spk_a",
                        "context_atoms": CTX,
                        "exposure_atoms": ["exp_ack_read"],
                        "event_token": "edonor",
                    },
                    {
                        "op": "ground",
                        "symbol": "box",
                        "paired": "hyp_blue",
                        "trial_id": "evt_edonor__v0",
                        "result": "success",
                        "provenance": "direct",
                    },
                ],
            },
            perspective=True,
        )
        assert fail is None, fail


def test_repetition_no_amplify() -> None:
    """Same claim without new exposure must not amplify report_alignment_margin."""
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm021_rep_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_perspective(s, policy)
        ag.reset_rho()
        CTX = ["scene", "fac_lab"]
        ag.observe_exposure(
            {
                "speaker_token": "spk_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "er",
            }
        )
        ag.observe_symbol_ground(
            {
                "symbol": "box",
                "paired": "hyp_red",
                "trial_id": "evt_er__v0",
                "result": "success",
                "provenance": "direct",
            }
        )
        for i in range(20):
            ag.observe_testimony(
                {
                    "speaker_token": "spk_a",
                    "context_atoms": CTX,
                    "claim_atoms": ["box", "hyp_red"],
                    "event_token": f"ert_{i}",
                }
            )
        m = ag.report_alignment_margin("spk_a", CTX)
        # n_unique=1 < n_min=2 → margin 0
        assert m == 0.0, m


if __name__ == "__main__":
    test_prereg_and_fixture()
    test_baseline_lock()
    test_smoke_and_candidate()
    test_unit_cells_and_mech()
    test_life_lock()
    test_wall_diagnostic()
    test_reliability_unchanged_default()
    test_make_perspective_opt_in()
    test_presence_does_not_attach()
    test_no_honesty_score_fields()
    test_world_unique_beats_testimony()
    test_donor_exposure_changes_alignment()
    test_repetition_no_amplify()
    print("test_tm021perspective: ok")
