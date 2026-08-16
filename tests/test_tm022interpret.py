"""TM.0.22.INTERPRET regression: baseline, cells, J-life, capacity, wall."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016relate import empty_birth
from experiments.run_tm019inquire import ensure_context_grounded
from experiments.run_tm021perspective import make_perspective
from experiments.run_tm022interpret import (
    AGENT_PY,
    BASELINE_LOCK,
    CANDIDATE_LOCK,
    CANDIDATE_V1_LOCK,
    FIXTURE_JSON,
    INTERPRET_LOCK,
    MECH_LOCK,
    PREREG_BASELINE,
    PREREG_MECH,
    PREREG_WALL,
    WALL_LOCK,
    make_interpret,
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
    ok_m, why_m, mlock = verify_mech_prereg()
    assert ok_m, why_m
    assert mlock["flag"] == "use_source_interpretation"
    assert mlock["no_jaccard"] is True
    assert mlock["independent_anchor_required"] is True
    ok_w, why_w, wlock = verify_wall_prereg()
    assert ok_w, why_w
    assert wlock["mechanism_changes_permitted"] is False


def test_baseline_lock() -> None:
    assert BASELINE_LOCK.exists()
    lock = json.loads(BASELINE_LOCK.read_text(encoding="utf-8"))
    assert lock["ok"] is True
    assert lock["fixture_sha"] == sha(FIXTURE_JSON)
    summary = run_baseline(write_lock=False)
    assert summary["ok"] is True


def test_smoke_and_candidate() -> None:
    smoke = run_smoke()
    assert smoke["ok"], smoke
    assert CANDIDATE_LOCK.exists()
    assert CANDIDATE_V1_LOCK.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand["flag"] == "use_source_interpretation"
    assert cand["no_jaccard"] is True
    assert "goal_cue" in cand.get("repair_goal", "")


def test_unit_cells_and_mech() -> None:
    cells = run_unit_cells()
    assert cells["ok"], cells
    assert cells["n_pass"] == cells["n_cells"] == 16
    assert MECH_LOCK.exists()


def test_life_lock() -> None:
    assert INTERPRET_LOCK.exists()
    lock = json.loads(INTERPRET_LOCK.read_text(encoding="utf-8"))
    assert lock["ok"] is True
    assert lock["life_last_stage_clear"] == "J15"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["twin_ok"] is True
    assert lock["capacity"]["ok"] is True


def test_wall_diagnostic() -> None:
    assert WALL_LOCK.exists()
    lock = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    assert lock["scored_probes_ok"] is True
    assert lock["first_fail_wall"]["id"] == "W_claim_understand"
    assert lock["first_fail_wall"]["actual"] == "diagnostic_fail"
    assert lock["next_primitive_hint"] == "honesty"
    summary = run_wall(write_lock=False)
    assert summary["scored_probes_ok"] is True
    not_run = [p for p in summary["probes"] if p["actual"] == "not_run"]
    assert not_run
    assert all(p["id"] != summary["first_fail_wall"]["id"] for p in not_run)


def test_perspective_unchanged_default() -> None:
    policy = UsePolicy(seed=12345)
    with tempfile.TemporaryDirectory(prefix="tm022_persp_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_perspective(s, policy)
        assert getattr(ag, "use_source_interpretation", False) is False
        out = ag.observe_source_consequence(
            {
                "source_token": "a",
                "interaction_token": "i",
                "exposure_event_token": "e",
                "consequence_event_token": "c",
                "context_symbols": ["scene"],
                "message_symbols": ["m"],
                "action_symbols": ["a"],
                "state_before": ["b"],
                "state_after": ["d"],
            }
        )
        assert out.get("why") == "interpretation_off"


def test_independent_anchor_required() -> None:
    policy = UsePolicy(seed=1)
    with tempfile.TemporaryDirectory(prefix="tm022_anchor_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_interpret(s, policy)
        ag.reset_rho()
        ensure_context_grounded(ag, ["scene", "fac_lab"], tag="a")
        CTX = ["scene", "fac_lab"]
        ag.observe_exposure(
            {
                "speaker_token": "src_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "e1",
            }
        )
        # consequence without independent grounding of action
        ag.observe_source_consequence(
            {
                "source_token": "src_a",
                "interaction_token": "ix1",
                "exposure_event_token": "e1",
                "consequence_event_token": "c1",
                "context_symbols": CTX,
                "message_symbols": ["tok"],
                "action_symbols": ["act_ungrounded"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        recon = ag.interpret_message(
            {
                "source_token": "src_a",
                "context_symbols": CTX,
                "ordered_symbols": ["tok"],
            }
        )
        assert recon.get("status") == "INSUFFICIENT", recon


def test_fit_never_takes_candidate() -> None:
    """interpretation_fit exact keys must not include candidate."""
    import inspect

    src = inspect.getsource(
        __import__("three_memory.agent", fromlist=["ThreeMemoryAgent"]).ThreeMemoryAgent.interpretation_fit
    )
    assert "candidate" not in src.split("required")[1].split(")")[0]


def test_no_honesty_or_cause_abi() -> None:
    text = AGENT_PY.read_text(encoding="utf-8")
    assert "honesty_score" not in text
    assert "def cause_" not in text


def test_banned_exact_token_not_substring() -> None:
    """Opaque vocab containing banned substrings must not be rejected."""
    policy = UsePolicy(seed=3)
    with tempfile.TemporaryDirectory(prefix="tm022_banned_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_interpret(s, policy)
        out = ag.observe_source_consequence(
            {
                "source_token": "src",
                "interaction_token": "ix",
                "exposure_event_token": "ev",
                "consequence_event_token": "cv",
                "context_symbols": ["scene", "fac_lab"],
                "message_symbols": ["because", "unsuccessful"],
                "action_symbols": ["act_a", "act_b"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        assert out.get("why") != "banned_token", out
        # Exact banned atom still rejects
        bad = ag.observe_source_consequence(
            {
                "source_token": "src",
                "interaction_token": "ix2",
                "exposure_event_token": "ev2",
                "consequence_event_token": "cv2",
                "context_symbols": ["scene", "fac_lab"],
                "message_symbols": ["cause"],
                "action_symbols": ["act_a"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        assert bad.get("why") == "banned_token", bad


def test_unequal_message_action_length_rejected() -> None:
    policy = UsePolicy(seed=4)
    with tempfile.TemporaryDirectory(prefix="tm022_len_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_interpret(s, policy)
        out = ag.observe_source_consequence(
            {
                "source_token": "src",
                "interaction_token": "ix",
                "exposure_event_token": "ev",
                "consequence_event_token": "cv",
                "context_symbols": ["scene", "fac_lab"],
                "message_symbols": ["m1", "m2"],
                "action_symbols": ["act_a"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        assert out.get("why") == "length_mismatch", out
        assert out.get("wrote") is False


def test_fit_requires_equal_lengths() -> None:
    policy = UsePolicy(seed=5)
    with tempfile.TemporaryDirectory(prefix="tm022_fitlen_") as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_interpret(s, policy)
        ag.reset_rho()
        ensure_context_grounded(ag, ["scene", "fac_lab", "world"], tag="fl")
        CTX = ["scene", "fac_lab"]
        for i in range(2):
            ag.observe_symbol_ground(
                {
                    "symbol": "act_a",
                    "paired": "mean_a",
                    "trial_id": f"fl_a{i}",
                    "result": "success",
                    "provenance": "direct",
                }
            )
            ag.observe_symbol_ground(
                {
                    "symbol": "act_b",
                    "paired": "mean_b",
                    "trial_id": f"fl_b{i}",
                    "result": "success",
                    "provenance": "direct",
                }
            )
        for i in range(2):
            ag.observe_exposure(
                {
                    "speaker_token": "src",
                    "context_atoms": CTX,
                    "exposure_atoms": ["exp_ack_read"],
                    "event_token": f"fl_e{i}",
                }
            )
            ag.observe_source_consequence(
                {
                    "source_token": "src",
                    "interaction_token": f"fl_i{i}",
                    "exposure_event_token": f"fl_e{i}",
                    "consequence_event_token": f"fl_c{i}",
                    "context_symbols": CTX,
                    "message_symbols": ["m1", "m2"],
                    "action_symbols": ["act_a", "act_b"],
                    "state_before": ["st_idle"],
                    "state_after": ["st_done"],
                }
            )
        fit = ag.interpretation_fit(
            {
                "source_token": "src",
                "context_symbols": CTX,
                "message_symbols": ["m1", "m2"],
                "action_symbols": ["act_a"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        assert fit.get("fit") == "UNKNOWN", fit
        assert fit.get("why") == "length_mismatch", fit


def test_capacity_never_forces_ok() -> None:
    text = (REPO_ROOT / "experiments" / "run_tm022interpret.py").read_text(
        encoding="utf-8"
    )
    assert "ok if not is_metric_only else True" not in text


if __name__ == "__main__":
    test_prereg_and_fixture()
    test_baseline_lock()
    test_smoke_and_candidate()
    test_unit_cells_and_mech()
    test_life_lock()
    test_wall_diagnostic()
    test_perspective_unchanged_default()
    test_independent_anchor_required()
    test_fit_never_takes_candidate()
    test_no_honesty_or_cause_abi()
    test_banned_exact_token_not_substring()
    test_unequal_message_action_length_rejected()
    test_fit_requires_equal_lengths()
    test_capacity_never_forces_ok()
    print("test_tm022interpret: ok")
