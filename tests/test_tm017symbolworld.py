"""TM.0.17.SYMBOLWORLD regression: baseline, grounding cells, developmental life."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm017symbolworld import (
    AGENT_PY,
    BASELINE_LOCK,
    CANDIDATE_LOCK,
    FIXTURE_JSON,
    GROUND_LOCK,
    PERSIST_LOCK,
    PREREG_GROUND,
    PREREG_WORLD,
    WORLD_LOCK,
    run_baseline,
    run_life,
    run_unit_cells,
    verify_ground_prereg,
    verify_world_prereg,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg_and_fixture() -> None:
    ok, why, lock = verify_world_prereg()
    assert ok, why
    assert lock["fixture_sha"] == sha(FIXTURE_JSON)
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["phase_a"]["agent_edits_permitted"] is False
    ok_g, why_g, glock = verify_ground_prereg()
    assert ok_g, why_g
    assert glock["flag"] == "use_symbol_ground"
    assert glock["flag_default"] is False
    assert glock["source"] == "experience_grounding"
    assert "agent_sha" not in glock


def test_baseline_hold() -> None:
    assert BASELINE_LOCK.exists()
    summary = run_baseline(seed=12345, write_lock=False)
    assert summary["ok"] is True
    assert summary["first_fail"] is None
    freeze = json.loads(BASELINE_LOCK.read_text(encoding="utf-8"))
    assert freeze["ok"] is True
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)


def test_unit_cells() -> None:
    assert CANDIDATE_LOCK.exists()
    cells = run_unit_cells(seed=12345)
    assert cells["ok"] is True
    assert cells["n_pass"] == cells["n_cells"] == 6
    assert GROUND_LOCK.exists()
    freeze = json.loads(GROUND_LOCK.read_text(encoding="utf-8"))
    assert freeze["ok"] is True


def test_developmental_life() -> None:
    assert WORLD_LOCK.exists()
    summary = run_life(seed=12345, write_lock=False)
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["agent_sha"] == sha(AGENT_PY)
    freeze = json.loads(WORLD_LOCK.read_text(encoding="utf-8"))
    assert freeze["last_stage_clear"] == summary["last_stage_clear"]
    assert freeze["first_fail_stage"] == summary["first_fail_stage"]
    assert freeze["fixture_sha"] == sha(FIXTURE_JSON)
    # Recorded run cleared through S10
    assert summary["ok"] is True
    assert summary["last_stage_clear"] == "S10"
    assert summary["first_fail"] is None
    assert summary["main"]["ok"] and summary["twin"]["ok"]


def test_no_pos_learners_in_agent() -> None:
    src = AGENT_PY.read_text(encoding="utf-8")
    for banned in ("def learn_noun", "def learn_verb", "def learn_color"):
        assert banned not in src


def test_counterfactual_coverage() -> None:
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    ids = {o.get("id") for o in fix["script_life"] if o.get("id")}
    assert "CF_swap_noun_glim" in ids
    assert "CF_swap_verb_zake" in ids
    assert "CF_rho_mid_vex" in ids
    assert "CF_iso_ball_still" in ids
    forks = [o for o in fix["script_life"] if o["op"] == "fork"]
    kinds = {f["kind"] for f in forks}
    assert "donor_swap" in kinds
    assert "strip_grounding" in kinds
    assert any(o.get("op") == "isolation_seed" for o in fix["script_life"])
    assert any(o.get("id") == "S8_twin_ball" for o in fix["script_twin"])


if __name__ == "__main__":
    test_prereg_and_fixture()
    test_baseline_hold()
    test_unit_cells()
    test_counterfactual_coverage()
    test_developmental_life()
    test_no_pos_learners_in_agent()
    print("ok")
