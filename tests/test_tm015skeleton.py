"""TM.0.15.SKELETON: observed-transition acquisition battery + freeze locks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm015skeleton import (
    CELL_IDS,
    GENOME_015_LOCK,
    PREREG_LOCK,
    PREREGISTERED_CLAIM,
    SKELETON_LOCK,
    make_skeleton,
    run_skeleton,
    verify_genome_015,
    verify_prereg_lock,
    verify_skeleton_lock,
)
from three_memory.policy import UsePolicy


def test_prereg_intact():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["observation_abi"] == "observe_symbol"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["preregistered_claim"] == PREREGISTERED_CLAIM
    assert "put final mechanism/cell SHAs into this prereg lock" in lock["refuse"]
    assert "agent_sha" not in lock


def test_skeleton_battery_and_locks():
    summary = run_skeleton(seed=12345, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 16
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["unexpected_writes"] == 0
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["D0_birth_unreachable"]["ok"]
    assert by["D1_symbol_life"]["can_reach"] is True
    assert by["D5_dual_strip"]["test_a_motor"] == "hold"
    assert by["D5_dual_strip"]["test_b_motor"] == "hold"
    assert by["D5_dual_strip"]["restore_motor"] == "press"
    assert by["D9_rho_ctx_probe"]["after_rho"] == "press"
    assert by["D15_stale_prev"]["stale_prev"] is True

    ok, why, _ = verify_genome_015()
    assert ok, why
    ok2, why2, _ = verify_skeleton_lock(summary["rows"])
    assert ok2, why2
    lock = json.loads(SKELETON_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["observation_abi"] == "observe_symbol"
    assert lock["cell_ids"] == list(CELL_IDS)
    g = json.loads(GENOME_015_LOCK.read_text(encoding="utf-8"))
    assert g["use_acquire_skel"] is True
    assert g["use_acquire_ctx"] is True
    assert g["earned_next"] is False


def test_skel_off_by_default():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="tm015_off_") as tmp:
        s = Path(tmp) / "s"
        s.mkdir()
        ag = make(s, None, UsePolicy(seed=1), explore_epsilon=0.0, use_context_kappa=True, use_acquire_ctx=True)
        assert getattr(ag, "use_acquire_skel", False) is False
        out = ag.observe_symbol("x")
        assert out.get("why") == "skel_off"
        ag2 = make_skeleton(s, UsePolicy(seed=1))
        assert ag2.use_acquire_skel is True


def test_make011compose_untouched():
    import inspect
    from experiments import run_tm011compose

    src = inspect.getsource(run_tm011compose.make)
    assert "use_acquire_skel" not in src
    assert "use_context_kappa" in src


def test_prereg_not_rewritten_with_freeze_shas():
    lock = json.loads(PREREG_LOCK.read_text(encoding="utf-8"))
    for banned in ("agent_sha", "make_skeleton_sha", "run_tm015skeleton_sha", "cell_shas"):
        assert banned not in lock


if __name__ == "__main__":
    test_prereg_intact()
    test_skeleton_battery_and_locks()
    test_skel_off_by_default()
    test_make011compose_untouched()
    test_prereg_not_rewritten_with_freeze_shas()
    print("ok")
