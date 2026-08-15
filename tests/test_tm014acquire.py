"""TM.0.14.ACQUIRE: developmental delta battery + freeze locks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm014acquire import (
    ACQUIRE_LOCK,
    GENOME_014_LOCK,
    diamond_skeleton,
    run_acquire,
    verify_acquire_lock,
    verify_genome_014,
    write_skeleton,
    write_skeleton_tags,
)
from three_memory.policy import UsePolicy


def test_acquire_battery_and_locks():
    summary = run_acquire(seed=12345, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 16
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["D0_birth_no_ctx"]["ok"]
    assert by["D1_life_a_only"]["a_motor"] == "press"
    assert by["D1_life_a_only"]["b_motor"] == "hold"
    assert by["D2_both_coexist"]["a_motor"] == "press"
    assert by["D2_both_coexist"]["b_motor"] == "tune"
    assert by["D15_nasty_five"]["ok"]

    ok, why, _ = verify_genome_014()
    assert ok, why
    ok2, why2, _ = verify_acquire_lock(summary["rows"])
    assert ok2, why2
    lock = json.loads(ACQUIRE_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert "stamp or pre-name Ex0S 0.0.005" in lock["refuse"]
    g = json.loads(GENOME_014_LOCK.read_text(encoding="utf-8"))
    assert g["use_acquire_ctx"] is True
    assert g["earned_next"] is False


def test_acquire_off_default_and_skeleton_refuses_ctx():
    with TemporaryDirectory(prefix="tm014_off_") as tmp:
        s = Path(tmp) / "s"
        write_skeleton(s, diamond_skeleton())
        ag = make(s, None, UsePolicy(seed=1), explore_epsilon=0.0, use_context_kappa=True)
        assert getattr(ag, "use_acquire_ctx", False) is False
        raised = False
        try:
            write_skeleton_tags(
                Path(tmp) / "bad",
                [{"fid": "x", "bind": "y", "did": "press", "ctx": "nope", "here": "chb"}],
            )
        except ValueError as e:
            raised = "ctx" in str(e).lower()
        assert raised


def test_make011compose_untouched():
    """Lineage: run_tm011compose.make source must stay the CONTEXT forwarder."""
    import inspect
    from experiments import run_tm011compose

    src = inspect.getsource(run_tm011compose.make)
    assert "use_acquire_ctx" not in src
    assert "use_context_kappa" in src


if __name__ == "__main__":
    test_make011compose_untouched()
    test_acquire_off_default_and_skeleton_refuses_ctx()
    test_acquire_battery_and_locks()
    print("ok")
