"""TM.0.11.BOUND: freeze, four-class scoring, expected boundaries, no 0.0.004 stamp."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011bound import (
    BOUND_LOCK,
    CLASS_EXPECTED,
    CLASS_WITHIN,
    acquired_edge_invariant,
    all_cells,
    edges_in_s,
    gen_local_opt,
    gen_reuse_a,
    gen_reuse_b,
    run_bound,
    run_cell,
    verify_bound_lock,
    write_world_s,
)
from experiments.run_tm011family import verify_freeze as verify_organism


def test_locks():
    ok, why, snap = verify_bound_lock()
    assert ok, why
    assert "local_optimum_dead_end" in snap["expected_boundary_cells"]
    assert snap["timeout_ms"] > 0
    org_ok, org_why, _ = verify_organism()
    assert org_ok, org_why
    assert BOUND_LOCK.exists()


def test_holdouts_and_boundary_preregistered():
    cells = all_cells()
    ids = {c.cell_id for c in cells}
    assert "nasty" in ids
    assert "local_optimum_dead_end" in ids
    assert "depth_16" in ids and "depth_20" in ids
    assert "ssize_d5_n1000" in ids
    boundaries = [c for c in cells if c.intended_class == "expected_boundary"]
    assert {c.cell_id for c in boundaries} >= {"local_optimum_dead_end", "reuse_a"}


def test_acquired_edge_invariant(tmp_path: Path):
    cell = gen_reuse_b(1)
    write_world_s(tmp_path, cell.relations)
    assert acquired_edge_invariant(tmp_path, cell.relations)
    # Plant unearned edge
    from three_memory.symbols import record_to_tagfile

    (tmp_path / "cheat.tag").write_text(
        record_to_tagfile(
            "cheat",
            {
                "bind": cell.cue,
                "did": "press",
                "here": "chb",
                "support": 1,
                "contradiction": 0,
                "wins": 1,
                "losses": 0,
                "trials": 1,
                "hyp": "supported",
            },
        ),
        encoding="utf-8",
    )
    assert not acquired_edge_invariant(tmp_path, cell.relations)
    assert (cell.cue.lower(), "press") in edges_in_s(tmp_path)


def test_local_opt_expected_hold():
    row = run_cell(
        {"cell_id": "local_optimum_dead_end", "seed": 12345, "dest": tempfile.mkdtemp()}
    )
    assert row["class"] == CLASS_EXPECTED
    assert row["expected_boundary_confirmed"]
    assert row["motor"] == "hold"
    assert not row["ok_for_solved"]


def test_reuse_a_discriminator():
    """here-filter and bind-only must disagree on expect_motor."""
    cell = gen_reuse_a(12345)
    by = {r.role: r for r in cell.relations}
    assert by["ya"].init[0] > by["yb"].init[0]
    assert by["ya"].here != by["yb"].here
    assert by["yb"].here == "chb"  # probe station
    assert by["ya"].here == "cha"
    # bind-only → strong wrong-here; here-filter → weak same-here
    assert cell.expect_motor == by["am"].did
    assert cell.expect_motor != by["bm"].did


def test_reuse_a_expected_boundary():
    row = run_cell({"cell_id": "reuse_a", "seed": 12345, "dest": tempfile.mkdtemp()})
    assert row["class"] == CLASS_EXPECTED
    assert row["expected_boundary_confirmed"]
    assert row["motor"] == gen_reuse_a(12345).expect_motor


def test_reuse_b_within_model():
    row = run_cell({"cell_id": "reuse_b", "seed": 12345, "dest": tempfile.mkdtemp()})
    assert row["class"] == CLASS_WITHIN
    assert row["ok_for_solved"]


def test_never_stamps_004():
    # Smoke subset via full runner would be heavy; assert aggregate contract on one cell path.
    from experiments.run_tm011bound import aggregate

    summary = aggregate(
        [
            {
                "axis": "depth",
                "level": 2,
                "holdout": False,
                "intended_class": "within_model",
                "class": CLASS_WITHIN,
                "ok_for_solved": True,
                "expected_boundary_confirmed": False,
                "decision_wall_ms": 1,
                "n_s": 2,
                "facts_examined": 4,
            }
        ],
        organism_ok=True,
        bound_ok=True,
    )
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert "0.0.004" not in str(summary.get("ex0s", ""))


if __name__ == "__main__":
    test_locks()
    test_holdouts_and_boundary_preregistered()
    with tempfile.TemporaryDirectory() as d:
        test_acquired_edge_invariant(Path(d))
    test_local_opt_expected_hold()
    test_reuse_a_discriminator()
    test_reuse_a_expected_boundary()
    test_reuse_b_within_model()
    test_never_stamps_004()
    print("ok")
