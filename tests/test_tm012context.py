"""TM.0.12.CONTEXT: freeze, H0–H4, C0–C7 discriminators, no 0.0.004 stamp."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import (
    CLASS_EXPECTED,
    CLASS_WITHIN,
    CONTEXT_LOCK,
    HYPOTHESES,
    MINIMALITY_SHAPE,
    aggregate,
    all_cells,
    gen_c0,
    gen_c1,
    gen_c2,
    gen_c3,
    gen_c4,
    gen_c7_a,
    gen_c7_b,
    run_cell,
    run_context,
    verify_context_lock,
    write_context_lock,
)
from experiments.run_tm011family import verify_freeze as verify_organism


def test_locks():
    write_context_lock()
    ok, why, snap = verify_context_lock()
    assert ok, why
    assert snap["ex0s_under_test"] == "0.0.003"
    assert HYPOTHESES["H4"]["role"] == "diagnostic_upper_bound"
    assert snap["hypotheses"]["H4"]["role"] == "diagnostic_upper_bound"
    assert MINIMALITY_SHAPE == snap["minimality_shape"]
    assert "c7_indistinguish_a" in snap["expected_boundary_cells"]
    assert "(token, here) as victory" in snap["refuse"]
    org_ok, org_why, _ = verify_organism()
    assert org_ok, org_why
    assert CONTEXT_LOCK.exists()


def test_family_split():
    cells = all_cells()
    ids = {c.cell_id for c in cells}
    assert ids >= {
        "c0_unique",
        "c1_benign_reuse",
        "c2_here_split",
        "c3_pred_split",
        "c4_pred_collision",
        "c5_path_depth",
        "c6_evidence_trap",
        "c7_indistinguish_a",
        "c7_indistinguish_b",
    }
    within = {c.cell_id for c in cells if c.intended_class == "within_model"}
    assert within == {"c0_unique", "c1_benign_reuse"}
    holdouts = {c.cell_id for c in cells if c.holdout}
    assert holdouts >= {"c4_pred_collision", "c5_path_depth", "c7_indistinguish_a", "c7_indistinguish_b"}


def test_c2_h0_ne_h1():
    cell = gen_c2(12345)
    preds = cell.hypothesis_preds
    assert preds["H0"] != preds["H1"]
    assert preds["H0_ne_H1"] is True
    # Frozen expects bind-only (H0); context expect on cue_z differs
    assert cell.probes[1].expect == preds["H0"]
    assert cell.probes[1].context_expect == preds["H1"]


def test_c3_here_identical_pred_differs():
    cell = gen_c3(12345)
    heres = {r.here for r in cell.relations if r.bind}
    # All on-path facts share chb for the Y rivals and arrivals
    y_edges = [r for r in cell.relations if r.role in ("yx", "yz", "xy", "zy")]
    assert all(r.here == "chb" for r in y_edges)
    assert cell.probes[0].context_expect != cell.probes[1].context_expect
    assert cell.probes[0].expect == cell.probes[1].expect  # frozen same


def test_c4_shared_predecessor():
    cell = gen_c4(12345)
    by = {r.role: r for r in cell.relations}
    assert by["xa"].did == by["za"].did == by["ay"].bind
    assert cell.hypothesis_preds["H2"] == "cannot_distinguish"


def test_c7_indistinguishable_state():
    a = gen_c7_a(12345)
    b = gen_c7_b(12345)
    assert [(r.bind, r.did, r.init, r.here) for r in a.relations] == [
        (r.bind, r.did, r.init, r.here) for r in b.relations
    ]
    assert a.probes[0].cue == b.probes[0].cue
    assert a.probes[0].expect == b.probes[0].expect == "hold"
    assert a.probes[0].context_expect != b.probes[0].context_expect


def test_c0_c1_within_runtime():
    for cid in ("c0_unique", "c1_benign_reuse"):
        row = run_cell({"cell_id": cid, "seed": 12345, "dest": tempfile.mkdtemp()})
        assert row["class"] == CLASS_WITHIN, (cid, row)
        assert row["ok_for_solved"]


def test_c2_boundary_runtime():
    row = run_cell({"cell_id": "c2_here_split", "seed": 12345, "dest": tempfile.mkdtemp()})
    assert row["class"] == CLASS_EXPECTED
    assert row["expected_boundary_confirmed"]


def test_c7_witness_runtime():
    a = run_cell({"cell_id": "c7_indistinguish_a", "seed": 12345, "dest": tempfile.mkdtemp()})
    b = run_cell({"cell_id": "c7_indistinguish_b", "seed": 12345, "dest": tempfile.mkdtemp()})
    assert a["class"] == CLASS_EXPECTED and b["class"] == CLASS_EXPECTED
    assert a["motors"] == b["motors"]
    assert a["context_expects"] != b["context_expects"]


def test_never_stamps_004():
    summary = aggregate(
        [
            {
                "cell_id": "c0_unique",
                "axis": "c0",
                "level": "x",
                "holdout": False,
                "intended_class": "within_model",
                "class": CLASS_WITHIN,
                "ok_for_solved": True,
                "expected_boundary_confirmed": False,
                "motors": ["press"],
                "context_expects": ["press"],
                "discriminates": "apparatus_intact",
            }
        ],
        organism_ok=True,
        context_ok=True,
    )
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"
    assert "0.0.004" not in str(summary.get("ex0s", ""))


if __name__ == "__main__":
    test_locks()
    test_family_split()
    test_c2_h0_ne_h1()
    test_c3_here_identical_pred_differs()
    test_c4_shared_predecessor()
    test_c7_indistinguishable_state()
    test_c0_c1_within_runtime()
    test_c2_boundary_runtime()
    test_c7_witness_runtime()
    test_never_stamps_004()
    # unused import keep for smoke discoverability
    _ = (gen_c0, gen_c1, run_context)
    print("ok")
