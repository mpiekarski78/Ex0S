"""TM.0.12.MINIMAP: rules — not the result table — and fail-closed CONTEXT pin."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012minimap import (
    CONTEXT_LOCK,
    MINIMAP_LOCK,
    NO_INCOMING_FID,
    NO_INCOMING_HERE,
    OUTCOME_APPARATUS_ERROR,
    OUTCOME_BENIGN,
    OUTCOME_COLLISION,
    OUTCOME_DISTINGUISHES,
    OUTCOME_INADMISSIBLE,
    OUTCOME_UNOBSERVABLE,
    VICTORY_CONTRASTS,
    answer_derived_outgoing_fid,
    extract_states,
    least_sufficient_candidate,
    locked_contrasts,
    path_and_frontier,
    refuse_answer_derived_fid,
    run_minimap,
    score_contrast,
    verify_minimap_lock,
)
from experiments.run_tm012context import all_cells, gen_c4, gen_c5, gen_c7_a, gen_c7_b


def test_locks_fail_closed_no_rewrite():
    ok, why, snap = verify_minimap_lock()
    assert ok, why
    assert snap["earned_next"] is False
    assert snap["c7_classification"] == "UNOBSERVABLE_FROM_PROVENANCE"
    assert snap["h4_rule"]["outgoing_answer_fid"] == "inadmissible_answer_id"
    assert MINIMAP_LOCK.exists()
    assert CONTEXT_LOCK.exists()
    pin = json.loads(MINIMAP_LOCK.read_text(encoding="utf-8"))["context_012_lock_sha"]
    live = hashlib.sha256(CONTEXT_LOCK.read_bytes()).hexdigest()
    assert pin == live
    # Tests must not import write helpers that rewrite CONTEXT.
    import experiments.run_tm012minimap as mm

    assert not hasattr(mm, "write_context_lock")


def test_contrasts_are_preregistered():
    c = locked_contrasts()
    assert set(c) == {"C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"}
    assert c["C0"]["role"] == "positive_control"
    assert c["C1"]["role"] == "benign"
    assert c["C7"]["role"] == "unobservable_from_provenance"
    assert c["C7"]["left"]["cell_id"] != c["C7"]["right"]["cell_id"]
    assert VICTORY_CONTRASTS == ("C2", "C3", "C4", "C5", "C6")


def test_c0_h0_distinguishes():
    cells = {x.cell_id: x for x in all_cells(12345)}
    c0 = cells["c0_unique"]
    px, pz = c0.probes[0], c0.probes[1]
    sx = extract_states(c0.relations, px.cue)
    sz = extract_states(c0.relations, pz.cue)
    assert sx["H0"] != sz["H0"]
    out = score_contrast(
        contrast_id="C0",
        role="positive_control",
        left_states=sx,
        right_states=sz,
        left_motor=px.context_expect,
        right_motor=pz.context_expect,
        candidate="H0",
    )
    assert out == OUTCOME_DISTINGUISHES


def test_c1_benign():
    cells = {x.cell_id: x for x in all_cells(12345)}
    c1 = cells["c1_benign_reuse"]
    px, pz = c1.probes[0], c1.probes[1]
    assert px.context_expect == pz.context_expect
    sx = extract_states(c1.relations, px.cue)
    sz = extract_states(c1.relations, pz.cue)
    assert sx["H0"] == sz["H0"]  # same frontier Y
    out = score_contrast(
        contrast_id="C1",
        role="benign",
        left_states=sx,
        right_states=sz,
        left_motor=px.context_expect,
        right_motor=pz.context_expect,
        candidate="H0",
    )
    assert out == OUTCOME_BENIGN


def test_c7_zero_length_sentinels_and_unobservable():
    a = gen_c7_a(12345)
    b = gen_c7_b(12345)
    sa = extract_states(a.relations, a.probes[0].cue)
    sb = extract_states(b.relations, b.probes[0].cue)
    y = a.probes[0].cue.lower()
    assert sa["H0"] == (y,)
    assert sa["H1"] == (y, NO_INCOMING_HERE)
    assert sa["H2"] == (y, y)
    assert sa["H3a"] == (y, y)
    assert sa["H3b"] == (y, (y,))
    assert sa["H4"] == (y, NO_INCOMING_FID)
    assert sa == sb
    assert a.probes[0].context_expect != b.probes[0].context_expect
    out = score_contrast(
        contrast_id="C7",
        role="unobservable_from_provenance",
        left_states=sa,
        right_states=sb,
        left_motor=a.probes[0].context_expect,
        right_motor=b.probes[0].context_expect,
        candidate="H3a",
    )
    assert out == OUTCOME_UNOBSERVABLE
    # Vacuous short-circuit refused: differing states must be apparatus_error.
    fake_b = {k: ("other",) for k in sb}
    bad = score_contrast(
        contrast_id="C7",
        role="unobservable_from_provenance",
        left_states=sa,
        right_states=fake_b,
        left_motor=a.probes[0].context_expect,
        right_motor=b.probes[0].context_expect,
        candidate="H0",
    )
    assert bad == OUTCOME_APPARATUS_ERROR


def test_c4_h4_incoming_fid_collides():
    cell = gen_c4(12345)
    px, pz = cell.probes[0], cell.probes[1]
    sx = extract_states(cell.relations, px.cue)
    sz = extract_states(cell.relations, pz.cue)
    assert sx["H4"] == sz["H4"]  # shared A→Y
    assert sx["H3a"] != sz["H3a"]  # origins differ
    assert sx["H2"] == sz["H2"]  # same pred A
    out = score_contrast(
        contrast_id="C4",
        role="provenance",
        left_states=sx,
        right_states=sz,
        left_motor=px.context_expect,
        right_motor=pz.context_expect,
        candidate="H4",
    )
    assert out == OUTCOME_COLLISION


def test_c5_origin_and_path_both_distinguish_here_collides():
    """C5 varies origin and path; cannot force full-path necessity."""
    cell = gen_c5(12345)
    px, pz = cell.probes[0], cell.probes[1]
    sx = extract_states(cell.relations, px.cue)
    sz = extract_states(cell.relations, pz.cue)
    assert sx["H1"] == sz["H1"]  # same arrival here
    assert sx["H3a"] != sz["H3a"]
    assert sx["H3b"] != sz["H3b"]
    assert len(sx["H3b"][1]) != len(sz["H3b"][1])
    for cand in ("H3a", "H3b"):
        assert (
            score_contrast(
                contrast_id="C5",
                role="provenance",
                left_states=sx,
                right_states=sz,
                left_motor=px.context_expect,
                right_motor=pz.context_expect,
                candidate=cand,
            )
            == OUTCOME_DISTINGUISHES
        )


def test_outgoing_answer_fid_inadmissible():
    cells = {x.cell_id: x for x in all_cells(12345)}
    c2 = cells["c2_here_split"]
    p = c2.probes[0]
    ans = answer_derived_outgoing_fid(c2.relations, p.cue, p.context_expect)
    assert ans is not None
    _path, frontier, incoming = path_and_frontier(c2.relations, p.cue)
    assert incoming is not None
    assert ans != incoming.fid
    assert refuse_answer_derived_fid(incoming.fid, ans) == OUTCOME_INADMISSIBLE


def test_provenance_same_motor_is_apparatus_error():
    cells = {x.cell_id: x for x in all_cells(12345)}
    c2 = cells["c2_here_split"]
    sx = extract_states(c2.relations, c2.probes[0].cue)
    sz = extract_states(c2.relations, c2.probes[1].cue)
    bad = score_contrast(
        contrast_id="C2",
        role="provenance",
        left_states=sx,
        right_states=sz,
        left_motor="press",
        right_motor="press",
        candidate="H0",
    )
    assert bad == OUTCOME_APPARATUS_ERROR


def test_benign_different_motor_is_apparatus_error():
    cells = {x.cell_id: x for x in all_cells(12345)}
    c1 = cells["c1_benign_reuse"]
    sx = extract_states(c1.relations, c1.probes[0].cue)
    sz = extract_states(c1.relations, c1.probes[1].cue)
    bad = score_contrast(
        contrast_id="C1",
        role="benign",
        left_states=sx,
        right_states=sz,
        left_motor="press",
        right_motor="tune",
        candidate="H0",
    )
    assert bad == OUTCOME_APPARATUS_ERROR


def test_computed_table_and_least_candidate():
    summary = run_minimap(seed=12345)
    assert summary["ok"], summary.get("why")
    assert summary["earned_next"] is False
    table = summary["table"]
    assert table["H0"]["C0"] == OUTCOME_DISTINGUISHES
    assert table["H0"]["C1"] == OUTCOME_BENIGN
    for v in VICTORY_CONTRASTS:
        assert table["H0"][v] == OUTCOME_COLLISION
    assert table["H1"]["C2"] == OUTCOME_DISTINGUISHES
    assert table["H1"]["C3"] == OUTCOME_COLLISION
    assert table["H2"]["C4"] == OUTCOME_COLLISION
    assert table["H3a"]["C4"] == OUTCOME_DISTINGUISHES
    assert table["H4"]["C4"] == OUTCOME_COLLISION
    assert table["H3a"]["C7"] == OUTCOME_UNOBSERVABLE
    assert least_sufficient_candidate(table) == "H3a"
    assert summary["least_sufficient_candidate"] == "H3a"
    assert "0.0.004" not in str(summary.get("ex0s", ""))
    assert "least-structured sufficient candidate" in summary["claim"]


def test_never_stamps_004():
    summary = run_minimap(seed=12345)
    assert summary["earned_next"] is False
    assert summary["ex0s_under_test"] == "0.0.003"


if __name__ == "__main__":
    test_locks_fail_closed_no_rewrite()
    test_contrasts_are_preregistered()
    test_c0_h0_distinguishes()
    test_c1_benign()
    test_c7_zero_length_sentinels_and_unobservable()
    test_c4_h4_incoming_fid_collides()
    test_c5_origin_and_path_both_distinguish_here_collides()
    test_outgoing_answer_fid_inadmissible()
    test_provenance_same_motor_is_apparatus_error()
    test_benign_different_motor_is_apparatus_error()
    test_computed_table_and_least_candidate()
    test_never_stamps_004()
    print("ok")
