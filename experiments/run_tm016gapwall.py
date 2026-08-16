"""TM.0.16.GAPWALL: continuity capacity wall on frozen ALIASFINGER-on.

Wall probe only. Product stays 0.0.004; earned_next=false; ex0s=null.
Empty-event bridging is reported only as empty-event skip semantics.
No object-continuity mechanism is implemented or claimed here.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import traverse_hold
from experiments.run_tm016aliasfinger import (
    ALIAS_FINGER_LOCK,
    CANDIDATE_LOCK as ALIAS_FINGER_CANDIDATE_LOCK,
    EVIDENCE_PREREG as ALIAS_EVIDENCE_PREREG,
    make_finger,
)
from experiments.run_tm016relate import (
    GENOME_016_LOCK,
    RELATE_LOCK,
    edge_support,
    empty_birth,
    evidence_winner_did,
    observe_event,
)
from three_memory.policy import UsePolicy

PREREG_LOCK = REPO_ROOT / "docs" / "gap_wall.prereg.lock"
GAP_WALL_LOCK = REPO_ROOT / "docs" / "gap_wall.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm016gapwall_results.md"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"

DEFAULT_SEED = 12345
CELL_IDS = (
    "G0_adjacent",
    "G1_empty_skip",
    "G2_episode_gap",
    "G3_distractor",
    "G4_one_reappear",
    "G5_two_reappear",
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


def _motor(trav: dict[str, Any]) -> str:
    return str(trav.get("action_name") or "hold").lower()


def fresh_world(tmp: Path, name: str, policy: UsePolicy) -> tuple[Path, Any]:
    """Fresh organism, empty S, and reset rho for one isolated cell."""
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_finger(s_dir, policy)
    ag.reset_rho()
    return s_dir, ag


def run_episode_capture(
    ag: Any, events: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for visible in events:
        focus = visible[0] if visible else None
        rows.append(observe_event(ag, visible, focus=focus))
    ag.end_event_episode()
    return rows


def _query(ag: Any, cue: str = "a") -> dict[str, Any]:
    ag.reset_rho()
    return traverse_hold(ag, cue)


def _gap_ok(result: dict[str, Any]) -> bool:
    return (
        result.get("ok") is True
        and result.get("why") == "empty_visible"
        and int(result.get("wrote") or 0) == 0
        and int(result.get("updated") or 0) == 0
    )


def cell_g0_adjacent(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, "g0", policy)
    for _ in range(3):
        run_episode_capture(ag, (["a"], ["b"], ["c"]))
    trav = _query(ag)
    ok = (
        edge_support(s_dir, "a", "b") == 3
        and edge_support(s_dir, "b", "c") == 3
        and evidence_winner_did(s_dir, "a") == "b"
        and evidence_winner_did(s_dir, "b") == "c"
        and str(trav.get("lived_bind") or "").lower() == "c"
        and bool(trav.get("lived_pending"))
        and not trav.get("evidence_tie")
    )
    return _cell(
        "G0_adjacent",
        ok,
        support_a_b=edge_support(s_dir, "a", "b"),
        support_b_c=edge_support(s_dir, "b", "c"),
        winner_a=evidence_winner_did(s_dir, "a"),
        winner_b=evidence_winner_did(s_dir, "b"),
        lived_bind=trav.get("lived_bind"),
        motor=_motor(trav),
        unique_route_completion=ok,
        continuity_not_at_issue=True,
    )


def cell_g1_empty_skip(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, "g1", policy)
    events = run_episode_capture(ag, (["a"], [], ["b"], ["c"]))
    gap = events[1]
    trav = _query(ag)
    route_ok = (
        str(trav.get("lived_bind") or "").lower() == "c"
        and bool(trav.get("lived_pending"))
        and not trav.get("evidence_tie")
    )
    ok = (
        _gap_ok(gap)
        and edge_support(s_dir, "a", "b") == 1
        and edge_support(s_dir, "b", "c") == 1
        and route_ok
    )
    return _cell(
        "G1_empty_skip",
        ok,
        gap_why=gap.get("why"),
        gap_wrote=int(gap.get("wrote") or 0),
        gap_updated=int(gap.get("updated") or 0),
        support_a_b=edge_support(s_dir, "a", "b"),
        support_b_c=edge_support(s_dir, "b", "c"),
        lived_bind=trav.get("lived_bind"),
        motor=_motor(trav),
        unique_route_completion=route_ok,
        empty_event_skip_semantics=True,
        object_continuity_claim=False,
    )


def cell_g2_episode_gap(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, "g2", policy)
    run_episode_capture(ag, (["a"],))
    run_episode_capture(ag, (["a"],))
    trav = _query(ag)
    ok = (
        edge_support(s_dir, "a", "a") == 0
        and evidence_winner_did(s_dir, "a") is None
        and _motor(trav) == "hold"
        and trav.get("lived_bind") is None
    )
    return _cell(
        "G2_episode_gap",
        ok,
        support_a_a=edge_support(s_dir, "a", "a"),
        winner_a=evidence_winner_did(s_dir, "a"),
        lived_bind=trav.get("lived_bind"),
        motor=_motor(trav),
        cross_episode_bridge=False,
        inherited_episode_frontier=False,
    )


def cell_g3_distractor(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, "g3", policy)
    run_episode_capture(ag, (["a"], ["d"], ["a"]))
    trav = _query(ag)
    structural = (
        edge_support(s_dir, "a", "d") == 1
        and edge_support(s_dir, "d", "a") == 1
        and edge_support(s_dir, "a", "a") == 0
    )
    ok = structural and _motor(trav) == "hold" and bool(trav.get("compose_hold"))
    return _cell(
        "G3_distractor",
        ok,
        support_a_d=edge_support(s_dir, "a", "d"),
        support_d_a=edge_support(s_dir, "d", "a"),
        support_a_a=edge_support(s_dir, "a", "a"),
        lived_bind=trav.get("lived_bind"),
        motor=_motor(trav),
        compose_hold=bool(trav.get("compose_hold")),
        distractor_edges_authored=structural,
        direct_continuity_privileged=False,
        unique_route_completion=False,
        honest_report="fragmented_through_distractor",
    )


def cell_g4_one_reappear(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, "g4", policy)
    events = run_episode_capture(ag, (["a"], [], ["u1"]))
    gap = events[1]
    trav = _query(ag)
    unique = (
        evidence_winner_did(s_dir, "a") == "u1"
        and str(trav.get("lived_bind") or "").lower() == "u1"
        and bool(trav.get("lived_pending"))
        and not trav.get("evidence_tie")
    )
    hold = _motor(trav) == "hold" and not unique
    ok = _gap_ok(gap) and edge_support(s_dir, "a", "u1") == 1 and (unique or hold)
    return _cell(
        "G4_one_reappear",
        ok,
        gap_why=gap.get("why"),
        support_a_u1=edge_support(s_dir, "a", "u1"),
        winner_a=evidence_winner_did(s_dir, "a"),
        lived_bind=trav.get("lived_bind"),
        motor=_motor(trav),
        unique_assignment=unique,
        hold=hold,
        measured_outcome="unique" if unique else "hold",
        measurement_only=True,
        honest_label=(
            "single_candidate_selected_by_empty_event_skip"
            if unique
            else "single_candidate_hold"
        ),
        object_continuity_claim=False,
    )


def cell_g5_two_reappear(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, "g5", policy)
    events = run_episode_capture(ag, (["a"], [], ["u1", "u2"]))
    gap = events[1]
    trav = _query(ag)
    lived = str(trav.get("lived_bind") or "").lower()
    no_unique = lived not in {"u1", "u2"}
    ok = (
        _gap_ok(gap)
        and edge_support(s_dir, "a", "u1") == 1
        and edge_support(s_dir, "a", "u2") == 1
        and evidence_winner_did(s_dir, "a") is None
        and _motor(trav) == "hold"
        and bool(trav.get("compose_hold"))
        and bool(trav.get("evidence_tie"))
        and no_unique
    )
    return _cell(
        "G5_two_reappear",
        ok,
        gap_why=gap.get("why"),
        support_a_u1=edge_support(s_dir, "a", "u1"),
        support_a_u2=edge_support(s_dir, "a", "u2"),
        winner_a=evidence_winner_did(s_dir, "a"),
        lived_bind=trav.get("lived_bind"),
        motor=_motor(trav),
        compose_hold=bool(trav.get("compose_hold")),
        evidence_tie=bool(trav.get("evidence_tie")),
        unique_assignment=False,
        irreducible_ambiguity=True,
        fail_closed_on_unique_peer=no_unique,
    )


CELLS = (
    cell_g0_adjacent,
    cell_g1_empty_skip,
    cell_g2_episode_gap,
    cell_g3_distractor,
    cell_g4_one_reappear,
    cell_g5_two_reappear,
)


def load_prereg(path: Path = PREREG_LOCK) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prereg_lock(path: Path = PREREG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/gap_wall.prereg.lock missing", {}
    lock = load_prereg(path)
    if lock.get("lab") != "TM.0.16.GAPWALL":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "product/earn drift", lock
    if lock.get("not_tm017") is not True:
        return False, "not_tm017 drift", lock
    if lock.get("cell_ids") != list(CELL_IDS):
        return False, "cell_ids drift", lock
    fixtures = lock.get("fixtures") or {}
    expected_episodes = {
        "G0_adjacent": [[["a"], ["b"], ["c"]]] * 3,
        "G1_empty_skip": [[["a"], [], ["b"], ["c"]]],
        "G2_episode_gap": [[["a"]], [["a"]]],
        "G3_distractor": [[["a"], ["d"], ["a"]]],
        "G4_one_reappear": [[["a"], [], ["u1"]]],
        "G5_two_reappear": [[["a"], [], ["u1", "u2"]]],
    }
    for cell, episodes in expected_episodes.items():
        if (fixtures.get(cell) or {}).get("episodes") != episodes:
            return False, f"fixture schedule drift: {cell}", lock
    g1_scorer = (fixtures.get("G1_empty_skip") or {}).get("scorer") or {}
    if g1_scorer.get("empty_event_skip_semantics") is not True:
        return False, "G1 skip label drift", lock
    if g1_scorer.get("object_continuity_claim") is not False:
        return False, "G1 continuity claim drift", lock
    g4_scorer = (fixtures.get("G4_one_reappear") or {}).get("scorer") or {}
    if g4_scorer.get("measurement_only") is not True:
        return False, "G4 measurement contract drift", lock
    g5_scorer = (fixtures.get("G5_two_reappear") or {}).get("scorer") or {}
    if g5_scorer.get("fail_closed_on_unique_peer") is not True:
        return False, "G5 fail-closed drift", lock
    priors = lock.get("prior_lock_shas") or {}
    paths = {
        "alias_finger.lock": ALIAS_FINGER_LOCK,
        "alias_finger.candidate.lock": ALIAS_FINGER_CANDIDATE_LOCK,
        "alias_evidence.prereg.lock": ALIAS_EVIDENCE_PREREG,
        "genome_016.lock": GENOME_016_LOCK,
        "relate_016.lock": RELATE_LOCK,
    }
    for name, prior_path in paths.items():
        if priors.get(name) != _sha_file(prior_path):
            return False, f"prior pin drift: {name}", lock
    serialized = json.dumps(lock, sort_keys=True)
    for banned in ('"run_tm016gapwall_sha"', '"agent_sha"'):
        if banned in serialized:
            return False, "prereg contains GAPWALL implementation SHA", lock
    return True, "gap_wall.prereg.lock intact", lock


def gap_wall_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.16.GAPWALL",
        "lab": "TM.0.16.GAPWALL",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "not_tm017": True,
        "label": "continuity capacity wall on frozen ALIASFINGER-on",
        "cell_ids": list(CELL_IDS),
        "cell_ok": {row["cell"]: bool(row.get("ok")) for row in rows},
        "observed": {
            "G1_empty_event_skip_semantics": bool(rows[1].get("empty_event_skip_semantics")),
            "G2_cross_episode_bridge": bool(rows[2].get("cross_episode_bridge")),
            "G3_distractor_report": rows[3].get("honest_report"),
            "G4_measured_outcome": rows[4].get("measured_outcome"),
            "G5_irreducible_ambiguity": bool(rows[5].get("irreducible_ambiguity")),
        },
        "agent_sha": _sha_file(AGENT_PY),
        "make_finger_sha": _sha_src(make_finger),
        "run_tm016gapwall_sha": _sha_file(Path(__file__)),
        "gap_wall_prereg_sha": _sha_file(PREREG_LOCK),
        "prior_lock_shas": {
            "alias_finger.lock": _sha_file(ALIAS_FINGER_LOCK),
            "alias_finger.candidate.lock": _sha_file(ALIAS_FINGER_CANDIDATE_LOCK),
            "alias_evidence.prereg.lock": _sha_file(ALIAS_EVIDENCE_PREREG),
            "genome_016.lock": _sha_file(GENOME_016_LOCK),
            "relate_016.lock": _sha_file(RELATE_LOCK),
        },
        "bounded_fact": "Frozen ALIASFINGER-on bridges an empty event only because empty visible is skipped, clears the frontier at end_event_episode, treats distractors as ordinary events, and HOLDS when two post-gap peers tie.",
        "refuse": [
            "learned object continuity claim",
            "editing agent.py / RELATE / ALIASFINGER / alias-evidence contract",
            "continuity-evidence contract or persistence candidate this pass",
            "TM.0.17 / 0.0.005 / FAMILY / LOOKAHEAD / pixels",
            "earned_next=true or non-null ex0s",
        ],
    }


def write_gap_wall_lock(rows: list[dict[str, Any]]) -> dict[str, Any]:
    snap = gap_wall_lock_snapshot(rows)
    GAP_WALL_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_gap_wall_lock(
    rows: list[dict[str, Any]] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    if not GAP_WALL_LOCK.exists():
        return False, "docs/gap_wall.lock missing", {}
    lock = json.loads(GAP_WALL_LOCK.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "product/earn drift", lock
    if lock.get("run_tm016gapwall_sha") != _sha_file(Path(__file__)):
        return False, "runner SHA drift", lock
    if lock.get("agent_sha") != _sha_file(AGENT_PY):
        return False, "agent.py drift", lock
    if lock.get("gap_wall_prereg_sha") != _sha_file(PREREG_LOCK):
        return False, "prereg SHA drift", lock
    priors = lock.get("prior_lock_shas") or {}
    paths = {
        "alias_finger.lock": ALIAS_FINGER_LOCK,
        "alias_finger.candidate.lock": ALIAS_FINGER_CANDIDATE_LOCK,
        "alias_evidence.prereg.lock": ALIAS_EVIDENCE_PREREG,
        "genome_016.lock": GENOME_016_LOCK,
        "relate_016.lock": RELATE_LOCK,
    }
    for name, prior_path in paths.items():
        if priors.get(name) != _sha_file(prior_path):
            return False, f"prior pin drift: {name}", lock
    if rows is not None:
        expected = {row["cell"]: bool(row.get("ok")) for row in rows}
        if lock.get("cell_ok") != expected:
            return False, "cell_ok drift", lock
    return True, "gap_wall.lock intact", lock


def run_gap_wall(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    prereg_ok, prereg_why, _ = verify_prereg_lock()
    if not prereg_ok:
        raise RuntimeError(prereg_why)
    policy = UsePolicy(seed=seed)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tm016gapwall_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            rows.append(fn(root, policy))
    n_ok = sum(1 for row in rows if row.get("ok"))
    summary = {
        "version": "TM.0.16.GAPWALL",
        "lab": "TM.0.16.GAPWALL",
        "label": "continuity capacity wall",
        "ok": n_ok == len(rows) == 6,
        "n_ok": n_ok,
        "n_cells": len(rows),
        "earned_next": False,
        "ex0s": None,
        "seed": seed,
        "rows": rows,
    }
    if write_lock:
        write_gap_wall_lock(rows)
        write_results_md(summary)
    return summary


def write_results_md(summary: dict[str, Any]) -> None:
    by = {row["cell"]: row for row in summary["rows"]}
    lines = [
        "# TM.0.16.GAPWALL results",
        "",
        f"**Recorded:** frozen G0–G5 → **{summary['n_ok']}/{summary['n_cells']}**",
        "",
        "- Product: `0.0.004`",
        "- `earned_next=false`",
        "- `ex0s=null`",
        "- Organism: frozen ALIASFINGER-on; `agent.py` unchanged",
        "",
        "| Cell | Result | Honest reading |",
        "|------|--------|----------------|",
        f"| G0 adjacent | {'PASS' if by['G0_adjacent']['ok'] else 'FAIL'} | Unique adjacent route; continuity not at issue. |",
        f"| G1 empty skip | {'PASS' if by['G1_empty_skip']['ok'] else 'FAIL'} | Bridge is existing empty-event skip semantics, not learned object continuity. |",
        f"| G2 episode gap | {'PASS' if by['G2_episode_gap']['ok'] else 'FAIL'} | No cross-episode frontier bridge. |",
        f"| G3 distractor | {'PASS' if by['G3_distractor']['ok'] else 'FAIL'} | The distractor is authored into the route; `a` is not privileged. |",
        f"| G4 one reappear | {'PASS' if by['G4_one_reappear']['ok'] else 'FAIL'} | Measured `{by['G4_one_reappear']['measured_outcome']}`; if unique, it is skip-driven only. |",
        f"| G5 two reappear | {'PASS' if by['G5_two_reappear']['ok'] else 'FAIL'} | Equal peers tie; behavior HOLDS without choosing either. |",
        "",
        "## Bounded fact",
        "",
        "Frozen ALIASFINGER-on preserves the pre-gap bag across an empty event because the empty event is skipped. It loses that frontier at an episode boundary, routes through a visible distractor rather than preserving an object, and cannot resolve two equally supported post-gap candidates.",
        "",
        "This is a capacity wall, not evidence of learned object continuity.",
        "",
        "## Next",
        "",
        "Freeze a separate continuity-evidence contract. Only after that contract may an opt-in persistence candidate be proposed.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m experiments.run_tm016gapwall --verify-prereg",
        "python tests/test_tm016gapwall.py",
        "```",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-prereg", action="store_true")
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.verify_prereg:
        ok, why, _ = verify_prereg_lock()
        print(json.dumps({"ok": ok, "why": why}, indent=2))
        raise SystemExit(0 if ok else 1)
    summary = run_gap_wall(seed=args.seed, write_lock=args.write_lock)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    for row in summary["rows"]:
        detail = {key: value for key, value in row.items() if key not in {"cell", "ok"}}
        print(f"  {row['cell']}: {row.get('ok')} {detail}")
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
