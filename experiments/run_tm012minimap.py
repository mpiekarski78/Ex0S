"""TM.0.12.MINIMAP: representation distinguishability on locked CONTEXT contrasts.

No agent.py, no policy, no probe of 0.0.003, no genome change, no Ex0S 0.0.004.
Scores only preregistered contrast IDs. Never rewrites docs/context_012.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import (  # noqa: E402
    CONTEXT_LOCK,
    DEFAULT_SEED,
    MOTORS,
    ProbeSpec,
    Rel,
    all_cells,
    seed_list_blob,
)

MINIMAP_LOCK = REPO_ROOT / "docs" / "minimap_012.lock"

NO_INCOMING_HERE = "NO_INCOMING_HERE"
NO_INCOMING_FID = "NO_INCOMING_FID"

OUTCOME_DISTINGUISHES = "distinguishes"
OUTCOME_COLLISION = "collision"
OUTCOME_BENIGN = "benign"
OUTCOME_UNOBSERVABLE = "unobservable"
OUTCOME_APPARATUS_ERROR = "apparatus_error"
OUTCOME_INADMISSIBLE = "inadmissible_answer_id"

ROLE_POSITIVE = "positive_control"
ROLE_BENIGN = "benign"
ROLE_PROVENANCE = "provenance"
ROLE_UNOBSERVABLE = "unobservable_from_provenance"

VICTORY_CONTRASTS = ("C2", "C3", "C4", "C5", "C6")
CANDIDATE_ORDER = ("H0", "H1", "H2", "H3a", "H3b", "H4")

# Less-structured → more-structured for “least sufficient candidate” (H4 is diagnostic, not ranked).
STRUCTURE_RANK = {
    "H0": 0,
    "H1": 1,
    "H2": 2,
    "H3a": 3,
    "H3b": 4,
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def locked_contrasts() -> dict[str, dict[str, Any]]:
    """Preregistered contrast IDs — never discover pairings from context_expect."""
    return {
        "C0": {
            "left": {"cell_id": "c0_unique", "probe_label": "cue_x"},
            "right": {"cell_id": "c0_unique", "probe_label": "cue_z"},
            "role": ROLE_POSITIVE,
        },
        "C1": {
            "left": {"cell_id": "c1_benign_reuse", "probe_label": "cue_x"},
            "right": {"cell_id": "c1_benign_reuse", "probe_label": "cue_z"},
            "role": ROLE_BENIGN,
        },
        "C2": {
            "left": {"cell_id": "c2_here_split", "probe_label": "cue_x"},
            "right": {"cell_id": "c2_here_split", "probe_label": "cue_z"},
            "role": ROLE_PROVENANCE,
        },
        "C3": {
            "left": {"cell_id": "c3_pred_split", "probe_label": "cue_x"},
            "right": {"cell_id": "c3_pred_split", "probe_label": "cue_z"},
            "role": ROLE_PROVENANCE,
        },
        "C4": {
            "left": {"cell_id": "c4_pred_collision", "probe_label": "cue_x"},
            "right": {"cell_id": "c4_pred_collision", "probe_label": "cue_z"},
            "role": ROLE_PROVENANCE,
        },
        "C5": {
            "left": {"cell_id": "c5_path_depth", "probe_label": "cue_x_short"},
            "right": {"cell_id": "c5_path_depth", "probe_label": "cue_z_long"},
            "role": ROLE_PROVENANCE,
        },
        "C6": {
            "left": {"cell_id": "c6_evidence_trap", "probe_label": "cue_x"},
            "right": {"cell_id": "c6_evidence_trap", "probe_label": "cue_z"},
            "role": ROLE_PROVENANCE,
        },
        "C7": {
            "left": {"cell_id": "c7_indistinguish_a", "probe_label": "cue_y"},
            "right": {"cell_id": "c7_indistinguish_b", "probe_label": "cue_y"},
            "role": ROLE_UNOBSERVABLE,
        },
    }


def candidate_definitions() -> dict[str, dict[str, str]]:
    return {
        "H0": {"name": "token", "state": "(Y,)"},
        "H1": {"name": "token_plus_here", "state": "(Y, incoming.here)"},
        "H2": {"name": "token_plus_predecessor", "state": "(Y, incoming.bind)"},
        "H3a": {"name": "token_plus_origin", "state": "(Y, path_start)"},
        "H3b": {"name": "token_plus_full_path", "state": "(Y, (n0,...,Y))"},
        "H4": {
            "name": "token_plus_incoming_fact_id",
            "state": "(Y, incoming.fid)",
            "role": "incoming_fact_identity_diagnostic",
            "note": "Not an upper bound; C4 shares A→Y so H4 collides",
        },
    }


def zero_length_sentinels() -> dict[str, Any]:
    return {
        "NO_INCOMING_HERE": NO_INCOMING_HERE,
        "NO_INCOMING_FID": NO_INCOMING_FID,
        "H0": ["Y"],
        "H1": ["Y", NO_INCOMING_HERE],
        "H2": ["Y", "Y"],
        "H3a": ["Y", "Y"],
        "H3b": ["Y", ["Y"]],
        "H4": ["Y", NO_INCOMING_FID],
        "note": "Cue is Y; no incoming edge. Exact sentinels — no None.",
    }


def _cell_by_id(seed: int) -> dict[str, Any]:
    return {c.cell_id: c for c in all_cells(seed)}


def _probe_by_label(cell: Any, label: str) -> ProbeSpec:
    for p in cell.probes:
        if p.label == label:
            return p
    raise KeyError(f"probe label {label!r} not in {cell.cell_id}")


def path_and_frontier(rels: list[Rel], cue: str) -> tuple[list[str], str, Rel | None]:
    """Unique walk from cue; frontier = last non-motor before motor choice (or cue if cue is Y).

    Returns (path_nodes, frontier_token, incoming_rel_or_None).
    """
    motors = {m.lower() for m in MOTORS}
    out: dict[str, list[Rel]] = defaultdict(list)
    for r in rels:
        out[r.bind.lower()].append(r)

    current = cue.lower()
    path = [current]
    visited = {current}
    incoming: Rel | None = None

    while True:
        edges = out.get(current, [])
        non_motor = [e for e in edges if e.did.lower() not in motors]
        motor = [e for e in edges if e.did.lower() in motors]
        if motor and not non_motor:
            # Decision frontier: choose among motors at current.
            break
        if len(non_motor) == 1:
            edge = non_motor[0]
            nxt = edge.did.lower()
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            incoming = edge
            current = nxt
            continue
        # No unique non-motor continuation (0 or many) — frontier is current.
        break

    return path, current, incoming


def extract_states(
    rels: list[Rel], cue: str
) -> dict[str, tuple[Any, ...]]:
    """Hypothetical representation states at the compose frontier."""
    path, frontier, incoming = path_and_frontier(rels, cue)
    y = frontier

    if incoming is None:
        # Zero-length / cue-is-frontier (C7).
        return {
            "H0": (y,),
            "H1": (y, NO_INCOMING_HERE),
            "H2": (y, y),
            "H3a": (y, y),
            "H3b": (y, tuple(path)),  # (Y,)
            "H4": (y, NO_INCOMING_FID),
        }

    origin = path[0]
    return {
        "H0": (y,),
        "H1": (y, incoming.here),
        "H2": (y, incoming.bind.lower()),
        "H3a": (y, origin),
        "H3b": (y, tuple(path)),
        "H4": (y, incoming.fid),
    }


def answer_derived_outgoing_fid(rels: list[Rel], cue: str, context_expect: str) -> str | None:
    """Fid of Y→motor matching context_expect — inadmissible as a representation."""
    _path, frontier, _inc = path_and_frontier(rels, cue)
    want = context_expect.lower()
    for r in rels:
        if r.bind.lower() == frontier and r.did.lower() == want:
            return r.fid
    return None


def score_contrast(
    *,
    contrast_id: str,
    role: str,
    left_states: dict[str, tuple[Any, ...]],
    right_states: dict[str, tuple[Any, ...]],
    left_motor: str,
    right_motor: str,
    candidate: str,
) -> str:
    """Score one candidate on a locked contrast. Fail closed on apparatus lies."""
    ls = left_states[candidate]
    rs = right_states[candidate]
    lm = left_motor.lower()
    rm = right_motor.lower()

    if role == ROLE_UNOBSERVABLE:
        # C7: required motors differ; provenance states must collide by construction.
        if lm == rm:
            return OUTCOME_APPARATUS_ERROR
        if ls != rs:
            return OUTCOME_APPARATUS_ERROR
        return OUTCOME_UNOBSERVABLE

    if role == ROLE_BENIGN:
        # C1: required motors must match; collision of states is allowed.
        if lm != rm:
            return OUTCOME_APPARATUS_ERROR
        return OUTCOME_BENIGN

    # positive_control / provenance: motors must differ or the contrast is vacuous.
    if lm == rm:
        return OUTCOME_APPARATUS_ERROR
    if ls == rs:
        return OUTCOME_COLLISION
    return OUTCOME_DISTINGUISHES


def refuse_answer_derived_fid(incoming_fid: str | None, answer_fid: str | None) -> str:
    """Outgoing Y→motor fid chosen by context_expect is never a valid representation."""
    if answer_fid is None:
        return OUTCOME_APPARATUS_ERROR
    if incoming_fid is not None and answer_fid == incoming_fid:
        return OUTCOME_APPARATUS_ERROR
    return OUTCOME_INADMISSIBLE

def context_generator_shas_live() -> dict[str, str]:
    """Match the hashing scheme used by run_tm012context.apparatus_snapshot."""
    from experiments import run_tm012context as ctx

    return {
        "gen_c0_sha": _sha_src(ctx.gen_c0),
        "gen_c1_sha": _sha_src(ctx.gen_c1),
        "gen_c2_sha": _sha_src(ctx.gen_c2),
        "gen_c3_sha": _sha_src(ctx.gen_c3),
        "gen_c4_sha": _sha_src(ctx.gen_c4),
        "gen_c5_sha": _sha_src(ctx.gen_c5),
        "gen_c6_sha": _sha_src(ctx.gen_c6),
        "gen_c7_sha": _sha_bytes(
            (
                inspect.getsource(ctx.gen_c7_a)
                + inspect.getsource(ctx.gen_c7_b)
                + inspect.getsource(ctx._c7_identical_world)
            ).encode()
        ),
    }


def apparatus_snapshot() -> dict[str, Any]:
    if not CONTEXT_LOCK.exists():
        raise FileNotFoundError("docs/context_012.lock missing — do not invent CONTEXT here")
    return {
        "version": "TM.0.12.MINIMAP",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": DEFAULT_SEED,
        "seed_list_sha": _sha_bytes(seed_list_blob().encode()),
        "context_012_lock_sha": _sha_file(CONTEXT_LOCK),
        "contrasts": locked_contrasts(),
        "candidates": candidate_definitions(),
        "zero_length_sentinels": zero_length_sentinels(),
        "victory_contrasts": list(VICTORY_CONTRASTS),
        "h4_rule": {
            "kind": "incoming_fact_identity_diagnostic",
            "outgoing_answer_fid": "inadmissible_answer_id",
            "note": "C4 shares A→Y; incoming fid collides. Not an upper bound.",
        },
        "c7_classification": "UNOBSERVABLE_FROM_PROVENANCE",
        "extractor_sha": _sha_src(extract_states),
        "path_sha": _sha_src(path_and_frontier),
        "scorer_sha": _sha_src(score_contrast),
        "contrast_def_sha": _sha_src(locked_contrasts),
        "refuse": [
            "rewrite context_012.lock",
            "probe 0.0.003 for this table",
            "encode result table in scorer",
            "C7 as CONTEXT victory",
            "outgoing fact_id as representation",
            "stamp Ex0S 0.0.004",
            "genome / agent change",
        ],
    }


def write_minimap_lock(path: Path = MINIMAP_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_minimap_lock(path: Path = MINIMAP_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not path.exists():
        return False, "docs/minimap_012.lock missing; write only via --write-lock", snap
    if not CONTEXT_LOCK.exists():
        return False, "docs/context_012.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    ctx = json.loads(CONTEXT_LOCK.read_text(encoding="utf-8"))

    if snap["context_012_lock_sha"] != lock.get("context_012_lock_sha"):
        return False, "context_012.lock SHA drifted from minimap pin", snap
    if snap["context_012_lock_sha"] != _sha_file(CONTEXT_LOCK):
        return False, "context_012.lock content SHA mismatch", snap

    live_gens = context_generator_shas_live()
    for key, val in live_gens.items():
        if ctx.get(key) != val:
            return False, f"CONTEXT generator drift vs context_012.lock: {key}", snap

    if snap["seed_list_sha"] != ctx.get("seed_list_sha"):
        return False, "seed_list_sha diverged from context_012.lock", snap
    if snap["seed"] != ctx.get("seed"):
        return False, "seed diverged from context_012.lock", snap

    for key in (
        "seed",
        "seed_list_sha",
        "extractor_sha",
        "path_sha",
        "scorer_sha",
        "contrast_def_sha",
        "victory_contrasts",
        "c7_classification",
        "contrasts",
        "candidates",
        "zero_length_sentinels",
        "h4_rule",
    ):
        if snap[key] != lock.get(key):
            return False, f"minimap apparatus drift: {key}", snap

    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    return True, "minimap apparatus frozen", snap


def run_minimap(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok, why, snap = verify_minimap_lock()
    if not ok:
        return {
            "ok": False,
            "why": why,
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
            "cells": {},
        }

    cells = _cell_by_id(seed)
    contrasts = locked_contrasts()
    table: dict[str, dict[str, str]] = {c: {} for c in CANDIDATE_ORDER}
    detail: dict[str, Any] = {}

    for cid, spec in contrasts.items():
        left_cell = cells[spec["left"]["cell_id"]]
        right_cell = cells[spec["right"]["cell_id"]]
        left_p = _probe_by_label(left_cell, spec["left"]["probe_label"])
        right_p = _probe_by_label(right_cell, spec["right"]["probe_label"])
        left_states = extract_states(left_cell.relations, left_p.cue)
        right_states = extract_states(right_cell.relations, right_p.cue)
        left_motor = left_p.context_expect or left_p.expect
        right_motor = right_p.context_expect or right_p.expect

        cell_detail = {
            "role": spec["role"],
            "left": {
                "cell_id": left_cell.cell_id,
                "cue": left_p.cue,
                "context_expect": left_motor,
                "states": {k: list(_serialize_state(v)) for k, v in left_states.items()},
                "path": path_and_frontier(left_cell.relations, left_p.cue)[0],
            },
            "right": {
                "cell_id": right_cell.cell_id,
                "cue": right_p.cue,
                "context_expect": right_motor,
                "states": {k: list(_serialize_state(v)) for k, v in right_states.items()},
                "path": path_and_frontier(right_cell.relations, right_p.cue)[0],
            },
            "outcomes": {},
        }

        for cand in CANDIDATE_ORDER:
            outcome = score_contrast(
                contrast_id=cid,
                role=spec["role"],
                left_states=left_states,
                right_states=right_states,
                left_motor=left_motor,
                right_motor=right_motor,
                candidate=cand,
            )
            table[cand][cid] = outcome
            cell_detail["outcomes"][cand] = outcome
            if outcome == OUTCOME_APPARATUS_ERROR:
                return {
                    "ok": False,
                    "why": f"apparatus_error on {cid}/{cand}",
                    "earned_next": False,
                    "ex0s_under_test": "0.0.003",
                    "table": table,
                    "detail": detail,
                    "cells": {},
                }

        # Answer-derived fid check (must remain inadmissible).
        left_ans = answer_derived_outgoing_fid(
            left_cell.relations, left_p.cue, left_motor
        )
        right_ans = answer_derived_outgoing_fid(
            right_cell.relations, right_p.cue, right_motor
        )
        _lp, _lf, left_inc = path_and_frontier(left_cell.relations, left_p.cue)
        _rp, _rf, right_inc = path_and_frontier(right_cell.relations, right_p.cue)
        cell_detail["answer_derived_fids"] = {"left": left_ans, "right": right_ans}
        cell_detail["answer_fid_refuse"] = {
            "left": refuse_answer_derived_fid(
                None if left_inc is None else left_inc.fid, left_ans
            ),
            "right": refuse_answer_derived_fid(
                None if right_inc is None else right_inc.fid, right_ans
            ),
        }
        detail[cid] = cell_detail

    # Any apparatus_error in table?
    for cand, row in table.items():
        for cid, outcome in row.items():
            if outcome == OUTCOME_APPARATUS_ERROR:
                return {
                    "ok": False,
                    "why": f"apparatus_error on {cid}/{cand}",
                    "earned_next": False,
                    "ex0s_under_test": "0.0.003",
                    "table": table,
                    "detail": detail,
                }

    least = least_sufficient_candidate(table)
    summary = {
        "ok": True,
        "why": why,
        "version": "TM.0.12.MINIMAP",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": seed,
        "table": table,
        "detail": detail,
        "least_sufficient_candidate": least,
        "claim": _claim_text(least),
        "victory_contrasts": list(VICTORY_CONTRASTS),
        "c7_classification": "UNOBSERVABLE_FROM_PROVENANCE",
        "lock": snap,
    }
    return summary


def _serialize_state(state: tuple[Any, ...]) -> list[Any]:
    out: list[Any] = []
    for x in state:
        if isinstance(x, tuple):
            out.append(list(x))
        else:
            out.append(x)
    return out


def least_sufficient_candidate(table: dict[str, dict[str, str]]) -> str | None:
    """Least-structured preregistered candidate that distinguishes all C2–C6.

    H4 is diagnostic and excluded from sufficiency ranking.
    """
    ranked = sorted(
        (c for c in CANDIDATE_ORDER if c != "H4"),
        key=lambda c: STRUCTURE_RANK[c],
    )
    for cand in ranked:
        outcomes = [table[cand][v] for v in VICTORY_CONTRASTS]
        if all(o == OUTCOME_DISTINGUISHES for o in outcomes):
            return cand
    return None


def _claim_text(least: str | None) -> str:
    if least == "H3a":
        return (
            "Among the preregistered candidate set, (token, origin) is the "
            "least-structured sufficient candidate representation for locked "
            "contrasts C2–C6. Full path is not required by this battery. "
            "C7 is UNOBSERVABLE_FROM_PROVENANCE and excluded from the CONTEXT victory set."
        )
    if least is None:
        return (
            "No preregistered structured candidate distinguished all C2–C6. "
            "C7 remains UNOBSERVABLE_FROM_PROVENANCE."
        )
    name = candidate_definitions()[least]["name"]
    return (
        f"Among the preregistered candidate set, {name} is the "
        "least-structured sufficient candidate representation for locked "
        "contrasts C2–C6. C7 is UNOBSERVABLE_FROM_PROVENANCE."
    )


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012minimap"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifacts(summary: dict[str, Any]) -> Path:
    run_dir = _run_dir()
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    table = summary.get("table") or {}
    header = "| Candidate | " + " | ".join(["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]) + " |"
    sep = "|-----------|" + "|".join(["------"] * 8) + "|"
    lines = [header, sep]
    names = {
        "H0": "H0 token",
        "H1": "H1 + here",
        "H2": "H2 + pred",
        "H3a": "H3a + origin",
        "H3b": "H3b + path",
        "H4": "H4 + incoming fid",
    }
    short = {
        OUTCOME_DISTINGUISHES: "D",
        OUTCOME_COLLISION: "collision",
        OUTCOME_BENIGN: "benign",
        OUTCOME_UNOBSERVABLE: "unobservable",
    }
    for cand in CANDIDATE_ORDER:
        row = table.get(cand, {})
        cells = [short.get(row.get(c, "?"), row.get(c, "?")) for c in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")]
        lines.append(f"| {names[cand]} | " + " | ".join(cells) + " |")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.MINIMAP · representation distinguishability

Apparatus: {summary.get('why')}
`earned_next`: false (no Ex0S 0.0.004)
Least-structured sufficient candidate (C2–C6): **{summary.get('least_sufficient_candidate')}**

{summary.get('claim')}

{chr(10).join(lines)}
""",
        encoding="utf-8",
    )
    summary["run_dir"] = str(run_dir)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.MINIMAP representation distinguishability")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_minimap_lock(), indent=2))
        return
    summary = run_minimap(seed=args.seed)
    if summary.get("ok"):
        write_run_artifacts(summary)
    print(
        json.dumps(
            {
                "ok": summary.get("ok"),
                "why": summary.get("why"),
                "least_sufficient_candidate": summary.get("least_sufficient_candidate"),
                "earned_next": summary.get("earned_next"),
                "table": summary.get("table"),
                "claim": summary.get("claim"),
                "run_dir": summary.get("run_dir"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
