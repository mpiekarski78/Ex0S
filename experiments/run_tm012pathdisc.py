"""TM.0.12.PATHDISC: same-S origin vs path discriminator.

One shared S, two apparatus-level observed traces. No path_and_frontier
discovery. No agent/probe/genome. No Ex0S 0.0.004.

If H3a collides and H3b distinguishes: origin alone is insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import (  # noqa: E402
    CONTEXT_LOCK,
    DEFAULT_SEED,
    MOTORS,
    Rel,
    _fid,
    _nonce,
    _two_motors,
)
from experiments.run_tm012minimap import (  # noqa: E402
    CANDIDATE_ORDER,
    MINIMAP_LOCK,
    OUTCOME_APPARATUS_ERROR,
    OUTCOME_COLLISION,
    OUTCOME_DISTINGUISHES,
    OUTCOME_INADMISSIBLE,
    refuse_answer_derived_fid,
    score_contrast,
)

PATHDISC_LOCK = REPO_ROOT / "docs" / "pathdisc_012.lock"

ROLE_ORIGIN_VS_PATH = "origin_vs_path"
CELL_ID = "c8_same_origin_path"
HERE = "chb"
SUPPORT = (1, 0)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


@dataclass
class TraceSpec:
    label: str
    nodes: tuple[str, ...]
    required_motor: str


@dataclass
class PathDiscCell:
    cell_id: str
    relations: list[Rel]
    origin: str
    frontier: str
    traces: list[TraceSpec]
    annotation: str = ""


def relations_content_hash(rels: list[Rel]) -> str:
    """Stable hash of shared S (order-independent)."""
    rows = sorted(
        (r.fid, r.bind.lower(), r.did.lower(), r.here, int(r.init[0]), int(r.init[1]), r.role)
        for r in rels
    )
    return _sha_bytes(json.dumps(rows, separators=(",", ":")).encode())


def find_edge(rels: list[Rel], bind: str, did: str) -> Rel | None:
    b, d = bind.lower(), did.lower()
    for r in rels:
        if r.bind.lower() == b and r.did.lower() == d:
            return r
    return None


def validate_trace(rels: list[Rel], trace: TraceSpec, *, origin: str, frontier: str) -> list[str]:
    """Return list of validation errors (empty ⇒ ok)."""
    errs: list[str] = []
    motors = {m.lower() for m in MOTORS}
    nodes = [n.lower() for n in trace.nodes]
    if len(nodes) < 2:
        errs.append("trace too short")
        return errs
    if nodes[0] != origin.lower():
        errs.append(f"trace origin {nodes[0]!r} != locked {origin.lower()!r}")
    if nodes[-1] != frontier.lower():
        errs.append(f"trace frontier {nodes[-1]!r} != locked {frontier.lower()!r}")
    for n in nodes:
        if n in motors:
            errs.append(f"path node is motor: {n}")
    for a, b in zip(nodes, nodes[1:]):
        if find_edge(rels, a, b) is None:
            errs.append(f"missing hop {a}→{b}")
    return errs


def validate_c8_pair(
    rels: list[Rel],
    route_a: TraceSpec,
    route_b: TraceSpec,
    *,
    origin: str,
    frontier: str,
) -> list[str]:
    errs: list[str] = []
    errs.extend(validate_trace(rels, route_a, origin=origin, frontier=frontier))
    errs.extend(validate_trace(rels, route_b, origin=origin, frontier=frontier))
    motors = {m.lower() for m in MOTORS}
    for trace in (route_a, route_b):
        m = trace.required_motor.lower()
        if m not in motors:
            errs.append(f"required motor not in MOTORS: {trace.required_motor!r}")
        elif find_edge(rels, frontier, m) is None:
            errs.append(f"missing outgoing Y→motor hop {frontier.lower()}→{m}")
    if route_a.required_motor.lower() == route_b.required_motor.lower():
        errs.append("required motors must differ")
    na = [n.lower() for n in route_a.nodes]
    nb = [n.lower() for n in route_b.nodes]
    if na[1:-1] == nb[1:-1]:
        errs.append("internal nodes must differ")
    # Same incoming here on last hop into Y.
    ea = find_edge(rels, na[-2], na[-1]) if len(na) >= 2 else None
    eb = find_edge(rels, nb[-2], nb[-1]) if len(nb) >= 2 else None
    if ea is None or eb is None:
        errs.append("incoming edges missing")
    else:
        if ea.here != eb.here:
            errs.append("incoming here must match (H1 cannot help)")
        if ea.init != eb.init:
            errs.append("incoming evidence must match")
    # Equal evidence on all path hops (locked support).
    for nodes in (na, nb):
        for a, b in zip(nodes, nodes[1:]):
            e = find_edge(rels, a, b)
            if e is not None and e.init != SUPPORT:
                errs.append(f"unequal path evidence on {a}→{b}")
    return errs


def extract_states_from_trace(rels: list[Rel], trace: TraceSpec) -> dict[str, tuple[Any, ...]]:
    """Project an apparatus-observed trace to H0–H4 from supplied nodes only."""
    nodes = [n.lower() for n in trace.nodes]
    if len(nodes) < 2:
        raise ValueError("trace too short")
    origin = nodes[0]
    frontier = nodes[-1]
    incoming = find_edge(rels, nodes[-2], nodes[-1])
    if incoming is None:
        raise ValueError(f"no incoming edge {nodes[-2]}→{nodes[-1]}")
    return {
        "H0": (frontier,),
        "H1": (frontier, incoming.here),
        "H2": (frontier, incoming.bind.lower()),
        "H3a": (frontier, origin),
        "H3b": (frontier, tuple(nodes)),
        "H4": (frontier, incoming.fid),
    }


def gen_c8(seed: int = DEFAULT_SEED) -> PathDiscCell:
    """One shared diamond S; two observed traces with same origin/frontier."""
    rng = np.random.default_rng(seed + 801)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    y = _nonce(rng, taken_n)
    a, b = _nonce(rng, taken_n), _nonce(rng, taken_n)
    c, d = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m_a, m_b = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), a, b, "ab", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), b, y, "by", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), x, c, "xc", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), c, d, "cd", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), d, y, "dy", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), y, m_a, "ym_a", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), y, m_b, "ym_b", SUPPORT, here=HERE),
    ]
    return PathDiscCell(
        cell_id=CELL_ID,
        relations=rels,
        origin=x,
        frontier=y,
        traces=[
            TraceSpec("route_a", (x, a, b, y), m_a),
            TraceSpec("route_b", (x, c, d, y), m_b),
        ],
        annotation=(
            "same S; same origin X and frontier Y; two observed traces; "
            "H3a must collide; H3b must distinguish"
        ),
    )


def locked_contrast() -> dict[str, Any]:
    return {
        "C8": {
            "cell_id": CELL_ID,
            "left": {"trace_label": "route_a"},
            "right": {"trace_label": "route_b"},
            "role": ROLE_ORIGIN_VS_PATH,
            "note": "same S; apparatus traces; not two stores; not path_and_frontier discovery",
        }
    }


def apparatus_snapshot() -> dict[str, Any]:
    if not CONTEXT_LOCK.exists():
        raise FileNotFoundError("docs/context_012.lock missing")
    if not MINIMAP_LOCK.exists():
        raise FileNotFoundError("docs/minimap_012.lock missing")
    return {
        "version": "TM.0.12.PATHDISC",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": DEFAULT_SEED,
        "context_012_lock_sha": _sha_file(CONTEXT_LOCK),
        "minimap_012_lock_sha": _sha_file(MINIMAP_LOCK),
        "contrast": locked_contrast(),
        "cell_id": CELL_ID,
        "role": ROLE_ORIGIN_VS_PATH,
        "here": HERE,
        "support": list(SUPPORT),
        "validation": [
            "consecutive hops exist in S",
            "traces start at locked X end at locked Y",
            "same origin and frontier",
            "internal nodes differ",
            "path nodes are non-motors",
            "incoming here equal",
            "path evidence equal",
            "required motors differ",
            "required motors in MOTORS with Y→motor edges in S",
            "one shared S content hash for both traces",
            "extract_states_from_trace must not call path_and_frontier",
            "outgoing Y→motor fid inadmissible",
        ],
        "gen_c8_sha": _sha_src(gen_c8),
        "extract_from_trace_sha": _sha_src(extract_states_from_trace),
        "validate_sha": _sha_src(validate_c8_pair),
        "scorer_sha": _sha_src(score_contrast),
        "scorer_reuse": "experiments.run_tm012minimap.score_contrast",
        "refuse": [
            "two-store C8",
            "path_and_frontier route discovery",
            "rewrite context_012.lock or minimap_012.lock",
            "declare full path as product",
            "implement H3c this pass",
            "stamp Ex0S 0.0.004",
            "genome / agent change",
        ],
    }


def write_pathdisc_lock(path: Path = PATHDISC_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_pathdisc_lock(path: Path = PATHDISC_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not path.exists():
        return False, "docs/pathdisc_012.lock missing; write only via --write-lock", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    if snap["context_012_lock_sha"] != lock.get("context_012_lock_sha"):
        return False, "context_012.lock SHA drifted from pathdisc pin", snap
    if snap["minimap_012_lock_sha"] != lock.get("minimap_012_lock_sha"):
        return False, "minimap_012.lock SHA drifted from pathdisc pin", snap
    if _sha_file(CONTEXT_LOCK) != lock.get("context_012_lock_sha"):
        return False, "context_012.lock content SHA mismatch", snap
    if _sha_file(MINIMAP_LOCK) != lock.get("minimap_012_lock_sha"):
        return False, "minimap_012.lock content SHA mismatch", snap
    for key in (
        "seed",
        "contrast",
        "cell_id",
        "role",
        "here",
        "support",
        "validation",
        "gen_c8_sha",
        "extract_from_trace_sha",
        "validate_sha",
        "scorer_sha",
        "refuse",
    ):
        if snap[key] != lock.get(key):
            return False, f"pathdisc apparatus drift: {key}", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if _sha_src(score_contrast) != lock.get("scorer_sha"):
        return False, "score_contrast SHA drifted from pathdisc pin", snap
    # extract_states_from_trace source must not call path_and_frontier.
    src = inspect.getsource(extract_states_from_trace)
    if "path_and_frontier" in src:
        return False, "extract_states_from_trace must not call path_and_frontier", snap
    return True, "pathdisc apparatus frozen", snap


def run_pathdisc(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok, why, snap = verify_pathdisc_lock()
    if not ok:
        return {
            "ok": False,
            "why": why,
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    cell = gen_c8(seed)
    route_a, route_b = cell.traces[0], cell.traces[1]
    verrs = validate_c8_pair(
        cell.relations,
        route_a,
        route_b,
        origin=cell.origin,
        frontier=cell.frontier,
    )
    if verrs:
        return {
            "ok": False,
            "why": "validation failed: " + "; ".join(verrs),
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    # One shared S: both traces' path + Y→motor edges must be subsets of the same store.
    s_hash = relations_content_hash(cell.relations)
    store_fids = {r.fid for r in cell.relations}

    def _trace_fids(trace: TraceSpec) -> set[str]:
        fids: set[str] = set()
        nodes = [n.lower() for n in trace.nodes]
        for a, b in zip(nodes, nodes[1:]):
            e = find_edge(cell.relations, a, b)
            if e is not None:
                fids.add(e.fid)
        m = find_edge(cell.relations, cell.frontier, trace.required_motor)
        if m is not None:
            fids.add(m.fid)
        return fids

    fa, fb = _trace_fids(route_a), _trace_fids(route_b)
    if not fa or not fb or not fa.issubset(store_fids) or not fb.issubset(store_fids):
        return {
            "ok": False,
            "why": "trace edges not contained in the single shared S",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    sa = extract_states_from_trace(cell.relations, route_a)
    sb = extract_states_from_trace(cell.relations, route_b)

    table: dict[str, dict[str, str]] = {c: {} for c in CANDIDATE_ORDER}
    outcomes: dict[str, str] = {}
    for cand in CANDIDATE_ORDER:
        out = score_contrast(
            contrast_id="C8",
            role=ROLE_ORIGIN_VS_PATH,
            left_states=sa,
            right_states=sb,
            left_motor=route_a.required_motor,
            right_motor=route_b.required_motor,
            candidate=cand,
        )
        table[cand]["C8"] = out
        outcomes[cand] = out
        if out == OUTCOME_APPARATUS_ERROR:
            return {
                "ok": False,
                "why": f"apparatus_error on C8/{cand}",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

    # Decisive: H3a must collide; H3b must distinguish. Else apparatus error.
    if outcomes["H3a"] == OUTCOME_DISTINGUISHES:
        return {
            "ok": False,
            "why": "apparatus_error: H3a distinguished (X/Y or traces drifted)",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }
    if outcomes["H3a"] != OUTCOME_COLLISION:
        return {
            "ok": False,
            "why": f"apparatus_error: H3a expected collision got {outcomes['H3a']}",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }
    if outcomes["H3b"] != OUTCOME_DISTINGUISHES:
        return {
            "ok": False,
            "why": f"apparatus_error: H3b expected distinguishes got {outcomes['H3b']}",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }

    # Outgoing Y→motor fids are answer-derived and inadmissible (direct lookup on
    # the shared S; do not infer a traversed route from branching structure).
    out_a = find_edge(cell.relations, cell.frontier, route_a.required_motor)
    out_b = find_edge(cell.relations, cell.frontier, route_b.required_motor)
    inc_a = find_edge(cell.relations, route_a.nodes[-2], route_a.nodes[-1])
    inc_b = find_edge(cell.relations, route_b.nodes[-2], route_b.nodes[-1])
    refuse_a = refuse_answer_derived_fid(
        None if inc_a is None else inc_a.fid,
        None if out_a is None else out_a.fid,
    )
    refuse_b = refuse_answer_derived_fid(
        None if inc_b is None else inc_b.fid,
        None if out_b is None else out_b.fid,
    )
    if refuse_a != OUTCOME_INADMISSIBLE or refuse_b != OUTCOME_INADMISSIBLE:
        return {
            "ok": False,
            "why": f"outgoing motor fid refuse failed: {refuse_a}/{refuse_b}",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }

    claim = (
        "Origin alone is insufficient. Some information about the traversed "
        "route beyond origin is necessary. H3b survives as a candidate only; "
        "H2 also distinguishes — do not conclude store-the-full-path. "
        "C4 kills predecessor alone; C8 kills origin alone. Next candidate "
        "question (not this pass): H3c=(token, origin, predecessor)."
    )
    return {
        "ok": True,
        "why": why,
        "version": "TM.0.12.PATHDISC",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": seed,
        "table": table,
        "outcomes": outcomes,
        "claim": claim,
        "origin_insufficient": True,
        "h2_survives": outcomes["H2"] == OUTCOME_DISTINGUISHES,
        "h3b_survives": outcomes["H3b"] == OUTCOME_DISTINGUISHES,
        "s_hash": s_hash,
        "origin": cell.origin,
        "frontier": cell.frontier,
        "traces": {
            "route_a": {
                "nodes": list(route_a.nodes),
                "motor": route_a.required_motor,
                "states": {k: list(_ser(v)) for k, v in sa.items()},
            },
            "route_b": {
                "nodes": list(route_b.nodes),
                "motor": route_b.required_motor,
                "states": {k: list(_ser(v)) for k, v in sb.items()},
            },
        },
        "answer_fid_refuse": {"route_a": refuse_a, "route_b": refuse_b},
        "lock": snap,
    }


def _ser(state: tuple[Any, ...]) -> list[Any]:
    out: list[Any] = []
    for x in state:
        out.append(list(x) if isinstance(x, tuple) else x)
    return out


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012pathdisc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifacts(summary: dict[str, Any]) -> Path:
    run_dir = _run_dir()
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    (run_dir / "metrics.json").write_text(
        json.dumps(clean, indent=2, default=str) + "\n", encoding="utf-8"
    )
    table = summary.get("table") or {}
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
        OUTCOME_APPARATUS_ERROR: "apparatus_error",
    }
    lines = ["| Candidate | C8 |", "|-----------|----|"]
    for cand in CANDIDATE_ORDER:
        o = (table.get(cand) or {}).get("C8", "?")
        lines.append(f"| {names[cand]} | {short.get(o, o)} |")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.PATHDISC · same-S origin vs path

Apparatus: {summary.get('why')}
`earned_next`: false (no Ex0S 0.0.004)
Origin insufficient: **{summary.get('origin_insufficient')}**
H2 survives: **{summary.get('h2_survives')}**

{summary.get('claim')}

{chr(10).join(lines)}
""",
        encoding="utf-8",
    )
    summary["run_dir"] = str(run_dir)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.PATHDISC same-S origin vs path")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_pathdisc_lock(), indent=2))
        return
    summary = run_pathdisc(seed=args.seed)
    if summary.get("ok"):
        write_run_artifacts(summary)
    print(
        json.dumps(
            {
                "ok": summary.get("ok"),
                "why": summary.get("why"),
                "earned_next": summary.get("earned_next"),
                "origin_insufficient": summary.get("origin_insufficient"),
                "h2_survives": summary.get("h2_survives"),
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
