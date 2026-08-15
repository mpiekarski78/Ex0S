"""TM.0.12.MIDPATH: same-S endpoint provenance vs interior route.

One shared S; same origin X, predecessor P, frontier Y; two length-4
apparatus traces that differ in exactly one interior token. No
path_and_frontier discovery. Scorer sees projected states + motors only —
raw TraceSpec is extractor-only ground truth. No agent/probe/genome.
No Ex0S 0.0.004.

If H3c collides and H3b distinguishes: endpoint provenance is insufficient.
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
    MINIMAP_LOCK,
    OUTCOME_APPARATUS_ERROR,
    OUTCOME_COLLISION,
    OUTCOME_DISTINGUISHES,
    OUTCOME_INADMISSIBLE,
    refuse_answer_derived_fid,
    score_contrast,
)
from experiments.run_tm012pathdisc import (  # noqa: E402
    PATHDISC_LOCK,
    TraceSpec,
    find_edge,
    relations_content_hash,
)

MIDPATH_LOCK = REPO_ROOT / "docs" / "midpath_012.lock"

ROLE_MIDPATH_VS_ENDPOINT = "midpath_vs_endpoint"
CELL_ID = "c9_same_endpoint_midpath"
HERE = "chb"
SUPPORT = (1, 0)
TRACE_LEN = 4

# H3c is local to MIDPATH — do not modify MINIMAP CANDIDATE_ORDER.
MIDPATH_CANDIDATES = ("H0", "H1", "H2", "H3a", "H3c", "H3b", "H4")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


@dataclass
class MidPathCell:
    cell_id: str
    relations: list[Rel]
    origin: str
    predecessor: str
    frontier: str
    traces: list[TraceSpec]
    annotation: str = ""


def extract_states_midpath(rels: list[Rel], trace: TraceSpec) -> dict[str, tuple[Any, ...]]:
    """Project apparatus trace to H0–H4 + H3c. Does not call path discovery."""
    nodes = [n.lower() for n in trace.nodes]
    if len(nodes) != TRACE_LEN:
        raise ValueError(f"C9 trace must have length {TRACE_LEN}")
    origin = nodes[0]
    predecessor = nodes[-2]
    frontier = nodes[-1]
    incoming = find_edge(rels, predecessor, frontier)
    if incoming is None:
        raise ValueError(f"no incoming edge {predecessor}→{frontier}")
    return {
        "H0": (frontier,),
        "H1": (frontier, incoming.here),
        "H2": (frontier, incoming.bind.lower()),
        "H3a": (frontier, origin),
        "H3c": (frontier, origin, predecessor),
        "H3b": (frontier, tuple(nodes)),
        "H4": (frontier, incoming.fid),
    }


def validate_c9_pair(
    rels: list[Rel],
    route_a: TraceSpec,
    route_b: TraceSpec,
    *,
    origin: str,
    predecessor: str,
    frontier: str,
) -> list[str]:
    """Strict C9 geometry + constant here/evidence on every relation."""
    errs: list[str] = []
    motors = {m.lower() for m in MOTORS}
    ox, pred, fr = origin.lower(), predecessor.lower(), frontier.lower()
    na = [n.lower() for n in route_a.nodes]
    nb = [n.lower() for n in route_b.nodes]

    for label, nodes in (("route_a", na), ("route_b", nb)):
        if len(nodes) != TRACE_LEN:
            errs.append(f"{label} length {len(nodes)} != {TRACE_LEN}")
            continue
        if nodes[0] != ox:
            errs.append(f"{label} origin {nodes[0]!r} != {ox!r}")
        if nodes[2] != pred:
            errs.append(f"{label} predecessor {nodes[2]!r} != {pred!r}")
        if nodes[3] != fr:
            errs.append(f"{label} frontier {nodes[3]!r} != {fr!r}")
        for n in nodes:
            if n in motors:
                errs.append(f"{label} path node is motor: {n}")
        for a, b in zip(nodes, nodes[1:]):
            if find_edge(rels, a, b) is None:
                errs.append(f"{label} missing hop {a}→{b}")

    if len(na) == TRACE_LEN and len(nb) == TRACE_LEN:
        if na[0] != nb[0]:
            errs.append("origins must match")
        if na[1] == nb[1]:
            errs.append("interior token A[1] must differ from B[1]")
        if na[2] != nb[2]:
            errs.append("predecessors must match")
        if na[3] != nb[3]:
            errs.append("frontiers must match")

    for trace in (route_a, route_b):
        m = trace.required_motor.lower()
        if m not in motors:
            errs.append(f"required motor not in MOTORS: {trace.required_motor!r}")
        elif find_edge(rels, frontier, m) is None:
            errs.append(f"missing outgoing Y→motor hop {fr}→{m}")
    if route_a.required_motor.lower() == route_b.required_motor.lower():
        errs.append("required motors must differ")

    # Exactly one shared P→Y edge (H4 must collide on identity, not luck).
    py = [
        r
        for r in rels
        if r.bind.lower() == pred and r.did.lower() == fr
    ]
    if len(py) != 1:
        errs.append(f"expected exactly one P→Y edge, found {len(py)}")

    # Every relation in S: identical here + support (incl. Y→motor).
    for r in rels:
        if r.here != HERE:
            errs.append(f"relation here drift {r.fid}: {r.here!r} != {HERE!r}")
        if r.init != SUPPORT:
            errs.append(f"relation evidence drift {r.fid}: {r.init!r}")

    return errs


def gen_c9(seed: int = DEFAULT_SEED) -> MidPathCell:
    """One shared fork-join S; two length-4 traces differing at interior only."""
    rng = np.random.default_rng(seed + 901)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    a = _nonce(rng, taken_n)
    b = _nonce(rng, taken_n)
    p = _nonce(rng, taken_n)
    y = _nonce(rng, taken_n)
    m_a, m_b = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), a, p, "ap", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), x, b, "xb", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), b, p, "bp", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), p, y, "py", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), y, m_a, "ym_a", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), y, m_b, "ym_b", SUPPORT, here=HERE),
    ]
    return MidPathCell(
        cell_id=CELL_ID,
        relations=rels,
        origin=x,
        predecessor=p,
        frontier=y,
        traces=[
            TraceSpec("route_a", (x, a, p, y), m_a),
            TraceSpec("route_b", (x, b, p, y), m_b),
        ],
        annotation=(
            "same S; same X/P/Y; length-4; one interior token differs; "
            "H3c must collide; H3b must distinguish; raw trace extractor-only"
        ),
    )


def locked_contrast() -> dict[str, Any]:
    return {
        "C9": {
            "cell_id": CELL_ID,
            "left": {"trace_label": "route_a"},
            "right": {"trace_label": "route_b"},
            "role": ROLE_MIDPATH_VS_ENDPOINT,
            "geometry": {
                "len": TRACE_LEN,
                "A": "X → A → P → Y",
                "B": "X → B → P → Y",
                "differs_at": 1,
            },
            "note": (
                "same S; apparatus traces; raw apparatus trace is extractor-only "
                "ground truth; unavailable after projection; not path discovery"
            ),
        }
    }


def apparatus_snapshot() -> dict[str, Any]:
    if not CONTEXT_LOCK.exists():
        raise FileNotFoundError("docs/context_012.lock missing")
    if not MINIMAP_LOCK.exists():
        raise FileNotFoundError("docs/minimap_012.lock missing")
    if not PATHDISC_LOCK.exists():
        raise FileNotFoundError("docs/pathdisc_012.lock missing")
    return {
        "version": "TM.0.12.MIDPATH",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": DEFAULT_SEED,
        "context_012_lock_sha": _sha_file(CONTEXT_LOCK),
        "minimap_012_lock_sha": _sha_file(MINIMAP_LOCK),
        "pathdisc_012_lock_sha": _sha_file(PATHDISC_LOCK),
        "contrast": locked_contrast(),
        "cell_id": CELL_ID,
        "role": ROLE_MIDPATH_VS_ENDPOINT,
        "candidates": list(MIDPATH_CANDIDATES),
        "here": HERE,
        "support": list(SUPPORT),
        "trace_len": TRACE_LEN,
        "validation": [
            "len(A)==len(B)==4",
            "A[0]==B[0]==X",
            "A[1]!=B[1]",
            "A[2]==B[2]==P",
            "A[3]==B[3]==Y",
            "all path and motor edges share identical here",
            "all relations have identical support (1,0)",
            "required motors differ and exist as Y→motor",
            "one shared S; both traces' path+motor fids ⊆ store",
            "runtime seed equals locked seed",
            "raw apparatus trace is extractor-only ground truth; unavailable after projection",
            "extract_states_midpath must not call path_and_frontier",
            "outgoing Y→motor fid inadmissible",
            "H4 incoming is shared P→Y",
            "exactly one P→Y edge in S",
        ],
        "gen_c9_sha": _sha_src(gen_c9),
        "validate_c9_sha": _sha_src(validate_c9_pair),
        "midpath_extractor_sha": _sha_src(extract_states_midpath),
        "score_contrast_sha": _sha_src(score_contrast),
        "find_edge_sha": _sha_src(find_edge),
        "relations_content_hash_sha": _sha_src(relations_content_hash),
        "refuse_answer_derived_fid_sha": _sha_src(refuse_answer_derived_fid),
        "trace_spec_sha": _sha_src(TraceSpec),
        "scorer_reuse": "experiments.run_tm012minimap.score_contrast",
        "refuse": [
            "two-store C9",
            "path_and_frontier route discovery",
            "raw TraceSpec / nodes as selector input",
            "rewrite context_012.lock / minimap_012.lock / pathdisc_012.lock",
            "modify MINIMAP CANDIDATE_ORDER or extractor for H3c",
            "declare full path as product",
            "ceremonial H3c-only scorer",
            "implement route-signature product this pass",
            "stamp Ex0S 0.0.004",
            "genome / agent change",
        ],
    }


def write_midpath_lock(path: Path = MIDPATH_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_midpath_lock(path: Path = MIDPATH_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not path.exists():
        return False, "docs/midpath_012.lock missing; write only via --write-lock", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key, label in (
        ("context_012_lock_sha", "context_012.lock"),
        ("minimap_012_lock_sha", "minimap_012.lock"),
        ("pathdisc_012_lock_sha", "pathdisc_012.lock"),
    ):
        if snap[key] != lock.get(key):
            return False, f"{label} SHA drifted from midpath pin", snap
    if _sha_file(CONTEXT_LOCK) != lock.get("context_012_lock_sha"):
        return False, "context_012.lock content SHA mismatch", snap
    if _sha_file(MINIMAP_LOCK) != lock.get("minimap_012_lock_sha"):
        return False, "minimap_012.lock content SHA mismatch", snap
    if _sha_file(PATHDISC_LOCK) != lock.get("pathdisc_012_lock_sha"):
        return False, "pathdisc_012.lock content SHA mismatch", snap
    for key in (
        "seed",
        "contrast",
        "cell_id",
        "role",
        "candidates",
        "here",
        "support",
        "trace_len",
        "validation",
        "gen_c9_sha",
        "validate_c9_sha",
        "midpath_extractor_sha",
        "score_contrast_sha",
        "find_edge_sha",
        "relations_content_hash_sha",
        "refuse_answer_derived_fid_sha",
        "trace_spec_sha",
        "refuse",
    ):
        if snap[key] != lock.get(key):
            return False, f"midpath apparatus drift: {key}", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if _sha_src(score_contrast) != lock.get("score_contrast_sha"):
        return False, "score_contrast SHA drifted from midpath pin", snap
    if _sha_src(TraceSpec) != lock.get("trace_spec_sha"):
        return False, "TraceSpec SHA drifted from midpath pin", snap
    src = inspect.getsource(extract_states_midpath)
    if "path_and_frontier" in src:
        return False, "extract_states_midpath must not call path_and_frontier", snap
    return True, "midpath apparatus frozen", snap


def run_midpath(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok, why, snap = verify_midpath_lock()
    if not ok:
        return {
            "ok": False,
            "why": why,
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }
    if seed != snap["seed"]:
        return {
            "ok": False,
            "why": f"runtime seed {seed} != locked seed {snap['seed']}",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    cell = gen_c9(seed)
    route_a, route_b = cell.traces[0], cell.traces[1]
    verrs = validate_c9_pair(
        cell.relations,
        route_a,
        route_b,
        origin=cell.origin,
        predecessor=cell.predecessor,
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

    # Projection then scorer — never pass TraceSpec / nodes into score_contrast.
    sa = extract_states_midpath(cell.relations, route_a)
    sb = extract_states_midpath(cell.relations, route_b)

    table: dict[str, dict[str, str]] = {c: {} for c in MIDPATH_CANDIDATES}
    outcomes: dict[str, str] = {}
    for cand in MIDPATH_CANDIDATES:
        out = score_contrast(
            contrast_id="C9",
            role=ROLE_MIDPATH_VS_ENDPOINT,
            left_states=sa,
            right_states=sb,
            left_motor=route_a.required_motor,
            right_motor=route_b.required_motor,
            candidate=cand,
        )
        table[cand]["C9"] = out
        outcomes[cand] = out
        if out == OUTCOME_APPARATUS_ERROR:
            return {
                "ok": False,
                "why": f"apparatus_error on C9/{cand}",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

    # Decisive: H3c must collide; H3b must distinguish.
    if outcomes["H3c"] == OUTCOME_DISTINGUISHES:
        return {
            "ok": False,
            "why": "apparatus_error: H3c distinguished (X/P/Y or traces drifted)",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }
    if outcomes["H3c"] != OUTCOME_COLLISION:
        return {
            "ok": False,
            "why": f"apparatus_error: H3c expected collision got {outcomes['H3c']}",
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
    for must_collide in ("H0", "H1", "H2", "H3a", "H4"):
        if outcomes[must_collide] != OUTCOME_COLLISION:
            return {
                "ok": False,
                "why": (
                    f"apparatus_error: {must_collide} expected collision "
                    f"got {outcomes[must_collide]}"
                ),
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

    out_a = find_edge(cell.relations, cell.frontier, route_a.required_motor)
    out_b = find_edge(cell.relations, cell.frontier, route_b.required_motor)
    inc = find_edge(cell.relations, cell.predecessor, cell.frontier)
    refuse_a = refuse_answer_derived_fid(
        None if inc is None else inc.fid,
        None if out_a is None else out_a.fid,
    )
    refuse_b = refuse_answer_derived_fid(
        None if inc is None else inc.fid,
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
        "Endpoint provenance is insufficient. Some information from the "
        "traversed route interior is necessary. H3b survives as a candidate "
        "only — do not conclude store-the-full-path. C4 kills predecessor "
        "alone; C8 kills origin alone; C9 kills origin+predecessor. Next "
        "(not this pass): route-signature minimality — which interior "
        "information can be deleted without required-output collisions."
    )
    return {
        "ok": True,
        "why": why,
        "version": "TM.0.12.MIDPATH",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": seed,
        "table": table,
        "outcomes": outcomes,
        "claim": claim,
        "endpoint_provenance_insufficient": True,
        "h3b_survives": outcomes["H3b"] == OUTCOME_DISTINGUISHES,
        "s_hash": s_hash,
        "origin": cell.origin,
        "predecessor": cell.predecessor,
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
        "scorer_inputs": "projected_states_and_motors_only",
        "lock": snap,
    }


def _ser(state: tuple[Any, ...]) -> list[Any]:
    out: list[Any] = []
    for x in state:
        out.append(list(x) if isinstance(x, tuple) else x)
    return out


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012midpath"
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
        "H3c": "H3c + origin+pred",
        "H3b": "H3b + path",
        "H4": "H4 + incoming fid",
    }
    short = {
        OUTCOME_DISTINGUISHES: "D",
        OUTCOME_COLLISION: "collision",
        OUTCOME_APPARATUS_ERROR: "apparatus_error",
    }
    lines = ["| Candidate | C9 |", "|-----------|----|"]
    for cand in MIDPATH_CANDIDATES:
        o = (table.get(cand) or {}).get("C9", "?")
        lines.append(f"| {names[cand]} | {short.get(o, o)} |")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.MIDPATH · endpoint provenance vs interior

Apparatus: {summary.get('why')}
`earned_next`: false (no Ex0S 0.0.004)
Endpoint provenance insufficient: **{summary.get('endpoint_provenance_insufficient')}**
H3b survives: **{summary.get('h3b_survives')}**

{summary.get('claim')}

{chr(10).join(lines)}
""",
        encoding="utf-8",
    )
    summary["run_dir"] = str(run_dir)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.MIDPATH endpoint vs interior")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_midpath_lock(), indent=2))
        return
    summary = run_midpath(seed=args.seed)
    if summary.get("ok"):
        write_run_artifacts(summary)
    print(
        json.dumps(
            {
                "ok": summary.get("ok"),
                "why": summary.get("why"),
                "earned_next": summary.get("earned_next"),
                "endpoint_provenance_insufficient": summary.get(
                    "endpoint_provenance_insufficient"
                ),
                "h3b_survives": summary.get("h3b_survives"),
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
