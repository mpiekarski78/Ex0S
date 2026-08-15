"""TM.0.12.ROUTESIG phase 1: what must a route summary preserve?

C10 commutable loops — same unordered traversed edge-fid set, different
order. Scorer sees projected R0–R4 states + motors only. Kappa API takes
origin + ordered path fids only (never TraceSpec / motor). No genome.
No Ex0S 0.0.004. No ROUTESIG.DEPTH this pass.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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
    relations_content_hash,
)
from experiments.run_tm012midpath import MIDPATH_LOCK  # noqa: E402

ROUTESIG_LOCK = REPO_ROOT / "docs" / "routesig_012.lock"

ROLE_ORDER = "route_order"
CELL_ID = "c10_order_commutable_loops"
HERE = "chb"
SUPPORT = (1, 0)
SUFFIX_K = 2
TRACE_LEN = 8  # X Q A Q B Q P Y  (or X Q B Q A Q P Y)

ROUTESIG_CANDIDATES = ("R0", "R1", "R2", "R3", "R4")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


@dataclass
class RouteSigCell:
    cell_id: str
    relations: list[Rel]
    origin: str
    hub: str
    predecessor: str
    frontier: str
    traces: list[TraceSpec]
    annotation: str = ""


def edge_fid(rels: Sequence[Rel], bind: str, did: str) -> str:
    """Return the sole fid for (bind,did). Fail closed if not unique."""
    b, d = bind.lower(), did.lower()
    matches = [r for r in rels if r.bind.lower() == b and r.did.lower() == d]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one edge {b}→{d}, found {len(matches)}")
    return matches[0].fid


def path_edge_fids(rels: Sequence[Rel], nodes: Sequence[str]) -> tuple[str, ...]:
    """Ordered fids along observed node hops (no motor edges)."""
    ns = [n.lower() for n in nodes]
    if len(ns) < 2:
        raise ValueError("path too short")
    return tuple(edge_fid(rels, a, b) for a, b in zip(ns, ns[1:]))


def kappa_seed(origin: str) -> str:
    """κ₀ from origin token only — no answer / outgoing-edge inputs."""
    return _sha_bytes(b"origin\0" + origin.lower().encode())


def kappa_step(previous_kappa: str, traversed_fid: str) -> str:
    """κ' = F(κ, fid) — traversed relation identity only."""
    return _sha_bytes(previous_kappa.encode() + b"\0" + traversed_fid.encode())


def route_kappa(origin: str, ordered_path_fids: Sequence[str]) -> str:
    """Accumulate κ over ordered path fids. Path-fid sequence only."""
    k = kappa_seed(origin)
    for fid in ordered_path_fids:
        k = kappa_step(k, fid)
    return k


def extract_states_routesig(
    rels: Sequence[Rel], nodes: Sequence[str]
) -> dict[str, tuple[Any, ...]]:
    """Project path nodes to R0–R4. Path materialization via unique edge_fid."""
    ns = [n.lower() for n in nodes]
    if len(ns) != TRACE_LEN:
        raise ValueError(f"C10 trace must have length {TRACE_LEN}")
    origin = ns[0]
    predecessor = ns[-2]
    frontier = ns[-1]
    fids = path_edge_fids(rels, ns)
    if len(fids) < SUFFIX_K:
        raise ValueError("path shorter than suffix k")
    return {
        "R0": (frontier, origin, predecessor),
        "R1": (frontier, frozenset(fids)),
        "R2": (frontier, fids[-SUFFIX_K:]),
        "R3": (frontier, fids),
        "R4": (frontier, route_kappa(origin, fids)),
    }


def validate_c10_pair(
    rels: list[Rel],
    route_a: TraceSpec,
    route_b: TraceSpec,
    *,
    origin: str,
    hub: str,
    predecessor: str,
    frontier: str,
) -> list[str]:
    errs: list[str] = []
    motors = {m.lower() for m in MOTORS}
    ox, hub_l, pred, fr = (
        origin.lower(),
        hub.lower(),
        predecessor.lower(),
        frontier.lower(),
    )
    na = [n.lower() for n in route_a.nodes]
    nb = [n.lower() for n in route_b.nodes]

    for label, nodes in (("route_a", na), ("route_b", nb)):
        if len(nodes) != TRACE_LEN:
            errs.append(f"{label} length {len(nodes)} != {TRACE_LEN}")
            continue
        if nodes[0] != ox:
            errs.append(f"{label} origin {nodes[0]!r} != {ox!r}")
        if nodes[-2] != pred:
            errs.append(f"{label} predecessor {nodes[-2]!r} != {pred!r}")
        if nodes[-1] != fr:
            errs.append(f"{label} frontier {nodes[-1]!r} != {fr!r}")
        if nodes[1] != hub_l:
            errs.append(f"{label} first hub hop {nodes[1]!r} != {hub_l!r}")
        for n in nodes:
            if n in motors:
                errs.append(f"{label} path node is motor: {n}")
        for a, b in zip(nodes, nodes[1:]):
            try:
                edge_fid(rels, a, b)
            except ValueError as e:
                errs.append(f"{label} {e}")

    if len(na) == TRACE_LEN and len(nb) == TRACE_LEN:
        if na[0] != nb[0] or na[-1] != nb[-1] or na[-2] != nb[-2]:
            errs.append("X/P/Y must match across traces")
        if len(na) != len(nb):
            errs.append("lengths must match")
        # Same unordered node multiset.
        if sorted(na) != sorted(nb):
            errs.append("unordered node multisets must match")
        try:
            fa = path_edge_fids(rels, na)
            fb = path_edge_fids(rels, nb)
            if frozenset(fa) != frozenset(fb):
                errs.append("unordered path-edge fid sets must match")
            if sorted(fa) != sorted(fb):
                errs.append("path-edge fid multisets must match")
            if fa == fb:
                errs.append("ordered path-edge fid sequences must differ")
            if fa[:1] != fb[:1]:
                errs.append("first edge must match")
            if fa[-2:] != fb[-2:]:
                errs.append("last two edges (Q→P→Y) must match")
        except ValueError as e:
            errs.append(str(e))

    for trace in (route_a, route_b):
        m = trace.required_motor.lower()
        if m not in motors:
            errs.append(f"required motor not in MOTORS: {trace.required_motor!r}")
        else:
            try:
                edge_fid(rels, frontier, m)
            except ValueError as e:
                errs.append(f"Y→motor: {e}")
    if route_a.required_motor.lower() == route_b.required_motor.lower():
        errs.append("required motors must differ")

    # Exactly one P→Y.
    py = [r for r in rels if r.bind.lower() == pred and r.did.lower() == fr]
    if len(py) != 1:
        errs.append(f"expected exactly one P→Y edge, found {len(py)}")

    # Every (bind,did) pair in S must be unique — file-order independence of edge_fid.
    pair_counts = Counter((r.bind.lower(), r.did.lower()) for r in rels)
    for (b, d), n in sorted(pair_counts.items()):
        if n != 1:
            errs.append(f"duplicate directed edge {b}→{d} count={n}")

    for r in rels:
        if r.here != HERE:
            errs.append(f"relation here drift {r.fid}: {r.here!r} != {HERE!r}")
        if r.init != SUPPORT:
            errs.append(f"relation evidence drift {r.fid}: {r.init!r}")

    return errs


def gen_c10(seed: int = DEFAULT_SEED) -> RouteSigCell:
    """Shared S with Q↔A and Q↔B loops; two traces, same edges, different order."""
    rng = np.random.default_rng(seed + 1001)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    q = _nonce(rng, taken_n)
    a = _nonce(rng, taken_n)
    b = _nonce(rng, taken_n)
    p = _nonce(rng, taken_n)
    y = _nonce(rng, taken_n)
    m_a, m_b = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, q, "xq", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), q, a, "qa", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), a, q, "aq", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), q, b, "qb", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), b, q, "bq", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), q, p, "qp", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), p, y, "py", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), y, m_a, "ym_a", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), y, m_b, "ym_b", SUPPORT, here=HERE),
    ]
    return RouteSigCell(
        cell_id=CELL_ID,
        relations=rels,
        origin=x,
        hub=q,
        predecessor=p,
        frontier=y,
        traces=[
            TraceSpec("route_a", (x, q, a, q, b, q, p, y), m_a),
            TraceSpec("route_b", (x, q, b, q, a, q, p, y), m_b),
        ],
        annotation=(
            "same S; same edge-fid set; different order via Q-A/Q-B loops; "
            "R1 must collide; R3/R4 must distinguish; kappa path-fids only"
        ),
    )


def locked_contrast() -> dict[str, Any]:
    return {
        "C10": {
            "cell_id": CELL_ID,
            "left": {"trace_label": "route_a"},
            "right": {"trace_label": "route_b"},
            "role": ROLE_ORDER,
            "geometry": {
                "len": TRACE_LEN,
                "A": "X → Q → A → Q → B → Q → P → Y",
                "B": "X → Q → B → Q → A → Q → P → Y",
                "isolates": "order_not_membership",
            },
            "note": (
                "same S; same unordered path-edge fid set; different order; "
                "raw apparatus trace is extractor-only; unavailable after projection"
            ),
        }
    }


def apparatus_snapshot() -> dict[str, Any]:
    for path, name in (
        (CONTEXT_LOCK, "context_012.lock"),
        (MINIMAP_LOCK, "minimap_012.lock"),
        (PATHDISC_LOCK, "pathdisc_012.lock"),
        (MIDPATH_LOCK, "midpath_012.lock"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"docs/{name} missing")
    return {
        "version": "TM.0.12.ROUTESIG",
        "phase": "order",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": DEFAULT_SEED,
        "context_012_lock_sha": _sha_file(CONTEXT_LOCK),
        "minimap_012_lock_sha": _sha_file(MINIMAP_LOCK),
        "pathdisc_012_lock_sha": _sha_file(PATHDISC_LOCK),
        "midpath_012_lock_sha": _sha_file(MIDPATH_LOCK),
        "contrast": locked_contrast(),
        "cell_id": CELL_ID,
        "role": ROLE_ORDER,
        "candidates": list(ROUTESIG_CANDIDATES),
        "candidate_meanings": {
            "R0": "endpoint (Y, X, P)",
            "R1": "unordered edge membership (Y, frozenset(path_fids))",
            "R2": f"bounded suffix k={SUFFIX_K} (Y, last {SUFFIX_K} path_fids)",
            "R3": "ordered route identity (Y, ordered path_fids)",
            "R4": "rolling route signature (Y, kappa)",
        },
        "suffix_k": SUFFIX_K,
        "here": HERE,
        "support": list(SUPPORT),
        "trace_len": TRACE_LEN,
        "kappa_api": [
            "kappa_seed(origin)",
            "kappa_step(previous_kappa, traversed_fid)",
            "route_kappa(origin, ordered_path_fids)",
        ],
        "kappa_requirements": [
            "output-blind: kappa must not depend on context_expect, PRESS/TUNE, or Y→motor",
            "incremental: kappa_step(previous, traversed_fid) only",
            "raw path unavailable after projection",
            "deterministic and file-order independent",
            "no facts in genome",
            "same route yields same kappa after reload",
            "differing routes that matter must not collide under R4 on C10",
            "C7 stays unsolvable because no route difference exists there",
        ],
        "validation": [
            "commutable Q-A / Q-B loops; same unordered path-edge fid set",
            "same path-edge fid multiset",
            "ordered path-edge fid sequences differ",
            "same X/P/Y; same length; same first edge; same last two edges",
            "same unordered node multiset",
            "exactly one relation per directed (bind,did) in S (edge_fid)",
            "all relations share identical here and support (1,0)",
            "required motors differ with Y→motor edges",
            "one shared S; path+motor fids ⊆ store",
            "runtime seed equals locked seed",
            "raw apparatus trace is extractor-only ground truth; unavailable after projection",
            "route_kappa never receives TraceSpec / motor / context_expect",
            "outgoing Y→motor fid inadmissible",
            f"R2 locked at suffix k={SUFFIX_K}",
        ],
        "gen_c10_sha": _sha_src(gen_c10),
        "validate_c10_sha": _sha_src(validate_c10_pair),
        "extract_routesig_sha": _sha_src(extract_states_routesig),
        "edge_fid_sha": _sha_src(edge_fid),
        "path_edge_fids_sha": _sha_src(path_edge_fids),
        "kappa_seed_sha": _sha_src(kappa_seed),
        "kappa_step_sha": _sha_src(kappa_step),
        "route_kappa_sha": _sha_src(route_kappa),
        "score_contrast_sha": _sha_src(score_contrast),
        "relations_content_hash_sha": _sha_src(relations_content_hash),
        "refuse_answer_derived_fid_sha": _sha_src(refuse_answer_derived_fid),
        "trace_spec_sha": _sha_src(TraceSpec),
        "scorer_reuse": "experiments.run_tm012minimap.score_contrast",
        "refuse": [
            "non-commutable C10 where edge-fid sets differ",
            "R1 as unordered node names only",
            "parameterized R2 / DEPTH battery this pass",
            "passing TraceSpec or motor into kappa_*",
            "silent first-match find_edge for path fids",
            "two-store C10",
            "path_and_frontier route discovery",
            "raw TraceSpec / nodes as selector input",
            "declare SHA-256 or full path as product",
            "stamp Ex0S 0.0.004",
            "genome / agent change",
            "rewrite context/minimap/pathdisc/midpath locks",
        ],
    }


def write_routesig_lock(path: Path = ROUTESIG_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_routesig_lock(path: Path = ROUTESIG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not path.exists():
        return False, "docs/routesig_012.lock missing; write only via --write-lock", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key, label in (
        ("context_012_lock_sha", "context_012.lock"),
        ("minimap_012_lock_sha", "minimap_012.lock"),
        ("pathdisc_012_lock_sha", "pathdisc_012.lock"),
        ("midpath_012_lock_sha", "midpath_012.lock"),
    ):
        if snap[key] != lock.get(key):
            return False, f"{label} SHA drifted from routesig pin", snap
    for key in (
        "seed",
        "phase",
        "contrast",
        "cell_id",
        "role",
        "candidates",
        "candidate_meanings",
        "suffix_k",
        "here",
        "support",
        "trace_len",
        "kappa_api",
        "kappa_requirements",
        "validation",
        "gen_c10_sha",
        "validate_c10_sha",
        "extract_routesig_sha",
        "edge_fid_sha",
        "path_edge_fids_sha",
        "kappa_seed_sha",
        "kappa_step_sha",
        "route_kappa_sha",
        "score_contrast_sha",
        "relations_content_hash_sha",
        "refuse_answer_derived_fid_sha",
        "trace_spec_sha",
        "refuse",
    ):
        if snap[key] != lock.get(key):
            return False, f"routesig apparatus drift: {key}", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if lock.get("suffix_k") != SUFFIX_K:
        return False, f"suffix_k must be {SUFFIX_K}", snap
    if _sha_src(score_contrast) != lock.get("score_contrast_sha"):
        return False, "score_contrast SHA drifted from routesig pin", snap
    if _sha_src(edge_fid) != lock.get("edge_fid_sha"):
        return False, "edge_fid SHA drifted from routesig pin", snap
    if _sha_src(route_kappa) != lock.get("route_kappa_sha"):
        return False, "route_kappa SHA drifted from routesig pin", snap
    if _sha_src(TraceSpec) != lock.get("trace_spec_sha"):
        return False, "TraceSpec SHA drifted from routesig pin", snap
    # Kappa API must not mention TraceSpec / motor in source.
    for fn in (kappa_seed, kappa_step, route_kappa):
        src = inspect.getsource(fn)
        for banned in ("TraceSpec", "required_motor", "context_expect", "PRESS", "TUNE"):
            if banned in src:
                return False, f"{fn.__name__} source must not reference {banned}", snap
    if "path_and_frontier" in inspect.getsource(extract_states_routesig):
        return False, "extract_states_routesig must not call path_and_frontier", snap
    return True, "routesig apparatus frozen", snap


def run_routesig(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok, why, snap = verify_routesig_lock()
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

    cell = gen_c10(seed)
    route_a, route_b = cell.traces[0], cell.traces[1]
    verrs = validate_c10_pair(
        cell.relations,
        route_a,
        route_b,
        origin=cell.origin,
        hub=cell.hub,
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
    fa = path_edge_fids(cell.relations, route_a.nodes)
    fb = path_edge_fids(cell.relations, route_b.nodes)
    motor_fids = {
        edge_fid(cell.relations, cell.frontier, route_a.required_motor),
        edge_fid(cell.relations, cell.frontier, route_b.required_motor),
    }
    if not set(fa).issubset(store_fids) or not set(fb).issubset(store_fids):
        return {
            "ok": False,
            "why": "path fids not contained in shared S",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }
    if not motor_fids.issubset(store_fids):
        return {
            "ok": False,
            "why": "motor fids not contained in shared S",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    # Project from nodes only — never pass TraceSpec into extract/kappa.
    sa = extract_states_routesig(cell.relations, route_a.nodes)
    sb = extract_states_routesig(cell.relations, route_b.nodes)

    table: dict[str, dict[str, str]] = {c: {} for c in ROUTESIG_CANDIDATES}
    outcomes: dict[str, str] = {}
    for cand in ROUTESIG_CANDIDATES:
        out = score_contrast(
            contrast_id="C10",
            role=ROLE_ORDER,
            left_states=sa,
            right_states=sb,
            left_motor=route_a.required_motor,
            right_motor=route_b.required_motor,
            candidate=cand,
        )
        table[cand]["C10"] = out
        outcomes[cand] = out
        if out == OUTCOME_APPARATUS_ERROR:
            return {
                "ok": False,
                "why": f"apparatus_error on C10/{cand}",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

    for must_collide in ("R0", "R1", "R2"):
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
    for must_dist in ("R3", "R4"):
        if outcomes[must_dist] != OUTCOME_DISTINGUISHES:
            return {
                "ok": False,
                "why": (
                    f"apparatus_error: {must_dist} expected distinguishes "
                    f"got {outcomes[must_dist]}"
                ),
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

    # Outgoing motor fids inadmissible (lookup via unique edge_fid).
    out_a = edge_fid(cell.relations, cell.frontier, route_a.required_motor)
    out_b = edge_fid(cell.relations, cell.frontier, route_b.required_motor)
    inc = edge_fid(cell.relations, cell.predecessor, cell.frontier)
    refuse_a = refuse_answer_derived_fid(inc, out_a)
    refuse_b = refuse_answer_derived_fid(inc, out_b)
    if refuse_a != OUTCOME_INADMISSIBLE or refuse_b != OUTCOME_INADMISSIBLE:
        return {
            "ok": False,
            "why": f"outgoing motor fid refuse failed: {refuse_a}/{refuse_b}",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }

    # Shuffle invariance of kappa given unique edges.
    shuffled = list(cell.relations)
    random.Random(seed).shuffle(shuffled)
    ka = route_kappa(cell.origin, fa)
    ka_shuf = route_kappa(cell.origin, path_edge_fids(shuffled, route_a.nodes))
    if ka != ka_shuf:
        return {
            "ok": False,
            "why": "kappa not file-order independent under relation shuffle",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }

    claim = (
        "The set of traversed relations is insufficient. Their order is "
        "necessary for this locked contrast. An output-blind incremental "
        "order-sensitive accumulator can preserve that distinction without "
        "retaining the raw path. R3/R4 survive as candidates only — do not "
        "conclude SHA-256 is the genome primitive, that full path is "
        "necessary, or that rolling hash is minimal. Next (not this pass): "
        "ROUTESIG.DEPTH — systematic bounded-suffix family."
    )
    return {
        "ok": True,
        "why": why,
        "version": "TM.0.12.ROUTESIG",
        "phase": "order",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": seed,
        "table": table,
        "outcomes": outcomes,
        "claim": claim,
        "order_necessary": True,
        "membership_insufficient": True,
        "r4_survives": outcomes["R4"] == OUTCOME_DISTINGUISHES,
        "s_hash": s_hash,
        "origin": cell.origin,
        "hub": cell.hub,
        "predecessor": cell.predecessor,
        "frontier": cell.frontier,
        "path_fids": {"route_a": list(fa), "route_b": list(fb)},
        "traces": {
            "route_a": {
                "nodes": list(route_a.nodes),
                "motor": route_a.required_motor,
                "states": {k: _ser(v) for k, v in sa.items()},
            },
            "route_b": {
                "nodes": list(route_b.nodes),
                "motor": route_b.required_motor,
                "states": {k: _ser(v) for k, v in sb.items()},
            },
        },
        "answer_fid_refuse": {"route_a": refuse_a, "route_b": refuse_b},
        "scorer_inputs": "projected_states_and_motors_only",
        "kappa_shuffle_invariant": True,
        "lock": snap,
    }


def _ser(state: tuple[Any, ...]) -> list[Any]:
    out: list[Any] = []
    for x in state:
        if isinstance(x, frozenset):
            out.append(sorted(x))
        elif isinstance(x, tuple):
            out.append(list(x))
        else:
            out.append(x)
    return out


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012routesig"
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
        "R0": "R0 endpoint",
        "R1": "R1 unordered edge membership",
        "R2": f"R2 suffix-{SUFFIX_K}",
        "R3": "R3 ordered identity",
        "R4": "R4 rolling kappa",
    }
    short = {
        OUTCOME_DISTINGUISHES: "D",
        OUTCOME_COLLISION: "collision",
        OUTCOME_APPARATUS_ERROR: "apparatus_error",
    }
    lines = ["| Candidate | C10 |", "|-----------|-----|"]
    for cand in ROUTESIG_CANDIDATES:
        o = (table.get(cand) or {}).get("C10", "?")
        lines.append(f"| {names[cand]} | {short.get(o, o)} |")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.ROUTESIG · order (phase 1)

Apparatus: {summary.get('why')}
`earned_next`: false (no Ex0S 0.0.004)
Order necessary: **{summary.get('order_necessary')}**
Membership insufficient: **{summary.get('membership_insufficient')}**

{summary.get('claim')}

{chr(10).join(lines)}
""",
        encoding="utf-8",
    )
    summary["run_dir"] = str(run_dir)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.ROUTESIG phase 1 order")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_routesig_lock(), indent=2))
        return
    summary = run_routesig(seed=args.seed)
    if summary.get("ok"):
        write_run_artifacts(summary)
    print(
        json.dumps(
            {
                "ok": summary.get("ok"),
                "why": summary.get("why"),
                "earned_next": summary.get("earned_next"),
                "order_necessary": summary.get("order_necessary"),
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
