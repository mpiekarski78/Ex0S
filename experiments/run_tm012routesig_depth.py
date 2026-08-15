"""TM.0.12.ROUTESIG.DEPTH: fixed windows vs rolling κ.

One shared S: order witness + chain Q→T1→…→T8. Each C11[k] frontiers at T_k
so suffix-k collides and suffix-(k+1) distinguishes for every k∈[1,8].
Reuses ORDER κ / edge_fid. No genome. No Ex0S 0.0.004. No identity/rename.
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
from experiments.run_tm012routesig import (  # noqa: E402
    ROUTESIG_LOCK,
    edge_fid,
    kappa_seed,
    kappa_step,
    path_edge_fids,
    route_kappa,
)

DEPTH_LOCK = REPO_ROOT / "docs" / "routesig_depth_012.lock"

ROLE_DEPTH = "route_depth"
CELL_FAMILY = "c11_order_depth"
HERE = "chb"
SUPPORT = (1, 0)
K_FAMILY = (1, 2, 3, 4, 5, 6, 7, 8)
MOTOR_A = "press"
MOTOR_B = "tune"
# Prefix edges: X→Q→A→Q→B→Q (or B/A swapped) = 5 edges / 6 nodes; then T1..Tk.
PREFIX_HOPS = 5
DEPTH_CANDIDATES = ("R1", "R2", "R2x", "R3", "R4")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


@dataclass
class DepthStore:
    """One shared S serving all C11[k]."""

    relations: list[Rel]
    origin: str
    hub: str
    a_node: str
    b_node: str
    chain: tuple[str, ...]  # T1..T8
    annotation: str = ""


def gen_c11_store(seed: int = DEFAULT_SEED) -> DepthStore:
    """Shared S: Q↔A / Q↔B loops + Q→T1→…→T8 with PRESS/TUNE on each Tk."""
    rng = np.random.default_rng(seed + 1101)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    q = _nonce(rng, taken_n)
    a = _nonce(rng, taken_n)
    b = _nonce(rng, taken_n)
    chain = tuple(_nonce(rng, taken_n) for _ in range(8))
    rels: list[Rel] = [
        Rel(_fid(rng, taken_f), x, q, "xq", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), q, a, "qa", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), a, q, "aq", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), q, b, "qb", SUPPORT, here=HERE),
        Rel(_fid(rng, taken_f), b, q, "bq", SUPPORT, here=HERE),
    ]
    # Q → T1 → … → T8
    prev = q
    for i, t in enumerate(chain):
        rels.append(Rel(_fid(rng, taken_f), prev, t, f"t{i}", SUPPORT, here=HERE))
        prev = t
    for t in chain:
        rels.append(Rel(_fid(rng, taken_f), t, MOTOR_A, "ym_a", SUPPORT, here=HERE))
        rels.append(Rel(_fid(rng, taken_f), t, MOTOR_B, "ym_b", SUPPORT, here=HERE))
    return DepthStore(
        relations=rels,
        origin=x,
        hub=q,
        a_node=a,
        b_node=b,
        chain=chain,
        annotation="order witness + T1..T8; C11[k] frontiers at Tk",
    )


def traces_for_k(store: DepthStore, k: int) -> tuple[TraceSpec, TraceSpec, str]:
    """Observed traces ending at frontier Tk. Motors fixed PRESS/TUNE."""
    if k not in K_FAMILY:
        raise ValueError(f"k={k} not in {K_FAMILY}")
    tail = store.chain[:k]
    frontier = store.chain[k - 1]
    nodes_a = (store.origin, store.hub, store.a_node, store.hub, store.b_node, store.hub) + tail
    nodes_b = (store.origin, store.hub, store.b_node, store.hub, store.a_node, store.hub) + tail
    return (
        TraceSpec(f"route_a_k{k}", nodes_a, MOTOR_A),
        TraceSpec(f"route_b_k{k}", nodes_b, MOTOR_B),
        frontier,
    )


def assert_geometry_k(
    rels: Sequence[Rel], nodes_a: Sequence[str], nodes_b: Sequence[str], k: int
) -> list[str]:
    """Fail closed unless distinction sits exactly one edge outside suffix-k."""
    errs: list[str] = []
    try:
        fa = path_edge_fids(rels, nodes_a)
        fb = path_edge_fids(rels, nodes_b)
    except ValueError as e:
        return [str(e)]
    if len(fa) < k + 1 or len(fb) < k + 1:
        errs.append(f"path shorter than k+1={k + 1}")
        return errs
    if fa[-k:] != fb[-k:]:
        errs.append(f"suffix-{k} must match: {fa[-k:]!r} vs {fb[-k:]!r}")
    if fa[-(k + 1)] == fb[-(k + 1)]:
        errs.append(f"edge immediately before suffix-{k} must differ")
    if fa[-(k + 1) + 1 :] != fb[-(k + 1) + 1 :]:
        errs.append(f"suffix after differing edge must match for k={k}")
    if frozenset(fa) != frozenset(fb):
        errs.append("unordered path-edge fid sets must match")
    if sorted(fa) != sorted(fb):
        errs.append("path-edge fid multisets must match")
    if fa == fb:
        errs.append("ordered sequences must differ")
    return errs


def validate_store(store: DepthStore) -> list[str]:
    errs: list[str] = []
    motors = {m.lower() for m in MOTORS}
    if MOTOR_A not in motors or MOTOR_B not in motors:
        errs.append("locked motors not in MOTORS")
    if MOTOR_A == MOTOR_B:
        errs.append("motors must differ")
    if len(store.chain) != 8 or len({t.lower() for t in store.chain}) != 8:
        errs.append("chain must be eight distinct T_k")
    pair_counts = Counter((r.bind.lower(), r.did.lower()) for r in store.relations)
    for (b, d), n in sorted(pair_counts.items()):
        if n != 1:
            errs.append(f"duplicate directed edge {b}→{d} count={n}")
    store_fids = {r.fid for r in store.relations}
    for r in store.relations:
        if r.here != HERE:
            errs.append(f"here drift {r.fid}")
        if r.init != SUPPORT:
            errs.append(f"support drift {r.fid}")
    frontiers: list[str] = []
    for k in K_FAMILY:
        ta, tb, fr = traces_for_k(store, k)
        frontiers.append(fr.lower())
        if ta.nodes[-1].lower() != fr.lower() or tb.nodes[-1].lower() != fr.lower():
            errs.append(f"k={k} frontier mismatch")
        if len(ta.nodes) != PREFIX_HOPS + 1 + k:
            errs.append(f"k={k} unexpected length {len(ta.nodes)}")
        if ta.required_motor.lower() != MOTOR_A or tb.required_motor.lower() != MOTOR_B:
            errs.append(f"k={k} motors must be locked PRESS/TUNE")
        errs.extend(assert_geometry_k(store.relations, ta.nodes, tb.nodes, k))
        try:
            fa = path_edge_fids(store.relations, ta.nodes)
            fb = path_edge_fids(store.relations, tb.nodes)
        except ValueError as e:
            errs.append(f"k={k} path fids: {e}")
            continue
        if not set(fa).issubset(store_fids) or not set(fb).issubset(store_fids):
            errs.append(f"k={k} path fids not contained in shared S")
        for tr in (ta, tb):
            try:
                edge_fid(store.relations, fr, tr.required_motor)
            except ValueError as e:
                errs.append(f"k={k} Y→motor: {e}")
    if len(set(frontiers)) != 8:
        errs.append("C11 frontiers must be eight distinct T_k")
    return errs


def extract_states_depth(
    rels: Sequence[Rel], nodes: Sequence[str], *, k: int
) -> dict[str, tuple[Any, ...]]:
    ns = [n.lower() for n in nodes]
    frontier = ns[-1]
    origin = ns[0]
    fids = path_edge_fids(rels, ns)
    if len(fids) < k + 1:
        raise ValueError("path shorter than k+1")
    return {
        "R1": (frontier, frozenset(fids)),
        "R2": (frontier, fids[-k:]),
        "R2x": (frontier, fids[-(k + 1) :]),
        "R3": (frontier, fids),
        "R4": (frontier, route_kappa(origin, fids)),
    }


def locked_family() -> dict[str, Any]:
    rows = {}
    for k in K_FAMILY:
        rows[f"C11[{k}]"] = {
            "k": k,
            "frontier": f"T{k}",
            "common_tail_edges": k,
            "trace_len_nodes": PREFIX_HOPS + 1 + k,
            "role": ROLE_DEPTH,
            "motors": {"route_a": MOTOR_A, "route_b": MOTOR_B},
        }
    return rows


def apparatus_snapshot() -> dict[str, Any]:
    for path, name in (
        (CONTEXT_LOCK, "context_012.lock"),
        (MINIMAP_LOCK, "minimap_012.lock"),
        (PATHDISC_LOCK, "pathdisc_012.lock"),
        (MIDPATH_LOCK, "midpath_012.lock"),
        (ROUTESIG_LOCK, "routesig_012.lock"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"docs/{name} missing")
    return {
        "version": "TM.0.12.ROUTESIG.DEPTH",
        "phase": "depth",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": DEFAULT_SEED,
        "context_012_lock_sha": _sha_file(CONTEXT_LOCK),
        "minimap_012_lock_sha": _sha_file(MINIMAP_LOCK),
        "pathdisc_012_lock_sha": _sha_file(PATHDISC_LOCK),
        "midpath_012_lock_sha": _sha_file(MIDPATH_LOCK),
        "routesig_012_lock_sha": _sha_file(ROUTESIG_LOCK),
        "cell_family": CELL_FAMILY,
        "k_family": list(K_FAMILY),
        "family": locked_family(),
        "candidates": list(DEPTH_CANDIDATES),
        "candidate_meanings": {
            "R1": "unordered edge membership",
            "R2": "suffix-k (under test)",
            "R2x": "suffix-(k+1) geometric diagnostic",
            "R3": "ordered route identity",
            "R4": "rolling kappa",
        },
        "motors": {"route_a": MOTOR_A, "route_b": MOTOR_B},
        "prefix_hops": PREFIX_HOPS,
        "here": HERE,
        "support": list(SUPPORT),
        "geometry_guards": [
            "fa[-k:] == fb[-k:]",
            "fa[-(k+1)] != fb[-(k+1)]",
            "fa[-(k+1)+1:] == fb[-(k+1)+1:]",
        ],
        "validation": [
            "one shared S; C11[k] frontiers at distinct T_k",
            "order witness prefixes then common tail of k edges",
            "same edge-fid set and multiset per k",
            "geometric R2/R2x placement before scoring",
            "unique directed edges in S",
            "same PRESS/TUNE motors for all k",
            "runtime seed equals locked seed",
            "raw TraceSpec extractor-only; kappa path-fids only",
            "TraceSpec construction pin; extract bans path_and_frontier",
            "path-edge fids subset of shared S",
            "C11[2] structurally equivalent to C10 suffix-2 condition (pattern only)",
        ],
        "gen_c11_store_sha": _sha_src(gen_c11_store),
        "traces_for_k_sha": _sha_src(traces_for_k),
        "assert_geometry_k_sha": _sha_src(assert_geometry_k),
        "extract_depth_sha": _sha_src(extract_states_depth),
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
        "kappa_reuse": "experiments.run_tm012routesig",
        "refuse": [
            "single fixed frontier for all k",
            "rewrite routesig_012.lock / C10",
            "claim C11[2] is the same frozen cell as C10",
            "universal no-finite-suffix claim",
            "SHA-as-genome / full-path necessity",
            "identity/rename battery this pass",
            "stamp Ex0S 0.0.004",
            "genome / agent change",
            "path_and_frontier route discovery",
            "raw TraceSpec as selector input",
        ],
    }


def write_depth_lock(path: Path = DEPTH_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_depth_lock(path: Path = DEPTH_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not path.exists():
        return False, "docs/routesig_depth_012.lock missing; write only via --write-lock", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key, label in (
        ("context_012_lock_sha", "context_012.lock"),
        ("minimap_012_lock_sha", "minimap_012.lock"),
        ("pathdisc_012_lock_sha", "pathdisc_012.lock"),
        ("midpath_012_lock_sha", "midpath_012.lock"),
        ("routesig_012_lock_sha", "routesig_012.lock"),
    ):
        if snap[key] != lock.get(key):
            return False, f"{label} SHA drifted from depth pin", snap
    for key in (
        "seed",
        "phase",
        "cell_family",
        "k_family",
        "family",
        "candidates",
        "candidate_meanings",
        "motors",
        "prefix_hops",
        "here",
        "support",
        "geometry_guards",
        "validation",
        "gen_c11_store_sha",
        "traces_for_k_sha",
        "assert_geometry_k_sha",
        "extract_depth_sha",
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
            return False, f"depth apparatus drift: {key}", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if tuple(lock.get("k_family") or ()) != K_FAMILY:
        return False, "k_family must be 1..8", snap
    if _sha_src(score_contrast) != lock.get("score_contrast_sha"):
        return False, "score_contrast SHA drifted from depth pin", snap
    if _sha_src(route_kappa) != lock.get("route_kappa_sha"):
        return False, "route_kappa SHA drifted", snap
    if _sha_src(edge_fid) != lock.get("edge_fid_sha"):
        return False, "edge_fid SHA drifted", snap
    if _sha_src(TraceSpec) != lock.get("trace_spec_sha"):
        return False, "TraceSpec SHA drifted from depth pin", snap
    for fn in (kappa_seed, kappa_step, route_kappa):
        src = inspect.getsource(fn)
        for banned in ("TraceSpec", "required_motor", "context_expect", "PRESS", "TUNE"):
            if banned in src:
                return False, f"{fn.__name__} source must not reference {banned}", snap
    if "path_and_frontier" in inspect.getsource(extract_states_depth):
        return False, "extract_states_depth must not call path_and_frontier", snap
    return True, "routesig depth apparatus frozen", snap


def run_depth(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok, why, snap = verify_depth_lock()
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

    store = gen_c11_store(seed)
    verrs = validate_store(store)
    if verrs:
        return {
            "ok": False,
            "why": "validation failed: " + "; ".join(verrs[:12]),
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    s_hash = relations_content_hash(store.relations)
    table: dict[str, dict[str, str]] = {c: {} for c in DEPTH_CANDIDATES}
    size_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for k in K_FAMILY:
        ta, tb, frontier = traces_for_k(store, k)
        geo = assert_geometry_k(store.relations, ta.nodes, tb.nodes, k)
        if geo:
            return {
                "ok": False,
                "why": f"geometry fail k={k}: " + "; ".join(geo),
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

        sa = extract_states_depth(store.relations, ta.nodes, k=k)
        sb = extract_states_depth(store.relations, tb.nodes, k=k)
        fa = path_edge_fids(store.relations, ta.nodes)
        fb = path_edge_fids(store.relations, tb.nodes)

        outcomes_k: dict[str, str] = {}
        for cand in DEPTH_CANDIDATES:
            out = score_contrast(
                contrast_id=f"C11[{k}]",
                role=ROLE_DEPTH,
                left_states=sa,
                right_states=sb,
                left_motor=ta.required_motor,
                right_motor=tb.required_motor,
                candidate=cand,
            )
            table[cand][f"C11[{k}]"] = out
            outcomes_k[cand] = out
            if out == OUTCOME_APPARATUS_ERROR:
                return {
                    "ok": False,
                    "why": f"apparatus_error on C11[{k}]/{cand}",
                    "earned_next": False,
                    "ex0s_under_test": "0.0.003",
                    "table": table,
                }

        if outcomes_k["R1"] != OUTCOME_COLLISION:
            return {
                "ok": False,
                "why": f"k={k} R1 expected collision got {outcomes_k['R1']}",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }
        if outcomes_k["R2"] != OUTCOME_COLLISION:
            return {
                "ok": False,
                "why": f"k={k} R2 expected collision got {outcomes_k['R2']}",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }
        for must_d in ("R2x", "R3", "R4"):
            if outcomes_k[must_d] != OUTCOME_DISTINGUISHES:
                return {
                    "ok": False,
                    "why": f"k={k} {must_d} expected distinguishes got {outcomes_k[must_d]}",
                    "earned_next": False,
                    "ex0s_under_test": "0.0.003",
                    "table": table,
                }

        # Outgoing motor refuse at this frontier (both routes).
        out_a = edge_fid(store.relations, frontier, MOTOR_A)
        out_b = edge_fid(store.relations, frontier, MOTOR_B)
        inc = fa[-1]
        refuse_a = refuse_answer_derived_fid(inc, out_a)
        refuse_b = refuse_answer_derived_fid(inc, out_b)
        if refuse_a != OUTCOME_INADMISSIBLE or refuse_b != OUTCOME_INADMISSIBLE:
            return {
                "ok": False,
                "why": f"k={k} answer fid refuse failed: {refuse_a}/{refuse_b}",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

        size_rows.append(
            {
                "k": k,
                "frontier": frontier,
                "n_path_edges": len(fa),
                "R2_fid_count": k,
                "R3_fid_count": len(fa),
                "R4_bits": len(sa["R4"][1]) * 4,  # hex digest → bits
            }
        )
        details[f"C11[{k}]"] = {
            "frontier": frontier,
            "outcomes": outcomes_k,
            "path_fids_a": list(fa),
            "path_fids_b": list(fb),
            "geometry": {
                "suffix_k_equal": fa[-k:] == fb[-k:],
                "edge_before_differs": fa[-(k + 1)] != fb[-(k + 1)],
            },
        }

    # Shuffle invariance of κ on k=8 path.
    ta8, _, _ = traces_for_k(store, 8)
    fa8 = path_edge_fids(store.relations, ta8.nodes)
    shuffled = list(store.relations)
    random.Random(seed).shuffle(shuffled)
    if route_kappa(store.origin, fa8) != route_kappa(
        store.origin, path_edge_fids(shuffled, ta8.nodes)
    ):
        return {
            "ok": False,
            "why": "kappa not file-order independent under shuffle",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }
    if any(row["R4_bits"] != 256 for row in size_rows):
        return {
            "ok": False,
            "why": "R4 kappa bit-width drifted from 256",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }

    # C11[2] ~ C10 suffix-2 pattern (not same cell) — fail closed.
    c112 = details["C11[2]"]["outcomes"]
    pattern_ok = (
        c112["R1"] == OUTCOME_COLLISION
        and c112["R2"] == OUTCOME_COLLISION
        and c112["R3"] == OUTCOME_DISTINGUISHES
        and c112["R4"] == OUTCOME_DISTINGUISHES
    )
    if not pattern_ok:
        return {
            "ok": False,
            "why": "C11[2] failed C10 suffix-2 outcome pattern",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
            "c11_2_matches_c10_suffix2_pattern": False,
        }

    claim = (
        "For each tested fixed window k=1..8, a route distinction placed "
        "exactly k+1 edges before the frontier is lost by suffix-k but "
        "retained by the incremental rolling accumulator. R3 grows with k; "
        "each fixed window of size k fails when the distinction is k+1 back; "
        "R4 stays constant-size across tested depths. Not: no finite suffix "
        "can ever suffice; SHA-as-genome; full-path necessity; 0.0.004. "
        "Next (not this pass): identity/rename robustness of κ."
    )
    return {
        "ok": True,
        "why": why,
        "version": "TM.0.12.ROUTESIG.DEPTH",
        "phase": "depth",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": seed,
        "table": table,
        "size_vs_k": size_rows,
        "details": details,
        "claim": claim,
        "depth_windows_insufficient": True,
        "kappa_preserves_tested_depths": True,
        "c11_2_matches_c10_suffix2_pattern": pattern_ok,
        "s_hash": s_hash,
        "origin": store.origin,
        "hub": store.hub,
        "chain": list(store.chain),
        "kappa_shuffle_invariant": True,
        "scorer_inputs": "projected_states_and_motors_only",
        "lock": snap,
    }


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012routesig_depth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifacts(summary: dict[str, Any]) -> Path:
    run_dir = _run_dir()
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    (run_dir / "metrics.json").write_text(
        json.dumps(clean, indent=2, default=str) + "\n", encoding="utf-8"
    )
    table = summary.get("table") or {}
    headers = ["Candidate"] + [f"C11[{k}]" for k in K_FAMILY]
    short = {
        OUTCOME_DISTINGUISHES: "D",
        OUTCOME_COLLISION: "collision",
        OUTCOME_APPARATUS_ERROR: "apparatus_error",
    }
    names = {
        "R1": "R1 unordered",
        "R2": "R2 suffix-k",
        "R2x": "R2x suffix-(k+1)",
        "R3": "R3 ordered",
        "R4": "R4 kappa",
    }
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["-----------"] * len(headers)) + "|"]
    for cand in DEPTH_CANDIDATES:
        row = [names[cand]]
        for k in K_FAMILY:
            o = (table.get(cand) or {}).get(f"C11[{k}]", "?")
            row.append(short.get(o, o))
        lines.append("| " + " | ".join(row) + " |")
    size_lines = ["| k | R2 fids | R3 fids | R4 bits |", "|---|---------|---------|---------|"]
    for row in summary.get("size_vs_k") or []:
        size_lines.append(
            f"| {row['k']} | {row['R2_fid_count']} | {row['R3_fid_count']} | {row['R4_bits']} |"
        )
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.ROUTESIG.DEPTH

Apparatus: {summary.get('why')}
`earned_next`: false (no Ex0S 0.0.004)
Depth windows insufficient: **{summary.get('depth_windows_insufficient')}**
κ preserves tested depths: **{summary.get('kappa_preserves_tested_depths')}**
C11[2] ~ C10 suffix-2 pattern: **{summary.get('c11_2_matches_c10_suffix2_pattern')}**

{summary.get('claim')}

{chr(10).join(lines)}

## Size vs k

{chr(10).join(size_lines)}
""",
        encoding="utf-8",
    )
    summary["run_dir"] = str(run_dir)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.ROUTESIG.DEPTH")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_depth_lock(), indent=2))
        return
    summary = run_depth(seed=args.seed)
    if summary.get("ok"):
        write_run_artifacts(summary)
    print(
        json.dumps(
            {
                "ok": summary.get("ok"),
                "why": summary.get("why"),
                "earned_next": summary.get("earned_next"),
                "depth_windows_insufficient": summary.get("depth_windows_insufficient"),
                "c11_2_matches_c10_suffix2_pattern": summary.get(
                    "c11_2_matches_c10_suffix2_pattern"
                ),
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
