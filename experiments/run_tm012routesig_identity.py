"""TM.0.12.ROUTESIG.IDENTITY: storage-row fid vs semantic relation identity.

C12A: bijective fid rename on frozen C10 — everything else identical.
C12B: same fid, last path hop's (bind,did) rewritten.
Kfid steps on fid; Ksem steps on canonical(bind, did) only.
Same rolling F (kappa_seed / kappa_step). No alpha-rename. No genome.
No Ex0S 0.0.004.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012context import (  # noqa: E402
    CONTEXT_LOCK,
    DEFAULT_SEED,
    Rel,
    _fid,
    _nonce,
)
from experiments.run_tm012minimap import MINIMAP_LOCK  # noqa: E402
from experiments.run_tm012pathdisc import (  # noqa: E402
    PATHDISC_LOCK,
    TraceSpec,
    relations_content_hash,
)
from experiments.run_tm012midpath import MIDPATH_LOCK  # noqa: E402
from experiments.run_tm012routesig import (  # noqa: E402
    ROUTESIG_LOCK,
    RouteSigCell,
    edge_fid,
    gen_c10,
    kappa_seed,
    kappa_step,
    path_edge_fids,
    route_kappa,
    validate_c10_pair,
)
from experiments.run_tm012routesig_depth import DEPTH_LOCK  # noqa: E402

IDENTITY_LOCK = REPO_ROOT / "docs" / "routesig_identity_012.lock"

ROLE_IDENTITY = "route_identity"
HERE = "chb"
SUPPORT = (1, 0)
OUTCOME_SAME = "same"
OUTCOME_DIFFERS = "differs"
IDENTITY_CANDIDATES = ("Kfid", "Ksem")
CELLS = ("C12A", "C12B")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def edge_sem(bind: str, did: str) -> str:
    """Canonical structural identity: directed endpoints only."""
    return bind.lower() + "\0" + did.lower()


def path_edge_sems(rels: Sequence[Rel], nodes: Sequence[str]) -> tuple[str, ...]:
    """Ordered semantic identities along an observed path that exists in S."""
    ns = [n.lower() for n in nodes]
    if len(ns) < 2:
        raise ValueError("path too short")
    path_edge_fids(rels, ns)  # fail closed if a hop is missing / non-unique
    return tuple(edge_sem(a, b) for a, b in zip(ns, ns[1:]))


def route_kappa_fid(origin: str, rels: Sequence[Rel], nodes: Sequence[str]) -> str:
    return route_kappa(origin, path_edge_fids(rels, nodes))


def route_kappa_sem(origin: str, rels: Sequence[Rel], nodes: Sequence[str]) -> str:
    return route_kappa(origin, path_edge_sems(rels, nodes))


def score_kappa_pair(left: str, right: str) -> str:
    """Compare two accumulator digests. No motors. Table not encoded here."""
    if left == right:
        return OUTCOME_SAME
    return OUTCOME_DIFFERS


@dataclass
class IdentityPair:
    cell_id: str
    left: RouteSigCell
    right: RouteSigCell
    annotation: str = ""


def _copy_rel(r: Rel, *, fid: str | None = None, did: str | None = None) -> Rel:
    return Rel(
        fid=r.fid if fid is None else fid,
        bind=r.bind,
        did=r.did if did is None else did,
        role=r.role,
        init=r.init,
        here=r.here,
    )


def rename_fids(rels: Sequence[Rel], seed: int) -> list[Rel]:
    """Bijective fid rename. New fids disjoint from old. Other fields copied."""
    rng = np.random.default_rng(seed + 1201)
    taken = {r.fid for r in rels}
    mapping = {r.fid: _fid(rng, taken) for r in rels}
    if set(mapping.values()) & {r.fid for r in rels}:
        raise ValueError("fid rename not disjoint")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("fid rename not injective")
    return [_copy_rel(r, fid=mapping[r.fid]) for r in rels]


def gen_c12a(seed: int = DEFAULT_SEED) -> IdentityPair:
    """Frozen C10 vs the same graph with only fids renamed."""
    left = gen_c10(seed)
    renamed = rename_fids(left.relations, seed)
    right = replace(left, relations=renamed, annotation="C12A fid-renamed clone")
    return IdentityPair(
        cell_id="C12A",
        left=left,
        right=right,
        annotation="opaque fid rename; bind/did/role/here/support/order/motors fixed",
    )


def gen_c12b(seed: int = DEFAULT_SEED) -> IdentityPair:
    """Frozen C10 vs same fids with last path hop P→Y rewritten to P→Z."""
    left = gen_c10(seed)
    taken_n = {r.bind.lower() for r in left.relations} | {r.did.lower() for r in left.relations}
    taken_n |= {n.lower() for tr in left.traces for n in tr.nodes}
    rng = np.random.default_rng(seed + 1202)
    z = _nonce(rng, taken_n)
    pred = left.predecessor
    old_fr = left.frontier
    rewritten: list[Rel] = []
    found = 0
    for r in left.relations:
        if r.bind.lower() == pred.lower() and r.did.lower() == old_fr.lower():
            rewritten.append(_copy_rel(r, did=z))
            found += 1
        else:
            rewritten.append(_copy_rel(r))
    if found != 1:
        raise ValueError(f"expected one P→Y to rewrite, found {found}")
    taken_f = {r.fid for r in rewritten}
    for motor in sorted({tr.required_motor.lower() for tr in left.traces}):
        rewritten.append(Rel(_fid(rng, taken_f), z, motor, "zm", SUPPORT, here=HERE))
    new_traces = [
        TraceSpec(
            tr.label,
            tuple(z if n.lower() == old_fr.lower() else n for n in tr.nodes),
            tr.required_motor,
        )
        for tr in left.traces
    ]
    right = RouteSigCell(
        cell_id="c12b_same_fid_rewrite",
        relations=rewritten,
        origin=left.origin,
        hub=left.hub,
        predecessor=pred,
        frontier=z,
        traces=new_traces,
        annotation="C12B same-fid last-hop rewrite",
    )
    return IdentityPair(
        cell_id="C12B",
        left=left,
        right=right,
        annotation="same path fids; last hop (bind,did) changed; not alpha-rename",
    )


def _rel_key(r: Rel) -> tuple[str, str]:
    return (r.bind.lower(), r.did.lower())


def _struct_fields(r: Rel) -> tuple[str, str, str, tuple[int, int], str]:
    return (r.bind.lower(), r.did.lower(), r.role, r.init, r.here)


def _unique_fids(rels: Sequence[Rel], label: str) -> list[str]:
    fids = [r.fid for r in rels]
    if len(fids) != len(set(fids)):
        return [f"{label} fids must be unique"]
    return []


def assert_c12a_geometry(pair: IdentityPair) -> list[str]:
    errs: list[str] = []
    L, R = pair.left, pair.right
    errs.extend(_unique_fids(L.relations, "C12A left"))
    errs.extend(_unique_fids(R.relations, "C12A right"))
    if len(L.traces) != 2 or len(R.traces) != 2:
        errs.append("C12A must keep both C10 traces")
        return errs
    for i, (tl, tr) in enumerate(zip(L.traces, R.traces)):
        if [n.lower() for n in tl.nodes] != [n.lower() for n in tr.nodes]:
            errs.append(f"C12A route_{i} node sequences must be identical")
        if tl.required_motor.lower() != tr.required_motor.lower():
            errs.append(f"C12A route_{i} motors must be identical")
    left_keys = [_rel_key(r) for r in L.relations]
    right_keys = [_rel_key(r) for r in R.relations]
    if sorted(left_keys) != sorted(right_keys) or len(left_keys) != len(set(left_keys)):
        errs.append("C12A (bind,did) sets must match with unique keys")
        return errs
    left_by = {_rel_key(r): r for r in L.relations}
    right_by = {_rel_key(r): r for r in R.relations}
    for key, lr in left_by.items():
        rr = right_by[key]
        if _struct_fields(lr) != _struct_fields(rr):
            errs.append(f"C12A structural fields drifted on {key}")
        if lr.fid == rr.fid:
            errs.append(f"C12A fid not renamed on {key}")
    if {r.fid for r in L.relations} & {r.fid for r in R.relations}:
        errs.append("C12A fid sets must be disjoint")
    for i, (tl, tr) in enumerate(zip(L.traces, R.traces)):
        fa = path_edge_fids(L.relations, tl.nodes)
        fb = path_edge_fids(R.relations, tr.nodes)
        if len(fa) != len(fb) or not fa or any(a == b for a, b in zip(fa, fb)):
            errs.append(f"C12A route_{i} every path fid must change")
        sa = path_edge_sems(L.relations, tl.nodes)
        sb = path_edge_sems(R.relations, tr.nodes)
        if sa != sb:
            errs.append(f"C12A route_{i} path sems must be identical")
    return errs


def assert_c12b_geometry(pair: IdentityPair) -> list[str]:
    errs: list[str] = []
    L, R = pair.left, pair.right
    errs.extend(_unique_fids(L.relations, "C12B left"))
    errs.extend(_unique_fids(R.relations, "C12B right"))
    if len(L.traces) != 2 or len(R.traces) != 2:
        errs.append("C12B must keep both C10 traces")
        return errs
    left_by_fid = {r.fid: r for r in L.relations}
    inherited = [r for r in R.relations if r.fid in left_by_fid]
    added = [r for r in R.relations if r.fid not in left_by_fid]
    if len(inherited) != len(L.relations):
        errs.append("C12B must inherit every left fid exactly once")
    changed = [
        (left_by_fid[r.fid], r)
        for r in inherited
        if _rel_key(left_by_fid[r.fid]) != _rel_key(r)
    ]
    if len(changed) != 1:
        errs.append(f"C12B must rewrite exactly one inherited (bind,did), found {len(changed)}")
    else:
        old, new = changed[0]
        if old.fid != new.fid:
            errs.append("C12B rewritten hop must keep its fid")
        if old.bind.lower() != new.bind.lower():
            errs.append("C12B rewrite must keep bind (predecessor)")
        if old.did.lower() == new.did.lower():
            errs.append("C12B rewrite must change did")
        if old.did.lower() != L.frontier.lower() or new.did.lower() != R.frontier.lower():
            errs.append("C12B rewrite must be the last hop P→Y to P→Z")
        if old.role != new.role or old.init != new.init or old.here != new.here:
            errs.append("C12B rewrite must not touch role/here/support")
    motors = {tr.required_motor.lower() for tr in L.traces}
    if len(added) != len(motors):
        errs.append(f"C12B must add exactly {len(motors)} new frontier→motor edges")
    for r in added:
        if r.bind.lower() != R.frontier.lower() or r.did.lower() not in motors:
            errs.append("C12B added edges must be Z→locked motors")
    for i, (tl, tr) in enumerate(zip(L.traces, R.traces)):
        na = [n.lower() for n in tl.nodes]
        nb = [n.lower() for n in tr.nodes]
        if len(na) != len(nb):
            errs.append(f"C12B route_{i} lengths must match")
            continue
        if na[:-1] != nb[:-1]:
            errs.append(f"C12B route_{i} may change only the last path token")
        if na[-1] == nb[-1]:
            errs.append(f"C12B route_{i} last token must change")
        if tl.required_motor.lower() != tr.required_motor.lower():
            errs.append(f"C12B route_{i} motors must match")
        if tl.nodes[0].lower() != L.origin.lower() or tr.nodes[0].lower() != R.origin.lower():
            errs.append(f"C12B route_{i} origin mismatch")
        fa = path_edge_fids(L.relations, tl.nodes)
        fb = path_edge_fids(R.relations, tr.nodes)
        if fa != fb:
            errs.append(f"C12B route_{i} ordered path fids must be identical")
        sa = path_edge_sems(L.relations, tl.nodes)
        sb = path_edge_sems(R.relations, tr.nodes)
        if sa[:-1] != sb[:-1] or sa[-1] == sb[-1]:
            errs.append(f"C12B route_{i} must differ at exactly the last semantic hop")
        if set(na) == set(nb):
            errs.append(f"C12B route_{i} must introduce a new frontier, not permute tokens")
        if (set(na) & set(nb)) != set(na[:-1]):
            errs.append(f"C12B route_{i} shared tokens must be the unchanged prefix")
    return errs


def validate_pair(pair: IdentityPair) -> list[str]:
    errs: list[str] = []
    for side, cell in (("left", pair.left), ("right", pair.right)):
        pair_counts = Counter((r.bind.lower(), r.did.lower()) for r in cell.relations)
        for (b, d), n in sorted(pair_counts.items()):
            if n != 1:
                errs.append(f"{side} duplicate directed edge {b}→{d} count={n}")
        for r in cell.relations:
            if r.here != HERE:
                errs.append(f"{side} here drift {r.fid}")
            if r.init != SUPPORT:
                errs.append(f"{side} support drift {r.fid}")
        c10_errs = validate_c10_pair(
            cell.relations,
            cell.traces[0],
            cell.traces[1],
            origin=cell.origin,
            hub=cell.hub,
            predecessor=cell.predecessor,
            frontier=cell.frontier,
        )
        errs.extend(f"{side} {e}" for e in c10_errs)
    if pair.cell_id == "C12A":
        errs.extend(assert_c12a_geometry(pair))
    elif pair.cell_id == "C12B":
        errs.extend(assert_c12b_geometry(pair))
    else:
        errs.append(f"unknown cell {pair.cell_id}")
    return errs


def extract_identity(
    rels: Sequence[Rel], nodes: Sequence[str]
) -> dict[str, str]:
    origin = nodes[0]
    return {
        "Kfid": route_kappa_fid(origin, rels, nodes),
        "Ksem": route_kappa_sem(origin, rels, nodes),
    }


def locked_cells() -> dict[str, Any]:
    return {
        "C12A": {
            "role": ROLE_IDENTITY,
            "perturbation": "opaque_fid_rename",
            "holds_fixed": ["bind", "did", "role", "here", "support", "route_order", "motors"],
            "varies": ["fid"],
            "witness": "frozen C10 via gen_c10",
        },
        "C12B": {
            "role": ROLE_IDENTITY,
            "perturbation": "same_fid_last_hop_rewrite",
            "holds_fixed": ["path_fids", "prefix_tokens", "motors"],
            "varies": ["last_did", "frontier_token"],
            "note": "diagnostic if revision always mints a new fid",
        },
    }


def apparatus_snapshot() -> dict[str, Any]:
    for path, name in (
        (CONTEXT_LOCK, "context_012.lock"),
        (MINIMAP_LOCK, "minimap_012.lock"),
        (PATHDISC_LOCK, "pathdisc_012.lock"),
        (MIDPATH_LOCK, "midpath_012.lock"),
        (ROUTESIG_LOCK, "routesig_012.lock"),
        (DEPTH_LOCK, "routesig_depth_012.lock"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"docs/{name} missing")
    return {
        "version": "TM.0.12.ROUTESIG.IDENTITY",
        "phase": "identity",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": DEFAULT_SEED,
        "context_012_lock_sha": _sha_file(CONTEXT_LOCK),
        "minimap_012_lock_sha": _sha_file(MINIMAP_LOCK),
        "pathdisc_012_lock_sha": _sha_file(PATHDISC_LOCK),
        "midpath_012_lock_sha": _sha_file(MIDPATH_LOCK),
        "routesig_012_lock_sha": _sha_file(ROUTESIG_LOCK),
        "routesig_depth_012_lock_sha": _sha_file(DEPTH_LOCK),
        "cells": locked_cells(),
        "candidates": list(IDENTITY_CANDIDATES),
        "candidate_meanings": {
            "Kfid": "rolling F over origin + ordered path fids (storage-row identity)",
            "Ksem": "same F over origin + ordered canonical(bind, did) (relation identity)",
        },
        "ksem_fields": ["bind", "did"],
        "ksem_excludes": ["fid", "here", "support", "role", "motor", "context_expect"],
        "kappa_reuse": [
            "kappa_seed(origin)",
            "kappa_step(previous_kappa, token)",
            "route_kappa(origin, ordered_tokens)",
        ],
        "here": HERE,
        "support": list(SUPPORT),
        "validation": [
            "C12A: bijective disjoint fid rename; structural fields identical",
            "C12A: path fids differ; path sems identical; nodes and motors identical",
            "C12B: ordered path fids identical; last semantic hop only differs",
            "C12B: prefix tokens fixed; not a full token alpha-rename",
            "C12B: exactly one inherited (bind,did) rewritten; fid kept; Z→motors added",
            "both C10 traces checked; unique fids; file-order shuffle invariance",
            "Ksem canonical(bind, did) only — no here/support/role/fid",
            "same rolling F as ORDER (live SHA pin); payload differs",
            "ORDER distinction preserved on both encodings (Ksem route_a != route_b)",
            "unique directed edges; runtime seed equals locked seed",
            "raw TraceSpec extractor-only; kappa path tokens only",
        ],
        "gen_c10_sha": _sha_src(gen_c10),
        "gen_c12a_sha": _sha_src(gen_c12a),
        "gen_c12b_sha": _sha_src(gen_c12b),
        "rename_fids_sha": _sha_src(rename_fids),
        "edge_sem_sha": _sha_src(edge_sem),
        "path_edge_sems_sha": _sha_src(path_edge_sems),
        "extract_identity_sha": _sha_src(extract_identity),
        "score_kappa_pair_sha": _sha_src(score_kappa_pair),
        "assert_c12a_sha": _sha_src(assert_c12a_geometry),
        "assert_c12b_sha": _sha_src(assert_c12b_geometry),
        "edge_fid_sha": _sha_src(edge_fid),
        "path_edge_fids_sha": _sha_src(path_edge_fids),
        "kappa_seed_sha": _sha_src(kappa_seed),
        "kappa_step_sha": _sha_src(kappa_step),
        "route_kappa_sha": _sha_src(route_kappa),
        "trace_spec_sha": _sha_src(TraceSpec),
        "refuse": [
            "alpha-rename / graph-isomorphism / token permutation battery",
            "require identical digest bits after token rename",
            "hash here/support/role/fid into Ksem",
            "claim Kfid fails cognition",
            "rewrite routesig_012.lock / routesig_depth_012.lock",
            "stamp Ex0S 0.0.004",
            "genome / agent / CONTEXT-in-M this pass",
            "path_and_frontier route discovery",
            "raw TraceSpec as selector input",
            "SHA-as-genome / full-path necessity",
        ],
    }


def write_identity_lock(path: Path = IDENTITY_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_identity_lock(path: Path = IDENTITY_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not path.exists():
        return False, "docs/routesig_identity_012.lock missing; write only via --write-lock", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key, label in (
        ("context_012_lock_sha", "context_012.lock"),
        ("minimap_012_lock_sha", "minimap_012.lock"),
        ("pathdisc_012_lock_sha", "pathdisc_012.lock"),
        ("midpath_012_lock_sha", "midpath_012.lock"),
        ("routesig_012_lock_sha", "routesig_012.lock"),
        ("routesig_depth_012_lock_sha", "routesig_depth_012.lock"),
    ):
        if snap[key] != lock.get(key):
            return False, f"{label} SHA drifted from identity pin", snap
    order = json.loads(ROUTESIG_LOCK.read_text(encoding="utf-8"))
    if snap["gen_c10_sha"] != order.get("gen_c10_sha"):
        return False, "gen_c10 SHA drifted from frozen ORDER lock", snap
    for k in (
        "kappa_seed_sha",
        "kappa_step_sha",
        "route_kappa_sha",
        "edge_fid_sha",
        "path_edge_fids_sha",
    ):
        if snap[k] != order.get(k):
            return False, f"{k} drifted from frozen ORDER lock", snap
    for key in (
        "seed",
        "phase",
        "cells",
        "candidates",
        "candidate_meanings",
        "ksem_fields",
        "ksem_excludes",
        "kappa_reuse",
        "here",
        "support",
        "validation",
        "gen_c10_sha",
        "gen_c12a_sha",
        "gen_c12b_sha",
        "rename_fids_sha",
        "edge_sem_sha",
        "path_edge_sems_sha",
        "extract_identity_sha",
        "score_kappa_pair_sha",
        "assert_c12a_sha",
        "assert_c12b_sha",
        "edge_fid_sha",
        "path_edge_fids_sha",
        "kappa_seed_sha",
        "kappa_step_sha",
        "route_kappa_sha",
        "trace_spec_sha",
        "refuse",
    ):
        if snap[key] != lock.get(key):
            return False, f"identity apparatus drift: {key}", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if lock.get("ksem_fields") != ["bind", "did"]:
        return False, "Ksem must be canonical(bind, did) only", snap
    if _sha_src(TraceSpec) != lock.get("trace_spec_sha"):
        return False, "TraceSpec SHA drifted from identity pin", snap
    for fn in (kappa_seed, kappa_step, route_kappa, edge_sem):
        src = inspect.getsource(fn)
        for banned in ("TraceSpec", "required_motor", "context_expect", "PRESS", "TUNE"):
            if banned in src:
                return False, f"{fn.__name__} source must not reference {banned}", snap
    sem_src = inspect.getsource(edge_sem)
    for banned in ("fid", "here", "support", "role", "init"):
        if banned in sem_src:
            return False, f"edge_sem must not mention {banned}", snap
    if "path_and_frontier" in inspect.getsource(extract_identity):
        return False, "extract_identity must not call path_and_frontier", snap
    return True, "routesig identity apparatus frozen", snap


def run_identity(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok, why, snap = verify_identity_lock()
    if not ok:
        return {"ok": False, "why": why, "earned_next": False, "ex0s_under_test": "0.0.003", "table": {}}
    if seed != snap["seed"]:
        return {
            "ok": False,
            "why": f"runtime seed {seed} != locked seed {snap['seed']}",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": {},
        }

    pairs = {"C12A": gen_c12a(seed), "C12B": gen_c12b(seed)}
    table: dict[str, dict[str, str]] = {c: {} for c in IDENTITY_CANDIDATES}
    details: dict[str, Any] = {}

    for cell_id, pair in pairs.items():
        verrs = validate_pair(pair)
        if verrs:
            return {
                "ok": False,
                "why": f"validation failed {cell_id}: " + "; ".join(verrs[:12]),
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }
        nodes_l = pair.left.traces[0].nodes
        nodes_r = pair.right.traces[0].nodes
        sl = extract_identity(pair.left.relations, nodes_l)
        sr = extract_identity(pair.right.relations, nodes_r)
        outcomes = {cand: score_kappa_pair(sl[cand], sr[cand]) for cand in IDENTITY_CANDIDATES}
        for cand, out in outcomes.items():
            table[cand][cell_id] = out

        # ORDER still holds on both encodings (semantic and fid).
        sl_b = extract_identity(pair.left.relations, pair.left.traces[1].nodes)
        sr_b = extract_identity(pair.right.relations, pair.right.traces[1].nodes)
        if sl["Ksem"] == sl_b["Ksem"] or sr["Ksem"] == sr_b["Ksem"]:
            return {
                "ok": False,
                "why": f"{cell_id} Ksem collapsed C10 route_a vs route_b",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }
        if sl["Kfid"] == sl_b["Kfid"] or sr["Kfid"] == sr_b["Kfid"]:
            return {
                "ok": False,
                "why": f"{cell_id} Kfid collapsed C10 route_a vs route_b",
                "earned_next": False,
                "ex0s_under_test": "0.0.003",
                "table": table,
            }

        for side, cell, nodes in (
            ("left", pair.left, nodes_l),
            ("right", pair.right, nodes_r),
        ):
            shuffled = list(cell.relations)
            random.Random(seed + (0 if side == "left" else 1)).shuffle(shuffled)
            if extract_identity(shuffled, nodes) != extract_identity(cell.relations, nodes):
                return {
                    "ok": False,
                    "why": f"{cell_id} {side} kappa not file-order independent under shuffle",
                    "earned_next": False,
                    "ex0s_under_test": "0.0.003",
                    "table": table,
                }

        details[cell_id] = {
            "outcomes": outcomes,
            "left_s_hash": relations_content_hash(pair.left.relations),
            "right_s_hash": relations_content_hash(pair.right.relations),
            "left_frontier": pair.left.frontier,
            "right_frontier": pair.right.frontier,
            "Kfid_bits": len(sl["Kfid"]) * 4,
            "Ksem_bits": len(sl["Ksem"]) * 4,
        }

    expected = {
        "C12A": {"Kfid": OUTCOME_DIFFERS, "Ksem": OUTCOME_SAME},
        "C12B": {"Kfid": OUTCOME_SAME, "Ksem": OUTCOME_DIFFERS},
    }
    for cell_id, want in expected.items():
        for cand, outcome in want.items():
            if table[cand][cell_id] != outcome:
                return {
                    "ok": False,
                    "why": f"{cell_id}/{cand} expected {outcome} got {table[cand][cell_id]}",
                    "earned_next": False,
                    "ex0s_under_test": "0.0.003",
                    "table": table,
                }

    if any(details[c]["Kfid_bits"] != 256 or details[c]["Ksem_bits"] != 256 for c in CELLS):
        return {
            "ok": False,
            "why": "kappa bit-width drifted from 256",
            "earned_next": False,
            "ex0s_under_test": "0.0.003",
            "table": table,
        }

    claim = (
        "On the frozen C10 witness, opaque fid rename changes Kfid and leaves Ksem "
        "unchanged; rewriting the last hop's (bind, did) while keeping its fid does "
        "the inverse. Same rolling F; the payload is the question. Fid-based κ is "
        "storage-identity dependent; semantic κ tracks directed relation identity "
        "canonical(bind, did). Not: Kfid fails cognition; alpha-rename equivariance; "
        "SHA-as-genome; 0.0.004. Next (not this pass): freeze the κ contract, then "
        "the first CONTEXT function in M."
    )
    return {
        "ok": True,
        "why": why,
        "version": "TM.0.12.ROUTESIG.IDENTITY",
        "phase": "identity",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "seed": seed,
        "table": table,
        "details": details,
        "claim": claim,
        "fid_is_storage_identity": True,
        "sem_is_relation_identity": True,
        "scorer_inputs": "kappa_digests_only",
        "lock": snap,
    }


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012routesig_identity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifacts(summary: dict[str, Any]) -> Path:
    run_dir = _run_dir()
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    (run_dir / "metrics.json").write_text(
        json.dumps(clean, indent=2, default=str) + "\n", encoding="utf-8"
    )
    table = summary.get("table") or {}
    headers = ["Candidate", "C12A fid-rename", "C12B same-fid rewrite"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["-----------"] * len(headers)) + "|",
    ]
    names = {"Kfid": "Kfid (storage row)", "Ksem": "Ksem canonical(bind, did)"}
    for cand in IDENTITY_CANDIDATES:
        row = [names[cand], table.get(cand, {}).get("C12A", "?"), table.get(cand, {}).get("C12B", "?")]
        lines.append("| " + " | ".join(row) + " |")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.ROUTESIG.IDENTITY

Apparatus: {summary.get('why')}
`earned_next`: false (no Ex0S 0.0.004)

{summary.get('claim')}

{chr(10).join(lines)}
""",
        encoding="utf-8",
    )
    summary["run_dir"] = str(run_dir)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.ROUTESIG.IDENTITY")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_identity_lock(), indent=2))
        return
    summary = run_identity(seed=args.seed)
    if summary.get("ok"):
        write_run_artifacts(summary)
    print(
        json.dumps(
            {
                "ok": summary.get("ok"),
                "why": summary.get("why"),
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
