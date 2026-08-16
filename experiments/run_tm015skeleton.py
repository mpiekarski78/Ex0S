"""TM.0.15.SKELETON: observed-transition acquisition.

Apparatus emits a temporal symbol sequence via observe_symbol.
Organism keeps one transient prev and authors experience_skel adjacency.
Compose over organism-authored edges rebuilds κ; ACQUIRE authors experience_ctx.

Not latent relation discovery. No FAMILY. No LOOKAHEAD. earned_next=false. ex0s=null.
Do not rewrite genome_014 / acquire_014 / family_014 / kappa_013 / genome_011.
Prereg: docs/skeleton_015.prereg.lock. Freeze locks after 16/16 only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm014acquire import (
    ACQUIRE_LOCK,
    GENOME_014_LOCK,
    reference_route_kappa,
    teacher_outcome,
    traverse_hold,
    probe_cue,
    verify_genome_014,
)
from experiments.run_tm040 import probe
from three_memory.kappa import CTX_ENCODING
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import TagStore

PREREG_LOCK = REPO_ROOT / "docs" / "skeleton_015.prereg.lock"
SKELETON_LOCK = REPO_ROOT / "docs" / "skeleton_015.lock"
GENOME_015_LOCK = REPO_ROOT / "docs" / "genome_015.lock"
FAMILY_014_LOCK = REPO_ROOT / "docs" / "family_014.lock"
GENOME_011_LOCK = REPO_ROOT / "docs" / "genome_011.lock"
KAPPA_LOCK = REPO_ROOT / "docs" / "kappa_013.lock"

DEFAULT_SEED = 12345
HERE = "chb"
SOURCE_SKEL = "experience_skel"
SOURCE_CTX = "experience_ctx"
MOTORS = frozenset({"press", "tune", "flip", "hold", "idle", "wait"})

TOKENS = {
    "x": "x",
    "a": "a",
    "b": "b",
    "y": "y",
    "press": "press",
    "tune": "tune",
}

PREREGISTERED_CLAIM = (
    "A frozen developmental recipe can convert an observed sequence of relational "
    "transitions into durable skeleton edges in S, then compose over those "
    "organism-authored edges to rebuild κ and author provenance-sensitive contextual "
    "continuations, without the apparatus writing the skeleton or contextual answers into S."
)

CELL_IDS = (
    "D0_birth_unreachable",
    "D1_symbol_life",
    "D2_diamond_coexist",
    "D3_reset_rho",
    "D4_newborn_reload",
    "D5_dual_strip",
    "D6_skel_swap",
    "D7_competing_support",
    "D8_ctx_over_authored",
    "D9_rho_ctx_probe",
    "D10_rename_fid",
    "D11_storage_order",
    "D12_channel_contract",
    "D13_oracle_score_only",
    "D14_weights_no_shortcut",
    "D15_stale_prev",
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


# --- Factory ------------------------------------------------------------------


def make_skeleton(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    pol = policy if policy is not None else UsePolicy(seed=1)
    return make(
        s_dir,
        None,
        pol,
        explore_epsilon=0.0,
        use_context_kappa=True,
        use_acquire_ctx=True,
        use_acquire_skel=True,
        **kwargs,
    )


# --- Birth / clutter (no X→Y path) --------------------------------------------


def empty_birth(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)


def clutter_birth(dest: Path) -> None:
    """Plant clutter that must NOT create an X→Y path."""
    empty_birth(dest)
    rows = [
        ("cl0", "p", "q"),
        ("cl1", "q", "r"),
        ("cl2", "m", "n"),
    ]
    for fid, bind, did in rows:
        tags = {
            "bind": bind,
            "did": did,
            "here": HERE,
            "w0": bind,
            "hyp": "supported",
            "trials": 1,
            "wins": 1,
            "losses": 0,
            "support": 1,
            "contradiction": 0,
            "source": "clutter",
        }
        (dest / f"{fid}.tag").write_text(record_to_tagfile(fid, tags), encoding="utf-8")


# --- Listing / reachability / strip -------------------------------------------


def list_by_source(s_dir: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(s_dir.glob("*.tag")):
        fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") != source:
            continue
        rows.append(
            {
                "fid": fid,
                "bind": str(tags.get("bind") or "").lower(),
                "did": str(tags.get("did") or "").lower(),
                "ctx": tags.get("ctx"),
                "support": int(tags.get("support") or 0),
                "contradiction": int(tags.get("contradiction") or 0),
                "source": source,
            }
        )
    return rows


def list_experience_skel(s_dir: Path) -> list[dict[str, Any]]:
    return list_by_source(s_dir, SOURCE_SKEL)


def list_experience_ctx(s_dir: Path) -> list[dict[str, Any]]:
    return list_by_source(s_dir, SOURCE_CTX)


def graph_edges(s_dir: Path) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        b = str(tags.get("bind") or "").lower()
        d = str(tags.get("did") or "").lower()
        if b and d and d not in MOTORS:
            edges.append((b, d))
    return edges


def can_reach(s_dir: Path, origin: str, goal: str) -> bool:
    """BFS through non-motor bind→did edges."""
    o, g = origin.lower(), goal.lower()
    adj: dict[str, list[str]] = {}
    for b, d in graph_edges(s_dir):
        adj.setdefault(b, []).append(d)
    seen = {o}
    q: deque[str] = deque([o])
    while q:
        cur = q.popleft()
        if cur == g:
            return True
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def clear_by_source(s_dir: Path, source: str) -> int:
    n = 0
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") == source:
            p.unlink()
            n += 1
    return n


def stash_by_source(s_dir: Path, source: str) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") == source:
            out.append((p.name, p.read_bytes()))
    return out


def restore_blobs(s_dir: Path, blobs: Sequence[tuple[str, bytes]]) -> None:
    for name, data in blobs:
        (s_dir / name).write_bytes(data)


def reload_store(ag: Any) -> None:
    if hasattr(ag.store, "reload"):
        ag.store.reload()


def observe_seq(ag: Any, symbols: Sequence[str]) -> list[dict[str, Any]]:
    return [ag.observe_symbol(s) for s in symbols]


def life_route_a(ag: Any) -> None:
    """Author X→A→Y via symbol stream."""
    observe_seq(ag, [TOKENS["x"], TOKENS["a"], TOKENS["y"]])


def life_route_b(ag: Any) -> None:
    observe_seq(ag, [TOKENS["x"], TOKENS["b"], TOKENS["y"]])


def oracle_ka() -> str:
    return reference_route_kappa(
        TOKENS["x"], [(TOKENS["x"], TOKENS["a"]), (TOKENS["a"], TOKENS["y"])]
    )


def oracle_kb() -> str:
    return reference_route_kappa(
        TOKENS["x"], [(TOKENS["x"], TOKENS["b"]), (TOKENS["b"], TOKENS["y"])]
    )


def apparatus_skel_sources(s_dir: Path) -> list[str]:
    """Non-organism relational sources that could fake a map (refuse in D1+)."""
    bad: list[str] = []
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        src = str(tags.get("source") or "")
        if src in ("skeleton", "planted", "apparatus"):
            bad.append(f"{p.name}:{src}")
        # Any non-skel/non-clutter/non-ctx relational edge that completes X→Y is suspect
        # for D1 empty birth: only experience_skel should create the path.
    return bad


def audit_skel_rows(s_dir: Path) -> list[str]:
    errs: list[str] = []
    for r in list_experience_skel(s_dir):
        if r.get("ctx"):
            errs.append(f"skel_has_ctx:{r['fid']}")
        if not r["bind"] or not r["did"]:
            errs.append(f"skel_empty:{r['fid']}")
        if r["did"] in MOTORS:
            errs.append(f"skel_did_motor:{r['fid']}")
        if r["source"] != SOURCE_SKEL:
            errs.append(f"skel_bad_source:{r['fid']}")
    return errs


def audit_ctx_rows(s_dir: Path) -> list[str]:
    errs: list[str] = []
    for r in list_experience_ctx(s_dir):
        if not r.get("ctx"):
            errs.append(f"ctx_missing:{r['fid']}")
        if r["did"] not in ("press", "tune", "flip"):
            errs.append(f"ctx_did_not_motor:{r['fid']}")
        if r["source"] != SOURCE_CTX:
            errs.append(f"ctx_bad_source:{r['fid']}")
    return errs


def ledger_from_dir(s_dir: Path, *, observed_symbols: int = 0) -> dict[str, int]:
    skel = list_experience_skel(s_dir)
    ctx = list_experience_ctx(s_dir)
    unexpected_skel = 0
    unexpected_ctx = 0
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        src = str(tags.get("source") or "")
        has_ctx = isinstance(tags.get("ctx"), str) and bool(tags.get("ctx"))
        if has_ctx and src != SOURCE_CTX:
            unexpected_ctx += 1
        if src == SOURCE_SKEL and has_ctx:
            unexpected_skel += 1
        if src == SOURCE_SKEL and str(tags.get("did") or "").lower() in MOTORS:
            unexpected_skel += 1
    return {
        "observed_symbols": observed_symbols,
        "created_skel_rows": len(skel),
        "updated_skel_rows": sum(max(0, r["support"] - 1) for r in skel),
        "created_ctx_rows": len(ctx),
        "unexpected_skel_writes": unexpected_skel,
        "unexpected_ctx_writes": unexpected_ctx,
    }


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


def _full_learn(s: Path, policy: UsePolicy) -> tuple[Any, dict[str, Any]]:
    """Birth empty → symbols → compose HOLD → teacher PRESS → authored ctx."""
    empty_birth(s)
    ag = make_skeleton(s, policy)
    w0 = ag.weight_hash()
    life_route_a(ag)
    if not can_reach(s, TOKENS["x"], TOKENS["y"]):
        return ag, {"ok": False, "why": "no_reach_after_skel", "w0": w0}
    errs = audit_skel_rows(s)
    if errs:
        return ag, {"ok": False, "why": ";".join(errs), "w0": w0}
    ag.reset_rho()
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending"):
        return ag, {"ok": False, "why": "no_lived", "trav": trav, "w0": w0}
    teacher_outcome(ag, TOKENS["press"], success=True)
    ctx = list_experience_ctx(s)
    if len(ctx) != 1 or ctx[0]["did"] != "press" or ctx[0]["ctx"] != oracle_ka():
        return ag, {"ok": False, "why": "ctx_mismatch", "ctx": ctx, "w0": w0}
    return ag, {"ok": True, "w0": w0, "kappa": oracle_ka()}


# --- Battery ------------------------------------------------------------------


def cell_d0_birth(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d0"
    clutter_birth(s)
    skel = list_experience_skel(s)
    ctx = list_experience_ctx(s)
    reach = can_reach(s, TOKENS["x"], TOKENS["y"])
    ok = len(skel) == 0 and len(ctx) == 0 and not reach
    return _cell(
        "D0_birth_unreachable",
        ok,
        n_skel=len(skel),
        n_ctx=len(ctx),
        can_reach=reach,
    )


def cell_d1_symbol_life(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d1"
    empty_birth(s)
    ag = make_skeleton(s, policy)
    outs = observe_seq(ag, [TOKENS["x"], TOKENS["a"], TOKENS["y"]])
    skel = list_experience_skel(s)
    edges = {(r["bind"], r["did"]) for r in skel}
    expect = {(TOKENS["x"], TOKENS["a"]), (TOKENS["a"], TOKENS["y"])}
    reach = can_reach(s, TOKENS["x"], TOKENS["y"])
    errs = audit_skel_rows(s)
    planted = apparatus_skel_sources(s)
    # Only experience_skel edges may form the graph path (no co-planted skeleton).
    all_edges = set(graph_edges(s))
    ok = (
        edges == expect
        and all_edges == expect
        and reach
        and not errs
        and not planted
        and outs[0].get("wrote") is False
        and outs[1].get("wrote") is True
        and outs[2].get("wrote") is True
        and list_experience_ctx(s) == []
        and ledger_from_dir(s)["unexpected_skel_writes"] == 0
        and ledger_from_dir(s)["unexpected_ctx_writes"] == 0
    )
    return _cell(
        "D1_symbol_life",
        ok,
        edges=sorted(edges),
        can_reach=reach,
        errs=errs,
        planted=planted,
        outs=[{k: o.get(k) for k in ("wrote", "updated", "why")} for o in outs],
    )


def cell_d2_diamond(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d2"
    empty_birth(s)
    ag = make_skeleton(s, policy)
    life_route_a(ag)
    # Clear prev so route B starts cleanly at X (new sequence).
    ag.reset_rho()
    life_route_b(ag)
    edges = {(r["bind"], r["did"]) for r in list_experience_skel(s)}
    expect = {
        (TOKENS["x"], TOKENS["a"]),
        (TOKENS["a"], TOKENS["y"]),
        (TOKENS["x"], TOKENS["b"]),
        (TOKENS["b"], TOKENS["y"]),
    }
    ok = edges == expect and can_reach(s, TOKENS["x"], TOKENS["y"])
    return _cell("D2_diamond_coexist", ok, edges=sorted(edges))


def cell_d3_reset_rho(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d3"
    empty_birth(s)
    ag = make_skeleton(s, policy)
    life_route_a(ag)
    before = list_experience_skel(s)
    ag.reset_rho()
    after = list_experience_skel(s)
    trav = traverse_hold(ag, TOKENS["x"])
    # Without ctx yet, compose should HOLD at Y with lived pending.
    ok = (
        before == after
        and can_reach(s, TOKENS["x"], TOKENS["y"])
        and bool(trav.get("lived_pending"))
        and trav.get("lived_bind") == TOKENS["y"]
    )
    return _cell("D3_reset_rho", ok, lived=trav.get("lived_bind"), hops=trav.get("compose_hops"))


def cell_d4_newborn(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d4"
    empty_birth(s)
    ag = make_skeleton(s, policy)
    life_route_a(ag)
    rows = list_experience_skel(s)
    nb = ag.clone_empty(store_enabled=True)
    nb.store = TagStore(s, enabled=True)
    assert getattr(nb, "use_acquire_skel", False) is True
    trav = traverse_hold(nb, TOKENS["x"])
    ok = (
        len(rows) == 2
        and can_reach(s, TOKENS["x"], TOKENS["y"])
        and bool(trav.get("lived_pending"))
        and getattr(nb, "_skel_prev", "x") is None
    )
    return _cell("D4_newborn_reload", ok, n_skel=len(rows), lived=trav.get("lived_bind"))


def cell_d5_dual_strip(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d5"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D5_dual_strip", False, why=meta.get("why"))
    skel_blobs = stash_by_source(s, SOURCE_SKEL)
    ctx_blobs = stash_by_source(s, SOURCE_CTX)

    # Test A: strip skel, leave ctx → HOLD
    clear_by_source(s, SOURCE_SKEL)
    reload_store(ag)
    ag.reset_rho()
    pa = probe_cue(ag, TOKENS["x"])
    reach_a = can_reach(s, TOKENS["x"], TOKENS["y"])
    hops_a = int(pa.get("compose_hops") or 0)
    ok_a = pa["motor"] == "hold" and not reach_a and hops_a == 0

    # Restore skel, strip ctx → HOLD at Y (compose reaches, no motor)
    restore_blobs(s, skel_blobs)
    clear_by_source(s, SOURCE_CTX)
    reload_store(ag)
    ag.reset_rho()
    pb = probe_cue(ag, TOKENS["x"])
    reach_b = can_reach(s, TOKENS["x"], TOKENS["y"])
    ok_b = pb["motor"] == "hold" and reach_b and int(pb.get("compose_hops") or 0) >= 2

    # Restore both → PRESS
    restore_blobs(s, ctx_blobs)
    reload_store(ag)
    ag.reset_rho()
    pc = probe_cue(ag, TOKENS["x"])
    ok_c = pc["motor"] == "press"

    ok = ok_a and ok_b and ok_c
    return _cell(
        "D5_dual_strip",
        ok,
        test_a_motor=pa["motor"],
        test_b_motor=pb["motor"],
        restore_motor=pc["motor"],
        reach_a=reach_a,
        reach_b=reach_b,
        hops_a=hops_a,
        hops_b=pb.get("compose_hops"),
    )


def cell_d6_skel_swap(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    """Two lives: A-route vs B-route skeletons; swap bytes → behavior follows memory."""
    sa = tmp / "d6a"
    sb = tmp / "d6b"
    empty_birth(sa)
    empty_birth(sb)
    aga = make_skeleton(sa, policy)
    agb = make_skeleton(sb, policy)
    life_route_a(aga)
    life_route_b(agb)
    # Author ctx on each with prefer via evidence: only one route present.
    for ag, s, motor, hops in (
        (aga, sa, "press", [(TOKENS["x"], TOKENS["a"]), (TOKENS["a"], TOKENS["y"])]),
        (agb, sb, "tune", [(TOKENS["x"], TOKENS["b"]), (TOKENS["b"], TOKENS["y"])]),
    ):
        ag.reset_rho()
        trav = traverse_hold(ag, TOKENS["x"])
        if not trav.get("lived_pending"):
            return _cell("D6_skel_swap", False, why=f"no_lived_{motor}")
        teacher_outcome(ag, motor, success=True)

    # Swap only experience_skel bytes between sa and sb; keep each store's ctx.
    skel_a = stash_by_source(sa, SOURCE_SKEL)
    skel_b = stash_by_source(sb, SOURCE_SKEL)
    clear_by_source(sa, SOURCE_SKEL)
    clear_by_source(sb, SOURCE_SKEL)
    restore_blobs(sa, skel_b)
    restore_blobs(sb, skel_a)

    aga2 = make_skeleton(sa, policy)
    agb2 = make_skeleton(sb, policy)
    # After swap: sa has B-route skel + A-ctx (κA). Compose rebuilds κB ≠ κA → HOLD.
    # Assert lived_kappa follows swapped topology (not merely HOLD).
    ta = traverse_hold(aga2, TOKENS["x"])
    tb = traverse_hold(agb2, TOKENS["x"])
    pa = probe_cue(aga2, TOKENS["x"])
    pb = probe_cue(agb2, TOKENS["x"])
    edges_a = {(r["bind"], r["did"]) for r in list_experience_skel(sa)}
    edges_b = {(r["bind"], r["did"]) for r in list_experience_skel(sb)}
    kappa_a_ok = ta.get("lived_kappa") == oracle_kb() and bool(ta.get("lived_pending"))
    kappa_b_ok = tb.get("lived_kappa") == oracle_ka() and bool(tb.get("lived_pending"))
    ok = (
        edges_a == {(TOKENS["x"], TOKENS["b"]), (TOKENS["b"], TOKENS["y"])}
        and edges_b == {(TOKENS["x"], TOKENS["a"]), (TOKENS["a"], TOKENS["y"])}
        and kappa_a_ok
        and kappa_b_ok
        and pa["motor"] == "hold"
        and pb["motor"] == "hold"
    )
    return _cell(
        "D6_skel_swap",
        ok,
        a_motor=pa["motor"],
        b_motor=pb["motor"],
        kappa_a=ta.get("lived_kappa"),
        kappa_b=tb.get("lived_kappa"),
        expect_ka=oracle_ka(),
        expect_kb=oracle_kb(),
        edges_a=sorted(edges_a),
        edges_b=sorted(edges_b),
    )


def cell_d7_competing_support(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d7"
    empty_birth(s)
    ag = make_skeleton(s, policy)
    # X→A twice → support 2; X→B once → support 1; then finish both to Y for compose.
    observe_seq(ag, [TOKENS["x"], TOKENS["a"]])
    ag.reset_rho()
    observe_seq(ag, [TOKENS["x"], TOKENS["a"]])
    ag.reset_rho()
    observe_seq(ag, [TOKENS["x"], TOKENS["b"]])
    # Complete routes to Y (one hop each from mid).
    ag.reset_rho()
    observe_seq(ag, [TOKENS["a"], TOKENS["y"]])
    ag.reset_rho()
    observe_seq(ag, [TOKENS["b"], TOKENS["y"]])

    by = {(r["bind"], r["did"]): r for r in list_experience_skel(s)}
    xa = by.get((TOKENS["x"], TOKENS["a"]))
    xb = by.get((TOKENS["x"], TOKENS["b"]))
    if not xa or not xb:
        return _cell("D7_competing_support", False, why="missing_edges", by=list(by))
    # Unequal: compose chooses A
    ag.reset_rho()
    trav = traverse_hold(ag, TOKENS["x"])
    chose_a = trav.get("lived_kappa") == oracle_ka() and bool(trav.get("lived_pending"))

    # Equalize: bump X→B to support 2
    ag.reset_rho()
    observe_seq(ag, [TOKENS["x"], TOKENS["b"]])
    by2 = {(r["bind"], r["did"]): r for r in list_experience_skel(s)}
    xa2, xb2 = by2[(TOKENS["x"], TOKENS["a"])], by2[(TOKENS["x"], TOKENS["b"])]
    ag.reset_rho()
    trav2 = traverse_hold(ag, TOKENS["x"])
    # Equal hop-0 evidence → evidence_tie HOLD; must not reach lived Y.
    tie_hold = bool(trav2.get("evidence_tie")) and not trav2.get("lived_pending")
    p2 = probe_cue(ag, TOKENS["x"])
    equal_hold = p2["motor"] == "hold" and bool(p2.get("evidence_tie"))

    # No contradiction invented
    no_contra = (
        xa["contradiction"] == 0
        and xb["contradiction"] == 0
        and xa2["contradiction"] == 0
        and xb2["contradiction"] == 0
    )

    ok = (
        xa["support"] == 2
        and xb["support"] == 1
        and chose_a
        and xa2["support"] == 2
        and xb2["support"] == 2
        and tie_hold
        and equal_hold
        and no_contra
    )
    return _cell(
        "D7_competing_support",
        ok,
        xa_support=xa["support"],
        xb_support=xb["support"],
        chose_a=chose_a,
        tie_hold=tie_hold,
        equal_hold=equal_hold,
        evidence_tie=p2.get("evidence_tie"),
        no_contra=no_contra,
    )


def cell_d8_ctx_over_authored(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d8"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D8_ctx_over_authored", False, why=meta.get("why"))
    # No apparatus skeleton source
    planted = [
        p.name
        for p in s.glob("*.tag")
        if str(parse_tagfile(p.read_text(encoding="utf-8"))[1].get("source") or "")
        == "skeleton"
    ]
    ctx = list_experience_ctx(s)
    ok = (
        not planted
        and len(ctx) == 1
        and ctx[0]["ctx"] == oracle_ka()
        and ctx[0]["did"] == "press"
        and can_reach(s, TOKENS["x"], TOKENS["y"])
        and not audit_skel_rows(s)
        and not audit_ctx_rows(s)
    )
    return _cell("D8_ctx_over_authored", ok, kappa=ctx[0]["ctx"] if ctx else None, planted=planted)


def cell_d9_rho_ctx_probe(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d9"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D9_rho_ctx_probe", False, why=meta.get("why"))
    ag.reset_rho()
    p1 = probe_cue(ag, TOKENS["x"])
    nb = ag.clone_empty(store_enabled=True)
    nb.store = TagStore(s, enabled=True)
    p2 = probe_cue(nb, TOKENS["x"])
    ok = p1["motor"] == "press" and p2["motor"] == "press"
    return _cell("D9_rho_ctx_probe", ok, after_rho=p1["motor"], after_newborn=p2["motor"])


def cell_d10_rename_fid(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d10"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D10_rename_fid", False, why=meta.get("why"))
    # Rename all tag files
    for i, p in enumerate(sorted(s.glob("*.tag"))):
        p.rename(s / f"renamed_{i:03d}.tag")
    ag2 = make_skeleton(s, policy)
    p = probe_cue(ag2, TOKENS["x"])
    ok = p["motor"] == "press" and can_reach(s, TOKENS["x"], TOKENS["y"])
    return _cell("D10_rename_fid", ok, motor=p["motor"])


def cell_d11_storage_order(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d11"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D11_storage_order", False, why=meta.get("why"))
    files = list(sorted(s.glob("*.tag")))
    order_before = [p.name for p in files]
    # Lexicographic permutation: rename so sorted(glob) order flips vs original.
    blobs = [(p.name, p.read_bytes()) for p in files]
    for p in files:
        p.unlink()
    new_names: list[str] = []
    for i, (_old, data) in enumerate(blobs):
        # Reverse rank in sort order: first original → zzz..., last → aaa...
        name = f"ord_{len(blobs) - 1 - i:03d}_{_old}"
        (s / name).write_bytes(data)
        new_names.append(name)
    order_after = [p.name for p in sorted(s.glob("*.tag"))]
    order_changed = order_after != sorted(order_before) and order_after == sorted(new_names)
    # Confirm TagStore load order actually differs from original content sequence.
    ag2 = make_skeleton(s, policy)
    loaded = [str(getattr(r, "fact_id", "")) for r in ag2.store.records()]
    # fact_ids come from tag headers, not filenames — order of records follows sorted filenames.
    rec_order = [p.name for p in sorted(s.glob("*.tag"))]
    p = probe_cue(ag2, TOKENS["x"])
    ok = (
        order_changed
        and rec_order == order_after
        and p["motor"] == "press"
        and can_reach(s, TOKENS["x"], TOKENS["y"])
    )
    return _cell(
        "D11_storage_order",
        ok,
        motor=p["motor"],
        order_before=order_before,
        order_after=order_after,
        loaded_fids=loaded,
    )


def cell_d12_channel_contract(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d12"
    empty_birth(s)
    ag = make_skeleton(s, policy)
    has_sym = callable(getattr(ag, "observe_symbol", None))
    has_out = callable(getattr(ag, "observe_outcome", None))
    # Teacher contract: info keys must be exactly {action} (same harden as ACQUIRE).
    src = inspect.getsource(teacher_outcome)
    compact = src.replace(" ", "")
    teacher_ok = (
        'set(info.keys())!={"action"}' in compact
        or 'set(info.keys()) != {"action"}' in src
    ) and "forbidden" in src
    import ast
    import textwrap

    sym_src = textwrap.dedent(inspect.getsource(type(ag).observe_symbol))
    tree = ast.parse(sym_src)
    calls: set[str] = set()

    class _V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            self.generic_visit(node)

    _V().visit(tree)
    channel_sep = "observe_outcome" not in calls

    # Extra keys via observe_outcome must not author experience_skel.
    from three_memory.dial_env import ChannelDialWorld

    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    ag.observe_outcome(obs, True, {"action": "press", "from": "x", "to": "a"})
    skel_empty = list_experience_skel(s) == []

    # Proper symbol channel works
    life_route_a(ag)
    skel_ok = len(list_experience_skel(s)) == 2
    ok = has_sym and has_out and teacher_ok and channel_sep and skel_empty and skel_ok
    return _cell(
        "D12_channel_contract",
        ok,
        has_sym=has_sym,
        teacher_ok=teacher_ok,
        channel_sep=channel_sep,
        skel_empty_before=skel_empty,
        skel_ok=skel_ok,
    )


def cell_d13_oracle_score_only(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d13"
    empty_birth(s)
    before = {p.name: p.read_bytes() for p in s.glob("*.tag")}
    k = oracle_ka()
    after = {p.name: p.read_bytes() for p in s.glob("*.tag")}
    # Oracle must not write
    ok = before == after and isinstance(k, str) and len(k) == 64
    # Also: score-only reference functions have no store.write in source
    src = inspect.getsource(reference_route_kappa)
    ok = ok and "write" not in src.lower()
    return _cell("D13_oracle_score_only", ok, kappa_len=len(k) if k else 0)


def cell_d14_weights_no_shortcut(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d14"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D14_weights_no_shortcut", False, why=meta.get("why"))
    stable = ag.weight_hash() == meta["w0"] and ag.weights_unchanged()
    shortcuts = [
        r
        for r in list_experience_skel(s) + list_experience_ctx(s)
        if r["bind"] == TOKENS["x"] and r["did"] in ("press", "tune")
    ]
    skel_ok = not audit_skel_rows(s)
    ok = stable and not shortcuts and skel_ok
    return _cell(
        "D14_weights_no_shortcut",
        ok,
        weights_stable=stable,
        n_shortcuts=len(shortcuts),
    )


def cell_d15_stale_prev(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    results: dict[str, bool] = {}

    # (1) observe X; reset_rho; observe A → must not author X→A
    s1 = tmp / "d15_1"
    empty_birth(s1)
    ag1 = make_skeleton(s1, policy)
    ag1.observe_symbol(TOKENS["x"])
    ag1.reset_rho()
    ag1.observe_symbol(TOKENS["a"])
    results["stale_prev"] = list_experience_skel(s1) == []

    # (2) no stream → no write (fresh agent, only one symbol)
    s2 = tmp / "d15_2"
    empty_birth(s2)
    ag2 = make_skeleton(s2, policy)
    ag2.observe_symbol(TOKENS["x"])
    results["single_symbol"] = list_experience_skel(s2) == []

    # (3) motor symbol noise does not author skel with motor did
    s3 = tmp / "d15_3"
    empty_birth(s3)
    ag3 = make_skeleton(s3, policy)
    observe_seq(ag3, [TOKENS["x"], TOKENS["press"]])
    results["motor_noise"] = list_experience_skel(s3) == []

    # (4) fail unseen motor after lived → no negative swarm
    s4 = tmp / "d15_4"
    empty_birth(s4)
    ag4 = make_skeleton(s4, policy)
    life_route_a(ag4)
    ag4.reset_rho()
    trav = traverse_hold(ag4, TOKENS["x"])
    if not trav.get("lived_pending"):
        results["unseen_fail"] = False
    else:
        teacher_outcome(ag4, TOKENS["press"], success=False)
        results["unseen_fail"] = list_experience_ctx(s4) == []

    # (5) repeat bumps support not duplicate rows
    s5 = tmp / "d15_5"
    empty_birth(s5)
    ag5 = make_skeleton(s5, policy)
    observe_seq(ag5, [TOKENS["x"], TOKENS["a"]])
    ag5.reset_rho()
    observe_seq(ag5, [TOKENS["x"], TOKENS["a"]])
    rows = list_experience_skel(s5)
    results["repeat"] = len(rows) == 1 and rows[0]["support"] == 2

    ok = all(results.values())
    return _cell("D15_stale_prev", ok, **results)


CELLS: list[Callable[[Path, UsePolicy], dict[str, Any]]] = [
    cell_d0_birth,
    cell_d1_symbol_life,
    cell_d2_diamond,
    cell_d3_reset_rho,
    cell_d4_newborn,
    cell_d5_dual_strip,
    cell_d6_skel_swap,
    cell_d7_competing_support,
    cell_d8_ctx_over_authored,
    cell_d9_rho_ctx_probe,
    cell_d10_rename_fid,
    cell_d11_storage_order,
    cell_d12_channel_contract,
    cell_d13_oracle_score_only,
    cell_d14_weights_no_shortcut,
    cell_d15_stale_prev,
]


# --- Run / locks --------------------------------------------------------------


def run_skeleton(*, seed: int = DEFAULT_SEED, write_locks: bool = False) -> dict[str, Any]:
    policy = UsePolicy(seed=seed)
    rows: list[dict[str, Any]] = []
    unexpected = 0
    with tempfile.TemporaryDirectory(prefix="tm015skel_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            rows.append(fn(root, policy))
        # Audit every cell store left on disk for unexpected provenance.
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            led = ledger_from_dir(d)
            unexpected += led["unexpected_skel_writes"] + led["unexpected_ctx_writes"]
            unexpected += len(audit_skel_rows(d)) + len(audit_ctx_rows(d))
        # Fresh full-learn ledger must also be clean.
        s = root / "_ledger_check"
        _ag, meta = _full_learn(s, policy)
        led = ledger_from_dir(s, observed_symbols=3)
        unexpected += led["unexpected_skel_writes"] + led["unexpected_ctx_writes"]
        if not meta.get("ok"):
            unexpected += 1

    n_ok = sum(1 for r in rows if r.get("ok"))
    summary: dict[str, Any] = {
        "version": "TM.0.15.SKELETON",
        "label": "observed-transition acquisition",
        "ok": n_ok == len(rows) == 16 and unexpected == 0,
        "n_ok": n_ok,
        "n_cells": len(rows),
        "earned_next": False,
        "ex0s": None,
        "seed": seed,
        "ctx_encoding": CTX_ENCODING,
        "claim": PREREGISTERED_CLAIM,
        "unexpected_writes": unexpected,
        "rows": rows,
    }
    if write_locks and summary["ok"]:
        write_genome_015_lock()
        write_skeleton_lock(rows)
    return summary


def verify_prereg_lock(path: Path = PREREG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/skeleton_015.prereg.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False:
        return False, "prereg earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "prereg ex0s must be null", lock
    if lock.get("observation_abi") != "observe_symbol":
        return False, "observation_abi", lock
    if lock.get("preregistered_claim") != PREREGISTERED_CLAIM:
        return False, "claim drift", lock
    if lock.get("cell_ids") != list(CELL_IDS):
        return False, "cell_ids drift", lock
    for key, path_lock in (
        ("genome_014_lock_sha", GENOME_014_LOCK),
        ("acquire_014_lock_sha", ACQUIRE_LOCK),
        ("family_014_lock_sha", FAMILY_014_LOCK),
        ("kappa_013_lock_sha", KAPPA_LOCK),
        ("genome_011_lock_sha", GENOME_011_LOCK),
    ):
        if lock.get(key) != _sha_file(path_lock):
            return False, f"prior lock pin {key}", lock
    # Prereg must not contain final mechanism SHAs
    banned = ("agent_sha", "make_skeleton_sha", "run_tm015skeleton_sha", "cell_shas")
    if any(k in lock for k in banned):
        return False, "prereg contains freeze SHAs", lock
    return True, "skeleton_015.prereg.lock intact", lock


def genome_015_snapshot() -> dict[str, Any]:
    from three_memory import agent as agent_mod

    with tempfile.TemporaryDirectory(prefix="tm015_g_") as tmp:
        s = Path(tmp) / "s"
        s.mkdir()
        ag = make_skeleton(s, UsePolicy(seed=1))
        wh = ag.weight_hash()
    return {
        "version": "TM.0.15.GENOME",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "use_context_kappa": True,
        "use_acquire_ctx": True,
        "use_acquire_skel": True,
        "ctx_encoding": CTX_ENCODING,
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "policy_sha": _sha_file(REPO_ROOT / "three_memory" / "policy.py"),
        "cortex_sha": _sha_file(REPO_ROOT / "three_memory" / "cortex.py"),
        "kappa_sha": _sha_file(REPO_ROOT / "three_memory" / "kappa.py"),
        "make011compose_sha": _sha_bytes(inspect.getsource(make).encode()),
        "make_skeleton_sha": _sha_src(make_skeleton),
        "observe_symbol_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_symbol),
        "run_tm015skeleton_sha": _sha_file(Path(__file__)),
        "cortex_weight_hash": wh,
        "n_feat": int(UsePolicy.n_feat),
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK),
        "acquire_014_lock_sha": _sha_file(ACQUIRE_LOCK),
        "family_014_lock_sha": _sha_file(FAMILY_014_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "clone_empty_copies_skel": "use_acquire_skel=self.use_acquire_skel"
        in inspect.getsource(agent_mod.ThreeMemoryAgent.clone_empty),
    }


def write_genome_015_lock(path: Path = GENOME_015_LOCK) -> dict[str, Any]:
    snap = genome_015_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_genome_015(path: Path = GENOME_015_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = genome_015_snapshot()
    if not path.exists():
        return False, "docs/genome_015.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "agent_sha",
        "policy_sha",
        "cortex_sha",
        "kappa_sha",
        "make011compose_sha",
        "make_skeleton_sha",
        "observe_symbol_sha",
        "cortex_weight_hash",
        "n_feat",
        "use_acquire_skel",
        "use_acquire_ctx",
        "ctx_encoding",
        "genome_014_lock_sha",
        "acquire_014_lock_sha",
        "prereg_lock_sha",
        "earned_next",
        "ex0s",
    ):
        if snap.get(key) != lock.get(key):
            return False, f"genome_015 drift: {key}", snap
    if not snap.get("clone_empty_copies_skel"):
        return False, "clone_empty missing use_acquire_skel", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if lock.get("ex0s") is not None:
        return False, "ex0s must be null", snap
    return True, "genome_015 SKELETON candidate intact", snap


def skeleton_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.15.SKELETON",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "observed-transition acquisition",
        "preregistered_claim": PREREGISTERED_CLAIM,
        "observation_abi": "observe_symbol",
        "ctx_encoding": CTX_ENCODING,
        "seed": DEFAULT_SEED,
        "n_cells": 16,
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "cell_ids": list(CELL_IDS),
        "cell_ok": {r["cell"]: bool(r.get("ok")) for r in rows},
        "make_skeleton_sha": _sha_src(make_skeleton),
        "run_tm015skeleton_sha": _sha_file(Path(__file__)),
        "teacher_outcome_sha": _sha_src(teacher_outcome),
        "genome_015_lock_sha": _sha_file(GENOME_015_LOCK) if GENOME_015_LOCK.exists() else None,
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK),
        "acquire_014_lock_sha": _sha_file(ACQUIRE_LOCK),
        "family_014_lock_sha": _sha_file(FAMILY_014_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "refuse": [
            "pair-event ABI as binding channel (observe_hop)",
            "smuggle skeleton through observe_outcome / teacher info",
            "claim latent relation discovery or organism inferred the map",
            "invent relation contradiction / retraction from X→B",
            "score D0 as target edges absent without X-unreachable-Y",
            "single-strip D5 (one learned layer)",
            "apparatus writing target skeleton or ctx into S",
            "LOOKAHEAD",
            "FAMILY / 288 worlds this pass",
            "stamp or pre-name Ex0S 0.0.005",
            "rewrite genome_014 / acquire_014 / family_014",
        ],
    }


def write_skeleton_lock(
    rows: list[dict[str, Any]], path: Path = SKELETON_LOCK
) -> dict[str, Any]:
    snap = skeleton_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_skeleton_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = SKELETON_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/skeleton_015.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    live = {
        "make_skeleton_sha": _sha_src(make_skeleton),
        "run_tm015skeleton_sha": _sha_file(Path(__file__)),
        "teacher_outcome_sha": _sha_src(teacher_outcome),
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK),
        "observation_abi": "observe_symbol",
        "earned_next": False,
        "ex0s": None,
        "cell_ids": list(CELL_IDS),
        "preregistered_claim": PREREGISTERED_CLAIM,
    }
    for key, val in live.items():
        if lock.get(key) != val:
            return False, f"skeleton lock drift: {key}", lock
    if lock.get("n_ok") != 16 or lock.get("n_cells") != 16:
        return False, "n_ok", lock
    if lock.get("earned_next") is not False:
        return False, "earned_next", lock
    if rows is not None:
        live_ok = {r["cell"]: bool(r.get("ok")) for r in rows}
        if live_ok != lock.get("cell_ok"):
            return False, "cell_ok drift", lock
        if not all(live_ok.get(c) for c in CELL_IDS):
            return False, "cell not all ok", lock
    return True, "skeleton_015.lock intact", lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    args = ap.parse_args()

    if args.verify_prereg:
        ok, why, _ = verify_prereg_lock()
        print(json.dumps({"ok": ok, "why": why}, indent=2))
        sys.exit(0 if ok else 1)

    summary = run_skeleton(seed=args.seed, write_locks=args.write_lock)
    out = _run_dir()
    (out / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    pub = {k: summary[k] for k in summary if k != "rows"}
    print(json.dumps(pub, indent=2))
    for r in summary["rows"]:
        mark = "OK" if r.get("ok") else "FAIL"
        print(f"  {mark} {r['cell']}" + (f" — {r.get('why')}" if not r.get("ok") else ""))
    if not summary["ok"]:
        sys.exit(1)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm015skeleton"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
