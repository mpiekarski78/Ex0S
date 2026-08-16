"""TM.0.16.RELATE: candidate relations under ambiguity.

Apparatus emits multi-symbol events via observe_event(visible, focus).
Organism reads visible only, authors all-pairs experience_skel candidates,
accumulates support across lives, and lets compose pick the unique evidence
winner (HOLD on ties). Losers stay inspectable in S. focus is unused.

Not vision. Not FAMILY. No LOOKAHEAD. earned_next=false. ex0s=null.
Prereg: docs/relate_016.prereg.lock. Freeze locks after 16/16 only.
Do not rewrite genome_015 / skeleton_015 / genome_014 / acquire_014 / family_014.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
import textwrap
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
)
from experiments.run_tm040 import probe
from three_memory.kappa import CTX_ENCODING
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import TagStore

PREREG_LOCK = REPO_ROOT / "docs" / "relate_016.prereg.lock"
RELATE_LOCK = REPO_ROOT / "docs" / "relate_016.lock"
GENOME_016_LOCK = REPO_ROOT / "docs" / "genome_016.lock"
GENOME_015_LOCK = REPO_ROOT / "docs" / "genome_015.lock"
SKELETON_015_LOCK = REPO_ROOT / "docs" / "skeleton_015.lock"
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
    "A frozen developmental recipe can accumulate candidate relations across "
    "repeated ambiguous multi-symbol event streams and use converging evidence "
    "to select an invariant relational route for later composition, while surface "
    "distractor relations remain inspectable competing hypotheses, without the "
    "apparatus choosing which visible-symbol transitions should control behavior."
)

CELL_IDS = (
    "D0_birth_unreachable",
    "D1_ambiguous_exposure_hold",
    "D2_varying_clutter_winner",
    "D3_full_route_resolves",
    "D4_surface_invariant",
    "D5_counterfactual_latent",
    "D6_reset_rho",
    "D7_newborn_reload",
    "D8_dual_strip",
    "D9_focus_not_relation_oracle",
    "D10_irreducible_ambiguity",
    "D11_episode_boundary",
    "D12_fid_storage_order",
    "D13_channel_oracle",
    "D14_weights_no_shortcut",
    "D15_nasty",
)

# Three lives: invariant X→A→Y under varying clutter (plan decisive chain).
LIVES_XA_Y = (
    (["x", "q"], ["a", "q"], ["y", "q"]),
    (["x", "r"], ["a", "r"], ["y", "s"]),
    (["x", "t"], ["a", "u"], ["y", "v"]),
)

LIVES_XB_Y = (
    (["x", "q"], ["b", "q"], ["y", "q"]),
    (["x", "r"], ["b", "r"], ["y", "s"]),
    (["x", "t"], ["b", "u"], ["y", "v"]),
)

# Surface A vs B — same latent X→A→Y, different distractor vocabularies.
LIVES_SURFACE_A = (
    (["x", "red", "cat"], ["a", "blue", "dog"], ["y", "green", "bird"]),
    (["x", "hot", "mug"], ["a", "cold", "cup"], ["y", "warm", "bowl"]),
    (["x", "one", "fox"], ["a", "two", "owl"], ["y", "three", "elk"]),
)

LIVES_SURFACE_B = (
    (["x", "17", "q"], ["a", "92", "r"], ["y", "3", "s"]),
    (["x", "44", "m"], ["a", "81", "n"], ["y", "9", "p"]),
    (["x", "55", "w"], ["a", "66", "z"], ["y", "7", "v"]),
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


# --- Factory ------------------------------------------------------------------


def make_relate(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    pol = policy if policy is not None else UsePolicy(seed=1)
    return make(
        s_dir,
        None,
        pol,
        explore_epsilon=0.0,
        use_context_kappa=True,
        use_acquire_ctx=True,
        use_acquire_skel=True,
        use_acquire_relate=True,
        **kwargs,
    )


# --- Birth / listing ----------------------------------------------------------


def empty_birth(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)


def clutter_birth(dest: Path) -> None:
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


def skel_map(s_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["bind"], r["did"]): r for r in list_experience_skel(s_dir)}


def edge_support(s_dir: Path, bind: str, did: str) -> int:
    r = skel_map(s_dir).get((bind.lower(), did.lower()))
    return int(r["support"]) if r else 0


def evidence_winner_did(s_dir: Path, bind: str) -> str | None:
    """Unique (support, -contradiction) winner among non-motor skel outs from bind."""
    cands = [
        r
        for r in list_experience_skel(s_dir)
        if r["bind"] == bind.lower() and r["did"] not in MOTORS
    ]
    if not cands:
        return None
    best = max((r["support"], -r["contradiction"]) for r in cands)
    winners = [r for r in cands if (r["support"], -r["contradiction"]) == best]
    dids = {r["did"] for r in winners}
    if len(dids) != 1:
        return None
    return winners[0]["did"]


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


def oracle_ka() -> str:
    return reference_route_kappa(
        TOKENS["x"], [(TOKENS["x"], TOKENS["a"]), (TOKENS["a"], TOKENS["y"])]
    )


def oracle_kb() -> str:
    return reference_route_kappa(
        TOKENS["x"], [(TOKENS["x"], TOKENS["b"]), (TOKENS["b"], TOKENS["y"])]
    )


# --- Event lives --------------------------------------------------------------


def observe_event(
    ag: Any,
    visible: Sequence[str],
    focus: Any = None,
) -> dict[str, Any]:
    """Harness may pass focus; organism must ignore it."""
    return ag.observe_event({"visible": list(visible), "focus": focus})


def run_life(ag: Any, events: Sequence[Sequence[str]], *, focus_mode: str = "first") -> None:
    """One episode of multi-symbol events; ends with end_event_episode (not reset_rho)."""
    for vis in events:
        if focus_mode == "first":
            focus = vis[0] if vis else None
        elif focus_mode == "last":
            focus = vis[-1] if vis else None
        elif focus_mode == "none":
            focus = None
        elif focus_mode == "random_token":
            focus = "zzzz_focus_noise"
        else:
            focus = focus_mode
        observe_event(ag, vis, focus=focus)
    ag.end_event_episode()


def run_lives(ag: Any, lives: Sequence[Sequence[Sequence[str]]], *, focus_mode: str = "first") -> None:
    for life in lives:
        run_life(ag, life, focus_mode=focus_mode)


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


def ledger_from_dir(s_dir: Path) -> dict[str, int]:
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
        "created_skel_rows": len(skel),
        "created_ctx_rows": len(ctx),
        "unexpected_skel_writes": unexpected_skel,
        "unexpected_ctx_writes": unexpected_ctx,
    }


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


def _infer_mid(lives: Sequence[Sequence[Sequence[str]]]) -> str:
    """Latent mid token from first life's mid-event (a or b)."""
    if lives and lives[0] and len(lives[0]) >= 2:
        mid_vis = {t.lower() for t in lives[0][1]}
        if TOKENS["b"] in mid_vis and TOKENS["a"] not in mid_vis:
            return TOKENS["b"]
        if TOKENS["a"] in mid_vis:
            return TOKENS["a"]
    return TOKENS["a"]


def _full_learn(
    s: Path,
    policy: UsePolicy,
    lives: Sequence[Sequence[Sequence[str]]] = LIVES_XA_Y,
    motor: str = "press",
) -> tuple[Any, dict[str, Any]]:
    """Ambiguous lives → unique winners → compose HOLD → teacher → ctx."""
    empty_birth(s)
    ag = make_relate(s, policy)
    w0 = ag.weight_hash()
    run_lives(ag, lives)
    mid = _infer_mid(lives)

    wx = evidence_winner_did(s, TOKENS["x"])
    wm = evidence_winner_did(s, mid)
    if wx != mid or wm != TOKENS["y"]:
        return ag, {
            "ok": False,
            "why": f"winners_x={wx}_mid={wm}_expect_mid={mid}",
            "w0": w0,
            "mid": mid,
        }
    errs = audit_skel_rows(s)
    if errs:
        return ag, {"ok": False, "why": ";".join(errs), "w0": w0, "mid": mid}
    ag.reset_rho()
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending") or str(trav.get("lived_bind") or "").lower() != TOKENS["y"]:
        return ag, {"ok": False, "why": "no_lived_y", "trav": trav, "w0": w0, "mid": mid}
    teacher_outcome(ag, motor, success=True)
    expect_k = oracle_ka() if mid == TOKENS["a"] else oracle_kb()
    ctx = list_experience_ctx(s)
    if len(ctx) != 1 or ctx[0]["did"] != motor or ctx[0]["ctx"] != expect_k:
        return ag, {"ok": False, "why": "ctx_mismatch", "ctx": ctx, "w0": w0, "mid": mid}
    return ag, {"ok": True, "w0": w0, "kappa": expect_k, "mid": mid, "motor": motor}


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


def cell_d1_ambiguous_exposure_hold(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d1"
    empty_birth(s)
    ag = make_relate(s, policy)
    run_life(ag, LIVES_XA_Y[0])
    skel = list_experience_skel(s)
    edges = {(r["bind"], r["did"]) for r in skel}
    # Candidate cloud must include competitors
    cloud_ok = ("x", "a") in edges and ("x", "q") in edges and ("a", "y") in edges and ("q", "y") in edges
    # Self-pair retained
    self_pair = ("q", "q") in edges
    sx_a = edge_support(s, "x", "a")
    sx_q = edge_support(s, "x", "q")
    equal = sx_a == sx_q == 1
    ag.reset_rho()
    trav = traverse_hold(ag, TOKENS["x"])
    hold_tie = bool(trav.get("compose_hold")) and bool(trav.get("evidence_tie"))
    no_unique = evidence_winner_did(s, "x") is None
    ok = cloud_ok and self_pair and equal and hold_tie and no_unique and len(skel) >= 4
    return _cell(
        "D1_ambiguous_exposure_hold",
        ok,
        n_skel=len(skel),
        support_xa=sx_a,
        support_xq=sx_q,
        evidence_tie=trav.get("evidence_tie"),
        compose_hold=trav.get("compose_hold"),
        self_pair=self_pair,
    )


def cell_d2_varying_clutter_winner(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d2"
    empty_birth(s)
    ag = make_relate(s, policy)
    run_lives(ag, LIVES_XA_Y)
    sm = skel_map(s)
    # Actual clutter outs from LIVES_XA_Y (x→t is never authored: t co-occurs with x).
    xa = sm.get(("x", "a"))
    loser_keys = (("x", "q"), ("x", "r"), ("x", "u"))
    losers = [k for k in loser_keys if k in sm]
    winner_ok = xa is not None and evidence_winner_did(s, "x") == "a"
    loser_ok = (
        losers == list(loser_keys)
        and all(sm[k]["support"] < xa["support"] for k in losers)
        and all(sm[k]["support"] >= 1 for k in losers)
    )
    ay = sm.get(("a", "y"))
    ay_ok = ay is not None and evidence_winner_did(s, "a") == "y"
    # Losers must remain inspectable — not pruned after winner emerges.
    pruned = any(k not in sm for k in loser_keys)
    ag.reset_rho()
    trav = traverse_hold(ag, TOKENS["x"])
    compose_ok = (
        bool(trav.get("lived_pending"))
        and str(trav.get("lived_bind") or "").lower() == "y"
        and not trav.get("evidence_tie")
        and int(trav.get("compose_hops") or 0) >= 2
    )
    ok = winner_ok and loser_ok and ay_ok and compose_ok and not pruned
    return _cell(
        "D2_varying_clutter_winner",
        ok,
        support_xa=xa["support"] if xa else 0,
        losers=sorted(f"{a}->{b}:{sm[a, b]['support']}" for a, b in losers),
        lived_bind=trav.get("lived_bind"),
        hops=trav.get("compose_hops"),
        pruned=pruned,
    )


def cell_d3_full_route_resolves(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d3"
    ag, meta = _full_learn(s, policy, LIVES_XA_Y)
    if not meta.get("ok"):
        return _cell("D3_full_route_resolves", False, why=meta.get("why"))
    # Losers still in S after full route
    sm = skel_map(s)
    losers_remain = ("x", "q") in sm and ("x", "r") in sm
    ag.reset_rho()
    p = probe_cue(ag, TOKENS["x"])
    ok = (
        p["motor"] == "press"
        and meta.get("kappa") == oracle_ka()
        and losers_remain
        and evidence_winner_did(s, "x") == "a"
        and evidence_winner_did(s, "a") == "y"
    )
    return _cell(
        "D3_full_route_resolves",
        ok,
        motor=p["motor"],
        kappa=meta.get("kappa"),
        losers_remain=losers_remain,
    )


def cell_d4_surface_invariant(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    sa = tmp / "d4a"
    sb = tmp / "d4b"
    aga, meta_a = _full_learn(sa, policy, LIVES_SURFACE_A)
    agb, meta_b = _full_learn(sb, policy, LIVES_SURFACE_B)
    if not meta_a.get("ok") or not meta_b.get("ok"):
        return _cell(
            "D4_surface_invariant",
            False,
            why=f"a={meta_a.get('why')};b={meta_b.get('why')}",
        )
    aga.reset_rho()
    agb.reset_rho()
    pa = probe_cue(aga, TOKENS["x"])
    pb = probe_cue(agb, TOKENS["x"])
    # Same operational winners and same κ/behavior despite different surfaces
    same_winners = (
        evidence_winner_did(sa, "x") == "a"
        and evidence_winner_did(sb, "x") == "a"
        and evidence_winner_did(sa, "a") == "y"
        and evidence_winner_did(sb, "a") == "y"
    )
    ok = (
        same_winners
        and pa["motor"] == pb["motor"] == "press"
        and meta_a["kappa"] == meta_b["kappa"] == oracle_ka()
    )
    return _cell(
        "D4_surface_invariant",
        ok,
        motor_a=pa["motor"],
        motor_b=pb["motor"],
        kappa_a=meta_a["kappa"],
        kappa_b=meta_b["kappa"],
    )


def cell_d5_counterfactual_latent(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    sa = tmp / "d5a"
    sb = tmp / "d5b"
    aga, meta_a = _full_learn(sa, policy, LIVES_XA_Y, motor="press")
    agb, meta_b = _full_learn(sb, policy, LIVES_XB_Y, motor="tune")
    if not meta_a.get("ok") or not meta_b.get("ok"):
        return _cell(
            "D5_counterfactual_latent",
            False,
            why=f"a={meta_a.get('why')};b={meta_b.get('why')}",
        )
    aga.reset_rho()
    agb.reset_rho()
    pa = probe_cue(aga, TOKENS["x"])
    pb = probe_cue(agb, TOKENS["x"])
    ok = (
        pa["motor"] == "press"
        and pb["motor"] == "tune"
        and meta_a["kappa"] == oracle_ka()
        and meta_b["kappa"] == oracle_kb()
        and meta_a["kappa"] != meta_b["kappa"]
        and evidence_winner_did(sa, "x") == "a"
        and evidence_winner_did(sb, "x") == "b"
    )
    return _cell(
        "D5_counterfactual_latent",
        ok,
        motor_a=pa["motor"],
        motor_b=pb["motor"],
        kappa_a=meta_a["kappa"],
        kappa_b=meta_b["kappa"],
    )


def cell_d6_reset_rho(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d6"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D6_reset_rho", False, why=meta.get("why"))
    before = list_experience_skel(s)
    ag.reset_rho()
    after = list_experience_skel(s)
    p = probe_cue(ag, TOKENS["x"])
    ok = (
        len(before) == len(after)
        and p["motor"] == "press"
        and getattr(ag, "_rel_prev_visible", "x") is None
    )
    return _cell("D6_reset_rho", ok, motor=p["motor"], n_skel=len(after))


def cell_d7_newborn_reload(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d7"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D7_newborn_reload", False, why=meta.get("why"))
    nb = ag.clone_empty(store_enabled=True)
    nb.store = TagStore(s, enabled=True)
    assert nb.use_acquire_relate is True
    p = probe_cue(nb, TOKENS["x"])
    ok = (
        p["motor"] == "press"
        and evidence_winner_did(s, "x") == "a"
        and getattr(nb, "_rel_prev_visible", "x") is None
    )
    return _cell("D7_newborn_reload", ok, motor=p["motor"])


def cell_d8_dual_strip(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d8"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D8_dual_strip", False, why=meta.get("why"))
    skel_blobs = stash_by_source(s, SOURCE_SKEL)
    ctx_blobs = stash_by_source(s, SOURCE_CTX)

    clear_by_source(s, SOURCE_SKEL)
    reload_store(ag)
    ag.reset_rho()
    pa = probe_cue(ag, TOKENS["x"])
    reach_a = can_reach(s, TOKENS["x"], TOKENS["y"])
    ok_a = pa["motor"] == "hold" and not reach_a

    restore_blobs(s, skel_blobs)
    clear_by_source(s, SOURCE_CTX)
    reload_store(ag)
    ag.reset_rho()
    pb = probe_cue(ag, TOKENS["x"])
    reach_b = can_reach(s, TOKENS["x"], TOKENS["y"])
    ok_b = pb["motor"] == "hold" and reach_b and int(pb.get("compose_hops") or 0) >= 2

    restore_blobs(s, ctx_blobs)
    reload_store(ag)
    ag.reset_rho()
    pc = probe_cue(ag, TOKENS["x"])
    ok_c = pc["motor"] == "press"

    ok = ok_a and ok_b and ok_c
    return _cell(
        "D8_dual_strip",
        ok,
        test_a_motor=pa["motor"],
        test_b_motor=pb["motor"],
        restore_motor=pc["motor"],
        reach_a=reach_a,
        reach_b=reach_b,
    )


def _focus_stream_xay(_life_i: int, ev_i: int, _vis: Sequence[str]) -> str:
    return ("x", "a", "y")[ev_i]


def _focus_stream_qrs(life_i: int, ev_i: int, _vis: Sequence[str]) -> str:
    """Clutter-aligned focus (plan QRS) — must not change authored relations."""
    streams = (
        ("q", "q", "q"),
        ("r", "r", "s"),
        ("t", "u", "v"),
    )
    return streams[life_i][ev_i]


def _focus_stream_random(life_i: int, ev_i: int, _vis: Sequence[str]) -> str:
    return f"zzzz_focus_{life_i}_{ev_i}"


def cell_d9_focus_not_relation_oracle(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    """Same visibles; focus XAY / QRS / random → identical skel + compose."""
    streams = (
        ("XAY", _focus_stream_xay),
        ("QRS", _focus_stream_qrs),
        ("random", _focus_stream_random),
    )
    snaps: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for name, foc_fn in streams:
        s = tmp / f"d9_{name}"
        empty_birth(s)
        ag = make_relate(s, policy)
        for life_i, life in enumerate(LIVES_XA_Y):
            for ev_i, vis in enumerate(life):
                observe_event(ag, vis, focus=foc_fn(life_i, ev_i, vis))
            ag.end_event_episode()
        rows = sorted(
            (r["bind"], r["did"], r["support"], r["contradiction"])
            for r in list_experience_skel(s)
        )
        snaps.append({"rows": rows, "winner_x": evidence_winner_did(s, "x")})
        ag.reset_rho()
        trav = traverse_hold(ag, TOKENS["x"])
        probes.append(
            {
                "lived_bind": trav.get("lived_bind"),
                "lived_kappa": trav.get("lived_kappa"),
                "hops": trav.get("compose_hops"),
                "tie": trav.get("evidence_tie"),
            }
        )

    # Focus-as-oracle kill: clutter visibles + path focus must not author X→A→Y.
    s_leak = tmp / "d9_leak"
    empty_birth(s_leak)
    ag_leak = make_relate(s_leak, policy)
    for _ in range(3):
        observe_event(ag_leak, ["q", "r"], focus="x")
        observe_event(ag_leak, ["s", "t"], focus="a")
        observe_event(ag_leak, ["u", "v"], focus="y")
        ag_leak.end_event_episode()
    leak_ok = edge_support(s_leak, "x", "a") == 0 and edge_support(s_leak, "a", "y") == 0
    clutter_votes = edge_support(s_leak, "q", "s") == 3

    identical_s = all(snaps[0] == sn for sn in snaps[1:])
    identical_b = all(probes[0] == pr for pr in probes[1:])
    ok = (
        identical_s
        and identical_b
        and snaps[0]["winner_x"] == "a"
        and leak_ok
        and clutter_votes
    )
    return _cell(
        "D9_focus_not_relation_oracle",
        ok,
        identical_skel=identical_s,
        identical_compose=identical_b,
        n_modes=len(streams),
        winner=snaps[0]["winner_x"] if snaps else None,
        focus_leak_blocked=leak_ok,
        clutter_votes=clutter_votes,
    )


def cell_d10_irreducible_ambiguity(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d10"
    empty_birth(s)
    ag = make_relate(s, policy)
    # Repeated equal competition: {X}→{A,Q}→{Y}
    for _ in range(3):
        run_life(ag, (["x"], ["a", "q"], ["y"]))
    sx_a = edge_support(s, "x", "a")
    sx_q = edge_support(s, "x", "q")
    equal = sx_a == sx_q == 3
    no_winner = evidence_winner_did(s, "x") is None
    ag.reset_rho()
    trav = traverse_hold(ag, TOKENS["x"])
    ok = (
        equal
        and no_winner
        and bool(trav.get("evidence_tie"))
        and bool(trav.get("compose_hold"))
        and not trav.get("lived_pending")
    )
    return _cell(
        "D10_irreducible_ambiguity",
        ok,
        support_xa=sx_a,
        support_xq=sx_q,
        evidence_tie=trav.get("evidence_tie"),
    )


def cell_d11_episode_boundary(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    results: dict[str, bool] = {}

    # (1) Without end_event_episode: Y→X seam across lives
    s1 = tmp / "d11_seam"
    empty_birth(s1)
    ag1 = make_relate(s1, policy)
    for vis in LIVES_XA_Y[0]:
        observe_event(ag1, vis, focus=vis[0])
    # No end_event_episode — start next life while prev=Y
    for vis in LIVES_XA_Y[1]:
        observe_event(ag1, vis, focus=vis[0])
    results["seam_without_boundary"] = edge_support(s1, "y", "x") >= 1

    # (2) With end_event_episode: no Y→X
    s2 = tmp / "d11_ok"
    empty_birth(s2)
    ag2 = make_relate(s2, policy)
    run_lives(ag2, LIVES_XA_Y[:2])
    results["no_seam_with_boundary"] = edge_support(s2, "y", "x") == 0

    # (3) reset_rho clears transient (no setattr)
    s3 = tmp / "d11_rho"
    empty_birth(s3)
    ag3 = make_relate(s3, policy)
    observe_event(ag3, ["x"], focus="x")
    ag3.reset_rho()
    observe_event(ag3, ["a"], focus="a")
    results["reset_rho_clears"] = (
        edge_support(s3, "x", "a") == 0 and getattr(ag3, "_rel_prev_visible", None) == ["a"]
    )

    # (4) newborn clears transient
    s4 = tmp / "d11_nb"
    empty_birth(s4)
    ag4 = make_relate(s4, policy)
    observe_event(ag4, ["x"], focus="x")
    nb = ag4.clone_empty(store_enabled=True)
    nb.store = TagStore(s4, enabled=True)
    observe_event(nb, ["a"], focus="a")
    results["newborn_clears"] = (
        edge_support(s4, "x", "a") == 0 and getattr(nb, "_rel_prev_visible", None) == ["a"]
    )

    # (5) Accumulation does not require reset_rho between lives
    s5 = tmp / "d11_acc"
    empty_birth(s5)
    ag5 = make_relate(s5, policy)
    run_lives(ag5, LIVES_XA_Y)  # only end_event_episode between lives
    results["accumulate_without_rho"] = (
        edge_support(s5, "x", "a") == 3 and evidence_winner_did(s5, "x") == "a"
    )

    ok = all(results.values())
    return _cell("D11_episode_boundary", ok, **results)


def cell_d12_fid_storage_order(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d12"
    ag, meta = _full_learn(s, policy)
    if not meta.get("ok"):
        return _cell("D12_fid_storage_order", False, why=meta.get("why"))

    # Rename all files
    for i, p in enumerate(sorted(s.glob("*.tag"))):
        p.rename(s / f"renamed_{i:03d}.tag")
    ag_r = make_relate(s, policy)
    pr = probe_cue(ag_r, TOKENS["x"])
    rename_ok = pr["motor"] == "press"

    # Permute storage order
    files = list(sorted(s.glob("*.tag")))
    blobs = [(p.name, p.read_bytes()) for p in files]
    for p in files:
        p.unlink()
    for i, (_old, data) in enumerate(blobs):
        name = f"ord_{len(blobs) - 1 - i:03d}_{_old}"
        (s / name).write_bytes(data)
    ag_o = make_relate(s, policy)
    po = probe_cue(ag_o, TOKENS["x"])
    order_ok = po["motor"] == "press"
    ok = rename_ok and order_ok
    return _cell(
        "D12_fid_storage_order",
        ok,
        rename_motor=pr["motor"],
        order_motor=po["motor"],
    )


def cell_d13_channel_oracle(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d13"
    empty_birth(s)
    ag = make_relate(s, policy)
    has_ev = callable(getattr(ag, "observe_event", None))
    has_end = callable(getattr(ag, "end_event_episode", None))
    has_out = callable(getattr(ag, "observe_outcome", None))

    src = inspect.getsource(teacher_outcome)
    compact = src.replace(" ", "")
    teacher_ok = (
        'set(info.keys())!={"action"}' in compact
        or 'set(info.keys()) != {"action"}' in src
    ) and "forbidden" in src

    # observe_event must not read focus for authoring
    ev_src = textwrap.dedent(inspect.getsource(type(ag).observe_event))
    tree = ast.parse(ev_src)
    focus_reads: list[str] = []

    class _FocusV(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            # event["focus"] / event['focus']
            if isinstance(node.value, ast.Name) and node.value.id == "event":
                sl = node.slice
                if isinstance(sl, ast.Constant) and sl.value == "focus":
                    focus_reads.append("subscript")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # event.get("focus", ...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "event"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "focus"
            ):
                focus_reads.append("get")
            self.generic_visit(node)

    _FocusV().visit(tree)
    focus_ignored = len(focus_reads) == 0

    # Channel separation: observe_event must not call observe_outcome
    calls: set[str] = set()

    class _CallV(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            self.generic_visit(node)

    _CallV().visit(tree)
    channel_sep = "observe_outcome" not in calls

    # Extra keys via observe_outcome must not author experience_skel
    from three_memory.dial_env import ChannelDialWorld

    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    ag.observe_outcome(obs, True, {"action": "press", "from": "x", "to": "a", "path": "xay"})
    skel_empty = list_experience_skel(s) == []

    # Proper event channel works
    run_life(ag, LIVES_XA_Y[0])
    skel_ok = len(list_experience_skel(s)) >= 4

    # Oracle score-only
    before = {p.name: p.read_bytes() for p in s.glob("*.tag")}
    k = oracle_ka()
    after = {p.name: p.read_bytes() for p in s.glob("*.tag")}
    oracle_ok = before == after and "write" not in inspect.getsource(reference_route_kappa).lower()

    ok = (
        has_ev
        and has_end
        and has_out
        and teacher_ok
        and focus_ignored
        and channel_sep
        and skel_empty
        and skel_ok
        and oracle_ok
        and isinstance(k, str)
        and len(k) == 64
    )
    return _cell(
        "D13_channel_oracle",
        ok,
        focus_ignored=focus_ignored,
        channel_sep=channel_sep,
        teacher_ok=teacher_ok,
        skel_empty_before=skel_empty,
        oracle_ok=oracle_ok,
    )


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


def cell_d15_nasty(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    results: dict[str, bool] = {}

    # Empty visible → no write, prev unchanged
    s0 = tmp / "d15_empty"
    empty_birth(s0)
    ag0 = make_relate(s0, policy)
    observe_event(ag0, ["x"], focus="x")
    out_empty = observe_event(ag0, [], focus="x")
    results["empty_no_write"] = (
        out_empty.get("why") == "empty_visible"
        and list_experience_skel(s0) == []
        and getattr(ag0, "_rel_prev_visible", None) == ["x"]
    )

    # Duplicate / case / whitespace normalize + motors skipped
    s1 = tmp / "d15_norm"
    empty_birth(s1)
    ag1 = make_relate(s1, policy)
    observe_event(ag1, ["  X ", "x", "X", "press"], focus="PRESS")
    observe_event(ag1, [" A ", "a", "tune"], focus="a")
    sm = skel_map(s1)
    results["normalize_dedupe"] = (
        set(sm.keys()) == {("x", "a")}
        and edge_support(s1, "x", "a") == 1
        and ("x", "press") not in sm
        and ("press", "a") not in sm
    )

    # Stale after ρ
    s2 = tmp / "d15_stale"
    empty_birth(s2)
    ag2 = make_relate(s2, policy)
    observe_event(ag2, ["x"], focus="x")
    ag2.reset_rho()
    observe_event(ag2, ["a"], focus="a")
    results["stale_after_rho"] = list_experience_skel(s2) == []

    # Full learn provenance clean
    s3 = tmp / "d15_prov"
    ag3, meta = _full_learn(s3, policy)
    led = ledger_from_dir(s3)
    results["provenance_clean"] = (
        bool(meta.get("ok"))
        and led["unexpected_skel_writes"] == 0
        and led["unexpected_ctx_writes"] == 0
        and not audit_skel_rows(s3)
        and not audit_ctx_rows(s3)
    )

    # relate off by default on plain make
    s4 = tmp / "d15_off"
    empty_birth(s4)
    ag4 = make(
        s4,
        None,
        policy,
        explore_epsilon=0.0,
        use_context_kappa=True,
        use_acquire_ctx=True,
        use_acquire_skel=True,
    )
    out_off = ag4.observe_event({"visible": ["x"], "focus": "x"})
    results["relate_off_default"] = (
        getattr(ag4, "use_acquire_relate", False) is False
        and out_off.get("why") == "relate_off"
    )

    # Legal N=1 teach: {X}→{A} may author X→A support 1 (not a universal one-obs ban)
    s_n1 = tmp / "d15_n1"
    empty_birth(s_n1)
    ag_n1 = make_relate(s_n1, policy)
    observe_event(ag_n1, ["x"], focus="x")
    observe_event(ag_n1, ["a"], focus="a")
    results["n1_legal_teach"] = edge_support(s_n1, "x", "a") == 1

    # Harness must not assign the RELATE transient (API only: end_event_episode / reset_rho).
    harness_src = Path(__file__).read_text(encoding="utf-8")
    assign_hits: list[str] = []
    tree = ast.parse(harness_src)

    class _AssignV(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "_rel_prev_visible":
                    assign_hits.append(ast.dump(t))
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            t = node.target
            if isinstance(t, ast.Attribute) and t.attr == "_rel_prev_visible":
                assign_hits.append(ast.dump(t))
            self.generic_visit(node)

    _AssignV().visit(tree)
    results["no_harness_setattr"] = assign_hits == []

    ok = all(results.values())
    return _cell("D15_nasty", ok, **results)


CELLS: Sequence[Callable[[Path, UsePolicy], dict[str, Any]]] = (
    cell_d0_birth,
    cell_d1_ambiguous_exposure_hold,
    cell_d2_varying_clutter_winner,
    cell_d3_full_route_resolves,
    cell_d4_surface_invariant,
    cell_d5_counterfactual_latent,
    cell_d6_reset_rho,
    cell_d7_newborn_reload,
    cell_d8_dual_strip,
    cell_d9_focus_not_relation_oracle,
    cell_d10_irreducible_ambiguity,
    cell_d11_episode_boundary,
    cell_d12_fid_storage_order,
    cell_d13_channel_oracle,
    cell_d14_weights_no_shortcut,
    cell_d15_nasty,
)


# --- Run / locks --------------------------------------------------------------


def run_relate(*, seed: int = DEFAULT_SEED, write_locks: bool = False) -> dict[str, Any]:
    policy = UsePolicy(seed=seed)
    rows: list[dict[str, Any]] = []
    unexpected = 0
    with tempfile.TemporaryDirectory(prefix="tm016rel_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            rows.append(fn(root, policy))
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            led = ledger_from_dir(d)
            unexpected += led["unexpected_skel_writes"] + led["unexpected_ctx_writes"]
            unexpected += len(audit_skel_rows(d)) + len(audit_ctx_rows(d))
        s = root / "_ledger_check"
        _ag, meta = _full_learn(s, policy)
        led = ledger_from_dir(s)
        unexpected += led["unexpected_skel_writes"] + led["unexpected_ctx_writes"]
        if not meta.get("ok"):
            unexpected += 1

    n_ok = sum(1 for r in rows if r.get("ok"))
    summary: dict[str, Any] = {
        "version": "TM.0.16.RELATE",
        "label": "candidate relations under ambiguity",
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
        write_genome_016_lock()
        write_relate_lock(rows)
    return summary


def verify_prereg_lock(path: Path = PREREG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/relate_016.prereg.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False:
        return False, "prereg earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "prereg ex0s must be null", lock
    if lock.get("observation_abi") != "observe_event":
        return False, "observation_abi", lock
    if lock.get("episode_boundary_api") != "end_event_episode":
        return False, "episode_boundary_api", lock
    if lock.get("preregistered_claim") != PREREGISTERED_CLAIM:
        return False, "claim drift", lock
    if lock.get("cell_ids") != list(CELL_IDS):
        return False, "cell_ids drift", lock
    cand = lock.get("candidate_rule") or {}
    if cand.get("reads_focus") is not False:
        return False, "candidate_rule.reads_focus must be false", lock
    if cand.get("prune_losers") is not False:
        return False, "candidate_rule.prune_losers must be false", lock
    for key, path_lock in (
        ("genome_015_lock_sha", GENOME_015_LOCK),
        ("skeleton_015_lock_sha", SKELETON_015_LOCK),
        ("genome_014_lock_sha", GENOME_014_LOCK),
        ("acquire_014_lock_sha", ACQUIRE_LOCK),
        ("family_014_lock_sha", FAMILY_014_LOCK),
        ("kappa_013_lock_sha", KAPPA_LOCK),
        ("genome_011_lock_sha", GENOME_011_LOCK),
    ):
        if lock.get(key) != _sha_file(path_lock):
            return False, f"prior lock pin {key}", lock
    banned = (
        "agent_sha",
        "make_relate_sha",
        "run_tm016relate_sha",
        "observe_event_sha",
        "cell_shas",
    )
    if any(k in lock for k in banned):
        return False, "prereg contains freeze SHAs", lock
    return True, "relate_016.prereg.lock intact", lock


def genome_016_snapshot() -> dict[str, Any]:
    from three_memory import agent as agent_mod

    with tempfile.TemporaryDirectory(prefix="tm016_g_") as tmp:
        s = Path(tmp) / "s"
        s.mkdir()
        ag = make_relate(s, UsePolicy(seed=1))
        wh = ag.weight_hash()
    return {
        "version": "TM.0.16.GENOME",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "use_context_kappa": True,
        "use_acquire_ctx": True,
        "use_acquire_skel": True,
        "use_acquire_relate": True,
        "ctx_encoding": CTX_ENCODING,
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "policy_sha": _sha_file(REPO_ROOT / "three_memory" / "policy.py"),
        "cortex_sha": _sha_file(REPO_ROOT / "three_memory" / "cortex.py"),
        "kappa_sha": _sha_file(REPO_ROOT / "three_memory" / "kappa.py"),
        "make011compose_sha": _sha_bytes(inspect.getsource(make).encode()),
        "make_relate_sha": _sha_src(make_relate),
        "observe_event_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_event),
        "end_event_episode_sha": _sha_src(agent_mod.ThreeMemoryAgent.end_event_episode),
        "run_tm016relate_sha": _sha_file(Path(__file__)),
        "cortex_weight_hash": wh,
        "n_feat": int(UsePolicy.n_feat),
        "genome_015_lock_sha": _sha_file(GENOME_015_LOCK),
        "skeleton_015_lock_sha": _sha_file(SKELETON_015_LOCK),
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK),
        "acquire_014_lock_sha": _sha_file(ACQUIRE_LOCK),
        "family_014_lock_sha": _sha_file(FAMILY_014_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "clone_empty_copies_relate": "use_acquire_relate=self.use_acquire_relate"
        in inspect.getsource(agent_mod.ThreeMemoryAgent.clone_empty),
        "observation_abi": "observe_event",
        "episode_boundary_api": "end_event_episode",
        "reads_focus": False,
        "prune_losers": False,
    }


def write_genome_016_lock(path: Path = GENOME_016_LOCK) -> dict[str, Any]:
    snap = genome_016_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_genome_016(path: Path = GENOME_016_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = genome_016_snapshot()
    if not path.exists():
        return False, "docs/genome_016.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "agent_sha",
        "policy_sha",
        "cortex_sha",
        "kappa_sha",
        "make011compose_sha",
        "make_relate_sha",
        "observe_event_sha",
        "end_event_episode_sha",
        "run_tm016relate_sha",
        "cortex_weight_hash",
        "n_feat",
        "use_acquire_relate",
        "use_acquire_skel",
        "use_acquire_ctx",
        "ctx_encoding",
        "genome_015_lock_sha",
        "skeleton_015_lock_sha",
        "genome_014_lock_sha",
        "acquire_014_lock_sha",
        "family_014_lock_sha",
        "kappa_013_lock_sha",
        "prereg_lock_sha",
        "observation_abi",
        "episode_boundary_api",
        "reads_focus",
        "prune_losers",
        "earned_next",
        "ex0s",
        "clone_empty_copies_relate",
    ):
        if lock.get(key) != snap.get(key):
            return False, f"genome_016 drift: {key}", snap
    return True, "genome_016 RELATE candidate intact", snap


def relate_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.16.RELATE",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "candidate relations under ambiguity",
        "preregistered_claim": PREREGISTERED_CLAIM,
        "observation_abi": "observe_event",
        "episode_boundary_api": "end_event_episode",
        "reads_focus": False,
        "prune_losers": False,
        "cell_ids": list(CELL_IDS),
        "cell_ok": {r["cell"]: bool(r.get("ok")) for r in rows},
        "make_relate_sha": _sha_src(make_relate),
        "run_tm016relate_sha": _sha_file(Path(__file__)),
        "teacher_outcome_sha": _sha_src(teacher_outcome),
        "genome_016_lock_sha": _sha_file(GENOME_016_LOCK) if GENOME_016_LOCK.exists() else None,
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "genome_015_lock_sha": _sha_file(GENOME_015_LOCK),
        "skeleton_015_lock_sha": _sha_file(SKELETON_015_LOCK),
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK),
        "acquire_014_lock_sha": _sha_file(ACQUIRE_LOCK),
        "family_014_lock_sha": _sha_file(FAMILY_014_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "refuse": [
            "pruning losers or support>=N earn flags",
            "RELATE reading focus (answer channel)",
            "cross-life Y→X via leftover _rel_prev_visible",
            "harness setattr to clear transients",
            "requiring reset_rho between lives to accumulate",
            "universal one-observation ban",
            "pair-hop / observe_symbol as the 0.16 binding channel",
            "FAMILY / 288 worlds this pass",
            "LOOKAHEAD",
            "pixels / raw audio / text streams",
            "stamp or pre-name Ex0S 0.0.005",
            "rewrite genome_015 / skeleton_015 / genome_014 / acquire_014 / family_014",
        ],
    }


def write_relate_lock(
    rows: list[dict[str, Any]], path: Path = RELATE_LOCK
) -> dict[str, Any]:
    snap = relate_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_relate_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = RELATE_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/relate_016.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    live = {
        "make_relate_sha": _sha_src(make_relate),
        "run_tm016relate_sha": _sha_file(Path(__file__)),
        "teacher_outcome_sha": _sha_src(teacher_outcome),
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "genome_015_lock_sha": _sha_file(GENOME_015_LOCK),
        "genome_016_lock_sha": _sha_file(GENOME_016_LOCK) if GENOME_016_LOCK.exists() else None,
        "observation_abi": "observe_event",
        "episode_boundary_api": "end_event_episode",
        "reads_focus": False,
        "prune_losers": False,
        "earned_next": False,
        "ex0s": None,
        "cell_ids": list(CELL_IDS),
        "preregistered_claim": PREREGISTERED_CLAIM,
    }
    for key, val in live.items():
        if lock.get(key) != val:
            return False, f"relate lock drift: {key}", lock
    if rows is not None:
        live_ok = {r["cell"]: bool(r.get("ok")) for r in rows}
        if live_ok != lock.get("cell_ok"):
            return False, "cell_ok drift", lock
        if not all(live_ok.get(c) for c in CELL_IDS):
            return False, "cell not all ok", lock
    return True, "relate_016.lock intact", lock


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

    summary = run_relate(seed=args.seed, write_locks=args.write_lock)
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
    d = REPO_ROOT / "runs" / f"{stamp}_tm016relate"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
