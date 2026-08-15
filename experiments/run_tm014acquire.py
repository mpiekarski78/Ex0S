"""TM.0.14.ACQUIRE: experience authors contextual continuations over a skeleton.

Apparatus plants untagged X→A→Y / X→B→Y only — never ctx=.
Organism retains compose-local (κ, frontier) on HOLD; teacher may pass only
motor + outcome; observe_outcome authors source=experience_ctx rows.

No FAMILY. No Ex0S stamp. earned_next always false. No LOOKAHEAD.
Do not modify run_tm011compose.make — kwargs forward use_acquire_ctx.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm040 import probe
from three_memory.dial_env import ChannelDialWorld
from three_memory.kappa import CTX_ENCODING
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import TagStore

ACQUIRE_LOCK = REPO_ROOT / "docs" / "acquire_014.lock"
GENOME_014_LOCK = REPO_ROOT / "docs" / "genome_014.lock"
GENOME_013_LOCK = REPO_ROOT / "docs" / "genome_013.lock"
KAPPA_LOCK = REPO_ROOT / "docs" / "kappa_013.lock"
DEFAULT_SEED = 12345
HERE = "chb"
SOURCE_EXPERIENCE = "experience_ctx"

TOKENS = {
    "x": "x",
    "a": "a",
    "b": "b",
    "y": "y",
    "press": "press",
    "tune": "tune",
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


# --- Score-only oracle (independent F; never write_s) ---------------------------


def reference_edge_sem(bind: str, did: str) -> str:
    return bind.lower() + "\0" + did.lower()


def reference_kappa_seed(origin: str) -> str:
    return _sha_bytes(b"origin\0" + origin.lower().encode())


def reference_kappa_step(previous_kappa: str, traversed_token: str) -> str:
    return _sha_bytes(previous_kappa.encode() + b"\0" + traversed_token.encode())


def reference_route_kappa(origin: str, hops: Sequence[tuple[str, str]]) -> str:
    """Scoring oracle — must never plant ctx into S."""
    k = reference_kappa_seed(origin)
    for bind, did in hops:
        k = reference_kappa_step(k, reference_edge_sem(bind, did))
    return k


@dataclass(frozen=True)
class SkelEdge:
    fid: str
    bind: str
    did: str
    support: int = 1
    contradiction: int = 0


def write_skeleton(dest: Path, edges: Sequence[SkelEdge]) -> None:
    """Plant untagged relational skeleton only. Runtime-refuse any ctx key."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for e in edges:
        if hasattr(e, "ctx") or (isinstance(e, dict) and "ctx" in e):  # type: ignore[arg-type]
            raise ValueError("skeleton writer refuses ctx")
        w, n = int(e.support), int(e.contradiction)
        tags: dict[str, Any] = {
            "bind": e.bind,
            "did": e.did,
            "here": HERE,
            "w0": e.bind,
            "hyp": "contradicted" if n else ("supported" if w else "untried"),
            "trials": w + n,
            "wins": w,
            "losses": n,
            "support": w,
            "contradiction": n,
            "source": "skeleton",
        }
        if "ctx" in tags:
            raise ValueError("skeleton writer refuses ctx")
        (dest / f"{e.fid}.tag").write_text(record_to_tagfile(e.fid, tags), encoding="utf-8")


def write_skeleton_tags(dest: Path, tag_rows: Sequence[dict[str, Any]]) -> None:
    """Low-level skeleton write; rejects ctx in any row."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for row in tag_rows:
        if "ctx" in row:
            raise ValueError("skeleton writer refuses ctx")
        fid = str(row["fid"])
        tags = {k: v for k, v in row.items() if k != "fid"}
        if "ctx" in tags:
            raise ValueError("skeleton writer refuses ctx")
        (dest / f"{fid}.tag").write_text(record_to_tagfile(fid, tags), encoding="utf-8")


def diamond_skeleton(*, prefer: str | None = None) -> list[SkelEdge]:
    """X→A→Y and X→B→Y. prefer='a'|'b' sets hop-0 evidence; None = equal."""
    sa, sb = 1, 1
    if prefer == "a":
        sa, sb = 10, 1
    elif prefer == "b":
        sa, sb = 1, 10
    return [
        SkelEdge("xa", TOKENS["x"], TOKENS["a"], support=sa),
        SkelEdge("ay", TOKENS["a"], TOKENS["y"], support=1),
        SkelEdge("xb", TOKENS["x"], TOKENS["b"], support=sb),
        SkelEdge("by", TOKENS["b"], TOKENS["y"], support=1),
    ]


def make_acquire(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    pol = policy if policy is not None else UsePolicy(seed=1)
    return make(
        s_dir,
        None,
        pol,
        explore_epsilon=0.0,
        use_context_kappa=True,
        use_acquire_ctx=True,
        **kwargs,
    )


def teacher_outcome(ag: Any, motor: str, *, success: bool) -> dict[str, Any]:
    """Hard contract: only motor identity + outcome. Never bind/κ/path/Y/destination."""
    forbidden = {
        "bind",
        "ctx",
        "kappa",
        "lived",
        "route",
        "path",
        "y",
        "frontier",
        "did",
        "destination",
        "origin",
        "hops",
    }
    info = {"action": str(motor).lower()}
    if set(info.keys()) != {"action"}:
        raise ValueError("teacher_outcome may only pass action")
    if set(info) & forbidden:
        raise ValueError("teacher_outcome internal contract broken")
    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    return ag.observe_outcome(obs, success, info)


def stash_experience_bytes(s_dir: Path) -> list[tuple[str, bytes]]:
    """Byte-copy organism-authored experience rows (no tag rewrite)."""
    out: list[tuple[str, bytes]] = []
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") == SOURCE_EXPERIENCE:
            out.append((p.name, p.read_bytes()))
    return out


def restore_experience_bytes(s_dir: Path, blobs: Sequence[tuple[str, bytes]]) -> None:
    for name, data in blobs:
        (s_dir / name).write_bytes(data)


def traverse_hold(ag: Any, cue: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """One act: compose along skeleton; expect HOLD when no contextual motor yet."""
    tokens = frozenset({cue.lower()})
    out = probe(ag, "probe_channel_b", seed, tokens=tokens)
    pol = out.get("policy") or {}
    out["compose_hold"] = bool(pol.get("compose_hold"))
    out["context_kappa"] = pol.get("context_kappa")
    out["compose_hops"] = pol.get("compose_hops")
    out["lived_pending"] = bool(getattr(ag, "_lived_pending", False))
    out["lived_bind"] = getattr(ag, "_lived_bind", None)
    out["lived_kappa"] = getattr(ag, "_lived_kappa", None)
    return out


def probe_cue(ag: Any, cue: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    tokens = frozenset({cue.lower()})
    out = probe(ag, "probe_channel_b", seed, tokens=tokens)
    pol = out.get("policy") or {}
    out["motor"] = str(out.get("action_name") or "hold").lower()
    out["compose_hold"] = bool(pol.get("compose_hold"))
    out["context_kappa"] = pol.get("context_kappa")
    out["compose_hops"] = pol.get("compose_hops")
    out["evidence_tie"] = bool(pol.get("evidence_tie"))
    return out


def list_experience_ctx(s_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(s_dir.glob("*.tag")):
        fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") != SOURCE_EXPERIENCE:
            continue
        if not isinstance(tags.get("ctx"), str) or not tags.get("ctx"):
            continue
        rows.append(
            {
                "fid": fid,
                "bind": str(tags.get("bind") or "").lower(),
                "did": str(tags.get("did") or "").lower(),
                "ctx": str(tags["ctx"]),
                "support": int(tags.get("support") or 0),
                "contradiction": int(tags.get("contradiction") or 0),
                "source": SOURCE_EXPERIENCE,
            }
        )
    return rows


def list_ctx_any(s_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(s_dir.glob("*.tag")):
        fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if isinstance(tags.get("ctx"), str) and tags.get("ctx"):
            rows.append({"fid": fid, **{k: tags[k] for k in tags}})
    return rows


def s_content_hash(s_dir: Path) -> str:
    parts: list[bytes] = []
    for p in sorted(s_dir.glob("*.tag")):
        parts.append(p.name.encode() + b"\0" + p.read_bytes())
    return _sha_bytes(b"".join(parts))


def copy_experience_rows(src: Path, dest: Path) -> None:
    """Copy only source=experience_ctx tag files onto dest (skeleton already there)."""
    for p in sorted(src.glob("*.tag")):
        fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") != SOURCE_EXPERIENCE:
            continue
        (dest / f"{fid}.tag").write_text(record_to_tagfile(fid, tags), encoding="utf-8")


def clear_experience_rows(s_dir: Path) -> None:
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") == SOURCE_EXPERIENCE:
            p.unlink()


def reload_store(ag: Any) -> None:
    if hasattr(ag.store, "reload"):
        ag.store.reload()

def oracle_ka() -> str:
    return reference_route_kappa(TOKENS["x"], [(TOKENS["x"], TOKENS["a"]), (TOKENS["a"], TOKENS["y"])])


def oracle_kb() -> str:
    return reference_route_kappa(TOKENS["x"], [(TOKENS["x"], TOKENS["b"]), (TOKENS["b"], TOKENS["y"])])


def life_a(ag: Any, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    trav = traverse_hold(ag, TOKENS["x"], seed=seed)
    if not trav.get("lived_pending"):
        return {"ok": False, "why": "life_a: lived not pending after traverse", "trav": trav}
    teach = teacher_outcome(ag, "press", success=True)
    return {"ok": True, "trav": trav, "teach": teach}


def life_b(ag: Any, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    trav = traverse_hold(ag, TOKENS["x"], seed=seed)
    if not trav.get("lived_pending"):
        return {"ok": False, "why": "life_b: lived not pending after traverse", "trav": trav}
    teach = teacher_outcome(ag, "tune", success=True)
    return {"ok": True, "trav": trav, "teach": teach}


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


# --- Battery D0–D15 ------------------------------------------------------------


def cell_d0_birth(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d0"
    write_skeleton(s, diamond_skeleton())
    ctx = list_ctx_any(s)
    return _cell("D0_birth_no_ctx", len(ctx) == 0, n_ctx=len(ctx), n_skel=4)


def cell_d1_life_a_only(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d1"
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    la = life_a(ag)
    if not la["ok"]:
        return _cell("D1_life_a_only", False, why=la.get("why"), **la)
    rows = list_experience_ctx(s)
    ka = oracle_ka()
    if len(rows) != 1 or rows[0]["did"] != "press" or rows[0]["ctx"] != ka:
        return _cell("D1_life_a_only", False, why="expected single Y→PRESS ctx=κA", rows=rows, ka=ka)
    # Prefer A: PRESS. Prefer B: HOLD (κB has no matching motor).
    _refresh_skel(s, "a")
    ag = make_acquire(s, policy)
    pa = probe_cue(ag, TOKENS["x"])
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    pb = probe_cue(ag, TOKENS["x"])
    ok = pa["motor"] == "press" and pb["motor"] == "hold"
    return _cell(
        "D1_life_a_only",
        ok,
        why=None if ok else f"A={pa['motor']} B={pb['motor']}",
        a_motor=pa["motor"],
        b_motor=pb["motor"],
        rows=rows,
    )


def _refresh_skel(s: Path, prefer: str) -> None:
    for e in diamond_skeleton(prefer=prefer):
        tags = {
            "bind": e.bind,
            "did": e.did,
            "here": HERE,
            "w0": e.bind,
            "hyp": "supported",
            "trials": e.support,
            "wins": e.support,
            "losses": 0,
            "support": e.support,
            "contradiction": 0,
            "source": "skeleton",
        }
        (s / f"{e.fid}.tag").write_text(record_to_tagfile(e.fid, tags), encoding="utf-8")


def cell_d2_both(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d2"
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    if not life_a(ag)["ok"]:
        return _cell("D2_both_coexist", False, why="life_a failed")
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    if not life_b(ag)["ok"]:
        return _cell("D2_both_coexist", False, why="life_b failed")
    rows = list_experience_ctx(s)
    ka, kb = oracle_ka(), oracle_kb()
    by_did = {r["did"]: r for r in rows}
    if set(by_did) != {"press", "tune"}:
        return _cell("D2_both_coexist", False, why="expected PRESS+TUNE", rows=rows)
    if by_did["press"]["ctx"] != ka or by_did["tune"]["ctx"] != kb:
        return _cell("D2_both_coexist", False, why="ctx mismatch oracle", rows=rows, ka=ka, kb=kb)
    _refresh_skel(s, "a")
    ag = make_acquire(s, policy)
    pa = probe_cue(ag, TOKENS["x"])
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    pb = probe_cue(ag, TOKENS["x"])
    ok = pa["motor"] == "press" and pb["motor"] == "tune"
    return _cell(
        "D2_both_coexist",
        ok,
        why=None if ok else f"A={pa['motor']} B={pb['motor']}",
        a_motor=pa["motor"],
        b_motor=pb["motor"],
        rows=rows,
    )


def _prep_both(s: Path, policy: UsePolicy) -> tuple[bool, str]:
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    if not life_a(ag)["ok"]:
        return False, "life_a"
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    if not life_b(ag)["ok"]:
        return False, "life_b"
    return True, ""


def cell_d3_reset_rho(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d3"
    ok0, why0 = _prep_both(s, policy)
    if not ok0:
        return _cell("D3_reset_rho", False, why=why0)
    ag = make_acquire(s, policy)
    before = list_experience_ctx(s)
    ag.reset_rho()
    after = list_experience_ctx(s)
    _refresh_skel(s, "a")
    ag = make_acquire(s, policy)
    ag.reset_rho()
    pa = probe_cue(ag, TOKENS["x"])
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    ag.reset_rho()
    pb = probe_cue(ag, TOKENS["x"])
    ok = before == after and pa["motor"] == "press" and pb["motor"] == "tune"
    return _cell("D3_reset_rho", ok, a_motor=pa["motor"], b_motor=pb["motor"], n=len(after))


def cell_d4_newborn(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d4"
    ok0, why0 = _prep_both(s, policy)
    if not ok0:
        return _cell("D4_newborn_reload", False, why=why0)
    rows = list_experience_ctx(s)
    _refresh_skel(s, "a")
    ag = make_acquire(s, policy)
    # Newborn: clone_empty + TagStore reload on same S path.
    nb = ag.clone_empty(store_enabled=True)
    nb.store = TagStore(s, enabled=True)
    pa = probe_cue(nb, TOKENS["x"])
    _refresh_skel(s, "b")
    nb2 = ag.clone_empty(store_enabled=True)
    nb2.store = TagStore(s, enabled=True)
    pb = probe_cue(nb2, TOKENS["x"])
    ok = pa["motor"] == "press" and pb["motor"] == "tune" and len(rows) == 2
    return _cell("D4_newborn_reload", ok, a_motor=pa["motor"], b_motor=pb["motor"], rows=rows)


def cell_d5_wipe(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d5"
    ok0, why0 = _prep_both(s, policy)
    if not ok0:
        return _cell("D5_wipe_hold", False, why=why0)
    if s.exists():
        shutil.rmtree(s)
    s.mkdir(parents=True)
    ag = make_acquire(s, policy)
    p = probe_cue(ag, TOKENS["x"])
    ok = p["motor"] == "hold"
    return _cell("D5_wipe_hold", ok, motor=p["motor"])


def cell_d6_swap_ctx_rows(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    """Identical skeletons; swap only experience_ctx rows → behavior follows memory."""
    sa = tmp / "d6a"
    sb = tmp / "d6b"
    write_skeleton(sa, diamond_skeleton(prefer="a"))
    write_skeleton(sb, diamond_skeleton(prefer="a"))
    aga = make_acquire(sa, policy)
    if not life_a(aga)["ok"]:
        return _cell("D6_ctx_row_swap", False, why="life_a on A")
    # Organism B: life B on prefer-b skeleton, then unify skeleton to prefer-a.
    write_skeleton(sb, diamond_skeleton(prefer="b"))
    agb = make_acquire(sb, policy)
    if not life_b(agb)["ok"]:
        return _cell("D6_ctx_row_swap", False, why="life_b on B")
    # Both get identical prefer-a skeletons; A has PRESS ctx, B has TUNE ctx.
    _refresh_skel(sa, "a")
    _refresh_skel(sb, "a")
    # Swap only experience rows via byte-copy (no apparatus tag rewrite of ctx).
    rows_a = list_experience_ctx(sa)
    rows_b = list_experience_ctx(sb)
    blobs_a = stash_experience_bytes(sa)
    blobs_b = stash_experience_bytes(sb)
    clear_experience_rows(sa)
    clear_experience_rows(sb)
    restore_experience_bytes(sa, blobs_b)
    restore_experience_bytes(sb, blobs_a)
    # After swap: sa has TUNE@κB but prefer-a → κA → HOLD; sb has PRESS@κA → PRESS.
    aga2 = make_acquire(sa, policy)
    agb2 = make_acquire(sb, policy)
    pa = probe_cue(aga2, TOKENS["x"])
    pb = probe_cue(agb2, TOKENS["x"])
    ok = pa["motor"] == "hold" and pb["motor"] == "press"
    return _cell(
        "D6_ctx_row_swap",
        ok,
        why=None if ok else f"sa={pa['motor']} sb={pb['motor']}",
        a_motor=pa["motor"],
        b_motor=pb["motor"],
        rows_a_before=rows_a,
        rows_b_before=rows_b,
    )


def cell_d7_evidence_math(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d7"
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    # PRESS success → (1,0)
    if not life_a(ag)["ok"]:
        return _cell("D7_evidence_math", False, why="press success")
    rows = list_experience_ctx(s)
    if not rows or rows[0]["support"] != 1 or rows[0]["contradiction"] != 0:
        return _cell("D7_evidence_math", False, why=f"after success {rows}")
    # Existing motor is selected by compose; failure revises via _update_chosen_hyp.
    def _fail_press() -> str | None:
        p = probe_cue(ag, TOKENS["x"])
        if p["motor"] != "press":
            return f"expected press got {p['motor']}"
        teacher_outcome(ag, "press", success=False)
        return None

    err = _fail_press()
    if err:
        return _cell("D7_evidence_math", False, why=err)
    rows = list_experience_ctx(s)
    if rows[0]["support"] != 1 or rows[0]["contradiction"] != 1:
        return _cell("D7_evidence_math", False, why=f"after fail1 {rows}")
    err = _fail_press()
    if err:
        return _cell("D7_evidence_math", False, why=err)
    rows = list_experience_ctx(s)
    if rows[0]["support"] != 1 or rows[0]["contradiction"] != 2:
        return _cell("D7_evidence_math", False, why=f"after fail2 {rows}")
    # TUNE under same κA: byte-stash PRESS so compose HOLDs, author TUNE, restore bytes.
    press_blobs = stash_experience_bytes(s)
    clear_experience_rows(s)
    reload_store(ag)
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending"):
        return _cell("D7_evidence_math", False, why="no lived for tune")
    teacher_outcome(ag, "tune", success=True)
    restore_experience_bytes(s, press_blobs)
    rows = list_experience_ctx(s)
    by = {r["did"]: r for r in rows}
    if "tune" not in by or by["tune"]["support"] != 1 or by["tune"]["contradiction"] != 0:
        return _cell("D7_evidence_math", False, why=f"tune row {rows}")
    if by["press"]["support"] != 1 or by["press"]["contradiction"] != 2:
        return _cell("D7_evidence_math", False, why=f"press row {rows}")
    # Evidence choose: TUNE (1,0) beats PRESS (1,-2)
    ag = make_acquire(s, policy)
    p = probe_cue(ag, TOKENS["x"])
    ok = p["motor"] == "tune"
    return _cell("D7_evidence_math", ok, motor=p["motor"], rows=rows)


def cell_d8_diff_histories(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d8"
    ok0, why0 = _prep_both(s, policy)
    if not ok0:
        return _cell("D8_diff_histories", False, why=why0)
    rows = list_experience_ctx(s)
    ka, kb = oracle_ka(), oracle_kb()
    ok = (
        len(rows) == 2
        and {r["ctx"] for r in rows} == {ka, kb}
        and ka != kb
    )
    return _cell("D8_diff_histories", ok, rows=rows, ka=ka, kb=kb)


def cell_d9_tie_hold(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d9"
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    if not life_a(ag)["ok"]:
        return _cell("D9_tie_hold", False, why="life_a")
    # Byte-stash PRESS so compose HOLDs; author equal-evidence TUNE; restore bytes.
    press_blobs = stash_experience_bytes(s)
    clear_experience_rows(s)
    reload_store(ag)
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending"):
        return _cell("D9_tie_hold", False, why="no lived")
    teacher_outcome(ag, "tune", success=True)
    restore_experience_bytes(s, press_blobs)
    rows = list_experience_ctx(s)
    if len(rows) != 2:
        return _cell("D9_tie_hold", False, why=f"rows {rows}")
    if rows[0]["support"] != rows[1]["support"] or rows[0]["contradiction"] != rows[1]["contradiction"]:
        return _cell("D9_tie_hold", False, why="unequal evidence")
    ag = make_acquire(s, policy)
    p = probe_cue(ag, TOKENS["x"])
    ok = p["motor"] == "hold" and bool(p.get("evidence_tie") or p.get("compose_hold"))
    return _cell("D9_tie_hold", ok, motor=p["motor"], rows=rows, tie=p.get("evidence_tie"))


def cell_d10_rename_fid(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d10"
    ok0, why0 = _prep_both(s, policy)
    if not ok0:
        return _cell("D10_rename_fid", False, why=why0)
    # Rename experience fids
    for r in list_experience_ctx(s):
        old = s / f"{r['fid']}.tag"
        text = old.read_text(encoding="utf-8")
        _fid, tags = parse_tagfile(text)
        new_fid = f"ren_{r['fid']}"
        old.unlink()
        (s / f"{new_fid}.tag").write_text(record_to_tagfile(new_fid, tags), encoding="utf-8")
    _refresh_skel(s, "a")
    ag = make_acquire(s, policy)
    pa = probe_cue(ag, TOKENS["x"])
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    pb = probe_cue(ag, TOKENS["x"])
    ok = pa["motor"] == "press" and pb["motor"] == "tune"
    return _cell("D10_rename_fid", ok, a_motor=pa["motor"], b_motor=pb["motor"])


def cell_d11_storage_order(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d11"
    ok0, why0 = _prep_both(s, policy)
    if not ok0:
        return _cell("D11_storage_order", False, why=why0)
    files = list(s.glob("*.tag"))
    # Rewrite in reverse order with temp names then restore
    blobs = [(p.name, p.read_bytes()) for p in sorted(files)]
    for p in files:
        p.unlink()
    for name, data in reversed(blobs):
        (s / name).write_bytes(data)
    _refresh_skel(s, "a")
    ag = make_acquire(s, policy)
    pa = probe_cue(ag, TOKENS["x"])
    _refresh_skel(s, "b")
    ag = make_acquire(s, policy)
    pb = probe_cue(ag, TOKENS["x"])
    ok = pa["motor"] == "press" and pb["motor"] == "tune"
    return _cell("D11_storage_order", ok, a_motor=pa["motor"], b_motor=pb["motor"])


def cell_d12_no_apparatus_ctx(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d12"
    refused = False
    try:
        write_skeleton_tags(
            s,
            [{"fid": "bad", "bind": "y", "did": "press", "ctx": "deadbeef", "here": HERE}],
        )
    except ValueError as e:
        refused = "ctx" in str(e).lower()
    src = inspect.getsource(teacher_outcome)
    ok_teacher = (
        'info={"action"' in src.replace(" ", "")
        or 'info = {"action"' in src.replace(" ", "")
    )
    ok_dest = "destination" in src
    ok_only_action = 'set(info.keys()) != {"action"}' in src.replace(" ", "") or (
        'set(info.keys())!={"action"}' in src.replace(" ", "")
    )
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    life_a(ag)
    rows = list_experience_ctx(s)
    ok = (
        refused
        and ok_teacher
        and ok_dest
        and ok_only_action
        and len(rows) == 1
        and rows[0]["source"] == SOURCE_EXPERIENCE
    )
    return _cell(
        "D12_no_apparatus_ctx",
        ok,
        refused=refused,
        ok_teacher=ok_teacher,
        ok_dest=ok_dest,
        ok_only_action=ok_only_action,
        rows=rows,
    )


def cell_d13_oracle_score_only(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d13"
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    trav = traverse_hold(ag, TOKENS["x"])
    lived = trav.get("lived_kappa")
    ka = oracle_ka()
    if lived != ka:
        return _cell("D13_oracle_agree", False, why=f"lived {lived} != oracle {ka}")
    teacher_outcome(ag, "press", success=True)
    rows = list_experience_ctx(s)
    ok = len(rows) == 1 and rows[0]["ctx"] == ka
    # Oracle functions must not appear in write_skeleton source as writers of ctx
    ws = inspect.getsource(write_skeleton)
    ok2 = "ctx" in ws and "refuses" in ws
    return _cell("D13_oracle_agree", ok and ok2, rows=rows, ka=ka)


def cell_d14_weights_no_shortcut(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s = tmp / "d14"
    write_skeleton(s, diamond_skeleton(prefer="a"))
    ag = make_acquire(s, policy)
    w0 = ag.weight_hash()
    h0 = s_content_hash(s)
    if not life_a(ag)["ok"]:
        return _cell("D14_weights_no_shortcut", False, why="life_a")
    h1 = s_content_hash(s)
    ag = make_acquire(s, policy)
    w1 = ag.weight_hash()
    p = probe_cue(ag, TOKENS["x"])
    h2 = s_content_hash(s)
    # Probe must not write shortcuts (X→press)
    shortcuts = []
    for row in list_ctx_any(s):
        if str(row.get("bind") or "").lower() == TOKENS["x"] and str(row.get("did") or "").lower() == "press":
            shortcuts.append(row)
    ok = (
        w0 == w1
        and ag.weights_unchanged()
        and h1 != h0
        and h2 == h1
        and p["motor"] == "press"
        and not shortcuts
    )
    return _cell(
        "D14_weights_no_shortcut",
        ok,
        w0=w0,
        w1=w1,
        motor=p["motor"],
        shortcuts=shortcuts,
    )


def cell_d15_nasty_five(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # (1) no κ + motor success → no contextual row
    s1 = tmp / "d15_1"
    write_skeleton(s1, diamond_skeleton(prefer="a"))
    ag = make_acquire(s1, policy)
    teacher_outcome(ag, "press", success=True)
    results["no_kappa"] = list_experience_ctx(s1) == []

    # (2) κ exists + non-motor outcome → no contextual row
    s2 = tmp / "d15_2"
    write_skeleton(s2, diamond_skeleton(prefer="a"))
    ag = make_acquire(s2, policy)
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending"):
        results["non_motor"] = False
    else:
        teacher_outcome(ag, "a", success=True)  # non-motor token
        results["non_motor"] = list_experience_ctx(s2) == []

    # (3) κA then unrelated later motor success → no stale write
    s3 = tmp / "d15_3"
    write_skeleton(s3, diamond_skeleton(prefer="a"))
    ag = make_acquire(s3, policy)
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending"):
        results["stale"] = False
    else:
        # New act with no matching cue clears lived without consuming.
        probe(ag, "probe_channel_b", DEFAULT_SEED + 1, tokens=frozenset())
        teacher_outcome(ag, "press", success=True)
        results["stale"] = list_experience_ctx(s3) == []

    # (4) repeated same (Y,motor,κ) → one row, increasing support
    s4 = tmp / "d15_4"
    write_skeleton(s4, diamond_skeleton(prefer="a"))
    ag = make_acquire(s4, policy)
    ok_rep = False
    if life_a(ag)["ok"]:
        ok_rep = True
        for _ in range(2):
            p = probe_cue(ag, TOKENS["x"])
            if p["motor"] != "press":
                ok_rep = False
                break
            teacher_outcome(ag, "press", success=True)
        rows4 = list_experience_ctx(s4)
        ok_rep = ok_rep and len(rows4) == 1 and rows4[0]["support"] == 3
    results["repeat"] = ok_rep

    # (5) failure for unseen (Y,motor,κ) → no negative-memory swarm
    s5 = tmp / "d15_5"
    write_skeleton(s5, diamond_skeleton(prefer="a"))
    ag = make_acquire(s5, policy)
    trav = traverse_hold(ag, TOKENS["x"])
    if not trav.get("lived_pending"):
        results["unseen_fail"] = False
    else:
        teacher_outcome(ag, "press", success=False)
        results["unseen_fail"] = list_experience_ctx(s5) == [] and list_ctx_any(s5) == []

    ok = all(bool(v) for v in results.values())
    return _cell("D15_nasty_five", ok, **results)


CELLS: list[Callable[[Path, UsePolicy], dict[str, Any]]] = [
    cell_d0_birth,
    cell_d1_life_a_only,
    cell_d2_both,
    cell_d3_reset_rho,
    cell_d4_newborn,
    cell_d5_wipe,
    cell_d6_swap_ctx_rows,
    cell_d7_evidence_math,
    cell_d8_diff_histories,
    cell_d9_tie_hold,
    cell_d10_rename_fid,
    cell_d11_storage_order,
    cell_d12_no_apparatus_ctx,
    cell_d13_oracle_score_only,
    cell_d14_weights_no_shortcut,
    cell_d15_nasty_five,
]


def run_acquire(*, seed: int = DEFAULT_SEED, write_locks: bool = False) -> dict[str, Any]:
    policy = UsePolicy(seed=seed)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tm014acquire_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            rows.append(fn(root, policy))
    n_ok = sum(1 for r in rows if r.get("ok"))
    summary: dict[str, Any] = {
        "version": "TM.0.14.ACQUIRE",
        "ok": n_ok == len(rows),
        "n_ok": n_ok,
        "n_cells": len(rows),
        "earned_next": False,
        "ex0s": None,
        "seed": seed,
        "ctx_encoding": CTX_ENCODING,
        "claim": (
            "A frozen developmental recipe can use experienced outcomes to author "
            "provenance-sensitive contextual continuations into S over an existing "
            "relational skeleton, then later use κ to select those organism-authored "
            "continuations after ρ reset, without contextual answers being planted "
            "by the apparatus."
        ),
        "rows": rows,
    }
    if write_locks:
        write_genome_014_lock()
        write_acquire_lock(rows)
    return summary


def genome_014_snapshot() -> dict[str, Any]:
    from three_memory import agent as agent_mod

    with tempfile.TemporaryDirectory(prefix="tm014_g_") as tmp:
        s = Path(tmp) / "s"
        s.mkdir()
        ag = make_acquire(s, UsePolicy(seed=1))
        wh = ag.weight_hash()
    return {
        "version": "TM.0.14.GENOME",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "use_context_kappa": True,
        "use_acquire_ctx": True,
        "ctx_encoding": CTX_ENCODING,
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "policy_sha": _sha_file(REPO_ROOT / "three_memory" / "policy.py"),
        "cortex_sha": _sha_file(REPO_ROOT / "three_memory" / "cortex.py"),
        "kappa_sha": _sha_file(REPO_ROOT / "three_memory" / "kappa.py"),
        "make011compose_sha": _sha_bytes(inspect.getsource(make).encode()),
        "make_acquire_sha": _sha_src(make_acquire),
        "run_tm014acquire_sha": _sha_file(Path(__file__)),
        "cortex_weight_hash": wh,
        "n_feat": int(UsePolicy.n_feat),
        "genome_013_lock_sha": _sha_file(GENOME_013_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "clone_empty_copies_acquire": "use_acquire_ctx=self.use_acquire_ctx"
        in inspect.getsource(agent_mod.ThreeMemoryAgent.clone_empty),
    }


def write_genome_014_lock(path: Path = GENOME_014_LOCK) -> dict[str, Any]:
    snap = genome_014_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_genome_014(path: Path = GENOME_014_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = genome_014_snapshot()
    if not path.exists():
        return False, "docs/genome_014.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "agent_sha",
        "policy_sha",
        "cortex_sha",
        "kappa_sha",
        "make011compose_sha",
        "make_acquire_sha",
        "cortex_weight_hash",
        "n_feat",
        "use_context_kappa",
        "use_acquire_ctx",
        "ctx_encoding",
        "genome_013_lock_sha",
        "kappa_013_lock_sha",
        "earned_next",
    ):
        if snap.get(key) != lock.get(key):
            return False, f"genome_014 drift: {key}", snap
    if not snap.get("clone_empty_copies_acquire"):
        return False, "clone_empty missing use_acquire_ctx", snap
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    return True, "genome_014 ACQUIRE candidate intact", snap


def acquire_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.14.ACQUIRE",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ctx_encoding": CTX_ENCODING,
        "seed": DEFAULT_SEED,
        "n_cells": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "cell_ids": [r["cell"] for r in rows],
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "genome_013_lock_sha": _sha_file(GENOME_013_LOCK),
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK) if GENOME_014_LOCK.exists() else None,
        "reference_route_kappa_sha": _sha_src(reference_route_kappa),
        "teacher_outcome_sha": _sha_src(teacher_outcome),
        "write_skeleton_sha": _sha_src(write_skeleton),
        "make_acquire_sha": _sha_src(make_acquire),
        "run_tm014acquire_sha": _sha_file(Path(__file__)),
        "refuse": [
            "apparatus ctx planting",
            "second kappa engine for lived state",
            "stale lived retention across act/outcome",
            "teacher supplying Y/kappa/path",
            "rewrite genome_011 or genome_013 lock",
            "modify run_tm011compose.py",
            "LOOKAHEAD",
            "FAMILY / 288 worlds this pass",
            "stamp or pre-name Ex0S 0.0.005",
            "claim full skeleton acquisition from life",
        ],
    }


def write_acquire_lock(rows: list[dict[str, Any]], path: Path = ACQUIRE_LOCK) -> dict[str, Any]:
    if not GENOME_014_LOCK.exists():
        raise FileNotFoundError("write genome_014.lock before acquire_014.lock")
    snap = acquire_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_acquire_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = ACQUIRE_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/acquire_014.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", lock
    if rows is not None:
        ids = [r["cell"] for r in rows]
        if ids != lock.get("cell_ids"):
            return False, "cell_ids drift", lock
        if sum(1 for r in rows if r.get("ok")) != lock.get("n_ok"):
            return False, "n_ok drift", lock
    for key, fn in (
        ("reference_route_kappa_sha", reference_route_kappa),
        ("teacher_outcome_sha", teacher_outcome),
        ("write_skeleton_sha", write_skeleton),
        ("make_acquire_sha", make_acquire),
    ):
        if _sha_src(fn) != lock.get(key):
            return False, f"lock drift: {key}", lock
    if _sha_file(Path(__file__)) != lock.get("run_tm014acquire_sha"):
        return False, "run_tm014acquire.py drift", lock
    if _sha_file(GENOME_013_LOCK) != lock.get("genome_013_lock_sha"):
        return False, "genome_013.lock pin drift", lock
    if _sha_file(KAPPA_LOCK) != lock.get("kappa_013_lock_sha"):
        return False, "kappa_013.lock pin drift", lock
    if GENOME_014_LOCK.exists() and _sha_file(GENOME_014_LOCK) != lock.get("genome_014_lock_sha"):
        return False, "genome_014.lock pin drift", lock
    return True, "acquire_014 lock intact", lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--verify-lock", action="store_true")
    args = ap.parse_args()
    summary = run_acquire(seed=args.seed, write_locks=args.write_lock)
    if args.verify_lock:
        ok, why, _ = verify_acquire_lock(summary["rows"])
        summary["lock_ok"] = ok
        summary["lock_why"] = why
        gok, gwhy, _ = verify_genome_014()
        summary["genome_014_ok"] = gok
        summary["genome_014_why"] = gwhy
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = REPO_ROOT / "runs" / f"{stamp}_tm014acquire"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    for r in summary["rows"]:
        mark = "OK" if r.get("ok") else "FAIL"
        print(f"  {mark} {r['cell']}" + (f" — {r.get('why')}" if not r.get("ok") else ""))
    if not summary["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
