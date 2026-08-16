"""TM.0.16.ALIASWALL: cross-episode alias equivalence wall on frozen RELATE.

Paired Control/Kill probe. No TM.0.17. No alias machinery. No occlusion.
Kill demonstrates dependence on externally supplied symbol equivalence;
it is not an earnable equivalence benchmark this pass.

Product stamp stays 0.0.004. earned_next=false. ex0s=null.
Prereg: docs/alias_wall.prereg.lock. Do not edit agent.py / relate_016.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import probe_cue, traverse_hold
from experiments.run_tm016relate import (
    GENOME_016_LOCK,
    LIVES_XA_Y,
    RELATE_LOCK,
    edge_support,
    empty_birth,
    evidence_winner_did,
    make_relate,
    observe_event,
    run_lives,
    skel_map,
    verify_genome_016,
)
from three_memory.policy import UsePolicy

PREREG_LOCK = REPO_ROOT / "docs" / "alias_wall.prereg.lock"
ALIAS_WALL_LOCK = REPO_ROOT / "docs" / "alias_wall.lock"

DEFAULT_SEED = 12345

WALL_CLAIM = (
    "With RELATE fixed, replacing repeated route symbols with fresh opaque aliases "
    "disperses support across disjoint edges; no reusable relation survives, so "
    "compose remains HOLD despite repetition of the same latent route."
)

CELL_IDS = (
    "W0_control",
    "W1_kill",
    "W2_schedule_twin",
    "W3_opacity",
    "W4_map_isolation",
    "W5_no_mechanism",
)

# Pinned in prereg — runner copies; no RNG.
KILL_ALIAS_TABLE = (
    {"life": 1, "origin": "kelm", "mid": "norb", "dest": "wift"},
    {"life": 2, "origin": "jasp", "mid": "clud", "dest": "vemq"},
    {"life": 3, "origin": "doth", "mid": "praq", "dest": "hunz"},
)

CONTROL_CLUTTER = (
    (("q",), ("q",), ("q",)),
    (("r",), ("r",), ("s",)),
    (("t",), ("u",), ("v",)),
)

LATENT_ROLES = ("x", "a", "y")
MOTORS = frozenset({"press", "tune", "flip", "hold", "idle", "wait"})
QUERY_LIFE = 1  # 1-indexed


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def load_prereg(path: Path = PREREG_LOCK) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def kill_schedule_from_table(
    table: Sequence[dict[str, str]] = KILL_ALIAS_TABLE,
    clutter: Sequence[Sequence[Sequence[str]]] = CONTROL_CLUTTER,
) -> tuple[tuple[list[str], ...], ...]:
    lives: list[tuple[list[str], ...]] = []
    for row, clut in zip(table, clutter):
        o, m, d = row["origin"], row["mid"], row["dest"]
        e0 = [o, *clut[0]]
        e1 = [m, *clut[1]]
        e2 = [d, *clut[2]]
        lives.append((e0, e1, e2))
    return tuple(lives)


def alias_to_role_map(table: Sequence[dict[str, str]] = KILL_ALIAS_TABLE) -> dict[str, str]:
    """Evaluator-only. Never passed to organism."""
    out: dict[str, str] = {}
    for row in table:
        out[row["origin"]] = "x"
        out[row["mid"]] = "a"
        out[row["dest"]] = "y"
    return out


def canonicalize_schedule(
    lives: Sequence[Sequence[Sequence[str]]],
    amap: dict[str, str],
) -> list[list[list[str]]]:
    """Replace Kill aliases with latent roles; leave clutter unchanged."""
    out: list[list[list[str]]] = []
    for life in lives:
        clife: list[list[str]] = []
        for ev in life:
            clife.append([amap.get(tok, tok) for tok in ev])
        out.append(clife)
    return out


def normalize_bags(lives: Sequence[Sequence[Sequence[str]]]) -> list[list[list[str]]]:
    """Order-insensitive bags per event (sorted), preserving event/life order."""
    return [[sorted(list(ev)) for ev in life] for life in lives]


def latent_path_edges(table: Sequence[dict[str, str]] = KILL_ALIAS_TABLE) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for row in table:
        edges.append((row["origin"], row["mid"]))
        edges.append((row["mid"], row["dest"]))
    return edges


def cross_life_seam_edges(
    lives: Sequence[Sequence[Sequence[str]]],
) -> list[tuple[str, str]]:
    """Edges that would be authored if an episode boundary leaked between lives."""
    edges: list[tuple[str, str]] = []
    for prev_life, next_life in zip(lives, lives[1:]):
        if not prev_life or not next_life:
            continue
        for a in prev_life[-1]:
            for b in next_life[0]:
                edges.append((str(a).lower(), str(b).lower()))
    return edges


def fresh_world(tmp: Path, name: str, policy: UsePolicy) -> tuple[Path, Any]:
    """Isolated empty S + new make_relate + reset ρ."""
    s = tmp / name
    empty_birth(s)
    ag = make_relate(s, policy)
    ag.reset_rho()
    return s, ag


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


# --- Cells --------------------------------------------------------------------


def cell_w0_control(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "w0", policy)
    run_lives(ag, LIVES_XA_Y)
    wx = evidence_winner_did(s, "x")
    wa = evidence_winner_did(s, "a")
    ag.reset_rho()
    trav = traverse_hold(ag, "x")
    lived = str(trav.get("lived_bind") or "").lower()
    ok = (
        wx == "a"
        and wa == "y"
        and lived == "y"
        and bool(trav.get("lived_pending"))
        and not trav.get("evidence_tie")
    )
    return _cell(
        "W0_control",
        ok,
        winner_x=wx,
        winner_a=wa,
        lived_bind=lived,
        hops=trav.get("compose_hops"),
        evidence_tie=bool(trav.get("evidence_tie")),
        s_dir=str(s.resolve()),
    )


def cell_w1_kill(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "w1", policy)
    kill_lives = kill_schedule_from_table()
    run_lives(ag, kill_lives)

    path_edges = latent_path_edges()
    supports = {f"{a}->{b}": edge_support(s, a, b) for a, b in path_edges}
    support_ok = all(v == 1 for v in supports.values())
    seam_supports = {
        f"{a}->{b}": edge_support(s, a, b)
        for a, b in cross_life_seam_edges(kill_lives)
    }
    seam_ok = all(v == 0 for v in seam_supports.values())

    tokens = [t for row in KILL_ALIAS_TABLE for t in (row["origin"], row["mid"], row["dest"])]
    distinct_ok = len(tokens) == len(set(tokens)) == 9
    no_shared_edge = len(set(path_edges)) == len(path_edges)

    qrow = KILL_ALIAS_TABLE[QUERY_LIFE - 1]
    cue = qrow["origin"]
    target = qrow["dest"]

    # Motor probe (fresh ρ on same isolated S)
    ag_m = make_relate(s, policy)
    ag_m.reset_rho()
    motor = str(probe_cue(ag_m, cue).get("motor") or "hold").lower()

    # Lived bind probe (fresh ρ)
    ag_t = make_relate(s, policy)
    ag_t.reset_rho()
    trav = traverse_hold(ag_t, cue)
    lived = trav.get("lived_bind")
    lived_l = str(lived).lower() if lived is not None else None

    motor_ok = motor == "hold"
    lived_ok = lived_l != target
    cue_ok = cue != "x" and cue == qrow["origin"]

    ok = (
        support_ok
        and seam_ok
        and distinct_ok
        and no_shared_edge
        and motor_ok
        and lived_ok
        and cue_ok
    )
    return _cell(
        "W1_kill",
        ok,
        supports=supports,
        seam_supports=seam_supports,
        distinct_tokens=distinct_ok,
        motor=motor,
        lived_bind=lived_l,
        target_alias=target,
        cue=cue,
        evidence_tie=bool(trav.get("evidence_tie")),
        hops=trav.get("compose_hops"),
        s_dir=str(s.resolve()),
    )


def cell_w2_schedule_twin(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    # Pure evaluator check, but preserve the preregistered per-cell birth isolation.
    s, _ag = fresh_world(tmp, "w2", policy)
    kill_lives = kill_schedule_from_table()
    amap = alias_to_role_map()
    canon = normalize_bags(canonicalize_schedule(kill_lives, amap))
    control = normalize_bags(LIVES_XA_Y)
    # Also check raw event counts / bag sizes before sort identity
    sizes_kill = [[len(ev) for ev in life] for life in kill_lives]
    sizes_ctrl = [[len(ev) for ev in life] for life in LIVES_XA_Y]
    query_life_ok = QUERY_LIFE == 1
    n_lives_ok = len(kill_lives) == len(LIVES_XA_Y) == 3
    ok = canon == control and sizes_kill == sizes_ctrl and query_life_ok and n_lives_ok
    return _cell(
        "W2_schedule_twin",
        ok,
        sizes_kill=sizes_kill,
        sizes_ctrl=sizes_ctrl,
        canon_match=canon == control,
        query_life=QUERY_LIFE,
        s_dir=str(s.resolve()),
    )


def cell_w3_opacity(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, _ag = fresh_world(tmp, "w3", policy)
    tokens = [t for row in KILL_ALIAS_TABLE for t in (row["origin"], row["mid"], row["dest"])]
    equal_len = all(len(t) == 4 for t in tokens)
    unique = len(set(tokens)) == 9
    clutter = {"q", "r", "s", "t", "u", "v"}
    no_clutter = not any(t in clutter for t in tokens)
    no_motors = not any(t in MOTORS for t in tokens)
    # Role-correlated morphology: digits, exact role token, x1-style, leading role char.
    role_leaks: list[str] = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            role_leaks.append(f"digit:{t}")
        if t in LATENT_ROLES:
            role_leaks.append(f"exact:{t}")
        if len(t) >= 2 and t[0] in LATENT_ROLES and t[1].isdigit():
            role_leaks.append(f"x1style:{t}")
        if t[0] in LATENT_ROLES:
            role_leaks.append(f"leading_role:{t}")
    ok = equal_len and unique and no_clutter and no_motors and not role_leaks
    return _cell(
        "W3_opacity",
        ok,
        equal_len=equal_len,
        unique=unique,
        role_leaks=role_leaks,
        s_dir=str(s.resolve()),
    )


def cell_w4_map_isolation(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    """Aliases may be in visible; association must not enter organism channels."""
    s, ag = fresh_world(tmp, "w4", policy)
    kill_lives = kill_schedule_from_table()
    alias_vals = {
        tok
        for row in KILL_ALIAS_TABLE
        for tok in (row["origin"], row["mid"], row["dest"])
    }

    payloads: list[dict[str, Any]] = []
    for life in kill_lives:
        for vis in life:
            ev = {"visible": list(vis), "focus": vis[0]}
            payloads.append(ev)
            observe_event(ag, vis, focus=vis[0])
        ag.end_event_episode()

    allowed_keys = {"visible", "focus"}
    meta_clean = all(set(ev.keys()) <= allowed_keys for ev in payloads)
    has_alias_visible = any(any(t in alias_vals for t in ev["visible"]) for ev in payloads)
    sm = skel_map(s)
    no_latent_in_s = not any(b in LATENT_ROLES or d in LATENT_ROLES for b, d in sm)

    src = inspect.getsource(make_relate)
    no_map_in_make = "alias_map" not in src and "role_alias" not in src

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    new_abi = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("observe_alias", "observe_equivalence", "observe_identity", "observe_role")
    ]

    ok = meta_clean and has_alias_visible and no_latent_in_s and no_map_in_make and not new_abi
    return _cell(
        "W4_map_isolation",
        ok,
        meta_clean=meta_clean,
        has_alias_visible=has_alias_visible,
        no_latent_in_s=no_latent_in_s,
        new_abi=new_abi,
        s_dir=str(s.resolve()),
    )


def cell_w5_no_mechanism(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "w5", policy)
    flags_ok = (
        ag.use_acquire_relate is True
        and ag.use_acquire_skel is True
        and ag.use_acquire_ctx is True
    )
    g_ok, g_why, _ = verify_genome_016()
    # No agent.py edits this pass — genome_016 agent_sha is the live file at freeze time;
    # verify_genome_016 must stay green.
    # No new equivalence-authoring helpers defined in this runner.
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    banned_defs = {
        "alias_merge",
        "unify_alias",
        "role_bind",
        "observe_alias",
        "observe_equivalence",
        "observe_identity",
    }
    defined = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    clean = not (defined & banned_defs)
    ok = flags_ok and g_ok and clean
    return _cell(
        "W5_no_mechanism",
        ok,
        flags_ok=flags_ok,
        genome_016=g_why,
        clean=clean,
        banned_hit=sorted(defined & banned_defs),
        s_dir=str(s.resolve()),
    )


CELLS: Sequence[Callable[[Path, UsePolicy], dict[str, Any]]] = (
    cell_w0_control,
    cell_w1_kill,
    cell_w2_schedule_twin,
    cell_w3_opacity,
    cell_w4_map_isolation,
    cell_w5_no_mechanism,
)


# --- Run / locks --------------------------------------------------------------


def run_alias_wall(*, seed: int = DEFAULT_SEED, write_locks: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tm016alias_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            # Same frozen initial weights, distinct mutable policy/organism state per cell.
            policy = UsePolicy(seed=seed)
            rows.append(fn(root, policy))

    n_ok = sum(1 for r in rows if r.get("ok"))
    summary: dict[str, Any] = {
        "version": "TM.0.16.ALIASWALL",
        "label": "cross-episode alias equivalence wall on frozen RELATE",
        "ok": n_ok == len(rows) == 6,
        "n_ok": n_ok,
        "n_cells": len(rows),
        "earned_next": False,
        "ex0s": None,
        "seed": seed,
        "claim": WALL_CLAIM,
        "rows": rows,
    }
    if write_locks and summary["ok"]:
        write_alias_wall_lock(rows)
    return summary


def verify_prereg_lock(path: Path = PREREG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/alias_wall.prereg.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False:
        return False, "prereg earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "prereg ex0s must be null", lock
    if lock.get("preregistered_claim") != WALL_CLAIM:
        return False, "claim drift", lock
    if lock.get("cell_ids") != list(CELL_IDS):
        return False, "cell_ids drift", lock
    if lock.get("not_tm017") is not True:
        return False, "must refuse TM.0.17 naming", lock
    if lock.get("genome_016_lock_sha") != _sha_file(GENOME_016_LOCK):
        return False, "genome_016_lock_sha pin", lock
    if lock.get("relate_016_lock_sha") != _sha_file(RELATE_LOCK):
        return False, "relate_016_lock_sha pin", lock
    # Pinned table must match runner constant
    table = lock.get("kill_alias_table") or []
    if len(table) != 3:
        return False, "alias table length", lock
    for i, row in enumerate(KILL_ALIAS_TABLE):
        pr = table[i]
        for k in ("origin", "mid", "dest"):
            if pr.get(k) != row[k]:
                return False, f"alias table drift life {i+1} {k}", lock
    if lock.get("control_schedule") != [list(map(list, life)) for life in LIVES_XA_Y]:
        # JSON may have lists; compare normalized
        ctrl = [[[str(t) for t in ev] for ev in life] for life in lock.get("control_schedule") or []]
        expect = [[[str(t) for t in ev] for ev in life] for life in LIVES_XA_Y]
        if ctrl != expect:
            return False, "control_schedule drift", lock
    kill_expect = [[[str(t) for t in ev] for ev in life] for life in kill_schedule_from_table()]
    kill_prereg = [[[str(t) for t in ev] for ev in life] for life in lock.get("kill_schedule") or []]
    if kill_prereg != kill_expect:
        return False, "kill_schedule drift", lock
    banned = (
        "agent_sha",
        "run_tm016aliaswall_sha",
        "alias_wall_lock_sha",
        "cell_shas",
        "make_relate_sha",
    )
    if any(k in lock for k in banned):
        return False, "prereg contains ALIASWALL artifact SHAs", lock
    return True, "alias_wall.prereg.lock intact", lock


def alias_wall_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.16.ALIASWALL",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "cross-episode alias equivalence wall on frozen RELATE",
        "preregistered_claim": WALL_CLAIM,
        "cell_ids": list(CELL_IDS),
        "cell_ok": {r["cell"]: bool(r.get("ok")) for r in rows},
        "kill_alias_table": [dict(r) for r in KILL_ALIAS_TABLE],
        "query_life": QUERY_LIFE,
        "run_tm016aliaswall_sha": _sha_file(Path(__file__)),
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "genome_016_lock_sha": _sha_file(GENOME_016_LOCK),
        "relate_016_lock_sha": _sha_file(RELATE_LOCK),
        "refuse": [
            "operand identity / object continuity",
            "occlusion / gap persistence",
            "earn alias equivalence this pass",
            "alias machinery in organism",
            "unpinned RNG aliases",
            "shared S between Control and Kill",
            "TM.0.17",
            "FAMILY / LOOKAHEAD / pixels / 0.0.005",
        ],
    }


def write_alias_wall_lock(
    rows: list[dict[str, Any]], path: Path = ALIAS_WALL_LOCK
) -> dict[str, Any]:
    snap = alias_wall_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_alias_wall_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = ALIAS_WALL_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/alias_wall.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    live = {
        "run_tm016aliaswall_sha": _sha_file(Path(__file__)),
        "prereg_lock_sha": _sha_file(PREREG_LOCK),
        "genome_016_lock_sha": _sha_file(GENOME_016_LOCK),
        "relate_016_lock_sha": _sha_file(RELATE_LOCK),
        "earned_next": False,
        "ex0s": None,
        "cell_ids": list(CELL_IDS),
        "preregistered_claim": WALL_CLAIM,
        "query_life": QUERY_LIFE,
    }
    for key, val in live.items():
        if lock.get(key) != val:
            return False, f"alias_wall lock drift: {key}", lock
    if lock.get("kill_alias_table") != [dict(r) for r in KILL_ALIAS_TABLE]:
        return False, "alias table drift in freeze lock", lock
    if rows is not None:
        live_ok = {r["cell"]: bool(r.get("ok")) for r in rows}
        if live_ok != lock.get("cell_ok"):
            return False, "cell_ok drift", lock
        if not all(live_ok.get(c) for c in CELL_IDS):
            return False, "cell not all ok", lock
    return True, "alias_wall.lock intact", lock


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

    summary = run_alias_wall(seed=args.seed, write_locks=args.write_lock)
    out = _run_dir()
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
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
    d = REPO_ROOT / "runs" / f"{stamp}_tm016aliaswall"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
