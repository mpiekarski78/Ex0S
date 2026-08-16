"""TM.0.16.ALIASFINGER: opt-in behavioral fingerprint candidate.

Under frozen alias_evidence.prereg.lock. Product stays 0.0.004.
earned_next=false. ex0s=null even on 7/7. No TM.0.17.

Order: prereg → implement → candidate.lock (full surface) → score once →
freeze or publish failure against unchanged candidate.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import traverse_hold
from experiments.run_tm016aliaswall import (
    ALIAS_WALL_LOCK,
    KILL_ALIAS_TABLE,
    kill_schedule_from_table,
)
from experiments.run_tm016relate import (
    GENOME_016_LOCK,
    RELATE_LOCK,
    clear_by_source,
    empty_birth,
    make_relate,
    reload_store,
    run_lives,
    stash_by_source,
    restore_blobs,
    verify_genome_016,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy

PREREG_LOCK = REPO_ROOT / "docs" / "alias_finger.prereg.lock"
EVIDENCE_PREREG = REPO_ROOT / "docs" / "alias_evidence.prereg.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "alias_finger.candidate.lock"
ALIAS_FINGER_LOCK = REPO_ROOT / "docs" / "alias_finger.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm016aliasfinger_results.md"

DEFAULT_SEED = 12345
QUERY_CUE = "kelm"
QUERY_TARGET = "wift"
ACTION = "press"
OUTCOME = "success"
CTX_ALPHA = "ctx_alpha"
CTX_BETA = "ctx_beta"
CTX_GAMMA = "ctx_gamma"
SOURCE_FP = "experience_fingerprint"

CLAIM = (
    "An opt-in recipe can author raw behavioral fingerprint evidence from exact "
    "observe_alias_probe tuples and apply pairwise-qualified equivalence as a "
    "derived compose view so that convergent fingerprints complete an ALIASWALL "
    "Kill route without rewriting experience_skel, while weak, colliding, "
    "contradictory, or stripped evidence HOLDs."
)

CELL_IDS = (
    "A0_wall",
    "A1_weak",
    "A2_convergent",
    "A3_collision",
    "A4_swap",
    "A5_contradiction",
    "A6_causality",
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def make_finger(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    return make_relate(s_dir, policy, use_alias_fingerprint=True, **kwargs)


def fresh_world(tmp: Path, name: str, policy: UsePolicy, *, finger: bool = True) -> tuple[Path, Any]:
    s = tmp / name
    empty_birth(s)
    ag = make_finger(s, policy) if finger else make_relate(s, policy)
    ag.reset_rho()
    return s, ag


def role_groups(table: Sequence[dict[str, str]] = KILL_ALIAS_TABLE) -> dict[str, list[str]]:
    return {
        "origin": [r["origin"] for r in table],
        "mid": [r["mid"] for r in table],
        "dest": [r["dest"] for r in table],
    }


def emit_probe(ag: Any, alias: str, ctx: str, action: str = ACTION, outcome: str = OUTCOME) -> dict[str, Any]:
    return ag.observe_alias_probe(
        {
            "alias": alias,
            "probe_context": ctx,
            "action": action,
            "observed_outcome": outcome,
        }
    )


def fingerprint_group(
    ag: Any,
    aliases: Sequence[str],
    contexts: Sequence[str],
    *,
    action: str = ACTION,
    outcome: str = OUTCOME,
) -> None:
    for alias in aliases:
        for ctx in contexts:
            out = emit_probe(ag, alias, ctx, action=action, outcome=outcome)
            if not out.get("ok"):
                raise RuntimeError(f"fingerprint failed for {alias}/{ctx}: {out}")


def fingerprint_all_roles(ag: Any, contexts: Sequence[str] = (CTX_ALPHA, CTX_BETA)) -> None:
    """Within-role pairing only: distinct outcomes so origins ≁ mids ≁ dests."""
    groups = role_groups()
    fingerprint_group(ag, groups["origin"], contexts, outcome="origin_ok")
    fingerprint_group(ag, groups["mid"], contexts, outcome="mid_ok")
    fingerprint_group(ag, groups["dest"], contexts, outcome="dest_ok")


def fingerprint_foil_pair(ag: Any, aliases: Sequence[str], contexts: Sequence[str] = (CTX_ALPHA, CTX_BETA)) -> None:
    fingerprint_group(ag, aliases, contexts, outcome="foil_ok")


def strip_fingerprints(s: Path, ag: Any) -> int:
    n = clear_by_source(s, SOURCE_FP)
    reload_store(ag)
    return n


def list_fp_files(s: Path) -> list[Path]:
    out: list[Path] = []
    from three_memory.symbols import parse_tagfile

    for p in sorted(s.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") == SOURCE_FP:
            out.append(p)
    return out


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


def _holds(trav: dict[str, Any]) -> bool:
    lived = str(trav.get("lived_bind") or "").lower()
    return lived != QUERY_TARGET and (
        bool(trav.get("compose_hold")) or lived in ("", "none") or not trav.get("lived_pending")
    )


def _completes(trav: dict[str, Any]) -> bool:
    return (
        str(trav.get("lived_bind") or "").lower() == QUERY_TARGET
        and bool(trav.get("lived_pending"))
        and not trav.get("evidence_tie")
    )


# --- Cells --------------------------------------------------------------------


def cell_a0_wall(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "a0", policy)
    run_lives(ag, kill_schedule_from_table())
    ag.reset_rho()
    trav = traverse_hold(ag, QUERY_CUE)
    ok = _holds(trav)
    return _cell(
        "A0_wall",
        ok,
        lived_bind=trav.get("lived_bind"),
        compose_hold=bool(trav.get("compose_hold")),
        motor="hold" if ok else "other",
    )


def cell_a1_weak(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "a1", policy)
    run_lives(ag, kill_schedule_from_table())
    fingerprint_all_roles(ag, contexts=(CTX_ALPHA,))
    ag.reset_rho()
    trav = traverse_hold(ag, QUERY_CUE)
    origins = role_groups()["origin"]
    pair_kj = ag._fingerprint_pair_ok(origins[0], origins[1])
    ok = _holds(trav) and not pair_kj
    return _cell(
        "A1_weak",
        ok,
        lived_bind=trav.get("lived_bind"),
        compose_hold=bool(trav.get("compose_hold")),
        motor="hold" if ok else "other",
        no_pair=not pair_kj,
        n_fp=len(list_fp_files(s)),
    )


def cell_a2_convergent(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "a2", policy)
    run_lives(ag, kill_schedule_from_table())
    fingerprint_all_roles(ag, contexts=(CTX_ALPHA, CTX_BETA))
    skel_before = stash_by_source(s, "experience_skel")
    ag.reset_rho()
    trav = traverse_hold(ag, QUERY_CUE)
    completes = _completes(trav)
    skel_after = stash_by_source(s, "experience_skel")
    skel_untouched = skel_before == skel_after
    # Strip fingerprints only → HOLD again (behavioral, not diagnostic-only).
    n_strip = strip_fingerprints(s, ag)
    ag.reset_rho()
    trav2 = traverse_hold(ag, QUERY_CUE)
    holds_after = _holds(trav2)
    ok = completes and n_strip > 0 and holds_after and skel_untouched
    return _cell(
        "A2_convergent",
        ok,
        lived_bind=trav.get("lived_bind"),
        lived_pending=bool(trav.get("lived_pending")),
        strip_n=n_strip,
        strip_lived_bind=trav2.get("lived_bind"),
        strip_hold=holds_after,
        skel_untouched=skel_untouched,
    )


def cell_a3_collision(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "a3", policy)
    run_lives(ag, kill_schedule_from_table())
    groups = role_groups()
    # Same first consequence; second differs across origins so no origin pair qualifies.
    fingerprint_group(ag, groups["origin"], (CTX_ALPHA,), outcome="origin_ok")
    emit_probe(ag, groups["origin"][0], CTX_BETA, outcome="origin_ok")
    emit_probe(ag, groups["origin"][1], CTX_BETA, outcome="fail")
    emit_probe(ag, groups["origin"][2], CTX_BETA, outcome="other")
    fingerprint_group(ag, groups["mid"], (CTX_ALPHA, CTX_BETA), outcome="mid_ok")
    fingerprint_group(ag, groups["dest"], (CTX_ALPHA, CTX_BETA), outcome="dest_ok")
    ag.reset_rho()
    trav = traverse_hold(ag, QUERY_CUE)
    pair_kj = ag._fingerprint_pair_ok(groups["origin"][0], groups["origin"][1])
    pair_kd = ag._fingerprint_pair_ok(groups["origin"][0], groups["origin"][2])
    ok = _holds(trav) and not pair_kj and not pair_kd
    return _cell(
        "A3_collision",
        ok,
        lived_bind=trav.get("lived_bind"),
        pair_kelm_jasp=pair_kj,
        pair_kelm_doth=pair_kd,
        no_pair_origins=not pair_kj and not pair_kd,
        motor="hold" if _holds(trav) else "other",
        clique_kelm=sorted(ag._fingerprint_clique(QUERY_CUE) or []),
    )


def cell_a4_swap(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    # Fresh good world: convergent fingerprints → completion.
    s_good, ag_good = fresh_world(tmp, "a4_good", policy)
    run_lives(ag_good, kill_schedule_from_table())
    fingerprint_all_roles(ag_good, contexts=(CTX_ALPHA, CTX_BETA))
    ag_good.reset_rho()
    trav_good = traverse_hold(ag_good, QUERY_CUE)

    # Fresh swapped world: same skel; fingerprints pair kelm only with foil (no route).
    s_swap, ag_swap = fresh_world(tmp, "a4_swap", policy)
    run_lives(ag_swap, kill_schedule_from_table())
    foil = "zzzz"
    fingerprint_foil_pair(ag_swap, [QUERY_CUE, foil], (CTX_ALPHA, CTX_BETA))
    others = [a for a in role_groups()["origin"] if a != QUERY_CUE]
    fingerprint_group(ag_swap, others, (CTX_ALPHA, CTX_BETA), outcome="origin_ok")
    fingerprint_group(ag_swap, role_groups()["mid"], (CTX_ALPHA, CTX_BETA), outcome="mid_ok")
    fingerprint_group(ag_swap, role_groups()["dest"], (CTX_ALPHA, CTX_BETA), outcome="dest_ok")
    ag_swap.reset_rho()
    trav_swap = traverse_hold(ag_swap, QUERY_CUE)

    pair_foil = ag_swap._fingerprint_pair_ok(QUERY_CUE, foil)
    pair_jasp = ag_swap._fingerprint_pair_ok(QUERY_CUE, "jasp")
    clique = ag_swap._fingerprint_clique(QUERY_CUE)
    ok = (
        _completes(trav_good)
        and _holds(trav_swap)
        and pair_foil
        and not pair_jasp
        and clique == {QUERY_CUE, foil}
    )
    return _cell(
        "A4_swap",
        ok,
        good_lived=trav_good.get("lived_bind"),
        swap_lived=trav_swap.get("lived_bind"),
        behavior_follows_swapped_evidence=ok,
        swap_clique=sorted(clique or []),
        pair_kelm_zzzz=pair_foil,
        pair_kelm_jasp=pair_jasp,
    )


def cell_a5_contradiction(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "a5", policy)
    run_lives(ag, kill_schedule_from_table())
    fingerprint_all_roles(ag, contexts=(CTX_ALPHA, CTX_BETA))
    n_before = len(list_fp_files(s))
    # Later conflicting evidence in same S on kelm/ctx_alpha — retain both.
    conflict = emit_probe(ag, QUERY_CUE, CTX_ALPHA, outcome="fail")
    n_after = len(list_fp_files(s))
    ag.reset_rho()
    trav = traverse_hold(ag, QUERY_CUE)
    conflicted = CTX_ALPHA in ag._fingerprint_conflicted_contexts(QUERY_CUE)
    pair_kj = ag._fingerprint_pair_ok(QUERY_CUE, "jasp")
    retained = n_after == n_before + 1
    ok = conflict.get("ok") and conflicted and not pair_kj and _holds(trav) and retained
    return _cell(
        "A5_contradiction",
        ok,
        lived_bind=trav.get("lived_bind"),
        conflicted_context=conflicted,
        pair_kelm_jasp=pair_kj,
        retained_both=retained,
        motor="hold" if _holds(trav) else "other",
        n_fp=n_after,
    )


def cell_a6_causality(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    # Base convergent world.
    s, ag = fresh_world(tmp, "a6", policy)
    run_lives(ag, kill_schedule_from_table())
    fingerprint_all_roles(ag, contexts=(CTX_ALPHA, CTX_BETA))
    ag.reset_rho()
    trav0 = traverse_hold(ag, QUERY_CUE)
    # 1) Reset ρ, S retained → still completes.
    ag.reset_rho()
    trav_rho = traverse_hold(ag, QUERY_CUE)
    rho_ok = _completes(trav0) and _completes(trav_rho)

    # 2) Strip only fingerprints → HOLD.
    fp_blobs = stash_by_source(s, SOURCE_FP)
    strip_fingerprints(s, ag)
    ag.reset_rho()
    trav_strip = traverse_hold(ag, QUERY_CUE)
    strip_ok = _holds(trav_strip)

    # Restore fingerprints for donor swap setup.
    restore_blobs(s, fp_blobs)
    reload_store(ag)

    # 3) Two otherwise identical S stores; swap fingerprint donors.
    s_a, ag_a = fresh_world(tmp, "a6_a", policy)
    run_lives(ag_a, kill_schedule_from_table())
    fingerprint_all_roles(ag_a, contexts=(CTX_ALPHA, CTX_BETA))
    good_fps = stash_by_source(s_a, SOURCE_FP)

    s_b, ag_b = fresh_world(tmp, "a6_b", policy)
    run_lives(ag_b, kill_schedule_from_table())
    # Foil-only fingerprints on B (no convergent origin clique with route peers).
    fingerprint_foil_pair(ag_b, [QUERY_CUE, "zzzz"], (CTX_ALPHA, CTX_BETA))
    foil_fps = stash_by_source(s_b, SOURCE_FP)

    # Swap: A gets foil fps, B gets good fps.
    clear_by_source(s_a, SOURCE_FP)
    clear_by_source(s_b, SOURCE_FP)
    restore_blobs(s_a, foil_fps)
    restore_blobs(s_b, good_fps)
    reload_store(ag_a)
    reload_store(ag_b)
    ag_a.reset_rho()
    ag_b.reset_rho()
    trav_a = traverse_hold(ag_a, QUERY_CUE)
    trav_b = traverse_hold(ag_b, QUERY_CUE)
    swap_ok = _holds(trav_a) and _completes(trav_b)

    ok = rho_ok and strip_ok and swap_ok
    return _cell(
        "A6_causality",
        ok,
        reset_rho_retains_completion=rho_ok,
        strip_fingerprint_only_hold=strip_ok,
        swap_fingerprint_donors_follow_s=swap_ok,
        lived_rho=trav_rho.get("lived_bind"),
        lived_strip=trav_strip.get("lived_bind"),
        lived_a=trav_a.get("lived_bind"),
        lived_b=trav_b.get("lived_bind"),
    )


CELLS: Sequence[Callable[[Path, UsePolicy], dict[str, Any]]] = (
    cell_a0_wall,
    cell_a1_weak,
    cell_a2_convergent,
    cell_a3_collision,
    cell_a4_swap,
    cell_a5_contradiction,
    cell_a6_causality,
)


# --- Locks / verify -----------------------------------------------------------


def candidate_lock_snapshot() -> dict[str, Any]:
    return {
        "version": "TM.0.16.ALIASFINGER.CANDIDATE",
        "lab": "TM.0.16.ALIASFINGER",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "pre-score pin of full ALIASFINGER scored surface",
        "preregistered_claim": CLAIM,
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "observe_alias_probe_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_alias_probe),
        "compose_select_sha": _sha_src(agent_mod.ThreeMemoryAgent._compose_select),
        "make_finger_sha": _sha_src(make_finger),
        "make_relate_sha": _sha_src(make_relate),
        "run_tm016aliasfinger_sha": _sha_file(Path(__file__)),
        "run_tm054_sha": _sha_file(REPO_ROOT / "experiments" / "run_tm054.py"),
        "run_tm062_sha": _sha_file(REPO_ROOT / "experiments" / "run_tm062.py"),
        "cell_ids": list(CELL_IDS),
        "alias_evidence_prereg_sha": _sha_file(EVIDENCE_PREREG),
        "alias_finger_prereg_sha": _sha_file(PREREG_LOCK),
        "alias_wall_lock_sha": _sha_file(ALIAS_WALL_LOCK),
        "genome_016_lock_sha": _sha_file(GENOME_016_LOCK),
        "relate_016_lock_sha": _sha_file(RELATE_LOCK),
        "note": "Written BEFORE canonical A0–A6. Never rewrite after the scored run.",
    }


def write_candidate_lock(path: Path = CANDIDATE_LOCK) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(
            f"{path} already exists — refuse rewrite after candidate pin "
            "(repair requires a new candidate)"
        )
    snap = candidate_lock_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_candidate_lock(path: Path = CANDIDATE_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/alias_finger.candidate.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    snap = candidate_lock_snapshot()
    for key in (
        "agent_sha",
        "observe_alias_probe_sha",
        "compose_select_sha",
        "make_finger_sha",
        "make_relate_sha",
        "run_tm016aliasfinger_sha",
        "run_tm054_sha",
        "run_tm062_sha",
        "alias_evidence_prereg_sha",
        "alias_finger_prereg_sha",
        "alias_wall_lock_sha",
        "genome_016_lock_sha",
        "relate_016_lock_sha",
        "cell_ids",
        "earned_next",
        "ex0s",
        "lab",
        "ex0s_under_test",
    ):
        if lock.get(key) != snap.get(key):
            return False, f"candidate drift: {key}", lock
    return True, "alias_finger.candidate.lock matches scored surface", lock


def verify_prereg_lock(path: Path = PREREG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/alias_finger.prereg.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.16.ALIASFINGER":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "ex0s must be null", lock
    if lock.get("preregistered_claim") != CLAIM:
        return False, "claim drift", lock
    if lock.get("cell_ids") != list(CELL_IDS):
        return False, "cell_ids drift", lock
    if lock.get("alias_evidence_prereg_sha") != _sha_file(EVIDENCE_PREREG):
        return False, "alias_evidence_prereg_sha pin", lock
    if lock.get("alias_wall_lock_sha") != _sha_file(ALIAS_WALL_LOCK):
        return False, "alias_wall_lock_sha pin", lock
    if lock.get("genome_016_lock_sha") != _sha_file(GENOME_016_LOCK):
        return False, "genome_016_lock_sha pin", lock
    if lock.get("relate_016_lock_sha") != _sha_file(RELATE_LOCK):
        return False, "relate_016_lock_sha pin", lock
    banned = (
        "agent_sha",
        "run_tm016aliasfinger_sha",
        "make_finger_sha",
        "observe_alias_probe_sha",
        "candidate_lock_sha",
    )
    if any(k in lock for k in banned):
        return False, "prereg contains candidate SHAs", lock
    return True, "alias_finger.prereg.lock intact", lock


def alias_finger_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.16.ALIASFINGER",
        "lab": "TM.0.16.ALIASFINGER",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "opt-in behavioral fingerprint candidate",
        "preregistered_claim": CLAIM,
        "cell_ids": list(CELL_IDS),
        "cell_ok": {r["cell"]: bool(r.get("ok")) for r in rows},
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "observe_alias_probe_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_alias_probe),
        "make_finger_sha": _sha_src(make_finger),
        "run_tm016aliasfinger_sha": _sha_file(Path(__file__)),
        "candidate_lock_sha": _sha_file(CANDIDATE_LOCK),
        "alias_finger_prereg_sha": _sha_file(PREREG_LOCK),
        "alias_evidence_prereg_sha": _sha_file(EVIDENCE_PREREG),
        "alias_wall_lock_sha": _sha_file(ALIAS_WALL_LOCK),
        "genome_016_lock_sha": _sha_file(GENOME_016_LOCK),
        "relate_016_lock_sha": _sha_file(RELATE_LOCK),
        "refuse": [
            "rewrite alias_evidence.prereg.lock",
            "rewrite alias_finger.candidate.lock after score",
            "exact-key subset checks",
            "overwrite contradictory witnesses",
            "implicit transitive pooling",
            "destructive skel canonicalization",
            "A2 diagnostic-only",
            "A4 late inject into same S",
            "A6 whole-S wipe only",
            "candidate SHAs in alias_finger.prereg.lock",
            "TM.0.17 / 0.0.005",
            "FAMILY / LOOKAHEAD / pixels / object continuity",
            "ambiguous clique falling back to raw evidence",
        ],
    }


def write_alias_finger_lock(
    rows: list[dict[str, Any]], path: Path = ALIAS_FINGER_LOCK
) -> dict[str, Any]:
    snap = alias_finger_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_alias_finger_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = ALIAS_FINGER_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/alias_finger.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if rows is not None:
        expect = {r["cell"]: bool(r.get("ok")) for r in rows}
        if lock.get("cell_ok") != expect:
            return False, "cell_ok drift", lock
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "ex0s must be null", lock
    if lock.get("candidate_lock_sha") != _sha_file(CANDIDATE_LOCK):
        return False, "candidate_lock_sha pin", lock
    return True, "alias_finger.lock intact", lock


def run_alias_finger(
    *,
    seed: int = DEFAULT_SEED,
    write_candidate: bool = False,
    write_locks: bool = False,
) -> dict[str, Any]:
    policy = UsePolicy(seed=seed)
    if write_candidate:
        write_candidate_lock()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tm016aliasfinger_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            rows.append(fn(root, policy))

    n_ok = sum(1 for r in rows if r.get("ok"))
    summary: dict[str, Any] = {
        "version": "TM.0.16.ALIASFINGER",
        "lab": "TM.0.16.ALIASFINGER",
        "label": "opt-in behavioral fingerprint candidate",
        "ok": n_ok == len(rows) == 7,
        "n_ok": n_ok,
        "n_cells": len(rows),
        "earned_next": False,
        "ex0s": None,
        "seed": seed,
        "claim": CLAIM,
        "rows": rows,
        "candidate_lock": str(CANDIDATE_LOCK) if CANDIDATE_LOCK.exists() else None,
    }
    if write_locks:
        if summary["ok"]:
            write_alias_finger_lock(rows)
        else:
            # Publish failure against unchanged candidate lock — do not rewrite candidate.
            fail_path = REPO_ROOT / "docs" / "alias_finger.failure.json"
            fail_path.write_text(
                json.dumps(
                    {
                        "version": "TM.0.16.ALIASFINGER.FAILURE",
                        "earned_next": False,
                        "ex0s": None,
                        "n_ok": n_ok,
                        "n_cells": len(rows),
                        "rows": rows,
                        "candidate_lock_sha": _sha_file(CANDIDATE_LOCK)
                        if CANDIDATE_LOCK.exists()
                        else None,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return summary


def write_results_md(summary: dict[str, Any]) -> None:
    lines = [
        "# TM.0.16.ALIASFINGER results",
        "",
        f"**Recorded:** canonical A0–A6 → **{summary['n_ok']}/{summary['n_cells']}**",
        "",
        f"- `ok`: {summary['ok']}",
        f"- `earned_next`: false",
        f"- `ex0s`: null",
        f"- lab: `TM.0.16.ALIASFINGER`",
        "",
        "| Cell | ok | notes |",
        "|------|----|-------|",
    ]
    for r in summary["rows"]:
        note = r.get("lived_bind") or r.get("why") or ""
        lines.append(f"| {r['cell']} | {r.get('ok')} | {note} |")
    lines.extend(
        [
            "",
            "## Locks",
            "",
            "- Contract: [`alias_evidence.prereg.lock`](alias_evidence.prereg.lock)",
            "- Prereg: [`alias_finger.prereg.lock`](alias_finger.prereg.lock)",
            "- Candidate (pre-score): [`alias_finger.candidate.lock`](alias_finger.candidate.lock)",
            "- Freeze: [`alias_finger.lock`](alias_finger.lock)"
            if summary.get("ok")
            else "- Failure: [`alias_finger.failure.json`](alias_finger.failure.json)",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m experiments.run_tm016aliasfinger --verify-prereg",
            "python tests/test_tm016aliasfinger.py",
            "```",
            "",
        ]
    )
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def smoke_compat() -> dict[str, Any]:
    """HEAD fingerprint-off must keep RELATE/ALIASWALL behavior; exact keys refuse."""
    g_ok, g_why, _ = verify_genome_016()
    from experiments.run_tm016relate import LIVES_XA_Y, run_lives, run_relate
    from experiments.run_tm016aliaswall import run_alias_wall

    relate = run_relate(seed=12345, write_locks=False)
    wall = run_alias_wall(seed=12345, write_locks=False)

    with tempfile.TemporaryDirectory(prefix="tm016af_smoke_") as tmp:
        root = Path(tmp)
        s, ag = fresh_world(root, "smoke", UsePolicy(seed=1))
        bad = ag.observe_alias_probe({"alias": "x", "probe_context": "c", "action": "press"})
        extra = ag.observe_alias_probe(
            {
                "alias": "x",
                "probe_context": "c",
                "action": "press",
                "observed_outcome": "success",
                "role": "origin",
            }
        )
        off = make_relate(root / "off", UsePolicy(seed=1))
        off_probe = off.observe_alias_probe(
            {
                "alias": "x",
                "probe_context": "c",
                "action": "press",
                "observed_outcome": "success",
            }
        )
        # Ambiguous overlapping peers must HOLD under fingerprint mode (no raw fallback).
        # x↔p1 and x↔p2, but p1!~p2 → clique(x) is None → HOLD even on Control skel.
        s_amb, ag_amb = fresh_world(root, "amb", UsePolicy(seed=1))
        run_lives(ag_amb, LIVES_XA_Y)
        emit_probe(ag_amb, "x", "ctx_a", outcome="ok")
        emit_probe(ag_amb, "p1", "ctx_a", outcome="ok")
        emit_probe(ag_amb, "p2", "ctx_a", outcome="ok")
        emit_probe(ag_amb, "x", "ctx_b", outcome="ok")
        emit_probe(ag_amb, "p1", "ctx_b", outcome="ok")
        emit_probe(ag_amb, "p2", "ctx_b", outcome="other")
        emit_probe(ag_amb, "x", "ctx_c", outcome="ok")
        emit_probe(ag_amb, "p1", "ctx_c", outcome="other")
        emit_probe(ag_amb, "p2", "ctx_c", outcome="ok")
        clique = ag_amb._fingerprint_clique("x")
        pair_xp1 = ag_amb._fingerprint_pair_ok("x", "p1")
        pair_xp2 = ag_amb._fingerprint_pair_ok("x", "p2")
        pair_p1p2 = ag_amb._fingerprint_pair_ok("p1", "p2")
        ag_amb.reset_rho()
        trav_amb = traverse_hold(ag_amb, "x")

    return {
        "genome_016": g_ok,
        "genome_why": g_why,
        "relate_16": bool(relate.get("ok")) and relate.get("n_ok") == 16,
        "aliaswall_6": bool(wall.get("ok")) and wall.get("n_ok") == 6,
        "exact_reject_missing": bad.get("why") == "exact_key_reject",
        "exact_reject_extra": extra.get("why") == "exact_key_reject",
        "off_is_noop": off_probe.get("why") == "fingerprint_off",
        "flag_default_false": off.use_alias_fingerprint is False,
        "ambiguous_pair_xp1": pair_xp1,
        "ambiguous_pair_xp2": pair_xp2,
        "ambiguous_pair_p1p2": pair_p1p2,
        "ambiguous_clique_none": clique is None,
        "ambiguous_holds": _holds(trav_amb),
        "ambiguous_lived": trav_amb.get("lived_bind"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.verify_prereg:
        ok, why, _ = verify_prereg_lock()
        print(json.dumps({"ok": ok, "why": why}, indent=2))
        sys.exit(0 if ok else 1)

    if args.smoke:
        print(json.dumps(smoke_compat(), indent=2))
        return

    summary = run_alias_finger(
        seed=args.seed,
        write_candidate=args.write_candidate,
        write_locks=args.write_lock,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for r in summary["rows"]:
        print(f"  {r['cell']}: {r.get('ok')} { {k: v for k, v in r.items() if k not in ('cell', 'ok')} }")
    if args.write_lock:
        write_results_md(summary)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
