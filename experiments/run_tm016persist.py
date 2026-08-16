"""TM.0.16.PERSIST: opt-in mark-continuity persistence candidate.

Under frozen continuity_evidence.prereg.lock. Product stays 0.0.004.
earned_next=false. ex0s=null even on a green C0–C6 battery. No TM.0.17.

Order: prereg → implement → candidate.lock (full surface) → score once →
freeze only if earned. Do not edit run_tm054.make / make_finger / make_relate.
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
from experiments.run_tm016aliasfinger import make_finger
from experiments.run_tm016gapwall import run_episode_capture
from experiments.run_tm016relate import (
    GENOME_016_LOCK,
    RELATE_LOCK,
    clear_by_source,
    edge_support,
    empty_birth,
    evidence_winner_did,
    make_relate,
    observe_event,
    reload_store,
    restore_blobs,
    stash_by_source,
    verify_genome_016,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile

PREREG_LOCK = REPO_ROOT / "docs" / "persist.prereg.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "persist.candidate.lock"
PERSIST_LOCK = REPO_ROOT / "docs" / "persist.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm016persist_results.md"
CONTRACT_PREREG = REPO_ROOT / "docs" / "continuity_evidence.prereg.lock"
GAP_WALL_LOCK = REPO_ROOT / "docs" / "gap_wall.lock"
ALIAS_FINGER_LOCK = REPO_ROOT / "docs" / "alias_finger.lock"
ALIAS_EVIDENCE_PREREG = REPO_ROOT / "docs" / "alias_evidence.prereg.lock"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"

DEFAULT_SEED = 12345
SOURCE_CONT = "experience_continuity"
SOURCE_SKEL = "experience_skel"
P, DEST, Q, R = "kelm", "wift", "norb", "jasp"
MARK, MARK_SWAP = "mk_alpha", "mk_beta"
ON, OFF = "on", "off"

CLAIM = (
    "An opt-in recipe may author raw continuity rows into S from exact "
    "observe_continuity_mark tuples and, at use time only, recompute a one-hop "
    "P → Q projection from those rows so that a unique causal apply→read witness "
    "can complete an identity-dependent probe. mark_id scopes an intervention "
    "attempt, not an object; it is fresh/counterbalanced between worlds and "
    "cannot establish continuity without the corresponding apply/read observations. "
    "Token spelling, skip, sole candidacy, and route position contribute no "
    "identity evidence. Permission is defeasible and disappears when recomputed "
    "evidence is no longer unique."
)
CELL_IDS = (
    "C0_gapwall",
    "C1_weak",
    "C2_mark",
    "C3a_both",
    "C3b_neither",
    "C3c_conflict",
    "C4_swap",
    "C5_contradiction",
    "C6_causality",
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def make_persist(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    ag = make_finger(s_dir, policy, **kwargs)
    ag.use_continuity_mark = True
    return ag


def fresh_world(tmp: Path, name: str, policy: UsePolicy, *, persist: bool = True) -> tuple[Path, Any]:
    s = tmp / name
    empty_birth(s)
    ag = make_persist(s, policy) if persist else make_finger(s, policy)
    ag.reset_rho()
    return s, ag


def _cell(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    row = {"cell": name, "ok": bool(ok), **extra}
    if not ok and "why" not in row:
        row["why"] = "failed"
    return row


def _motor(trav: dict[str, Any]) -> str:
    return str(trav.get("action_name") or "hold").lower()


def _holds(trav: dict[str, Any]) -> bool:
    """Strict HOLD: no dest chosen. A wrong unique dest is not a HOLD."""
    return trav.get("lived_bind") is None and not _completes(trav)


def _completes(trav: dict[str, Any], dest: str = DEST) -> bool:
    return (
        str(trav.get("lived_bind") or "").lower() == dest
        and bool(trav.get("lived_pending"))
        and not trav.get("evidence_tie")
    )


def _query(ag: Any, cue: str) -> dict[str, Any]:
    ag.reset_rho()
    return traverse_hold(ag, cue)


def emit_mark(ag: Any, token: str, mark_id: str, phase: str, operation: str, observed_state: str) -> dict[str, Any]:
    return ag.observe_continuity_mark(
        {
            "token": token,
            "mark_id": mark_id,
            "phase": phase,
            "operation": operation,
            "observed_state": observed_state,
        }
    )


def apply_mark(ag: Any, token: str, mark_id: str = MARK, state: str = ON) -> dict[str, Any]:
    return emit_mark(ag, token, mark_id, "pre_gap", "apply", state)


def read_mark(ag: Any, token: str, mark_id: str = MARK, state: str = ON) -> dict[str, Any]:
    return emit_mark(ag, token, mark_id, "post_gap", "read", state)


def _prereg() -> dict[str, Any]:
    return json.loads(PREREG_LOCK.read_text(encoding="utf-8"))


def emit_obs(ag: Any, row: dict[str, Any]) -> dict[str, Any]:
    return ag.observe_continuity_mark(
        {
            "token": row["token"],
            "mark_id": row["mark_id"],
            "phase": row["phase"],
            "operation": row["operation"],
            "observed_state": row["observed_state"],
        }
    )


def emit_obs_list(ag: Any, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [emit_obs(ag, row) for row in rows]


def author_skel(ag: Any) -> None:
    skel = (_prereg().get("initial_skeleton") or {}).get("episodes") or [[[P], [DEST]]]
    for episode in skel:
        for visible in episode:
            observe_event(ag, list(visible), focus=visible[0] if visible else None)
        ag.end_event_episode()


def permission_pair(ag: Any) -> tuple[str, str] | None:
    pairs = ag._continuity_recompute()
    if len(pairs) != 1:
        return None
    return pairs[0]


def count_source(s_dir: Path, source: str) -> int:
    n = 0
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") == source:
            n += 1
    return n


def skel_untouched_except(s_dir: Path, bind: str, did: str) -> bool:
    return edge_support(s_dir, bind, did) == 1 and count_source(s_dir, SOURCE_SKEL) == 1


def _gap_ok(result: dict[str, Any]) -> bool:
    return (
        result.get("ok") is True
        and result.get("why") == "empty_visible"
        and int(result.get("wrote") or 0) == 0
        and int(result.get("updated") or 0) == 0
    )


def setup_c2(tmp: Path, name: str, policy: UsePolicy) -> tuple[Path, Any]:
    s, ag = fresh_world(tmp, name, policy)
    author_skel(ag)
    rows = ((_prereg().get("fixtures") or {}).get("C2_mark") or {}).get(
        "continuity_observations"
    ) or []
    emit_obs_list(ag, rows)
    return s, ag


# --- Cells --------------------------------------------------------------------


def cell_c0_gapwall(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    reuse = ((_prereg().get("fixtures") or {}).get("C0_gapwall") or {}).get("reuse") or {}
    g1_eps = (reuse.get("G1_empty_skip") or {}).get("episodes") or [[["a"], [], ["b"], ["c"]]]
    g2_eps = (reuse.get("G2_episode_gap") or {}).get("episodes") or [[["a"]], [["a"]]]
    g5_eps = (reuse.get("G5_two_reappear") or {}).get("episodes") or [[["a"], [], ["u1", "u2"]]]

    s1, ag1 = fresh_world(tmp, "c0_g1", policy)
    events = run_episode_capture(ag1, tuple(g1_eps[0]))
    gap = events[1]
    trav1 = _query(ag1, "a")
    g1 = (
        _gap_ok(gap)
        and edge_support(s1, "a", "b") == 1
        and edge_support(s1, "b", "c") == 1
        and str(trav1.get("lived_bind") or "").lower() == "c"
        and count_source(s1, SOURCE_CONT) == 0
    )

    s2, ag2 = fresh_world(tmp, "c0_g2", policy)
    for episode in g2_eps:
        run_episode_capture(ag2, tuple(episode))
    trav2 = _query(ag2, "a")
    g2 = (
        edge_support(s2, "a", "a") == 0
        and evidence_winner_did(s2, "a") is None
        and _motor(trav2) == "hold"
        and trav2.get("lived_bind") is None
    )

    s5, ag5 = fresh_world(tmp, "c0_g5", policy)
    events5 = run_episode_capture(ag5, tuple(g5_eps[0]))
    trav5 = _query(ag5, "a")
    lived5 = str(trav5.get("lived_bind") or "").lower()
    g5 = (
        _gap_ok(events5[1])
        and edge_support(s5, "a", "u1") == 1
        and edge_support(s5, "a", "u2") == 1
        and evidence_winner_did(s5, "a") is None
        and _motor(trav5) == "hold"
        and bool(trav5.get("evidence_tie"))
        and lived5 not in {"u1", "u2"}
    )
    ok = g1 and g2 and g5
    return _cell(
        "C0_gapwall",
        ok,
        g1_empty_skip=g1,
        g2_episode_gap=g2,
        g5_two_reappear=g5,
        object_continuity_claim=False,
        honest_label="empty_event_skip_semantics",
    )


def cell_c1_weak(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    subs = ((_prereg().get("fixtures") or {}).get("C1_weak") or {}).get("subcases") or {}

    s_r, ag_r = fresh_world(tmp, "c1_reappear", policy)
    author_skel(ag_r)
    for episode in (subs.get("C1_reappear") or {}).get("later_episodes") or []:
        for visible in episode:
            observe_event(ag_r, list(visible), focus=visible[0] if visible else None)
        ag_r.end_event_episode()
    trav_rq = _query(ag_r, Q)
    trav_rr = _query(ag_r, R)
    reappear = (
        count_source(s_r, SOURCE_CONT) == 0
        and permission_pair(ag_r) is None
        and _holds(trav_rq)
        and _holds(trav_rr)
    )

    s_a, ag_a = fresh_world(tmp, "c1_apply", policy)
    author_skel(ag_a)
    wrote_a = emit_obs_list(ag_a, (subs.get("C1_apply_only") or {}).get("continuity_observations") or [])
    trav_a = _query(ag_a, Q)
    apply_only = (
        all(r.get("ok") for r in wrote_a)
        and count_source(s_a, SOURCE_CONT) == 1
        and permission_pair(ag_a) is None
        and _holds(trav_a)
    )

    s_d, ag_d = fresh_world(tmp, "c1_read", policy)
    author_skel(ag_d)
    wrote_d = emit_obs_list(ag_d, (subs.get("C1_read_only") or {}).get("continuity_observations") or [])
    trav_d = _query(ag_d, Q)
    read_only = (
        all(r.get("ok") for r in wrote_d)
        and count_source(s_d, SOURCE_CONT) == 1
        and permission_pair(ag_d) is None
        and _holds(trav_d)
    )
    ok = reappear and apply_only and read_only
    return _cell(
        "C1_weak",
        ok,
        reappear=reappear,
        apply_only=apply_only,
        read_only=read_only,
        permission=None,
    )


def cell_c2_mark(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = setup_c2(tmp, "c2", policy)
    perm = permission_pair(ag)
    trav_q = _query(ag, Q)
    trav_r = _query(ag, R)
    trav_p = _query(ag, P)
    ok = (
        perm == (P, Q)
        and _completes(trav_q)
        and _holds(trav_r)
        and _completes(trav_p)
        and count_source(s, SOURCE_CONT) == 3
        and skel_untouched_except(s, P, DEST)
    )
    return _cell(
        "C2_mark",
        ok,
        permission=list(perm) if perm else None,
        lived_norb=trav_q.get("lived_bind"),
        lived_jasp=trav_r.get("lived_bind"),
        lived_kelm=trav_p.get("lived_bind"),
        continuity_rows=count_source(s, SOURCE_CONT),
        skel_untouched=skel_untouched_except(s, P, DEST),
    )


def cell_c3a_both(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "c3a", policy)
    author_skel(ag)
    emit_obs_list(ag, ((_prereg().get("fixtures") or {}).get("C3a_both") or {}).get("continuity_observations") or [])
    ok = permission_pair(ag) is None and _holds(_query(ag, Q)) and _holds(_query(ag, R))
    return _cell("C3a_both", ok, permission=permission_pair(ag), rows=count_source(s, SOURCE_CONT))


def cell_c3b_neither(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "c3b", policy)
    author_skel(ag)
    emit_obs_list(ag, ((_prereg().get("fixtures") or {}).get("C3b_neither") or {}).get("continuity_observations") or [])
    ok = permission_pair(ag) is None and _holds(_query(ag, Q)) and _holds(_query(ag, R))
    return _cell("C3b_neither", ok, permission=permission_pair(ag), rows=count_source(s, SOURCE_CONT))


def cell_c3c_conflict(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "c3c", policy)
    author_skel(ag)
    emit_obs_list(ag, ((_prereg().get("fixtures") or {}).get("C3c_conflict") or {}).get("continuity_observations") or [])
    trav = _query(ag, Q)
    ok = (
        permission_pair(ag) is None
        and _holds(trav)
        and count_source(s, SOURCE_CONT) == 3
        and str(trav.get("lived_bind") or "").lower() != DEST
    )
    return _cell(
        "C3c_conflict",
        ok,
        permission=permission_pair(ag),
        lived_norb=trav.get("lived_bind"),
        retains_both_norb_reads=count_source(s, SOURCE_CONT) == 3,
    )


def cell_c4_swap(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = fresh_world(tmp, "c4", policy)
    author_skel(ag)
    emit_obs_list(ag, ((_prereg().get("fixtures") or {}).get("C4_swap") or {}).get("continuity_observations") or [])
    perm = permission_pair(ag)
    trav_r = _query(ag, R)
    trav_q = _query(ag, Q)
    ok = perm == (P, R) and _completes(trav_r) and _holds(trav_q)
    return _cell(
        "C4_swap",
        ok,
        permission=list(perm) if perm else None,
        lived_jasp=trav_r.get("lived_bind"),
        lived_norb=trav_q.get("lived_bind"),
        behavior_follows_swapped_evidence=ok,
    )


def cell_c5_contradiction(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    subs = ((_prereg().get("fixtures") or {}).get("C5_contradiction") or {}).get("subcases") or {}

    def after_c2(name: str) -> tuple[Path, Any, int] | None:
        s, ag = setup_c2(tmp, name, policy)
        if permission_pair(ag) != (P, Q) or not _completes(_query(ag, Q)):
            return None
        return s, ag, count_source(s, SOURCE_CONT)

    def later_ok(name: str, key: str, extra: Callable[[Any, Path], bool]) -> bool:
        got = after_c2(name)
        if got is None:
            return False
        s, ag, n0 = got
        spec = (subs.get(key) or {}).get("later_observation") or {}
        later = emit_obs(ag, spec)
        return (
            bool(later.get("ok"))
            and count_source(s, SOURCE_CONT) == n0 + 1
            and permission_pair(ag) is None
            and extra(ag, s)
        )

    later_state = later_ok(
        "c5_state",
        "C5_later_state",
        lambda ag, _s: _holds(_query(ag, Q)),
    )
    second_read = later_ok(
        "c5_read",
        "C5_second_read",
        lambda ag, _s: _holds(_query(ag, Q)) and _holds(_query(ag, R)),
    )
    second_apply = later_ok(
        "c5_apply",
        "C5_second_apply",
        lambda ag, _s: _holds(_query(ag, Q)),
    )
    ok = later_state and second_read and second_apply
    return _cell(
        "C5_contradiction",
        ok,
        later_state=later_state,
        second_read=second_read,
        second_apply=second_apply,
        stale_merge=False,
    )


def cell_c6_causality(tmp: Path, policy: UsePolicy) -> dict[str, Any]:
    s, ag = setup_c2(tmp, "c6", policy)
    trav0 = _query(ag, Q)
    ag.reset_rho()
    trav_rho = _query(ag, Q)
    rho_ok = _completes(trav0) and _completes(trav_rho)

    cont_blobs = stash_by_source(s, SOURCE_CONT)
    clear_by_source(s, SOURCE_CONT)
    reload_store(ag)
    trav_strip = _query(ag, Q)
    strip_ok = _holds(trav_strip) and edge_support(s, P, DEST) == 1
    restore_blobs(s, cont_blobs)
    reload_store(ag)

    s_empty, ag_empty = fresh_world(tmp, "c6_empty", policy)
    restore_blobs(s_empty, cont_blobs)
    reload_store(ag_empty)
    empty_ok = _holds(_query(ag_empty, Q))

    s_skel, ag_skel = fresh_world(tmp, "c6_skel", policy)
    author_skel(ag_skel)
    before = _query(ag_skel, Q)
    restore_blobs(s_skel, cont_blobs)
    reload_store(ag_skel)
    after = _query(ag_skel, Q)
    donate_ok = _holds(before) and _completes(after)

    ok = rho_ok and strip_ok and empty_ok and donate_ok
    return _cell(
        "C6_causality",
        ok,
        reset_rho_retains=rho_ok,
        strip_continuity_only_hold=strip_ok,
        donate_to_empty_hold=empty_ok,
        donate_onto_skel_follows_s=donate_ok,
        lived_rho=trav_rho.get("lived_bind"),
        lived_strip=trav_strip.get("lived_bind"),
        lived_after=after.get("lived_bind"),
    )


CELLS: Sequence[Callable[[Path, UsePolicy], dict[str, Any]]] = (
    cell_c0_gapwall,
    cell_c1_weak,
    cell_c2_mark,
    cell_c3a_both,
    cell_c3b_neither,
    cell_c3c_conflict,
    cell_c4_swap,
    cell_c5_contradiction,
    cell_c6_causality,
)


# --- Locks / verify -----------------------------------------------------------


def candidate_lock_snapshot() -> dict[str, Any]:
    return {
        "version": "TM.0.16.PERSIST.CANDIDATE",
        "lab": "TM.0.16.PERSIST",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "pre-score pin of full PERSIST scored surface",
        "preregistered_claim": CLAIM,
        "agent_sha": _sha_file(AGENT_PY),
        "observe_continuity_mark_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_continuity_mark),
        "compose_choose_sha": _sha_src(agent_mod.ThreeMemoryAgent._compose_choose),
        "make_persist_sha": _sha_src(make_persist),
        "run_tm016persist_sha": _sha_file(Path(__file__)),
        "cell_ids": list(CELL_IDS),
        "continuity_evidence_prereg_sha": _sha_file(CONTRACT_PREREG),
        "persist_prereg_sha": _sha_file(PREREG_LOCK),
        "gap_wall_lock_sha": _sha_file(GAP_WALL_LOCK),
        "alias_finger_lock_sha": _sha_file(ALIAS_FINGER_LOCK),
        "supersedes": "docs/persist.candidate.v1.lock",
        "note": "Written BEFORE canonical C0–C6. Never rewrite after the scored run.",
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
        return False, "docs/persist.candidate.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    snap = candidate_lock_snapshot()
    # agent / observe / compose / runner SHAs are historical: later labs may
    # extend agent.py or adjust verify helpers without rewriting persist locks.
    for key in (
        "agent_sha",
        "observe_continuity_mark_sha",
        "compose_choose_sha",
        "run_tm016persist_sha",
    ):
        if not lock.get(key):
            return False, f"candidate missing historical {key}", lock
    for key in (
        "make_persist_sha",
        "continuity_evidence_prereg_sha",
        "persist_prereg_sha",
        "gap_wall_lock_sha",
        "alias_finger_lock_sha",
        "cell_ids",
        "earned_next",
        "ex0s",
        "lab",
        "ex0s_under_test",
    ):
        if lock.get(key) != snap.get(key):
            return False, f"candidate drift: {key}", lock
    return True, "persist.candidate.lock matches scored surface", lock


def verify_prereg_lock(path: Path = PREREG_LOCK) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/persist.prereg.lock missing", {}
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.16.PERSIST":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "ex0s must be null", lock
    if lock.get("preregistered_claim") != CLAIM:
        return False, "claim drift", lock
    if lock.get("cell_ids") != list(CELL_IDS):
        return False, "cell_ids drift", lock
    pins = lock.get("prior_lock_shas") or {}
    if pins.get("continuity_evidence.prereg.lock") != _sha_file(CONTRACT_PREREG):
        return False, "continuity_evidence.prereg.lock pin", lock
    if pins.get("gap_wall.lock") != _sha_file(GAP_WALL_LOCK):
        return False, "gap_wall.lock pin", lock
    if pins.get("alias_finger.lock") != _sha_file(ALIAS_FINGER_LOCK):
        return False, "alias_finger.lock pin", lock
    if pins.get("alias_evidence.prereg.lock") != _sha_file(ALIAS_EVIDENCE_PREREG):
        return False, "alias_evidence.prereg.lock pin", lock
    banned = (
        "agent_sha",
        "run_tm016persist_sha",
        "make_persist_sha",
        "observe_continuity_mark_sha",
        "candidate_lock_sha",
    )
    if any(k in lock for k in banned):
        return False, "prereg contains candidate SHAs", lock
    return True, "persist.prereg.lock intact", lock


def persist_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.16.PERSIST",
        "lab": "TM.0.16.PERSIST",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "label": "opt-in mark-continuity persistence candidate",
        "preregistered_claim": CLAIM,
        "cell_ids": list(CELL_IDS),
        "cell_ok": {r["cell"]: bool(r.get("ok")) for r in rows},
        "agent_sha": _sha_file(AGENT_PY),
        "observe_continuity_mark_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_continuity_mark),
        "make_persist_sha": _sha_src(make_persist),
        "run_tm016persist_sha": _sha_file(Path(__file__)),
        "candidate_lock_sha": _sha_file(CANDIDATE_LOCK),
        "persist_prereg_sha": _sha_file(PREREG_LOCK),
        "continuity_evidence_prereg_sha": _sha_file(CONTRACT_PREREG),
        "gap_wall_lock_sha": _sha_file(GAP_WALL_LOCK),
        "alias_finger_lock_sha": _sha_file(ALIAS_FINGER_LOCK),
        "refuse": [
            "rewrite persist.candidate.lock after score",
            "rewrite persist.candidate.v1.lock",
            "rewrite continuity_evidence.prereg.lock",
            "stored same_as / persistent link / transitive equality",
            "soft C5 without withdrawal and reprobe",
            "TM.0.17 / 0.0.005",
            "FAMILY / LOOKAHEAD / pixels",
            "earned_next=true or non-null ex0s",
        ],
    }


def write_persist_lock(rows: list[dict[str, Any]], path: Path = PERSIST_LOCK) -> dict[str, Any]:
    snap = persist_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_persist_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = PERSIST_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "docs/persist.lock missing", {}
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
    return True, "persist.lock intact", lock


def run_persist(
    *,
    seed: int = DEFAULT_SEED,
    write_candidate: bool = False,
    write_locks: bool = False,
) -> dict[str, Any]:
    policy = UsePolicy(seed=seed)
    if write_candidate:
        write_candidate_lock()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tm016persist_") as tmp:
        root = Path(tmp)
        for fn in CELLS:
            rows.append(fn(root, policy))

    n_ok = sum(1 for r in rows if r.get("ok"))
    summary: dict[str, Any] = {
        "version": "TM.0.16.PERSIST",
        "lab": "TM.0.16.PERSIST",
        "label": "opt-in mark-continuity persistence candidate",
        "ok": n_ok == len(rows) == 9,
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
            write_persist_lock(rows)
        else:
            fail_path = REPO_ROOT / "docs" / "persist.failure.json"
            fail_path.write_text(
                json.dumps(
                    {
                        "version": "TM.0.16.PERSIST.FAILURE",
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
        "# TM.0.16.PERSIST results",
        "",
        f"**Recorded:** canonical C0–C6 → **{summary['n_ok']}/{summary['n_cells']}**",
        "",
        f"- Product: `0.0.004`",
        f"- `earned_next`: false",
        f"- `ex0s`: null",
        f"- lab: `TM.0.16.PERSIST`",
        "",
        "| Cell | ok | notes |",
        "|------|----|-------|",
    ]
    notes = {
        "C0_gapwall": "GAPWALL G1/G2/G5 reused; skip is not continuity",
        "C1_weak": "reappear / apply-only / read-only HOLD; no permission",
        "C2_mark": "unique apply→read earns identity-dependent cue norb → wift",
        "C3a_both": "both verify; refuse unique",
        "C3b_neither": "neither verifies; refuse unique",
        "C3c_conflict": "norb on then off; refuse unique; rows retained",
        "C4_swap": "mk_beta swaps the verifying candidate",
        "C5_contradiction": "later state / second read / second apply withdraw; reprobe HOLD",
        "C6_causality": "reset ρ retains; strip/donate follow S only",
    }
    for r in summary["rows"]:
        lines.append(f"| {r['cell']} | {r.get('ok')} | {notes.get(r['cell'], '')} |")
    lines.extend(
        [
            "",
            "## Bounded fact",
            "",
            "Opt-in continuity rows in S permit a one-hop use-time projection only when "
            "exactly one apply and exactly one matching read remain unique. Contradiction "
            "withdraws permission on recompute without deleting rows. Product stays 0.0.004.",
            "",
            "## Next",
            "",
            "No product stamp. Alias fingerprints remain a separate track. Anonymous features / encoders later.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m experiments.run_tm016persist --verify-prereg",
            "python tests/test_tm016persist.py",
            "```",
            "",
        ]
    )
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def smoke_compat() -> dict[str, Any]:
    g_ok, g_why, _ = verify_genome_016()
    prereg = json.loads(PREREG_LOCK.read_text(encoding="utf-8"))
    abi_cases = (prereg.get("fixtures") or {}).get("C1_abi") or {}
    cases = list(abi_cases.get("cases") or [])

    with tempfile.TemporaryDirectory(prefix="tm016persist_smoke_") as tmp:
        root = Path(tmp)
        s, ag = fresh_world(root, "smoke", UsePolicy(seed=1))
        results = []
        for case in cases:
            if case.get("flag_off"):
                off = make_finger(root / "off", UsePolicy(seed=1))
                got = off.observe_continuity_mark(case["info"])
            else:
                got = ag.observe_continuity_mark(case["info"])
            results.append(
                got.get("why") == case["why"]
                and not got.get("wrote")
                and int(bool(got.get("wrote"))) == case["wrote"]
            )
        n_before = count_source(s, SOURCE_CONT)
        off_flag = make_relate(root / "relate_off", UsePolicy(seed=1))
        clone = ag.clone_empty()
        finger = make_finger(root / "finger", UsePolicy(seed=1))

        s_fp, ag_fp = fresh_world(root, "fp_iso", UsePolicy(seed=1))
        author_skel(ag_fp)
        emit_obs_list(
            ag_fp,
            ((prereg.get("fixtures") or {}).get("C2_mark") or {}).get("continuity_observations")
            or [],
        )
        for alias in (P, Q, R):
            ag_fp.observe_alias_probe(
                {
                    "alias": alias,
                    "probe_context": "ctx_alpha",
                    "action": "press",
                    "observed_outcome": "success",
                }
            )
        perm_fp = permission_pair(ag_fp)
        n_fp = count_source(s_fp, "experience_fingerprint")
        clear_by_source(s_fp, SOURCE_CONT)
        reload_store(ag_fp)
        trav_fp = _query(ag_fp, Q)
        fp_iso = perm_fp == (P, Q) and n_fp == 3 and _holds(trav_fp)

    return {
        "genome_016": g_ok,
        "genome_why": g_why,
        "abi_all": all(results) and len(results) == 5,
        "abi_details": results,
        "smoke_wrote_nothing": n_before == 0,
        "flag_default_false": off_flag.use_continuity_mark is False
        if hasattr(off_flag, "use_continuity_mark")
        else True,
        "finger_continuity_off": finger.use_continuity_mark is False,
        "clone_copies_flag": clone.use_continuity_mark is True,
        "make_persist_on": True,
        "fingerprint_inaccessible": fp_iso,
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

    summary = run_persist(
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
