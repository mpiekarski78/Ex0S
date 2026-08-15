"""TM.0.13.CONTEXT: first cognitive function in M — provenance-sensitive compose.

Freeze κ contract (docs/kappa_013.lock). Carry (Y, κ) after selected non-motor
hops. Plant ctx via independent reference_route_kappa. Causal evidence routes.
genome_011.lock immutable. No Ex0S 0.0.004. earned_next always false.
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
from experiments.run_tm011family import (
    ORGANISM_BASELINE_COMMIT,
    verify_011_compatibility,
    verify_historical_freeze,
)
from experiments.run_tm040 import probe
from three_memory.kappa import CTX_ENCODING, route_kappa_hops
from three_memory.kappa import edge_sem as live_edge_sem
from three_memory.kappa import kappa_seed as live_kappa_seed
from three_memory.kappa import kappa_step as live_kappa_step
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile

MOTORS = ("press", "tune", "flip")
BANNED = frozenset(MOTORS + ("hold", "idle", "push", "adjust", "open", "wait", "use", "pick"))
CONS = "bcdfghjklmnpqrstvwxz"
VOW = "aeiou"
HERE = "chb"

KAPPA_LOCK = REPO_ROOT / "docs" / "kappa_013.lock"
CONTEXT_LOCK = REPO_ROOT / "docs" / "context_013.lock"
GENOME_013_LOCK = REPO_ROOT / "docs" / "genome_013.lock"
GENOME_011_LOCK = REPO_ROOT / "docs" / "genome_011.lock"
DEFAULT_SEED = 12345


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def _motor(name: str) -> str:
    return str(name or "hold").lower()


def _nonce(rng: np.random.Generator, taken: set[str]) -> str:
    while True:
        w = "".join(str(rng.choice(list(CONS))) + str(rng.choice(list(VOW))) for _ in range(2))
        if w not in BANNED and w not in taken:
            taken.add(w)
            return w


def _fid(rng: np.random.Generator, taken: set[str]) -> str:
    while True:
        name = f"n{int(rng.integers(0, 100_000)):05d}"
        if name not in taken:
            taken.add(name)
            return name


# --- Independent planter oracle (must not be three_memory.kappa) ---------------
# Deliberately duplicated F so a shared live bug cannot plant and retrieve the
# same wrong ctx. Known-answer vectors gate both sides.


def reference_edge_sem(bind: str, did: str) -> str:
    return bind.lower() + "\0" + did.lower()


def reference_kappa_seed(origin: str) -> str:
    return _sha_bytes(b"origin\0" + origin.lower().encode())


def reference_kappa_step(previous_kappa: str, traversed_token: str) -> str:
    return _sha_bytes(previous_kappa.encode() + b"\0" + traversed_token.encode())


def reference_route_kappa(origin: str, hops: Sequence[tuple[str, str]]) -> str:
    """Planting oracle — independent of three_memory.kappa."""
    k = reference_kappa_seed(origin)
    for bind, did in hops:
        k = reference_kappa_step(k, reference_edge_sem(bind, did))
    return k


@dataclass
class Rel:
    fid: str
    bind: str
    did: str
    role: str
    init: tuple[int, int] = (1, 0)
    ctx: str | None = None


def write_s(dest: Path, rels: Sequence[Rel]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in rels:
        w, n = rel.init
        tags: dict[str, Any] = {
            "bind": rel.bind,
            "did": rel.did,
            "here": HERE,
            "w0": rel.bind,
            "hyp": "contradicted" if n else ("supported" if w else "untried"),
            "trials": w + n,
            "wins": w,
            "losses": n,
            "support": w,
            "contradiction": n,
        }
        if rel.ctx:
            tags["ctx"] = rel.ctx
        (dest / f"{rel.fid}.tag").write_text(record_to_tagfile(rel.fid, tags), encoding="utf-8")


def s_content_hash(s_dir: Path) -> str:
    parts: list[bytes] = []
    for p in sorted(s_dir.glob("*.tag")):
        parts.append(p.name.encode() + b"\0" + p.read_bytes())
    return _sha_bytes(b"".join(parts))


def make_context(s_dir: Path, policy: UsePolicy) -> Any:
    return make(
        s_dir,
        None,
        policy,
        explore_epsilon=0.0,
        use_context_kappa=True,
    )


def probe_cue(
    policy: UsePolicy,
    s_dir: Path | None,
    seed: int,
    cue: str | None,
    *,
    use_context_kappa: bool = True,
) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm013ctx_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make(
            store,
            None,
            policy,
            explore_epsilon=0.0,
            use_context_kappa=use_context_kappa,
        )
        ag.reset_rho()
        out = probe(ag, "probe_channel_b", seed, tokens=tokens)
        out["cue"] = cue
        pol = out.get("policy") or {}
        out["compose_hops"] = pol.get("compose_hops")
        out["evidence_resolved"] = bool(pol.get("evidence_resolved"))
        out["evidence_tie"] = bool(pol.get("evidence_tie"))
        out["compose_hold"] = bool(pol.get("compose_hold"))
        out["context_kappa"] = pol.get("context_kappa")
        out["weight_hash"] = ag.weight_hash()
        return out


# Fixed developed tokens (lockable). Nonces used for hold-outs.
TOKENS = {
    "x": "x",
    "q": "q",
    "a": "a",
    "b": "b",
    "y": "y",
    "press": "press",
    "tune": "tune",
}

HOPS_A_THEN_B: list[tuple[str, str]] = [
    ("x", "q"),
    ("q", "a"),
    ("a", "q"),
    ("q", "b"),
    ("b", "q"),
    ("q", "y"),
]
HOPS_B_THEN_A: list[tuple[str, str]] = [
    ("x", "q"),
    ("q", "b"),
    ("b", "q"),
    ("q", "a"),
    ("a", "q"),
    ("q", "y"),
]


def verify_kappa_vectors() -> tuple[bool, str, dict[str, Any]]:
    """Live kappa.py must match frozen known-answer vectors. Fail closed."""
    snap: dict[str, Any] = {"version": "TM.0.13.KAPPA"}
    if not KAPPA_LOCK.exists():
        return False, "docs/kappa_013.lock missing", snap
    lock = json.loads(KAPPA_LOCK.read_text(encoding="utf-8"))
    if lock.get("ctx_encoding") != CTX_ENCODING:
        return False, "ctx_encoding drifted", snap
    if lock.get("kappa_module_sha") != _sha_file(REPO_ROOT / "three_memory" / "kappa.py"):
        return False, "kappa.py module sha drifted", snap
    for name in (
        "kappa_seed",
        "kappa_step",
        "edge_sem",
        "route_kappa",
        "route_kappa_hops",
    ):
        from three_memory import kappa as kmod

        fn = getattr(kmod, name)
        if _sha_src(fn) != lock.get(f"{name}_sha"):
            return False, f"kappa function drift: {name}", snap
    vectors = lock.get("known_answer_vectors") or {}
    for vname, vec in vectors.items():
        origin = vec["origin"]
        hops = [tuple(h) for h in vec["hops"]]
        k = live_kappa_seed(origin)
        if vec["steps"][0]["kappa"] != k:
            return False, f"vector {vname} seed mismatch", snap
        for i, (b, d) in enumerate(hops, start=1):
            k = live_kappa_step(k, live_edge_sem(b, d))
            if vec["steps"][i]["kappa"] != k:
                return False, f"vector {vname} step {i} mismatch", snap
        if route_kappa_hops(origin, hops) != vec["steps"][-1]["kappa"]:
            return False, f"vector {vname} final mismatch", snap
        # Reference oracle must agree with frozen vectors (independent code).
        if reference_route_kappa(origin, hops) != vec["steps"][-1]["kappa"]:
            return False, f"reference disagree on {vname}", snap
    snap["vectors_ok"] = True
    snap["n_vectors"] = len(vectors)
    return True, "kappa vectors + reference agree", snap


def kappa_final(hops: Sequence[tuple[str, str]], origin: str = "x") -> str:
    return reference_route_kappa(origin, hops)


def base_graph_a_then_b(
    *,
    fids: dict[str, str],
    kappa_ab: str,
    kappa_ba: str,
    qa_support: int = 3,
    qb_support: int = 2,
    qy_support: int = 1,
) -> list[Rel]:
    """Evidence-causal A-then-B route + both contextual motors at Y."""
    return [
        Rel(fids["xq"], "x", "q", "xq", (1, 0)),
        Rel(fids["qa"], "q", "a", "qa", (qa_support, 0)),
        Rel(fids["aq"], "a", "q", "aq", (1, 0)),
        Rel(fids["qb"], "q", "b", "qb", (qb_support, 0)),
        Rel(fids["bq"], "b", "q", "bq", (1, 0)),
        Rel(fids["qy"], "q", "y", "qy", (qy_support, 0)),
        Rel(fids["yp"], "y", "press", "yp", (1, 0), ctx=kappa_ab),
        Rel(fids["yt"], "y", "tune", "yt", (1, 0), ctx=kappa_ba),
    ]


def default_fids(rng: np.random.Generator | None = None) -> dict[str, str]:
    if rng is None:
        return {
            "xq": "n00001",
            "qa": "n00002",
            "aq": "n00003",
            "qb": "n00004",
            "bq": "n00005",
            "qy": "n00006",
            "yp": "n00007",
            "yt": "n00008",
        }
    taken: set[str] = set()
    return {k: _fid(rng, taken) for k in ("xq", "qa", "aq", "qb", "bq", "qy", "yp", "yt")}


def cell_route_order_split(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Same cue X, same motors at Y; evidence history chooses route → κ → motor."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    assert kappa_ab != kappa_ba
    fids = default_fids()
    s_ab = dest / "s_ab"
    s_ba = dest / "s_ba"
    write_s(s_ab, base_graph_a_then_b(fids=fids, kappa_ab=kappa_ab, kappa_ba=kappa_ba))
    write_s(
        s_ba,
        base_graph_a_then_b(
            fids=fids,
            kappa_ab=kappa_ab,
            kappa_ba=kappa_ba,
            qa_support=2,
            qb_support=3,
        ),
    )
    before_ab = s_content_hash(s_ab)
    before_ba = s_content_hash(s_ba)
    w0 = probe_cue(policy, s_ab, 1, "x")["weight_hash"]
    out_ab = probe_cue(policy, s_ab, 11, "x")
    out_ba = probe_cue(policy, s_ba, 12, "x")
    after_ab = s_content_hash(s_ab)
    after_ba = s_content_hash(s_ba)
    ok = (
        _motor(out_ab["action_name"]) == "press"
        and _motor(out_ba["action_name"]) == "tune"
        and out_ab.get("context_kappa") == kappa_ab
        and out_ba.get("context_kappa") == kappa_ba
        and before_ab == after_ab
        and before_ba == after_ba
        and out_ab["weight_hash"] == w0
        and out_ba["weight_hash"] == w0
    )
    return {
        "cell": "C13_route_order",
        "ok": ok,
        "ab_motor": _motor(out_ab["action_name"]),
        "ba_motor": _motor(out_ba["action_name"]),
        "ab_kappa": out_ab.get("context_kappa"),
        "ba_kappa": out_ba.get("context_kappa"),
        "expect_ab_kappa": kappa_ab,
        "expect_ba_kappa": kappa_ba,
        "s_stable": before_ab == after_ab and before_ba == after_ba,
        "weights_stable": out_ab["weight_hash"] == w0 == out_ba["weight_hash"],
    }


def cell_c7_equal_evidence_hold(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Two equal-evidence motors with the same κ → evidence tie → HOLD."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    fids = default_fids()
    rels = base_graph_a_then_b(fids=fids, kappa_ab=kappa_ab, kappa_ba=kappa_ab)
    # Both motors carry same κ; equal support → tie.
    rels = [r for r in rels if r.role not in ("yp", "yt")] + [
        Rel(fids["yp"], "y", "press", "yp", (1, 0), ctx=kappa_ab),
        Rel(fids["yt"], "y", "tune", "yt", (1, 0), ctx=kappa_ab),
    ]
    s_dir = dest / "s_c7"
    write_s(s_dir, rels)
    out = probe_cue(policy, s_dir, 21, "x")
    ok = _motor(out["action_name"]) == "hold" and (
        out.get("evidence_tie") or out.get("compose_hold")
    )
    return {
        "cell": "C13_c7_tie_hold",
        "ok": ok,
        "motor": _motor(out["action_name"]),
        "evidence_tie": out.get("evidence_tie"),
        "compose_hold": out.get("compose_hold"),
    }


def cell_wipe_hold(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    s_dir = dest / "s_wipe"
    write_s(
        s_dir,
        base_graph_a_then_b(
            fids=default_fids(), kappa_ab=kappa_ab, kappa_ba=kappa_ba
        ),
    )
    shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    out = probe_cue(policy, s_dir, 31, "x")
    ok = _motor(out["action_name"]) == "hold"
    return {"cell": "C13_wipe", "ok": ok, "motor": _motor(out["action_name"])}


def cell_donor_revise(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """In-place evidence revise on same S: swap Q→A / Q→B support → motor flips."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    fids = default_fids()
    s_dir = dest / "s_revise"
    write_s(
        s_dir,
        base_graph_a_then_b(
            fids=fids, kappa_ab=kappa_ab, kappa_ba=kappa_ba, qa_support=3, qb_support=2
        ),
    )
    before = s_content_hash(s_dir)
    out1 = probe_cue(policy, s_dir, 35, "x")
    # Revise first-hop evidence in place (same fids / ctx tags).
    write_s(
        s_dir,
        base_graph_a_then_b(
            fids=fids, kappa_ab=kappa_ab, kappa_ba=kappa_ba, qa_support=2, qb_support=3
        ),
    )
    after_plant = s_content_hash(s_dir)
    out2 = probe_cue(policy, s_dir, 36, "x")
    after_use = s_content_hash(s_dir)
    ok = (
        _motor(out1["action_name"]) == "press"
        and _motor(out2["action_name"]) == "tune"
        and out1.get("context_kappa") == kappa_ab
        and out2.get("context_kappa") == kappa_ba
        and before != after_plant  # revise changed S
        and after_plant == after_use  # use did not write shortcuts
    )
    return {
        "cell": "C13_donor_revise",
        "ok": ok,
        "before_motor": _motor(out1["action_name"]),
        "after_motor": _motor(out2["action_name"]),
        "s_changed_on_revise": before != after_plant,
        "s_stable_on_use": after_plant == after_use,
    }


def cell_retarget_ctx(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Swap which motor carries κ(A-then-B) → output changes."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    fids = default_fids()
    # Retarget: PRESS gets BA κ, TUNE gets AB κ.
    rels = base_graph_a_then_b(fids=fids, kappa_ab=kappa_ba, kappa_ba=kappa_ab)
    s_dir = dest / "s_retarget"
    write_s(s_dir, rels)
    out = probe_cue(policy, s_dir, 41, "x")
    ok = _motor(out["action_name"]) == "tune" and out.get("context_kappa") == kappa_ab
    return {
        "cell": "C13_retarget",
        "ok": ok,
        "motor": _motor(out["action_name"]),
        "kappa": out.get("context_kappa"),
    }


def cell_reset_rho(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    s_dir = dest / "s_rho"
    write_s(
        s_dir,
        base_graph_a_then_b(
            fids=default_fids(), kappa_ab=kappa_ab, kappa_ba=kappa_ba
        ),
    )
    ag = make_context(s_dir, policy)
    w0 = ag.weight_hash()
    ag.reset_rho()
    out1 = probe(ag, "probe_channel_b", 51, tokens=frozenset({"x"}))
    ag.reset_rho()
    out2 = probe(ag, "probe_channel_b", 52, tokens=frozenset({"x"}))
    m1 = _motor(out1["action_name"])
    m2 = _motor(out2["action_name"])
    ok = m1 == m2 == "press" and ag.weight_hash() == w0
    return {
        "cell": "C13_reset_rho",
        "ok": ok,
        "motor1": m1,
        "motor2": m2,
        "weights_stable": ag.weight_hash() == w0,
    }


def cell_fid_rename(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Semantic κ: renaming fids must not change contextual motor."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    fids_a = default_fids()
    fids_b = default_fids(np.random.default_rng(99))
    assert set(fids_a.values()).isdisjoint(set(fids_b.values()))
    s_a = dest / "s_fid_a"
    s_b = dest / "s_fid_b"
    write_s(s_a, base_graph_a_then_b(fids=fids_a, kappa_ab=kappa_ab, kappa_ba=kappa_ba))
    write_s(s_b, base_graph_a_then_b(fids=fids_b, kappa_ab=kappa_ab, kappa_ba=kappa_ba))
    out_a = probe_cue(policy, s_a, 61, "x")
    out_b = probe_cue(policy, s_b, 62, "x")
    ok = (
        _motor(out_a["action_name"]) == _motor(out_b["action_name"]) == "press"
        and out_a.get("context_kappa") == out_b.get("context_kappa") == kappa_ab
    )
    return {
        "cell": "C13_fid_rename",
        "ok": ok,
        "motor_a": _motor(out_a["action_name"]),
        "motor_b": _motor(out_b["action_name"]),
        "kappa": out_a.get("context_kappa"),
    }


def cell_support_trap_contextual_wins(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Matching ctx motor support=1 beats untagged wrong motor support=1000."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    fids = default_fids()
    taken = set(fids.values())
    rng = np.random.default_rng(7)
    trap = _fid(rng, taken)
    rels = base_graph_a_then_b(fids=fids, kappa_ab=kappa_ab, kappa_ba=kappa_ba)
    # Untagged high-support wrong motor at Y.
    rels.append(Rel(trap, "y", "tune", "trap", (1000, 0), ctx=None))
    # Contextual PRESS stays support=1.
    s_dir = dest / "s_trap"
    write_s(s_dir, rels)
    out = probe_cue(policy, s_dir, 71, "x")
    ok = _motor(out["action_name"]) == "press"
    return {
        "cell": "C13_ctx_beats_untagged",
        "ok": ok,
        "motor": _motor(out["action_name"]),
    }


def cell_mismatch_no_fallback(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """ctx facts exist but none match κ; untagged motor exists → HOLD."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    wrong = kappa_final([("x", "q"), ("q", "z"), ("z", "y")])
    fids = default_fids()
    taken = set(fids.values())
    trap = _fid(np.random.default_rng(8), taken)
    rels = [
        Rel(fids["xq"], "x", "q", "xq", (1, 0)),
        Rel(fids["qa"], "q", "a", "qa", (3, 0)),
        Rel(fids["aq"], "a", "q", "aq", (1, 0)),
        Rel(fids["qb"], "q", "b", "qb", (2, 0)),
        Rel(fids["bq"], "b", "q", "bq", (1, 0)),
        Rel(fids["qy"], "q", "y", "qy", (1, 0)),
        Rel(fids["yp"], "y", "press", "yp", (1, 0), ctx=wrong),
        Rel(fids["yt"], "y", "tune", "yt", (1, 0), ctx=kappa_ba),  # not AB route
        Rel(trap, "y", "flip", "untagged", (1000, 0), ctx=None),
    ]
    s_dir = dest / "s_nofallback"
    write_s(s_dir, rels)
    out = probe_cue(policy, s_dir, 81, "x")
    ok = (
        _motor(out["action_name"]) == "hold"
        and out.get("context_kappa") == kappa_ab
    )
    return {
        "cell": "C13_no_fallback_untagged",
        "ok": ok,
        "motor": _motor(out["action_name"]),
        "kappa": out.get("context_kappa"),
        "expect_kappa": kappa_ab,
    }


def cell_hop1_motor_no_kappa(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Direct motor on hop 1: return motor; no κ init / no κ step."""
    s_dir = dest / "s_hop1"
    # Motor wins first-hop evidence. Competing non-motor exists but loses.
    write_s(
        s_dir,
        [
            Rel("n1", "x", "press", "xm", (3, 0)),
            Rel("n2", "x", "q", "xq", (1, 0)),
            Rel("n3", "q", "tune", "qt", (1000, 0), ctx="deadbeef"),
        ],
    )
    out = probe_cue(policy, s_dir, 91, "x")
    ok = _motor(out["action_name"]) == "press" and out.get("context_kappa") is None
    return {
        "cell": "C13_hop1_motor_no_kappa",
        "ok": ok,
        "motor": _motor(out["action_name"]),
        "kappa": out.get("context_kappa"),
    }


def cell_depth_holdout(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Preregistered DEPTH-shaped hold-out from kappa vector C."""
    lock = json.loads(KAPPA_LOCK.read_text(encoding="utf-8"))
    vec = lock["known_answer_vectors"]["C_depth_shaped"]
    hops = [tuple(h) for h in vec["hops"]]
    kappa = reference_route_kappa(vec["origin"], hops)
    other = kappa_final(HOPS_A_THEN_B)
    assert kappa == vec["steps"][-1]["kappa"]
    s_dir = dest / "s_depth"
    write_s(
        s_dir,
        [
            Rel("d01", "x", "q", "xq", (1, 0)),
            Rel("d02", "q", "t1", "qt1", (1, 0)),
            Rel("d03", "t1", "t2", "t12", (1, 0)),
            Rel("d04", "t2", "t3", "t23", (1, 0)),
            Rel("d05", "t3", "y", "t3y", (1, 0)),
            Rel("d06", "y", "press", "yp", (1, 0), ctx=kappa),
            Rel("d07", "y", "tune", "yt", (1, 0), ctx=other),
        ],
    )
    out = probe_cue(policy, s_dir, 101, "x")
    ok = _motor(out["action_name"]) == "press" and out.get("context_kappa") == kappa
    return {
        "cell": "C13_depth_holdout",
        "ok": ok,
        "motor": _motor(out["action_name"]),
        "kappa": out.get("context_kappa"),
        "holdout": True,
    }


def cell_new_nonce_order(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """New-nonce order witness — same topology, fresh tokens."""
    rng = np.random.default_rng(DEFAULT_SEED + 313)
    taken: set[str] = set()
    x, q, a, b, y = (_nonce(rng, taken) for _ in range(5))
    hops_ab = [(x, q), (q, a), (a, q), (q, b), (b, q), (q, y)]
    hops_ba = [(x, q), (q, b), (b, q), (q, a), (a, q), (q, y)]
    kab = reference_route_kappa(x, hops_ab)
    kba = reference_route_kappa(x, hops_ba)
    fids = default_fids(rng)
    s_ab = dest / "nonce_ab"
    s_ba = dest / "nonce_ba"
    write_s(
        s_ab,
        [
            Rel(fids["xq"], x, q, "xq", (1, 0)),
            Rel(fids["qa"], q, a, "qa", (3, 0)),
            Rel(fids["aq"], a, q, "aq", (1, 0)),
            Rel(fids["qb"], q, b, "qb", (2, 0)),
            Rel(fids["bq"], b, q, "bq", (1, 0)),
            Rel(fids["qy"], q, y, "qy", (1, 0)),
            Rel(fids["yp"], y, "press", "yp", (1, 0), ctx=kab),
            Rel(fids["yt"], y, "tune", "yt", (1, 0), ctx=kba),
        ],
    )
    write_s(
        s_ba,
        [
            Rel(fids["xq"], x, q, "xq", (1, 0)),
            Rel(fids["qa"], q, a, "qa", (2, 0)),
            Rel(fids["aq"], a, q, "aq", (1, 0)),
            Rel(fids["qb"], q, b, "qb", (3, 0)),
            Rel(fids["bq"], b, q, "bq", (1, 0)),
            Rel(fids["qy"], q, y, "qy", (1, 0)),
            Rel(fids["yp"], y, "press", "yp", (1, 0), ctx=kab),
            Rel(fids["yt"], y, "tune", "yt", (1, 0), ctx=kba),
        ],
    )
    out_ab = probe_cue(policy, s_ab, 111, x)
    out_ba = probe_cue(policy, s_ba, 112, x)
    ok = (
        _motor(out_ab["action_name"]) == "press"
        and _motor(out_ba["action_name"]) == "tune"
        and out_ab.get("context_kappa") == kab
        and out_ba.get("context_kappa") == kba
    )
    return {
        "cell": "C13_new_nonce_order",
        "ok": ok,
        "holdout": True,
        "ab_motor": _motor(out_ab["action_name"]),
        "ba_motor": _motor(out_ba["action_name"]),
        "tokens": {"x": x, "q": q, "a": a, "b": b, "y": y},
    }


def cell_feature_off_compat(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """With use_context_kappa=False, ctx is ignored: untagged high support wins."""
    kappa_ab = kappa_final(HOPS_A_THEN_B)
    kappa_ba = kappa_final(HOPS_B_THEN_A)
    fids = default_fids()
    taken = set(fids.values())
    trap = _fid(np.random.default_rng(13), taken)
    # Matching ctx PRESS support=1; untagged TUNE support=1000.
    # Feature ON → PRESS. Feature OFF → TUNE (untagged evidence wins).
    rels = base_graph_a_then_b(fids=fids, kappa_ab=kappa_ab, kappa_ba=kappa_ba)
    rels = [r for r in rels if r.role != "yt"] + [
        Rel(trap, "y", "tune", "trap", (1000, 0), ctx=None),
    ]
    s_dir = dest / "s_off"
    write_s(s_dir, rels)
    out_on = probe_cue(policy, s_dir, 121, "x", use_context_kappa=True)
    out_off = probe_cue(policy, s_dir, 122, "x", use_context_kappa=False)
    ok = (
        _motor(out_on["action_name"]) == "press"
        and _motor(out_off["action_name"]) == "tune"
    )
    return {
        "cell": "C13_feature_off_untagged_wins",
        "ok": ok,
        "on_motor": _motor(out_on["action_name"]),
        "off_motor": _motor(out_off["action_name"]),
    }


def cell_visited_ctx_no_poison(policy: UsePolicy, dest: Path) -> dict[str, Any]:
    """Consumed ctx fact must not poison later untagged match at the same bind."""
    s_dir = dest / "s_visited"
    # x --ctx=poison,s=3--> a --> x --untagged--> press
    write_s(
        s_dir,
        [
            Rel("n1", "x", "a", "xa", (3, 0), ctx="poison"),
            Rel("n2", "a", "x", "ax", (1, 0)),
            Rel("n3", "x", "press", "xp", (1, 0), ctx=None),
        ],
    )
    out = probe_cue(policy, s_dir, 131, "x")
    ok = _motor(out["action_name"]) == "press" and int(out.get("compose_hops") or 0) == 3
    return {
        "cell": "C13_visited_ctx_no_poison",
        "ok": ok,
        "motor": _motor(out["action_name"]),
        "compose_hops": out.get("compose_hops"),
        "kappa": out.get("context_kappa"),
    }


CELLS: list[Callable[[UsePolicy, Path], dict[str, Any]]] = [
    cell_route_order_split,
    cell_c7_equal_evidence_hold,
    cell_wipe_hold,
    cell_donor_revise,
    cell_retarget_ctx,
    cell_reset_rho,
    cell_fid_rename,
    cell_support_trap_contextual_wins,
    cell_mismatch_no_fallback,
    cell_hop1_motor_no_kappa,
    cell_visited_ctx_no_poison,
    cell_depth_holdout,
    cell_new_nonce_order,
    cell_feature_off_compat,
]


def genome_013_snapshot() -> dict[str, Any]:
    from three_memory import agent as agent_mod

    ag = make(
        REPO_ROOT / "runs" / "_context_013_probe",
        None,
        UsePolicy(seed=1),
        enabled=False,
        use_context_kappa=True,
    )
    return {
        "version": "TM.0.13.GENOME",
        "ex0s_under_test": "0.0.003",
        "use_context_kappa": True,
        "ctx_encoding": CTX_ENCODING,
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "policy_sha": _sha_file(REPO_ROOT / "three_memory" / "policy.py"),
        "cortex_sha": _sha_file(REPO_ROOT / "three_memory" / "cortex.py"),
        "kappa_sha": _sha_file(REPO_ROOT / "three_memory" / "kappa.py"),
        "make011compose_sha": _sha_bytes(inspect.getsource(make).encode()),
        "cortex_weight_hash": ag.weight_hash(),
        "n_feat": int(UsePolicy.n_feat),
        "organism_baseline_commit": ORGANISM_BASELINE_COMMIT,
        "genome_011_lock_sha": _sha_file(GENOME_011_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "clone_empty_copies_flag": "use_context_kappa=self.use_context_kappa"
        in inspect.getsource(agent_mod.ThreeMemoryAgent.clone_empty),
    }


def write_genome_013_lock(path: Path = GENOME_013_LOCK) -> dict[str, Any]:
    snap = genome_013_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_genome_013(path: Path = GENOME_013_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = genome_013_snapshot()
    if not path.exists():
        return False, "docs/genome_013.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "policy_sha",
        "cortex_sha",
        "kappa_sha",
        "make011compose_sha",
        "cortex_weight_hash",
        "n_feat",
        "use_context_kappa",
        "ctx_encoding",
        "genome_011_lock_sha",
        "kappa_013_lock_sha",
    ):
        if snap.get(key) != lock.get(key):
            return False, f"genome_013 drift: {key}", snap
    # agent_sha is a historical CONTEXT freeze pin — HEAD may add ACQUIRE later.
    if not lock.get("agent_sha"):
        return False, "genome_013 missing historical agent_sha", snap
    if not snap.get("clone_empty_copies_flag"):
        return False, "clone_empty missing use_context_kappa", snap
    # Immutable 0.11 lock content must not have been rewritten.
    if _sha_file(GENOME_011_LOCK) != lock.get("genome_011_lock_sha"):
        return False, "genome_011.lock rewritten", snap
    return True, "genome_013 CONTEXT-on candidate intact (agent historical)", snap


def context_lock_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "TM.0.13.CONTEXT",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "ctx_encoding": CTX_ENCODING,
        "seed": DEFAULT_SEED,
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "genome_011_lock_sha": _sha_file(GENOME_011_LOCK),
        "genome_013_lock_sha": _sha_file(GENOME_013_LOCK) if GENOME_013_LOCK.exists() else None,
        "reference_route_kappa_sha": _sha_src(reference_route_kappa),
        "reference_kappa_seed_sha": _sha_src(reference_kappa_seed),
        "reference_kappa_step_sha": _sha_src(reference_kappa_step),
        "reference_edge_sem_sha": _sha_src(reference_edge_sem),
        "cell_ids": [r["cell"] for r in rows],
        "n_cells": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "refuse": [
            "rewrite genome_011.lock",
            "stamp Ex0S 0.0.004",
            "encode kappa to motor in genome",
            "step kappa on motor edges",
            "apply ctx filter on observation MATCH",
            "fallback to untagged when any ctx exists",
            "plant family ctx only from live kappa without reference",
            "252-world CONTEXT generator this pass",
        ],
    }


def write_context_lock(rows: list[dict[str, Any]], path: Path = CONTEXT_LOCK) -> dict[str, Any]:
    snap = context_lock_snapshot(rows)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_context_lock(
    rows: list[dict[str, Any]] | None = None, path: Path = CONTEXT_LOCK
) -> tuple[bool, str, dict[str, Any]]:
    snap = context_lock_snapshot(rows or [])
    if not path.exists():
        return False, "docs/context_013.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", snap
    if lock.get("ex0s_under_test") != "0.0.003":
        return False, "ex0s_under_test drifted", snap
    if lock.get("ctx_encoding") != CTX_ENCODING:
        return False, "ctx_encoding drifted", snap
    if _sha_file(KAPPA_LOCK) != lock.get("kappa_013_lock_sha"):
        return False, "kappa_013.lock pin drifted", snap
    if _sha_file(GENOME_011_LOCK) != lock.get("genome_011_lock_sha"):
        return False, "genome_011.lock pin drifted", snap
    if not GENOME_013_LOCK.exists():
        return False, "docs/genome_013.lock missing", snap
    if _sha_file(GENOME_013_LOCK) != lock.get("genome_013_lock_sha"):
        return False, "genome_013.lock pin drifted", snap
    for key, fn in (
        ("reference_route_kappa_sha", reference_route_kappa),
        ("reference_kappa_seed_sha", reference_kappa_seed),
        ("reference_kappa_step_sha", reference_kappa_step),
        ("reference_edge_sem_sha", reference_edge_sem),
    ):
        if _sha_src(fn) != lock.get(key):
            return False, f"context lock drift: {key}", snap
    if "stamp Ex0S 0.0.004" not in (lock.get("refuse") or []):
        return False, "refuse missing 0.0.004 ban", snap
    # When rows provided (recorded/family run), pin apparatus outcomes fail-closed.
    if rows is not None:
        cell_ids = [r["cell"] for r in rows]
        n_ok = sum(1 for r in rows if r.get("ok"))
        if cell_ids != lock.get("cell_ids"):
            return False, "context lock drift: cell_ids", snap
        if len(rows) != lock.get("n_cells"):
            return False, "context lock drift: n_cells", snap
        if n_ok != lock.get("n_ok"):
            return False, "context lock drift: n_ok", snap
        if n_ok != len(rows):
            return False, "family cells not all ok", snap
    snap["verified_with_rows"] = rows is not None
    return True, "context_013.lock intact", snap


def run_context(
    *,
    seed: int = DEFAULT_SEED,
    write_locks: bool = False,
) -> dict[str, Any]:
    run_dir = REPO_ROOT / "runs" / (
        datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_tm013context"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_ok, vec_why, vec_snap = verify_kappa_vectors()
    hist_ok, hist_why, hist_snap = verify_historical_freeze()
    compat_ok, compat_why, compat_snap = verify_011_compatibility()

    if write_locks:
        write_genome_013_lock()

    g013_ok, g013_why, g013_snap = verify_genome_013()
    if write_locks and not GENOME_013_LOCK.exists():
        g013_ok, g013_why, g013_snap = False, "genome_013 write failed", {}

    policy = UsePolicy(seed=seed, lr=0.2)
    rows: list[dict[str, Any]] = []
    source_before = genome_013_snapshot()
    for fn in CELLS:
        cell_dir = run_dir / fn.__name__
        cell_dir.mkdir(parents=True, exist_ok=True)
        row = fn(policy, cell_dir)
        rows.append(row)

    source_after = genome_013_snapshot()
    sources_stable = source_before == source_after
    # Re-check genome_013 after run (hashes must be unchanged during recorded run).
    g013_after_ok, g013_after_why, _ = verify_genome_013()

    if write_locks:
        write_context_lock(rows)

    ctx_ok, ctx_why, ctx_snap = verify_context_lock(rows)

    n_ok = sum(1 for r in rows if r.get("ok"))
    all_ok = (
        vec_ok
        and hist_ok
        and compat_ok
        and g013_ok
        and g013_after_ok
        and ctx_ok
        and sources_stable
        and n_ok == len(rows)
    )
    summary: dict[str, Any] = {
        "version": "TM.0.13.CONTEXT",
        "ex0s_under_test": "0.0.003",
        "ex0s": None,
        "earned_next": False,
        "ok": all_ok,
        "n_cells": len(rows),
        "n_ok": n_ok,
        "rows": rows,
        "kappa_vectors_ok": vec_ok,
        "kappa_vectors_why": vec_why,
        "historical_freeze_ok": hist_ok,
        "historical_freeze_why": hist_why,
        "compatibility_ok": compat_ok,
        "compatibility_why": compat_why,
        "genome_013_ok": g013_ok and g013_after_ok,
        "genome_013_why": g013_why if not g013_ok else g013_after_why,
        "context_lock_ok": ctx_ok,
        "context_lock_why": ctx_why,
        "sources_stable_during_run": sources_stable,
        "run_dir": str(run_dir),
        "freeze": {**hist_snap, **compat_snap},
        "kappa": vec_snap,
        "genome_013": g013_snap,
        "context_lock": ctx_snap,
        "head_matches_lock_agent": hist_snap.get("head_matches_lock_agent"),
        "claim": (
            "The organism carries compact transient context while traversing "
            "externally acquired knowledge, allowing identical frontier symbols "
            "to support different learned continuations without encoding those "
            "continuations in the genome."
            if all_ok
            else None
        ),
        "note": "Candidate CONTEXT-on under test. Not stamped Ex0S 0.0.004. earned_next false.",
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TM.0.13.CONTEXT")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--write-locks",
        action="store_true",
        help="Write genome_013.lock and context_013.lock (not genome_011).",
    )
    args = p.parse_args(argv)
    summary = run_context(seed=args.seed, write_locks=args.write_locks)
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2, default=str))
    for r in summary["rows"]:
        print(f"  {r['cell']}: {'OK' if r.get('ok') else 'FAIL'} { {k:v for k,v in r.items() if k not in ('cell','ok')} }")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
