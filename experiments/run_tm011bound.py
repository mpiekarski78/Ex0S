"""TM.0.11.BOUND: map the capacity envelope of frozen Ex0S 0.0.003.

Not a recipe jump. No Ex0S stamp. Four outcome classes:
  within_model_pass | unexpected_fail | expected_boundary | resource_boundary
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm011family import verify_freeze as verify_organism_freeze
from experiments.run_tm040 import probe
from three_memory.dial_env import ChannelDialWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile

MOTORS = ("press", "tune", "flip")
BANNED = frozenset(MOTORS + ("hold", "idle", "push", "adjust", "open", "wait", "use", "pick"))
CONS = "bcdfghjklmnpqrstvwxz"
VOW = "aeiou"

GENOME_LOCK = REPO_ROOT / "docs" / "genome_011.lock"
BOUND_LOCK = REPO_ROOT / "docs" / "bound_011.lock"

# Preregistered resource budget (also written into bound_011.lock).
TIMEOUT_MS = 30_000
MEMORY_NOTE = "rss via resource.getrusage; soft ceiling advisory only"

CLASS_WITHIN = "within_model_pass"
CLASS_UNEXPECTED = "unexpected_fail"
CLASS_EXPECTED = "expected_boundary"
CLASS_RESOURCE = "resource_boundary"

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


def _two_motors(rng: np.random.Generator) -> tuple[str, str]:
    m1, m2 = (str(t) for t in rng.permutation(MOTORS[:2]))
    return m1, m2


@dataclass
class Rel:
    fid: str
    bind: str
    did: str
    role: str
    init: tuple[int, int] = (1, 0)
    here: str = "chb"


@dataclass
class Cell:
    cell_id: str
    axis: str
    level: Any
    holdout: bool
    # within_model | expected_boundary
    intended_class: str
    expect_motor: str
    expect_hops: int | None
    relations: list[Rel]
    cue: str
    phases: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    upstream_roles: list[str] = field(default_factory=list)
    annotation: str = ""
    depth: int = 0
    n_offpath: int = 0
    branch: int = 1


def acquired_edges(rels: list[Rel]) -> set[tuple[str, str]]:
    return {(r.bind.lower(), r.did.lower()) for r in rels}


def edges_in_s(s_dir: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for path in s_dir.glob("*.tag"):
        _fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        b, d = tags.get("bind"), tags.get("did")
        if isinstance(b, str) and isinstance(d, str) and b and d:
            out.add((b.lower(), d.lower()))
    return out


def acquired_edge_invariant(s_dir: Path, rels: list[Rel]) -> bool:
    return edges_in_s(s_dir) - acquired_edges(rels) == set()


def write_world_s(dest: Path, rels: list[Rel]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in rels:
        w, n = rel.init
        tags: dict[str, Any] = {
            "bind": rel.bind,
            "did": rel.did,
            "here": rel.here,
            "w0": rel.bind,
            "hyp": "contradicted" if n else ("supported" if w else "untried"),
            "trials": w + n,
            "wins": w,
            "losses": n,
            "support": w,
            "contradiction": n,
        }
        (dest / f"{rel.fid}.tag").write_text(record_to_tagfile(rel.fid, tags), encoding="utf-8")


def fact_body_hash(s_dir: Path, role: str, rels: list[Rel]) -> str | None:
    by_role = {r.role: r for r in rels}
    rel = by_role.get(role)
    if rel is None:
        return None
    for p in s_dir.glob("*.tag"):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if tags.get("bind") == rel.bind and tags.get("did") == rel.did:
            return _sha_bytes(p.read_bytes())
    return None


def probe_cue(policy: UsePolicy, s_dir: Path | None, seed: int, cue: str | None) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm011bound_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make(store, None, policy, explore_epsilon=0.0)
        ag.reset_rho()
        t0 = time.perf_counter()
        out = probe(ag, "probe_channel_b", seed, tokens=tokens)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        pol = out.get("policy") or {}
        hops = int(pol.get("compose_hops") or 0)
        n_s = len(list(store.glob("*.tag"))) if store.exists() else 0
        out["cue"] = cue
        out["compose_hops"] = pol.get("compose_hops")
        out["evidence_resolved"] = bool(pol.get("evidence_resolved"))
        out["evidence_tie"] = bool(pol.get("evidence_tie"))
        out["compose_hold"] = bool(pol.get("compose_hold"))
        out["decision_wall_ms"] = wall_ms
        out["compose_iterations"] = hops
        out["facts_examined"] = hops * n_s
        out["eligible_facts"] = None
        out["evidence_comparisons"] = hops
        out["n_s"] = n_s
        out["peak_memory_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out["timed_out"] = False
        return out


def probe_cue_budget(
    policy: UsePolicy, s_dir: Path | None, seed: int, cue: str | None, *, timeout_ms: int = TIMEOUT_MS
) -> dict[str, Any]:
    """Run probe_cue under the preregistered wall-clock budget."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(probe_cue, policy, s_dir, seed, cue)
        try:
            return fut.result(timeout=max(timeout_ms, 1) / 1000.0)
        except FuturesTimeout:
            return {
                "action_name": "hold",
                "cue": cue,
                "compose_hops": None,
                "timed_out": True,
                "decision_wall_ms": float(timeout_ms),
                "facts_examined": None,
                "n_s": None,
                "peak_memory_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }


def _ids_by_role(s_dir: Path, rels: list[Rel]) -> dict[str, str]:
    by_pair = {}
    for path in s_dir.glob("*.tag"):
        fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        by_pair[(tags.get("bind"), tags.get("did"))] = fid
    return {r.role: by_pair.get((r.bind, r.did), r.fid) for r in rels}


def _earn(ag: Any, s_dir: Path, rels: list[Rel], steps: list[tuple[str, bool]]) -> None:
    ids = _ids_by_role(s_dir, rels)
    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    for role, ok in steps:
        ag._last_chosen_ids = [ids[role]]
        ag.observe_outcome(obs, ok, {"opened": ok})


def _chain_rels(
    rng: np.random.Generator,
    depth: int,
    motor: str,
    *,
    taken_n: set[str] | None = None,
    taken_f: set[str] | None = None,
    support: tuple[int, int] = (1, 0),
) -> tuple[list[Rel], list[str], str]:
    """Linear chain of `depth` hops ending in motor. Returns rels, nodes, cue."""
    taken_n = taken_n if taken_n is not None else set()
    taken_f = taken_f if taken_f is not None else set()
    nodes = [_nonce(rng, taken_n) for _ in range(depth)]
    nodes.append(motor)
    rels = []
    for i in range(depth):
        rels.append(
            Rel(_fid(rng, taken_f), nodes[i], nodes[i + 1], f"e{i}", support)
        )
    return rels, nodes, nodes[0]


def _offpath_junk(
    rng: np.random.Generator,
    n: int,
    *,
    taken_n: set[str],
    taken_f: set[str],
    wrong_motor: str,
    support: tuple[int, int] = (1000, 0),
) -> list[Rel]:
    out = []
    for i in range(n):
        b = _nonce(rng, taken_n)
        out.append(Rel(_fid(rng, taken_f), b, wrong_motor, f"junk{i}", support))
    return out


# --- generators (hold-outs + expected-boundary hashed into bound_011.lock) ---


def gen_depth(seed: int, depth: int, *, holdout: bool = False) -> Cell:
    rng = np.random.default_rng(seed + 17 * depth)
    m1, m2 = _two_motors(rng)
    rels, nodes, cue = _chain_rels(rng, depth, m1)
    return Cell(
        cell_id=f"depth_{depth}",
        axis="depth",
        level=depth,
        holdout=holdout,
        intended_class="within_model",
        expect_motor=m1,
        expect_hops=depth,
        relations=rels,
        cue=cue,
        probes=[{"after": "plant", "expect": m1, "hops": depth}],
        phases=[{"name": "plant", "steps": []}],
        depth=depth,
        annotation="linear chain; acquired-edge invariant",
    )


def gen_ssize(seed: int, depth: int, n_s: int, *, holdout: bool = False) -> Cell:
    rng = np.random.default_rng(seed + 31 * depth + n_s)
    m1, m2 = _two_motors(rng)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    rels, nodes, cue = _chain_rels(rng, depth, m1, taken_n=taken_n, taken_f=taken_f)
    need = max(0, n_s - len(rels))
    rels.extend(_offpath_junk(rng, need, taken_n=taken_n, taken_f=taken_f, wrong_motor=m2))
    return Cell(
        cell_id=f"ssize_d{depth}_n{n_s}",
        axis="ssize",
        level={"depth": depth, "n_s": n_s},
        holdout=holdout,
        intended_class="within_model",
        expect_motor=m1,
        expect_hops=depth,
        relations=rels,
        cue=cue,
        probes=[{"after": "plant", "expect": m1, "hops": depth}],
        phases=[{"name": "plant", "steps": []}],
        depth=depth,
        n_offpath=need,
        annotation="off-path wrong-motor padding; MATCH must drop",
    )


def gen_branch(seed: int, k: int) -> Cell:
    rng = np.random.default_rng(seed + 41 * k)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    mids = [_nonce(rng, taken_n) for _ in range(k)]
    rels = []
    # First mid strongest; others weaker. Each mid → motor (stronger path → m1).
    for i, mid in enumerate(mids):
        supp = (10 - i, 0) if i < 10 else (1, 0)
        rels.append(Rel(_fid(rng, taken_f), x, mid, f"x{i}", supp))
        motor = m1 if i == 0 else m2
        rels.append(Rel(_fid(rng, taken_f), mid, motor, f"m{i}", (1, 0)))
    return Cell(
        cell_id=f"branch_{k}",
        axis="branch",
        level=k,
        holdout=False,
        intended_class="within_model",
        expect_motor=m1 if k >= 1 else "hold",
        expect_hops=2,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": m1, "hops": 2}],
        phases=[{"name": "plant", "steps": []}],
        branch=k,
        depth=2,
        annotation="first-hop evidence only among X→mid rivals",
    )


def gen_branch_tie(seed: int) -> Cell:
    rng = np.random.default_rng(seed + 43)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    y, z = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), x, z, "xz", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
        Rel(_fid(rng, taken_f), z, m2, "zm", (1000, 0)),
    ]
    return Cell(
        cell_id="branch_tie",
        axis="branch",
        level="tie",
        holdout=False,
        intended_class="within_model",
        expect_motor="hold",
        expect_hops=None,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": "hold", "hops": None}],
        phases=[{"name": "plant", "steps": []}],
        branch=2,
        depth=2,
        annotation="equal first-hop; HOLD (downstream trap must not win)",
    )


def gen_distractors(seed: int, n_junk: int) -> Cell:
    rng = np.random.default_rng(seed + 47 + n_junk)
    m1, m2 = _two_motors(rng)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    rels, nodes, cue = _chain_rels(rng, 2, m1, taken_n=taken_n, taken_f=taken_f)
    rels.extend(_offpath_junk(rng, n_junk, taken_n=taken_n, taken_f=taken_f, wrong_motor=m2))
    return Cell(
        cell_id=f"distractors_{n_junk}",
        axis="distractors",
        level=n_junk,
        holdout=False,
        intended_class="within_model",
        expect_motor=m1,
        expect_hops=2,
        relations=rels,
        cue=cue,
        probes=[{"after": "plant", "expect": m1, "hops": 2}],
        phases=[{"name": "plant", "steps": []}],
        depth=2,
        n_offpath=n_junk,
        annotation="off-path wrong-motor distractors only",
    )


def gen_cycle_only(seed: int) -> Cell:
    rng = np.random.default_rng(seed + 53)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, z, "yz", (1, 0)),
        Rel(_fid(rng, taken_f), z, x, "zx", (1, 0)),
    ]
    return Cell(
        cell_id="cycle_only",
        axis="cycles",
        level="only",
        holdout=False,
        intended_class="within_model",
        expect_motor="hold",
        expect_hops=None,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": "hold", "hops": None}],
        phases=[{"name": "plant", "steps": []}],
        annotation="cycle; visited-fact exclusion ⇒ HOLD",
    )


def gen_cycle_exit(seed: int) -> Cell:
    rng = np.random.default_rng(seed + 59)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, _m2 = _two_motors(rng)
    # At Z: cycle back weaker; motor exit stronger (first-hop at Z).
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, z, "yz", (1, 0)),
        Rel(_fid(rng, taken_f), z, x, "zx", (1, 0)),
        Rel(_fid(rng, taken_f), z, m1, "zm", (2, 0)),
    ]
    return Cell(
        cell_id="cycle_exit",
        axis="cycles",
        level="exit",
        holdout=False,
        intended_class="within_model",
        expect_motor=m1,
        expect_hops=3,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": m1, "hops": 3}],
        phases=[{"name": "plant", "steps": []}],
        depth=3,
        annotation="cycle + stronger motor exit at Z by first-hop evidence",
    )


def gen_nomotor(seed: int, depth: int = 5) -> Cell:
    rng = np.random.default_rng(seed + 61)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    nodes = [_nonce(rng, taken_n) for _ in range(depth + 1)]
    rels = [
        Rel(_fid(rng, taken_f), nodes[i], nodes[i + 1], f"e{i}", (1, 0))
        for i in range(depth)
    ]
    return Cell(
        cell_id=f"nomotor_{depth}",
        axis="nomotor",
        level=depth,
        holdout=False,
        intended_class="within_model",
        expect_motor="hold",
        expect_hops=None,
        relations=rels,
        cue=nodes[0],
        probes=[{"after": "plant", "expect": "hold", "hops": None}],
        phases=[{"name": "plant", "steps": []}],
        depth=depth,
        annotation="long chain, no motor ⇒ HOLD",
    )


def gen_deep_tie(seed: int, hop_k: int) -> Cell:
    """Tie at hop k (1-indexed from cue)."""
    rng = np.random.default_rng(seed + 67 * hop_k)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    m1, m2 = _two_motors(rng)
    if hop_k == 1:
        x = _nonce(rng, taken_n)
        y, z = _nonce(rng, taken_n), _nonce(rng, taken_n)
        rels = [
            Rel(_fid(rng, taken_f), x, y, "a", (1, 0)),
            Rel(_fid(rng, taken_f), x, z, "b", (1, 0)),
            Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
            Rel(_fid(rng, taken_f), z, m2, "zm", (1, 0)),
        ]
        cue = x
    else:
        # Prefix of hop_k-1 edges to frontier F, then equal F→Y / F→Z.
        nodes = [_nonce(rng, taken_n) for _ in range(hop_k)]  # cue .. F
        F = nodes[-1]
        cue = nodes[0]
        rels = [
            Rel(_fid(rng, taken_f), nodes[i], nodes[i + 1], f"e{i}", (1, 0))
            for i in range(hop_k - 1)
        ]
        y, z = _nonce(rng, taken_n), _nonce(rng, taken_n)
        rels.extend(
            [
                Rel(_fid(rng, taken_f), F, y, "fy", (1, 0)),
                Rel(_fid(rng, taken_f), F, z, "fz", (1, 0)),
                Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
                Rel(_fid(rng, taken_f), z, m2, "zm", (1, 0)),
            ]
        )
    return Cell(
        cell_id=f"deep_tie_k{hop_k}",
        axis="deep_tie",
        level=hop_k,
        holdout=False,
        intended_class="within_model",
        expect_motor="hold",
        expect_hops=None,
        relations=rels,
        cue=cue,
        probes=[{"after": "plant", "expect": "hold", "hops": None}],
        phases=[{"name": "plant", "steps": []}],
        annotation=f"evidence tie at hop {hop_k} ⇒ HOLD",
    )


def gen_file_order(seed: int, shuffle_i: int) -> Cell:
    """Same acquired edges; Rel list shuffled before write.

    TagStore loads sorted by filename, so this is a content-invariance check
    (not an unsorted-iteration adversarial). Motor must match the unshuffled chain.
    """
    rng = np.random.default_rng(seed + 71)
    m1, m2 = _two_motors(rng)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    rels, nodes, cue = _chain_rels(rng, 3, m1, taken_n=taken_n, taken_f=taken_f)
    rels.extend(_offpath_junk(rng, 5, taken_n=taken_n, taken_f=taken_f, wrong_motor=m2))
    shuf = np.random.default_rng(seed + 1000 + shuffle_i)
    order = list(rels)
    shuf.shuffle(order)
    return Cell(
        cell_id=f"file_order_{shuffle_i}",
        axis="file_order",
        level=shuffle_i,
        holdout=False,
        intended_class="within_model",
        expect_motor=m1,
        expect_hops=3,
        relations=order,
        cue=cue,
        probes=[{"after": "plant", "expect": m1, "hops": 3}],
        phases=[{"name": "plant", "steps": []}],
        depth=3,
        annotation="identical acquired edges; list shuffle before write (store sorts by fid)",
    )


def gen_reuse_a(seed: int) -> Cell:
    """Expected boundary: compose frontier is bind-only; here does not filter.

    Discriminator (must differ under here-filter vs bind-only):
      Y -> A   support=10, here=cha  (strong, wrong station)
      Y -> B   support=2,  here=chb  (weak, probe station = channel B)
      A -> m_strongpath, B -> m_weakpath

    If compose filtered by here=chb: only Y→B ⇒ weakpath motor.
    If bind-only (0.0.003): Y→A wins ⇒ strongpath motor.
    Expected boundary confirmation requires the strongpath motor.
    """
    rng = np.random.default_rng(seed + 79)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y = _nonce(rng, taken_n), _nonce(rng, taken_n)
    a, b = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m_strong, m_weak = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0), here="chb"),
        Rel(_fid(rng, taken_f), y, a, "ya", (10, 0), here="cha"),  # strong, wrong here
        Rel(_fid(rng, taken_f), y, b, "yb", (2, 0), here="chb"),  # weak, probe here
        Rel(_fid(rng, taken_f), a, m_strong, "am", (1, 0), here="cha"),
        Rel(_fid(rng, taken_f), b, m_weak, "bm", (1, 0), here="chb"),
    ]
    return Cell(
        cell_id="reuse_a",
        axis="reuse_a",
        level="here_split",
        holdout=False,
        intended_class="expected_boundary",
        expect_motor=m_strong,  # bind-only ⇒ strong wrong-here wins
        expect_hops=3,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": m_strong, "hops": 3}],
        phases=[{"name": "plant", "steps": []}],
        annotation="expected limit: compose frontier is bind-only (no here filter)",
    )


def gen_reuse_b(seed: int) -> Cell:
    rng = np.random.default_rng(seed + 83)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y = _nonce(rng, taken_n), _nonce(rng, taken_n)
    a, b = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, a, "ya", (5, 0)),
        Rel(_fid(rng, taken_f), y, b, "yb", (1, 0)),
        Rel(_fid(rng, taken_f), a, m1, "am", (1, 0)),
        Rel(_fid(rng, taken_f), b, m2, "bm", (1000, 0)),
    ]
    return Cell(
        cell_id="reuse_b",
        axis="reuse_b",
        level="same_context",
        holdout=False,
        intended_class="within_model",
        expect_motor=m1,
        expect_hops=3,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": m1, "hops": 3}],
        phases=[{"name": "plant", "steps": []}],
        annotation="same context; first-hop evidence at Y picks A not B",
    )


def gen_revise_far(seed: int, depth: int) -> Cell:
    """Chain of depth hops to D, then D→PRESS and D→TUNE both exist; revise D only."""
    rng = np.random.default_rng(seed + 89 * depth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    m1, m2 = _two_motors(rng)
    # depth hops to final non-motor F, then F→m1 and F→m2.
    # Total compose hops to motor = depth + 1.
    nodes = [_nonce(rng, taken_n) for _ in range(depth)]
    F = _nonce(rng, taken_n)
    nodes.append(F)
    rels = []
    upstream_roles = []
    for i in range(depth):
        role = f"e{i}"
        rels.append(Rel(_fid(rng, taken_f), nodes[i], nodes[i + 1], role, (1, 0)))
        upstream_roles.append(role)
    rels.append(Rel(_fid(rng, taken_f), F, m1, "d_press", (2, 0)))
    rels.append(Rel(_fid(rng, taken_f), F, m2, "d_tune", (0, 0)))
    hops = depth + 1
    return Cell(
        cell_id=f"revise_far_{depth}",
        axis="revise_far",
        level=depth,
        holdout=False,
        intended_class="within_model",
        expect_motor=m2,  # after revise
        expect_hops=hops,
        relations=rels,
        cue=nodes[0],
        upstream_roles=upstream_roles,
        phases=[
            {"name": "learn", "steps": []},
            {
                "name": "revise",
                "steps": [
                    ("d_press", False),
                    ("d_press", False),
                    ("d_tune", True),
                    ("d_tune", True),
                    ("d_tune", True),
                ],
                "reset_rho": True,
            },
        ],
        probes=[
            {"after": "learn", "expect": m1, "hops": hops},
            {"after": "revise", "expect": m2, "hops": hops},
        ],
        depth=hops,
        annotation="both D motors exist before revise; only D evidence changes",
    )


def gen_nasty(seed: int, *, holdout: bool = True) -> Cell:
    rng = np.random.default_rng(seed + 97)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    m1, m2 = _two_motors(rng)
    # X→A→B→C→D then D→PRESS / D→TUNE
    labs = [_nonce(rng, taken_n) for _ in range(5)]  # X,A,B,C,D
    x, a, b, c, d = labs
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", (1, 0)),
        Rel(_fid(rng, taken_f), a, b, "ab", (1, 0)),
        Rel(_fid(rng, taken_f), b, c, "bc", (1, 0)),
        Rel(_fid(rng, taken_f), c, d, "cd", (1, 0)),
        Rel(_fid(rng, taken_f), d, m1, "d_press", (2, 0)),
        Rel(_fid(rng, taken_f), d, m2, "d_tune", (0, 0)),
    ]
    rels.extend(_offpath_junk(rng, 300, taken_n=taken_n, taken_f=taken_f, wrong_motor=m2))
    upstream = ["xa", "ab", "bc", "cd"]
    return Cell(
        cell_id="nasty",
        axis="nasty",
        level="dirty_revise",
        holdout=holdout,
        intended_class="within_model",
        expect_motor=m2,
        expect_hops=5,
        relations=rels,
        cue=x,
        upstream_roles=upstream,
        phases=[
            {"name": "learn", "steps": []},
            {
                "name": "revise",
                "steps": [
                    ("d_press", False),
                    ("d_press", False),
                    ("d_tune", True),
                    ("d_tune", True),
                    ("d_tune", True),
                ],
                "reset_rho": True,
            },
        ],
        probes=[
            {"after": "learn", "expect": m1, "hops": 5},
            {"after": "revise", "expect": m2, "hops": 5},
        ],
        depth=5,
        n_offpath=300,
        annotation="factorized revise through dirty S; off-path only",
    )


def gen_local_opt(seed: int, *, holdout: bool = True) -> Cell:
    """Preregistered expected boundary: locally stronger dead end ⇒ HOLD."""
    rng = np.random.default_rng(seed + 101)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    a, b, dead = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, _m2 = _two_motors(rng)
    # DEAD is a nonce (not a motor) so path dies; nowhere continuation.
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", (2, 0)),
        Rel(_fid(rng, taken_f), x, dead, "xdead", (10, 0)),
        Rel(_fid(rng, taken_f), a, b, "ab", (1, 0)),
        Rel(_fid(rng, taken_f), b, m1, "bm", (1, 0)),
        # no outgoing from dead
    ]
    return Cell(
        cell_id="local_optimum_dead_end",
        axis="local_opt",
        level="boundary",
        holdout=holdout,
        intended_class="expected_boundary",
        expect_motor="hold",
        expect_hops=None,
        relations=rels,
        cue=x,
        probes=[{"after": "plant", "expect": "hold", "hops": None}],
        phases=[{"name": "plant", "steps": []}],
        annotation="not a compose bug; expected limit: no lookahead/backtracking",
    )


def all_cells(seed: int = DEFAULT_SEED) -> list[Cell]:
    cells: list[Cell] = []
    # depth developed
    for d in (2, 3, 4, 5, 8, 12):
        cells.append(gen_depth(seed, d, holdout=False))
    # depth hold-out
    for d in (16, 20):
        cells.append(gen_depth(seed, d, holdout=True))
    # |S| strata
    for depth in (2, 5):
        for n in (10, 100, 1000):
            hold = depth == 5 and n == 1000
            cells.append(gen_ssize(seed, depth, n, holdout=hold))
    for k in (1, 2, 4, 8):
        cells.append(gen_branch(seed, k))
    cells.append(gen_branch_tie(seed))
    for n in (0, 10, 100, 500):
        cells.append(gen_distractors(seed, n))
    cells.append(gen_cycle_only(seed))
    cells.append(gen_cycle_exit(seed))
    cells.append(gen_nomotor(seed, 5))
    for k in (1, 2, 3):
        cells.append(gen_deep_tie(seed, k))
    for i in range(8):
        cells.append(gen_file_order(seed, i))
    cells.append(gen_reuse_a(seed))
    cells.append(gen_reuse_b(seed))
    for d in (3, 5, 8):
        cells.append(gen_revise_far(seed, d))
    cells.append(gen_nasty(seed, holdout=True))
    cells.append(gen_local_opt(seed, holdout=True))
    return cells


def seed_list_blob(seed: int = DEFAULT_SEED) -> str:
    return "\n".join(c.cell_id for c in all_cells(seed))


def score_cell(
    cell: Cell,
    *,
    motor: str,
    hops: Any,
    invariant_ok: bool,
    upstream_ok: bool,
    timed_out: bool,
    probes_ok: bool,
) -> dict[str, Any]:
    if timed_out:
        return {
            "class": CLASS_RESOURCE,
            "ok_for_solved": False,
            "expected_boundary_confirmed": False,
            "why": f"timeout after {TIMEOUT_MS}ms",
        }
    if cell.intended_class == "expected_boundary":
        confirmed = motor == cell.expect_motor and invariant_ok
        if cell.expect_hops is not None:
            confirmed = confirmed and hops == cell.expect_hops
        return {
            "class": CLASS_EXPECTED if confirmed else CLASS_UNEXPECTED,
            "ok_for_solved": False,
            "expected_boundary_confirmed": confirmed,
            "why": cell.annotation if confirmed else f"boundary not confirmed: motor={motor} hops={hops}",
        }
    # within_model
    semantic_ok = probes_ok and invariant_ok and upstream_ok
    if semantic_ok:
        return {
            "class": CLASS_WITHIN,
            "ok_for_solved": True,
            "expected_boundary_confirmed": False,
            "why": "within-model pass",
        }
    return {
        "class": CLASS_UNEXPECTED,
        "ok_for_solved": False,
        "expected_boundary_confirmed": False,
        "why": f"semantic fail motor={motor} hops={hops} inv={invariant_ok} up={upstream_ok} probes={probes_ok}",
    }


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    cell_id = job["cell_id"]
    seed = int(job["seed"])
    dest = Path(job["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    # Rebuild cell from id via registry
    cell = next(c for c in all_cells(seed) if c.cell_id == cell_id)
    s_dir = dest / "S"
    write_world_s(s_dir, cell.relations)
    policy = UsePolicy(seed=7, lr=0.2)
    ag = make(s_dir, None, policy, explore_epsilon=0.0)
    upstream_before = {
        role: fact_body_hash(s_dir, role, cell.relations) for role in cell.upstream_roles
    }
    probe_results: list[dict[str, Any]] = []
    invariant_ok = True
    timed_out = False
    t_cell0 = time.perf_counter()
    try:
        for ph in cell.phases:
            if ph.get("reset_rho"):
                ag.reset_rho()
            _earn(ag, s_dir, cell.relations, list(ph.get("steps") or []))
            for i, spec in enumerate(cell.probes):
                if spec["after"] != ph["name"]:
                    continue
                hit = probe_cue_budget(policy, s_dir, seed + 20 + i, cell.cue)
                if hit.get("timed_out"):
                    timed_out = True
                    probe_results.append({**hit, "spec": spec})
                    break
                probe_results.append({**hit, "spec": spec})
                if not acquired_edge_invariant(s_dir, cell.relations):
                    invariant_ok = False
            if timed_out or (time.perf_counter() - t_cell0) * 1000.0 > TIMEOUT_MS:
                timed_out = True
                break
    except Exception as exc:  # noqa: BLE001 — record as resource/unexpected
        return {
            "cell_id": cell.cell_id,
            "axis": cell.axis,
            "level": cell.level,
            "holdout": cell.holdout,
            "intended_class": cell.intended_class,
            "class": CLASS_UNEXPECTED,
            "ok_for_solved": False,
            "expected_boundary_confirmed": False,
            "why": f"exception: {exc}",
            "motor": None,
            "n_s": len(cell.relations),
            "depth": cell.depth,
            "decision_wall_ms": None,
            "facts_examined": None,
            "annotation": cell.annotation,
        }

    upstream_after = {
        role: fact_body_hash(s_dir, role, cell.relations) for role in cell.upstream_roles
    }
    upstream_ok = all(
        upstream_before.get(r) is not None and upstream_before[r] == upstream_after.get(r)
        for r in cell.upstream_roles
    ) if cell.upstream_roles else True

    probes_ok = True
    last_motor = "hold"
    last_hops = None
    wall_sum = 0.0
    facts_sum = 0
    for pr in probe_results:
        spec = pr["spec"]
        got = _motor(pr.get("action_name"))
        last_motor = got
        last_hops = pr.get("compose_hops")
        wall_sum += float(pr.get("decision_wall_ms") or 0)
        facts_sum += int(pr.get("facts_examined") or 0)
        if got != spec["expect"]:
            probes_ok = False
        if spec.get("hops") is not None and spec["expect"] != "hold":
            if pr.get("compose_hops") != spec["hops"]:
                probes_ok = False

    # Final motor expectation for boundary / single-probe cells uses last probe
    # or expect_motor when multi-phase (revise uses final).
    if cell.probes:
        final_spec = cell.probes[-1]
        final_motor = _motor(probe_results[-1]["action_name"]) if probe_results else "hold"
        final_hops = probe_results[-1].get("compose_hops") if probe_results else None
    else:
        final_motor, final_hops = last_motor, last_hops

    scored = score_cell(
        cell,
        motor=final_motor,
        hops=final_hops,
        invariant_ok=invariant_ok and acquired_edge_invariant(s_dir, cell.relations),
        upstream_ok=upstream_ok,
        timed_out=timed_out,
        probes_ok=probes_ok,
    )
    return {
        "cell_id": cell.cell_id,
        "axis": cell.axis,
        "level": cell.level,
        "holdout": cell.holdout,
        "intended_class": cell.intended_class,
        "class": scored["class"],
        "ok_for_solved": scored["ok_for_solved"],
        "expected_boundary_confirmed": scored["expected_boundary_confirmed"],
        "why": scored["why"],
        "motor": final_motor,
        "compose_hops": final_hops,
        "n_s": len(list(s_dir.glob("*.tag"))),
        "depth": cell.depth,
        "branch": cell.branch,
        "n_offpath": cell.n_offpath,
        "decision_wall_ms": wall_sum,
        "facts_examined": facts_sum,
        "peak_memory_kb": probe_results[-1].get("peak_memory_kb") if probe_results else None,
        "invariant_ok": invariant_ok,
        "upstream_ok": upstream_ok,
        "annotation": cell.annotation,
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
    }


def apparatus_snapshot() -> dict[str, Any]:
    return {
        "version": "TM.0.11.BOUND",
        "ex0s_under_test": "0.0.003",
        "timeout_ms": TIMEOUT_MS,
        "memory_note": MEMORY_NOTE,
        "seed": DEFAULT_SEED,
        "seed_list_sha": _sha_bytes(seed_list_blob().encode()),
        "scorer_sha": _sha_src(score_cell),
        "gen_depth_sha": _sha_src(gen_depth),
        "gen_ssize_sha": _sha_src(gen_ssize),
        "gen_branch_sha": _sha_src(gen_branch),
        "gen_distractors_sha": _sha_src(gen_distractors),
        "gen_nasty_sha": _sha_src(gen_nasty),
        "gen_local_opt_sha": _sha_src(gen_local_opt),
        "gen_reuse_a_sha": _sha_src(gen_reuse_a),
        "gen_revise_far_sha": _sha_src(gen_revise_far),
        "held_out_cells": [
            c.cell_id for c in all_cells() if c.holdout
        ],
        "expected_boundary_cells": {
            "local_optimum_dead_end": {
                "expect_motor": "hold",
                "annotation": "not a compose bug; expected limit: no lookahead/backtracking",
            },
            "reuse_a": {
                "expect": "strong wrong-here path wins (bind-only); here-filter would pick weak same-here path",
                "annotation": "expected limit: compose frontier is bind-only (no here filter)",
            },
        },
    }


def write_bound_lock(path: Path = BOUND_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_bound_lock() -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not BOUND_LOCK.exists():
        return False, "docs/bound_011.lock missing", snap
    lock = json.loads(BOUND_LOCK.read_text(encoding="utf-8"))
    for key in (
        "timeout_ms",
        "seed_list_sha",
        "scorer_sha",
        "gen_depth_sha",
        "gen_ssize_sha",
        "gen_nasty_sha",
        "gen_local_opt_sha",
        "gen_reuse_a_sha",
        "held_out_cells",
    ):
        if snap[key] != lock.get(key):
            return False, f"bound apparatus drift: {key}", snap
    if "local_optimum_dead_end" not in (lock.get("expected_boundary_cells") or {}):
        return False, "local_optimum_dead_end not preregistered", snap
    return True, "bound apparatus frozen", snap


def aggregate(rows: list[dict[str, Any]], *, organism_ok: bool, bound_ok: bool) -> dict[str, Any]:
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_axis.setdefault(r["axis"], []).append(r)
    within = [r for r in rows if r["intended_class"] == "within_model"]
    solved = [r for r in within if r.get("ok_for_solved")]
    boundaries = [r for r in rows if r["intended_class"] == "expected_boundary"]
    confirmed = [r for r in boundaries if r.get("expected_boundary_confirmed")]
    classes = {
        CLASS_WITHIN: sum(1 for r in rows if r["class"] == CLASS_WITHIN),
        CLASS_UNEXPECTED: sum(1 for r in rows if r["class"] == CLASS_UNEXPECTED),
        CLASS_EXPECTED: sum(1 for r in rows if r["class"] == CLASS_EXPECTED),
        CLASS_RESOURCE: sum(1 for r in rows if r["class"] == CLASS_RESOURCE),
    }
    curves = {
        "depth": [
            {"level": r["level"], "class": r["class"], "wall_ms": r.get("decision_wall_ms"), "n_s": r.get("n_s")}
            for r in rows
            if r["axis"] == "depth"
        ],
        "ssize": [
            {"level": r["level"], "class": r["class"], "wall_ms": r.get("decision_wall_ms"), "facts_examined": r.get("facts_examined")}
            for r in rows
            if r["axis"] == "ssize"
        ],
        "branch": [
            {"level": r["level"], "class": r["class"]}
            for r in rows
            if r["axis"] == "branch"
        ],
        "distractors": [
            {"level": r["level"], "class": r["class"]}
            for r in rows
            if r["axis"] == "distractors"
        ],
        "revise_far": [
            {"level": r["level"], "class": r["class"], "wall_ms": r.get("decision_wall_ms")}
            for r in rows
            if r["axis"] == "revise_far"
        ],
    }
    return {
        "version": "TM.0.11.BOUND",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "organism_ok": organism_ok,
        "bound_ok": bound_ok,
        "n_cells": len(rows),
        "n_within_model": len(within),
        "within_model_pass": len(solved),
        "within_model_pass_frac": (len(solved) / len(within)) if within else 0.0,
        "expected_boundary_n": len(boundaries),
        "expected_boundary_confirmed": len(confirmed),
        "classes": classes,
        "curves": curves,
        "axes": {
            ax: {
                "n": len(chunk),
                "within_pass": sum(1 for r in chunk if r.get("ok_for_solved")),
                "unexpected": sum(1 for r in chunk if r["class"] == CLASS_UNEXPECTED),
                "expected_boundary": sum(1 for r in chunk if r["class"] == CLASS_EXPECTED),
                "resource": sum(1 for r in chunk if r["class"] == CLASS_RESOURCE),
            }
            for ax, chunk in by_axis.items()
        },
        "holdout": {
            "n": sum(1 for r in rows if r["holdout"]),
            "unexpected": sum(1 for r in rows if r["holdout"] and r["class"] == CLASS_UNEXPECTED),
            "within_pass": sum(1 for r in rows if r["holdout"] and r.get("ok_for_solved")),
            "boundary_confirmed": sum(
                1 for r in rows if r["holdout"] and r.get("expected_boundary_confirmed")
            ),
        },
        "intervention": {
            "required_genome_changes": 0,
            "apparatus_interventions": 0,
            "note": "No organism edits. Hold-outs and expected-boundary cells preregistered.",
        },
    }


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm011bound"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_bound(*, seed: int = DEFAULT_SEED, workers: int = 4) -> dict[str, Any]:
    run_dir = _run_dir()
    organism_ok, org_why, _ = verify_organism_freeze()
    bound_ok, bound_why, bound_snap = verify_bound_lock()
    cells = all_cells(seed)
    jobs = [
        {"cell_id": c.cell_id, "seed": seed, "dest": str(run_dir / c.axis / c.cell_id)}
        for c in cells
    ]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for j in jobs:
            rows.append(run_cell(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_cell, jobs))
    rows.sort(key=lambda r: (r["axis"], str(r["level"]), r["cell_id"]))
    summary = aggregate(rows, organism_ok=organism_ok, bound_ok=bound_ok)
    summary["organism_why"] = org_why
    summary["bound_why"] = bound_why
    summary["run_dir"] = str(run_dir)
    summary["seed"] = seed
    summary["bound_lock"] = bound_snap
    (run_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    # summary.md
    axis_lines = "\n".join(
        f"| {ax} | {v['n']} | {v['within_pass']} | {v['unexpected']} | {v['expected_boundary']} | {v['resource']} |"
        for ax, v in summary["axes"].items()
    )
    (run_dir / "summary.md").write_text(
        f"""# TM.0.11.BOUND · Ex0S 0.0.003 under test

Organism: {org_why}
Apparatus: {bound_why}
Within-model pass: **{summary['within_model_pass']}/{summary['n_within_model']}** ({summary['within_model_pass_frac']:.3f})
Expected boundaries confirmed: **{summary['expected_boundary_confirmed']}/{summary['expected_boundary_n']}**
Classes: {summary['classes']}
`earned_next`: false (no Ex0S 0.0.004)

| Axis | N | Within pass | Unexpected | Exp. boundary | Resource |
|------|---|-------------|------------|---------------|----------|
{axis_lines}
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.11.BOUND capacity envelope of frozen compose")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_bound_lock(), indent=2))
        return
    s = run_bound(seed=args.seed, workers=args.workers)
    print(
        json.dumps(
            {
                "within_model_pass_frac": s["within_model_pass_frac"],
                "within_model_pass": s["within_model_pass"],
                "n_within_model": s["n_within_model"],
                "expected_boundary_confirmed": s["expected_boundary_confirmed"],
                "classes": s["classes"],
                "earned_next": s["earned_next"],
                "organism_ok": s["organism_ok"],
                "bound_ok": s["bound_ok"],
                "holdout": s["holdout"],
                "run_dir": s["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
