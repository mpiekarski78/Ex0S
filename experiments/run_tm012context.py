"""TM.0.12.CONTEXT: representation audit of provenance at compose frontiers.

Freeze Ex0S 0.0.003. No genome patch. No Ex0S 0.0.004 stamp.
Hypotheses H0–H4; worlds C0–C7. earned_next always false.
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
from three_memory.symbols import parse_tagfile, record_to_tagfile

MOTORS = ("press", "tune", "flip")
BANNED = frozenset(MOTORS + ("hold", "idle", "push", "adjust", "open", "wait", "use", "pick"))
CONS = "bcdfghjklmnpqrstvwxz"
VOW = "aeiou"

GENOME_LOCK = REPO_ROOT / "docs" / "genome_011.lock"
CONTEXT_LOCK = REPO_ROOT / "docs" / "context_012.lock"

TIMEOUT_MS = 30_000
DEFAULT_SEED = 12345

CLASS_WITHIN = "within_model_pass"
CLASS_UNEXPECTED = "unexpected_fail"
CLASS_EXPECTED = "expected_boundary"
CLASS_RESOURCE = "resource_boundary"

HYPOTHESES = {
    "H0": {"name": "bare_token", "state": "Y", "role": "baseline"},
    "H1": {"name": "here_station", "state": "(Y, here)", "role": "structured"},
    "H2": {"name": "predecessor", "state": "(Y, pred)", "role": "structured"},
    "H3": {"name": "path_origin", "state": "(Y, origin or path)", "role": "structured"},
    "H4": {
        "name": "opaque_fact_id",
        "state": "(Y, fact_id)",
        "role": "diagnostic_upper_bound",
        "note": "Must not become Ex0S 0.0.004",
    },
}

MINIMALITY_SHAPE = [
    "bare_token",
    "(token, here)",
    "(token, predecessor)",
    "(token, path-origin)",
    "opaque_fact_id",
]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
class ProbeSpec:
    after: str
    cue: str
    expect: str  # frozen 0.0.003 expected motor
    hops: int | None = None
    label: str = ""
    context_expect: str | None = None  # what path-provenance would want (audit only)


@dataclass
class Cell:
    cell_id: str
    axis: str
    level: Any
    holdout: bool
    intended_class: str  # within_model | expected_boundary
    relations: list[Rel]
    probes: list[ProbeSpec]
    phases: list[dict[str, Any]] = field(default_factory=lambda: [{"name": "plant", "steps": []}])
    annotation: str = ""
    discriminates: str = ""
    hypothesis_preds: dict[str, Any] = field(default_factory=dict)
    depth: int = 0


def acquired_edges(rels: list[Rel]) -> set[tuple[str, str]]:
    return {(r.bind.lower(), r.did.lower()) for r in rels}


def edges_in_s(s_dir: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for path in s_dir.glob("*.tag"):
        _fid_v, tags = parse_tagfile(path.read_text(encoding="utf-8"))
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


def probe_cue(policy: Any, s_dir: Path | None, seed: int, cue: str | None) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm012ctx_empty_") as tmp:
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
        out["decision_wall_ms"] = wall_ms
        out["facts_examined"] = hops * n_s
        out["n_s"] = n_s
        out["peak_memory_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out["timed_out"] = False
        return out


def probe_cue_budget(
    policy: Any, s_dir: Path | None, seed: int, cue: str | None, *, timeout_ms: int = TIMEOUT_MS
) -> dict[str, Any]:
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


# ----- generators C0–C7 -----


def gen_c0(seed: int) -> Cell:
    """Unique intermediates — within-model control."""
    rng = np.random.default_rng(seed + 201)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z, q = (_nonce(rng, taken_n) for _ in range(4))
    m1, m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
        Rel(_fid(rng, taken_f), z, q, "zq", (1, 0)),
        Rel(_fid(rng, taken_f), q, m2, "qm", (1, 0)),
    ]
    return Cell(
        cell_id="c0_unique",
        axis="c0",
        level="unique_control",
        holdout=False,
        intended_class="within_model",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m1, 2, "cue_x", context_expect=m1),
            ProbeSpec("plant", z, m2, 2, "cue_z", context_expect=m2),
        ],
        annotation="unique intermediates; compose must still work",
        discriminates="apparatus_intact",
        hypothesis_preds={"H0": "pass", "note": "no reuse collision"},
        depth=2,
    )


def gen_c1(seed: int) -> Cell:
    """Benign reuse — same continuation; reuse alone not pathological."""
    rng = np.random.default_rng(seed + 211)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = (_nonce(rng, taken_n) for _ in range(3))
    m1, _m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), z, y, "zy", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
    ]
    return Cell(
        cell_id="c1_benign_reuse",
        axis="c1",
        level="benign_reuse",
        holdout=False,
        intended_class="within_model",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m1, 2, "cue_x", context_expect=m1),
            ProbeSpec("plant", z, m1, 2, "cue_z", context_expect=m1),
        ],
        annotation="same Y reused; both cues require same motor",
        discriminates="reuse_not_inherently_pathological",
        hypothesis_preds={"H0": "pass"},
        depth=2,
    )


def gen_c2(seed: int) -> Cell:
    """Same token, different here — H1 vs H0.

    Probe station is always chb.
    Y→m_strong @ cha (10); Y→m_weak @ chb (2).
    H0 bind-only → m_strong; H1 here-filter → m_weak.
    Frozen expect: m_strong for both cues.
    CONTEXT-correct: cue via cha path → m_strong; via chb path → m_weak.
    """
    rng = np.random.default_rng(seed + 221)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = (_nonce(rng, taken_n) for _ in range(3))
    m_strong, m_weak = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0), here="cha"),
        Rel(_fid(rng, taken_f), z, y, "zy", (1, 0), here="chb"),
        Rel(_fid(rng, taken_f), y, m_strong, "ys", (10, 0), here="cha"),
        Rel(_fid(rng, taken_f), y, m_weak, "yw", (2, 0), here="chb"),
    ]
    return Cell(
        cell_id="c2_here_split",
        axis="c2",
        level="here_split",
        holdout=False,
        intended_class="expected_boundary",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m_strong, 2, "cue_x", context_expect=m_strong),
            ProbeSpec("plant", z, m_strong, 2, "cue_z", context_expect=m_weak),
        ],
        annotation="expected limit: bind-only; here does not filter at frontier Y",
        discriminates="H1_vs_H0",
        hypothesis_preds={
            "H0": m_strong,
            "H1": m_weak,  # at probe chb
            "disagree": True,
            "H0_ne_H1": m_strong != m_weak,
        },
        depth=2,
    )


def gen_c3(seed: int) -> Cell:
    """Same here, different predecessor — H2 vs H1 (here cannot help).

    Both arrivals tagged chb. CONTEXT wants X→m_x, Z→m_z.
    Frozen bind-only: stronger Y-edge wins for both cues.
    """
    rng = np.random.default_rng(seed + 231)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = (_nonce(rng, taken_n) for _ in range(3))
    m_x, m_z = _two_motors(rng)
    # Stronger edge is m_x — wrong for cue Z under CONTEXT.
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0), here="chb"),
        Rel(_fid(rng, taken_f), z, y, "zy", (1, 0), here="chb"),
        Rel(_fid(rng, taken_f), y, m_x, "yx", (10, 0), here="chb"),
        Rel(_fid(rng, taken_f), y, m_z, "yz", (2, 0), here="chb"),
    ]
    return Cell(
        cell_id="c3_pred_split",
        axis="c3",
        level="pred_split",
        holdout=False,
        intended_class="expected_boundary",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m_x, 2, "cue_x", context_expect=m_x),
            ProbeSpec("plant", z, m_x, 2, "cue_z", context_expect=m_z),
        ],
        annotation="same here; predecessor would distinguish; bind-only collapses",
        discriminates="H2_vs_H1",
        hypothesis_preds={
            "H1": "cannot_distinguish",  # same here
            "H2": {"from_x": m_x, "from_z": m_z},
            "H0_frozen": m_x,
        },
        depth=2,
    )


def gen_c4(seed: int, *, holdout: bool = True) -> Cell:
    """Predecessor collision — H3 vs H2.

    X→A→Y and Z→A→Y share immediate predecessor A.
    """
    rng = np.random.default_rng(seed + 241)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, a, y, z = (_nonce(rng, taken_n) for _ in range(4))
    m_x, m_z = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", (1, 0)),
        Rel(_fid(rng, taken_f), z, a, "za", (1, 0)),
        Rel(_fid(rng, taken_f), a, y, "ay", (1, 0)),
        Rel(_fid(rng, taken_f), y, m_x, "yx", (10, 0)),
        Rel(_fid(rng, taken_f), y, m_z, "yz", (2, 0)),
    ]
    return Cell(
        cell_id="c4_pred_collision",
        axis="c4",
        level="pred_collision",
        holdout=holdout,
        intended_class="expected_boundary",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m_x, 3, "cue_x", context_expect=m_x),
            ProbeSpec("plant", z, m_x, 3, "cue_z", context_expect=m_z),
        ],
        annotation="same pred A; origin cue needed; bind-only → strong Y edge",
        discriminates="H3_vs_H2",
        hypothesis_preds={
            "H2": "cannot_distinguish",
            "H3": {"from_x": m_x, "from_z": m_z},
            "H0_frozen": m_x,
        },
        depth=3,
    )


def gen_c5(seed: int, *, holdout: bool = True) -> Cell:
    """Path-depth provenance — different-length paths converge on Y."""
    rng = np.random.default_rng(seed + 251)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z, p, q = (_nonce(rng, taken_n) for _ in range(5))
    m_short, m_long = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),  # short → Y
        Rel(_fid(rng, taken_f), z, p, "zp", (1, 0)),
        Rel(_fid(rng, taken_f), p, q, "pq", (1, 0)),
        Rel(_fid(rng, taken_f), q, y, "qy", (1, 0)),  # long → Y
        Rel(_fid(rng, taken_f), y, m_short, "ys", (10, 0)),
        Rel(_fid(rng, taken_f), y, m_long, "yl", (2, 0)),
    ]
    return Cell(
        cell_id="c5_path_depth",
        axis="c5",
        level="path_depth",
        holdout=holdout,
        intended_class="expected_boundary",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m_short, 2, "cue_x_short", context_expect=m_short),
            ProbeSpec("plant", z, m_short, 4, "cue_z_long", context_expect=m_long),
        ],
        annotation="path length/history lost at bare Y; bind-only → strong edge",
        discriminates="path_history",
        hypothesis_preds={
            "H0_frozen": m_short,
            "path_aware": {"short": m_short, "long": m_long},
        },
        depth=4,
    )


def gen_c6(seed: int) -> Cell:
    """Evidence trap — wrong-context Y edge much stronger."""
    rng = np.random.default_rng(seed + 261)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = (_nonce(rng, taken_n) for _ in range(3))
    m_right_x, m_wrong = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0), here="cha"),
        Rel(_fid(rng, taken_f), z, y, "zy", (1, 0), here="chb"),
        Rel(_fid(rng, taken_f), y, m_wrong, "ywrong", (100, 0), here="chb"),
        Rel(_fid(rng, taken_f), y, m_right_x, "yright", (2, 0), here="cha"),
    ]
    return Cell(
        cell_id="c6_evidence_trap",
        axis="c6",
        level="evidence_trap",
        holdout=False,
        intended_class="expected_boundary",
        relations=rels,
        probes=[
            ProbeSpec("plant", x, m_wrong, 2, "cue_x", context_expect=m_right_x),
            ProbeSpec("plant", z, m_wrong, 2, "cue_z", context_expect=m_wrong),
        ],
        annotation="EVIDENCE cannot repair representational collapse",
        discriminates="evidence_vs_provenance",
        hypothesis_preds={"H0_frozen": m_wrong, "context_x": m_right_x},
        depth=2,
    )


def _c7_identical_world(seed: int) -> tuple[list[Rel], str, str, str]:
    """Shared concrete S + cue for indistinguishability witness.

    Tied Y→PRESS and Y→TUNE; cue is Y. Frozen ⇒ HOLD.
    """
    rng = np.random.default_rng(seed + 271)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    # Deterministic labels from a fixed stream so a/b match.
    y = _nonce(rng, taken_n)
    m1, m2 = "press", "tune"
    rels = [
        Rel(_fid(rng, taken_f), y, m1, "yp", (1, 0)),
        Rel(_fid(rng, taken_f), y, m2, "yt", (1, 0)),
    ]
    return rels, y, m1, m2


def gen_c7_a(seed: int, *, holdout: bool = True) -> Cell:
    rels, y, m_press, m_tune = _c7_identical_world(seed)
    return Cell(
        cell_id="c7_indistinguish_a",
        axis="c7",
        level="witness_a",
        holdout=holdout,
        intended_class="expected_boundary",
        relations=rels,
        probes=[ProbeSpec("plant", y, "hold", None, "cue_y", context_expect=m_press)],
        annotation="indistinguishability witness A: context wants PRESS; frozen state ⇒ HOLD",
        discriminates="structural_indistinguishability",
        hypothesis_preds={
            "frozen_state_id": "c7_shared",
            "context_expect": m_press,
            "sibling": "c7_indistinguish_b",
            "sibling_context_expect": m_tune,
        },
        depth=1,
    )


def gen_c7_b(seed: int, *, holdout: bool = True) -> Cell:
    rels, y, m_press, m_tune = _c7_identical_world(seed)
    return Cell(
        cell_id="c7_indistinguish_b",
        axis="c7",
        level="witness_b",
        holdout=holdout,
        intended_class="expected_boundary",
        relations=rels,
        probes=[ProbeSpec("plant", y, "hold", None, "cue_y", context_expect=m_tune)],
        annotation="indistinguishability witness B: context wants TUNE; same frozen state as A",
        discriminates="structural_indistinguishability",
        hypothesis_preds={
            "frozen_state_id": "c7_shared",
            "context_expect": m_tune,
            "sibling": "c7_indistinguish_a",
            "sibling_context_expect": m_press,
        },
        depth=1,
    )


def all_cells(seed: int = DEFAULT_SEED) -> list[Cell]:
    return [
        gen_c0(seed),
        gen_c1(seed),
        gen_c2(seed),
        gen_c3(seed),
        gen_c4(seed, holdout=True),
        gen_c5(seed, holdout=True),
        gen_c6(seed),
        gen_c7_a(seed, holdout=True),
        gen_c7_b(seed, holdout=True),
    ]


def seed_list_blob(seed: int = DEFAULT_SEED) -> str:
    return "\n".join(c.cell_id for c in all_cells(seed))


def score_cell(
    cell: Cell,
    *,
    probes_ok: bool,
    invariant_ok: bool,
    timed_out: bool,
    motors: list[str],
) -> dict[str, Any]:
    if timed_out:
        return {
            "class": CLASS_RESOURCE,
            "ok_for_solved": False,
            "expected_boundary_confirmed": False,
            "why": f"timeout after {TIMEOUT_MS}ms",
        }
    if cell.intended_class == "expected_boundary":
        confirmed = probes_ok and invariant_ok
        return {
            "class": CLASS_EXPECTED if confirmed else CLASS_UNEXPECTED,
            "ok_for_solved": False,
            "expected_boundary_confirmed": confirmed,
            "why": cell.annotation if confirmed else f"boundary not confirmed motors={motors}",
        }
    if probes_ok and invariant_ok:
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
        "why": f"semantic fail motors={motors} inv={invariant_ok} probes={probes_ok}",
    }


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    from three_memory.policy import UsePolicy

    cell_id = job["cell_id"]
    seed = int(job["seed"])
    dest = Path(job["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    cell = next(c for c in all_cells(seed) if c.cell_id == cell_id)
    s_dir = dest / "S"
    write_world_s(s_dir, cell.relations)
    policy = UsePolicy(seed=7, lr=0.2)
    probe_results: list[dict[str, Any]] = []
    timed_out = False
    try:
        for i, spec in enumerate(cell.probes):
            hit = probe_cue_budget(policy, s_dir, seed + 20 + i, spec.cue)
            if hit.get("timed_out"):
                timed_out = True
                probe_results.append({**hit, "spec": spec.__dict__})
                break
            probe_results.append({**hit, "spec": spec.__dict__})
    except Exception as exc:  # noqa: BLE001
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
            "motors": [],
            "n_s": len(cell.relations),
            "annotation": cell.annotation,
            "discriminates": cell.discriminates,
            "ex0s_under_test": "0.0.003",
            "earned_next": False,
        }

    invariant_ok = acquired_edge_invariant(s_dir, cell.relations)
    probes_ok = True
    motors: list[str] = []
    wall_sum = 0.0
    facts_sum = 0
    for pr in probe_results:
        spec = pr["spec"]
        got = _motor(pr.get("action_name"))
        motors.append(got)
        wall_sum += float(pr.get("decision_wall_ms") or 0)
        facts_sum += int(pr.get("facts_examined") or 0)
        if got != spec["expect"]:
            probes_ok = False
        if spec.get("hops") is not None and spec["expect"] != "hold":
            if pr.get("compose_hops") != spec["hops"]:
                probes_ok = False

    scored = score_cell(
        cell,
        probes_ok=probes_ok,
        invariant_ok=invariant_ok,
        timed_out=timed_out,
        motors=motors,
    )
    context_expects = [p.context_expect for p in cell.probes]
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
        "motors": motors,
        "context_expects": context_expects,
        "compose_hops": [pr.get("compose_hops") for pr in probe_results],
        "n_s": len(list(s_dir.glob("*.tag"))),
        "depth": cell.depth,
        "decision_wall_ms": wall_sum,
        "facts_examined": facts_sum,
        "invariant_ok": invariant_ok,
        "annotation": cell.annotation,
        "discriminates": cell.discriminates,
        "hypothesis_preds": cell.hypothesis_preds,
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
    }


def apparatus_snapshot() -> dict[str, Any]:
    return {
        "version": "TM.0.12.CONTEXT",
        "ex0s_under_test": "0.0.003",
        "timeout_ms": TIMEOUT_MS,
        "seed": DEFAULT_SEED,
        "seed_list_sha": _sha_bytes(seed_list_blob().encode()),
        "scorer_sha": _sha_src(score_cell),
        "gen_c0_sha": _sha_src(gen_c0),
        "gen_c1_sha": _sha_src(gen_c1),
        "gen_c2_sha": _sha_src(gen_c2),
        "gen_c3_sha": _sha_src(gen_c3),
        "gen_c4_sha": _sha_src(gen_c4),
        "gen_c5_sha": _sha_src(gen_c5),
        "gen_c6_sha": _sha_src(gen_c6),
        "gen_c7_sha": _sha_bytes(
            (inspect.getsource(gen_c7_a) + inspect.getsource(gen_c7_b) + inspect.getsource(_c7_identical_world)).encode()
        ),
        "hypotheses": HYPOTHESES,
        "minimality_shape": MINIMALITY_SHAPE,
        "held_out_cells": [c.cell_id for c in all_cells() if c.holdout],
        "expected_boundary_cells": {
            c.cell_id: {
                "frozen_expects": [p.expect for p in c.probes],
                "context_expects": [p.context_expect for p in c.probes],
                "annotation": c.annotation,
                "discriminates": c.discriminates,
            }
            for c in all_cells()
            if c.intended_class == "expected_boundary"
        },
        "within_model_cells": [c.cell_id for c in all_cells() if c.intended_class == "within_model"],
        "refuse": [
            "(token, here) as victory",
            "GUID / opaque fact_id as Ex0S 0.0.004",
            "LOOKAHEAD in this battery",
            "pre-name 0.0.004 Abstraction",
            "genome cue→motor maps",
        ],
    }


def write_context_lock(path: Path = CONTEXT_LOCK) -> dict[str, Any]:
    snap = apparatus_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_context_lock() -> tuple[bool, str, dict[str, Any]]:
    snap = apparatus_snapshot()
    if not CONTEXT_LOCK.exists():
        return False, "docs/context_012.lock missing", snap
    lock = json.loads(CONTEXT_LOCK.read_text(encoding="utf-8"))
    for key in (
        "timeout_ms",
        "seed_list_sha",
        "scorer_sha",
        "gen_c0_sha",
        "gen_c1_sha",
        "gen_c2_sha",
        "gen_c3_sha",
        "gen_c4_sha",
        "gen_c5_sha",
        "gen_c6_sha",
        "gen_c7_sha",
        "held_out_cells",
        "minimality_shape",
    ):
        if snap[key] != lock.get(key):
            return False, f"context apparatus drift: {key}", snap
    if "c7_indistinguish_a" not in (lock.get("expected_boundary_cells") or {}):
        return False, "c7 witness not preregistered", snap
    if lock.get("hypotheses", {}).get("H4", {}).get("role") != "diagnostic_upper_bound":
        return False, "H4 must remain diagnostic_upper_bound", snap
    return True, "context apparatus frozen", snap


def aggregate(rows: list[dict[str, Any]], *, organism_ok: bool, context_ok: bool) -> dict[str, Any]:
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
    elimination = []
    for r in rows:
        elimination.append(
            {
                "cell_id": r["cell_id"],
                "class": r["class"],
                "discriminates": r.get("discriminates"),
                "motors": r.get("motors"),
                "context_expects": r.get("context_expects"),
                "confirmed_boundary": r.get("expected_boundary_confirmed"),
                "within": r.get("ok_for_solved"),
            }
        )
    # C7 structural check
    c7a = next((r for r in rows if r["cell_id"] == "c7_indistinguish_a"), None)
    c7b = next((r for r in rows if r["cell_id"] == "c7_indistinguish_b"), None)
    c7_witness = None
    if c7a and c7b:
        c7_witness = {
            "same_frozen_motors": c7a.get("motors") == c7b.get("motors"),
            "context_expects_differ": c7a.get("context_expects") != c7b.get("context_expects"),
            "both_confirmed": bool(
                c7a.get("expected_boundary_confirmed") and c7b.get("expected_boundary_confirmed")
            ),
        }
    return {
        "version": "TM.0.12.CONTEXT",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "organism_ok": organism_ok,
        "context_ok": context_ok,
        "n_cells": len(rows),
        "n_within_model": len(within),
        "within_model_pass": len(solved),
        "within_model_pass_frac": (len(solved) / len(within)) if within else 0.0,
        "expected_boundary_n": len(boundaries),
        "expected_boundary_confirmed": len(confirmed),
        "classes": classes,
        "elimination": elimination,
        "c7_witness": c7_witness,
        "hypotheses": HYPOTHESES,
        "minimality_shape": MINIMALITY_SHAPE,
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
            "note": "Representation audit only. No organism edits. No 0.0.004.",
        },
    }


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm012context"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_context(*, seed: int = DEFAULT_SEED, workers: int = 4) -> dict[str, Any]:
    run_dir = _run_dir()
    organism_ok, org_why, _ = verify_organism_freeze()
    context_ok, ctx_why, ctx_snap = verify_context_lock()
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
    summary = aggregate(rows, organism_ok=organism_ok, context_ok=context_ok)
    summary["organism_why"] = org_why
    summary["context_why"] = ctx_why
    summary["run_dir"] = str(run_dir)
    summary["seed"] = seed
    summary["context_lock"] = ctx_snap
    (run_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    elim_lines = "\n".join(
        f"| {e['cell_id']} | {e['class']} | {e.get('discriminates')} | {e.get('motors')} | {e.get('context_expects')} |"
        for e in summary["elimination"]
    )
    (run_dir / "summary.md").write_text(
        f"""# TM.0.12.CONTEXT · Ex0S 0.0.003 under test

Organism: {org_why}
Apparatus: {ctx_why}
Within-model: **{summary['within_model_pass']}/{summary['n_within_model']}**
Expected boundaries: **{summary['expected_boundary_confirmed']}/{summary['expected_boundary_n']}**
Classes: {summary['classes']}
`earned_next`: false (no Ex0S 0.0.004)

| Cell | Class | Discriminates | Frozen motors | Context expects |
|------|-------|---------------|---------------|-----------------|
{elim_lines}

C7 witness: {summary.get('c7_witness')}
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.12.CONTEXT representation audit")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_context_lock(), indent=2))
        return
    s = run_context(seed=args.seed, workers=args.workers)
    print(
        json.dumps(
            {
                "within_model_pass_frac": s["within_model_pass_frac"],
                "within_model_pass": s["within_model_pass"],
                "n_within_model": s["n_within_model"],
                "expected_boundary_confirmed": s["expected_boundary_confirmed"],
                "classes": s["classes"],
                "c7_witness": s.get("c7_witness"),
                "earned_next": s["earned_next"],
                "organism_ok": s["organism_ok"],
                "context_ok": s["context_ok"],
                "holdout": s["holdout"],
                "run_dir": s["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
