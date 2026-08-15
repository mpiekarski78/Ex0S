"""TM.0.14.FAMILY: kill-or-earn on frozen ACQUIRE candidate.

Generated developmental histories (skeleton + life schedules), never planted ctx.
Seal full World including intervention branches. earned_next may become true;
ex0s stays null (no product stamp naming in the runner).

Reuse ACQUIRE helpers: write_skeleton, teacher_outcome, make_acquire.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import (
    ACQUIRE_LOCK,
    GENOME_014_LOCK,
    HERE,
    SOURCE_EXPERIENCE,
    SkelEdge,
    clear_experience_rows,
    list_ctx_any,
    list_experience_ctx,
    make_acquire,
    probe_cue,
    reference_route_kappa,
    restore_experience_bytes,
    stash_experience_bytes,
    teacher_outcome,
    traverse_hold,
    verify_genome_014,
    write_skeleton,
)
from three_memory.kappa import CTX_ENCODING
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import TagStore

DEVELOP = ("A", "B", "C", "D")
HOLDOUT = ("E", "F", "G", "H")
FAMILIES = DEVELOP + HOLDOUT
MOTORS = frozenset({"press", "tune", "flip"})
BANNED = frozenset(MOTORS | {"hold", "idle", "push", "adjust", "open", "wait", "use", "pick"})
CONS = "bcdfghjklmnpqrstvwxz"
VOW = "aeiou"

FAMILY_LOCK = REPO_ROOT / "docs" / "family_014.lock"
KAPPA_LOCK = REPO_ROOT / "docs" / "kappa_013.lock"
GENOME_013_LOCK = REPO_ROOT / "docs" / "genome_013.lock"
GENOME_011_LOCK = REPO_ROOT / "docs" / "genome_011.lock"
DEFAULT_SEED = 12345
DEFAULT_PER_FAMILY = 12
DEFAULT_BIRTHS = 3
EXPECTED_N = len(FAMILIES) * DEFAULT_PER_FAMILY * DEFAULT_BIRTHS  # 288

PREREGISTERED_CLAIM = (
    "A frozen developmental recipe generalizes across unseen generated life "
    "histories by converting experienced outcomes into provenance-sensitive "
    "contextual continuations in S, which persist across working-state reset "
    "and causally steer later behavior without contextual answers being "
    "supplied by the apparatus."
)

GLOBAL_MEASURES = (
    "birth_no_ctx",
    "authored_after_life",
    "persist_rho",
    "persist_newborn",
    "strip_experience_hold",
    "no_apparatus_ctx",
    "weights_stable",
    "genome_delta",
    "no_shortcut_writes",
)

BRANCH_MEASURES = ("donor_transfer", "counterfactual_life")

FAMILY_EXTRA = {
    "A": (),
    "B": ("evidence_accumulation",),
    "C": ("revision_math",),
    "D": ("irrelevant_no_proliferate",),
    "E": (),
    "F": ("multi_kappa_split",),
    "G": ("interleave_newborn",),
    "H": ("order_equiv", "route_association"),
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_json(obj: Any) -> str:
    return _sha_bytes(_canonical_json(obj).encode())


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


# --- World schema (fully sealed) ----------------------------------------------


@dataclass
class LifeEp:
    prefer: str  # 'a' | 'b' | route key
    motor: str
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {"prefer": self.prefer, "motor": self.motor, "success": bool(self.success)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LifeEp":
        return LifeEp(prefer=str(d["prefer"]), motor=str(d["motor"]), success=bool(d["success"]))


@dataclass
class ProbeSpec:
    prefer: str
    expect: str
    phase: str = "after_primary"

    def to_dict(self) -> dict[str, Any]:
        return {"prefer": self.prefer, "expect": self.expect, "phase": self.phase}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ProbeSpec":
        return ProbeSpec(
            prefer=str(d["prefer"]),
            expect=str(d["expect"]),
            phase=str(d.get("phase") or "after_primary"),
        )


@dataclass
class Intervention:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Intervention":
        return Intervention(kind=str(d["kind"]), payload=dict(d.get("payload") or {}))


@dataclass
class World:
    family: str
    holdout: bool
    seed: int
    birth: int
    cue: str
    birth_edges: list[dict[str, Any]]
    primary_life: list[LifeEp]
    counterfactual_life: list[LifeEp]
    donor_life: list[LifeEp]
    probes: list[ProbeSpec]
    interventions: list[Intervention]
    hops_a: list[list[str]]
    hops_b: list[list[str]]
    kappa_a: str
    kappa_b: str
    expect_a: str
    expect_b: str
    apply_donor: bool
    apply_counterfactual: bool
    annotation: str = ""
    depth: int = 2

    def to_manifest(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "holdout": self.holdout,
            "seed": self.seed,
            "birth": self.birth,
            "cue": self.cue,
            "birth_edges": self.birth_edges,
            "primary_life": [e.to_dict() for e in self.primary_life],
            "counterfactual_life": [e.to_dict() for e in self.counterfactual_life],
            "donor_life": [e.to_dict() for e in self.donor_life],
            "probes": [p.to_dict() for p in self.probes],
            "interventions": [i.to_dict() for i in self.interventions],
            "hops_a": self.hops_a,
            "hops_b": self.hops_b,
            "kappa_a": self.kappa_a,
            "kappa_b": self.kappa_b,
            "expect_a": self.expect_a,
            "expect_b": self.expect_b,
            "apply_donor": self.apply_donor,
            "apply_counterfactual": self.apply_counterfactual,
            "annotation": self.annotation,
            "depth": self.depth,
        }

    @staticmethod
    def from_manifest(d: dict[str, Any]) -> "World":
        return World(
            family=str(d["family"]),
            holdout=bool(d["holdout"]),
            seed=int(d["seed"]),
            birth=int(d["birth"]),
            cue=str(d["cue"]),
            birth_edges=list(d["birth_edges"]),
            primary_life=[LifeEp.from_dict(x) for x in d["primary_life"]],
            counterfactual_life=[LifeEp.from_dict(x) for x in d.get("counterfactual_life") or []],
            donor_life=[LifeEp.from_dict(x) for x in d.get("donor_life") or []],
            probes=[ProbeSpec.from_dict(x) for x in d["probes"]],
            interventions=[Intervention.from_dict(x) for x in d.get("interventions") or []],
            hops_a=[list(h) for h in d["hops_a"]],
            hops_b=[list(h) for h in d["hops_b"]],
            kappa_a=str(d["kappa_a"]),
            kappa_b=str(d["kappa_b"]),
            expect_a=str(d["expect_a"]),
            expect_b=str(d["expect_b"]),
            apply_donor=bool(d.get("apply_donor")),
            apply_counterfactual=bool(d.get("apply_counterfactual")),
            annotation=str(d.get("annotation") or ""),
            depth=int(d.get("depth") or 2),
        )


def world_manifest_sha(w: World) -> str:
    return _sha_json(w.to_manifest())


def lint_world_no_ctx(w: World) -> None:
    for e in w.birth_edges:
        if "ctx" in e:
            raise ValueError("world birth_edges refuse ctx")


# --- Skeleton builders --------------------------------------------------------


def _edges_from_dicts(rows: Sequence[dict[str, Any]]) -> list[SkelEdge]:
    out: list[SkelEdge] = []
    for r in rows:
        if "ctx" in r:
            raise ValueError("skeleton refuses ctx")
        out.append(
            SkelEdge(
                fid=str(r["fid"]),
                bind=str(r["bind"]),
                did=str(r["did"]),
                support=int(r.get("support") or 1),
                contradiction=int(r.get("contradiction") or 0),
            )
        )
    return out


def _apply_prefer_to_skel(
    edges: Sequence[dict[str, Any]], prefer: str, mid_a: str, mid_b: str, cue: str
) -> list[SkelEdge]:
    rows: list[SkelEdge] = []
    for e in edges:
        if "ctx" in e:
            raise ValueError("refuses ctx")
        s = int(e.get("support") or 1)
        if str(e["bind"]).lower() == cue.lower():
            did = str(e["did"]).lower()
            if prefer == "a":
                s = 10 if did == mid_a.lower() else 1
            elif prefer == "b":
                s = 10 if did == mid_b.lower() else 1
        rows.append(
            SkelEdge(
                fid=str(e["fid"]),
                bind=str(e["bind"]),
                did=str(e["did"]),
                support=s,
                contradiction=int(e.get("contradiction") or 0),
            )
        )
    return rows


def _diamond_tokens(rng: np.random.Generator, *, depth: int = 2) -> dict[str, Any]:
    """Build dual routes of exactly `depth` hops from cue to shared frontier y."""
    taken: set[str] = set()
    cue = _nonce(rng, taken)
    mid_a = _nonce(rng, taken)
    mid_b = _nonce(rng, taken)
    y = _nonce(rng, taken)
    hops_a: list[tuple[str, str]] = [(cue, mid_a)]
    hops_b: list[tuple[str, str]] = [(cue, mid_b)]
    edges: list[dict[str, Any]] = []
    fids: set[str] = set()
    prev_a, prev_b = mid_a, mid_b
    edges.append(
        {
            "fid": _fid(rng, fids),
            "bind": cue,
            "did": mid_a,
            "support": 1,
            "contradiction": 0,
        }
    )
    edges.append(
        {
            "fid": _fid(rng, fids),
            "bind": cue,
            "did": mid_b,
            "support": 1,
            "contradiction": 0,
        }
    )
    # After cue→mid, need (depth-1) more hops ending at y.
    # depth=2: one hop mid→y. depth=3: mid→z→y, etc.
    remaining = max(depth - 1, 1)
    for i in range(remaining):
        last = i == remaining - 1
        nxt_a = y if last else _nonce(rng, taken)
        nxt_b = y if last else _nonce(rng, taken)
        hops_a.append((prev_a, nxt_a))
        hops_b.append((prev_b, nxt_b))
        edges.append(
            {
                "fid": _fid(rng, fids),
                "bind": prev_a,
                "did": nxt_a,
                "support": 1,
                "contradiction": 0,
            }
        )
        edges.append(
            {
                "fid": _fid(rng, fids),
                "bind": prev_b,
                "did": nxt_b,
                "support": 1,
                "contradiction": 0,
            }
        )
        prev_a, prev_b = nxt_a, nxt_b
    ka = reference_route_kappa(cue, hops_a)
    kb = reference_route_kappa(cue, hops_b)
    assert ka != kb
    assert len(hops_a) == depth and len(hops_b) == depth
    return {
        "cue": cue,
        "mid_a": mid_a,
        "mid_b": mid_b,
        "y": y,
        "edges": edges,
        "hops_a": [list(h) for h in hops_a],
        "hops_b": [list(h) for h in hops_b],
        "kappa_a": ka,
        "kappa_b": kb,
        "fids": fids,
        "taken": taken,
        "rng": rng,
    }


def _base_interventions(*, donor: bool, counterfactual: bool) -> list[Intervention]:
    out = [
        Intervention("persist_rho", {}),
        Intervention("persist_newborn", {}),
        Intervention("strip_experience", {}),
    ]
    if donor:
        out.append(Intervention("donor_apply", {}))
    if counterfactual:
        out.append(Intervention("counterfactual_run", {}))
    return out


def _make_world(
    family: str,
    seed: int,
    birth: int,
    *,
    depth: int,
    primary: list[LifeEp],
    counterfactual: list[LifeEp],
    donor: list[LifeEp],
    probes: list[ProbeSpec],
    apply_donor: bool,
    apply_counterfactual: bool,
    annotation: str,
    clutter: int = 0,
    extra_interventions: Sequence[Intervention] = (),
) -> World:
    rng = np.random.default_rng(seed + 17 * birth + 1009 * ord(family[0]))
    tok = _diamond_tokens(rng, depth=depth)
    edges = list(tok["edges"])
    fids = set(tok["fids"])
    taken = set(tok["taken"])
    for _ in range(clutter):
        a = _nonce(rng, taken)
        b = _nonce(rng, taken)
        edges.append(
            {
                "fid": _fid(rng, fids),
                "bind": a,
                "did": b,
                "support": 1,
                "contradiction": 0,
            }
        )
    interventions = _base_interventions(donor=apply_donor, counterfactual=apply_counterfactual)
    interventions.extend(extra_interventions)
    w = World(
        family=family,
        holdout=family in HOLDOUT,
        seed=seed,
        birth=birth,
        cue=tok["cue"],
        birth_edges=edges,
        primary_life=primary,
        counterfactual_life=counterfactual,
        donor_life=donor,
        probes=probes,
        interventions=interventions,
        hops_a=tok["hops_a"],
        hops_b=tok["hops_b"],
        kappa_a=tok["kappa_a"],
        kappa_b=tok["kappa_b"],
        expect_a="press",
        expect_b="tune",
        apply_donor=apply_donor,
        apply_counterfactual=apply_counterfactual,
        annotation=annotation,
        depth=depth,
    )
    # stash mid tokens for prefer steering
    w.annotation = json.dumps(
        {
            "note": annotation,
            "mid_a": tok["mid_a"],
            "mid_b": tok["mid_b"],
            "y": tok["y"],
        },
        sort_keys=True,
    )
    lint_world_no_ctx(w)
    return w


def _mids(w: World) -> tuple[str, str]:
    meta = json.loads(w.annotation)
    return str(meta["mid_a"]), str(meta["mid_b"])


# --- Generators A–H -----------------------------------------------------------


def gen_A(seed: int, birth: int) -> World:
    """One/two contextual experiences (route-split)."""
    apply_d = birth % 3 == 0
    apply_c = birth % 3 == 1 or birth % 3 == 0
    return _make_world(
        "A",
        seed,
        birth,
        depth=2,
        primary=[LifeEp("a", "press", True), LifeEp("b", "tune", True)],
        counterfactual=[LifeEp("a", "tune", True), LifeEp("b", "press", True)] if apply_c else [],
        donor=[LifeEp("a", "tune", True)] if apply_d else [],
        probes=[
            ProbeSpec("a", "press", "after_primary"),
            ProbeSpec("b", "tune", "after_primary"),
        ],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation="develop_route_split",
    )


def gen_B(seed: int, birth: int) -> World:
    """Repeated success + evidence accumulation."""
    n = 2 + (birth % 2)
    primary = [LifeEp("a", "press", True) for _ in range(n)]
    apply_d = birth % 4 == 0
    apply_c = birth % 4 == 1
    return _make_world(
        "B",
        seed,
        birth,
        depth=2,
        primary=primary,
        counterfactual=[LifeEp("a", "tune", True)] if apply_c else [],
        donor=[LifeEp("b", "tune", True)] if apply_d else [],
        probes=[ProbeSpec("a", "press", "after_primary")],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation=f"accumulate_n={n}",
        extra_interventions=[Intervention("check_support", {"prefer": "a", "motor": "press", "min_support": n})],
    )


def gen_C(seed: int, birth: int) -> World:
    """Contradiction/revision: success then failures; probe still uses evidence."""
    primary = [
        LifeEp("a", "press", True),
        LifeEp("a", "press", False),
        LifeEp("a", "press", False),
    ]
    apply_d = birth % 3 == 0
    apply_c = birth % 3 != 2
    return _make_world(
        "C",
        seed,
        birth,
        depth=2,
        primary=primary,
        counterfactual=[LifeEp("a", "tune", True)] if apply_c else [],
        donor=[LifeEp("b", "tune", True)] if apply_d else [],
        probes=[ProbeSpec("a", "press", "after_primary")],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation="revision_press_fails",
        extra_interventions=[
            Intervention(
                "check_evidence",
                {"prefer": "a", "motor": "press", "support": 1, "contradiction": 2},
            )
        ],
    )


def gen_D(seed: int, birth: int) -> World:
    """Clutter + irrelevant outcomes (non-motor teacher after lived clear)."""
    apply_d = birth % 2 == 0
    apply_c = birth % 2 == 1
    return _make_world(
        "D",
        seed,
        birth,
        depth=2,
        primary=[LifeEp("a", "press", True), LifeEp("b", "tune", True)],
        counterfactual=[LifeEp("a", "tune", True)] if apply_c else [],
        donor=[LifeEp("a", "press", True)] if apply_d else [],
        probes=[
            ProbeSpec("a", "press", "after_primary"),
            ProbeSpec("b", "tune", "after_primary"),
        ],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation="clutter_irrelevant",
        clutter=4 + birth,
        extra_interventions=[Intervention("irrelevant_outcomes", {"n": 2})],
    )


def gen_E(seed: int, birth: int) -> World:
    """Unseen deeper lives."""
    depth = 3 + (birth % 2)
    apply_d = birth % 3 == 0
    apply_c = birth % 3 == 1
    return _make_world(
        "E",
        seed,
        birth,
        depth=depth,
        primary=[LifeEp("a", "press", True), LifeEp("b", "tune", True)],
        counterfactual=[LifeEp("a", "tune", True), LifeEp("b", "press", True)] if apply_c else [],
        donor=[LifeEp("b", "tune", True)] if apply_d else [],
        probes=[
            ProbeSpec("a", "press", "after_primary"),
            ProbeSpec("b", "tune", "after_primary"),
        ],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation=f"deep_{depth}",
    )


def gen_F(seed: int, birth: int) -> World:
    """Multiple κ at same frontier."""
    apply_d = True
    apply_c = True
    return _make_world(
        "F",
        seed,
        birth,
        depth=2,
        primary=[LifeEp("a", "press", True), LifeEp("b", "tune", True)],
        counterfactual=[LifeEp("a", "tune", True), LifeEp("b", "press", True)],
        donor=[LifeEp("a", "tune", True)],
        probes=[
            ProbeSpec("a", "press", "after_primary"),
            ProbeSpec("b", "tune", "after_primary"),
        ],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation="multi_kappa",
        extra_interventions=[Intervention("assert_two_kappa", {})],
    )


def gen_G(seed: int, birth: int) -> World:
    """Interleaved lives + newborn between episodes."""
    apply_d = birth % 2 == 0
    apply_c = birth % 2 == 1
    return _make_world(
        "G",
        seed,
        birth,
        depth=2,
        primary=[LifeEp("a", "press", True)],
        counterfactual=[LifeEp("b", "tune", True)] if apply_c else [],
        donor=[LifeEp("b", "tune", True)] if apply_d else [],
        probes=[
            ProbeSpec("a", "press", "after_primary"),
            ProbeSpec("b", "tune", "after_interleave"),
        ],
        apply_donor=apply_d,
        apply_counterfactual=apply_c,
        annotation="interleave_newborn",
        extra_interventions=[
            Intervention("newborn_between", {}),
            Intervention("life_after_newborn", {"prefer": "b", "motor": "tune", "success": True}),
        ],
    )


def gen_H(seed: int, birth: int) -> World:
    """Mixed adversarial: order-equiv + route-association counterfactual."""
    # Same multiset of outcomes, different temporal order → same final evidence.
    primary = [
        LifeEp("a", "press", True),
        LifeEp("a", "press", False),
        LifeEp("b", "tune", True),
    ]
    order_alt = [
        LifeEp("a", "press", True),
        LifeEp("b", "tune", True),
        LifeEp("a", "press", False),
    ]
    extra = [
        Intervention("order_equiv", {"alt_life": [e.to_dict() for e in order_alt]}),
        Intervention("route_association", {}),
        Intervention("assert_two_kappa", {}),
    ]
    return _make_world(
        "H",
        seed,
        birth,
        depth=2 + (birth % 2),
        primary=primary,
        counterfactual=[LifeEp("a", "tune", True), LifeEp("b", "press", True)],
        donor=[LifeEp("a", "tune", True)],
        probes=[
            ProbeSpec("a", "press", "after_primary"),
            ProbeSpec("b", "tune", "after_primary"),
        ],
        apply_donor=True,
        apply_counterfactual=True,
        annotation="order_and_association",
        clutter=2,
        extra_interventions=extra,
    )


FAMILY_GENERATORS: dict[str, Callable[[int, int], World]] = {
    "A": gen_A,
    "B": gen_B,
    "C": gen_C,
    "D": gen_D,
    "E": gen_E,
    "F": gen_F,
    "G": gen_G,
    "H": gen_H,
}


def generate_world(family: str, seed: int, birth: int) -> World:
    return FAMILY_GENERATORS[family](seed, birth)


def seed_jobs(
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
    families: Sequence[str] = FAMILIES,
) -> list[tuple[str, int, int]]:
    jobs: list[tuple[str, int, int]] = []
    for fam in families:
        fi = FAMILIES.index(fam)
        for w in range(per_family):
            for b in range(births):
                jobs.append((fam, seed + 1000 * fi + w, b))
    return jobs


def seed_jobs_sha(
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> str:
    return _sha_json(seed_jobs(seed, per_family, births))


def is_full_canonical(
    seed: int, per_family: int, births: int, families: Sequence[str]
) -> bool:
    return (
        seed == DEFAULT_SEED
        and per_family == DEFAULT_PER_FAMILY
        and births == DEFAULT_BIRTHS
        and tuple(families) == FAMILIES
    )


# --- Provenance ledger + scoring ----------------------------------------------


@dataclass
class Ledger:
    birth_ctx: int = 0
    created_by_observe_outcome: int = 0
    copied_for_donor: int = 0
    removed_for_strip: int = 0
    unexpected_ctx_writes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "birth_ctx": self.birth_ctx,
            "created_by_observe_outcome": self.created_by_observe_outcome,
            "copied_for_donor": self.copied_for_donor,
            "removed_for_strip": self.removed_for_strip,
            "unexpected_ctx_writes": self.unexpected_ctx_writes,
        }


def _audit_provenance(s_dir: Path, ledger: Ledger) -> list[str]:
    errors: list[str] = []
    for p in sorted(s_dir.glob("*.tag")):
        fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        has_ctx = isinstance(tags.get("ctx"), str) and bool(tags.get("ctx"))
        src = str(tags.get("source") or "")
        if has_ctx and src != SOURCE_EXPERIENCE:
            ledger.unexpected_ctx_writes += 1
            errors.append(f"ctx_without_experience:{fid}")
        if src == SOURCE_EXPERIENCE:
            if not has_ctx:
                ledger.unexpected_ctx_writes += 1
                errors.append(f"experience_without_ctx:{fid}")
            did = str(tags.get("did") or "").lower()
            bind = str(tags.get("bind") or "").lower()
            if did not in MOTORS:
                ledger.unexpected_ctx_writes += 1
                errors.append(f"experience_non_motor:{fid}")
            if bind in MOTORS or not bind:
                ledger.unexpected_ctx_writes += 1
                errors.append(f"experience_bad_bind:{fid}")
    return errors


def _write_birth(s_dir: Path, w: World, prefer: str) -> None:
    mid_a, mid_b = _mids(w)
    edges = _apply_prefer_to_skel(w.birth_edges, prefer, mid_a, mid_b, w.cue)
    write_skeleton(s_dir, edges)


def _run_life_eps(
    ag: Any,
    s_dir: Path,
    w: World,
    eps: Sequence[LifeEp],
    ledger: Ledger,
    *,
    seed: int,
) -> list[str]:
    """Author episodes: first success needs lived HOLD; bump/revise via probe."""
    errors: list[str] = []
    mid_a, mid_b = _mids(w)
    for ep in eps:
        edges = _apply_prefer_to_skel(w.birth_edges, ep.prefer, mid_a, mid_b, w.cue)
        exp = stash_experience_bytes(s_dir)
        write_skeleton(s_dir, edges)
        restore_experience_bytes(s_dir, exp)
        if hasattr(ag.store, "reload"):
            ag.store.reload()
        before = {r["fid"] for r in list_experience_ctx(s_dir)}
        expect_kappa = w.kappa_a if ep.prefer == "a" else w.kappa_b
        existing = [
            r
            for r in list_experience_ctx(s_dir)
            if r["did"] == ep.motor and r["ctx"] == expect_kappa
        ]
        if ep.success and not existing:
            trav = traverse_hold(ag, w.cue, seed=seed)
            if not trav.get("lived_pending"):
                errors.append(f"no_lived_for_{ep.motor}")
                continue
            teacher_outcome(ag, ep.motor, success=True)
        else:
            # Support bump on existing (κ,motor) or revise/fail after select.
            p = probe_cue(ag, w.cue, seed=seed)
            if ep.success and existing and p["motor"] != ep.motor:
                errors.append(f"bump_probe_{p['motor']}_ne_{ep.motor}")
                continue
            teacher_outcome(ag, ep.motor, success=ep.success)
        after = list_experience_ctx(s_dir)
        new_fids = {r["fid"] for r in after} - before
        if ep.success and new_fids:
            ledger.created_by_observe_outcome += len(new_fids)
        errors.extend(_audit_provenance(s_dir, ledger))
    return errors


def _probe_prefer(s_dir: Path, w: World, prefer: str, policy: UsePolicy, seed: int) -> str:
    mid_a, mid_b = _mids(w)
    exp = stash_experience_bytes(s_dir)
    edges = _apply_prefer_to_skel(w.birth_edges, prefer, mid_a, mid_b, w.cue)
    write_skeleton(s_dir, edges)
    restore_experience_bytes(s_dir, exp)
    ag = make_acquire(s_dir, policy)
    return probe_cue(ag, w.cue, seed=seed)["motor"]


def score_world(
    w: World,
    dest: Path,
    *,
    policy: UsePolicy | None = None,
    genome_ok: bool = True,
) -> dict[str, Any]:
    """Execute only committed lives/interventions. No post-hoc life invention."""
    pol = policy or UsePolicy(seed=w.seed)
    ledger = Ledger()
    measures: dict[str, Any] = {m: None for m in GLOBAL_MEASURES}
    for m in BRANCH_MEASURES:
        measures[m] = None
    for m in FAMILY_EXTRA.get(w.family, ()):
        measures[m] = None
    errors: list[str] = []
    s_dir = dest / "primary"
    if s_dir.exists():
        shutil.rmtree(s_dir)

    # Birth
    _write_birth(s_dir, w, prefer="a")
    ledger.birth_ctx = len(list_ctx_any(s_dir))
    measures["birth_no_ctx"] = ledger.birth_ctx == 0
    if ledger.birth_ctx != 0:
        errors.append("birth_has_ctx")

    ag = make_acquire(s_dir, pol)
    w0 = ag.weight_hash()

    # Primary life
    errors.extend(_run_life_eps(ag, s_dir, w, w.primary_life, ledger, seed=w.seed))
    rows = list_experience_ctx(s_dir)
    # Authored expectation: at least one experience_ctx after successful primary eps
    n_success = sum(1 for e in w.primary_life if e.success)
    measures["authored_after_life"] = len(rows) >= 1 and n_success >= 1
    if not measures["authored_after_life"]:
        errors.append("no_authored")

    # Oracle: primary association is press↔κa, tune↔κb (not membership alone).
    for r in rows:
        if r["did"] == "press" and r["ctx"] != w.kappa_a:
            errors.append("oracle_press_mismatch")
        if r["did"] == "tune" and r["ctx"] != w.kappa_b:
            errors.append("oracle_tune_mismatch")

    # Probes after primary
    for ps in w.probes:
        if ps.phase != "after_primary":
            continue
        got = _probe_prefer(s_dir, w, ps.prefer, pol, w.seed)
        if got != ps.expect:
            errors.append(f"probe_{ps.prefer}_{got}_ne_{ps.expect}")

    # Interventions (committed only)
    for iv in w.interventions:
        kind = iv.kind
        if kind == "persist_rho":
            ag2 = make_acquire(s_dir, pol)
            ag2.reset_rho()
            ok = True
            for ps in w.probes:
                if ps.phase != "after_primary":
                    continue
                if _probe_prefer(s_dir, w, ps.prefer, pol, w.seed + 1) != ps.expect:
                    ok = False
            measures["persist_rho"] = ok
        elif kind == "persist_newborn":
            ag2 = make_acquire(s_dir, pol)
            nb = ag2.clone_empty(store_enabled=True)
            nb.store = TagStore(s_dir, enabled=True)
            ok = True
            for ps in w.probes:
                if ps.phase != "after_primary":
                    continue
                mid_a, mid_b = _mids(w)
                exp = stash_experience_bytes(s_dir)
                edges = _apply_prefer_to_skel(w.birth_edges, ps.prefer, mid_a, mid_b, w.cue)
                write_skeleton(s_dir, edges)
                restore_experience_bytes(s_dir, exp)
                nb.store = TagStore(s_dir, enabled=True)
                if probe_cue(nb, w.cue, seed=w.seed)["motor"] != ps.expect:
                    ok = False
            measures["persist_newborn"] = ok
        elif kind == "strip_experience":
            # Work on a copy
            strip_dir = dest / "strip"
            if strip_dir.exists():
                shutil.rmtree(strip_dir)
            shutil.copytree(s_dir, strip_dir)
            before = len(list_experience_ctx(strip_dir))
            clear_experience_rows(strip_dir)
            ledger.removed_for_strip += before
            got = _probe_prefer(strip_dir, w, "a", pol, w.seed)
            measures["strip_experience_hold"] = got == "hold"
        elif kind == "donor_apply" and w.apply_donor and w.donor_life:
            donor_dir = dest / "donor"
            if donor_dir.exists():
                shutil.rmtree(donor_dir)
            _write_birth(donor_dir, w, prefer="a")
            dag = make_acquire(donor_dir, pol)
            dled = Ledger()
            errors.extend(_run_life_eps(dag, donor_dir, w, w.donor_life, dled, seed=w.seed + 9))
            # Recipient: same birth skeleton, donor experience bytes
            recv = dest / "recv"
            if recv.exists():
                shutil.rmtree(recv)
            _write_birth(recv, w, prefer="a")
            blobs = stash_experience_bytes(donor_dir)
            restore_experience_bytes(recv, blobs)
            ledger.copied_for_donor += len(blobs)
            # Behavior follows donor: probe prefer matching donor's first success
            dep = next((e for e in w.donor_life if e.success), None)
            if dep:
                got = _probe_prefer(recv, w, dep.prefer, pol, w.seed)
                measures["donor_transfer"] = got == dep.motor
            else:
                measures["donor_transfer"] = False
        elif kind == "counterfactual_run" and w.apply_counterfactual and w.counterfactual_life:
            cf = dest / "cf"
            if cf.exists():
                shutil.rmtree(cf)
            _write_birth(cf, w, prefer="a")
            cag = make_acquire(cf, pol)
            cled = Ledger()
            errors.extend(
                _run_life_eps(cag, cf, w, w.counterfactual_life, cled, seed=w.seed + 3)
            )
            # Behavioral divergence required (list inequality is not enough).
            p_mot = _probe_prefer(s_dir, w, "a", pol, w.seed)
            c_mot = _probe_prefer(cf, w, "a", pol, w.seed)
            measures["counterfactual_life"] = p_mot != c_mot
        elif kind == "check_support":
            prefer = str(iv.payload.get("prefer") or "a")
            motor = str(iv.payload.get("motor") or "press")
            mn = int(iv.payload.get("min_support") or 1)
            rows = [r for r in list_experience_ctx(s_dir) if r["did"] == motor]
            ok = bool(rows) and rows[0]["support"] >= mn
            measures["evidence_accumulation"] = ok
        elif kind == "check_evidence":
            motor = str(iv.payload.get("motor") or "press")
            s = int(iv.payload.get("support") or 0)
            c = int(iv.payload.get("contradiction") or 0)
            rows = [r for r in list_experience_ctx(s_dir) if r["did"] == motor]
            measures["revision_math"] = (
                bool(rows) and rows[0]["support"] == s and rows[0]["contradiction"] == c
            )
        elif kind == "irrelevant_outcomes":
            # After lived clear via empty-cue act, non-motor success must not write
            agi = make_acquire(s_dir, pol)
            from experiments.run_tm040 import probe as _probe

            _probe(agi, "probe_channel_b", w.seed + 77, tokens=frozenset())
            before = len(list_experience_ctx(s_dir))
            teacher_outcome(agi, "a", success=True)
            after = len(list_experience_ctx(s_dir))
            measures["irrelevant_no_proliferate"] = after == before
        elif kind == "assert_two_kappa":
            ctxs = {r["ctx"] for r in list_experience_ctx(s_dir)}
            measures["multi_kappa_split"] = w.kappa_a in ctxs and w.kappa_b in ctxs
        elif kind == "newborn_between":
            # Phase marker only — leave measure None until life_after_newborn.
            pass
        elif kind == "life_after_newborn":
            nb = make_acquire(s_dir, pol).clone_empty(store_enabled=True)
            nb.store = TagStore(s_dir, enabled=True)
            ep = LifeEp(
                prefer=str(iv.payload["prefer"]),
                motor=str(iv.payload["motor"]),
                success=bool(iv.payload["success"]),
            )
            errors.extend(_run_life_eps(nb, s_dir, w, [ep], ledger, seed=w.seed + 5))
            got = _probe_prefer(s_dir, w, ep.prefer, pol, w.seed)
            ok = got == ep.motor
            for ps in w.probes:
                if ps.phase == "after_interleave":
                    if _probe_prefer(s_dir, w, ps.prefer, pol, w.seed) != ps.expect:
                        ok = False
            measures["interleave_newborn"] = ok
        elif kind == "order_equiv":
            alt = [LifeEp.from_dict(x) for x in iv.payload.get("alt_life") or []]
            alt_dir = dest / "order_alt"
            if alt_dir.exists():
                shutil.rmtree(alt_dir)
            _write_birth(alt_dir, w, prefer="a")
            aag = make_acquire(alt_dir, pol)
            aled = Ledger()
            errors.extend(_run_life_eps(aag, alt_dir, w, alt, aled, seed=w.seed + 11))
            # Compare evidence bags
            def bag(d: Path) -> set[tuple[str, str, int, int]]:
                return {
                    (r["did"], r["ctx"], r["support"], r["contradiction"])
                    for r in list_experience_ctx(d)
                }

            same = bag(s_dir) == bag(alt_dir)
            pa = _probe_prefer(s_dir, w, "a", pol, w.seed)
            pb = _probe_prefer(alt_dir, w, "a", pol, w.seed)
            measures["order_equiv"] = same and pa == pb
        elif kind == "route_association":
            # Same skeleton/motors/counts; swapped κ↔motor association → different mind.
            # Scorer must not invent lives — counterfactual_life is sealed in the World.
            if not w.counterfactual_life:
                errors.append("route_association_missing_counterfactual_life")
                measures["route_association"] = False
                continue
            cf = dest / "assoc_cf"
            if cf.exists():
                shutil.rmtree(cf)
            _write_birth(cf, w, prefer="a")
            cag = make_acquire(cf, pol)
            cled = Ledger()
            errors.extend(
                _run_life_eps(cag, cf, w, w.counterfactual_life, cled, seed=w.seed + 13)
            )
            measures["route_association"] = _probe_prefer(
                s_dir, w, "a", pol, w.seed
            ) != _probe_prefer(cf, w, "a", pol, w.seed)

    # Global remaining
    errors.extend(_audit_provenance(s_dir, ledger))
    measures["no_apparatus_ctx"] = ledger.unexpected_ctx_writes == 0 and not any(
        "ctx_without" in e for e in errors
    )
    ag_end = make_acquire(s_dir, pol)
    measures["weights_stable"] = ag_end.weight_hash() == w0 and ag_end.weights_unchanged()
    measures["genome_delta"] = bool(genome_ok)
    # no shortcuts: X→motor with ctx should not appear as cue→motor
    shortcuts = [
        r
        for r in list_experience_ctx(s_dir)
        if r["bind"] == w.cue.lower() and r["did"] in MOTORS
    ]
    measures["no_shortcut_writes"] = len(shortcuts) == 0

    # Branch N/A when not applicable
    if not w.apply_donor:
        if measures["donor_transfer"] is None:
            measures["donor_transfer"] = "n/a"
    if not w.apply_counterfactual:
        if measures["counterfactual_life"] is None:
            measures["counterfactual_life"] = "n/a"

    applicable_ok = True
    for k, v in measures.items():
        if v is None:
            applicable_ok = False
        elif v == "n/a":
            continue
        elif v is not True:
            applicable_ok = False

    solved = (
        applicable_ok
        and measures["birth_no_ctx"] is True
        and measures["authored_after_life"] is True
        and measures["no_apparatus_ctx"] is True
        and ledger.unexpected_ctx_writes == 0
        and not errors
    )
    return {
        "family": w.family,
        "seed": w.seed,
        "birth": w.birth,
        "holdout": w.holdout,
        "solved": solved,
        "measures": measures,
        "ledger": ledger.to_dict(),
        "errors": errors,
        "manifest_sha": world_manifest_sha(w),
        "n_experience": len(list_experience_ctx(s_dir)),
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    fam = str(job["family"])
    seed = int(job["seed"])
    birth = int(job["birth"])
    dest = Path(job["dest"])
    allow = bool(job.get("allow_holdout_behavior"))
    genome_ok = bool(job.get("genome_ok", True))
    w = generate_world(fam, seed, birth)
    if w.holdout and not allow:
        return {
            "family": fam,
            "seed": seed,
            "birth": birth,
            "holdout": True,
            "solved": False,
            "sealed": True,
            "errors": ["holdout_behavior_sealed"],
            "measures": {},
            "ledger": {},
            "manifest_sha": world_manifest_sha(w),
        }
    dest.mkdir(parents=True, exist_ok=True)
    return score_world(w, dest, genome_ok=genome_ok)


# --- Lock / seal / earn -------------------------------------------------------


def holdout_manifests(
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per: dict[str, list[str]] = {f: [] for f in HOLDOUT}
    for fam, s, b in seed_jobs(seed, per_family, births, HOLDOUT):
        w = generate_world(fam, s, b)
        lint_world_no_ctx(w)
        sha = world_manifest_sha(w)
        rows.append({"family": fam, "seed": s, "birth": b, "manifest_sha": sha})
        per[fam].append(sha)
    return {
        "holdout_manifest_sha": _sha_json(rows),
        "per_family_sha": {f: _sha_json(per[f]) for f in HOLDOUT},
        "n_holdout_worlds": len(rows),
        "rows": rows,
    }


def committed_holdout_row_shas(path: Path = FAMILY_LOCK) -> set[str]:
    """Committed E–H world manifest SHAs from the sealed lock (not live regen)."""
    if not path.exists():
        return set()
    lock = json.loads(path.read_text(encoding="utf-8"))
    rows = lock.get("holdout_row_shas")
    if not isinstance(rows, list):
        return set()
    return {str(x) for x in rows}


def earn_gate(summary: dict[str, Any], *, full_canonical: bool) -> dict[str, Any]:
    out = {
        "earned_next": False,
        "ex0s": None,
        "claim": PREREGISTERED_CLAIM,
        "why": "",
    }
    if not full_canonical:
        out["why"] = "not full canonical dims"
        return out
    if not summary.get("family_lock_ok"):
        out["why"] = "family_lock"
        return out
    if summary.get("n_worlds") != EXPECTED_N:
        out["why"] = "n_worlds"
        return out
    if summary.get("unique_jobs") != EXPECTED_N:
        out["why"] = "unique_jobs"
        return out
    fams = summary.get("families") or {}
    for f in FAMILIES:
        if (fams.get(f) or {}).get("solved") != DEFAULT_PER_FAMILY * DEFAULT_BIRTHS:
            out["why"] = f"family_{f}"
            return out
    if summary.get("holdout_solved") != 144:
        out["why"] = "holdout"
        return out
    if not summary.get("genome_014_ok"):
        out["why"] = "genome_014"
        return out
    if summary.get("unexpected_ctx_writes", 1) != 0:
        out["why"] = "unexpected_ctx"
        return out
    if summary.get("birth_ctx_total", 1) != 0:
        out["why"] = "birth_ctx"
        return out
    if not summary.get("holdout_manifest_ok"):
        out["why"] = "manifests"
        return out
    if not summary.get("holdout_rows_match_lock"):
        out["why"] = "holdout_rows_lock"
        return out
    if summary.get("n_errors", 1) != 0:
        out["why"] = "errors"
        return out
    # Branch coverage: every family has ≥1 donor and ≥1 counterfactual applicable true
    cov = summary.get("branch_coverage") or {}
    for f in FAMILIES:
        c = cov.get(f) or {}
        if not c.get("donor"):
            out["why"] = f"coverage_donor_{f}"
            return out
        if not c.get("counterfactual"):
            out["why"] = f"coverage_cf_{f}"
            return out
    out["earned_next"] = True
    out["why"] = "earn_gate_pass"
    return out


def family_lock_snapshot() -> dict[str, Any]:
    manifests = holdout_manifests()
    return {
        "version": "TM.0.14.FAMILY",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s_if_earned": None,
        "preregistered_claim": PREREGISTERED_CLAIM,
        "ctx_encoding": CTX_ENCODING,
        "seed": DEFAULT_SEED,
        "per_family": DEFAULT_PER_FAMILY,
        "births": DEFAULT_BIRTHS,
        "expected_n": EXPECTED_N,
        "develop": list(DEVELOP),
        "holdout": list(HOLDOUT),
        "seed_jobs_sha": seed_jobs_sha(),
        "generator_sha": {f: _sha_src(FAMILY_GENERATORS[f]) for f in FAMILIES},
        "scorer_sha": _sha_src(score_world),
        "earn_gate_sha": _sha_src(earn_gate),
        "run_one_sha": _sha_src(run_one),
        "holdout_manifest_sha": manifests["holdout_manifest_sha"],
        "holdout_per_family_sha": manifests["per_family_sha"],
        "holdout_row_shas": [r["manifest_sha"] for r in manifests["rows"]],
        "n_holdout_worlds": manifests["n_holdout_worlds"],
        "genome_014_lock_sha": _sha_file(GENOME_014_LOCK),
        "acquire_014_lock_sha": _sha_file(ACQUIRE_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "genome_013_lock_sha": _sha_file(GENOME_013_LOCK),
        "genome_011_lock_sha": _sha_file(GENOME_011_LOCK),
        "global_measures": list(GLOBAL_MEASURES),
        "branch_measures": list(BRANCH_MEASURES),
        "refuse": [
            "behavioral contact with E-H before canonical run",
            "invent lives in scorer after peek",
            "apparatus ctx planting",
            "teacher supplying Y/kappa/path",
            "mid-life stash to remake lived kappa",
            "LOOKAHEAD",
            "pre-name Ex0S product version",
            "claim full skeleton acquisition from life",
            "claim organism chose open-ended experiences",
            "rewrite genome_014 / acquire_014 mid-holdout after E-H peek",
        ],
    }


def write_family_lock(path: Path = FAMILY_LOCK) -> dict[str, Any]:
    snap = family_lock_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_family_lock(path: Path = FAMILY_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = family_lock_snapshot()
    if not path.exists():
        return False, "docs/family_014.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "seed_jobs_sha",
        "scorer_sha",
        "earn_gate_sha",
        "run_one_sha",
        "holdout_manifest_sha",
        "holdout_row_shas",
        "genome_014_lock_sha",
        "acquire_014_lock_sha",
        "kappa_013_lock_sha",
        "expected_n",
        "ctx_encoding",
        "ex0s_if_earned",
        "preregistered_claim",
        "refuse",
        "seed",
        "per_family",
        "births",
    ):
        if snap.get(key) != lock.get(key):
            return False, f"family lock drift: {key}", snap
    for f in FAMILIES:
        if snap["generator_sha"].get(f) != (lock.get("generator_sha") or {}).get(f):
            return False, f"generator drift {f}", snap
    for f in HOLDOUT:
        if snap["holdout_per_family_sha"].get(f) != (lock.get("holdout_per_family_sha") or {}).get(f):
            return False, f"holdout manifest drift {f}", snap
    if lock.get("ex0s_if_earned") is not None:
        return False, "ex0s_if_earned must be null", snap
    if lock.get("earned_next") is not False:
        return False, "family lock earned_next must stay false until stamp", snap
    if lock.get("preregistered_claim") != PREREGISTERED_CLAIM:
        return False, "preregistered_claim drift", snap
    return True, "family_014.lock intact", snap


def verify_holdout_sealed(path: Path = FAMILY_LOCK) -> tuple[bool, str, dict[str, Any]]:
    """E–H: schema/manifest/oracle/no-ctx — no organism answers."""
    ok, why, snap = verify_family_lock(path)
    if not ok:
        return False, why, snap
    for fam in HOLDOUT:
        w = generate_world(fam, DEFAULT_SEED + 1000 * FAMILIES.index(fam), 0)
        if not w.holdout:
            return False, f"{fam} not holdout", snap
        if w.kappa_a == w.kappa_b:
            return False, f"{fam} kappa collision", snap
        lint_world_no_ctx(w)
        if any("ctx" in e for e in w.birth_edges):
            return False, f"{fam} birth ctx", snap
        w2 = World.from_manifest(w.to_manifest())
        if world_manifest_sha(w) != world_manifest_sha(w2):
            return False, f"{fam} round-trip", snap
        if not w.primary_life:
            return False, f"{fam} empty primary_life", snap
        if fam == "E" and w.depth < 3:
            return False, f"{fam} depth too shallow", snap
        if len(w.hops_a) != w.depth or len(w.hops_b) != w.depth:
            return False, f"{fam} hop depth mismatch", snap
    if len(committed_holdout_row_shas(path)) != 144:
        return False, "holdout row count", snap
    snap["holdout_sealed_ok"] = True
    return True, "holdout E–H sealed (no organism)", snap


def run_family(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
    workers: int = 4,
    families: Sequence[str] | None = None,
    allow_holdout_behavior: bool = False,
) -> dict[str, Any]:
    fams = list(families) if families is not None else list(FAMILIES)
    if allow_holdout_behavior and any(f in HOLDOUT for f in fams):
        if not is_full_canonical(seed, per_family, births, fams):
            raise ValueError("refuse holdout behavior outside full canonical dims")
    jobs_spec = seed_jobs(seed, per_family, births, fams)
    g_ok, g_why, _ = verify_genome_014()
    fam_ok, fam_why, _ = verify_family_lock()
    with tempfile.TemporaryDirectory(prefix="tm014family_") as tmp:
        root = Path(tmp)
        jobs = []
        for fam, s, b in jobs_spec:
            jobs.append(
                {
                    "family": fam,
                    "seed": s,
                    "birth": b,
                    "dest": str(root / f"{fam}_{s}_{b}"),
                    "allow_holdout_behavior": allow_holdout_behavior,
                    "genome_ok": g_ok,
                }
            )
        rows: list[dict[str, Any]] = []
        if workers <= 1:
            for j in jobs:
                rows.append(run_one(j))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_one, j) for j in jobs]
                for fut in as_completed(futs):
                    rows.append(fut.result())

    # Aggregate
    by_fam: dict[str, dict[str, Any]] = {f: {"solved": 0, "n": 0} for f in fams}
    unexpected = 0
    birth_ctx_total = 0
    n_errors = 0
    branch_cov: dict[str, dict[str, bool]] = {f: {"donor": False, "counterfactual": False} for f in fams}
    for r in rows:
        f = r["family"]
        by_fam[f]["n"] += 1
        if r.get("solved"):
            by_fam[f]["solved"] += 1
        led = r.get("ledger") or {}
        unexpected += int(led.get("unexpected_ctx_writes") or 0)
        birth_ctx_total += int(led.get("birth_ctx") or 0)
        n_errors += len(r.get("errors") or [])
        m = r.get("measures") or {}
        if m.get("donor_transfer") is True:
            branch_cov[f]["donor"] = True
        if m.get("counterfactual_life") is True:
            branch_cov[f]["counterfactual"] = True

    holdout_solved = sum(by_fam[f]["solved"] for f in fams if f in HOLDOUT)
    n_worlds = len(rows)
    unique_jobs = len({(r["family"], r["seed"], r["birth"]) for r in rows})
    solved_n = sum(1 for r in rows if r.get("solved"))

    # Manifest pin: scored holdout rows must match sealed lock (not only live regen).
    holdout_manifest_ok = True
    holdout_rows_match_lock = True
    if allow_holdout_behavior and is_full_canonical(seed, per_family, births, fams):
        live = holdout_manifests(seed, per_family, births)
        lock_rows = committed_holdout_row_shas()
        scored = {
            r["manifest_sha"]
            for r in rows
            if r.get("holdout") and not r.get("sealed")
        }
        holdout_rows_match_lock = (
            scored == lock_rows
            and len(scored) == 144
            and None not in scored
        )
        if FAMILY_LOCK.exists():
            lock = json.loads(FAMILY_LOCK.read_text(encoding="utf-8"))
            holdout_manifest_ok = live["holdout_manifest_sha"] == lock.get(
                "holdout_manifest_sha"
            )
        else:
            holdout_manifest_ok = False

    summary: dict[str, Any] = {
        "version": "TM.0.14.FAMILY",
        "n_worlds": n_worlds,
        "unique_jobs": unique_jobs,
        "solved_n": solved_n,
        "solved_frac": solved_n / max(n_worlds, 1),
        "families": by_fam,
        "holdout_solved": holdout_solved,
        "genome_014_ok": g_ok,
        "genome_014_why": g_why,
        "family_lock_ok": fam_ok,
        "family_lock_why": fam_why,
        "unexpected_ctx_writes": unexpected,
        "birth_ctx_total": birth_ctx_total,
        "n_errors": n_errors,
        "holdout_manifest_ok": holdout_manifest_ok,
        "holdout_rows_match_lock": holdout_rows_match_lock,
        "branch_coverage": branch_cov,
        "allow_holdout_behavior": allow_holdout_behavior,
        "claim": PREREGISTERED_CLAIM,
        "rows": rows,
    }
    full = is_full_canonical(seed, per_family, births, fams) and allow_holdout_behavior
    gate = earn_gate(summary, full_canonical=full)
    summary["earned_next"] = bool(gate["earned_next"]) if full else False
    summary["ex0s"] = None
    summary["earn_why"] = gate["why"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--per-family", type=int, default=DEFAULT_PER_FAMILY)
    ap.add_argument("--births", type=int, default=DEFAULT_BIRTHS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--develop-only", action="store_true")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--verify-sealed", action="store_true")
    args = ap.parse_args()

    if args.write_lock:
        snap = write_family_lock()
        print(json.dumps({"wrote": str(FAMILY_LOCK), "n_holdout": snap["n_holdout_worlds"]}, indent=2))
        return
    if args.verify_sealed:
        ok, why, snap = verify_holdout_sealed()
        print(json.dumps({"ok": ok, "why": why, "holdout_sealed_ok": snap.get("holdout_sealed_ok")}, indent=2))
        sys.exit(0 if ok else 1)

    families = DEVELOP if args.develop_only or not args.canonical else FAMILIES
    allow = bool(args.canonical)
    summary = run_family(
        seed=args.seed,
        per_family=args.per_family,
        births=args.births,
        workers=args.workers,
        families=families,
        allow_holdout_behavior=allow,
    )
    out = _run_dir()
    (out / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "rows.json").write_text(json.dumps(summary["rows"], indent=2) + "\n", encoding="utf-8")
    pub = {k: summary[k] for k in summary if k != "rows"}
    print(json.dumps(pub, indent=2))
    for f in families:
        st = summary["families"][f]
        print(f"  {f}: {st['solved']}/{st['n']}")
    if summary["solved_frac"] < 1.0:
        sys.exit(1)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm014family"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
