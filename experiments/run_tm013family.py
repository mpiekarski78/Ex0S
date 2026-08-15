"""TM.0.13.FAMILY: preregistered attack on frozen CONTEXT-on candidate.

A–D: development (behavioral OK). E–H: cryptographically committed, sealed
until the canonical 288. No new mechanism. Stamp Ex0S 0.0.004 only if earned.

Planted S wording only — not experience-acquired contextual memory.
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm013context import (  # noqa: E402
    CONTEXT_LOCK,
    GENOME_011_LOCK,
    GENOME_013_LOCK,
    KAPPA_LOCK,
    reference_edge_sem,
    reference_kappa_seed,
    reference_kappa_step,
    reference_route_kappa,
    verify_genome_013,
    verify_kappa_vectors,
)
from experiments.run_tm040 import probe
from three_memory.kappa import CTX_ENCODING
from three_memory.policy import UsePolicy
from three_memory.symbols import record_to_tagfile

DEVELOP = ("A", "B", "C", "D")
HOLDOUT = ("E", "F", "G", "H")
FAMILIES = DEVELOP + HOLDOUT
MOTORS = ("press", "tune", "flip")
BANNED = frozenset(MOTORS + ("hold", "idle", "push", "adjust", "open", "wait", "use", "pick"))
CONS = "bcdfghjklmnpqrstvwxz"
VOW = "aeiou"
HERE = "chb"

FAMILY_LOCK = REPO_ROOT / "docs" / "family_013.lock"
DEFAULT_SEED = 12345
DEFAULT_PER_FAMILY = 12
DEFAULT_BIRTHS = 3
EXPECTED_N = len(FAMILIES) * DEFAULT_PER_FAMILY * DEFAULT_BIRTHS  # 288

MANDATORY_MEASURES = (
    "context_route",
    "ctx_beats_untagged",
    "ctx_no_fallback",
    "tie_hold",
    "retarget_ctx",
    "revise_evidence",
    "revise_route",
    "s_necessity",
    "rho_reset_same_agent",
    "newborn_reload",
    "storage_identity_order_invariance",
    "feature_off_compat",
    "no_shortcut_writes",
    "weights_stable",
    "genome_delta",
)


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


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_json(obj: Any) -> str:
    return _sha_bytes(_canonical_json(obj).encode())


@dataclass
class Rel:
    fid: str
    bind: str
    did: str
    role: str
    init: tuple[int, int] = (1, 0)
    ctx: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fid": self.fid,
            "bind": self.bind,
            "did": self.did,
            "role": self.role,
            "init": [int(self.init[0]), int(self.init[1])],
            "ctx": self.ctx,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Rel":
        init = d.get("init") or [1, 0]
        return Rel(
            fid=str(d["fid"]),
            bind=str(d["bind"]),
            did=str(d["did"]),
            role=str(d["role"]),
            init=(int(init[0]), int(init[1])),
            ctx=d.get("ctx"),
        )


@dataclass
class World:
    family: str
    holdout: bool
    seed: int
    birth: int
    depth: int
    cue: str
    relations_primary: list[Rel]
    relations_alt: list[Rel]
    hops_primary: list[tuple[str, str]]
    hops_alt: list[tuple[str, str]]
    kappa_primary: str
    kappa_alt: str
    expect_primary: str
    expect_alt: str
    feature_off_expect: str
    role_qa: str = "qa"
    role_qb: str = "qb"
    clutter: list[Rel] = field(default_factory=list)
    annotation: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "holdout": self.holdout,
            "seed": self.seed,
            "birth": self.birth,
            "depth": self.depth,
            "cue": self.cue,
            "relations_primary": [r.to_dict() for r in self.relations_primary],
            "relations_alt": [r.to_dict() for r in self.relations_alt],
            "hops_primary": [list(h) for h in self.hops_primary],
            "hops_alt": [list(h) for h in self.hops_alt],
            "kappa_primary": self.kappa_primary,
            "kappa_alt": self.kappa_alt,
            "expect_primary": self.expect_primary,
            "expect_alt": self.expect_alt,
            "feature_off_expect": self.feature_off_expect,
            "role_qa": self.role_qa,
            "role_qb": self.role_qb,
            "clutter": [r.to_dict() for r in self.clutter],
            "annotation": self.annotation,
        }

    @staticmethod
    def from_manifest(d: dict[str, Any]) -> "World":
        return World(
            family=d["family"],
            holdout=bool(d["holdout"]),
            seed=int(d["seed"]),
            birth=int(d["birth"]),
            depth=int(d["depth"]),
            cue=d["cue"],
            relations_primary=[Rel.from_dict(r) for r in d["relations_primary"]],
            relations_alt=[Rel.from_dict(r) for r in d["relations_alt"]],
            hops_primary=[tuple(h) for h in d["hops_primary"]],
            hops_alt=[tuple(h) for h in d["hops_alt"]],
            kappa_primary=d["kappa_primary"],
            kappa_alt=d["kappa_alt"],
            expect_primary=d["expect_primary"],
            expect_alt=d["expect_alt"],
            feature_off_expect=d["feature_off_expect"],
            role_qa=d.get("role_qa", "qa"),
            role_qb=d.get("role_qb", "qb"),
            clutter=[Rel.from_dict(r) for r in d.get("clutter") or []],
            annotation=d.get("annotation") or "",
        )


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


def lexical_edge_order(rels: Sequence[Rel]) -> list[tuple[str, str]]:
    """Order of (bind,did) as TagStore would load: sorted by fid filename."""
    ordered = sorted(rels, key=lambda r: r.fid)
    return [(r.bind.lower(), r.did.lower()) for r in ordered]


def make_context(s_dir: Path, policy: UsePolicy, *, use_context_kappa: bool = True):
    return make(
        s_dir,
        None,
        policy,
        explore_epsilon=0.0,
        use_context_kappa=use_context_kappa,
    )


def probe_once(
    policy: UsePolicy,
    s_dir: Path | None,
    seed: int,
    cue: str | None,
    *,
    use_context_kappa: bool = True,
    agent=None,
) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    if agent is not None:
        out = probe(agent, "probe_channel_b", seed, tokens=tokens)
        out["cue"] = cue
        pol = out.get("policy") or {}
        out["compose_hops"] = pol.get("compose_hops")
        out["evidence_resolved"] = bool(pol.get("evidence_resolved"))
        out["evidence_tie"] = bool(pol.get("evidence_tie"))
        out["compose_hold"] = bool(pol.get("compose_hold"))
        out["context_kappa"] = pol.get("context_kappa")
        out["weight_hash"] = agent.weight_hash()
        return out
    with tempfile.TemporaryDirectory(prefix="tm013fam_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make_context(store, policy, use_context_kappa=use_context_kappa)
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


def seed_jobs(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
    families: Sequence[str] = FAMILIES,
) -> list[tuple[str, int, int]]:
    jobs: list[tuple[str, int, int]] = []
    for fam in FAMILIES:
        if fam not in families:
            continue
        fi = FAMILIES.index(fam)
        for w in range(per_family):
            for b in range(births):
                jobs.append((fam, seed + 1000 * fi + w, b))
    return jobs


def is_full_canonical(
    *,
    seed: int,
    per_family: int,
    births: int,
    families: Sequence[str],
    allow_holdout_behavior: bool,
) -> bool:
    return (
        allow_holdout_behavior
        and seed == DEFAULT_SEED
        and per_family == DEFAULT_PER_FAMILY
        and births == DEFAULT_BIRTHS
        and set(families) == set(FAMILIES)
    )


def seed_jobs_sha(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> str:
    jobs = seed_jobs(seed=seed, per_family=per_family, births=births)
    blob = ";".join(f"{f}:{s}:{b}" for f, s, b in jobs)
    return _sha_bytes(blob.encode())


def _base_rng(family: str, seed: int, birth: int) -> np.random.Generator:
    return np.random.default_rng(seed + 19 * (ord(family) + 1) + 1103 * birth)


# --- Graph builders ------------------------------------------------------------


def _diamond_tokens(rng: np.random.Generator, *, fixed: bool) -> dict[str, str]:
    if fixed:
        return {"x": "x", "q": "q", "a": "a", "b": "b", "y": "y"}
    taken: set[str] = set()
    return {k: _nonce(rng, taken) for k in ("x", "q", "a", "b", "y")}


def _fids(rng: np.random.Generator, n: int) -> list[str]:
    taken: set[str] = set()
    return [_fid(rng, taken) for _ in range(n)]


def build_order_world(
    *,
    family: str,
    holdout: bool,
    seed: int,
    birth: int,
    depth_extra: int = 0,
    n_clutter: int = 0,
    use_nonces: bool = False,
    cycle_revisit: bool = False,
    annotation: str = "",
) -> World:
    """Planted A-then-B vs B-then-A contextual route split at shared Y."""
    rng = _base_rng(family, seed, birth)
    tok = _diamond_tokens(rng, fixed=not use_nonces)
    x, q, a, b, y = tok["x"], tok["q"], tok["a"], tok["b"], tok["y"]
    m1, m2 = "press", "tune"

    mid: list[str] = []
    for i in range(depth_extra):
        mid.append(_nonce(rng, set(tok.values()) | set(mid) | set(MOTORS)))

    # Primary hops: x→q→a→q→b→q→(mids)→y
    hops_ab: list[tuple[str, str]] = [(x, q), (q, a), (a, q), (q, b), (b, q)]
    hops_ba: list[tuple[str, str]] = [(x, q), (q, b), (b, q), (q, a), (a, q)]
    chain_tail: list[tuple[str, str]] = []
    prev = q
    for node in mid:
        chain_tail.append((prev, node))
        prev = node
    chain_tail.append((prev, y))
    hops_ab = hops_ab + chain_tail
    hops_ba = hops_ba + chain_tail

    if cycle_revisit:
        # Extra loop q→z→q then continue; both routes share it after branch merge.
        z = _nonce(rng, set(tok.values()) | set(mid) | {y} | set(MOTORS))
        # Insert before final approach: after last q return via z.
        # Simpler: x→q→a→q→b→q→z→q→y with z on both after merge.
        hops_ab = [(x, q), (q, a), (a, q), (q, b), (b, q), (q, z), (z, q), (q, y)]
        hops_ba = [(x, q), (q, b), (b, q), (q, a), (a, q), (q, z), (z, q), (q, y)]
        mid = [z]

    kap_ab = reference_route_kappa(x, hops_ab)
    kap_ba = reference_route_kappa(x, hops_ba)
    assert kap_ab != kap_ba

    fids = _fids(rng, 20 + n_clutter)

    def graph(qa_s: int, qb_s: int) -> list[Rel]:
        rels = [
            Rel(fids[0], x, q, "xq", (1, 0)),
            Rel(fids[1], q, a, "qa", (qa_s, 0)),
            Rel(fids[2], a, q, "aq", (1, 0)),
            Rel(fids[3], q, b, "qb", (qb_s, 0)),
            Rel(fids[4], b, q, "bq", (1, 0)),
        ]
        idx = 5
        if cycle_revisit:
            z = mid[0]
            # Evidence ladder at q: first-hop winner > second > qz > qy.
            # Primary uses qa_s > qb_s > 3 > 1 so path is a then b then z then y.
            qz_s = min(qa_s, qb_s) - 1
            if qz_s < 2:
                qz_s = 2
            rels += [
                Rel(fids[idx], q, z, "qz", (qz_s, 0)),
                Rel(fids[idx + 1], z, q, "zq", (1, 0)),
                Rel(fids[idx + 2], q, y, "qy", (1, 0)),
            ]
            idx += 3
        else:
            prev_n = q
            for node in mid:
                rels.append(Rel(fids[idx], prev_n, node, f"m{idx}", (1, 0)))
                prev_n = node
                idx += 1
            rels.append(Rel(fids[idx], prev_n, y, "qy", (1, 0)))
            idx += 1
        rels += [
            Rel(fids[idx], y, m1, "yp", (1, 0), ctx=kap_ab),
            Rel(fids[idx + 1], y, m2, "yt", (1, 0), ctx=kap_ba),
        ]
        return rels

    primary = graph(5, 4) if cycle_revisit else graph(3, 2)
    alt = graph(4, 5) if cycle_revisit else graph(2, 3)
    clutter: list[Rel] = []
    taken = {r.fid for r in primary}
    for i in range(n_clutter):
        fb = _fid(rng, taken)
        taken.add(fb)
        junk = _nonce(rng, set(tok.values()) | set(mid) | {y} | set(MOTORS))
        clutter.append(Rel(fb, junk, junk + "z" if len(junk) < 6 else "zz", f"cl{i}", (1, 0)))

    depth = len(hops_ab)
    return World(
        family=family,
        holdout=holdout,
        seed=seed,
        birth=birth,
        depth=depth,
        cue=x,
        relations_primary=primary,
        relations_alt=alt,
        hops_primary=hops_ab,
        hops_alt=hops_ba,
        kappa_primary=kap_ab,
        kappa_alt=kap_ba,
        expect_primary=m1,
        expect_alt=m2,
        # κ off: both motors equal support at Y → HOLD (exact 0.0.003 evidence tie)
        feature_off_expect="hold",
        clutter=clutter,
        annotation=annotation,
    )


def generate_family_A(seed: int, birth: int) -> World:
    return build_order_world(
        family="A", holdout=False, seed=seed, birth=birth, depth_extra=0, annotation="simple route-order"
    )


def build_converge_world(*, family: str, seed: int, birth: int, annotation: str) -> World:
    """Shared mid token after branch: x→q→a→m→y vs x→q→b→m→y."""
    rng = _base_rng(family, seed, birth)
    tok = _diamond_tokens(rng, fixed=True)
    x, q, a, b, y = tok["x"], tok["q"], tok["a"], tok["b"], tok["y"]
    m = "m"
    m1, m2 = "press", "tune"
    hops_ab = [(x, q), (q, a), (a, m), (m, y)]
    hops_ba = [(x, q), (q, b), (b, m), (m, y)]
    kap_ab = reference_route_kappa(x, hops_ab)
    kap_ba = reference_route_kappa(x, hops_ba)
    fids = _fids(rng, 12)

    def graph(qa_s: int, qb_s: int) -> list[Rel]:
        return [
            Rel(fids[0], x, q, "xq", (1, 0)),
            Rel(fids[1], q, a, "qa", (qa_s, 0)),
            Rel(fids[2], a, m, "am", (1, 0)),
            Rel(fids[3], q, b, "qb", (qb_s, 0)),
            Rel(fids[4], b, m, "bm", (1, 0)),
            Rel(fids[5], m, y, "my", (1, 0)),
            Rel(fids[6], y, m1, "yp", (1, 0), ctx=kap_ab),
            Rel(fids[7], y, m2, "yt", (1, 0), ctx=kap_ba),
        ]

    return World(
        family=family,
        holdout=False,
        seed=seed,
        birth=birth,
        depth=len(hops_ab),
        cue=x,
        relations_primary=graph(3, 1),
        relations_alt=graph(1, 3),
        hops_primary=hops_ab,
        hops_alt=hops_ba,
        kappa_primary=kap_ab,
        kappa_alt=kap_ba,
        expect_primary=m1,
        expect_alt=m2,
        feature_off_expect="hold",
        annotation=annotation,
    )


def generate_family_B(seed: int, birth: int) -> World:
    return build_converge_world(
        family="B",
        seed=seed,
        birth=birth,
        annotation="convergence via shared mid before Y",
    )


def generate_family_C(seed: int, birth: int) -> World:
    """Diamond plus planted untagged high-support motor: feature-off exact = flip."""
    w = build_order_world(
        family="C",
        holdout=False,
        seed=seed,
        birth=birth,
        depth_extra=0,
        annotation="evidence/revision; planted untagged trap in primary S",
    )
    y = w.hops_primary[-1][1]
    trap = Rel(f"c{seed % 10000:04d}t{birth}", y, "flip", "trap0", (1000, 0), ctx=None)
    w.relations_primary = list(w.relations_primary) + [trap]
    w.relations_alt = list(w.relations_alt) + [
        Rel(f"c{seed % 10000:04d}u{birth}", y, "flip", "trap0", (1000, 0), ctx=None)
    ]
    w.feature_off_expect = "flip"
    return w


def generate_family_D(seed: int, birth: int) -> World:
    return build_order_world(
        family="D",
        holdout=False,
        seed=seed,
        birth=birth,
        depth_extra=0,
        n_clutter=40,
        annotation="clutter + storage identity",
    )


def generate_family_E(seed: int, birth: int) -> World:
    # Holdout depths 7–8 (depth_extra 1 → 7 hops; 2 → 8). Developed max is 6.
    extra = 1 + (seed + birth) % 2
    return build_order_world(
        family="E",
        holdout=True,
        seed=seed,
        birth=birth,
        depth_extra=extra,
        annotation="unseen greater depth",
    )


def generate_family_F(seed: int, birth: int) -> World:
    return build_order_world(
        family="F",
        holdout=True,
        seed=seed,
        birth=birth,
        cycle_revisit=True,
        annotation="cycle revisit + converge",
    )


def generate_family_G(seed: int, birth: int) -> World:
    return build_order_world(
        family="G",
        holdout=True,
        seed=seed,
        birth=birth,
        depth_extra=1 + (birth % 2),
        use_nonces=True,
        annotation="unseen nonce vocabulary",
    )


def generate_family_H(seed: int, birth: int) -> World:
    return build_order_world(
        family="H",
        holdout=True,
        seed=seed,
        birth=birth,
        depth_extra=1 + (seed % 2),
        n_clutter=25,
        use_nonces=True,
        cycle_revisit=bool(birth % 2),
        annotation="mixed adversarial",
    )


FAMILY_GENERATORS: dict[str, Callable[[int, int], World]] = {
    "A": generate_family_A,
    "B": generate_family_B,
    "C": generate_family_C,
    "D": generate_family_D,
    "E": generate_family_E,
    "F": generate_family_F,
    "G": generate_family_G,
    "H": generate_family_H,
}


def generate_world(family: str, seed: int, birth: int) -> World:
    return FAMILY_GENERATORS[family](seed, birth)


def world_manifest_sha(world: World) -> str:
    return _sha_json(world.to_manifest())


def committed_holdout_row_shas(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> set[str]:
    """SHA of each committed E–H world manifest (no organism)."""
    out: set[str] = set()
    for fam, s, b in seed_jobs(seed=seed, per_family=per_family, births=births, families=HOLDOUT):
        out.add(world_manifest_sha(generate_world(fam, s, b)))
    return out


def holdout_manifests(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> dict[str, Any]:
    """Generate E–H world specs without running the organism."""
    by_fam: dict[str, list[dict[str, Any]]] = {f: [] for f in HOLDOUT}
    for fam, s, b in seed_jobs(seed=seed, per_family=per_family, births=births, families=HOLDOUT):
        w = generate_world(fam, s, b)
        by_fam[fam].append(w.to_manifest())
    per_family_sha = {f: _sha_json(by_fam[f]) for f in HOLDOUT}
    all_flat = [m for f in HOLDOUT for m in by_fam[f]]
    return {
        "per_family_sha": per_family_sha,
        "holdout_manifest_sha": _sha_json(all_flat),
        "n_holdout_worlds": len(all_flat),
        "families": {f: len(by_fam[f]) for f in HOLDOUT},
    }


def order_changing_fid_remap(rels: Sequence[Rel], seed: int) -> list[Rel]:
    """Bijective fid remap such that lexical (bind,did) order changes."""
    rng = np.random.default_rng(seed + 9091)
    original_order = lexical_edge_order(rels)
    taken = {r.fid for r in rels}
    # Try until sorted-fid edge sequence differs.
    for _ in range(200):
        mapping = {}
        used: set[str] = set()
        for r in rels:
            nf = _fid(rng, used | taken)
            used.add(nf)
            mapping[r.fid] = nf
        remapped = [
            Rel(mapping[r.fid], r.bind, r.did, r.role, r.init, r.ctx) for r in rels
        ]
        if lexical_edge_order(remapped) != original_order:
            return remapped
    raise RuntimeError("could not find order-changing fid remap")


def _with_supports(rels: list[Rel], role: str, support: int) -> list[Rel]:
    out = []
    for r in rels:
        if r.role == role:
            out.append(Rel(r.fid, r.bind, r.did, r.role, (support, 0), r.ctx))
        else:
            out.append(r)
    return out


def _swap_ctx_motors(rels: list[Rel]) -> list[Rel]:
    yp = next(r for r in rels if r.role == "yp")
    yt = next(r for r in rels if r.role == "yt")
    out = []
    for r in rels:
        if r.role == "yp":
            out.append(Rel(r.fid, r.bind, r.did, r.role, r.init, yt.ctx))
        elif r.role == "yt":
            out.append(Rel(r.fid, r.bind, r.did, r.role, r.init, yp.ctx))
        else:
            out.append(r)
    return out


def _same_kappa_tie(rels: list[Rel], kappa: str) -> list[Rel]:
    out = []
    for r in rels:
        if r.role in ("yp", "yt"):
            out.append(Rel(r.fid, r.bind, r.did, r.role, (1, 0), kappa))
        else:
            out.append(r)
    return out


def _add_untagged_trap(rels: list[Rel], fid: str, y: str, motor: str, support: int) -> list[Rel]:
    return list(rels) + [Rel(fid, y, motor, "trap", (support, 0), ctx=None)]


def _wrong_kappa_nofallback(rels: list[Rel], wrong_kappa: str, alt_kappa: str, fid: str, y: str) -> list[Rel]:
    out = []
    for r in rels:
        if r.role == "yp":
            out.append(Rel(r.fid, r.bind, r.did, r.role, r.init, wrong_kappa))
        elif r.role == "yt":
            out.append(Rel(r.fid, r.bind, r.did, r.role, r.init, alt_kappa))
        else:
            out.append(r)
    out.append(Rel(fid, y, "flip", "untagged", (1000, 0), ctx=None))
    return out


# --- Scoring -------------------------------------------------------------------


def score_world(
    world: World,
    *,
    measures: dict[str, bool | None],
    genome_ok: bool,
    errors: list[str],
) -> dict[str, Any]:
    measures = dict(measures)
    measures["genome_delta"] = genome_ok
    missing = [m for m in MANDATORY_MEASURES if measures.get(m) is None]
    failed = [m for m in MANDATORY_MEASURES if measures.get(m) is False]
    solved = (not missing) and (not failed) and (not errors) and all(
        measures.get(m) is True for m in MANDATORY_MEASURES
    )
    return {
        "family": world.family,
        "holdout": world.holdout,
        "seed": world.seed,
        "birth": world.birth,
        "depth": world.depth,
        "measures": measures,
        "missing_measures": missing,
        "failed_measures": failed,
        "errors": errors,
        "solved": solved,
        "manifest_sha": world_manifest_sha(world),
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    family = job["family"]
    seed = int(job["seed"])
    birth = int(job["birth"])
    dest = Path(job["dest"])
    genome_ok = bool(job["genome_ok"])
    allow_holdout = bool(job.get("allow_holdout_behavior", False))
    dest.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    measures: dict[str, bool | None] = {m: None for m in MANDATORY_MEASURES}

    world = generate_world(family, seed, birth)
    if world.holdout and not allow_holdout:
        return {
            "family": family,
            "holdout": True,
            "seed": seed,
            "birth": birth,
            "depth": world.depth,
            "measures": measures,
            "missing_measures": list(MANDATORY_MEASURES),
            "failed_measures": [],
            "errors": ["holdout_behavior_sealed"],
            "solved": False,
            "manifest_sha": world_manifest_sha(world),
            "sealed": True,
        }

    policy = UsePolicy(seed=7, lr=0.2)
    y = world.hops_primary[-1][1]
    shortcut_ok = True
    weights_ok = True
    kappa_ok = True

    def use_probe(s_dir: Path, cue: str, pseed: int, *, use_ctx: bool = True) -> dict[str, Any]:
        nonlocal shortcut_ok, weights_ok
        before = s_content_hash(s_dir)
        w_before = None
        ag = make_context(s_dir, policy, use_context_kappa=use_ctx)
        w_before = ag.weight_hash()
        ag.reset_rho()
        out = probe_once(policy, s_dir, pseed, cue, use_context_kappa=use_ctx, agent=ag)
        after = s_content_hash(s_dir)
        if before != after:
            shortcut_ok = False
        if out.get("weight_hash") != w_before:
            weights_ok = False
        return out

    try:
        # --- primary / alt planted histories ---
        s_pri = dest / "s_primary"
        s_alt = dest / "s_alt"
        write_s(s_pri, world.relations_primary + world.clutter)
        write_s(s_alt, world.relations_alt + world.clutter)

        out_p = use_probe(s_pri, world.cue, seed + 1)
        out_a = use_probe(s_alt, world.cue, seed + 2)
        if out_p.get("context_kappa") != world.kappa_primary:
            kappa_ok = False
        if out_a.get("context_kappa") != world.kappa_alt:
            kappa_ok = False
        measures["context_route"] = (
            _motor(out_p["action_name"]) == world.expect_primary
            and _motor(out_a["action_name"]) == world.expect_alt
            and kappa_ok
        )

        # --- ctx beats untagged ---
        s_trap = dest / "s_trap"
        trap_fid = f"t{seed % 10000:04d}{birth}"
        write_s(
            s_trap,
            _add_untagged_trap(world.relations_primary, trap_fid, y, world.expect_alt, 1000)
            + world.clutter,
        )
        out_trap = use_probe(s_trap, world.cue, seed + 3)
        measures["ctx_beats_untagged"] = _motor(out_trap["action_name"]) == world.expect_primary

        # --- ctx no fallback ---
        s_nf = dest / "s_nofallback"
        wrong = reference_route_kappa(world.cue, [(world.cue, "zz"), ("zz", y)])
        write_s(
            s_nf,
            _wrong_kappa_nofallback(
                world.relations_primary, wrong, world.kappa_alt, f"u{seed % 10000:04d}", y
            )
            + world.clutter,
        )
        out_nf = use_probe(s_nf, world.cue, seed + 4)
        measures["ctx_no_fallback"] = (
            _motor(out_nf["action_name"]) == "hold"
            and out_nf.get("context_kappa") == world.kappa_primary
        )

        # --- tie hold ---
        s_tie = dest / "s_tie"
        write_s(s_tie, _same_kappa_tie(world.relations_primary, world.kappa_primary) + world.clutter)
        out_tie = use_probe(s_tie, world.cue, seed + 5)
        measures["tie_hold"] = _motor(out_tie["action_name"]) == "hold"

        # --- retarget ctx ---
        s_rt = dest / "s_retarget"
        write_s(s_rt, _swap_ctx_motors(world.relations_primary) + world.clutter)
        out_rt = use_probe(s_rt, world.cue, seed + 6)
        measures["retarget_ctx"] = _motor(out_rt["action_name"]) == world.expect_alt

        # --- revise evidence (boost primary motor support only; topology fixed) ---
        s_re = dest / "s_revise_ev"
        # Start from equal same-κ tie then boost press
        base = _same_kappa_tie(world.relations_primary, world.kappa_primary)
        boosted = []
        for r in base:
            if r.role == "yp":
                boosted.append(Rel(r.fid, r.bind, r.did, r.role, (5, 0), r.ctx))
            else:
                boosted.append(r)
        write_s(s_re, boosted + world.clutter)
        out_re = use_probe(s_re, world.cue, seed + 7)
        measures["revise_evidence"] = _motor(out_re["action_name"]) == world.expect_primary

        # --- revise route ---
        s_rr = dest / "s_revise_route"
        write_s(s_rr, world.relations_alt + world.clutter)
        out_rr = use_probe(s_rr, world.cue, seed + 8)
        measures["revise_route"] = (
            _motor(out_rr["action_name"]) == world.expect_alt
            and out_rr.get("context_kappa") == world.kappa_alt
        )

        # --- wipe ---
        s_wipe = dest / "s_wipe"
        s_wipe.mkdir(parents=True, exist_ok=True)
        before_w = s_content_hash(s_wipe)
        out_w = probe_once(policy, s_wipe, seed + 9, world.cue)
        if s_content_hash(s_wipe) != before_w:
            shortcut_ok = False
        measures["s_necessity"] = _motor(out_w["action_name"]) == "hold"

        # --- rho_reset_same_agent ---
        s_rho = dest / "s_rho"
        write_s(s_rho, world.relations_primary + world.clutter)
        ag = make_context(s_rho, policy)
        w0 = ag.weight_hash()
        before_rho = s_content_hash(s_rho)
        ag.reset_rho()
        o1 = probe_once(policy, s_rho, seed + 10, world.cue, agent=ag)
        ag.reset_rho()
        o2 = probe_once(policy, s_rho, seed + 11, world.cue, agent=ag)
        if s_content_hash(s_rho) != before_rho:
            shortcut_ok = False
        measures["rho_reset_same_agent"] = (
            _motor(o1["action_name"]) == _motor(o2["action_name"]) == world.expect_primary
            and o1.get("context_kappa") == o2.get("context_kappa") == world.kappa_primary
            and ag.weight_hash() == w0
        )

        # --- newborn_reload ---
        s_nb_src = dest / "s_newborn_src"
        s_nb = dest / "s_newborn"
        write_s(s_nb_src, world.relations_primary + world.clutter)
        if s_nb.exists():
            shutil.rmtree(s_nb)
        shutil.copytree(s_nb_src, s_nb)
        out_nb = use_probe(s_nb, world.cue, seed + 12)
        measures["newborn_reload"] = (
            _motor(out_nb["action_name"]) == world.expect_primary
            and out_nb.get("context_kappa") == world.kappa_primary
        )

        # --- storage identity order invariance ---
        remapped = order_changing_fid_remap(world.relations_primary + world.clutter, seed)
        assert lexical_edge_order(remapped) != lexical_edge_order(
            world.relations_primary + world.clutter
        )
        s_ord = dest / "s_order"
        write_s(s_ord, remapped)
        out_ord = use_probe(s_ord, world.cue, seed + 13)
        measures["storage_identity_order_invariance"] = (
            _motor(out_ord["action_name"]) == world.expect_primary
            and out_ord.get("context_kappa") == world.kappa_primary
        )

        # --- feature off exact 0.0.003 expect ---
        out_off = use_probe(s_pri, world.cue, seed + 14, use_ctx=False)
        measures["feature_off_compat"] = _motor(out_off["action_name"]) == world.feature_off_expect

        measures["no_shortcut_writes"] = shortcut_ok
        measures["weights_stable"] = weights_ok

    except Exception as exc:  # noqa: BLE001 — fail closed into score
        errors.append(f"{type(exc).__name__}: {exc}")

    return score_world(world, measures=measures, genome_ok=genome_ok, errors=errors)


def earn_gate(
    rows: list[dict[str, Any]],
    *,
    genome_ok_start: bool,
    genome_ok_end: bool,
    freeze_why: str,
    seed: int,
    per_family: int,
    births: int,
    holdout_manifest_ok: bool,
) -> dict[str, Any]:
    """Paranoid stamp gate — omitted measures cannot silently succeed."""
    n = len(rows)
    by_fam: dict[str, list[dict[str, Any]]] = {f: [] for f in FAMILIES}
    for r in rows:
        by_fam[r["family"]].append(r)

    jobs = seed_jobs(seed=seed, per_family=per_family, births=births)
    job_set = {(f, s, b) for f, s, b in jobs}
    row_set = {(r["family"], r["seed"], r["birth"]) for r in rows}
    unique_ok = len(row_set) == n == EXPECTED_N and row_set == job_set
    per_fam_ok = all(len(by_fam[f]) == DEFAULT_PER_FAMILY * DEFAULT_BIRTHS for f in FAMILIES)
    jobs_sha_ok = seed_jobs_sha(seed=seed, per_family=per_family, births=births) == seed_jobs_sha()

    holdout = [r for r in rows if r["holdout"]]
    committed = committed_holdout_row_shas(seed=seed, per_family=per_family, births=births)
    scored_holdout_shas = {r.get("manifest_sha") for r in holdout}
    holdout_rows_match_lock = (
        len(holdout) == DEFAULT_PER_FAMILY * DEFAULT_BIRTHS * len(HOLDOUT)
        and scored_holdout_shas == committed
        and None not in scored_holdout_shas
    )

    def fam_frac(fam: str) -> float:
        chunk = by_fam[fam]
        return (sum(1 for r in chunk if r.get("solved")) / len(chunk)) if chunk else 0.0

    developed = [r for r in rows if not r["holdout"]]
    all_frac = (sum(1 for r in rows if r.get("solved")) / n) if n else 0.0
    developed_frac = (
        sum(1 for r in developed if r.get("solved")) / len(developed) if developed else 0.0
    )
    holdout_frac = sum(1 for r in holdout if r.get("solved")) / len(holdout) if holdout else 0.0

    measures_complete = all(
        all(r["measures"].get(m) is not None for m in MANDATORY_MEASURES)
        and not r.get("errors")
        for r in rows
    )

    full_battery = (
        seed == DEFAULT_SEED
        and per_family == DEFAULT_PER_FAMILY
        and births == DEFAULT_BIRTHS
        and unique_ok
        and per_fam_ok
        and jobs_sha_ok
        and holdout_manifest_ok
        and holdout_rows_match_lock
        and measures_complete
        and genome_ok_start
        and genome_ok_end
    )

    families = {
        f: {
            "holdout": f in HOLDOUT,
            "n": len(by_fam[f]),
            "solved": sum(1 for r in by_fam[f] if r.get("solved")),
            "solved_frac": fam_frac(f),
            "depth": by_fam[f][0]["depth"] if by_fam[f] else None,
        }
        for f in FAMILIES
    }

    earned = (
        full_battery
        and all_frac == 1.0
        and developed_frac == 1.0
        and holdout_frac == 1.0
        and all(families[f]["solved_frac"] == 1.0 for f in FAMILIES)
    )

    return {
        "version": "TM.0.13.FAMILY",
        "ex0s_under_test": "0.0.003",
        "ex0s": "0.0.004" if earned else None,
        "earned_next": earned,
        "claim_name": "Contextual Composition" if earned else None,
        "n_worlds": n,
        "solved": sum(1 for r in rows if r.get("solved")),
        "solved_frac": all_frac,
        "developed_solved_frac": developed_frac,
        "holdout_solved_frac": holdout_frac,
        "families": families,
        "full_battery": full_battery,
        "unique_jobs_ok": unique_ok,
        "per_family_counts_ok": per_fam_ok,
        "seed_jobs_sha_ok": jobs_sha_ok,
        "holdout_manifest_ok": holdout_manifest_ok,
        "holdout_rows_match_lock": holdout_rows_match_lock,
        "measures_complete": measures_complete,
        "genome_ok_start": genome_ok_start,
        "genome_ok_end": genome_ok_end,
        "freeze_why": freeze_why,
        "seed": seed,
        "per_family": per_family,
        "births": births,
        "mandatory_measures": list(MANDATORY_MEASURES),
        "note": (
            "Ex0S 0.0.004 Contextual Composition earned."
            if earned
            else "Not stamped. Planted S FAMILY; ACQUIRE is later."
        ),
    }


# --- Lock / verify -------------------------------------------------------------


def family_lock_snapshot() -> dict[str, Any]:
    manifests = holdout_manifests()
    return {
        "version": "TM.0.13.FAMILY",
        "ex0s_under_test": "0.0.003",
        "earned_next": False,
        "preregistered_claim": (
            "A frozen CONTEXT recipe carries bounded provenance-sensitive state "
            "through externally acquired relation graphs and uses that state to "
            "distinguish otherwise identical frontiers across unseen generated "
            "world families, while acquired continuations remain in S and "
            "cognitive weights remain unchanged."
        ),
        "claim_name_if_earned": "Contextual Composition",
        "ex0s_if_earned": "0.0.004",
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
        "n_holdout_worlds": manifests["n_holdout_worlds"],
        "genome_013_lock_sha": _sha_file(GENOME_013_LOCK),
        "kappa_013_lock_sha": _sha_file(KAPPA_LOCK),
        "context_013_lock_sha": _sha_file(CONTEXT_LOCK),
        "genome_011_lock_sha": _sha_file(GENOME_011_LOCK),
        "reference_route_kappa_sha": _sha_src(reference_route_kappa),
        "reference_kappa_seed_sha": _sha_src(reference_kappa_seed),
        "reference_kappa_step_sha": _sha_src(reference_kappa_step),
        "reference_edge_sem_sha": _sha_src(reference_edge_sem),
        "mandatory_measures": list(MANDATORY_MEASURES),
        "refuse": [
            "behavioral contact with E-H before canonical run",
            "rewrite genome_011.lock",
            "rewrite genome_013.lock mid-holdout after E-H peek",
            "stamp Ex0S 0.0.004 on smoke or partial battery",
            "plant ctx only from live kappa without reference",
            "LOOKAHEAD / new mechanism this pass",
            "call stamp Abstraction or reasoning",
            "TM.0.14 ACQUIRE this pass",
        ],
    }


def write_family_lock(path: Path = FAMILY_LOCK) -> dict[str, Any]:
    snap = family_lock_snapshot()
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_family_lock(path: Path = FAMILY_LOCK) -> tuple[bool, str, dict[str, Any]]:
    snap = family_lock_snapshot()
    if not path.exists():
        return False, "docs/family_013.lock missing", snap
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "seed_jobs_sha",
        "scorer_sha",
        "earn_gate_sha",
        "run_one_sha",
        "holdout_manifest_sha",
        "genome_013_lock_sha",
        "kappa_013_lock_sha",
        "context_013_lock_sha",
        "genome_011_lock_sha",
        "reference_route_kappa_sha",
        "expected_n",
        "ctx_encoding",
    ):
        if snap.get(key) != lock.get(key):
            return False, f"family lock drift: {key}", snap
    for f in FAMILIES:
        if snap["generator_sha"].get(f) != (lock.get("generator_sha") or {}).get(f):
            return False, f"family lock drift: generator {f}", snap
    for f in HOLDOUT:
        if snap["holdout_per_family_sha"].get(f) != (lock.get("holdout_per_family_sha") or {}).get(
            f
        ):
            return False, f"family lock drift: holdout manifest {f}", snap
    if lock.get("earned_next") is not False:
        return False, "family lock earned_next must stay false until stamp", snap
    if lock.get("mandatory_measures") != list(MANDATORY_MEASURES):
        return False, "mandatory measures drifted", snap
    return True, "family_013.lock intact", snap


def verify_holdout_sealed(path: Path = FAMILY_LOCK) -> tuple[bool, str, dict[str, Any]]:
    """E–H: hashes / schema / manifests / oracle — no organism answers."""
    ok, why, snap = verify_family_lock(path)
    if not ok:
        return False, why, snap
    for fam in HOLDOUT:
        w = generate_world(fam, DEFAULT_SEED + 1000 * FAMILIES.index(fam), 0)
        if not w.holdout:
            return False, f"{fam} not marked holdout", snap
        if w.kappa_primary == w.kappa_alt:
            return False, f"{fam} κ collision", snap
        if w.expect_primary == w.expect_alt:
            return False, f"{fam} motor collision", snap
        if not any(r.role == "yp" and r.ctx for r in w.relations_primary):
            return False, f"{fam} missing ctx motor", snap
        if not any(r.role == "yt" and r.ctx for r in w.relations_primary):
            return False, f"{fam} missing ctx motor yt", snap
        w2 = World.from_manifest(w.to_manifest())
        if world_manifest_sha(w) != world_manifest_sha(w2):
            return False, f"{fam} manifest round-trip drift", snap
    vec_ok, vec_why, _ = verify_kappa_vectors()
    if not vec_ok:
        return False, vec_why, snap
    # Row-set size matches lock commitment
    if len(committed_holdout_row_shas()) != 144:
        return False, "committed holdout row count != 144", snap
    snap["holdout_sealed_ok"] = True
    return True, "holdout E–H sealed verification OK (no organism answers)", snap


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm013family"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_family(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
    workers: int = 4,
    families: Sequence[str] | None = None,
    allow_holdout_behavior: bool = False,
) -> dict[str, Any]:
    """Run FAMILY battery.

    allow_holdout_behavior=False (default for CI): E–H are not probed.
    Canonical recorded run must pass allow_holdout_behavior=True with full dims.
    """
    run_dir = _run_dir()
    fams = tuple(families) if families is not None else FAMILIES

    # Refuse any behavioral holdout peek outside the full canonical battery.
    if allow_holdout_behavior and any(f in HOLDOUT for f in fams):
        if not is_full_canonical(
            seed=seed,
            per_family=per_family,
            births=births,
            families=fams,
            allow_holdout_behavior=True,
        ):
            raise ValueError(
                "refuse: behavioral E–H contact only allowed for full canonical "
                f"seed={DEFAULT_SEED} per_family={DEFAULT_PER_FAMILY} births={DEFAULT_BIRTHS} "
                "families=A–H"
            )

    vec_ok, vec_why, _ = verify_kappa_vectors()
    g_ok, g_why, g_snap = verify_genome_013()
    fam_ok, fam_why, fam_snap = verify_family_lock()
    genome_ok_start = bool(vec_ok and g_ok and fam_ok)
    freeze_why = "; ".join(x for x in (vec_why, g_why, fam_why) if x)

    holdout_ok = False
    if fam_ok and FAMILY_LOCK.exists():
        live = holdout_manifests(seed=DEFAULT_SEED)
        lock = json.loads(FAMILY_LOCK.read_text(encoding="utf-8"))
        holdout_ok = live["holdout_manifest_sha"] == lock.get("holdout_manifest_sha")

    jobs = []
    for fam, world_seed, b in seed_jobs(
        seed=seed, per_family=per_family, births=births, families=fams
    ):
        w_idx = (world_seed - seed) % 1000
        jobs.append(
            {
                "family": fam,
                "seed": world_seed,
                "birth": b,
                "dest": str(run_dir / fam / f"w{w_idx}_b{b}"),
                "genome_ok": genome_ok_start,
                "allow_holdout_behavior": allow_holdout_behavior,
            }
        )

    if workers <= 1 or len(jobs) <= 1:
        rows = [run_one(j) for j in jobs]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())
    rows.sort(key=lambda r: (r["family"], r["seed"], r["birth"]))

    g_ok_end, g_why_end, _ = verify_genome_013()
    genome_ok_end = bool(g_ok_end)

    summary = earn_gate(
        rows,
        genome_ok_start=genome_ok_start,
        genome_ok_end=genome_ok_end,
        freeze_why=freeze_why if genome_ok_start else freeze_why + f"; end:{g_why_end}",
        seed=seed,
        per_family=per_family,
        births=births,
        holdout_manifest_ok=holdout_ok,
    )
    # Smoke / partial never stamps
    if not is_full_canonical(
        seed=seed,
        per_family=per_family,
        births=births,
        families=fams,
        allow_holdout_behavior=allow_holdout_behavior,
    ):
        summary["ex0s"] = None
        summary["earned_next"] = False
        summary["full_battery"] = False
        summary["claim_name"] = None

    summary["run_dir"] = str(run_dir)
    summary["allow_holdout_behavior"] = allow_holdout_behavior
    summary["genome_013"] = g_snap
    summary["family_lock"] = fam_snap
    summary["rows"] = rows

    (run_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in summary.items()}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TM.0.13.FAMILY")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--per-family", type=int, default=DEFAULT_PER_FAMILY)
    p.add_argument("--births", type=int, default=DEFAULT_BIRTHS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--write-lock",
        action="store_true",
        help="Write family_013.lock (holdout manifests; no organism answers).",
    )
    p.add_argument(
        "--develop-only",
        action="store_true",
        help="Behavioral run on A–D only (safe pre-canonical).",
    )
    p.add_argument(
        "--canonical",
        action="store_true",
        help="Full 288 with holdout behavioral contact (first E–H peek).",
    )
    p.add_argument(
        "--verify-sealed",
        action="store_true",
        help="Verify E–H manifests/hashes without organism answers.",
    )
    args = p.parse_args(argv)

    if args.write_lock:
        snap = write_family_lock()
        print(json.dumps({k: snap[k] for k in snap if k != "preregistered_claim"}, indent=2))
        print("wrote", FAMILY_LOCK)
        return 0

    if args.verify_sealed:
        ok, why, snap = verify_holdout_sealed()
        print(why)
        return 0 if ok else 1

    if args.canonical:
        summary = run_family(
            seed=args.seed,
            per_family=args.per_family,
            births=args.births,
            workers=args.workers,
            allow_holdout_behavior=True,
        )
    elif args.develop_only:
        summary = run_family(
            seed=args.seed,
            per_family=args.per_family,
            births=args.births,
            workers=args.workers,
            families=DEVELOP,
            allow_holdout_behavior=False,
        )
    else:
        # Default: A–D behavioral only (safe)
        summary = run_family(
            seed=args.seed,
            per_family=min(args.per_family, 1),
            births=min(args.births, 1),
            workers=args.workers,
            families=DEVELOP,
            allow_holdout_behavior=False,
        )

    pub = {k: summary[k] for k in summary if k not in ("rows", "genome_013", "family_lock")}
    print(json.dumps(pub, indent=2, default=str))
    for fam in (DEVELOP if args.develop_only or not args.canonical else FAMILIES):
        f = summary["families"].get(fam, {})
        print(f"  {fam}: {f.get('solved')}/{f.get('n')} depth={f.get('depth')}")
    return 0 if summary.get("solved_frac") == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
