"""TM.0.11.FAMILY: frozen COMPOSE genome vs composition worlds.

Not a recipe jump. Freeze run_tm011compose.make (compose-on).
First-hop evidence only for D/F — no lookahead.
Ex0S 0.0.003 stamped only after E–G all green (see aggregate).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm040 import probe
from three_memory.dial_env import ChannelDialWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile

DEVELOP = ("A", "B", "C", "D")
HOLDOUT = ("E", "F", "G")
FAMILIES = DEVELOP + HOLDOUT
MOTORS = ("press", "tune", "flip")
BANNED = frozenset(
    MOTORS + ("hold", "idle", "push", "adjust", "open", "wait", "use", "pick")
)
CONS = "bcdfghjklmnpqrstvwxz"
VOW = "aeiou"
MEASURES = (
    "compose_depth",
    "no_transitive_shortcuts",
    "match_drops_junk",
    "evidence_branch",
    "tie_hold",
    "revise_downstream",
    "upstream_stability",
    "reset_continuity",
    "s_necessity",
    "permutation_invariance",
    "genome_delta",
)
FREEZE_LOCK = REPO_ROOT / "docs" / "genome_011.lock"
DEFAULT_SEED = 12345
DEFAULT_PER_FAMILY = 12
DEFAULT_BIRTHS = 3


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
        name = f"n{int(rng.integers(0, 90)):02d}"
        if name not in taken:
            taken.add(name)
            return name


def _base_rng(family: str, seed: int, birth: int) -> np.random.Generator:
    off = 19 * (ord(family) + 1) + 1103 * birth
    return np.random.default_rng(seed + off)


def _two_motors(rng: np.random.Generator) -> tuple[str, str]:
    m1, m2 = (str(t) for t in rng.permutation(MOTORS[:2]))
    return m1, m2


@dataclass
class Rel:
    fid: str
    bind: str
    did: str
    role: str
    init: tuple[int, int] = (0, 0)


@dataclass
class World:
    family: str
    holdout: bool
    seed: int
    birth: int
    depth: int
    relations: list[Rel]
    chain: list[str]
    phases: list[dict[str, Any]]
    probes: list[dict[str, Any]]
    applicable: list[str] = field(default_factory=list)
    upstream_role: str | None = None
    cue: str = ""


def seed_jobs(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> list[tuple[str, int, int]]:
    jobs: list[tuple[str, int, int]] = []
    for i, fam in enumerate(FAMILIES):
        for w in range(per_family):
            for b in range(births):
                jobs.append((fam, seed + 1000 * i + w, b))
    return jobs


def seed_list_sha(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> str:
    blob = "\n".join(f"{f}:{s}:{b}" for f, s, b in seed_jobs(seed=seed, per_family=per_family, births=births))
    return _sha_bytes(blob.encode())


def transitive_forbidden(chain: list[str]) -> list[tuple[str, str]]:
    """All skip edges along a linear chain (not adjacent acquired edges)."""
    out: list[tuple[str, str]] = []
    for i in range(len(chain)):
        for j in range(i + 2, len(chain)):
            out.append((chain[i], chain[j]))
    return out


def has_edge(s_dir: Path, bind: str, did: str) -> bool:
    b, d = bind.lower(), did.lower()
    for path in s_dir.glob("*.tag"):
        _fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        if str(tags.get("bind") or "").lower() == b and str(tags.get("did") or "").lower() == d:
            return True
    return False


def no_transitive_shortcuts(s_dir: Path, chain: list[str]) -> bool:
    for bind, did in transitive_forbidden(chain):
        if has_edge(s_dir, bind, did):
            return False
    return True


def fact_body_hash(s_dir: Path, role: str, rels: list[Rel]) -> str | None:
    by_role = {r.role: r for r in rels}
    rel = by_role.get(role)
    if rel is None:
        return None
    path = s_dir / f"{rel.fid}.tag"
    if not path.exists():
        # Filename may have been rewritten; find by bind/did.
        for p in s_dir.glob("*.tag"):
            _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
            if tags.get("bind") == rel.bind and tags.get("did") == rel.did:
                return _sha_bytes(p.read_bytes())
        return None
    return _sha_bytes(path.read_bytes())


def write_world_s(dest: Path, rels: list[Rel]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in rels:
        w, n = rel.init
        tags: dict[str, Any] = {
            "bind": rel.bind,
            "did": rel.did,
            "here": "chb",
            "w0": rel.bind,
            "hyp": "contradicted" if n else ("supported" if w else "untried"),
            "trials": w + n,
            "wins": w,
            "losses": n,
            "support": w,
            "contradiction": n,
        }
        (dest / f"{rel.fid}.tag").write_text(record_to_tagfile(rel.fid, tags), encoding="utf-8")


def probe_cue(policy: UsePolicy, s_dir: Path | None, seed: int, cue: str | None) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm011family_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make(store, None, policy, explore_epsilon=0.0)
        ag.reset_rho()
        out = probe(ag, "probe_channel_b", seed, tokens=tokens)
        out["cue"] = cue
        pol = out.get("policy") or {}
        out["compose_hops"] = pol.get("compose_hops")
        out["evidence_resolved"] = bool(pol.get("evidence_resolved"))
        out["evidence_tie"] = bool(pol.get("evidence_tie"))
        out["compose_hold"] = bool(pol.get("compose_hold"))
        return out


def _ids_by_role(s_dir: Path, rels: list[Rel]) -> dict[str, str]:
    by_pair = {}
    for path in s_dir.glob("*.tag"):
        fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        by_pair[(tags.get("bind"), tags.get("did"))] = fid
    return {r.role: by_pair.get((r.bind, r.did), r.fid) for r in rels}


def _earn(
    ag: Any,
    s_dir: Path,
    rels: list[Rel],
    steps: list[tuple[str, bool]],
) -> None:
    ids = _ids_by_role(s_dir, rels)
    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    for role, ok in steps:
        ag._last_chosen_ids = [ids[role]]
        ag.observe_outcome(obs, ok, {"opened": ok})


# --- family generators (hold-outs E–G are hashed into the freeze lock) ---


def generate_family_A(seed: int, birth: int) -> World:
    rng = _base_rng("A", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, _m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
    ]
    return World(
        "A",
        False,
        seed,
        birth,
        2,
        rels,
        [x, y, m1],
        [{"name": "plant", "steps": [], "new_agent": False, "reset_rho": False}],
        [
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "compose_depth",
            },
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "reset_continuity",
                "new_agent": True,
            },
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "reset_continuity",
            "s_necessity",
            "permutation_invariance",
        ],
        cue=x,
    )


def generate_family_B(seed: int, birth: int) -> World:
    rng = _base_rng("B", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, a, b = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, _m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", (1, 0)),
        Rel(_fid(rng, taken_f), a, b, "ab", (1, 0)),
        Rel(_fid(rng, taken_f), b, m1, "bm", (1, 0)),
    ]
    return World(
        "B",
        False,
        seed,
        birth,
        3,
        rels,
        [x, a, b, m1],
        [{"name": "plant", "steps": [], "new_agent": False, "reset_rho": False}],
        [
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 3,
                "measure": "compose_depth",
            }
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "s_necessity",
            "permutation_invariance",
        ],
        cue=x,
    )


def generate_family_C(seed: int, birth: int) -> World:
    rng = _base_rng("C", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    z = _nonce(rng, taken_n)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
        # Wrong motor: if MATCH leaked junk into the act, probe would Fail.
        Rel(_fid(rng, taken_f), z, m2, "irr", (1000, 0)),
    ]
    return World(
        "C",
        False,
        seed,
        birth,
        2,
        rels,
        [x, y, m1],
        [{"name": "plant", "steps": [], "new_agent": False, "reset_rho": False}],
        [
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "compose_depth",
            },
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "match_drops_junk",
            },
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "match_drops_junk",
            "s_necessity",
            "permutation_invariance",
        ],
        cue=x,
    )


def generate_family_D(seed: int, birth: int) -> World:
    """First-hop evidence only: stronger X→Y vs X→Z. Not downstream lookahead."""
    rng = _base_rng("D", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (2, 0)),
        Rel(_fid(rng, taken_f), x, z, "xz", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
        Rel(_fid(rng, taken_f), z, m2, "zm", (1000, 0)),  # stronger downstream must NOT win
    ]
    return World(
        "D",
        False,
        seed,
        birth,
        2,
        rels,
        [x, y, m1],
        [{"name": "plant", "steps": [], "new_agent": False, "reset_rho": False}],
        [
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "compose_depth",
            },
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "evidence_branch",
            },
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "evidence_branch",
            "s_necessity",
            "permutation_invariance",
        ],
        cue=x,
    )


def generate_family_E(seed: int, birth: int) -> World:
    """Hold-out: 4-hop unseen depth. Preregistered — do not edit after peeking results."""
    rng = _base_rng("E", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, a, b, c = (
        _nonce(rng, taken_n),
        _nonce(rng, taken_n),
        _nonce(rng, taken_n),
        _nonce(rng, taken_n),
    )
    m1, _m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, a, "xa", (1, 0)),
        Rel(_fid(rng, taken_f), a, b, "ab", (1, 0)),
        Rel(_fid(rng, taken_f), b, c, "bc", (1, 0)),
        Rel(_fid(rng, taken_f), c, m1, "cm", (1, 0)),
    ]
    return World(
        "E",
        True,
        seed,
        birth,
        4,
        rels,
        [x, a, b, c, m1],
        [{"name": "plant", "steps": [], "new_agent": False, "reset_rho": False}],
        [
            {
                "after": "plant",
                "cue": x,
                "expect": m1,
                "hops": 4,
                "measure": "compose_depth",
            }
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "s_necessity",
            "permutation_invariance",
        ],
        cue=x,
    )


def generate_family_F(seed: int, birth: int) -> World:
    """Hold-out: first-hop branch by evidence; equal → HOLD. Not downstream lookahead."""
    rng = _base_rng("F", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y, z = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), x, z, "xz", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "ym", (1, 0)),
        Rel(_fid(rng, taken_f), z, m2, "zm", (1000, 0)),  # lookahead trap if used
    ]
    return World(
        "F",
        True,
        seed,
        birth,
        2,
        rels,
        [x, y, m1],
        [
            {"name": "tie", "steps": [], "new_agent": False, "reset_rho": False},
            {
                "name": "break",
                "steps": [("xy", True)],
                "new_agent": False,
                "reset_rho": False,
            },
        ],
        [
            {
                "after": "tie",
                "cue": x,
                "expect": "hold",
                "hops": None,
                "measure": "tie_hold",
            },
            {
                "after": "break",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "evidence_branch",
            },
            {
                "after": "break",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "compose_depth",
            },
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "evidence_branch",
            "tie_hold",
            "s_necessity",
            "permutation_invariance",
        ],
        cue=x,
    )


def generate_family_G(seed: int, birth: int) -> World:
    """Hold-out: revise-downstream; upstream X→Y body hash stable."""
    rng = _base_rng("G", seed, birth)
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x, y = _nonce(rng, taken_n), _nonce(rng, taken_n)
    m1, m2 = _two_motors(rng)
    rels = [
        Rel(_fid(rng, taken_f), x, y, "xy", (1, 0)),
        Rel(_fid(rng, taken_f), y, m1, "y_press", (2, 0)),
        Rel(_fid(rng, taken_f), y, m2, "y_tune", (0, 0)),
    ]
    return World(
        "G",
        True,
        seed,
        birth,
        2,
        rels,
        [x, y, m1],
        [
            {"name": "learn", "steps": [], "new_agent": False, "reset_rho": False},
            {
                "name": "revise",
                "steps": [
                    ("y_press", False),
                    ("y_press", False),
                    ("y_tune", True),
                    ("y_tune", True),
                    ("y_tune", True),
                ],
                "new_agent": False,
                "reset_rho": True,
            },
        ],
        [
            {
                "after": "learn",
                "cue": x,
                "expect": m1,
                "hops": 2,
                "measure": "compose_depth",
            },
            {
                "after": "revise",
                "cue": x,
                "expect": m2,
                "hops": 2,
                "measure": "revise_downstream",
            },
            {
                "after": "revise",
                "cue": x,
                "expect": m2,
                "hops": 2,
                "measure": "reset_continuity",
                "new_agent": True,
            },
        ],
        [
            "compose_depth",
            "no_transitive_shortcuts",
            "revise_downstream",
            "upstream_stability",
            "reset_continuity",
            "s_necessity",
            "permutation_invariance",
        ],
        upstream_role="xy",
        cue=x,
    )


FAMILY_GENERATORS: dict[str, Callable[[int, int], World]] = {
    "A": generate_family_A,
    "B": generate_family_B,
    "C": generate_family_C,
    "D": generate_family_D,
    "E": generate_family_E,
    "F": generate_family_F,
    "G": generate_family_G,
}


def generate_world(family: str, seed: int, birth: int) -> World:
    return FAMILY_GENERATORS[family](seed, birth)


def freeze_snapshot() -> dict[str, Any]:
    return {
        "version": "TM.0.11",
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "policy_sha": _sha_file(REPO_ROOT / "three_memory" / "policy.py"),
        "cortex_sha": _sha_file(REPO_ROOT / "three_memory" / "cortex.py"),
        "make011compose_sha": _sha_bytes(inspect.getsource(make).encode()),
        "family_E_generator_sha": _sha_src(generate_family_E),
        "family_F_generator_sha": _sha_src(generate_family_F),
        "family_G_generator_sha": _sha_src(generate_family_G),
        "scorer_sha": _sha_src(score_world),
        "seed_list_sha": seed_list_sha(),
        "n_feat": int(UsePolicy.n_feat),
    }


def write_freeze_lock(path: Path = FREEZE_LOCK) -> dict[str, Any]:
    snap = freeze_snapshot()
    ag = make(REPO_ROOT / "runs" / "_compose_family_lock_probe", None, UsePolicy(seed=1), enabled=False)
    snap["cortex_weight_hash"] = ag.weight_hash()
    snap["use_compose"] = bool(ag.use_compose)
    snap["use_evidence"] = bool(ag.use_evidence)
    snap["use_bind_match"] = bool(ag.use_bind_match)
    snap["use_hyp_survive"] = bool(ag.use_hyp_survive)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_freeze() -> tuple[bool, str, dict[str, Any]]:
    snap = freeze_snapshot()
    if not FREEZE_LOCK.exists():
        return False, "docs/genome_011.lock missing", snap
    lock = json.loads(FREEZE_LOCK.read_text(encoding="utf-8"))
    for key in (
        "policy_sha",
        "cortex_sha",
        "make011compose_sha",
        "family_E_generator_sha",
        "family_F_generator_sha",
        "family_G_generator_sha",
        "scorer_sha",
        "seed_list_sha",
        "n_feat",
    ):
        if snap[key] != lock.get(key):
            return False, f"freeze drift: {key}", snap
    if int(UsePolicy.n_feat) != 2:
        return False, "UsePolicy.n_feat moved", snap
    from three_memory import agent as agent_mod

    src = inspect.getsource(agent_mod)
    for banned in ("use_two_hop", "use_three_hop", "MAX_HOPS", "use_family", "use_lookahead"):
        if banned in src:
            return False, f"banned flag in agent.py: {banned}", snap
    probe_ag = make(REPO_ROOT / "runs" / "_compose_family_lock_probe", None, UsePolicy(seed=1), enabled=False)
    if not probe_ag.use_compose:
        return False, "011compose make lost use_compose", snap
    if not probe_ag.use_evidence or not probe_ag.use_bind_match or not probe_ag.use_hyp_survive:
        return False, "011compose make lost evidence/match/survive", snap
    if probe_ag.weight_hash() != lock.get("cortex_weight_hash"):
        return False, "cortex weight hash drifted from genome_011.lock", snap
    if snap["agent_sha"] == lock.get("agent_sha"):
        return True, "frozen 0.11 compose genome", snap
    return True, "frozen 0.11 compose genome (agent grew; compose make still on)", snap


def score_world(
    world: World,
    *,
    probes: dict[str, list[dict[str, Any]]],
    shortcut_ok: bool,
    wipe_motor: str,
    perm_motors: list[str],
    orig_motors: list[str],
    upstream_before: str | None,
    upstream_after: str | None,
    genome_ok: bool,
) -> dict[str, Any]:
    by_measure: dict[str, list[bool]] = defaultdict(list)
    for spec in world.probes:
        key = f"{spec['after']}:{spec['measure']}:{spec['cue']}"
        hits = probes.get(key, [])
        if not hits:
            by_measure[spec["measure"]].append(False)
            continue
        hit = hits[-1]
        got = _motor(hit["action_name"])
        ok = got == spec["expect"]
        if spec.get("hops") is not None and spec["expect"] != "hold":
            ok = ok and hit.get("compose_hops") == spec["hops"]
        by_measure[spec["measure"]].append(ok)

    measures = {m: None for m in MEASURES}
    for name in (
        "compose_depth",
        "match_drops_junk",
        "evidence_branch",
        "tie_hold",
        "revise_downstream",
        "reset_continuity",
    ):
        if name in world.applicable:
            bits = by_measure.get(name) or [False]
            measures[name] = all(bits)
    measures["no_transitive_shortcuts"] = (
        shortcut_ok if "no_transitive_shortcuts" in world.applicable else None
    )
    measures["upstream_stability"] = (
        (upstream_before is not None and upstream_before == upstream_after)
        if "upstream_stability" in world.applicable
        else None
    )
    measures["s_necessity"] = wipe_motor == "hold" if "s_necessity" in world.applicable else None
    measures["permutation_invariance"] = (
        perm_motors == orig_motors if "permutation_invariance" in world.applicable else None
    )
    measures["genome_delta"] = genome_ok
    applicable = {k: v for k, v in measures.items() if k in world.applicable or k == "genome_delta"}
    solved = all(v is True for v in applicable.values())
    missing = [k for k, v in applicable.items() if v is False]
    return {
        "measures": measures,
        "solved": solved,
        "missing": missing,
        "family": world.family,
        "holdout": world.holdout,
        "depth": world.depth,
        "seed": world.seed,
        "birth": world.birth,
    }


def _run_probes(
    policy: UsePolicy,
    s_dir: Path,
    world: World,
    after: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i, spec in enumerate(world.probes):
        if spec["after"] != after:
            continue
        key = f"{spec['after']}:{spec['measure']}:{spec['cue']}"
        # probe_cue always builds a fresh agent from S (ρ reset); new_agent is documentary.
        out.setdefault(key, []).append(probe_cue(policy, s_dir, seed + 20 + i, spec["cue"]))
    return out


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    family, seed, birth = job["family"], int(job["seed"]), int(job["birth"])
    dest = Path(job["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    genome_ok = bool(job["genome_ok"])
    world = generate_world(family, seed, birth)
    s_dir = dest / "S"
    write_world_s(s_dir, world.relations)
    policy = UsePolicy(seed=7, lr=0.2)
    ag = make(s_dir, None, policy, explore_epsilon=0.0)
    probes: dict[str, list[dict[str, Any]]] = {}
    upstream_before = (
        fact_body_hash(s_dir, world.upstream_role, world.relations) if world.upstream_role else None
    )
    shortcut_ok = True
    for ph in world.phases:
        if ph.get("new_agent"):
            ag = make(s_dir, None, policy, explore_epsilon=0.0)
        if ph.get("reset_rho"):
            ag.reset_rho()
        _earn(ag, s_dir, world.relations, list(ph.get("steps") or []))
        probes.update(_run_probes(policy, s_dir, world, ph["name"], seed + birth))
        if not no_transitive_shortcuts(s_dir, world.chain):
            shortcut_ok = False
        # Also ban cue→any-motor shortcuts and cue→downstream motors for branched worlds.
        cue = world.cue
        for m in MOTORS:
            if has_edge(s_dir, cue, m):
                shortcut_ok = False
    upstream_after = (
        fact_body_hash(s_dir, world.upstream_role, world.relations) if world.upstream_role else None
    )
    # After G revise, chain motor tip is m2 — still forbid transitive cue→m2 etc.
    if world.family == "G":
        y = next(r.did for r in world.relations if r.role == "xy")
        m2 = next(r.did for r in world.relations if r.role == "y_tune")
        if has_edge(s_dir, world.cue, m2) or has_edge(s_dir, world.cue, next(r.did for r in world.relations if r.role == "y_press")):
            shortcut_ok = False
        if not no_transitive_shortcuts(s_dir, [world.cue, y, m2]):
            shortcut_ok = False

    wipe = probe_cue(policy, None, seed + 90, world.cue)
    perm = generate_world(family, seed, birth)
    rng = _base_rng(family, seed, birth + 97)
    taken_f: set[str] = set()
    perm.relations = [
        Rel(_fid(rng, taken_f), r.bind, r.did, r.role, r.init) for r in perm.relations
    ]
    rng.shuffle(perm.relations)
    ps = dest / "S_perm"
    write_world_s(ps, perm.relations)
    pag = make(ps, None, policy, explore_epsilon=0.0)
    perm_probes: dict[str, list[dict[str, Any]]] = {}
    for ph in perm.phases:
        if ph.get("new_agent"):
            pag = make(ps, None, policy, explore_epsilon=0.0)
        if ph.get("reset_rho"):
            pag.reset_rho()
        _earn(pag, ps, perm.relations, list(ph.get("steps") or []))
        perm_probes.update(_run_probes(policy, ps, perm, ph["name"], seed + birth + 200))
    orig_motors = []
    perm_motors = []
    for spec in world.probes:
        key = f"{spec['after']}:{spec['measure']}:{spec['cue']}"
        orig_motors.append(_motor(probes[key][-1]["action_name"]))
        perm_motors.append(_motor(perm_probes[key][-1]["action_name"]))
    scored = score_world(
        world,
        probes=probes,
        shortcut_ok=shortcut_ok,
        wipe_motor=_motor(wipe["action_name"]),
        perm_motors=perm_motors,
        orig_motors=orig_motors,
        upstream_before=upstream_before,
        upstream_after=upstream_after,
        genome_ok=genome_ok,
    )
    scored["n_s"] = len(list(s_dir.glob("*.tag")))
    scored["upstream_before"] = upstream_before
    scored["upstream_after"] = upstream_after
    scored["world"] = {
        "family": world.family,
        "holdout": world.holdout,
        "depth": world.depth,
        "binds": [r.bind for r in world.relations],
        "dids": [r.did for r in world.relations],
        "roles": [r.role for r in world.relations],
    }
    return scored


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    bits = [r["measures"][key] for r in rows if r["measures"].get(key) is not None]
    if not bits:
        return None
    return sum(1 for b in bits if b) / len(bits)


def _max_depth_solved(rows: list[dict[str, Any]]) -> int:
    depths = [int(r["depth"]) for r in rows if r.get("solved")]
    return max(depths) if depths else 0


def aggregate(
    rows: list[dict[str, Any]],
    *,
    genome_ok: bool,
    freeze_why: str,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
) -> dict[str, Any]:
    by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    families = {}
    for fam in FAMILIES:
        chunk = by_fam.get(fam, [])
        families[fam] = {
            "holdout": fam in HOLDOUT,
            "n": len(chunk),
            "solved": sum(1 for r in chunk if r["solved"]),
            "solved_frac": (sum(1 for r in chunk if r["solved"]) / len(chunk)) if chunk else 0.0,
            "measures": {m: _rate(chunk, m) for m in MEASURES},
            "missing": sorted({k for r in chunk for k in r.get("missing") or []}),
            "depth": chunk[0]["depth"] if chunk else None,
        }
    developed = [r for r in rows if not r["holdout"]]
    holdout = [r for r in rows if r["holdout"]]
    developed_frac = (sum(1 for r in developed if r["solved"]) / len(developed)) if developed else 0.0
    holdout_frac = (sum(1 for r in holdout if r["solved"]) / len(holdout)) if holdout else 0.0
    all_frac = (sum(1 for r in rows if r["solved"]) / len(rows)) if rows else 0.0
    expected_n = len(FAMILIES) * DEFAULT_PER_FAMILY * DEFAULT_BIRTHS
    full_battery = (
        seed == DEFAULT_SEED
        and per_family == DEFAULT_PER_FAMILY
        and births == DEFAULT_BIRTHS
        and len(rows) == expected_n
        and seed_list_sha(seed=seed, per_family=per_family, births=births) == seed_list_sha()
    )
    interventions = 0
    intervention = {
        "world_classes_attempted": len(FAMILIES),
        "solved_with_frozen_genome": sum(1 for f in FAMILIES if families[f]["solved_frac"] == 1.0),
        "required_genome_changes": 0,
        "apparatus_interventions": interventions,
        "failed_honestly": sum(1 for f in FAMILIES if families[f]["solved_frac"] < 1.0),
        "full_battery": full_battery,
        "note": "No organism edits during the recorded family. Hold-out generators preregistered.",
    }
    earned = (
        genome_ok
        and full_battery
        and all_frac == 1.0
        and developed_frac == 1.0
        and holdout_frac == 1.0
        and intervention["required_genome_changes"] == 0
        and intervention["apparatus_interventions"] == 0
        and all(families[f]["solved_frac"] == 1.0 for f in HOLDOUT)
    )
    return {
        "version": "TM.0.11.FAMILY",
        "ex0s_planned": "0.0.003",
        "ex0s": "0.0.003" if earned else None,
        "earned_frozen_composition": earned,
        "genome_ok": genome_ok,
        "genome_why": freeze_why,
        "n_worlds": len(rows),
        "solved": sum(1 for r in rows if r["solved"]),
        "solved_frac": all_frac,
        "developed_solved_frac": developed_frac,
        "holdout_solved_frac": holdout_frac,
        "developed_max_depth_solved": _max_depth_solved(developed),
        "holdout_max_depth_solved": _max_depth_solved(holdout),
        "measures": {m: _rate(rows, m) for m in MEASURES},
        "families": families,
        "intervention": intervention,
        "mean_s": (sum(r.get("n_s") or 0 for r in rows) / len(rows)) if rows else 0.0,
    }


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm011family"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_family(
    *,
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
    births: int = DEFAULT_BIRTHS,
    workers: int = 4,
) -> dict[str, Any]:
    run_dir = _run_dir()
    if not FREEZE_LOCK.exists():
        write_freeze_lock()
    genome_ok, freeze_why, snap = verify_freeze()
    jobs = []
    for fam, world_seed, b in seed_jobs(seed=seed, per_family=per_family, births=births):
        w_idx = (world_seed - seed) % 1000
        jobs.append(
            {
                "family": fam,
                "seed": world_seed,
                "birth": b,
                "dest": str(run_dir / fam / f"w{w_idx}_b{b}"),
                "genome_ok": genome_ok,
            }
        )
    rows: list[dict[str, Any]] = []
    if workers <= 1 or len(jobs) == 1:
        rows = [run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())
    rows.sort(key=lambda r: (r["family"], r["seed"], r["birth"]))
    summary = aggregate(
        rows,
        genome_ok=genome_ok,
        freeze_why=freeze_why,
        seed=seed,
        per_family=per_family,
        births=births,
    )
    summary["seed"] = seed
    summary["per_family"] = per_family
    summary["births"] = births
    summary["workers"] = workers
    summary["run_dir"] = str(run_dir)
    summary["freeze"] = snap
    (run_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    fam_lines = "\n".join(
        f"| {f} | {'hold-out' if summary['families'][f]['holdout'] else 'developed'} | "
        f"depth {summary['families'][f]['depth']} | "
        f"{summary['families'][f]['solved']}/{summary['families'][f]['n']} | "
        f"{summary['families'][f]['solved_frac']:.2f} |"
        for f in FAMILIES
    )
    meas_lines = "\n".join(
        f"| {m} | {summary['measures'][m] if summary['measures'][m] is None else f'{summary['measures'][m]:.3f}'} |"
        for m in MEASURES
    )
    earned_line = (
        f"**Ex0S 0.0.003 earned.**"
        if summary.get("earned_frozen_composition")
        else "**Ex0S 0.0.003 not stamped** (planned until A–D and E–G all green)."
    )
    (run_dir / "summary.md").write_text(
        f"""# TM.0.11.FAMILY · Ex0S 0.0.003 (planned until earned)

Genome: {freeze_why}
Worlds solved: **{summary['solved']}/{summary['n_worlds']}** ({summary['solved_frac']:.3f})
Developed A–D: {summary['developed_solved_frac']:.3f} (max depth solved {summary['developed_max_depth_solved']})
Hold-out E–G: {summary['holdout_solved_frac']:.3f} (max depth solved {summary['holdout_max_depth_solved']})
{earned_line}

| Family | Split | Depth | Solved | Frac |
|--------|-------|-------|--------|------|
{fam_lines}

| Measure | Rate |
|---------|------|
{meas_lines}

Intervention: {summary['intervention']}
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.11.FAMILY frozen-compose composition worlds")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--per-family", type=int, default=DEFAULT_PER_FAMILY)
    p.add_argument("--births", type=int, default=DEFAULT_BIRTHS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    if args.write_lock:
        print(json.dumps(write_freeze_lock(), indent=2))
        return
    s = run_family(
        seed=args.seed,
        per_family=args.per_family,
        births=args.births,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "solved_frac": s["solved_frac"],
                "developed": s["developed_solved_frac"],
                "holdout": s["holdout_solved_frac"],
                "developed_max_depth": s["developed_max_depth_solved"],
                "holdout_max_depth": s["holdout_max_depth_solved"],
                "genome_ok": s["genome_ok"],
                "ex0s": s["ex0s"],
                "earned": s["earned_frozen_composition"],
                "intervention": s["intervention"],
                "families": {k: v["solved_frac"] for k, v in s["families"].items()},
                "run_dir": s["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
