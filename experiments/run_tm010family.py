"""TM.0.10.FAMILY: frozen 0.9.4 genome vs procedural relation worlds.

Not a recipe jump. No organism edits. Record failures before touching anything.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm093 import _counts, _motor, probe_cue
from experiments.run_tm094 import make
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
    "acquisition",
    "survival",
    "match",
    "evidence_choice",
    "tie_handling",
    "revision",
    "reset_continuity",
    "s_necessity",
    "permutation_invariance",
    "genome_delta",
)
FREEZE_LOCK = REPO_ROOT / "docs" / "genome_094.lock"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def freeze_snapshot() -> dict[str, Any]:
    return {
        "version": "TM.0.9.4",
        "agent_sha": _sha_file(REPO_ROOT / "three_memory" / "agent.py"),
        "policy_sha": _sha_file(REPO_ROOT / "three_memory" / "policy.py"),
        "cortex_sha": _sha_file(REPO_ROOT / "three_memory" / "cortex.py"),
        "make094_sha": _sha_bytes(inspect.getsource(make).encode()),
        "n_feat": int(UsePolicy.n_feat),
    }


def write_freeze_lock(path: Path = FREEZE_LOCK) -> dict[str, Any]:
    snap = freeze_snapshot()
    ag = make(REPO_ROOT / "runs" / "_family_lock_probe", None, UsePolicy(seed=1), enabled=False)
    snap["cortex_weight_hash"] = ag.weight_hash()
    snap["use_evidence"] = bool(ag.use_evidence)
    snap["use_bind_match"] = bool(ag.use_bind_match)
    snap["use_hyp_survive"] = bool(ag.use_hyp_survive)
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_freeze() -> tuple[bool, str, dict[str, Any]]:
    snap = freeze_snapshot()
    if not FREEZE_LOCK.exists():
        return False, "docs/genome_094.lock missing", snap
    lock = json.loads(FREEZE_LOCK.read_text(encoding="utf-8"))
    for key in ("policy_sha", "cortex_sha", "make094_sha", "n_feat"):
        if snap[key] != lock.get(key):
            return False, f"freeze drift: {key}", snap
    if int(UsePolicy.n_feat) != 2:
        return False, "UsePolicy.n_feat moved", snap
    from three_memory import agent as agent_mod

    src = inspect.getsource(agent_mod)
    if "use_family" in src or "use_revision" in src:
        return False, "new family/revision flag in agent.py", snap
    # Agent may grow (e.g. compose default-off). 0.9.4 make must stay compose-off.
    probe_ag = make(REPO_ROOT / "runs" / "_family_lock_probe", None, UsePolicy(seed=1), enabled=False)
    if getattr(probe_ag, "use_compose", False):
        return False, "094 make enabled compose; 0.10 freeze is no longer 0.9.4", snap
    if not probe_ag.use_evidence or not probe_ag.use_bind_match:
        return False, "094 make lost evidence/match", snap
    if probe_ag.weight_hash() != lock.get("cortex_weight_hash"):
        return False, "cortex weight hash drifted from genome_094.lock", snap
    if snap["agent_sha"] == lock.get("agent_sha"):
        return True, "frozen 0.9.4 genome", snap
    return True, "frozen 0.9.4 genome (agent grew; 094 make compose off)", snap


def expected_motor(rels: list[dict[str, Any]], cue: str) -> str:
    """Published 0.9.3 rule: unique max (support, -contradiction) among MATCH-eligible."""
    matched = [r for r in rels if r["bind"] == cue]
    if not matched:
        return "hold"
    best = max((int(r["support"]), -int(r["contradiction"])) for r in matched)
    winners = [r for r in matched if (int(r["support"]), -int(r["contradiction"])) == best]
    dids = {r["did"] for r in winners}
    if len(dids) != 1:
        return "hold"
    return str(next(iter(dids)))


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


def _apply_steps(counts: dict[str, list[int]], steps: list[tuple[str, bool]]) -> None:
    for role, ok in steps:
        pair = counts.setdefault(role, [0, 0])
        if ok:
            pair[0] += 1
        else:
            pair[1] += 1


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
    relations: list[Rel]
    phases: list[dict[str, Any]]
    probes: list[dict[str, Any]]
    applicable: list[str] = field(default_factory=list)


def _base_rng(family: str, seed: int, birth: int) -> np.random.Generator:
    off = 17 * (ord(family) + 1) + 1009 * birth
    return np.random.default_rng(seed + off)


def _two_rivals(rng: np.random.Generator) -> tuple[list[Rel], dict[str, str], set[str], set[str]]:
    taken_n: set[str] = set()
    taken_f: set[str] = set()
    x = _nonce(rng, taken_n)
    m1, m2 = (str(t) for t in rng.permutation(MOTORS[:2]))
    rels = [
        Rel(_fid(rng, taken_f), x, m1, "m1"),
        Rel(_fid(rng, taken_f), x, m2, "m2"),
    ]
    return rels, {"x": x, "m1": m1, "m2": m2}, taken_n, taken_f


def generate_world(family: str, seed: int, birth: int) -> World:
    rng = _base_rng(family, seed, birth)
    holdout = family in HOLDOUT
    if family == "A":
        rels, names, _, _ = _two_rivals(rng)
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [
                {"name": "start", "steps": [], "new_agent": False, "reset_rho": False},
                {
                    "name": "early",
                    "steps": [("m1", True), ("m1", True), ("m2", False)],
                    "new_agent": False,
                    "reset_rho": False,
                },
                {
                    "name": "late",
                    "steps": [("m1", False), ("m1", False), ("m2", True), ("m2", True), ("m2", True)],
                    "new_agent": False,
                    "reset_rho": True,
                },
            ],
            [
                {"after": "start", "cue": names["x"], "expect": "hold", "measure": "tie_handling"},
                {"after": "early", "cue": names["x"], "expect": names["m1"], "measure": "evidence_choice"},
                {"after": "late", "cue": names["x"], "expect": names["m2"], "measure": "revision"},
                {"after": "late", "cue": names["x"], "expect": names["m2"], "measure": "reset_continuity"},
            ],
            [
                "acquisition",
                "survival",
                "evidence_choice",
                "tie_handling",
                "revision",
                "reset_continuity",
                "s_necessity",
                "permutation_invariance",
            ],
        )
    if family == "B":
        rels, names, taken_n, taken_f = _two_rivals(rng)
        y = _nonce(rng, taken_n)
        rels.append(Rel(_fid(rng, taken_f), y, "flip", "irr0", (40, 0)))
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [
                {
                    "name": "early",
                    "steps": [("m1", True), ("m2", False), ("m1", True)],
                    "new_agent": False,
                    "reset_rho": False,
                }
            ],
            [
                {"after": "early", "cue": names["x"], "expect": names["m1"], "measure": "evidence_choice"},
                {"after": "early", "cue": names["x"], "expect": names["m1"], "measure": "match"},
                {"after": "early", "cue": y, "expect": "flip", "measure": "match"},
            ],
            ["acquisition", "survival", "match", "evidence_choice", "s_necessity", "permutation_invariance"],
        )
    if family == "C":
        rels, names, taken_n, taken_f = _two_rivals(rng)
        n_irr = int(rng.integers(4, 9))
        for i in range(n_irr):
            rels.append(
                Rel(_fid(rng, taken_f), _nonce(rng, taken_n), "flip", f"irr{i}", (1000, 0))
            )
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [
                {
                    "name": "early",
                    "steps": [("m1", True), ("m1", True), ("m2", False)],
                    "new_agent": False,
                    "reset_rho": False,
                }
            ],
            [
                {"after": "early", "cue": names["x"], "expect": names["m1"], "measure": "evidence_choice"},
                {"after": "early", "cue": names["x"], "expect": names["m1"], "measure": "match"},
                {"after": "early", "cue": _nonce(rng, taken_n), "expect": "hold", "measure": "match"},
            ],
            ["acquisition", "survival", "match", "evidence_choice", "s_necessity", "permutation_invariance"],
        )
    if family == "D":
        taken_n: set[str] = set()
        taken_f: set[str] = set()
        x, y, z = _nonce(rng, taken_n), _nonce(rng, taken_n), _nonce(rng, taken_n)
        shared, other = (str(t) for t in rng.permutation(MOTORS[:2]))
        rels = [
            Rel(_fid(rng, taken_f), x, shared, "cx"),
            Rel(_fid(rng, taken_f), y, shared, "cy"),
            Rel(_fid(rng, taken_f), z, other, "cz"),
        ]
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [{"name": "early", "steps": [], "new_agent": False, "reset_rho": False}],
            [
                {"after": "early", "cue": x, "expect": shared, "measure": "match"},
                {"after": "early", "cue": y, "expect": shared, "measure": "match"},
                {"after": "early", "cue": z, "expect": other, "measure": "match"},
            ],
            ["acquisition", "survival", "match", "s_necessity", "permutation_invariance"],
        )
    if family == "E":
        rels, names, _, _ = _two_rivals(rng)
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [
                {
                    "name": "life1",
                    "steps": [("m1", True)],
                    "new_agent": False,
                    "reset_rho": False,
                },
                {
                    "name": "life2",
                    "steps": [("m1", True)],
                    "new_agent": True,
                    "reset_rho": True,
                },
            ],
            [
                {"after": "life1", "cue": names["x"], "expect": names["m1"], "measure": "evidence_choice"},
                {"after": "life2", "cue": names["x"], "expect": names["m1"], "measure": "reset_continuity"},
            ],
            ["acquisition", "survival", "evidence_choice", "reset_continuity", "s_necessity", "permutation_invariance"],
        )
    if family == "F":
        rels, names, _, _ = _two_rivals(rng)
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [
                {
                    "name": "tie",
                    "steps": [("m1", True), ("m2", True)],
                    "new_agent": False,
                    "reset_rho": False,
                },
                {
                    "name": "break",
                    "steps": [("m1", True)],
                    "new_agent": False,
                    "reset_rho": False,
                },
            ],
            [
                {"after": "tie", "cue": names["x"], "expect": "hold", "measure": "tie_handling"},
                {"after": "break", "cue": names["x"], "expect": names["m1"], "measure": "evidence_choice"},
            ],
            ["acquisition", "survival", "evidence_choice", "tie_handling", "s_necessity", "permutation_invariance"],
        )
    if family == "G":
        rels, names, _, _ = _two_rivals(rng)
        return World(
            family,
            holdout,
            seed,
            birth,
            rels,
            [
                {
                    "name": "r1",
                    "steps": [("m1", True), ("m1", True), ("m2", False)],
                    "new_agent": False,
                    "reset_rho": False,
                },
                {
                    "name": "r2",
                    "steps": [("m1", False), ("m1", False), ("m2", True), ("m2", True), ("m2", True)],
                    "new_agent": False,
                    "reset_rho": True,
                },
                {
                    "name": "r3",
                    "steps": [("m1", True), ("m1", True)],
                    "new_agent": False,
                    "reset_rho": False,
                },
            ],
            [
                {"after": "r1", "cue": names["x"], "expect": names["m1"], "measure": "evidence_choice"},
                {"after": "r2", "cue": names["x"], "expect": names["m2"], "measure": "revision"},
                {"after": "r2", "cue": names["x"], "expect": names["m2"], "measure": "reset_continuity"},
                {"after": "r3", "cue": names["x"], "expect": names["m1"], "measure": "revision"},
            ],
            [
                "acquisition",
                "survival",
                "evidence_choice",
                "revision",
                "reset_continuity",
                "s_necessity",
                "permutation_invariance",
            ],
        )
    raise ValueError(family)


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


def _ids_by_role(s_dir: Path, rels: list[Rel]) -> dict[str, str]:
    by_pair = {}
    for path in s_dir.glob("*.tag"):
        fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        by_pair[(tags.get("bind"), tags.get("did"))] = fid
    return {r.role: by_pair.get((r.bind, r.did), r.fid) for r in rels}


def _rel_state(s_dir: Path, rels: list[Rel]) -> list[dict[str, Any]]:
    counts = _counts(s_dir)
    ids = _ids_by_role(s_dir, rels)
    out = []
    for r in rels:
        c = counts.get(ids.get(r.role, r.fid), {})
        out.append(
            {
                "fid": ids.get(r.role, r.fid),
                "bind": r.bind,
                "did": r.did,
                "role": r.role,
                "support": int(c.get("support") or 0),
                "contradiction": int(c.get("contradiction") or 0),
                "present": ids.get(r.role) in counts,
            }
        )
    return out


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


def _planned_counts(rels: list[Rel], phases: list[dict[str, Any]], until: str) -> dict[str, list[int]]:
    counts = {r.role: [r.init[0], r.init[1]] for r in rels}
    for ph in phases:
        _apply_steps(counts, list(ph.get("steps") or []))
        if ph["name"] == until:
            break
    return counts


def score_world(
    world: World,
    *,
    probes: dict[str, list[dict[str, Any]]],
    states: dict[str, list[dict[str, Any]]],
    wipe_motor: str,
    perm_motors: list[str],
    orig_motors: list[str],
    genome_ok: bool,
) -> dict[str, Any]:
    rels = world.relations
    last = states[world.phases[-1]["name"]]
    roles_present = {r["role"] for r in last if r.get("present")}
    intended = {r.role for r in rels}
    acquisition = intended <= roles_present
    for ph in world.phases:
        planned = _planned_counts(rels, world.phases, ph["name"])
        got = {r["role"]: [r["support"], r["contradiction"]] for r in states[ph["name"]]}
        if any(got.get(role) != planned[role] for role in planned):
            acquisition = False
    rivals = [r.role for r in rels if r.role.startswith("m") or r.role.startswith("c")]
    survival = all(role in roles_present for role in rivals)
    by_measure: dict[str, list[bool]] = defaultdict(list)
    for spec in world.probes:
        hits = probes.get(f"{spec['after']}:{spec['measure']}:{spec['cue']}", [])
        if not hits:
            by_measure[spec["measure"]].append(False)
            continue
        got = _motor(hits[-1]["action_name"])
        by_measure[spec["measure"]].append(got == spec["expect"])
        if spec["measure"] == "match" and spec["expect"] != "flip":
            by_measure["match"].append(got != "flip")
    if "match" in world.applicable and "match" not in by_measure:
        by_measure["match"] = [True]
    measures = {m: None for m in MEASURES}
    measures["acquisition"] = acquisition
    measures["survival"] = survival if "survival" in world.applicable else None
    for name in ("match", "evidence_choice", "tie_handling", "revision", "reset_continuity"):
        if name in world.applicable:
            bits = by_measure.get(name) or [False]
            measures[name] = all(bits)
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
    states: dict[str, list[dict[str, Any]]] = {}
    for ph in world.phases:
        if ph.get("new_agent"):
            ag = make(s_dir, None, policy, explore_epsilon=0.0)
        if ph.get("reset_rho"):
            ag.reset_rho()
        _earn(ag, s_dir, world.relations, list(ph.get("steps") or []))
        states[ph["name"]] = _rel_state(s_dir, world.relations)
        probes.update(_run_probes(policy, s_dir, world, ph["name"], seed + birth))
    wipe = probe_cue(policy, None, seed + 90, world.probes[-1]["cue"])
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
        states=states,
        wipe_motor=_motor(wipe["action_name"]),
        perm_motors=perm_motors,
        orig_motors=orig_motors,
        genome_ok=genome_ok,
    )
    scored["n_s"] = len(list(s_dir.glob("*.tag")))
    scored["world"] = {
        "family": world.family,
        "holdout": world.holdout,
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


def aggregate(rows: list[dict[str, Any]], *, genome_ok: bool, freeze_why: str) -> dict[str, Any]:
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
        }
    developed = [r for r in rows if not r["holdout"]]
    holdout = [r for r in rows if r["holdout"]]
    intervention = {
        "world_classes_attempted": len(FAMILIES),
        "solved_with_frozen_genome": sum(1 for f in FAMILIES if families[f]["solved_frac"] == 1.0),
        "required_genome_changes": 0,
        "failed_honestly": sum(1 for f in FAMILIES if families[f]["solved_frac"] < 1.0),
        "note": "No organism edits during the recorded family.",
    }
    return {
        "version": "TM.0.10.FAMILY",
        "ex0s": "0.0.001",
        "genome_ok": genome_ok,
        "genome_why": freeze_why,
        "n_worlds": len(rows),
        "solved": sum(1 for r in rows if r["solved"]),
        "solved_frac": (sum(1 for r in rows if r["solved"]) / len(rows)) if rows else 0.0,
        "developed_solved_frac": (sum(1 for r in developed if r["solved"]) / len(developed)) if developed else 0.0,
        "holdout_solved_frac": (sum(1 for r in holdout if r["solved"]) / len(holdout)) if holdout else 0.0,
        "measures": {m: _rate(rows, m) for m in MEASURES},
        "families": families,
        "intervention": intervention,
        "mean_s": (sum(r.get("n_s") or 0 for r in rows) / len(rows)) if rows else 0.0,
    }


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm010family"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_family(
    *,
    seed: int = 12345,
    per_family: int = 12,
    births: int = 3,
    workers: int = 4,
) -> dict[str, Any]:
    run_dir = _run_dir()
    if not FREEZE_LOCK.exists():
        write_freeze_lock()
    genome_ok, freeze_why, snap = verify_freeze()
    jobs = []
    for i, fam in enumerate(FAMILIES):
        for w in range(per_family):
            for b in range(births):
                world_seed = seed + 1000 * i + w
                jobs.append(
                    {
                        "family": fam,
                        "seed": world_seed,
                        "birth": b,
                        "dest": str(run_dir / fam / f"w{w}_b{b}"),
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
    summary = aggregate(rows, genome_ok=genome_ok, freeze_why=freeze_why)
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
        f"{summary['families'][f]['solved']}/{summary['families'][f]['n']} | "
        f"{summary['families'][f]['solved_frac']:.2f} |"
        for f in FAMILIES
    )
    meas_lines = "\n".join(
        f"| {m} | {summary['measures'][m] if summary['measures'][m] is None else f'{summary['measures'][m]:.3f}'} |"
        for m in MEASURES
    )
    (run_dir / "summary.md").write_text(
        f"""# TM.0.10.FAMILY · Ex0S 0.0.001

Genome: {freeze_why}
Worlds solved: **{summary['solved']}/{summary['n_worlds']}** ({summary['solved_frac']:.3f})
Developed A–D: {summary['developed_solved_frac']:.3f}
Hold-out E–G: {summary['holdout_solved_frac']:.3f}

| Family | Split | Solved | Frac |
|--------|-------|--------|------|
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
    p = argparse.ArgumentParser(description="TM.0.10.FAMILY frozen-genome relation worlds")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--per-family", type=int, default=12)
    p.add_argument("--births", type=int, default=3)
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
                "genome_ok": s["genome_ok"],
                "intervention": s["intervention"],
                "families": {k: v["solved_frac"] for k, v in s["families"].items()},
                "run_dir": s["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
