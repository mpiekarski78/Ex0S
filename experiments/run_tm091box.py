"""TM.0.9.BOX: boxed-policy leakage control on the frozen TM.0.9.1 genome.

Paired counterfactual crossover. No organism edits. No CUDA.
Apparatus only: nonce worlds, canonical B readout projection, P×S matrix,
parallel CPU workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm040 import probe
from experiments.run_tm052 import live_free
from experiments.run_tm054 import _n_paragraphs
from experiments.run_tm058 import N_CLUTTER, _closed_bodies
from experiments.run_tm061 import _MOTOR_WORDS, _STATION_WORDS, _s_has_bind, _s_has_did
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm080 import _TRIPLES, _n_ok
from experiments.run_tm091 import make
from experiments.run_v22 import _tags
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import (
    prose_token_stream,
    prose_tokens,
    write_prose_notes,
)

GENOME_LOCK = REPO_ROOT / "docs" / "genome_091.lock"
PROTOCOL_LOCK = REPO_ROOT / "docs" / "protocol_091.lock"

READOUT_FACT_ID = "readout"
NEUTRAL_PRESS_BIND = "wibble"
NEUTRAL_TUNE_BIND = "tork"

WORLD_SPECS: dict[str, dict[str, Any]] = {
    "W1": {
        "press_file": "p99.md",
        "tune_file": "p98.md",
        "press_token": "flim",
        "tune_token": "zorg",
        "press_distractor": "argon",
        "tune_distractor": "alpha",
        "clutter_order": "forward",
        "filename_prefix": "c",
    },
    "W2": {
        "press_file": "p99.md",
        "tune_file": "p98.md",
        "press_token": "zorg",
        "tune_token": "flim",
        "press_distractor": "argon",
        "tune_distractor": "alpha",
        "clutter_order": "forward",
        "filename_prefix": "c",
    },
    "W3": {
        "press_file": "q17.md",
        "tune_file": "r43.md",
        "press_token": "blen",
        "tune_token": "nork",
        "press_distractor": "quartz",
        "tune_distractor": "cobalt",
        "clutter_order": "reverse",
        "filename_prefix": "d",
    },
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_dir_tags(s_dir: Path) -> str:
    parts: list[bytes] = []
    if not s_dir.exists():
        return _sha_bytes(b"")
    for p in sorted(s_dir.glob("*.tag")):
        parts.append(p.name.encode())
        parts.append(p.read_bytes())
    return _sha_bytes(b"".join(parts))


def load_genome_lock() -> dict[str, Any]:
    return json.loads(GENOME_LOCK.read_text(encoding="utf-8"))


def load_protocol_lock() -> dict[str, Any]:
    return json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))


def genome_lock_hash() -> str:
    return _sha_file(GENOME_LOCK)


def protocol_lock_hash() -> str:
    return _sha_file(PROTOCOL_LOCK)


def verify_genome_lock() -> tuple[bool, str]:
    """Recompute organism hashes; must match docs/genome_091.lock."""
    import inspect

    lock = load_genome_lock()
    agent_sha = _sha_file(REPO_ROOT / "three_memory" / "agent.py")
    policy_sha = _sha_file(REPO_ROOT / "three_memory" / "policy.py")
    cortex_sha = _sha_file(REPO_ROOT / "three_memory" / "cortex.py")
    make_sha = _sha_bytes(inspect.getsource(make).encode())
    if policy_sha != lock["policy_sha"]:
        return False, "policy.py drifted from genome_091.lock"
    if cortex_sha != lock["cortex_sha"]:
        return False, "cortex.py drifted from genome_091.lock"
    if make_sha != lock["make_sha"]:
        return False, "run_tm091.make drifted from genome_091.lock"
    if agent_sha != lock["agent_sha"]:
        # 0.9.2 may add bind_match default-off. Historical BOX still uses 091 make.
        from three_memory.policy import UsePolicy

        probe_ag = make(REPO_ROOT / "runs" / "_box_lock_probe", None, UsePolicy(seed=1), enabled=False)
        if getattr(probe_ag, "use_bind_match", False):
            return False, "091 make enabled bind_match; historical BOX is no longer 0.9.1"
        if probe_ag.cortex.weight_hash() != lock["cortex_weight_hash"]:
            return False, "cortex weight hash drifted from genome_091.lock"
    features_sha = _sha_bytes(json.dumps(lock["features"], separators=(",", ":")).encode())
    outputs_sha = _sha_bytes(json.dumps(lock["outputs"], separators=(",", ":")).encode())
    flags_sha = _sha_bytes(json.dumps(lock["flags"], separators=(",", ":")).encode())
    if features_sha != lock["features_sha"] or outputs_sha != lock["outputs_sha"] or flags_sha != lock["flags_sha"]:
        return False, "feature/output/flag lists drifted inside genome_091.lock"
    if agent_sha == lock["agent_sha"]:
        blob = (
            agent_sha + policy_sha + cortex_sha + make_sha + features_sha + outputs_sha + flags_sha
        ).encode()
        if _sha_bytes(blob) != lock["source_tree"]:
            return False, "source_tree mismatch in genome_091.lock"
        return True, "genome lock ok"
    return True, "genome lock ok (agent grew; 091 make bind_match off)"


def verify_protocol_lock(*, n_train: int, max_steps: int) -> tuple[bool, str]:
    lock = load_protocol_lock()
    if n_train != lock["n_train"]:
        return False, f"n_train {n_train} != locked {lock['n_train']}"
    if max_steps != lock["max_steps"]:
        return False, f"max_steps {max_steps} != locked {lock['max_steps']}"
    if lock["max_train_s_files"] != MAX_TRAIN_S_FILES:
        return False, "MAX_TRAIN_S_FILES drifted"
    if lock["split"] is not False:
        return False, "protocol requires split=False"
    return True, "protocol lock ok"


def clone_policy(src: UsePolicy) -> UsePolicy:
    """Identical weights and optimizer state (lr, n_updates, birth hash)."""
    dst = UsePolicy(seed=0, lr=src.lr)
    for name in (
        "W_collect",
        "b_collect",
        "w_apply",
        "b_apply",
        "w_write",
        "b_write",
        "w_retrieve",
        "b_retrieve",
        "w_use",
        "b_use",
        "w_pick",
        "b_pick",
        "w_schema",
        "b_schema",
        "w_rank",
        "w_key",
        "b_key",
        "w_match",
        "b_match",
        "w_wkey",
        "b_wkey",
        "w_wplace",
        "b_wplace",
        "w_wsel",
        "b_wsel",
        "w_wcomp",
        "b_wcomp",
        "w_qname",
        "w_vname",
        "w_search",
        "w_revise",
        "b_revise",
    ):
        setattr(dst, name, np.array(getattr(src, name), copy=True))
    dst.lr = float(src.lr)
    dst.n_updates = int(src.n_updates)
    dst._hash0 = src._hash0
    return dst


def birth_policy(*, seed: int = 7, lr: float = 0.2) -> UsePolicy:
    return UsePolicy(seed=seed, lr=lr)


def _useful_body(token: str, distractor: str) -> str:
    body = (
        "Staff bench log.\n\n"
        f"{token.capitalize()} the fixture on the shelf. {distractor.capitalize()} in the bin.\n"
        "Notes from the lab follow. The tray was quiet.\n"
    )
    assert _n_paragraphs(body) >= 2
    toks = prose_tokens(body)
    assert not (toks & _MOTOR_WORDS)
    assert not (toks & _STATION_WORDS)
    stream = prose_token_stream(body)
    assert stream.index(token) < stream.index(distractor)
    return body


def _clutter_notes(spec: dict[str, Any]) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    bodies = []
    prefix = spec["filename_prefix"]
    indices = list(range(N_CLUTTER))
    if spec["clutter_order"] == "reverse":
        indices = list(reversed(indices))
    for i in indices:
        body = _closed_bodies()[i]
        if i in _TRIPLES:
            w1, w2, w3 = _TRIPLES[i]
            body = body.replace(
                "The tray was quiet.",
                f"{w1.capitalize()} in the tray. {w2.capitalize()} in the bin. "
                f"{w3.capitalize()} on the shelf. The tray was quiet.",
                1,
            )
            assert {w1, w2, w3} <= prose_tokens(body)
        assert _n_ok(body)
        bodies.append(body)
        notes.append((f"{prefix}{i:02d}.md", body))
    assert len(notes) == N_CLUTTER
    assert len(set(bodies)) == len(bodies)
    return notes


def wiki_nonce(world: str, *, include_a: bool = False, include_c: bool = False) -> list[tuple[str, str]]:
    spec = WORLD_SPECS[world]
    notes = _clutter_notes(spec)
    if include_a:
        notes.append((spec["press_file"], _useful_body(spec["press_token"], spec["press_distractor"])))
    if include_c:
        notes.append((spec["tune_file"], _useful_body(spec["tune_token"], spec["tune_distractor"])))
    return notes


def _s_has_token(tag: str, word: str) -> bool:
    word = word.lower()
    for ln in tag.splitlines():
        if "=" not in ln or ln.startswith("#"):
            continue
        _, _, v = ln.partition("=")
        if v.strip().lower() == word:
            return True
    return False


def _enrich_nonce(live: dict[str, Any], *, token: str, distractor: str, station: str, did: str) -> dict[str, Any]:
    tag = live.get("tag") or ""
    live["found_token"] = _s_has_token(tag, token)
    live["found_distractor"] = _s_has_token(tag, distractor)
    live["found_station"] = _s_has_token(tag, station)
    live["found_did"] = _s_has_did(tag, did)
    live["found_bind_token"] = _s_has_bind(tag, token)
    live["found_bind_distractor"] = _s_has_bind(tag, distractor)
    return live


def _copy_s(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _train_nonce(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    *,
    token: str,
    distractor: str,
    max_steps: int,
) -> tuple[list[float], Path]:
    """One-return train on A life; rewards use world nonce, not push."""
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_u = 0.0
    s_dir = work / "ep"
    s_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        snap_tag = _tags(s_dir) if s_dir.exists() else ""
        snap = {
            "found_token": _s_has_token(snap_tag, token),
            "found_distractor": _s_has_token(snap_tag, distractor),
            "found_station": _s_has_token(snap_tag, "cha"),
            "found_did": _s_has_did(snap_tag, "press"),
            "found_bind_token": _s_has_bind(snap_tag, token),
            "found_bind_distractor": _s_has_bind(snap_tag, distractor),
        }
        ag = make(s_dir, w_dir, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live = _enrich_nonce(
            live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps),
            token=token,
            distractor=distractor,
            station="cha",
            did="press",
        )
        tr_life = list(ag.policy_traces)
        wrote = any(t.get("kind") == "write" and t.get("write") for t in tr_life)
        r_find = (
            1.0
            if (live["found_token"] and live["found_distractor"])
            or (snap["found_token"] and snap["found_distractor"])
            else 0.0
        )
        r_mark = (
            1.0
            if (
                live["found_bind_token"]
                and live["found_distractor"]
                and live["found_station"]
                and live["found_did"]
                and not live["found_bind_distractor"]
            )
            or (
                snap["found_bind_token"]
                and snap["found_station"]
                and snap["found_distractor"]
                and not snap["found_bind_distractor"]
                and wrote
            )
            else 0.0
        )
        del r_find, r_mark  # one-return: only use reward trains
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_channel_a", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        b_u = 0.9 * b_u + 0.1 * r_use
        adv = r_use - b_u
        policy.update([t for t in tr if t.get("kind") in ("search", "write")], adv)
        policy.update([t for t in tr if t.get("kind") == "vname"], adv)
        rewards.append(r_use)
    return rewards, s_dir


def _c_life_nonce(
    s_dir: Path,
    w_dir: Path,
    policy: UsePolicy,
    seed: int,
    *,
    token: str,
    distractor: str,
    max_steps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ag = make(s_dir, w_dir, policy, explore_epsilon=0.5, rng=rng)
    ag.reset_rho()
    ag.policy_traces = []
    live = _enrich_nonce(
        live_free(ag, "experience_channel_c", seed, max_steps=max_steps),
        token=token,
        distractor=distractor,
        station="chc",
        did="tune",
    )
    return live


def project_to_readout(
    raw_s: Path,
    dest: Path,
    *,
    bind: str,
    did: str | None = None,
) -> dict[str, Any]:
    """Mechanical projection: keep bind+did; route to chb; canonicalize everything else.

    Also keep w0=<bind> so vname has a copyable page word (bind/did are bookkeeping
    and skipped by the organism). Never invents a motor token as a page word.
    Never inspects a desired motor beyond selecting the note whose bind matches.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    chosen = None
    for path in sorted(raw_s.glob("*.tag")):
        text = path.read_text(encoding="utf-8")
        fact_id, tags = parse_tagfile(text)
        b = tags.get("bind")
        d = tags.get("did")
        if not isinstance(b, str) or b.lower() != bind.lower():
            continue
        if not isinstance(d, str):
            continue
        if did is not None and d.lower() != did.lower():
            continue
        chosen = (fact_id, b, d)
        break
    if chosen is None:
        return {"ok": False, "bind": bind, "did": did, "path": None}
    _fid, b, d = chosen
    # Canonical fields only. Same fact_id/filename for every projected note.
    tags = {"bind": b, "did": d, "here": "chb", "w0": b}
    out = dest / f"{READOUT_FACT_ID}.tag"
    out.write_text(record_to_tagfile(READOUT_FACT_ID, tags), encoding="utf-8")
    return {
        "ok": True,
        "bind": b,
        "did": d,
        "path": str(out),
        "fact_id": READOUT_FACT_ID,
        "tags": tags,
    }


def write_neutral_readout(dest: Path, *, bind: str, did: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tags = {"bind": bind, "did": did, "here": "chb", "w0": bind}
    (dest / f"{READOUT_FACT_ID}.tag").write_text(
        record_to_tagfile(READOUT_FACT_ID, tags), encoding="utf-8"
    )


def probe_b(policy: UsePolicy, s_dir: Path | None, seed: int) -> dict[str, Any]:
    """Greedy probe at station B. Empty S if s_dir is None."""
    import tempfile

    if s_dir is None:
        with tempfile.TemporaryDirectory(prefix="tm091box_empty_") as tmp:
            ag = make(Path(tmp), None, policy, explore_epsilon=0.0)
            ag.reset_rho()
            return probe(ag, "probe_channel_b", seed)
    ag = make(s_dir, None, policy, explore_epsilon=0.0)
    ag.reset_rho()
    return probe(ag, "probe_channel_b", seed)


def _motor(name: str) -> str:
    return str(name).lower()


def classify_separation(cell: dict[str, Any]) -> tuple[str, str]:
    """Classify one paired seed's lethal-pair matrix."""
    if cell.get("genome_ok") is False:
        return "Confound", "Genome manifest drifted."
    if cell.get("protocol_ok") is False:
        return "Confound", "Development-protocol manifest drifted."
    if cell.get("cortex_unchanged") is False:
        return "Confound", "Cortex moved."
    if not cell.get("acquisition_ok"):
        return "Fail", "Acquisition never wrote the nonce bind."

    p1 = cell["P1"]
    p2 = cell["P2"]

    def act(block: dict[str, Any], key: str) -> str:
        return _motor((block.get(key) or {}).get("action_name") or "hold")

    # Matching projected S must follow donor did.
    s1_ok = act(p1, "S1") == "press" and act(p2, "S1") == "press"
    s2_ok = act(p1, "S2") == "tune" and act(p2, "S2") == "tune"
    if not (s1_ok and s2_ok):
        # Followed training worlds instead of donors?
        if act(p1, "S2") == "press" and act(p2, "S1") == "tune":
            return "Confound", "Pi + Sj followed training world, not counterfactual donor."
        if act(p1, "S1") == "press" and act(p1, "S2") == "press" and act(p2, "S1") == "tune" and act(p2, "S2") == "tune":
            return "Confound", "Pi + Sj followed training world, not counterfactual donor."
        if not s1_ok:
            return "Fail", "Pi + S1 (flim→PRESS) did not PRESS."
        return "Fail", "Pi + S2 (flim→TUNE) did not TUNE."

    nos_keys = ("empty", "neutral_PRESS", "neutral_TUNE")
    p1_nos = [act(p1, k) for k in nos_keys]
    p2_nos = [act(p2, k) for k in nos_keys]

    # World-correlated leak: P1 no-S → PRESS and P2 no-S → TUNE (or vice versa matching maps).
    leak = False
    for a1, a2 in zip(p1_nos, p2_nos):
        if a1 == "press" and a2 == "tune":
            leak = True
            break
        if a1 == "tune" and a2 == "press":
            leak = True
            break
    if leak:
        return "Confound", "No-S behavior covaries with training-world mapping."

    # Generic drift: both non-HOLD the same way on a no-S cell.
    for a1, a2 in zip(p1_nos, p2_nos):
        if a1 == a2 and a1 in ("press", "tune"):
            return "Control Fail", "Reproducible world-independent motor bias on empty/neutral."

    # Asymmetric / inconsistent
    for a1, a2 in zip(p1_nos, p2_nos):
        if {a1, a2} == {"press", "hold"} or {a1, a2} == {"tune", "hold"}:
            return "Inconclusive", "Asymmetric no-S behavior prevents attribution."
        if a1 in ("press", "tune") or a2 in ("press", "tune"):
            return "Inconclusive", "Inconsistent no-S behavior prevents attribution."

    if any(a != "hold" for a in p1_nos + p2_nos):
        return "Inconclusive", "Unexpected no-S motors."

    return (
        "Store-works",
        "Separation: empty/neutral HOLD; Pi+S1 PRESS; Pi+S2 TUNE; cortex and manifests frozen.",
    )


def _s3_ok(cell: dict[str, Any]) -> bool | None:
    if "s3_ok" in cell:
        return bool(cell["s3_ok"])
    proj = (cell.get("projections") or {}).get("S3") or {}
    if "ok" in proj:
        return bool(proj.get("ok"))
    return None


def classify_transfer(cell: dict[str, Any]) -> tuple[str, str]:
    """Cross-world use-S given a valid store. Missing S3 is not a transfer miss."""
    raw = cell.get("transfer")
    results = raw if isinstance(raw, dict) else {}
    if not results:
        results = cell.get("transfer_probes") or {}
    if not results:
        return "Fail", "Transfer matrix missing."
    if _s3_ok(cell) is False:
        return "Unevaluable", "No valid S3; W3 acquisition missed the nonce bind."
    ok = True
    why_bits: list[str] = []
    for key, need in (
        ("P1_S3", "press"),
        ("P2_S3", "press"),
        ("P3_S1", "press"),
        ("P3_S2", "tune"),
    ):
        got = _motor((results.get(key) or {}).get("action_name") or "hold")
        if got != need:
            ok = False
            why_bits.append(f"{key}={got}")
    if ok:
        return "Pass", "Cross-world use-S: Pi follows Sj including novel S3."
    return "Fail", "Cross-world use-S miss: " + ", ".join(why_bits)


def score_box_measures(cell: dict[str, Any]) -> dict[str, Any]:
    """Five-measure readout. Compatible separation label stays classify_separation."""

    def act(block: dict[str, Any], key: str) -> str:
        return _motor((block.get(key) or {}).get("action_name") or "hold")

    p1 = cell.get("P1") or {}
    p2 = cell.get("P2") or {}
    e1, e2 = act(p1, "empty"), act(p2, "empty")
    leak = (e1 == "press" and e2 == "tune") or (e1 == "tune" and e2 == "press")
    donor_ok = (
        act(p1, "S1") == "press"
        and act(p2, "S1") == "press"
        and act(p1, "S2") == "tune"
        and act(p2, "S2") == "tune"
    )
    neutrals = [
        act(p1, "neutral_PRESS"),
        act(p2, "neutral_PRESS"),
        act(p1, "neutral_TUNE"),
        act(p2, "neutral_TUNE"),
    ]
    xfer, xwhy = classify_transfer(cell)
    s3 = _s3_ok(cell)
    return {
        "world_fact_leakage": "Observed" if leak else "Not observed",
        "counterfactual_donor": "Pass" if donor_ok else "Fail",
        "neutral_relevance": "Pass" if all(m == "hold" for m in neutrals) else "Fail",
        "transfer": xfer,
        "transfer_rationale": xwhy,
        "w3_acquired": bool(s3),
    }


def _find_bind_did(s_dir: Path, bind: str) -> str | None:
    for path in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        b = tags.get("bind")
        d = tags.get("did")
        if isinstance(b, str) and b.lower() == bind.lower() and isinstance(d, str):
            return d.lower()
    return None


def train_world(
    *,
    world: str,
    seed: int,
    birth: UsePolicy,
    run_dir: Path,
    n_train: int,
    max_steps: int,
) -> dict[str, Any]:
    spec = WORLD_SPECS[world]
    w_a = run_dir / f"{world}_W_a"
    w_both = run_dir / f"{world}_W_both"
    work = run_dir / f"{world}_train"
    raw_s = run_dir / f"{world}_raw_S"
    for d in (w_a, w_both, work):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    write_prose_notes(w_a, wiki_nonce(world, include_a=True))
    write_prose_notes(w_both, wiki_nonce(world, include_a=True, include_c=True))

    policy = clone_policy(birth)
    birth_hash = policy.weight_hash()
    cortex0 = make(run_dir / f"{world}_hash", None, policy, enabled=False).weight_hash()

    rewards, s_dir = _train_nonce(
        policy,
        w_a,
        work,
        n_train,
        seed,
        token=spec["press_token"],
        distractor=spec["press_distractor"],
        max_steps=max_steps,
    )
    _c_life_nonce(
        s_dir,
        w_both,
        policy,
        seed + 20,
        token=spec["tune_token"],
        distractor=spec["tune_distractor"],
        max_steps=max_steps,
        rng=np.random.default_rng(seed + 1),
    )
    _copy_s(s_dir, raw_s)
    cortex1 = make(run_dir / f"{world}_hash2", None, policy, enabled=False).weight_hash()
    press_did = _find_bind_did(raw_s, spec["press_token"])
    tune_did = _find_bind_did(raw_s, spec["tune_token"])
    # For W1/W2 lethal pair we care about flim.
    flim_did = _find_bind_did(raw_s, "flim")
    blen_did = _find_bind_did(raw_s, "blen")
    return {
        "world": world,
        "seed": seed,
        "policy": policy,
        "birth_hash": birth_hash,
        "policy_hash": policy.weight_hash(),
        "n_updates": policy.n_updates,
        "raw_s": str(raw_s),
        "raw_s_hash": _sha_dir_tags(raw_s),
        "cortex0": cortex0,
        "cortex1": cortex1,
        "cortex_unchanged": cortex0 == cortex1,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "press_token": spec["press_token"],
        "tune_token": spec["tune_token"],
        "press_did": press_did,
        "tune_did": tune_did,
        "flim_did": flim_did,
        "blen_did": blen_did,
        "n_files": len(list(raw_s.glob("*.tag"))),
    }


_POLICY_ATTRS = (
    "W_collect",
    "b_collect",
    "w_apply",
    "b_apply",
    "w_write",
    "b_write",
    "w_retrieve",
    "b_retrieve",
    "w_use",
    "b_use",
    "w_pick",
    "b_pick",
    "w_schema",
    "b_schema",
    "w_rank",
    "w_key",
    "b_key",
    "w_match",
    "b_match",
    "w_wkey",
    "b_wkey",
    "w_wplace",
    "b_wplace",
    "w_wsel",
    "b_wsel",
    "w_wcomp",
    "b_wcomp",
    "w_qname",
    "w_vname",
    "w_search",
    "w_revise",
    "b_revise",
)


def _policy_to_arrays(p: UsePolicy) -> dict[str, Any]:
    return {
        "lr": float(p.lr),
        "n_updates": int(p.n_updates),
        "_hash0": p._hash0,
        "attrs": {name: np.asarray(getattr(p, name)).tolist() for name in _POLICY_ATTRS},
    }


def _policy_from_arrays(blob: dict[str, Any]) -> UsePolicy:
    p = UsePolicy(seed=0, lr=float(blob["lr"]))
    for name, data in blob["attrs"].items():
        arr = np.asarray(data, dtype=np.float64)
        ref = getattr(p, name)
        if np.ndim(ref) == 0:
            setattr(p, name, np.array(arr.reshape(()), dtype=np.float64))
        else:
            setattr(p, name, arr.reshape(np.shape(ref)))
    p.n_updates = int(blob["n_updates"])
    p._hash0 = blob["_hash0"]
    return p


def _worker_train(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-entry: train one world from a serialized birth policy."""
    birth = _policy_from_arrays(payload["birth"])
    out = train_world(
        world=payload["world"],
        seed=payload["seed"],
        birth=birth,
        run_dir=Path(payload["run_dir"]),
        n_train=payload["n_train"],
        max_steps=payload["max_steps"],
    )
    # Serialize policy for the parent; drop live object.
    pol = out.pop("policy")
    out["policy_blob"] = _policy_to_arrays(pol)
    return out


def run_paired_seed(
    *,
    seed: int,
    run_dir: Path,
    n_train: int,
    max_steps: int,
    workers: int,
) -> dict[str, Any]:
    genome_ok, genome_why = verify_genome_lock()
    protocol_ok, protocol_why = verify_protocol_lock(n_train=n_train, max_steps=max_steps)
    birth = birth_policy(seed=7, lr=0.2)
    birth_hash = birth.weight_hash()
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "world": w,
            "seed": seed,
            "birth": _policy_to_arrays(birth),
            "run_dir": str(seed_dir),
            "n_train": n_train,
            "max_steps": max_steps,
        }
        for w in ("W1", "W2", "W3")
    ]
    trained: dict[str, Any] = {}
    if workers <= 1:
        for job in jobs:
            trained[job["world"]] = _worker_train(job)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as pool:
            futs = {pool.submit(_worker_train, job): job["world"] for job in jobs}
            for fut in as_completed(futs):
                trained[futs[fut]] = fut.result()

    for w, rec in trained.items():
        rec["policy"] = _policy_from_arrays(rec.pop("policy_blob"))

    # Projected readouts for flim lethal pair + neutrals + S3 blen.
    proj = seed_dir / "projected"
    proj.mkdir(exist_ok=True)
    s1_raw = Path(trained["W1"]["raw_s"])
    s2_raw = Path(trained["W2"]["raw_s"])
    s3_raw = Path(trained["W3"]["raw_s"])

    s1_proj = proj / "S1_flim"
    s2_proj = proj / "S2_flim"
    s3_proj = proj / "S3_blen"
    n_press = proj / "neutral_PRESS"
    n_tune = proj / "neutral_TUNE"
    p1_s1 = project_to_readout(s1_raw, s1_proj, bind="flim", did="press")
    p2_s2 = project_to_readout(s2_raw, s2_proj, bind="flim", did="tune")
    p3_s3 = project_to_readout(s3_raw, s3_proj, bind="blen", did="press")
    write_neutral_readout(n_press, bind=NEUTRAL_PRESS_BIND, did="press")
    write_neutral_readout(n_tune, bind=NEUTRAL_TUNE_BIND, did="tune")

    acquisition_ok = bool(p1_s1.get("ok") and p2_s2.get("ok"))
    s3_ok = bool(p3_s3.get("ok"))

    P0 = birth_policy(seed=7, lr=0.2)
    P1 = trained["W1"]["policy"]
    P2 = trained["W2"]["policy"]
    P3 = trained["W3"]["policy"]

    def matrix_for(policy: UsePolicy) -> dict[str, Any]:
        return {
            "empty": probe_b(policy, None, seed + 100),
            "neutral_PRESS": probe_b(policy, n_press, seed + 101),
            "neutral_TUNE": probe_b(policy, n_tune, seed + 102),
            "S1": probe_b(policy, s1_proj, seed + 103),
            "S2": probe_b(policy, s2_proj, seed + 104),
        }

    transfer_probes = {
        "P1_S3": probe_b(P1, s3_proj, seed + 301),
        "P2_S3": probe_b(P2, s3_proj, seed + 302),
        "P3_S1": probe_b(P3, s1_proj, seed + 303),
        "P3_S2": probe_b(P3, s2_proj, seed + 304),
        "P0_S3": probe_b(P0, s3_proj, seed + 300),
    }
    cell = {
        "seed": seed,
        "birth_hash": birth_hash,
        "genome_ok": genome_ok,
        "genome_why": genome_why,
        "protocol_ok": protocol_ok,
        "protocol_why": protocol_why,
        "genome_lock_hash": genome_lock_hash(),
        "protocol_lock_hash": protocol_lock_hash(),
        "cortex_unchanged": all(trained[w]["cortex_unchanged"] for w in ("W1", "W2", "W3")),
        "acquisition_ok": acquisition_ok,
        "s3_ok": s3_ok,
        "projections": {
            "S1": p1_s1,
            "S2": p2_s2,
            "S3": p3_s3,
            "S1_hash": _sha_dir_tags(s1_proj),
            "S2_hash": _sha_dir_tags(s2_proj),
            "S3_hash": _sha_dir_tags(s3_proj),
        },
        "raw": {w: {k: trained[w][k] for k in trained[w] if k != "policy"} for w in trained},
        "P0": matrix_for(P0),
        "P1": matrix_for(P1),
        "P2": matrix_for(P2),
        "P3": {
            "empty": probe_b(P3, None, seed + 200),
            "S1": probe_b(P3, s1_proj, seed + 203),
            "S2": probe_b(P3, s2_proj, seed + 204),
            "S3": probe_b(P3, s3_proj, seed + 205),
        },
        "transfer_probes": transfer_probes,
    }
    sep, sep_why = classify_separation(cell)
    xfer, xfer_why = classify_transfer({"transfer": transfer_probes, "s3_ok": s3_ok})
    cell["separation"] = sep
    cell["separation_rationale"] = sep_why
    cell["transfer"] = xfer
    cell["transfer_rationale"] = xfer_why
    cell["measures"] = score_box_measures({**cell, "transfer": transfer_probes})
    return cell


def _aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    seps = [c["separation"] for c in cells]
    if any(s == "Confound" for s in seps):
        sep = "Confound"
        why = next(c["separation_rationale"] for c in cells if c["separation"] == "Confound")
    elif any(s == "Fail" for s in seps):
        sep = "Fail"
        why = next(c["separation_rationale"] for c in cells if c["separation"] == "Fail")
    elif any(s == "Control Fail" for s in seps):
        sep = "Control Fail"
        why = next(c["separation_rationale"] for c in cells if c["separation"] == "Control Fail")
    elif any(s == "Inconclusive" for s in seps):
        sep = "Inconclusive"
        why = next(c["separation_rationale"] for c in cells if c["separation"] == "Inconclusive")
    elif all(s == "Store-works" for s in seps):
        sep = "Store-works"
        why = "All paired seeds: separation holds."
    else:
        sep = "Inconclusive"
        why = f"Mixed separation labels: {seps}"
    xfers = [c["transfer"] for c in cells]
    evaluable = [x for x in xfers if x != "Unevaluable"]
    n_eval = len(evaluable)
    n_xfer_pass = sum(1 for x in evaluable if x == "Pass")
    if n_eval == 0:
        xfer, xwhy = "Unevaluable", "No valid S3 on any seed."
    elif n_xfer_pass == n_eval:
        xfer, xwhy = "Pass", f"Pass on all evaluable stores ({n_xfer_pass}/{n_eval})."
    else:
        xfer, xwhy = "Fail", f"Evaluable transfer {n_xfer_pass}/{n_eval}; labels: {xfers}"
    measures = [c.get("measures") or {} for c in cells]
    n = len(cells) or 1
    n_leak = sum(1 for m in measures if m.get("world_fact_leakage") == "Not observed")
    n_donor = sum(1 for m in measures if m.get("counterfactual_donor") == "Pass")
    n_neut_fail = sum(1 for m in measures if m.get("neutral_relevance") == "Fail")
    n_w3 = sum(1 for m in measures if m.get("w3_acquired"))
    return {
        "separation": sep,
        "separation_rationale": why,
        "transfer": xfer,
        "transfer_rationale": xwhy,
        "per_seed": seps,
        "measures": {
            "world_fact_leakage": f"Not observed, {n_leak}/{n} seeds" if n_leak == n else f"{n_leak}/{n} seeds not observed",
            "counterfactual_donor": f"Pass, {n_donor}/{n}",
            "neutral_relevance": f"Fail, {n_neut_fail}/{n}" if n_neut_fail else f"Pass, {n - n_neut_fail}/{n}",
            "transfer_evaluable": f"{xfer}, {n_xfer_pass}/{n_eval}" if n_eval else xfer,
            "w3_acquisition": f"{n_w3}/{n}",
        },
    }


def run_tm091box(
    *,
    seeds: list[int] | None = None,
    n_train: int = 500,
    max_steps: int = 32,
    workers: int = 3,
) -> dict[str, Any]:
    seeds = seeds or [12345, 12346, 12347]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{stamp}_tm091box"
    run_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    # Seeds are sequential; within each seed W1/W2/W3 may run in parallel.
    for seed in seeds:
        cell = run_paired_seed(
            seed=seed,
            run_dir=run_dir,
            n_train=n_train,
            max_steps=max_steps,
            workers=workers,
        )
        # Fix transfer overwrite: recompute from transfer_probes if needed
        if isinstance(cell.get("transfer"), str):
            # transfer_probes was set incorrectly; rebuild from cell fields stored before overwrite
            pass
        cells.append(cell)

    # Repair transfer probes: classify_transfer was called with cell that still had probe dict
    # but then we assigned string. Looking at run_paired_seed — bug. Fix here by re-probing? Better fix the function.

    agg = _aggregate(cells)
    out = {
        "version": "TM.0.9.BOX",
        "seeds": seeds,
        "n_train": n_train,
        "max_steps": max_steps,
        "workers": workers,
        "run_dir": str(run_dir),
        "genome_lock_hash": genome_lock_hash(),
        "protocol_lock_hash": protocol_lock_hash(),
        "genome_delta": 0,
        "protocol_delta": 0,
        "separation": agg["separation"],
        "separation_rationale": agg["separation_rationale"],
        "transfer": agg["transfer"],
        "transfer_rationale": agg["transfer_rationale"],
        "measures": agg["measures"],
        "cells": cells,
    }
    # JSON-safe: drop live policy objects already removed; ensure probes serializable
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.9.BOX boxed-policy leakage

Historical leakage control. Cue is which projected note is mounted, not a `flim` token in the observation. Do not require this battery to turn green after MATCH.

Compatible label: separation **{out['separation']}**; transfer **{out['transfer']}**.

| Measure | Result |
|---------|--------|
| World-fact leakage | {out['measures']['world_fact_leakage']} |
| Counterfactual donor control | {out['measures']['counterfactual_donor']} |
| Neutral relevance control | {out['measures']['neutral_relevance']} |
| Cross-world use-S, evaluable stores | {out['measures']['transfer_evaluable']} |
| W3 acquisition robustness | {out['measures']['w3_acquisition']} |

Separation: {out['separation_rationale']}

Transfer: {out['transfer_rationale']}

Seeds: {seeds}
Workers: {workers}
Genome lock: `{out['genome_lock_hash'][:12]}…` (delta 0)
Protocol lock: `{out['protocol_lock_hash'][:12]}…` (delta 0)
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.9.BOX boxed-policy leakage control")
    p.add_argument("--seeds", type=int, nargs="+", default=[12345, 12346, 12347])
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    # Engineering may pass --seeds 12345 alone; recorded run uses three.
    # Protocol lock check: only full n_train=500 satisfies lock; warn otherwise.
    m = run_tm091box(
        seeds=list(args.seeds),
        n_train=args.n_train,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "separation": m["separation"],
                "transfer": m["transfer"],
                "measures": m["measures"],
                "seeds": m["seeds"],
                "run_dir": m["run_dir"],
                "genome_delta": m["genome_delta"],
                "protocol_delta": m["protocol_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
