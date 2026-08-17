"""TM.0.24.LINEAGE runner — generator, observable teacher, unscored ES smoke.

Capability scoring is forbidden until lineage_engine.candidate.lock exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import build_observe, make_cortex, physics, torch_env
from three_memory.cortex_lineage import (
    AdamState,
    adam_step,
    antithetic_children,
    apply_arm_c_theta,
    cluster_bootstrap_lower,
    defaults_theta,
    f_search,
    freeze_plasticity,
    g_k,
    layout_sha,
    load_layout,
    pack_arm_c_from_cortex,
    rank_centered,
    refuse_audit,
    sample_birth_from_arm_d,
    sha_file,
)
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_PREREG = REPO_ROOT / "docs" / "lineage_world_generator.prereg.lock"
GEN_LOCK = REPO_ROOT / "docs" / "lineage_world_generator.lock"
THIS = Path(__file__).resolve()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def opaque_spelling(rng: np.random.Generator, prefix: str) -> str:
    return f"{prefix}_{int(rng.integers(0, 1_000_000_000)):09d}"


def make_synthetic_world(seed: int, *, teacher_convention: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    handles = [opaque_spelling(rng, "h") for _ in range(4)]
    symbols = [opaque_spelling(rng, "s") for _ in range(6)]
    beneficial = int(rng.integers(0, 2))
    a, b = handles[0], handles[1]
    if beneficial == 1:
        a, b = b, a
    effects = {
        a: {"state": ["st_p"], "delta": [0.25, -0.1, 0.15, 0.0]},
        b: {"state": ["st_h"], "delta": [-0.35, 0.45, -0.15, 0.0]},
        handles[2]: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
        handles[3]: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
    }
    # Public convention: which two symbols the teacher pairs with actuators.
    if teacher_convention % 2 == 0:
        pair = (symbols[0], symbols[1])
    else:
        pair = (symbols[1], symbols[0])
    return {
        "domain": "TM024.LINEAGE.SYNTH.",
        "seed": int(seed),
        "handles": handles,
        "symbols": symbols,
        "teacher_convention": int(teacher_convention),
        "teacher_pair": list(pair),
        "latent": {"act_effects": effects},
        "beneficial": a,
        "timeout": 3,
    }


class ObservableTeacher:
    """Deterministic given visible history + teacher RNG. Never reads scorers."""

    def __init__(self, world: dict[str, Any], rng_seed: int):
        self.world = world
        self.rng = np.random.default_rng(int(rng_seed))
        self.history: list[dict[str, Any]] = []
        self.hold_streak = 0
        self.fail_streak = 0

    def note(self, visible: dict[str, Any]) -> None:
        self.history.append(dict(visible))
        op = str(visible.get("op") or "")
        if op == "HOLD":
            self.hold_streak += 1
        else:
            self.hold_streak = 0
        if visible.get("incompatible"):
            self.fail_streak += 1
        else:
            self.fail_streak = 0

    def next_symbols(self) -> list[str]:
        pair = list(self.world["teacher_pair"])
        simplify = self.hold_streak >= self.world["timeout"] or self.fail_streak >= 2
        if simplify:
            return [pair[0]]
        if self.hold_streak > 0:
            return list(pair)
        return list(pair) if self.rng.random() < 0.7 else [pair[int(self.rng.integers(0, 2))]]


def teacher_audit_identical_histories() -> bool:
    w = make_synthetic_world(7, teacher_convention=0)
    vis = [{"op": "HOLD", "incompatible": False} for _ in range(4)]
    t1 = ObservableTeacher(w, 99)
    t2 = ObservableTeacher(w, 99)
    s1, s2 = [], []
    for v in vis:
        scorer_only = {"correct": True, "stage": "L0"}  # must not be passed
        t1.note(v)
        t2.note(v)
        assert "correct" not in t1.history[-1]
        _ = scorer_only
        s1.append(t1.next_symbols())
        s2.append(t2.next_symbols())
    return s1 == s2


def live_once(
    ag: NeuralCortex,
    world: dict[str, Any],
    *,
    n_wake: int,
    n_replay: int,
    teacher_seed: int,
) -> dict[str, Any]:
    ag.bind_actuators(list(world["handles"]))
    teacher = ObservableTeacher(world, teacher_seed)
    body = [1.0, 0.0, 1.0, 0.0]
    state = ["st_idle"]
    latent = world["latent"]
    ops: dict[str, int] = {}
    energies: list[float] = []
    for i in range(int(n_wake)):
        syms = teacher.next_symbols()
        out = ag.observe(
            build_observe(
                interaction_token=f"w{world['seed']}_{i}",
                source_token="src_teacher",
                ordered_symbols=syms,
                observable_state=state,
                body_state=body,
            )
        )
        action = out.get("action") or {}
        op = str(action.get("op") or "HOLD")
        ops[op] = ops.get(op, 0) + 1
        incompatible = False
        if op == "ACT":
            tok = action.get("token")
            if tok not in (world["latent"]["act_effects"]):
                incompatible = True
            state, body = physics(body, tok, latent)
        teacher.note({"op": op, "incompatible": incompatible, "symbols": list(syms)})
        energies.append(float(body[0]))
    rest = ag.rest_epoch(int(n_replay), body=np.asarray(body, dtype=np.float64))
    hold_rate = ops.get("HOLD", 0) / max(sum(ops.values()), 1)
    return {
        "ops": ops,
        "hold_rate": hold_rate,
        "energy_last": energies[-1] if energies else 0.0,
        "energy_mean": float(np.mean(energies)) if energies else 0.0,
        "rest": {"n": rest.get("n"), "ops": rest.get("op_counts")},
        "dev_epoch": int(getattr(ag, "dev_epoch", 0)),
        "n_s": len(ag.memory.records()),
    }


def probe_beneficial(ag: NeuralCortex, world: dict[str, Any], *, n_probe: int = 20) -> float:
    """Adult L0 unit: fraction of probe ticks that ACT the beneficial handle. No physics."""
    beneficial = world["beneficial"]
    hits = 0
    body = [1.0, 0.0, 1.0, 0.0]
    state = ["st_idle"]
    pair = list(world["teacher_pair"])
    for i in range(int(n_probe)):
        out = ag.observe(
            build_observe(
                interaction_token=f"p{world['seed']}_{i}",
                source_token="src_probe",
                ordered_symbols=pair,
                observable_state=state,
                body_state=body,
            )
        )
        tok = (out.get("action") or {}).get("token")
        if tok == beneficial:
            hits += 1
    return hits / max(int(n_probe), 1)


def viability(summary: dict[str, Any]) -> bool:
    if summary["hold_rate"] >= 0.98:
        return False
    return True


def evaluate_theta(
    theta: np.ndarray,
    arm: str,
    *,
    world_seeds: list[int],
    birth_seeds: list[int],
    teacher_seed: int,
    n_wake: int,
    n_replay: int,
    device: str,
    plasticity: bool,
    scored: bool = False,
    n_probe: int = 20,
) -> dict[str, Any]:
    adults: list[float] = []
    holds: list[float] = []
    cells: list[tuple[int, int, float]] = []
    for ws in world_seeds:
        world = make_synthetic_world(ws)
        for bs in birth_seeds:
            with tempfile.TemporaryDirectory(prefix="lin_") as tmp:
                if arm == "D":
                    ag = sample_birth_from_arm_d(theta, life_seed=bs, s_dir=Path(tmp) / "s", device=device)
                else:
                    ag = apply_arm_c_theta(theta, life_seed=bs, s_dir=Path(tmp) / "s", device=device)
                if not plasticity:
                    freeze_plasticity(ag)
                summary = live_once(
                    ag, world, n_wake=n_wake, n_replay=n_replay, teacher_seed=teacher_seed ^ ws ^ bs
                )
                if not viability(summary):
                    adult = 0.0
                elif scored:
                    adult = float(probe_beneficial(ag, world, n_probe=n_probe))
                else:
                    adult = float(summary["energy_mean"])
                adults.append(adult)
                holds.append(float(summary["hold_rate"]))
                cells.append((int(ws), int(bs), adult))
    rob = float(np.std(adults)) if adults else 1.0
    eff = float(np.mean(holds)) if holds else 1.0
    return {
        "adults": adults,
        "cells": cells,
        "f": f_search(adults, rob, eff),
        "robustness": rob,
        "efficiency": eff,
    }


def es_step(
    theta: np.ndarray,
    arm: str,
    *,
    n_pairs: int,
    sigma: float,
    mut_seed: int,
    world_seeds: list[int],
    birth_seeds: list[int],
    teacher_seed: int,
    n_wake: int,
    n_replay: int,
    device: str,
    adam: AdamState,
    scored: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(mut_seed))
    dim = theta.size
    fits: list[float] = []
    noises: list[np.ndarray] = []
    for i in range(int(n_pairs)):
        eps = rng.normal(0.0, 1.0, size=dim)
        plus, minus = antithetic_children(theta, eps, sigma)
        # identical world/birth/teacher seeds for the pair
        fp = evaluate_theta(
            plus, arm,
            world_seeds=world_seeds, birth_seeds=birth_seeds, teacher_seed=teacher_seed,
            n_wake=n_wake, n_replay=n_replay, device=device, plasticity=True, scored=scored,
        )
        fm = evaluate_theta(
            minus, arm,
            world_seeds=world_seeds, birth_seeds=birth_seeds, teacher_seed=teacher_seed,
            n_wake=n_wake, n_replay=n_replay, device=device, plasticity=True, scored=scored,
        )
        # scalarize lexicographic F for ES: primary quartile only (search gradient)
        fits.extend([fp["f"][0], fm["f"][0]])
        noises.extend([eps, -eps])
    ranks = rank_centered(np.asarray(fits, dtype=np.float64))
    grad = np.zeros(dim, dtype=np.float64)
    for r, e in zip(ranks, noises, strict=True):
        grad += r * e
    grad /= max(len(noises), 1)
    grad /= max(sigma, 1e-8)
    new_theta = adam_step_clip(theta, grad, adam, arm)
    return {"theta": new_theta, "mean_f": float(np.mean(fits)), "grad_norm": float(np.linalg.norm(grad))}


def adam_step_clip(theta: np.ndarray, grad: np.ndarray, st: AdamState, arm: str) -> np.ndarray:
    nxt = adam_step(theta, grad, st)
    layout = load_layout()
    for sl in layout["arms"][arm]["slices"]:
        if sl["kind"] != "scalar":
            continue
        off = int(sl["offset"])
        nxt[off] = float(np.clip(nxt[off], float(sl["lo"]), float(sl["hi"])))
    return nxt


def write_generator_lock() -> dict[str, Any]:
    prereg = json.loads(GEN_PREREG.read_text(encoding="utf-8"))
    lock = {
        "version": "TM.0.24.LINEAGE.WORLD.GENERATOR",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "prereg": "docs/lineage_world_generator.prereg.lock",
        "prereg_sha": sha_file(GEN_PREREG),
        "generator_module": "experiments.run_tm024lineage",
        "generator_sha": sha_file(THIS),
        "teacher": "experiments.run_tm024lineage.ObservableTeacher",
        "layout_sha": layout_sha(),
        "qual_seed_commitment": prereg["qual_seed_commitment"],
        "eval_seed_commitment": prereg["eval_seed_commitment"],
        "physics": "experiments.run_tm023cortex.physics",
        "note": "Pinned after generator code exists. Unscored synthetic worlds only until engine candidate.",
    }
    GEN_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def unscored_smoke(device: str = "cpu") -> dict[str, Any]:
    layout = load_layout()
    theta = defaults_theta("D", layout)
    audit = refuse_audit(theta, "D", layout)
    assert audit["ok"], audit
    adam = AdamState(m=np.zeros_like(theta), v=np.zeros_like(theta), lr=0.02)
    step = es_step(
        theta, "D",
        n_pairs=2, sigma=0.05, mut_seed=11,
        world_seeds=[101, 202], birth_seeds=[3, 4], teacher_seed=9,
        n_wake=12, n_replay=4, device=device, adam=adam,
    )
    # Arm C pack/apply roundtrip from a v27 make_cortex
    with tempfile.TemporaryDirectory(prefix="cpack_") as tmp:
        v27 = make_cortex(Path(tmp) / "s", device=device)
        ctheta = pack_arm_c_from_cortex(v27, layout)
        ctheta[layout["arms"]["C"]["slices"][0]["offset"]] += 0.0  # keep
        ag2 = apply_arm_c_theta(ctheta, life_seed=12345, s_dir=Path(tmp) / "s2", device=device)
        live_once(ag2, make_synthetic_world(5), n_wake=4, n_replay=2, teacher_seed=1)
    conv = teacher_audit_identical_histories()
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "refuse_ok": audit["ok"],
        "es_mean_f": step["mean_f"],
        "teacher_audit": conv,
        "env": torch_env(),
        "device": device,
    }


def engine_shas() -> dict[str, str]:
    files = {
        "neural_cortex": REPO_ROOT / "three_memory" / "neural_cortex.py",
        "cortex_memory": REPO_ROOT / "three_memory" / "cortex_memory.py",
        "cortex_lineage": REPO_ROOT / "three_memory" / "cortex_lineage.py",
        "runner": THIS,
        "layout": REPO_ROOT / "docs" / "lineage_genome_layout.json",
        "generator_lock": GEN_LOCK,
        "fitness_contract": REPO_ROOT / "docs" / "lineage_fitness_contract.md",
        "architecture_contract": REPO_ROOT / "docs" / "lineage_architecture_contract.md",
        "prereg": REPO_ROOT / "docs" / "lineage.prereg.lock",
        "world_prereg": GEN_PREREG,
        "compat": REPO_ROOT / "docs" / "lineage_v27_default_compat.lock",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def write_compat_lock() -> dict[str, Any]:
    """v27-default make_cortex() C4/C5/C6. Does not score lineage genomes."""
    from experiments.cortex_mact_boundary import control_c4_v6
    from experiments.cortex_v7_stats import run_c5_population, run_c6_population

    c4 = control_c4_v6()
    c5 = run_c5_population()
    c6 = run_c6_population()
    lock = {
        "version": "TM.0.24.LINEAGE.V27_DEFAULT_COMPAT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "neural_cortex_sha": sha_file(REPO_ROOT / "three_memory" / "neural_cortex.py"),
        "cortex_memory_sha": sha_file(REPO_ROOT / "three_memory" / "cortex_memory.py"),
        "ancestor_neural_sha": "71bece5917893fae03c3a95c276cf93bc0e34fce6a7bfb6a99adf093bb7ebc08",
        "C4": {"ok": bool(c4.get("ok")), "why": c4.get("why"), "id": c4.get("id")},
        "C5": {
            "ok": bool(c5.get("ok")),
            "n_trained_beats_frozen": c5.get("n_trained_beats_frozen"),
            "mean_delta": c5.get("mean_delta"),
            "id": c5.get("id"),
        },
        "C6": {"ok": bool(c6.get("ok")), "why": c6.get("why"), "id": c6.get("id")},
        "phrase_replay_absent": True,
        "note": "make_cortex() with lineage_params=None. REST not invoked. Age scales default 1.0.",
    }
    path = REPO_ROOT / "docs" / "lineage_v27_default_compat.lock"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    if not (c4.get("ok") and c5.get("ok") and c6.get("ok")):
        raise RuntimeError(f"v27-default C4/C5/C6 not green: {lock}")
    return lock


def dev_stream_seed() -> bytes:
    prereg = json.loads((REPO_ROOT / "docs" / "lineage.prereg.lock").read_text(encoding="utf-8"))
    return bytes.fromhex(prereg["dev_stream_seed_hex"])


def next_dev_triplet(index: int, *, n_worlds: int = 2) -> dict[str, Any]:
    raw = hashlib.sha256(dev_stream_seed() + int(index).to_bytes(8, "big")).digest()
    rng = np.random.default_rng(int.from_bytes(raw[:8], "big"))
    used: set[int] = set()

    def take() -> list[int]:
        out: list[int] = []
        while len(out) < n_worlds:
            s = int(rng.integers(1, 2_000_000_000))
            if s not in used:
                used.add(s)
                out.append(s)
        return out

    return {"A": take(), "B": take(), "C": take(), "index": int(index)}


def causal_scores(
    theta: np.ndarray,
    arm: str,
    *,
    world_seed: int,
    birth_seed: int,
    teacher_seed: int,
    n_wake: int,
    n_replay: int,
    device: str,
    n_probe: int = 20,
) -> dict[str, float]:
    """Plasticity-on/off and birth share the same life_seed, world, and teacher seeds."""
    world = make_synthetic_world(world_seed)
    ts = teacher_seed ^ world_seed ^ birth_seed

    def one(*, plasticity: bool, teach: bool) -> float:
        with tempfile.TemporaryDirectory(prefix="lin_cau_") as tmp:
            if arm == "D":
                ag = sample_birth_from_arm_d(theta, life_seed=birth_seed, s_dir=Path(tmp) / "s", device=device)
            else:
                ag = apply_arm_c_theta(theta, life_seed=birth_seed, s_dir=Path(tmp) / "s", device=device)
            if not plasticity:
                freeze_plasticity(ag)
            if teach:
                live_once(ag, world, n_wake=n_wake, n_replay=n_replay, teacher_seed=ts)
            return float(probe_beneficial(ag, world, n_probe=n_probe))

    birth = one(plasticity=True, teach=False)
    adult = one(plasticity=True, teach=True)
    off = one(plasticity=False, teach=True)
    return {"adult": adult, "birth": birth, "plasticity_off": off}


def evaluate_champion(
    theta: np.ndarray,
    arm: str,
    *,
    world_seeds: list[int],
    birth_seeds: list[int],
    teacher_seed: int,
    n_wake: int,
    n_replay: int,
    device: str,
    tau: float,
    n_boot: int = 9999,
) -> dict[str, Any]:
    cells: list[tuple[int, int, float]] = []
    adults: list[float] = []
    births: list[float] = []
    offs: list[float] = []
    for ws in world_seeds:
        for bs in birth_seeds:
            c = causal_scores(
                theta, arm,
                world_seed=ws, birth_seed=bs, teacher_seed=teacher_seed,
                n_wake=n_wake, n_replay=n_replay, device=device,
            )
            cells.append((int(ws), int(bs), c["adult"]))
            adults.append(c["adult"])
            births.append(c["birth"])
            offs.append(c["plasticity_off"])
    adult_mean = float(np.mean(adults)) if adults else 0.0
    birth_mean = float(np.mean(births)) if births else 0.0
    off_mean = float(np.mean(offs)) if offs else 0.0
    lo = cluster_bootstrap_lower(cells, n_boot=n_boot)
    gk = g_k(adult_mean, birth_mean, off_mean, tau, 0.05, 0.05)
    return {
        "adult_mean": adult_mean,
        "birth_mean": birth_mean,
        "plasticity_off_mean": off_mean,
        "ci_lower": lo,
        "tau": tau,
        "G_k": gk,
        "panel_clear": bool(lo >= tau and gk),
        "n_cells": len(cells),
    }


def checkpoint_champion(
    theta: np.ndarray,
    arm: str,
    *,
    stream_index: int,
    amend: dict[str, Any],
    device: str,
    failed_panels: list[str],
) -> dict[str, Any]:
    """Consume the next unused DEV triplet. Failed panels are never reused."""
    if failed_panels:
        # still consume a fresh unused triplet; do not go back
        pass
    triplet = next_dev_triplet(stream_index, n_worlds=int(amend["W"]))
    tau = 0.60  # L0
    panels = {}
    for name in ("A", "B", "C"):
        panels[name] = evaluate_champion(
            theta, arm,
            world_seeds=list(triplet[name]),
            birth_seeds=list(range(int(amend["base_replication"]["B"]))),
            teacher_seed=17,
            n_wake=int(amend["N_wake"]),
            n_replay=int(amend["N_replay"]),
            device=device,
            tau=tau,
        )
        if not panels[name]["panel_clear"]:
            failed_panels.append(f"{stream_index}:{name}")
    return {
        "stream_index": stream_index,
        "triplet": {k: triplet[k] for k in ("A", "B", "C")},
        "panels": panels,
        "all_clear": all(panels[n]["panel_clear"] for n in ("A", "B", "C")),
        "failed_panels": list(failed_panels),
        "note": "X-clears-A Y-clears-B Z-clears-C does not qualify Z. Consolidation is separate.",
    }


def consolidation_triplet(
    theta: np.ndarray,
    arm: str,
    *,
    amend: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Exact genome, A+B+C together, confirmatory tier, no training between panels."""
    triplet = next_dev_triplet(10_000, n_worlds=int(amend["confirmatory_replication"]["W"]))
    tau = 0.60
    panels = {}
    for name in ("A", "B", "C"):
        panels[name] = evaluate_champion(
            theta, arm,
            world_seeds=list(triplet[name]),
            birth_seeds=list(range(int(amend["confirmatory_replication"]["B"]))),
            teacher_seed=19,
            n_wake=int(amend["N_wake"]),
            n_replay=int(amend["N_replay"]),
            device=device,
            tau=tau,
        )
    return {
        "triplet": {k: triplet[k] for k in ("A", "B", "C")},
        "panels": panels,
        "all_clear": all(panels[n]["panel_clear"] for n in ("A", "B", "C")),
        "training_between_panels": False,
        "tier": "confirmatory",
    }


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def both_prospects_on_origin_main() -> dict[str, Any]:
    d = REPO_ROOT / "docs" / "lineage_prospect.lock"
    c = REPO_ROOT / "docs" / "lineage_prospect_c.lock"
    return {
        "arm_d": d.exists(),
        "arm_c": c.exists(),
        "clean": _git_clean(),
        "head": _git_head(),
        "ok": d.exists() and c.exists() and _git_clean(),
    }


def refuse_shared_qual() -> dict[str, Any]:
    gate = both_prospects_on_origin_main()
    if not gate["ok"]:
        return {
            "refuse": True,
            "why": "both Arm D and Arm C prospects must be frozen on clean origin/main before shared QUAL",
            "gate": gate,
            "superiority_claim": False,
        }
    return {"refuse": False, "gate": gate}


def phase0b(device: str = "cpu") -> dict[str, Any]:
    theta = defaults_theta("D")
    n_wake, n_replay = 40, 8
    birth_seeds = [11, 12, 13, 14]
    world_seeds = [101, 202, 303]
    teacher_seeds = [7, 8]

    def one(bs: int, ws: int, ts: int) -> dict[str, Any]:
        world = make_synthetic_world(ws)
        with tempfile.TemporaryDirectory(prefix="p0b_") as tmp:
            ag = sample_birth_from_arm_d(theta, life_seed=bs, s_dir=Path(tmp) / "s", device=device)
            t0 = time.perf_counter()
            summary = live_once(ag, world, n_wake=n_wake, n_replay=n_replay, teacher_seed=ts)
            elapsed = time.perf_counter() - t0
        summary["seconds"] = elapsed
        summary["ticks_per_second"] = (n_wake + n_replay) / elapsed if elapsed else 0.0
        return summary

    cells = []
    for bs in birth_seeds:
        for ws in world_seeds:
            cells.append({"birth": bs, "world": ws, "teacher": teacher_seeds[0], **one(bs, ws, teacher_seeds[0])})
    teacher_cells = [one(birth_seeds[0], world_seeds[0], ts)["energy_mean"] for ts in teacher_seeds]

    by_world: dict[int, list[float]] = {}
    for c in cells:
        by_world.setdefault(int(c["world"]), []).append(float(c["energy_mean"]))
    birth_var = float(np.mean([np.var(v) for v in by_world.values()])) if by_world else 0.0
    by_birth: dict[int, list[float]] = {}
    for c in cells:
        by_birth.setdefault(int(c["birth"]), []).append(float(c["energy_mean"]))
    world_var = float(np.mean([np.var(v) for v in by_birth.values()])) if by_birth else 0.0
    teacher_var = float(np.var(teacher_cells))
    tps = float(np.median([c["ticks_per_second"] for c in cells]))
    t_tick = 1.0 / tps if tps else None

    def tgen(p: int, b: int, w: int, e: int, nw: int, nr: int) -> dict[str, Any]:
        steps = 2 * p * b * w * e * (nw + nr)
        seconds = steps * (t_tick or 0.0)
        return {"steps": steps, "hours": seconds / 3600.0}

    out = {
        "version": "TM.0.24.LINEAGE.ENGINE.PREFLIGHT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "device": device,
        "env": torch_env(),
        "shas": engine_shas(),
        "n_wake": n_wake,
        "n_replay": n_replay,
        "ticks_per_second_p50": tps,
        "t_tick_seconds": t_tick,
        "variance": {
            "within_world_birth": birth_var,
            "across_world": world_var,
            "teacher": teacher_var,
        },
        "analytical": {
            "illustrative_128x4x4x1": tgen(128, 4, 4, 1, 500, 25),
            "frozen_proposal_16x2x2x1": tgen(16, 2, 2, 1, 80, 10),
        },
        "rest_used": True,
        "arm": "D",
        "note": "Real Arm D siblings + REST + teacher. Not a capability score.",
    }
    path = REPO_ROOT / "docs" / "lineage_engine.preflight.lock"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_compute_amendment(preflight: dict[str, Any]) -> dict[str, Any]:
    """Sample size and budget only. Does not move τ/δ/L floors."""
    hours16 = float((preflight.get("analytical") or {}).get("frozen_proposal_16x2x2x1", {}).get("hours") or 0)
    amend = {
        "version": "TM.0.24.LINEAGE.COMPUTE.AMENDMENT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "preflight_sha": sha_file(REPO_ROOT / "docs" / "lineage_engine.preflight.lock"),
        "scope_reduction": (
            "P=16 pairs (not 128) because Phase 0A/0B show sequential n=64 is CPU-bound; "
            "scientific floors unchanged."
        ),
        "P": 16,
        "islands": 2,
        "pairs_per_island": 8,
        "B": 2,
        "W": 2,
        "E": 1,
        "N_wake": 80,
        "N_replay": 10,
        "sigma": 0.05,
        "adam_lr": 0.02,
        "base_replication": {"B": 2, "W": 2},
        "confirmatory_replication": {"B": 4, "W": 4},
        "hard_maximum": {"B": 8, "W": 8},
        "max_generations": 20,
        "checkpoint_every": 5,
        "expected_hours_per_generation": hours16,
        "gpu_hour_budget": 48.0,
        "preferred_device": "cpu",
        "note": "If confirmatory precision is unaffordable, record underpowered. Do not move τ_k.",
    }
    path = REPO_ROOT / "docs" / "lineage_compute.amendment.lock"
    path.write_text(json.dumps(amend, indent=2) + "\n", encoding="utf-8")
    return amend


def write_engine_candidate(preflight: dict[str, Any], amend: dict[str, Any]) -> dict[str, Any]:
    shas = engine_shas()
    if shas != preflight.get("shas"):
        raise RuntimeError("engine SHAs drifted vs Phase 0B — new versioned preflight required")
    prereg = json.loads(GEN_PREREG.read_text(encoding="utf-8"))
    cand = {
        "version": "TM.0.24.LINEAGE.ENGINE.CANDIDATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "preflight": "docs/lineage_engine.preflight.lock",
        "compute_amendment": "docs/lineage_compute.amendment.lock",
        "shas": shas,
        "mutation": {"distribution": "isotropic_gaussian", "sigma": amend["sigma"], "antithetic": True},
        "optimizer": {"name": "Adam", "lr": amend["adam_lr"]},
        "rank_transform": True,
        "islands": {
            "n": amend["islands"],
            "pairs": amend["pairs_per_island"],
            "no_migration_pilot": True,
            "no_migration_first_30_main": True,
        },
        "qual_seed_commitment": prereg["qual_seed_commitment"],
        "eval_seed_commitment": prereg["eval_seed_commitment"],
        "note": "No implementation change after this lock without a new lineage version.",
    }
    path = REPO_ROOT / "docs" / "lineage_engine.candidate.lock"
    path.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return cand


def assert_engine_frozen() -> dict[str, Any]:
    cand_p = REPO_ROOT / "docs" / "lineage_engine.candidate.lock"
    if not cand_p.exists():
        raise RuntimeError("no engine candidate — refuse scored evolution")
    cand = json.loads(cand_p.read_text(encoding="utf-8"))
    live = engine_shas()
    if live != cand["shas"]:
        raise RuntimeError("implementation changed after engine candidate")
    return cand


def write_wall_lock(result: dict[str, Any]) -> dict[str, Any]:
    wall = {
        "version": "TM.0.24.LINEAGE.WALL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "L0_unlocked": bool(result.get("L0_unlocked")),
        "first_unacquired_level": "L0" if not result.get("L0_unlocked") else None,
        "interpretation": (
            "bounded unresolved substrate/search wall (not proven impossibility)"
            if not result.get("L0_unlocked")
            else "L0 unlocked under frozen bounded conditions; not 0.0.005"
        ),
        "dense_theta_run": bool(result.get("arm_c_mean_f") is not None),
        "qual_revealed": False,
        "eval_revealed": False,
        "note": result.get("note"),
    }
    path = REPO_ROOT / "docs" / "lineage_wall.lock"
    path.write_text(json.dumps(wall, indent=2) + "\n", encoding="utf-8")
    return wall


def scored_evolution(device: str = "cpu", *, n_generations: int | None = None, n_pairs: int | None = None) -> dict[str, Any]:
    cand = assert_engine_frozen()
    _ = cand
    amend = json.loads((REPO_ROOT / "docs" / "lineage_compute.amendment.lock").read_text(encoding="utf-8"))
    layout = load_layout()
    theta = defaults_theta("D", layout)
    adam = AdamState(m=np.zeros_like(theta), v=np.zeros_like(theta), lr=float(amend["adam_lr"]))
    gens = int(n_generations if n_generations is not None else min(int(amend["max_generations"]), 5))
    pairs = int(n_pairs if n_pairs is not None else min(int(amend["P"]), 8))
    history: list[dict[str, Any]] = []
    failed_panels: list[str] = []
    champ_reports: list[dict[str, Any]] = []
    stream_index = 0
    for g in range(gens):
        train = next_dev_triplet(1000 + g, n_worlds=int(amend["W"]))
        step = es_step(
            theta, "D",
            n_pairs=pairs,
            sigma=float(amend["sigma"]),
            mut_seed=21 + g,
            world_seeds=train["A"],
            birth_seeds=list(range(int(amend["B"]))),
            teacher_seed=13,
            n_wake=int(amend["N_wake"]),
            n_replay=int(amend["N_replay"]),
            device=device,
            adam=adam,
            scored=True,
        )
        theta = step["theta"]
        history.append({"generation": g, "mean_f": step["mean_f"], "grad_norm": step["grad_norm"]})
        if (g + 1) % int(amend["checkpoint_every"]) == 0 or g == gens - 1:
            report = checkpoint_champion(
                theta, "D", stream_index=stream_index, amend=amend, device=device, failed_panels=failed_panels,
            )
            champ_reports.append(report)
            stream_index += 1
    # Arm C control: one scored generation from packed v27 birth, not a superiority claim
    arm_c_mean = None
    with tempfile.TemporaryDirectory(prefix="armc_") as tmp:
        v27 = make_cortex(Path(tmp) / "s", device=device)
        ctheta = pack_arm_c_from_cortex(v27, layout)
        cadam = AdamState(m=np.zeros_like(ctheta), v=np.zeros_like(ctheta), lr=float(amend["adam_lr"]))
        cstep = es_step(
            ctheta, "C",
            n_pairs=2, sigma=float(amend["sigma"]), mut_seed=31,
            world_seeds=next_dev_triplet(2000, n_worlds=int(amend["W"]))["A"],
            birth_seeds=[0], teacher_seed=15,
            n_wake=max(20, int(amend["N_wake"]) // 4),
            n_replay=max(4, int(amend["N_replay"]) // 2),
            device=device, adam=cadam, scored=True,
        )
        arm_c_mean = cstep["mean_f"]
    l0 = bool(champ_reports and champ_reports[-1]["all_clear"])
    cons = None
    if l0:
        cons = consolidation_triplet(theta, "D", amend=amend, device=device)
        l0 = bool(cons["all_clear"])
    qual = refuse_shared_qual()
    out = {
        "version": "TM.0.24.LINEAGE.SCORED",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "L0_unlocked": l0,
        "history": history,
        "champion_reports": champ_reports,
        "consolidation": cons,
        "arm_c_mean_f": arm_c_mean,
        "qual": qual,
        "eval_revealed": False,
        "failed_panels": failed_panels,
        "n_generations": gens,
        "n_pairs": pairs,
        "note": (
            "F_search always. QUAL/EVAL remain sealed. Both-arm prospects were not frozen. "
            "Not 0.0.005."
        ),
    }
    (REPO_ROOT / "docs" / "tm024lineage_results.md").write_text(
        "# TM.0.24.LINEAGE results\n\n"
        "Product remains **0.0.004**. `earned_next=false`. `ex0s=null`.\n\n"
        f"Scored generations: {gens} (P={pairs}). Last mean F_search: "
        f"{history[-1]['mean_f'] if history else 'n/a'}.\n\n"
        f"L0 unlocked: **{l0}**. QUAL refused until both arm prospects are on clean "
        "`origin/main`. EVAL not revealed.\n\n"
        "A rigorous L0 wall under frozen architecture/search/data/compute is a primary result. "
        "This is not 0.0.005.\n",
        encoding="utf-8",
    )
    write_wall_lock(out)
    return out


def freeze_engine(device: str = "cpu") -> dict[str, Any]:
    gen = write_generator_lock()
    pf = phase0b(device=device)
    am = write_compute_amendment(pf)
    cand = write_engine_candidate(pf, am)
    return {
        "generator_sha": gen["generator_sha"],
        "preflight_tps": pf["ticks_per_second_p50"],
        "amendment_P": am["P"],
        "candidate_version": cand["version"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--write-generator-lock", action="store_true")
    p.add_argument("--unscored-smoke", action="store_true")
    p.add_argument("--write-compat", action="store_true")
    p.add_argument("--phase0b", action="store_true")
    p.add_argument("--write-engine-candidate", action="store_true")
    p.add_argument("--freeze-engine", action="store_true")
    p.add_argument("--scored-smoke", action="store_true")
    p.add_argument("--scored", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    ran = False
    if args.write_compat:
        print(json.dumps(write_compat_lock(), indent=2, default=str))
        ran = True
    if args.write_generator_lock:
        print(json.dumps(write_generator_lock(), indent=2))
        ran = True
    if args.unscored_smoke:
        print(json.dumps(unscored_smoke(device=args.device), indent=2, default=str))
        ran = True
    if args.phase0b:
        pf = phase0b(device=args.device)
        am = write_compute_amendment(pf)
        print(json.dumps({"preflight_tps": pf["ticks_per_second_p50"], "amendment_P": am["P"]}, indent=2))
        ran = True
    if args.write_engine_candidate:
        pf = json.loads((REPO_ROOT / "docs" / "lineage_engine.preflight.lock").read_text(encoding="utf-8"))
        am = json.loads((REPO_ROOT / "docs" / "lineage_compute.amendment.lock").read_text(encoding="utf-8"))
        print(json.dumps(write_engine_candidate(pf, am), indent=2))
        ran = True
    if args.freeze_engine:
        print(json.dumps(freeze_engine(device=args.device), indent=2))
        ran = True
    if args.scored_smoke:
        print(json.dumps(scored_evolution(device=args.device, n_generations=1, n_pairs=2), indent=2, default=str))
        ran = True
    if args.scored:
        print(json.dumps(scored_evolution(device=args.device), indent=2, default=str))
        ran = True
    if not ran:
        p.print_help()


if __name__ == "__main__":
    main()
