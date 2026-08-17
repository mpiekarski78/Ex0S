"""TM.0.24.COLLISIONMAP — cross-cue collision / interference diagnostic.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Scoring requires docs/lineage_collisionmap.runner.lock on clean origin/main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import make_cortex, physics, torch_env
from experiments.run_tm024actorcredit import (
    MID_BODY,
    clone_frozen,
    harmful_handle,
    motor_scores,
    observe_cue,
    op_logits,
    p_handle,
    prep_eval,
)
from experiments.run_tm024lineage import make_synthetic_world
from experiments.run_tm024statemap import teach_one
from three_memory.cortex_lineage import freeze_plasticity, sha_file
from three_memory.neural_cortex import OPS, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_collisionmap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_collisionmap_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_collisionmap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_collisionmap.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_collisionmap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024collisionmap_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v29.lock"
STATEMAP = REPO_ROOT / "docs" / "lineage_statemap.decision.lock"

CELLS_DOMAIN = "TM024.COLLISIONMAP.CELLS."
TWIN_DOMAIN = "TM024.COLLISIONMAP.TWIN."


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def thr() -> dict[str, Any]:
    return load_prereg()["thresholds"]


def fit_spec() -> dict[str, Any]:
    return load_prereg()["frozen_readout_fit"]


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def make_cell_world(index: int, domain: str = CELLS_DOMAIN) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def collisionmap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v29": CANDIDATE,
        "statemap_decision": STATEMAP,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no collisionmap runner.lock — refuse cell scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if collisionmap_shas() != lock.get("shas"):
        raise RuntimeError("collisionmap implementation drifted after runner.lock")
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if sha_file(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from v29 candidate")
    if cand.get("genome", {}).get("n") != 64:
        raise RuntimeError("n must stay 64")
    return lock


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    return ag


def restore_plasticity(ag: NeuralCortex, eta_pred: float, eta_act: float, beta: float) -> None:
    ag.genome.eta_pred = float(eta_pred)
    ag.genome.eta_act = float(eta_act)
    ag.genome.beta = float(beta)


def warmup_vocab(ag: NeuralCortex, world: dict[str, Any], extra: list[str] | None = None) -> None:
    eta = (float(ag.genome.eta_pred), float(ag.genome.eta_act), float(ag.genome.beta))
    freeze_plasticity(ag)
    for i, s in enumerate(list(world["symbols"]) + list(extra or [])):
        observe_cue(ag, world, tag=f"vocab{i}", body=list(MID_BODY), symbols=[s])
    prep_eval(ag)
    ag.reset_rho()
    restore_plasticity(ag, *eta)


def cue_pair(world: dict[str, Any]) -> tuple[str, str]:
    pair = set(world["teacher_pair"])
    rest = [s for s in world["symbols"] if s not in pair]
    return str(world["teacher_pair"][0]), str(rest[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def distinct(cos: float, dist: float, t: dict[str, Any] | None = None) -> bool:
    t = t or thr()
    return bool(cos < float(t["cos_distinct_max"]) or dist > float(t["l2_distinct_min"]))


def readout_at(ag: NeuralCortex, world: dict[str, Any], rho: np.ndarray) -> dict[str, float]:
    saved = ag._from_t(ag.rho)
    ag.rho = ag._to_t(np.asarray(rho, dtype=np.float64))
    ben = world["beneficial"]
    harm = harmful_handle(world)
    scores = motor_scores(ag)
    logits = op_logits(ag)
    out = {
        "act_logit": float(logits[OPS.index("ACT")]),
        "hold_logit": float(logits[OPS.index("HOLD")]),
        "motor_ben": float(scores.get(ben, 0.0)),
        "motor_harm": float(scores.get(harm, 0.0)),
        "p_handle_ben": p_handle(ag, ben),
    }
    ag.rho = ag._to_t(saved)
    return out


def parse_stages(ag: NeuralCortex) -> dict[str, np.ndarray]:
    sens = [np.asarray(x, dtype=np.float64) for x in ag.sensory_trajectory]
    full = [np.asarray(x, dtype=np.float64) for x in ag.last_trajectory]
    if not sens:
        raise RuntimeError("empty sensory trajectory")
    stages: dict[str, np.ndarray] = {
        "start": sens[0],
        "cue": sens[1] if len(sens) > 1 else sens[0],
        "event_end": sens[-2] if len(sens) >= 3 else sens[-1],
        "observable": sens[-1],
    }
    motor = full[len(sens) :]
    for i, r in enumerate(motor):
        stages[f"motor_{i}"] = r
    stages["motor_last"] = motor[-1] if motor else sens[-1]
    stages["final"] = full[-1]
    if ag.last_action is not None:
        stages["rho_elig"] = np.asarray(ag.last_action["rho_elig"], dtype=np.float64).copy()
    else:
        stages["rho_elig"] = stages["final"]
    stages["_n_sensory"] = np.asarray([len(sens)], dtype=np.float64)
    stages["_n_motor"] = np.asarray([len(motor)], dtype=np.float64)
    return stages


def one_trace(ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str) -> dict[str, Any]:
    probe = clone_frozen(ag)
    probe.reset_rho()
    prep_eval(probe)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    stages = parse_stages(probe)
    readouts = {
        k: readout_at(probe, world, v)
        for k, v in stages.items()
        if not k.startswith("_")
    }
    return {
        "stages": {k: v for k, v in stages.items() if not k.startswith("_")},
        "n_sensory": int(stages["_n_sensory"][0]),
        "n_motor": int(stages["_n_motor"][0]),
        "readouts": readouts,
    }


COMPARE_KEYS = ("start", "cue", "event_end", "observable", "motor_last", "final", "rho_elig")


def pair_metrics(tr_a: dict[str, Any], tr_b: dict[str, Any], t: dict[str, Any] | None = None) -> dict[str, Any]:
    t = t or thr()
    rows: dict[str, Any] = {}
    for k in COMPARE_KEYS:
        if k not in tr_a["stages"] or k not in tr_b["stages"]:
            continue
        cos = cosine(tr_a["stages"][k], tr_b["stages"][k])
        dist = l2(tr_a["stages"][k], tr_b["stages"][k])
        rows[k] = {
            "cosine": cos,
            "l2": dist,
            "distinct": distinct(cos, dist, t),
            "a": tr_a["readouts"][k],
            "b": tr_b["readouts"][k],
        }
    motor_keys = sorted(
        [k for k in tr_a["stages"] if k.startswith("motor_") and k != "motor_last"],
        key=lambda s: int(s.split("_")[1]),
    )
    motor_rows = []
    for k in motor_keys:
        if k not in tr_b["stages"]:
            continue
        cos = cosine(tr_a["stages"][k], tr_b["stages"][k])
        dist = l2(tr_a["stages"][k], tr_b["stages"][k])
        motor_rows.append({"tick": k, "cosine": cos, "l2": dist, "distinct": distinct(cos, dist, t)})
    cue_d = bool(rows.get("cue", {}).get("distinct"))
    elig_d = bool(rows.get("rho_elig", {}).get("distinct"))
    obs_d = bool(rows.get("observable", {}).get("distinct"))
    motor_d = bool(rows.get("motor_last", {}).get("distinct"))
    return {
        "stages": rows,
        "motor_ticks": motor_rows,
        "distinct_cue": cue_d,
        "distinct_observable": obs_d,
        "distinct_motor_last": motor_d,
        "distinct_rho_elig": elig_d,
        "attractor_collapse": bool(cue_d and not elig_d),
        "sensory_collapse": bool(cue_d and not obs_d),
        "remain_distinct": bool(cue_d and elig_d),
        "n_motor_a": tr_a["n_motor"],
        "n_motor_b": tr_b["n_motor"],
    }


def ranking(pol: dict[str, float], *, prefer: str) -> bool:
    if prefer == "ben":
        return bool(pol["motor_ben"] > pol["motor_harm"] + 1e-9)
    return bool(pol["motor_harm"] > pol["motor_ben"] + 1e-9)


def ready_pair(tmp: str, tag: str, world: dict[str, Any], extra: list[str] | None = None) -> NeuralCortex:
    ag = _fresh(tmp, tag, world)
    warmup_vocab(ag, world, extra=extra)
    return ag


def pair_trace_from(ag: NeuralCortex, world: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    cue_a, cue_b = cue_pair(world)
    tr_a = one_trace(ag, world, cue_a, tag=f"{prefix}_a")
    tr_b = one_trace(ag, world, cue_b, tag=f"{prefix}_b")
    return pair_metrics(tr_a, tr_b)


def run_c0(*, domain: str = CELLS_DOMAIN, index: int = 0) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    t = thr()
    with tempfile.TemporaryDirectory(prefix="cm_c0_") as tmp:
        ag = ready_pair(tmp, "s", world)
        m = pair_trace_from(ag, world, prefix="c0")
    passed = True  # measurement cell; collapse is a diagnosis bit, not a skip
    return {
        "id": "C0",
        "passed": passed,
        "attractor_collapse": m["attractor_collapse"],
        "sensory_collapse": m["sensory_collapse"],
        "remain_distinct": m["remain_distinct"],
        "distinct_cue": m["distinct_cue"],
        "distinct_rho_elig": m["distinct_rho_elig"],
        "trace": m,
        "thresholds": t,
        "domain": domain,
    }


def run_c1() -> dict[str, Any]:
    world = make_cell_world(1)
    cue_a, _cue_b = cue_pair(world)
    with tempfile.TemporaryDirectory(prefix="cm_c1_") as tmp:
        ag = ready_pair(tmp, "s", world)
        birth = pair_trace_from(ag, world, prefix="c1b")
        teach_one(ag, world, world["beneficial"], tag="c1t", symbols=[cue_a])
        after = pair_trace_from(ag, world, prefix="c1a")
    return {
        "id": "C1",
        "passed": True,
        "birth_attractor_collapse": birth["attractor_collapse"],
        "after_a_attractor_collapse": after["attractor_collapse"],
        "birth_distinct_cue": birth["distinct_cue"],
        "after_a_distinct_cue": after["distinct_cue"],
        "birth_distinct_rho_elig": birth["distinct_rho_elig"],
        "after_a_distinct_rho_elig": after["distinct_rho_elig"],
        "birth": birth,
        "after_a": after,
    }


def run_c2(*, domain: str = CELLS_DOMAIN, index: int = 2) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    cue_a, cue_b = cue_pair(world)
    ben = world["beneficial"]
    harm = harmful_handle(world)
    with tempfile.TemporaryDirectory(prefix="cm_c2_") as tmp:
        ag = ready_pair(tmp, "s", world)
        ta = teach_one(ag, world, ben, tag="c2a", symbols=[cue_a])
        after_a = one_trace(ag, world, cue_a, tag="c2pa")
        after_a_b = one_trace(ag, world, cue_b, tag="c2pab")
        a_ok_after_a = ranking(after_a["readouts"]["rho_elig"], prefer="ben")
        tb = teach_one(ag, world, harm, tag="c2b", symbols=[cue_b])
        live_a = one_trace(ag, world, cue_a, tag="c2la")
        live_b = one_trace(ag, world, cue_b, tag="c2lb")
        a_ok_after_b = ranking(live_a["readouts"]["rho_elig"], prefer="ben")
        b_ok_after_b = ranking(live_b["readouts"]["rho_elig"], prefer="harm")
        probe = clone_frozen(ag)
        probe.reset_rho()
        prep_eval(probe)
        observe_cue(probe, world, tag="c2sub", body=list(MID_BODY), symbols=[cue_a])
        sub = readout_at(probe, world, ta["rho_teach"])
        proj = readout_at(probe, world, live_b["stages"]["rho_elig"])
        pair = pair_metrics(live_a, live_b)
        destroyed = bool(a_ok_after_a and not a_ok_after_b)
        sub_restores = bool(ranking(sub, prefer="ben"))
    return {
        "id": "C2",
        "passed": True,
        "a_ok_after_a": a_ok_after_a,
        "a_ok_after_b": a_ok_after_b,
        "b_ok_after_b": b_ok_after_b,
        "teaching_b_destroys_a": destroyed,
        "substitute_rho_a_restores": sub_restores,
        "remain_distinct_after_ab": pair["remain_distinct"],
        "attractor_collapse_after_ab": pair["attractor_collapse"],
        "adv_a": ta["adv"],
        "adv_b": tb["adv"],
        "live_a": live_a["readouts"]["rho_elig"],
        "live_b": live_b["readouts"]["rho_elig"],
        "after_a": after_a["readouts"]["rho_elig"],
        "untought_b_after_a": after_a_b["readouts"]["rho_elig"],
        "substitute": sub,
        "other_elig_readout": proj,
        "pair_after_ab": pair,
        "domain": domain,
    }


def ridge_readout(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """W (d, n) minimizing ||W X - Y|| with columns of X/Y as samples."""
    n = X.shape[0]
    xt = X @ X.T + float(lam) * np.eye(n)
    return Y @ X.T @ np.linalg.pinv(xt)


def ridge_classifier(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    a = xb.T @ xb + float(lam) * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y)


def effective_rank(mat: np.ndarray, rel_tol: float) -> int:
    s = np.linalg.svd(mat, compute_uv=False)
    if s.size == 0:
        return 0
    cut = float(rel_tol) * float(s[0] + 1e-12)
    return int(np.sum(s > cut))


def run_c3(*, domain: str = CELLS_DOMAIN, index: int = 3) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    t = thr()
    spec = fit_spec()
    n_cues = int(t["n_rank_cues"])
    n_rep = int(t["n_sep_repeats"])
    extra = [f"cm_rank_{i}" for i in range(max(0, n_cues - len(world["symbols"])))]
    cues = list(world["symbols"]) + extra
    cues = cues[:n_cues]
    cue_a, cue_b = cue_pair(world)
    with tempfile.TemporaryDirectory(prefix="cm_c3_") as tmp:
        ag = ready_pair(tmp, "s", world, extra=extra)
        rhos = []
        for i, c in enumerate(cues):
            tr = one_trace(ag, world, c, tag=f"c3r{i}")
            rhos.append(tr["stages"]["rho_elig"])
        mat = np.stack(rhos, axis=1)
        rank_elig = effective_rank(mat, float(t["rank_rel_tol"]))
        xa, xb = [], []
        for i in range(n_rep):
            xa.append(one_trace(ag, world, cue_a, tag=f"c3a{i}")["stages"]["rho_elig"])
            xb.append(one_trace(ag, world, cue_b, tag=f"c3b{i}")["stages"]["rho_elig"])
        xa_m, xb_m = np.stack(xa), np.stack(xb)
        half = n_rep // 2
        x_tr = np.concatenate([xa_m[:half], xb_m[:half]], axis=0)
        y_tr = np.concatenate([np.ones(half), -np.ones(half)])
        x_te = np.concatenate([xa_m[half:], xb_m[half:]], axis=0)
        y_te = np.concatenate([np.ones(n_rep - half), -np.ones(n_rep - half)])
        w = ridge_classifier(x_tr, y_tr, float(spec["lambda"]))
        xb_te = np.concatenate([x_te, np.ones((x_te.shape[0], 1))], axis=1)
        pred = np.sign(xb_te @ w)
        pred[pred == 0] = 0
        acc = float(np.mean(pred == y_te))
        sep_ok = bool(acc >= float(t["linear_sep_acc_min"]))
        mean_cos = cosine(np.mean(xa_m, axis=0), np.mean(xb_m, axis=0))
        mean_l2 = l2(np.mean(xa_m, axis=0), np.mean(xb_m, axis=0))
    passed = True
    return {
        "id": "C3",
        "passed": passed,
        "eff_rank_rho_elig": rank_elig,
        "n_cues": n_cues,
        "linear_sep_acc": acc,
        "linear_sep_ok": sep_ok,
        "mean_cosine_a_b": mean_cos,
        "mean_l2_a_b": mean_l2,
        "singular_rel_tol": float(t["rank_rel_tol"]),
        "rank_ge2": bool(rank_elig >= 2),
        "domain": domain,
    }


def run_c4(*, domain: str = CELLS_DOMAIN, index: int = 4) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    spec = fit_spec()
    t = thr()
    cue_a, cue_b = cue_pair(world)
    ben = world["beneficial"]
    harm = harmful_handle(world)
    lam = float(spec["lambda"])
    with tempfile.TemporaryDirectory(prefix="cm_c4_") as tmp:
        ag = ready_pair(tmp, "s", world)
        tr_a = one_trace(ag, world, cue_a, tag="c4a")
        tr_b = one_trace(ag, world, cue_b, tag="c4b")
        rho_a = tr_a["stages"]["rho_elig"]
        rho_b = tr_b["stages"]["rho_elig"]
        rho_a_cue = tr_a["stages"]["cue"]
        rho_b_cue = tr_b["stages"]["cue"]
        va = ag.motor_vocab[ben]
        vb = ag.motor_vocab[harm]
        x = np.stack([rho_a, rho_b], axis=1)
        y = np.stack([va, vb], axis=1)
        w = ridge_readout(x, y, lam)
        qa, qb = w @ rho_a, w @ rho_b
        fit_a = cosine(qa, va) > cosine(qa, vb)
        fit_b = cosine(qb, vb) > cosine(qb, va)
        x_c = np.stack([rho_a_cue, rho_b_cue], axis=1)
        w_c = ridge_readout(x_c, y, lam)
        qac, qbc = w_c @ rho_a_cue, w_c @ rho_b_cue
        fit_cue_a = cosine(qac, va) > cosine(qac, vb)
        fit_cue_b = cosine(qbc, vb) > cosine(qbc, va)
        teach_one(ag, world, ben, tag="c4ta", symbols=[cue_a])
        teach_one(ag, world, harm, tag="c4tb", symbols=[cue_b])
        v29_a = one_trace(ag, world, cue_a, tag="c4va")["readouts"]["rho_elig"]
        v29_b = one_trace(ag, world, cue_b, tag="c4vb")["readouts"]["rho_elig"]
        v29_ok = bool(ranking(v29_a, prefer="ben") and ranking(v29_b, prefer="harm"))
        fit_ok = bool(fit_a and fit_b)
        fit_cue_ok = bool(fit_cue_a and fit_cue_b)
    return {
        "id": "C4",
        "passed": True,
        "direct_fit_elig_ok": fit_ok,
        "direct_fit_cue_ok": fit_cue_ok,
        "v29_sequential_ok": v29_ok,
        "geometry_failure": bool(fit_ok and not v29_ok),
        "lambda": lam,
        "state": spec["state"],
        "fit_elig": {"a_prefers_ben": bool(fit_a), "b_prefers_harm": bool(fit_b)},
        "fit_cue": {"a_prefers_ben": bool(fit_cue_a), "b_prefers_harm": bool(fit_cue_b)},
        "v29": {"a": v29_a, "b": v29_b},
        "elig_cosine": cosine(rho_a, rho_b),
        "elig_l2": l2(rho_a, rho_b),
        "cue_cosine": cosine(rho_a_cue, rho_b_cue),
        "cue_l2": l2(rho_a_cue, rho_b_cue),
        "domain": domain,
    }


def diagnosis_bits(c0: dict[str, Any], c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any], c4: dict[str, Any]) -> dict[str, bool]:
    collapse = bool(c0.get("attractor_collapse") or c1.get("after_a_attractor_collapse") or c2.get("attractor_collapse_after_ab"))
    remain = bool(c0.get("remain_distinct") or c2.get("remain_distinct_after_ab"))
    return {
        "attractor_collapse": collapse,
        "sequential_destroy": bool(c2.get("teaching_b_destroys_a")),
        "remain_distinct": remain,
        "linear_sep_ok": bool(c3.get("linear_sep_ok")),
        "rank_ge2": bool(c3.get("rank_ge2")),
        "direct_fit_ok": bool(c4.get("direct_fit_elig_ok")),
        "v29_ok": bool(c4.get("v29_sequential_ok")),
    }


def run_c5(c0: dict[str, Any], c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any], c4: dict[str, Any]) -> dict[str, Any]:
    t0 = run_c0(domain=TWIN_DOMAIN, index=0)
    t2 = run_c2(domain=TWIN_DOMAIN, index=0)
    t3 = run_c3(domain=TWIN_DOMAIN, index=0)
    t4 = run_c4(domain=TWIN_DOMAIN, index=0)
    # C1 is post-A attractor; rerun via C0/C2 bits on twin
    primary = diagnosis_bits(c0, c1, c2, c3, c4)
    twin = diagnosis_bits(t0, {"after_a_attractor_collapse": t0["attractor_collapse"]}, t2, t3, t4)
    same = primary == twin
    return {
        "id": "C5",
        "passed": bool(same),
        "same_bits": same,
        "primary": primary,
        "twin": twin,
        "domain": TWIN_DOMAIN,
    }


def decide(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by = {c["id"]: c for c in cells}
    bits = diagnosis_bits(by["C0"], by["C1"], by["C2"], by["C3"], by["C4"])
    if bits["attractor_collapse"]:
        code = "attractor_collapse"
        note = "A/B separate after cue ingestion, then are not distinct at ρ_elig."
    elif bits["remain_distinct"] and bits["sequential_destroy"]:
        code = "sequential_plastic_write_interference"
        note = "A/B remain distinct at ρ_elig, but teaching B destroys A’s ranking."
    elif (not bits["linear_sep_ok"]) or (not bits["rank_ge2"]):
        code = "representation_rank_failure"
        note = "Frozen linear discriminator or effective rank at ρ_elig cannot support two cues."
    elif bits["direct_fit_ok"] and not bits["v29_ok"]:
        code = "plastic_update_geometry_failure"
        note = "Frozen ridge readout assigns opposite handles; v29 sequential teaching does not."
    else:
        code = "unresolved_cross_cue_collision"
        note = "Preregistered four-way table did not uniquely fire."
    return {
        "code": code,
        "note": note,
        "bits": bits,
        "amendment_authorized": False,
        "two_timescale_authorized": False,
        "increase_n": False,
        "another_lineage_run": False,
    }


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_cell_world(0)
    with tempfile.TemporaryDirectory(prefix="cm_smk_") as tmp:
        ag = ready_pair(tmp, "s", w)
        m = pair_trace_from(ag, w, prefix="smk")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "distinct_cue": m["distinct_cue"],
        "distinct_rho_elig": m["distinct_rho_elig"],
        "cells": prereg["cells"],
        "domain": CELLS_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "lambda": prereg["frozen_readout_fit"]["lambda"],
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if not _git_clean():
        raise RuntimeError("write runner.lock only on a clean tree")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.COLLISIONMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "shas": collisionmap_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "isolation_sha": sha_file(ISOLATION),
        "candidate_v29_sha": sha_file(CANDIDATE),
        "statemap_decision_sha": sha_file(STATEMAP),
        "n": 64,
        "domain": CELLS_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "cells": prereg["cells"],
        "thresholds": prereg["thresholds"],
        "frozen_readout_fit": prereg["frozen_readout_fit"],
        "git_head": _git_head(),
        "note": "Frozen C-cell runner. Score only after this lock is on origin/main. No neural edit. v29 live.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_all() -> dict[str, Any]:
    assert_runner_frozen()
    c0 = run_c0()
    c1 = run_c1()
    c2 = run_c2()
    c3 = run_c3()
    c4 = run_c4()
    c5 = run_c5(c0, c1, c2, c3, c4)
    cells = [c0, c1, c2, c3, c4, c5]
    decision = decide(cells)
    out = {
        "version": "TM.0.24.COLLISIONMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "neural_edit": False,
        "n": 64,
        "n_pass": sum(1 for c in cells if c["passed"]),
        "n_cells": len(cells),
        "cells": cells,
        "decision": decision,
        "another_lineage_run": False,
        "q3": False,
        "amendment_authorized": False,
        "two_timescale_authorized": False,
        "increase_n": False,
        "git_head": _git_head(),
        "env": torch_env(),
        "note": "C0–C5 on unused TM024.COLLISIONMAP.CELLS. / TWIN. Diagnosis only. Not 0.0.005.",
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    lines = ["# TM.0.24.COLLISIONMAP results\n"]
    for c in cells:
        lines.append(f"- `{c['id']}`: **{'PASS' if c['passed'] else 'FAIL'}**")
    lines.append(f"\nPrimary diagnosis: `{decision['code']}` — {decision['note']}")
    lines.append("\nn stays 64. Product 0.0.004. Amendment not authorized this package.\n")
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cell", choices=["C0", "C1", "C2", "C3", "C4", "C5"])
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
    elif args.score:
        print(json.dumps(run_all(), indent=2, default=str))
    elif args.cell:
        if args.cell == "C5":
            print(json.dumps(run_c5(run_c0(), run_c1(), run_c2(), run_c3(), run_c4()), indent=2, default=str))
        else:
            fn = {"C0": run_c0, "C1": run_c1, "C2": run_c2, "C3": run_c3, "C4": run_c4}[args.cell]
            print(json.dumps(fn(), indent=2, default=str))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
