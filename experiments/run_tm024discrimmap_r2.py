"""TM.0.24.DISCRIMMAP.R2 — pinned runner-only separability diagnostic.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Historical DISCRIMMAP DEV/decision/runner locks are immutable.
DEV on unused TM024.DISCRIMMAP.R2.DEV. after this freeze is on origin/main.
SCORE reserved and unopened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024collisionmap import ridge_classifier
from experiments.run_tm024eligmap import (
    EligTrace,
    ProtoBank,
    capacity_world,
    clipnorm,
    collect_stream,
    domain_seed,
    mapping_pairs,
    parse_lam,
    tick_static,
    unit_or_zero,
)
from experiments.run_tm024motorpersist import TEACH_ORDERS
from three_memory.cortex_lineage import sha_file

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_discrimmap.r2.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_discrimmap.r2.contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_discrimmap.r2.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_discrimmap.r2.runner.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_discrimmap.r2.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_discrimmap.r2.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024discrimmap_r2_results.md"
HIST_DEC = REPO_ROOT / "docs" / "lineage_discrimmap.decision.lock"
HIST_DEV = REPO_ROOT / "docs" / "lineage_discrimmap.dev.lock"
HIST_RUN = REPO_ROOT / "docs" / "lineage_discrimmap.runner.lock"
HIST_ADD = REPO_ROOT / "docs" / "lineage_discrimmap.decision.addendum.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"

DEV_DOMAIN = "TM024.DISCRIMMAP.R2.DEV."
TWIN_DOMAIN = "TM024.DISCRIMMAP.R2.TWIN."
SCORE_DOMAIN = "TM024.DISCRIMMAP.R2.SCORE."
SCORE_MARKERS = ("TM024.DISCRIMMAP.R2.SCORE.", "TM024.DISCRIMMAP.SCORE.")
EPS = 1e-12
ACCEPTED_STATUS = ("optimal", "infeasible", "not_applicable")
ARMS = ("D0", "D1", "D2", "D3", "D4")
EXPECTED_N_RANK = 4 * 5 * 3 * 2 * 2
EXPECTED_N_TWIN = 4 * 5 * 2
EXPECTED_N_CELLS = EXPECTED_N_RANK + EXPECTED_N_TWIN


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def r2_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "historical_decision": HIST_DEC,
        "historical_dev": HIST_DEV,
        "historical_runner": HIST_RUN,
        "historical_addendum": HIST_ADD,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, arm: str, aid: str, n_cues: int, order: str, world: int) -> str:
    return f"{kind}|{arm}|{aid}|c{n_cues}|{order}|w{world}"


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no discrimmap R2 runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    current = r2_shas()
    if current != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    return lock


def refuse_rerun() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("same frozen DEV execution requested again")


def require_accepted(status: str) -> None:
    if status not in ACCEPTED_STATUS:
        raise RuntimeError(f"unaccepted solver status {status}")


def require_finite(name: str, *arrays: Any) -> None:
    for a in arrays:
        x = np.asarray(a, dtype=np.float64)
        if x.size and not np.all(np.isfinite(x)):
            raise RuntimeError(f"{name} solver returned NaN/inf")


def refuse_score_markers(payload: str) -> None:
    for mark in SCORE_MARKERS:
        if mark in payload:
            raise RuntimeError("SCORE identifier appeared in DEV payload")


def geometric_margin(w: np.ndarray, b: float, x: np.ndarray, y: float) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(w))
    if not np.isfinite(nrm) or nrm <= EPS:
        return 0.0
    return float(y) * (float(np.dot(w, x)) + float(b)) / nrm


def min_geometric_margin(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
    if X.size == 0:
        return 0.0
    return float(min(geometric_margin(w, b, X[i], float(y[i])) for i in range(len(y))))


def assert_y_pm1(y: np.ndarray) -> None:
    u = set(float(v) for v in np.unique(np.asarray(y, dtype=np.float64)))
    if not u or not u.issubset({-1.0, 1.0}):
        raise RuntimeError(f"target encoding must be y in {{-1,+1}}, got {u}")


@dataclass(frozen=True)
class TeachSplit:
    X: np.ndarray
    y: np.ndarray
    handles: tuple[str, ...]
    chosen: tuple[str, ...]
    cues: tuple[str, ...]
    fingerprint: bytes


@dataclass(frozen=True)
class ProbeSplit:
    X: np.ndarray
    y: np.ndarray
    handles: tuple[str, ...]
    chosen: tuple[str, ...]
    cues: tuple[str, ...]
    fingerprint: bytes


def _fp(X: np.ndarray, y: np.ndarray) -> bytes:
    return np.asarray(X, dtype=np.float64).tobytes() + np.asarray(y, dtype=np.float64).tobytes()


def assert_no_probe_in_fit(teach: TeachSplit, probe: ProbeSplit) -> None:
    if teach.X is probe.X or teach.y is probe.y:
        raise RuntimeError("probe rows passed to fit")


def hard_margin_linear(X: np.ndarray, y: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    """Exact hard-margin SVM via SV-subset KKT. No soft-margin fallback."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    require_finite("D1 input", X, y)
    assert_y_pm1(y)
    if spec.get("soft_margin") or spec.get("soft_margin_C") is not None:
        raise RuntimeError("soft-margin is refused")
    n, d = X.shape
    dim = int(d)
    empty = {
        "w": np.zeros(dim, dtype=np.float64),
        "b": 0.0,
        "status": "infeasible",
        "n_sv": 0,
        "constraint_min": 0.0,
    }
    if n == 0 or len(set(float(v) for v in y)) < 2:
        return empty
    alpha_min = float(spec["alpha_min"])
    kkt_max = float(spec["kkt_residual_max"])
    ctor = float(spec["constraint_tol"])
    best: dict[str, Any] | None = None
    best_norm = float("inf")
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        y_s = y[idx]
        if len(set(float(v) for v in y_s)) < 2:
            continue
        X_s = X[idx]
        m = len(idx)
        Q = (y_s[:, None] * y_s[None, :]) * (X_s @ X_s.T)
        A = np.zeros((m + 1, m + 1), dtype=np.float64)
        A[:m, :m] = Q
        A[:m, m] = y_s
        A[m, :m] = y_s
        rhs = np.zeros(m + 1, dtype=np.float64)
        rhs[:m] = 1.0
        try:
            z = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            z, residuals, _rank, _s = np.linalg.lstsq(A, rhs, rcond=None)
            if residuals.size and float(residuals[0]) > kkt_max:
                continue
        require_finite("D1 KKT", z)
        resid = float(np.linalg.norm(A @ z - rhs))
        if resid > kkt_max:
            continue
        alpha = z[:m]
        b = float(z[m])
        if np.any(alpha < alpha_min - ctor):
            continue
        w = X_s.T @ (alpha * y_s)
        require_finite("D1 w", w, b)
        g = y * (X @ w + b)
        if float(np.min(g)) < 1.0 - ctor:
            continue
        nrm = float(np.linalg.norm(w))
        if nrm < best_norm:
            best_norm = nrm
            best = {
                "w": w,
                "b": b,
                "status": "optimal",
                "n_sv": int(m),
                "constraint_min": float(np.min(g)),
            }
    return best if best is not None else empty


def ridge_fit(X: np.ndarray, y: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    require_finite("D2 input", X, y)
    assert_y_pm1(y)
    if spec.get("mean_center") or spec.get("per_feature_std"):
        raise RuntimeError("D2 feature centering/std is refused")
    if not spec.get("row_unit_l2") or not spec.get("intercept") or spec.get("intercept_in_norm"):
        raise RuntimeError("D2 intercept/normalization freeze violated")
    coef = ridge_classifier(X, y, float(spec["lambda"]))
    require_finite("D2", coef)
    w, b = coef[:-1], float(coef[-1])
    return {"w": w, "b": b, "status": "optimal", "n_sv": 0, "constraint_min": 0.0}


def competitive_fit(X: np.ndarray, handles: tuple[str, ...], chosen: tuple[str, ...], spec: dict[str, Any]) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    require_finite("D3 input", X)
    if spec.get("shuffle") or spec.get("error_only") or spec.get("pool_orders") or spec.get("intercept"):
        raise RuntimeError("D3 freeze violated")
    if spec.get("init") != "zeros" or spec.get("sample_order") != "teach_sequence":
        raise RuntimeError("D3 init/order freeze violated")
    eta = float(spec["eta"])
    epochs = int(spec["epochs"])
    c_max = float(spec["c_max"])
    h0, h1 = handles[0], handles[1]
    rows = {h: np.zeros(X.shape[1], dtype=np.float64) for h in handles}
    for _ep in range(epochs):
        for i, h in enumerate(chosen):
            ehat = unit_or_zero(X[i])
            for hh in handles:
                delta = float(eta) * ehat if hh == h else -float(eta) * ehat
                rows[hh] = clipnorm(rows[hh] + delta, c_max)
    w = rows[h0] - rows[h1]
    require_finite("D3", w)
    return {"w": w, "b": 0.0, "status": "optimal", "n_sv": 0, "constraint_min": 0.0}


def kernel_ridge_fit(X_tr: np.ndarray, y_tr: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64).reshape(-1)
    require_finite("D4 input", X_tr, y_tr)
    assert_y_pm1(y_tr)
    if spec.get("intercept"):
        raise RuntimeError("D4 intercept freeze violated")
    gamma = float(spec["rbf_gamma"])
    lam = float(spec["lambda"])
    K = _rbf(X_tr, X_tr, gamma)
    try:
        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y_tr)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("D4 solver unaccepted status") from exc
    require_finite("D4", alpha, K)
    rkhs = float(np.sqrt(max(float(alpha @ K @ alpha), 0.0)))
    return {"alpha": alpha, "rkhs": rkhs, "status": "optimal", "X_tr": X_tr, "gamma": gamma}


def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
    return np.exp(-float(gamma) * d2)


def capture_xy(
    rec: dict[str, Any],
    *,
    aid: str,
    handles: list[str],
    pairs: list[tuple[str, str]],
) -> tuple[TeachSplit, ProbeSplit]:
    lam = parse_lam(aid)
    tracer = EligTrace(64, handles, lam if lam is not None else 0.0)

    def addr(tick: dict[str, Any], tr: EligTrace) -> np.ndarray:
        tr.ingest(tick["trajectory"])
        if lam is not None:
            return unit_or_zero(tr.address())
        return unit_or_zero(tick_static(tick, aid))

    X_tr: list[np.ndarray] = []
    y_tr: list[float] = []
    h_tr: list[str] = []
    c_tr: list[str] = []
    for tick in rec["ticks"]:
        e = addr(tick, tracer)
        if tick["kind"] == "select" and tick.get("handle") and float(tick.get("adv") or 0.0) != 0.0:
            h = str(tick["handle"])
            X_tr.append(e)
            y_tr.append(1.0 if h == handles[0] else -1.0)
            h_tr.append(h)
            c_tr.append(str(tick.get("cue") or ""))
    X_te: list[np.ndarray] = []
    y_te: list[float] = []
    h_te: list[str] = []
    c_te: list[str] = []
    for (cue, handle), pt in zip(pairs, rec["probes"], strict=True):
        tr = tracer.copy()
        e = addr(pt, tr)
        X_te.append(e)
        y_te.append(1.0 if handle == handles[0] else -1.0)
        h_te.append(handle)
        c_te.append(cue)
    Xt = np.stack(X_tr) if X_tr else np.zeros((0, 64))
    yt = np.asarray(y_tr, dtype=np.float64)
    Xe = np.stack(X_te) if X_te else np.zeros((0, 64))
    ye = np.asarray(y_te, dtype=np.float64)
    teach = TeachSplit(Xt, yt, tuple(handles), tuple(h_tr), tuple(c_tr), _fp(Xt, yt))
    probe = ProbeSplit(Xe, ye, tuple(handles), tuple(h_te), tuple(c_te), _fp(Xe, ye))
    assert_no_probe_in_fit(teach, probe)
    return teach, probe


def perturb_sign_stable(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    *,
    domain: str,
    key: str,
) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    n_ok = 0
    for _i in range(n):
        good = True
        for j in range(len(y)):
            noise = rng.normal(0.0, sigma, size=X[j].shape)
            xp = unit_or_zero(X[j] + noise)
            if geometric_margin(w, b, xp, float(y[j])) <= 0.0:
                good = False
                break
        if good:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def perturb_kernel_stable(
    X_tr: np.ndarray,
    alpha: np.ndarray,
    X_te: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float,
    domain: str,
    key: str,
) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sig = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    n_ok = 0
    for _i in range(n):
        good = True
        for j in range(len(y)):
            noise = rng.normal(0.0, sig, size=X_te[j].shape)
            xp = unit_or_zero(X_te[j] + noise).reshape(1, -1)
            sc = float((_rbf(xp, X_tr, gamma) @ alpha)[0])
            if sc * float(y[j]) <= 0.0:
                good = False
                break
        if good:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def _linear_pass(
    *,
    status: str,
    train_rank: bool,
    probe_rank: bool,
    train_g: float,
    probe_g: float,
    gmin: float,
    stab: bool,
) -> bool:
    return bool(
        status == "optimal"
        and train_rank
        and probe_rank
        and train_g >= gmin
        and probe_g >= gmin
        and stab
    )


def eval_arm(
    rec: dict[str, Any],
    *,
    arm: str,
    aid: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    tag: str,
) -> dict[str, Any]:
    p = load_prereg()
    handles = list(world["handles"])
    teach, probe = capture_xy(rec, aid=aid, handles=handles, pairs=pairs)
    gmin = float(p["margin"]["geometric_margin_min"])
    w = np.zeros(64)
    b = 0.0
    train_g = 0.0
    probe_g = 0.0
    ranking_ok = False
    train_rank = False
    train_clean = False
    d0_margins: list[float] = []
    status = "not_applicable"
    n_sv = 0
    constraint_min = 0.0
    stab: dict[str, Any] = {"stable": False, "n_ok": 0, "n": 20}
    if arm == "D0":
        bank = ProtoBank(handles, "N0", eta=float(p["arms"]["D3"]["eta"]), c_max=1.0)
        lam = parse_lam(aid)
        tracer = EligTrace(64, handles, lam if lam is not None else 0.0)
        for tick in rec["ticks"]:
            tracer.ingest(tick["trajectory"])
            if tick["kind"] == "select" and tick.get("handle") and float(tick.get("adv") or 0.0) != 0.0:
                e = tracer.address() if lam is not None else tick_static(tick, aid)
                bank.update(str(tick["handle"]), float(tick["adv"]), e)
        winners = []
        for i, (_cue, handle) in enumerate(pairs):
            tr = tracer.copy()
            tr.ingest(rec["probes"][i]["trajectory"])
            e = tr.address() if lam is not None else tick_static(rec["probes"][i], aid)
            scores = bank.scores(e)
            win = max(scores, key=lambda h: scores[h]) if scores else None
            winners.append(win == handle)
            others = [v for k, v in scores.items() if k != win] if win else [0.0]
            d0_margins.append(float(scores.get(win or "", 0.0) - (max(others) if others else 0.0)))
        ranking_ok = all(winners)
        train_rank = ranking_ok
        probe_g = min(d0_margins) if d0_margins else 0.0
        passed = bool(ranking_ok and probe_g >= float(p["d0_control"]["cosine_margin_min"]))
        status = "not_applicable"
    elif arm == "D1":
        fit = hard_margin_linear(teach.X, teach.y, p["arms"]["D1"])
        w, b, status = fit["w"], fit["b"], fit["status"]
        n_sv, constraint_min = int(fit["n_sv"]), float(fit["constraint_min"])
        train_g = min_geometric_margin(w, b, teach.X, teach.y)
        probe_g = min_geometric_margin(w, b, probe.X, probe.y)
        train_rank = bool(len(teach.y) and np.all((teach.X @ w + b) * teach.y > 0.0))
        ranking_ok = bool(len(probe.y) and np.all((probe.X @ w + b) * probe.y > 0.0))
        train_clean = bool(status == "optimal" and train_rank and train_g >= gmin)
        if status == "optimal":
            stab = perturb_sign_stable(w, b, probe.X, probe.y, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = _linear_pass(
            status=status,
            train_rank=train_rank,
            probe_rank=ranking_ok,
            train_g=train_g,
            probe_g=probe_g,
            gmin=gmin,
            stab=bool(stab["stable"]),
        )
    elif arm == "D2":
        fit = ridge_fit(teach.X, teach.y, p["arms"]["D2"])
        w, b, status = fit["w"], fit["b"], fit["status"]
        train_g = min_geometric_margin(w, b, teach.X, teach.y)
        probe_g = min_geometric_margin(w, b, probe.X, probe.y)
        train_rank = bool(len(teach.y) and np.all((teach.X @ w + b) * teach.y > 0.0))
        ranking_ok = bool(len(probe.y) and np.all((probe.X @ w + b) * probe.y > 0.0))
        train_clean = bool(status == "optimal" and train_rank and train_g >= gmin)
        stab = perturb_sign_stable(w, b, probe.X, probe.y, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = _linear_pass(
            status=status,
            train_rank=train_rank,
            probe_rank=ranking_ok,
            train_g=train_g,
            probe_g=probe_g,
            gmin=gmin,
            stab=bool(stab["stable"]),
        )
    elif arm == "D3":
        fit = competitive_fit(teach.X, teach.handles, teach.chosen, p["arms"]["D3"])
        w, b, status = fit["w"], fit["b"], fit["status"]
        train_g = min_geometric_margin(w, b, teach.X, teach.y)
        probe_g = min_geometric_margin(w, b, probe.X, probe.y)
        train_rank = bool(len(teach.y) and np.all((teach.X @ w + b) * teach.y > 0.0))
        ranking_ok = bool(len(probe.y) and np.all((probe.X @ w + b) * probe.y > 0.0))
        train_clean = bool(status == "optimal" and train_rank and train_g >= gmin)
        stab = perturb_sign_stable(w, b, probe.X, probe.y, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = _linear_pass(
            status=status,
            train_rank=train_rank,
            probe_rank=ranking_ok,
            train_g=train_g,
            probe_g=probe_g,
            gmin=gmin,
            stab=bool(stab["stable"]),
        )
    elif arm == "D4":
        fit = kernel_ridge_fit(teach.X, teach.y, p["arms"]["D4"])
        status = fit["status"]
        alpha, rkhs, gamma = fit["alpha"], float(fit["rkhs"]), float(fit["gamma"])
        train_s = _rbf(teach.X, teach.X, gamma) @ alpha
        probe_s = _rbf(probe.X, teach.X, gamma) @ alpha
        train_rank = bool(len(teach.y) and np.all(train_s * teach.y > 0.0))
        ranking_ok = bool(len(probe.y) and np.all(probe_s * probe.y > 0.0))
        train_g = 0.0 if rkhs <= EPS else float(np.min(teach.y * train_s / rkhs))
        probe_g = 0.0 if rkhs <= EPS else float(np.min(probe.y * probe_s / rkhs))
        train_clean = bool(status == "optimal" and train_rank and train_g >= gmin)
        stab = perturb_kernel_stable(
            teach.X, alpha, probe.X, probe.y, gamma=gamma, domain=world["domain"], key=f"{tag}_{arm}_{aid}"
        )
        passed = _linear_pass(
            status=status,
            train_rank=train_rank,
            probe_rank=ranking_ok,
            train_g=train_g,
            probe_g=probe_g,
            gmin=gmin,
            stab=bool(stab["stable"]),
        )
    else:
        raise RuntimeError(arm)
    require_accepted(status)
    return {
        "arm": arm,
        "address": aid,
        "passed": bool(passed),
        "solver_status": status,
        "n_sv": int(n_sv),
        "constraint_min": float(constraint_min),
        "ranking_ok": bool(ranking_ok),
        "train_ranking_ok": bool(train_rank),
        "train_clean": bool(train_clean),
        "train_geometric_margin": float(train_g),
        "probe_geometric_margin": float(probe_g),
        "perturb_stable": bool(stab.get("stable")),
        "perturb_n_ok": int(stab.get("n_ok") or 0),
        "perturb_n": int(stab.get("n") or 0),
        "n_train": int(len(teach.y)),
        "n_probe": int(len(probe.y)),
        "w_norm": float(np.linalg.norm(w)),
        "d0_cosine_margins": d0_margins,
        "geometric_margin_min": gmin if arm != "D0" else float(p["d0_control"]["cosine_margin_min"]),
        "fit_fingerprint": teach.fingerprint.hex()[:16],
        "probe_fingerprint": probe.fingerprint.hex()[:16],
    }


def smoke() -> dict[str, Any]:
    p = load_prereg()
    x = np.ones(64, dtype=np.float64)
    x = x / np.linalg.norm(x)
    w = x.copy()
    g1 = geometric_margin(w, 0.0, x, 1.0)
    g10 = geometric_margin(10.0 * w, 0.0, x, 1.0)
    g_b = geometric_margin(w, 100.0, x, 1.0)
    scale_ok = bool(abs(g1 - g10) <= 1e-12 and abs(g1 - 1.0) <= 1e-9)
    bias_excluded = bool(abs(g_b - 101.0) <= 1e-9)
    Xsep = np.stack([x, -x])
    ysep = np.asarray([1.0, -1.0])
    sep = hard_margin_linear(Xsep, ysep, p["arms"]["D1"])
    Xsame = np.stack([x, x])
    yopp = np.asarray([1.0, -1.0])
    inf = hard_margin_linear(Xsame, yopp, p["arms"]["D1"])
    world = capacity_world(0, "TM024.DISCRIMMAP.R2.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    rec = collect_stream(world, pairs, tag="dmr2_smk", probe_pairs=pairs)
    d0 = eval_arm(rec, arm="D0", aid="E1", world=world, pairs=pairs, tag="smk")
    d1 = eval_arm(rec, arm="D1", aid="E1", world=world, pairs=pairs, tag="smk")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "geometric_margin_scale_invariant": scale_ok,
        "geometric_margin_excludes_bias": bias_excluded,
        "hard_margin_separable_status": sep["status"],
        "hard_margin_infeasible_status": inf["status"],
        "no_soft_margin": p["arms"]["D1"]["no_automatic_soft_margin"],
        "geometric_margin_min": p["margin"]["geometric_margin_min"],
        "d0_e1_2cue_passed": d0["passed"],
        "d1_e1_2cue_status": d1["solver_status"],
        "d1_e1_2cue_train_g": d1["train_geometric_margin"],
        "d1_e1_2cue_probe_g": d1["probe_geometric_margin"],
        "expected_n_cells": EXPECTED_N_CELLS,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "eligibility_budget_installed": False,
        "env": torch_env(),
    }


def _eval_block(
    rec: dict[str, Any],
    *,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    n_cues: int,
    wi: int,
    kind: str,
    tag: str,
    required: bool,
) -> list[dict[str, Any]]:
    p = load_prereg()
    rows = []
    for aid in p["addresses"]:
        for arm in ARMS:
            out = eval_arm(rec, arm=arm, aid=aid, world=world, pairs=pairs, tag=f"{tag}_{arm}_{aid}")
            cid = cell_id(kind, arm, aid, n_cues, order, wi)
            out.update(
                {
                    "id": cid,
                    "kind": kind,
                    "order": order,
                    "n_cues": n_cues,
                    "n_handles": 2,
                    "world": wi,
                    "required": required,
                    "domain": world["domain"],
                }
            )
            rows.append(out)
    return rows


def _decision(all_cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, bool]]:
    def rows(arm: str, n_cues: int, kind: str) -> list[dict[str, Any]]:
        return [
            c
            for c in all_cells
            if c["arm"] == arm and c["n_cues"] == n_cues and c["kind"] == kind
        ]

    def all_pass(rs: list[dict[str, Any]]) -> bool:
        return bool(rs) and all(bool(r["passed"]) for r in rs)

    d1_8 = rows("D1", 8, "rank")
    d1_twin = rows("D1", 2, "twin")
    d3_8 = rows("D3", 8, "rank")
    d3_twin = rows("D3", 2, "twin")
    d4_8 = rows("D4", 8, "rank")
    d4_twin = rows("D4", 2, "twin")
    d1_robust = all_pass(d1_8) and all_pass(d1_twin)
    d3_robust = all_pass(d3_8) and all_pass(d3_twin)
    d4_robust = all_pass(d4_8) and all_pass(d4_twin)
    d1_train_clean = bool(d1_8) and all(bool(c["train_clean"]) for c in d1_8)
    flags = {
        "d1_robust": d1_robust,
        "d3_robust": d3_robust,
        "d4_robust": d4_robust,
        "d1_train_clean": d1_train_clean,
        "d0_8cue_pass": all_pass(rows("D0", 8, "rank")),
    }
    ladder = p["decision_ladder"]
    if d1_robust and d3_robust:
        return ladder[0]["id"], ladder[0]["then"], flags
    if d1_robust and not d3_robust:
        return ladder[1]["id"], ladder[1]["then"], flags
    if (not d1_robust) and d4_robust:
        return ladder[2]["id"], ladder[2]["then"], flags
    if (not d1_robust) and d1_train_clean:
        return ladder[3]["id"], ladder[3]["then"], flags
    return ladder[4]["id"], ladder[4]["then"], flags


def run_dev() -> dict[str, Any]:
    if CANDIDATE_V31.exists():
        raise RuntimeError("v31 candidate must not exist")
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    cells: list[dict[str, Any]] = []
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
                raise RuntimeError("SCORE identifier appeared in DEV payload")
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
                rec = collect_stream(world, seq, tag=f"dmr2_{wi}_{n_cues}_{order}", probe_pairs=pairs)
                assert rec["ticks"] and rec["probes"] and rec["taught"]
                assert len(rec["probes"]) == n_cues
                cells.extend(
                    _eval_block(
                        rec,
                        world=world,
                        pairs=pairs,
                        order=order,
                        n_cues=n_cues,
                        wi=wi,
                        kind="rank",
                        tag=f"rank{wi}_{n_cues}_{order}",
                        required=True,
                    )
                )
    twin_cells: list[dict[str, Any]] = []
    for order in TEACH_ORDERS:
        world = capacity_world(0, TWIN_DOMAIN, n_cues=2, n_handles=2)
        pairs = mapping_pairs(world, flip=False)
        seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
        rec = collect_stream(world, seq, tag=f"dmr2_twin_{order}", probe_pairs=pairs)
        twin_cells.extend(
            _eval_block(
                rec,
                world=world,
                pairs=pairs,
                order=order,
                n_cues=2,
                wi=0,
                kind="twin",
                tag=f"twin_{order}",
                required=True,
            )
        )
    all_cells = cells + twin_cells
    ids = [c["id"] for c in all_cells]
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    for c in all_cells:
        if int(c["n_train"]) != int(c["n_cues"]) or int(c["n_probe"]) != int(c["n_cues"]):
            raise RuntimeError(f"empty or mismatched teach/probe {c['id']}")
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        require_accepted(str(c["solver_status"]))
    code, then, flags = _decision(all_cells, p)
    out = {
        "version": "TM.0.24.DISCRIMMAP.R2.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain_opened": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "d0_8cue_pass": flags["d0_8cue_pass"],
        "d1_robust": flags["d1_robust"],
        "d3_robust": flags["d3_robust"],
        "d4_robust": flags["d4_robust"],
        "d1_train_clean": flags["d1_train_clean"],
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(all_cells),
        "n_rank": len(cells),
        "n_twin": len(twin_cells),
        "cells": all_cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": r2_shas(),
        "note": "R2 DEV only. Historical DISCRIMMAP worlds not rescored. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("discrimmap R2 runner.lock already exists")
    if CANDIDATE_V31.exists():
        raise RuntimeError("v31 candidate must not exist")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.DISCRIMMAP.R2.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "shas": r2_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "addresses": prereg["addresses"],
        "arms": list(ARMS),
        "d1": prereg["arms"]["D1"],
        "d2": prereg["arms"]["D2"],
        "d3": prereg["arms"]["D3"],
        "d4": prereg["arms"]["D4"],
        "geometric_margin_min": prereg["margin"]["geometric_margin_min"],
        "w_norm_excludes_intercept": True,
        "reject_raw_linear_margin": True,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "expected_n_rank": EXPECTED_N_RANK,
        "expected_n_twin": EXPECTED_N_TWIN,
        "expected_n_cells": EXPECTED_N_CELLS,
        "git_head": _git_head(),
        "note": "Frozen R2 runner. DEV lock only after this file is on origin/main. No neural edit.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    refuse_rerun()
    blob = json.dumps(out, default=str)
    refuse_score_markers(blob)
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("discrimmap R2 decision lock already exists")
    out = {
        "version": "TM.0.24.DISCRIMMAP.R2.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "scored_worlds": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "candidate_v31": False,
        "candidate_v32": False,
        "lineage_reopened": False,
        "q3": False,
        "eligibility_budget_installed": False,
        "declared_budget_remains_closed": 1536,
        "n1n2_secondary": True,
        "historical_discrimmap_preserved": True,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "d1_robust": bool(dev.get("d1_robust")),
            "d3_robust": bool(dev.get("d3_robust")),
            "d4_robust": bool(dev.get("d4_robust")),
            "d1_train_clean": bool(dev.get("d1_train_clean")),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "Pinned R2 runner-only diagnostic. Historical DISCRIMMAP preserved. "
            "No v31/v32 candidate. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.DISCRIMMAP.R2 DEV\n\n"
        f"Decision: **{out['decision']['code']}**. "
        f"D1 robust: **{out['decision']['d1_robust']}**. "
        f"D3 robust: **{out['decision']['d3_robust']}**. "
        f"D4 robust: **{out['decision']['d4_robust']}**. "
        f"D1 train clean: **{out['decision']['d1_train_clean']}**.\n\n"
        f"Next: `{out['decision']['then']}`. Historical DISCRIMMAP not rescored. "
        "SCORE unopened. No neural candidate. 1536 eligibility budget stays closed. "
        "Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze authorizes a competitive law on origin/main")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("DISCRIMMAP R2 DEV lock requires runner.lock on origin/main after this freeze")
    assert_runner_frozen()
    refuse_rerun()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--write-dev-lock", action="store_true")
    ap.add_argument("--write-decision", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.verify_prereg:
        p = load_prereg()
        assert p["n"] == 64
        assert p["neural_edit"] is False
        assert p["implementation_authorized"] is False
        assert p["margin"]["geometric_margin_min"] == 0.01
        assert p["margin"]["w_norm_excludes_intercept"] is True
        assert p["arms"]["D1"]["no_automatic_soft_margin"] is True
        assert p["arms"]["D1"]["soft_margin_C"] is None
        assert p["arms"]["D2"]["y_encoding"] == [-1, 1]
        assert p["arms"]["D2"]["intercept_in_norm"] is False
        assert p["arms"]["D3"]["epochs"] == 1
        assert p["arms"]["D3"]["error_only"] is False
        assert p["arms"]["D3"]["shuffle"] is False
        assert p["arms"]["D4"]["rbf_gamma"] == 0.5
        assert p["arms"]["D4"]["v_eligible"] is False
        assert p["declared_budget_remains_closed"] == 1536
        print(json.dumps({"ok": True, "product": p["product"], "expected_n_cells": EXPECTED_N_CELLS}, indent=2))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2, default=str))
    elif args.dev:
        out = run_dev()
        brief = {k: v for k, v in out.items() if k != "cells"}
        print(json.dumps(brief, indent=2, default=str))
    elif args.write_dev_lock:
        refuse_dev_lock()
        out = run_dev()
        write_dev_lock(out)
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_decision:
        if not DEV_LOCK.exists():
            raise RuntimeError("write DEV lock first")
        dev = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        print(json.dumps(write_decision(dev), indent=2, default=str))
    elif args.score:
        refuse_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
