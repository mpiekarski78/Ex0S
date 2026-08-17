"""TM.0.24.DISCRIMMAP — runner-only separability diagnostic.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
DEV on unused TM024.DISCRIMMAP.DEV. after this freeze is on origin/main.
SCORE reserved and unopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024collisionmap import ridge_classifier
from experiments.run_tm024eligmap import (
    EligTrace,
    ProtoBank,
    capacity_world,
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
PREREG = REPO_ROOT / "docs" / "lineage_discrimmap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_discrimmap_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_discrimmap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_discrimmap.runner.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_discrimmap.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_discrimmap.decision.lock"
ELIG_DEC = REPO_ROOT / "docs" / "lineage_eligmap.decision.lock"
ELIG_ADD = REPO_ROOT / "docs" / "lineage_eligmap.decision.addendum.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"

DEV_DOMAIN = "TM024.DISCRIMMAP.DEV."
TWIN_DOMAIN = "TM024.DISCRIMMAP.TWIN."
SCORE_DOMAIN = "TM024.DISCRIMMAP.SCORE."
EPS = 1e-12


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def discrimmap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "eligmap_decision": ELIG_DEC,
        "eligmap_addendum": ELIG_ADD,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def geometric_margin(w: np.ndarray, b: float, x: np.ndarray, y: float) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(w))
    if not np.isfinite(n) or n <= EPS:
        return 0.0
    return float(y) * (float(np.dot(w, x)) + float(b)) / n


def min_geometric_margin(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
    if X.size == 0:
        return 0.0
    return float(min(geometric_margin(w, b, X[i], float(y[i])) for i in range(len(y))))


def max_margin_linear(X: np.ndarray, y: np.ndarray, *, steps: int = 4000, eta: float = 0.02) -> tuple[np.ndarray, float]:
    """Hard-margin SVM dual (projected gradient). No sklearn."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if n == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 64, dtype=np.float64), 0.0
    K = (X @ X.T) * np.outer(y, y)
    alpha = np.zeros(n, dtype=np.float64)
    yy = float(np.dot(y, y)) + EPS
    for _t in range(int(steps)):
        grad = 1.0 - K @ alpha
        alpha = alpha + float(eta) * grad
        alpha = np.maximum(alpha, 0.0)
        s = float(np.dot(alpha, y))
        alpha = alpha - (s / yy) * y
        alpha = np.maximum(alpha, 0.0)
    w = X.T @ (alpha * y)
    sv = alpha > (1e-8 * (float(alpha.max()) + EPS))
    if np.any(sv):
        b = float(np.mean(y[sv] - X[sv] @ w))
    else:
        b = 0.0
    return w, b


def competitive_fit(
    X: np.ndarray, handles: list[str], chosen: list[str], *, eta: float
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    h0, h1 = handles[0], handles[1]
    rows = {h: np.zeros(X.shape[1], dtype=np.float64) for h in handles}
    for i, h in enumerate(chosen):
        ehat = unit_or_zero(X[i])
        for hh in handles:
            rows[hh] = rows[hh] + (float(eta) * ehat if hh == h else -float(eta) * ehat)
    w = rows[h0] - rows[h1]
    return w, 0.0, rows


def kernel_ridge_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, *, lam: float, sigma: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """RBF kernel ridge. Returns scores, RKHS norm, train alpha."""
    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        aa = np.sum(A * A, axis=1)[:, None]
        bb = np.sum(B * B, axis=1)[None, :]
        d2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
        return np.exp(-d2 / (2.0 * float(sigma) ** 2 + EPS))

    K = rbf(X_tr, X_tr)
    alpha = np.linalg.solve(K + float(lam) * np.eye(K.shape[0]), y_tr)
    rkhs = float(np.sqrt(max(float(alpha @ K @ alpha), 0.0)))
    scores = rbf(X_te, X_tr) @ alpha
    return scores, rkhs, alpha


def capture_xy(
    rec: dict[str, Any],
    *,
    aid: str,
    handles: list[str],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
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
    for tick in rec["ticks"]:
        e = addr(tick, tracer)
        if tick["kind"] == "select" and tick.get("handle"):
            h = str(tick["handle"])
            X_tr.append(e)
            y_tr.append(1.0 if h == handles[0] else -1.0)
            h_tr.append(h)
    X_te: list[np.ndarray] = []
    y_te: list[float] = []
    h_te: list[str] = []
    for (cue, handle), pt in zip(pairs, rec["probes"], strict=True):
        tr = tracer.copy()
        e = addr(pt, tr)
        X_te.append(e)
        y_te.append(1.0 if handle == handles[0] else -1.0)
        h_te.append(handle)
    return {
        "X_tr": np.stack(X_tr) if X_tr else np.zeros((0, 64)),
        "y_tr": np.asarray(y_tr, dtype=np.float64),
        "h_tr": h_tr,
        "X_te": np.stack(X_te) if X_te else np.zeros((0, 64)),
        "y_te": np.asarray(y_te, dtype=np.float64),
        "h_te": h_te,
        "cues": [c for c, _h in pairs],
    }


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
    xy = capture_xy(rec, aid=aid, handles=handles, pairs=pairs)
    X_tr, y_tr, X_te, y_te = xy["X_tr"], xy["y_tr"], xy["X_te"], xy["y_te"]
    gmin = float(p["margin"]["geometric_margin_min"])
    w = np.zeros(64)
    b = 0.0
    train_g = 0.0
    probe_g = 0.0
    ranking_ok = False
    d0_margins: list[float] = []
    if arm == "D0":
        bank = ProtoBank(
            handles, "N0", eta=float(p["arms"]["D3"]["eta"]), c_max=1.0
        )
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
        probe_g = min(d0_margins) if d0_margins else 0.0
        passed = bool(ranking_ok and probe_g >= float(p["d0_control"]["cosine_margin_min"]))
        stab = {"stable": False, "n_ok": 0, "n": 20}
    elif arm == "D1":
        w, b = max_margin_linear(X_tr, y_tr)
        train_g = min_geometric_margin(w, b, X_tr, y_tr)
        probe_g = min_geometric_margin(w, b, X_te, y_te)
        ranking_ok = bool(np.all(np.sign(X_te @ w + b) == y_te)) if len(y_te) else False
        stab = perturb_sign_stable(w, b, X_te, y_te, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = bool(ranking_ok and probe_g >= gmin and stab["stable"])
    elif arm == "D2":
        coef = ridge_classifier(X_tr, y_tr, float(p["arms"]["D2"]["lambda"]))
        w, b = coef[:-1], float(coef[-1])
        train_g = min_geometric_margin(w, b, X_tr, y_tr)
        probe_g = min_geometric_margin(w, b, X_te, y_te)
        ranking_ok = bool(np.all(np.sign(X_te @ w + b) == y_te)) if len(y_te) else False
        stab = perturb_sign_stable(w, b, X_te, y_te, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = bool(ranking_ok and probe_g >= gmin and stab["stable"])
    elif arm == "D3":
        w, b, _rows = competitive_fit(X_tr, handles, xy["h_tr"], eta=float(p["arms"]["D3"]["eta"]))
        train_g = min_geometric_margin(w, b, X_tr, y_tr)
        probe_g = min_geometric_margin(w, b, X_te, y_te)
        ranking_ok = bool(np.all(np.sign(X_te @ w + b) == y_te)) if len(y_te) else False
        stab = perturb_sign_stable(w, b, X_te, y_te, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = bool(ranking_ok and probe_g >= gmin and stab["stable"])
    elif arm == "D4":
        scores, rkhs, _a = kernel_ridge_predict(
            X_tr,
            y_tr,
            X_te,
            lam=float(p["arms"]["D4"]["lambda"]),
            sigma=float(p["arms"]["D4"]["rbf_sigma"]),
        )
        ranking_ok = bool(np.all(np.sign(scores) == y_te)) if len(y_te) else False
        probe_g = 0.0 if rkhs <= EPS else float(np.min(y_te * scores / rkhs))
        train_s, _, _ = kernel_ridge_predict(
            X_tr, y_tr, X_tr, lam=float(p["arms"]["D4"]["lambda"]), sigma=float(p["arms"]["D4"]["rbf_sigma"])
        )
        train_g = 0.0 if rkhs <= EPS else float(np.min(y_tr * train_s / rkhs))
        stab = {"stable": ranking_ok, "n_ok": 20 if ranking_ok else 0, "n": 20}
        passed = bool(ranking_ok and probe_g >= gmin)
    else:
        raise RuntimeError(arm)
    return {
        "arm": arm,
        "address": aid,
        "passed": passed,
        "ranking_ok": ranking_ok,
        "train_geometric_margin": float(train_g),
        "probe_geometric_margin": float(probe_g),
        "perturb_stable": bool(stab.get("stable")),
        "n_train": int(len(y_tr)),
        "n_probe": int(len(y_te)),
        "w_norm": float(np.linalg.norm(w)),
        "d0_cosine_margins": d0_margins,
    }


def smoke() -> dict[str, Any]:
    p = load_prereg()
    x = np.ones(64, dtype=np.float64)
    x = x / np.linalg.norm(x)
    w = x.copy()
    g1 = geometric_margin(w, 0.0, x, 1.0)
    g10 = geometric_margin(10.0 * w, 0.0, x, 1.0)
    scale_ok = bool(abs(g1 - g10) <= 1e-12 and abs(g1 - 1.0) <= 1e-9)
    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    rec = collect_stream(world, pairs, tag="dm_smk", probe_pairs=pairs)
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
        "geometric_margin_min": p["margin"]["geometric_margin_min"],
        "d0_e1_2cue_passed": d0["passed"],
        "d1_e1_2cue_ranking": d1["ranking_ok"],
        "d1_e1_2cue_probe_g": d1["probe_geometric_margin"],
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "eligibility_budget_installed": False,
        "env": torch_env(),
    }


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze authorizes a competitive law on origin/main")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("DISCRIMMAP DEV lock requires runner.lock on origin/main after this freeze")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--write-dev-lock", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.verify_prereg:
        p = load_prereg()
        assert p["n"] == 64
        assert p["neural_edit"] is False
        assert p["implementation_authorized"] is False
        assert p["margin"]["geometric_margin_min"] == 0.01
        assert p["arms"]["D4"]["v_eligible"] is False
        assert p["declared_budget_remains_closed"] == 1536
        print(json.dumps({"ok": True, "product": p["product"]}, indent=2))
    elif args.write_dev_lock:
        refuse_dev_lock()
    elif args.score:
        refuse_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
