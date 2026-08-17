"""TM.0.24.DISCRIMMAP — runner-only separability diagnostic.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
DEV on unused TM024.DISCRIMMAP.DEV. after this freeze is on origin/main.
SCORE reserved and unopened.
"""

from __future__ import annotations

import argparse
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


ARMS = ("D0", "D1", "D2", "D3", "D4")
EXPECTED_N_RANK = 4 * 5 * 3 * 2 * 2  # addr × arm × cue × order × world
EXPECTED_N_TWIN = 4 * 5 * 2  # addr × arm × order at 2-cue TWIN
EXPECTED_N_CELLS = EXPECTED_N_RANK + EXPECTED_N_TWIN


def cell_id(kind: str, arm: str, aid: str, n_cues: int, order: str, world: int) -> str:
    return f"{kind}|{arm}|{aid}|c{n_cues}|{order}|w{world}"


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no discrimmap runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if discrimmap_shas() != lock.get("shas"):
        raise RuntimeError("discrimmap implementation drifted after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    return lock


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
        if tick["kind"] == "select" and tick.get("handle") and float(tick.get("adv") or 0.0) != 0.0:
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


def _rbf(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
    return np.exp(-d2 / (2.0 * float(sigma) ** 2 + EPS))


def perturb_kernel_stable(
    X_tr: np.ndarray,
    alpha: np.ndarray,
    X_te: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
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
            sc = float((_rbf(xp, X_tr, sigma) @ alpha)[0])
            if sc * float(y[j]) <= 0.0:
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
        ranking_ok = bool(len(y_te) and np.all((X_te @ w + b) * y_te > 0.0))
        train_rank = bool(len(y_tr) and np.all((X_tr @ w + b) * y_tr > 0.0))
        stab = perturb_sign_stable(w, b, X_te, y_te, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = bool(
            train_rank and ranking_ok and train_g >= gmin and probe_g >= gmin and stab["stable"]
        )
    elif arm == "D2":
        coef = ridge_classifier(X_tr, y_tr, float(p["arms"]["D2"]["lambda"]))
        w, b = coef[:-1], float(coef[-1])
        train_g = min_geometric_margin(w, b, X_tr, y_tr)
        probe_g = min_geometric_margin(w, b, X_te, y_te)
        ranking_ok = bool(len(y_te) and np.all((X_te @ w + b) * y_te > 0.0))
        train_rank = bool(len(y_tr) and np.all((X_tr @ w + b) * y_tr > 0.0))
        stab = perturb_sign_stable(w, b, X_te, y_te, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = bool(
            train_rank and ranking_ok and train_g >= gmin and probe_g >= gmin and stab["stable"]
        )
    elif arm == "D3":
        w, b, _rows = competitive_fit(X_tr, handles, xy["h_tr"], eta=float(p["arms"]["D3"]["eta"]))
        train_g = min_geometric_margin(w, b, X_tr, y_tr)
        probe_g = min_geometric_margin(w, b, X_te, y_te)
        ranking_ok = bool(len(y_te) and np.all((X_te @ w + b) * y_te > 0.0))
        train_rank = bool(len(y_tr) and np.all((X_tr @ w + b) * y_tr > 0.0))
        stab = perturb_sign_stable(w, b, X_te, y_te, domain=world["domain"], key=f"{tag}_{arm}_{aid}")
        passed = bool(
            train_rank and ranking_ok and train_g >= gmin and probe_g >= gmin and stab["stable"]
        )
    elif arm == "D4":
        scores, rkhs, alpha = kernel_ridge_predict(
            X_tr,
            y_tr,
            X_te,
            lam=float(p["arms"]["D4"]["lambda"]),
            sigma=float(p["arms"]["D4"]["rbf_sigma"]),
        )
        ranking_ok = bool(len(y_te) and np.all(scores * y_te > 0.0))
        probe_g = 0.0 if rkhs <= EPS else float(np.min(y_te * scores / rkhs))
        train_s, _, _ = kernel_ridge_predict(
            X_tr, y_tr, X_tr, lam=float(p["arms"]["D4"]["lambda"]), sigma=float(p["arms"]["D4"]["rbf_sigma"])
        )
        train_g = 0.0 if rkhs <= EPS else float(np.min(y_tr * train_s / rkhs))
        train_rank = bool(len(y_tr) and np.all(train_s * y_tr > 0.0))
        stab = perturb_kernel_stable(
            X_tr,
            alpha,
            X_te,
            y_te,
            sigma=float(p["arms"]["D4"]["rbf_sigma"]),
            domain=world["domain"],
            key=f"{tag}_{arm}_{aid}",
        )
        passed = bool(
            train_rank and ranking_ok and train_g >= gmin and probe_g >= gmin and stab["stable"]
        )
    else:
        raise RuntimeError(arm)
    if arm == "D0":
        train_rank = ranking_ok
    return {
        "arm": arm,
        "address": aid,
        "passed": passed,
        "ranking_ok": ranking_ok,
        "train_ranking_ok": bool(train_rank),
        "train_geometric_margin": float(train_g),
        "probe_geometric_margin": float(probe_g),
        "perturb_stable": bool(stab.get("stable")),
        "perturb_n_ok": int(stab.get("n_ok") or 0),
        "perturb_n": int(stab.get("n") or 0),
        "n_train": int(len(y_tr)),
        "n_probe": int(len(y_te)),
        "w_norm": float(np.linalg.norm(w)),
        "d0_cosine_margins": d0_margins,
        "geometric_margin_min": gmin if arm != "D0" else float(p["d0_control"]["cosine_margin_min"]),
    }


def smoke() -> dict[str, Any]:
    p = load_prereg()
    x = np.ones(64, dtype=np.float64)
    x = x / np.linalg.norm(x)
    w = x.copy()
    g1 = geometric_margin(w, 0.0, x, 1.0)
    g10 = geometric_margin(10.0 * w, 0.0, x, 1.0)
    scale_ok = bool(abs(g1 - g10) <= 1e-12 and abs(g1 - 1.0) <= 1e-9)
    world = capacity_world(0, "TM024.DISCRIMMAP.SMOKE.", n_cues=2, n_handles=2)
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


def run_dev() -> dict[str, Any]:
    if CANDIDATE_V31.exists():
        raise RuntimeError("v31 candidate must not exist")
    p = load_prereg()
    cells: list[dict[str, Any]] = []
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
                rec = collect_stream(
                    world, seq, tag=f"dm_{wi}_{n_cues}_{order}", probe_pairs=pairs
                )
                assert rec["ticks"] and rec["probes"] and rec["taught"]
                assert len(rec["probes"]) == n_cues
                assert len(rec["taught"]) == n_cues
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
        rec = collect_stream(world, seq, tag=f"dm_twin_{order}", probe_pairs=pairs)
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
        raise RuntimeError(f"cell coverage {len(ids)} unique {len(set(ids))} expected {EXPECTED_N_CELLS}")
    for c in all_cells:
        if int(c["n_train"]) != int(c["n_cues"]) or int(c["n_probe"]) != int(c["n_cues"]):
            raise RuntimeError(f"empty or mismatched teach/probe {c['id']}")
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")

    def _arm_addr(arm: str, aid: str, n_cues: int, kind: str) -> list[dict[str, Any]]:
        return [
            c
            for c in all_cells
            if c["arm"] == arm and c["address"] == aid and c["n_cues"] == n_cues and c["kind"] == kind
        ]

    def _all_pass(rows: list[dict[str, Any]]) -> bool:
        return bool(rows) and all(bool(r["passed"]) for r in rows)

    aids = list(p["addresses"])
    d1_8 = all(_all_pass(_arm_addr("D1", a, 8, "rank")) for a in aids)
    d1_twin = all(_all_pass(_arm_addr("D1", a, 2, "twin")) for a in aids)
    d1_robust = bool(d1_8 and d1_twin)
    d3_8 = all(_all_pass(_arm_addr("D3", a, 8, "rank")) for a in aids)
    d3_twin = all(_all_pass(_arm_addr("D3", a, 2, "twin")) for a in aids)
    d3_robust = bool(d3_8 and d3_twin)
    d4_8 = all(_all_pass(_arm_addr("D4", a, 8, "rank")) for a in aids)
    d0_8 = all(_all_pass(_arm_addr("D0", a, 8, "rank")) for a in aids)
    d1_train_any = any(
        bool(c["train_ranking_ok"]) and float(c["train_geometric_margin"]) >= 0.01
        for c in all_cells
        if c["arm"] == "D1" and c["kind"] == "rank" and c["n_cues"] == 8
    )
    d1_twin_or_perturb_fail = any(
        (not bool(c["passed"])) and bool(c["train_ranking_ok"])
        for c in all_cells
        if c["arm"] == "D1"
    )
    ladder = p["decision_ladder"]
    if not d1_robust:
        code, then = ladder[0]["id"], ladder[0]["then"]
    elif d1_robust and not d3_robust:
        code, then = ladder[1]["id"], ladder[1]["then"]
    elif d1_robust and d3_robust:
        code, then = ladder[2]["id"], ladder[2]["then"]
    elif d4_8 and not d1_robust:
        code, then = ladder[3]["id"], ladder[3]["then"]
    elif d1_train_any and d1_twin_or_perturb_fail:
        code, then = ladder[4]["id"], ladder[4]["then"]
    else:
        code, then = ladder[0]["id"], ladder[0]["then"]

    return {
        "version": "TM.0.24.DISCRIMMAP.DEV",
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
        "d0_8cue_pass": d0_8,
        "d1_robust": d1_robust,
        "d3_robust": d3_robust,
        "d4_8cue_pass": d4_8,
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(all_cells),
        "n_rank": len(cells),
        "n_twin": len(twin_cells),
        "cells": all_cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": discrimmap_shas(),
        "note": "DEV only. SCORE unopened. No neural edit. Product remains 0.0.004.",
    }


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("discrimmap runner.lock already exists")
    if CANDIDATE_V31.exists():
        raise RuntimeError("v31 candidate must not exist")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.DISCRIMMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "shas": discrimmap_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "addresses": prereg["addresses"],
        "arms": list(ARMS),
        "geometric_margin_min": prereg["margin"]["geometric_margin_min"],
        "reject_raw_linear_margin": True,
        "expected_n_rank": EXPECTED_N_RANK,
        "expected_n_twin": EXPECTED_N_TWIN,
        "expected_n_cells": EXPECTED_N_CELLS,
        "git_head": _git_head(),
        "note": "Frozen D-arm runner. DEV lock only after this file is on origin/main. No neural edit.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    if DEV_LOCK.exists():
        raise RuntimeError("discrimmap DEV lock already exists")
    if "TM024.DISCRIMMAP.SCORE." in json.dumps(out):
        raise RuntimeError("SCORE domain leaked into DEV lock")
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("discrimmap decision lock already exists")
    out = {
        "version": "TM.0.24.DISCRIMMAP.DECISION",
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
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "d1_robust": bool(dev.get("d1_robust")),
            "d3_robust": bool(dev.get("d3_robust")),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "Runner-only separability diagnostic. SCORE unopened. "
            "No v31/v32 candidate. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD = REPO_ROOT / "docs" / "tm024discrimmap_results.md"
    RESULT_MD.write_text(
        "# TM.0.24.DISCRIMMAP DEV\n\n"
        f"Decision: **{out['decision']['code']}**. "
        f"D1 robust: **{out['decision']['d1_robust']}**. "
        f"D3 robust: **{out['decision']['d3_robust']}**.\n\n"
        f"Next: `{out['decision']['then']}`. SCORE unopened. No neural candidate. "
        "1536 eligibility budget stays closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze authorizes a competitive law on origin/main")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("DISCRIMMAP DEV lock requires runner.lock on origin/main after this freeze")
    assert_runner_frozen()


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
