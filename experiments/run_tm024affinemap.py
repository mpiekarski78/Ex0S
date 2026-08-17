"""TM.0.24.AFFINEMAP — affine vs homogeneous acquisition on the frozen L2 store.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
LIFECYCLEMARGINMAP / R2 / V1 locks are immutable. Write-geometry closed. W1 not resurrected.
A3 is diagnostic only and not authorized. DEV on unused TM024.AFFINEMAP.DEV. after this
freeze is on origin/main. SCORE reserved and unopened. No trace or neural candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024convergencemap import PassiveAggressive, desired_pair
from experiments.run_tm024discrimmap_r2 import hard_margin_linear
from experiments.run_tm024eligmap import _fresh, record_rest, unit_or_zero
from experiments.run_tm024lifecyclemarginmap import (
    DEV_DOMAIN as LMM_DEV_DOMAIN,
    HardMarginOracle,
    attach_probe_traj,
    fit_oracle_traced,
    load_prereg as load_lmm_prereg,
    make_store as make_lmm_store,
    replay_traced,
)
from experiments.run_tm024memorylifecyclemap_r2 import (
    EpisodeStore,
    desired_winner,
    ingest,
    probe_pairs,
    snapshot_rows,
)
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024tracebridge import require_query, teach_bridged
from experiments.run_tm024writegeom import capacity_world, mapping_pairs, ranking_margin
from three_memory.cortex_lineage import sha_file

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_affinemap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_affinemap.contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_affinemap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_affinemap.runner.lock"
MANIFEST = REPO_ROOT / "docs" / "lineage_affinemap.manifest.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_affinemap.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_affinemap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024affinemap_results.md"
LMM_PREREG = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.prereg.lock"
LMM_CONTRACT = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.contract.md"
LMM_ISOLATION = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.isolation.lock"
LMM_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.runner.lock"
LMM_DEV = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock"
LMM_DEC = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.lock"
LMM_ADD = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.addendum.lock"
LMM_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024lifecyclemarginmap.py"
R2_PREREG = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.prereg.lock"
R2_CONTRACT = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.contract.md"
R2_ISOLATION = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.isolation.lock"
R2_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.runner.lock"
R2_DEV = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.dev.lock"
R2_DEC = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.lock"
R2_ADD = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.addendum.lock"
R2_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap_r2.py"
V1_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock"
V1_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py"
CVG_DEC = REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock"
CVG_ADD = REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock"
CVG_RUNNER = REPO_ROOT / "experiments" / "run_tm024convergencemap.py"
TB_RUNNER = REPO_ROOT / "experiments" / "run_tm024tracebridge.py"
WG_RUNNER = REPO_ROOT / "experiments" / "run_tm024writegeom.py"
D1_PREREG = REPO_ROOT / "docs" / "lineage_discrimmap.r2.prereg.lock"
D1_RUNNER = REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.AFFINEMAP.DEV."
TWIN_DOMAIN = "TM024.AFFINEMAP.TWIN."
SCORE_DOMAIN = "TM024.AFFINEMAP.SCORE."
SCORE_MARKERS = (
    "TM024.AFFINEMAP.SCORE.",
    "TM024.LIFECYCLEMARGINMAP.SCORE.",
    "TM024.MEMORYLIFECYCLEMAP.R2.SCORE.",
    "TM024.MEMORYLIFECYCLEMAP.SCORE.",
)
ARMS = ("A0", "A1", "A2", "A3")
BATCH_ARMS = ("A0", "A1")
KINDS = ("acquire", "stable", "twin")
LMM_DEC_SHA = "851d4a9312a7a8164600f53b857f65d3f50fc22fba136e52f00d3266422ddff0"
LMM_DEV_SHA = "57015fef334b533a77173bb06323e3f28e8d9bc5ad41e3453bfe126ee4a34bf8"
LMM_ADD_SHA = "d4dd4ca797d4c6c0aff6725fa79723abd870491a59f4cae41f73ca03fd75f794"
LMM_RUNNER_LOCK_SHA = "d0e5eee16752a7ae89bdfc16f3e0294ce14cfd4726aa8e71f7a8eba1c7c848dd"
LMM_RUNNER_PY_SHA = "5f7dc1a79e49c42edc45ccd7d12d4c4a8d2989a067071becc433b46c6234ddce"
LMM_PREREG_SHA = "4a753d811a14b428321f88767ce5d018a42dc047eccaa959ab83ed9f4c1ee8e2"
R2_DEC_SHA = "484c38d90582b650633e76a9a92481022a5d3c97308c72e8d51d30d6c9b266dd"
R2_DEV_SHA = "9321e57bb4f3bd1f4fe108c8fcb7751eca4fdb9da3d23401da5e5e2abd09eaed"
R2_ADD_SHA = "92321043267e95092863e1d6e0ac08256d36cdf313c11078f352471fb25c7228"
R2_RUNNER_LOCK_SHA = "c05c9254e9e1b1d6b6039d7cee43b487f83a53b2d1c0b46b60feb039dd6a1077"
R2_RUNNER_PY_SHA = "30f3c4ee67fb4e6524088cc545232682ea7c189758c06d11db0a80428af825a2"
R2_PREREG_SHA = "d1063d354cbe4162355c377d8e7a42cf6508035c9f7c6273291096477c8a2924"
V1_RUNNER_LOCK_SHA = "28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593"
V1_RUNNER_PY_SHA = "edec8809938f3f1ab77948feb3661bea4fc3e6bb1abf573a81089ae628dfc974"
D1_RUNNER_PY_SHA = "06f5f2c6edc0dffef570e75295708ea2816ea737cd0af9dab157cd94f4c26b41"
CVG_RUNNER_PY_SHA = "232cffa23619de1fcdbde7b8c82fc3de8e1c2fbe84a014a40bc27f3723cbbcf6"
EPS = 1e-12
EXPECTED_N_ACQUIRE = 48
EXPECTED_N_STABLE = 48
EXPECTED_N_TWIN = 8
EXPECTED_N_CELLS = EXPECTED_N_ACQUIRE + EXPECTED_N_STABLE + EXPECTED_N_TWIN
N_SLOTS = 8
ROW_DIM = 64
MAX_STATE_SCALARS = N_SLOTS * ROW_DIM
GMIN = 0.01


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def affinemap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "manifest": MANIFEST,
        "candidate_v30": CANDIDATE_V30,
        "lmm_decision": LMM_DEC,
        "lmm_dev": LMM_DEV,
        "lmm_addendum": LMM_ADD,
        "lmm_runner_lock": LMM_RUNNER_LOCK,
        "lmm_runner": LMM_RUNNER_PY,
        "lmm_prereg": LMM_PREREG,
        "r2_decision": R2_DEC,
        "r2_dev": R2_DEV,
        "r2_addendum": R2_ADD,
        "r2_runner_lock": R2_RUNNER_LOCK,
        "r2_runner": R2_RUNNER_PY,
        "r2_prereg": R2_PREREG,
        "v1_runner_lock": V1_RUNNER_LOCK,
        "v1_runner": V1_RUNNER_PY,
        "convergencemap_decision": CVG_DEC,
        "convergencemap_addendum": CVG_ADD,
        "convergencemap_runner": CVG_RUNNER,
        "tracebridge_runner": TB_RUNNER,
        "writegeom_runner": WG_RUNNER,
        "discrimmap_r2_prereg": D1_PREREG,
        "discrimmap_r2_runner": D1_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, arm: str, n_cues: int, order: str, world: int) -> str:
    if kind not in KINDS:
        raise RuntimeError(f"kind must be one of {KINDS}, got {kind}")
    if arm not in ARMS:
        raise RuntimeError(f"unknown arm {arm}")
    return f"{kind}|{arm}|c{n_cues}|{order}|w{world}"


def assert_historical_frozen() -> None:
    if sha_file(LMM_DEC) != LMM_DEC_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP decision.lock must remain the published freeze")
    if sha_file(LMM_DEV) != LMM_DEV_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP DEV lock must remain the published freeze")
    if sha_file(LMM_ADD) != LMM_ADD_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP decision addendum must remain the published freeze")
    if sha_file(LMM_RUNNER_LOCK) != LMM_RUNNER_LOCK_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP runner.lock must remain the published freeze")
    if sha_file(LMM_RUNNER_PY) != LMM_RUNNER_PY_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP runner.py must remain the published freeze")
    if sha_file(LMM_PREREG) != LMM_PREREG_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP prereg.lock must remain the published freeze")
    if sha_file(R2_DEC) != R2_DEC_SHA:
        raise RuntimeError("R2 decision.lock must remain the published freeze")
    if sha_file(R2_DEV) != R2_DEV_SHA:
        raise RuntimeError("R2 DEV lock must remain the published freeze")
    if sha_file(R2_ADD) != R2_ADD_SHA:
        raise RuntimeError("R2 decision addendum must remain the published freeze")
    if sha_file(R2_RUNNER_LOCK) != R2_RUNNER_LOCK_SHA:
        raise RuntimeError("R2 runner.lock must remain the published freeze")
    if sha_file(R2_RUNNER_PY) != R2_RUNNER_PY_SHA:
        raise RuntimeError("R2 runner.py must remain the published freeze")
    if sha_file(R2_PREREG) != R2_PREREG_SHA:
        raise RuntimeError("R2 prereg.lock must remain the published freeze")
    if sha_file(V1_RUNNER_LOCK) != V1_RUNNER_LOCK_SHA:
        raise RuntimeError("V1 runner.lock must remain the published freeze")
    if sha_file(V1_RUNNER_PY) != V1_RUNNER_PY_SHA:
        raise RuntimeError("V1 runner.py must remain the published freeze")
    if sha_file(D1_RUNNER) != D1_RUNNER_PY_SHA:
        raise RuntimeError("DISCRIMMAP R2 runner.py must remain the published freeze")
    if sha_file(CVG_RUNNER) != CVG_RUNNER_PY_SHA:
        raise RuntimeError("CONVERGENCEMAP runner.py must remain the published freeze")


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no affinemap runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    shas = affinemap_shas()
    if shas != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    assert_historical_frozen()
    return lock


def refuse_rerun() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("same frozen DEV execution requested again")


def refuse_score_markers(payload: str) -> None:
    for mark in SCORE_MARKERS:
        if mark in payload:
            raise RuntimeError("SCORE identifier appeared in DEV payload")


def refuse_v31() -> None:
    if CANDIDATE_V31.exists() or CANDIDATE_V32.exists():
        raise RuntimeError("v31/v32 candidate must not exist")


def hard_margin_homogeneous(X: np.ndarray, y: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    """Exact hard-margin SVM constrained to b=0. Does not edit DISCRIMMAP R2."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    empty = {
        "w": np.zeros(d, dtype=np.float64),
        "b": 0.0,
        "status": "infeasible",
        "n_sv": 0,
        "constraint_min": 0.0,
    }
    if spec.get("soft_margin") or spec.get("soft_margin_C") is not None:
        raise RuntimeError("soft-margin is refused")
    if spec.get("intercept") is True:
        raise RuntimeError("homogeneous solver must not enable intercept")
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
        q = (y_s[:, None] * y_s[None, :]) * (X_s @ X_s.T)
        rhs = np.ones(m, dtype=np.float64)
        try:
            alpha = np.linalg.solve(q, rhs)
        except np.linalg.LinAlgError:
            alpha, residuals, _rank, _s = np.linalg.lstsq(q, rhs, rcond=None)
            if residuals.size and float(residuals[0]) > kkt_max:
                continue
        resid = float(np.linalg.norm(q @ alpha - rhs))
        if resid > kkt_max:
            continue
        if np.any(alpha < alpha_min - ctor):
            continue
        w = X_s.T @ (alpha * y_s)
        g = y * (X @ w)
        if float(np.min(g)) < 1.0 - ctor:
            continue
        nrm = float(np.linalg.norm(w))
        if nrm < best_norm:
            best_norm = nrm
            best = {
                "w": w,
                "b": 0.0,
                "status": "optimal",
                "n_sv": int(m),
                "constraint_min": float(np.min(g)),
            }
    return best if best is not None else empty


class HomogeneousOracle(HardMarginOracle):
    """Runner-only D1 ceiling constrained to b=0. Not installed."""

    homogeneous = True

    def fit(self, rows: list[dict[str, Any]]) -> None:
        h0, h1 = self.handles
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for r in rows:
            want = desired_winner(self.handles, str(r["handle"]), float(r["adv"]))
            if want not in (h0, h1):
                continue
            xs.append(unit_or_zero(r["p1"]))
            ys.append(1.0 if want == h0 else -1.0)
        if len(xs) < 2 or len(set(ys)) < 2:
            self.w = np.zeros(64, dtype=np.float64)
            self.b = 0.0
            self.status = "infeasible"
            self.n_sv = 0
            self.n_updates += 1
            return
        fit = hard_margin_homogeneous(np.stack(xs, axis=0), np.asarray(ys, dtype=np.float64), self.spec)
        self.w = np.asarray(fit["w"], dtype=np.float64).reshape(-1)
        self.b = 0.0
        self.status = str(fit["status"])
        self.n_sv = int(fit.get("n_sv") or 0)
        self.n_updates += 1


class AffineSep:
    def __init__(self, handles: list[str], w: np.ndarray, b: float):
        self.handles = list(handles)
        self.w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.b = float(b)

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        x = unit_or_zero(addr)
        s = float(np.dot(self.w, x) + self.b)
        h0, h1 = self.handles
        return {h0: s, h1: -s}


class PassiveAggressiveBias:
    """Diagnostic PA plus per-handle bias. Not an instinct. Not authorized."""

    def __init__(self, handles: list[str], *, gamma: float):
        self.handles = list(handles)
        self.gamma = float(gamma)
        self.pa = PassiveAggressive(self.handles, gamma=self.gamma)
        self.bias = {h: 0.0 for h in self.handles}
        self.n_updates = 0
        self.n_bias_updates = 0
        self.last_tau_bias = 0.0

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        sc = self.pa.scores(addr)
        return {h: float(sc[h]) + float(self.bias[h]) for h in self.handles}

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        self.pa.update(addr, chosen, adv)
        self.n_updates = int(self.pa.n_updates)
        if abs(float(adv)) <= EPS or chosen not in self.bias:
            self.last_tau_bias = 0.0
            return
        ch, ot = desired_pair(self.handles, chosen, adv)
        sc = self.scores(addr)
        gap = float(sc[ch]) - float(sc[ot])
        tau = 0.5 * max(0.0, self.gamma - gap)
        self.last_tau_bias = float(tau)
        if tau <= EPS:
            return
        self.bias[ch] = float(self.bias[ch] + tau)
        self.bias[ot] = float(self.bias[ot] - tau)
        self.n_bias_updates += 1


def design_xy(handles: list[str], rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    h0, h1 = handles
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for r in rows:
        want = desired_winner(handles, str(r["handle"]), float(r["adv"]))
        if want not in (h0, h1):
            continue
        xs.append(unit_or_zero(r["p1"]))
        ys.append(1.0 if want == h0 else -1.0)
    if not xs:
        return np.zeros((0, 64), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float64)


def geo_min(w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray) -> float:
    nrm = float(np.linalg.norm(w))
    if nrm <= EPS or len(y) == 0:
        return 0.0
    return float(np.min(y * (x @ w + b) / nrm))


def stored_rank_ok(w: np.ndarray, b: float, handles: list[str], rows: list[dict[str, Any]]) -> bool:
    h0, h1 = handles
    if not rows:
        return False
    for r in rows:
        want = desired_winner(handles, str(r["handle"]), float(r["adv"]))
        x = unit_or_zero(r["p1"])
        s = float(np.dot(w, x) + b)
        sc = {h0: s, h1: -s}
        wins = [h for h, v in sc.items() if v == max(sc.values())]
        if len(wins) != 1 or wins[0] != want:
            return False
    return True


def stored_min_rank(w: np.ndarray, b: float, handles: list[str], rows: list[dict[str, Any]]) -> float:
    h0, h1 = handles
    margins: list[float] = []
    for r in rows:
        want = desired_winner(handles, str(r["handle"]), float(r["adv"]))
        x = unit_or_zero(r["p1"])
        s = float(np.dot(w, x) + b)
        sc = {h0: s, h1: -s}
        margins.append(float(ranking_margin(sc, str(want)) if want else 0.0))
    return float(min(margins) if margins else 0.0)


def make_learner(arm: str, handles: list[str], p: dict[str, Any]) -> Any:
    if p.get("no_new_eta_grid") is not True:
        raise RuntimeError("new eta grid is refused")
    if p.get("a3_implementation_authorized") is True:
        raise RuntimeError("A3 must remain unauthorized")
    if arm == "A0":
        spec = p["learner"]["d1_hard_margin_affine"]
        if spec.get("intercept") is not True:
            raise RuntimeError("A0 must keep intercept free")
        return HardMarginOracle(handles, spec)
    if arm == "A1":
        spec = p["learner"]["d1_hard_margin_homogeneous"]
        if spec.get("intercept") is not False:
            raise RuntimeError("A1 must constrain b=0")
        return HomogeneousOracle(handles, spec)
    if arm == "A2":
        a = p["learner"]["c3_passive_aggressive"]
        if a.get("learning_rate_grid") or a.get("clipnorm") or a.get("intercept"):
            raise RuntimeError("A2 must remain frozen PA with b=0")
        if float(a["geometric_margin_target"]) != GMIN:
            raise RuntimeError("C3 geometric margin target must stay 0.01")
        return PassiveAggressive(handles, gamma=float(a["geometric_margin_target"]))
    if arm == "A3":
        a = p["learner"]["c3_passive_aggressive_local_bias"]
        if a.get("implementation_authorized") or a.get("not_an_instinct") is not True:
            raise RuntimeError("A3 must remain diagnostic-only")
        if float(a["bias_init"]) != 0.0:
            raise RuntimeError("A3 bias must initialize at 0")
        if float(a["geometric_margin_target"]) != GMIN:
            raise RuntimeError("A3 geometric margin target must stay 0.01")
        return PassiveAggressiveBias(handles, gamma=float(a["geometric_margin_target"]))
    raise RuntimeError(f"unknown arm {arm}")


def make_store(p: dict[str, Any]) -> EpisodeStore:
    radius = float(p["match"]["radius"])
    if radius != 0.05:
        raise RuntimeError("match radius must stay 0.05")
    if p.get("no_lifecycle_changes") is not True:
        raise RuntimeError("lifecycle changes are refused")
    return EpisodeStore(policy="replace", n_slots=N_SLOTS, match_l2=radius)


def fit_traced(learner: Any, handles: list[str], rows: list[dict[str, Any]], gmin: float) -> dict[str, Any]:
    traj = fit_oracle_traced(learner, handles, rows, gmin)
    traj["affine_b"] = float(getattr(learner, "b", 0.0))
    traj["n_sv"] = int(getattr(learner, "n_sv", 0))
    traj["homogeneous"] = bool(getattr(learner, "homogeneous", False))
    return traj


def train_store(
    arm: str,
    learner: Any,
    store: EpisodeStore,
    handles: list[str],
    rows_epochs: int,
    gmin: float,
    p: dict[str, Any],
) -> dict[str, Any]:
    if int(rows_epochs) != 16:
        raise RuntimeError("replay budget must stay 16")
    if float(p["match"]["radius"]) != 0.05:
        raise RuntimeError("match radius must stay 0.05")
    rows = store.valid_rows()
    if arm in BATCH_ARMS:
        return fit_traced(learner, handles, rows, gmin)
    return replay_traced(learner, handles, rows, rows_epochs, gmin)


def eval_phased_map(
    *,
    arm: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    learner = make_learner(arm, handles, p)
    store = make_store(p)
    captured: list[dict[str, Any]] = []
    n_live = 0
    with tempfile.TemporaryDirectory(prefix="aff_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        for i, (cue, handle) in enumerate(seq):
            rec = teach_bridged(ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}")
            n_live += 1
            ingest(store, captured, rec["addr"], rec["handle"], rec["adv"], world, cue=cue, tag=f"{tag}_cap{i}")
        if world.get("purpose") == "rename_twin":
            store.score_twin()
        store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
        traj = train_store(arm, learner, store, handles, int(p["arms"][arm]["replay_epochs"]), gmin, p)
        acquire = probe_pairs(ag, world, pairs, learner, tag=f"{tag}_acq", gmin=gmin)
        record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
        stable = probe_pairs(ag, world, pairs, learner, tag=f"{tag}_st", gmin=gmin)
    attach_probe_traj(traj, acquire, stable)
    if arm == "A3":
        traj["n_bias_updates"] = int(getattr(learner, "n_bias_updates", 0))
        traj["bias"] = {h: float(v) for h, v in getattr(learner, "bias", {}).items()}
    shared = {
        "n_live_teaches": int(n_live),
        "n_replay_updates": int(traj["n_replay_calls"]),
        "n_updates": int(getattr(learner, "n_updates", traj["n_actual_updates"])),
        "store": store.stats(),
        "matcher": store.ledger.summary(),
        "ceiling_only": arm in BATCH_ARMS,
        "a3_diagnostic_only": arm == "A3",
        "n_cues": len(pairs),
        "n_live_reversal_updates": 0,
        "matched_live_reversal": False,
        "stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_match_sanity_used_for_pass": False,
        "margin_trajectory": traj,
        "replay_learner": str(p["arms"][arm]["replay_learner"]),
        "store_policy": str(p["arms"][arm]["policy"]),
        "affine_b": traj.get("affine_b"),
        "d1_status": traj.get("d1_status"),
    }
    acq_out = {**shared, **acquire, "passed": bool(acquire["ranking_ok"]), "phase": "acquisition"}
    stab_out = {**shared, **stable, "passed": bool(stable["robust_ok"]), "phase": "stability"}
    return acq_out, stab_out


def decorate(
    out: dict[str, Any],
    *,
    kind: str,
    arm: str,
    n_cues: int,
    order: str,
    world: int,
    domain: str,
) -> dict[str, Any]:
    out.update(
        {
            "id": cell_id(kind, arm, n_cues, order, world),
            "kind": kind,
            "arm": arm,
            "n_cues": n_cues,
            "order": order,
            "world": world,
            "domain": domain,
            "required": True,
            "ceiling_only": arm in BATCH_ARMS,
            "a3_diagnostic_only": arm == "A3",
        }
    )
    return out


def acquire_all_ok(cells: list[dict[str, Any]], arm: str, n_cues: int) -> bool:
    rows = [c for c in cells if c["arm"] == arm and c["kind"] == "acquire" and int(c["n_cues"]) == int(n_cues)]
    return bool(rows) and all(bool(c["passed"]) and bool(c.get("ranking_ok")) for c in rows)


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    flags = {
        "A0_acquire_4": acquire_all_ok(cells, "A0", 4),
        "A0_acquire_8": acquire_all_ok(cells, "A0", 8),
        "A1_acquire_4": acquire_all_ok(cells, "A1", 4),
        "A1_acquire_8": acquire_all_ok(cells, "A1", 8),
        "A2_acquire_4": acquire_all_ok(cells, "A2", 4),
        "A2_acquire_8": acquire_all_ok(cells, "A2", 8),
        "A3_acquire_4": acquire_all_ok(cells, "A3", 4),
        "A3_acquire_8": acquire_all_ok(cells, "A3", 8),
        "m1_previously_failed_four_cue_acquire": True,
        "a3_implementation_authorized": bool(p.get("a3_implementation_authorized")),
    }
    a0_48 = flags["A0_acquire_4"] and flags["A0_acquire_8"]
    a1_48 = flags["A1_acquire_4"] and flags["A1_acquire_8"]
    a2_48 = flags["A2_acquire_4"] and flags["A2_acquire_8"]
    a3_48 = flags["A3_acquire_4"] and flags["A3_acquire_8"]
    a1_fail = (not flags["A1_acquire_4"]) or (not flags["A1_acquire_8"])
    a2_fail = (not flags["A2_acquire_4"]) or (not flags["A2_acquire_8"])
    both_fail_4 = (not flags["A0_acquire_4"]) and (not flags["A1_acquire_4"])
    both_fail_8 = (not flags["A0_acquire_8"]) and (not flags["A1_acquire_8"])
    ladder = [(r["id"], r["then"]) for r in p["decision_ladder"]]
    if a0_48 and a1_fail:
        code, then = ladder[0]
    elif a1_48 and a2_fail:
        code, then = ladder[1]
    elif a2_48:
        code, then = ladder[2]
    elif a3_48 and a2_fail:
        code, then = ladder[3]
    elif both_fail_4 or both_fail_8:
        code, then = ladder[4]
    else:
        code, then = ladder[5]
    return code, then, flags


def expected_cell_ids() -> list[str]:
    ids: list[str] = []
    for arm in ARMS:
        for n in (2, 4, 8):
            for wi in range(2):
                for order in TEACH_ORDERS:
                    ids.append(cell_id("acquire", arm, n, order, wi))
                    ids.append(cell_id("stable", arm, n, order, wi))
        for order in TEACH_ORDERS:
            ids.append(cell_id("twin", arm, 2, order, 1))
    return ids


def cell_manifest_hash(cells: list[dict[str, Any]]) -> str:
    rows = []
    for c in cells:
        traj = c.get("margin_trajectory") or {}
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "passed": c["passed"],
                "ranking_ok": c.get("ranking_ok"),
                "min_probe_margin": c.get("min_probe_margin"),
                "affine_b": traj.get("affine_b"),
                "d1_status": traj.get("d1_status"),
                "pre_rest_min_probe_margin": traj.get("pre_rest_min_probe_margin"),
                "post_rest_min_probe_margin": traj.get("post_rest_min_probe_margin"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_cell_coverage(cells: list[dict[str, Any]]) -> str:
    ids = [c["id"] for c in cells]
    expected = expected_cell_ids()
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    if set(ids) != set(expected):
        raise RuntimeError("cell IDs do not match frozen AFFINEMAP grid")
    kinds = Counter(c["kind"] for c in cells)
    if dict(kinds) != {
        "acquire": EXPECTED_N_ACQUIRE,
        "stable": EXPECTED_N_STABLE,
        "twin": EXPECTED_N_TWIN,
    }:
        raise RuntimeError(f"kind counts {dict(kinds)}")
    for c in cells:
        if c["kind"] in ("eco", "spec"):
            raise RuntimeError("eco/spec cells are refused")
        if int(c.get("n_live_reversal_updates") or 0) != 0:
            raise RuntimeError(f"live reversal on {c['id']}")
        if c["arm"] in BATCH_ARMS and not c.get("ceiling_only", False):
            raise RuntimeError("A0/A1 must remain ceiling-only")
        if c["arm"] == "A3" and not c.get("a3_diagnostic_only", False):
            raise RuntimeError("A3 must remain diagnostic-only")
        if SCORE_DOMAIN in str(c.get("domain") or ""):
            raise RuntimeError("SCORE identifier appeared in DEV payload")
        store = c.get("store")
        if store is not None and int(store.get("n_p1_scalars") or 0) > MAX_STATE_SCALARS:
            raise RuntimeError("episode store exceeded 512 state scalars")
        if c.get("margin_trajectory") is None:
            raise RuntimeError(f"missing margin trajectory {c['id']}")
        if c.get("bounded_match_sanity_used_for_pass"):
            raise RuntimeError(f"bounded match used as stability gate {c['id']}")
    return cell_manifest_hash(cells)


def _append_arm(cells: list[dict[str, Any]], arm: str) -> None:
    p = load_prereg()
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
                raise RuntimeError("SCORE identifier appeared in DEV payload")
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                acq, stab = eval_phased_map(
                    arm=arm,
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"aff_{arm}_{wi}_{n_cues}_{order}",
                )
                cells.append(
                    decorate(acq, kind="acquire", arm=arm, n_cues=n_cues, order=order, world=wi, domain=world["domain"])
                )
                cells.append(
                    decorate(stab, kind="stable", arm=arm, n_cues=n_cues, order=order, world=wi, domain=world["domain"])
                )
    world_t = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
    world_t["purpose"] = "rename_twin"
    pairs_t = mapping_pairs(world_t, flip=False)
    for order in TEACH_ORDERS:
        acq, _stab = eval_phased_map(
            arm=arm, world=world_t, pairs=pairs_t, order=order, tag=f"aff_{arm}_twin_{order}"
        )
        cells.append(decorate(acq, kind="twin", arm=arm, n_cues=2, order=order, world=1, domain=world_t["domain"]))


def run_dev() -> dict[str, Any]:
    refuse_v31()
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    if p.get("skip_eco_spec") is not True or p.get("no_lifecycle_changes") is not True:
        raise RuntimeError("lifecycle freeze violated")
    if p.get("a3_implementation_authorized") is True:
        raise RuntimeError("A3 must remain unauthorized")
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        _append_arm(cells, arm)
    for c in cells:
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        if c.get("stability_gate") != "ranking_perturb_sigma_0.01":
            raise RuntimeError(f"stability gate missing {c['id']}")
        if int(c.get("n_live_reversal_updates") or 0) != 0:
            raise RuntimeError(f"live reversal {c['id']}")
    manifest = assert_cell_coverage(cells)
    code, then, extra = _decision(cells, p)
    out = {
        "version": "TM.0.24.AFFINEMAP.DEV",
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
        "a3_implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "phase_flags": extra,
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells),
        "manifest_sha": manifest,
        "cells": cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": affinemap_shas(),
        "note": "AFFINEMAP DEV only. LIFECYCLEMARGINMAP freeze preserved. A3 diagnostic-only. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def published_m3_live(n_cues: int, wi: int, order: str) -> dict[str, Any] | None:
    if not LMM_DEV.exists():
        return None
    dev = json.loads(LMM_DEV.read_text(encoding="utf-8"))
    cid = f"acquire|M3|c{n_cues}|{order}|w{wi}"
    for c in dev["cells"]:
        if c["id"] == cid:
            return {
                "passed": c["passed"],
                "ranking_ok": c.get("ranking_ok"),
                "min_probe_margin": c.get("min_probe_margin"),
                "d1_status": (c.get("margin_trajectory") or {}).get("d1_status"),
            }
    return None


def capture_m3(n_cues: int, wi: int, order: str) -> dict[str, Any]:
    p = load_lmm_prereg()
    spec = p["learner"]["d1_hard_margin"]
    gmin = float(p["margin"]["native_ranking_min"])
    hom_spec = {
        **spec,
        "intercept": False,
    }
    world = capacity_world(wi, LMM_DEV_DOMAIN, n_cues=n_cues, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    learner = HardMarginOracle(handles, spec)
    store = make_lmm_store("M3", p)
    captured: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="aff_m3_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        for i, (cue, handle) in enumerate(seq):
            rec = teach_bridged(ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"aff_m3_{i}")
            ingest(store, captured, rec["addr"], rec["handle"], rec["adv"], world, cue=cue, tag=f"aff_m3_{i}")
        rows = snapshot_rows(store.valid_rows())
        learner.fit(rows)
        x, y = design_xy(handles, rows)
        w = np.asarray(learner.w, dtype=np.float64).reshape(-1)
        b = float(learner.b)
        hom = hard_margin_homogeneous(x, y, hom_spec)
        w0 = np.asarray(hom["w"], dtype=np.float64).reshape(-1)
        acquire = probe_pairs(ag, world, pairs, learner, tag="aff_m3_acq", gmin=gmin)
        zero_live = probe_pairs(ag, world, pairs, AffineSep(handles, w, 0.0), tag="aff_m3_zb", gmin=gmin)
        hom_live = probe_pairs(ag, world, pairs, AffineSep(handles, w0, 0.0), tag="aff_m3_hom", gmin=gmin)
    return {
        "id": f"acquire|M3|c{n_cues}|{order}|w{wi}",
        "n_cues": n_cues,
        "world": wi,
        "order": order,
        "n_rows": int(len(rows)),
        "affine_status": str(learner.status),
        "affine_b": b,
        "affine_w_norm": float(np.linalg.norm(w)),
        "affine_n_sv": int(learner.n_sv),
        "affine_geo_min": geo_min(w, b, x, y),
        "affine_stored_rank_ok": stored_rank_ok(w, b, handles, rows),
        "affine_stored_min_rank": stored_min_rank(w, b, handles, rows),
        "affine_live_min_probe": float(acquire["min_probe_margin"]),
        "affine_live_ranking_ok": bool(acquire["ranking_ok"]),
        "zero_b_geo_min": geo_min(w, 0.0, x, y),
        "zero_b_stored_rank_ok": stored_rank_ok(w, 0.0, handles, rows),
        "zero_b_stored_min_rank": stored_min_rank(w, 0.0, handles, rows),
        "zero_b_live_min_probe": float(zero_live["min_probe_margin"]),
        "zero_b_live_ranking_ok": bool(zero_live["ranking_ok"]),
        "hom_status": str(hom["status"]),
        "hom_b": 0.0,
        "hom_w_norm": float(np.linalg.norm(w0)),
        "hom_n_sv": int(hom.get("n_sv") or 0),
        "hom_geo_min": geo_min(w0, 0.0, x, y),
        "hom_stored_rank_ok": stored_rank_ok(w0, 0.0, handles, rows),
        "hom_stored_min_rank": stored_min_rank(w0, 0.0, handles, rows),
        "hom_live_min_probe": float(hom_live["min_probe_margin"]),
        "hom_live_ranking_ok": bool(hom_live["ranking_ok"]),
        "affine_crosscheck": hard_margin_linear(x, y, spec)["status"] if len(y) else "infeasible",
        "published": published_m3_live(n_cues, wi, order),
    }


def extract_m3() -> list[dict[str, Any]]:
    assert_historical_frozen()
    if sha_file(LMM_DEV) != LMM_DEV_SHA:
        raise RuntimeError("extract must use the frozen LIFECYCLEMARGINMAP DEV lock")
    rows: list[dict[str, Any]] = []
    for n_cues in (2, 4, 8):
        for wi in range(2):
            for order in TEACH_ORDERS:
                rec = capture_m3(n_cues, wi, order)
                rows.append(rec)
    return rows


def smoke() -> dict[str, Any]:
    p = load_prereg()
    world = capacity_world(0, "TM024.AFFINEMAP.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    a0, a0s = eval_phased_map(arm="A0", world=world, pairs=pairs, order="A_then_B", tag="affsmk0")
    a1, _a1s = eval_phased_map(arm="A1", world=world, pairs=pairs, order="A_then_B", tag="affsmk1")
    ids = expected_cell_ids()
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "w1_resurrected": False,
        "act_score_mode": "query",
        "a0_acquire_n_probe": a0["n_probe"],
        "a0_stable_n_probe": a0s["n_probe"],
        "a0_affine_b": a0.get("affine_b"),
        "a1_d1_status": a1.get("d1_status"),
        "a0_traj_keys": sorted((a0.get("margin_trajectory") or {}).keys()),
        "stability_gate": a0.get("stability_gate"),
        "expected_id_count": len(ids),
        "a3_diagnostic_only": True,
        "replay_epochs": int(p["arms"]["A2"]["replay_epochs"]),
        "match_radius": float(p["match"]["radius"]),
        "skip_eco_spec": True,
    }


def expected_ids_sha(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        raise RuntimeError("affinemap manifest already exists")
    refuse_v31()
    ids = expected_cell_ids()
    shas = {k: v for k, v in affinemap_shas().items() if k != "manifest"}
    out = {
        "version": "TM.0.24.AFFINEMAP.MANIFEST",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "lab": "TM.0.24.AFFINEMAP",
        "expected_n_cells": EXPECTED_N_CELLS,
        "expected_kind_counts": {
            "acquire": EXPECTED_N_ACQUIRE,
            "stable": EXPECTED_N_STABLE,
            "twin": EXPECTED_N_TWIN,
        },
        "id_format": "{kind}|{arm}|c{n_cues}|{order}|w{world}",
        "expected_cell_ids": ids,
        "expected_ids_sha": expected_ids_sha(ids),
        "domains": {"DEV": DEV_DOMAIN, "TWIN": TWIN_DOMAIN, "SCORE": SCORE_DOMAIN},
        "skip_eco_spec": True,
        "a3_diagnostic_only": True,
        "lifecycle_stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_cannot_satisfy_stability_gate": True,
        "historical_lmm_decision_sha": LMM_DEC_SHA,
        "shas": shas,
        "n": 64,
        "note": "AFFINEMAP cell-ID manifest. LIFECYCLEMARGINMAP freeze preserved. Product remains 0.0.004.",
    }
    MANIFEST.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("affinemap runner.lock already exists")
    refuse_v31()
    assert_historical_frozen()
    if not MANIFEST.exists():
        write_manifest()
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.AFFINEMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "a3_implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "shas": affinemap_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "arms": list(ARMS),
        "skip_eco_spec": True,
        "no_lifecycle_changes": True,
        "match_radius": float(prereg["match"]["radius"]),
        "replay_epochs": 16,
        "no_new_eta_grid": True,
        "lifecycle_stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_cannot_satisfy_stability_gate": True,
        "n_slots": N_SLOTS,
        "max_state_scalars": MAX_STATE_SCALARS,
        "a0_a1_ceiling_only": True,
        "a3_diagnostic_only": True,
        "trace_budget_unopened": 512,
        "declared_budget_remains_closed": 1536,
        "expected_n_cells": EXPECTED_N_CELLS,
        "historical_lmm_decision_sha": LMM_DEC_SHA,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen AFFINEMAP runner. LIFECYCLEMARGINMAP locks preserved. DEV lock only after this file is on origin/main. No neural edit.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    refuse_rerun()
    manifest = assert_cell_coverage(out["cells"])
    if out.get("manifest_sha") != manifest:
        raise RuntimeError("DEV manifest hash must be asserted before write")
    refuse_score_markers(json.dumps(out, default=str))
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("affinemap decision lock already exists")
    out = {
        "version": "TM.0.24.AFFINEMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "scored_worlds": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "a3_implementation_authorized": False,
        "candidate_v31": False,
        "candidate_v32": False,
        "w1_resurrected": False,
        "lineage_reopened": False,
        "eligibility_budget_installed": False,
        "trace_rows_installed": False,
        "declared_budget_remains_closed": 1536,
        "write_geometry_branch_closed": True,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "phase_flags": dev.get("phase_flags"),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": "AFFINEMAP runner-only hypothesis-class diagnostic. LIFECYCLEMARGINMAP preserved. A3 unauthorized. No v31. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.AFFINEMAP DEV\n\n"
        f"Decision: **{out['decision']['code']}**.\n\n"
        f"Phase flags: `{out['decision']['phase_flags']}`.\n\n"
        "Write-geometry closed. SCORE unopened. No neural candidate. "
        "A3 remains diagnostic-only. 512/1536 budgets stay closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze installs a sufficient write rule")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("AFFINEMAP DEV lock requires runner.lock on origin/main after this freeze")
    assert_runner_frozen()
    refuse_rerun()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--extract-m3", action="store_true")
    ap.add_argument("--extract-out", default="/tmp/affinemap_m3_extract.json")
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
        assert p["act_score_mode"] == "query"
        assert p["match"]["radius"] == 0.05
        assert p["no_new_eta_grid"] is True
        assert p["no_lifecycle_changes"] is True
        assert p["skip_eco_spec"] is True
        assert p["a3_implementation_authorized"] is False
        assert p["arms"]["A0"]["replay_learner"] == "d1_hard_margin_affine"
        assert p["arms"]["A1"]["replay_learner"] == "d1_hard_margin_homogeneous"
        assert p["arms"]["A2"]["replay_learner"] == "c3_passive_aggressive"
        assert p["arms"]["A3"]["replay_learner"] == "c3_passive_aggressive_local_bias"
        assert p["expected_n_cells"] == EXPECTED_N_CELLS
        assert_historical_frozen()
        print(json.dumps({"ok": True, "product": p["product"], "expected_n_cells": EXPECTED_N_CELLS}, indent=2))
    elif args.write_manifest:
        print(json.dumps(write_manifest(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2, default=str))
    elif args.extract_m3:
        rows = extract_m3()
        Path(args.extract_out).write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
        summary = []
        for r in rows:
            summary.append(
                {
                    "id": r["id"],
                    "affine_b": r["affine_b"],
                    "affine_live_ranking_ok": r["affine_live_ranking_ok"],
                    "zero_b_live_ranking_ok": r["zero_b_live_ranking_ok"],
                    "hom_status": r["hom_status"],
                    "hom_live_ranking_ok": r["hom_live_ranking_ok"],
                    "published": r.get("published"),
                }
            )
        print(json.dumps({"n": len(rows), "out": args.extract_out, "cells": summary}, indent=2, default=str))
    elif args.dev:
        out = run_dev()
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_dev_lock:
        refuse_dev_lock()
    elif args.write_decision:
        if not DEV_LOCK.exists():
            raise RuntimeError("DEV lock required before decision")
        dev = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        print(json.dumps(write_decision(dev), indent=2, default=str))
    elif args.score:
        refuse_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
