"""Pinned numerical joint min-change SOCP for ACT-query consolidation.

This is a solver, not an exact projector and not local plasticity.
Product 0.0.004.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

CVXPY_VERSION = "1.7.3"
CLARABEL_VERSION = "0.11.1"
SOLVER_NAME = "CLARABEL"
DTYPE = np.float64
# Clarabel 0.11.1 published defaults, except verbose and thread count.
# max_threads=1 and direct_solve_method=qdldl are deterministic pins, not fitted tols.
CLARABEL_OPTIONS: dict[str, Any] = {
    "verbose": False,
    "max_iter": 200,
    "max_threads": 1,
    "direct_solve_method": "qdldl",
    "tol_gap_abs": 1e-8,
    "tol_gap_rel": 1e-8,
    "tol_feas": 1e-8,
    "tol_infeas_abs": 1e-8,
    "tol_infeas_rel": 1e-8,
    "equilibrate_enable": True,
}


def weight_hash(W: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(W, dtype=DTYPE))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def pin_versions() -> dict[str, str]:
    import clarabel
    import cvxpy as cp

    return {
        "solver": SOLVER_NAME,
        "cvxpy": str(cp.__version__),
        "clarabel": str(clarabel.__version__),
        "required_cvxpy": CVXPY_VERSION,
        "required_clarabel": CLARABEL_VERSION,
    }


def assert_pinned_solver() -> dict[str, str]:
    pins = pin_versions()
    if pins["cvxpy"] != CVXPY_VERSION:
        raise RuntimeError(f"cvxpy pin drifted {pins['cvxpy']} != {CVXPY_VERSION}")
    if pins["clarabel"] != CLARABEL_VERSION:
        raise RuntimeError(f"clarabel pin drifted {pins['clarabel']} != {CLARABEL_VERSION}")
    return pins


def solve_min_change_socp(
    W0: np.ndarray,
    constraints: list[dict[str, np.ndarray]],
    tau: float,
    proto_eps: float,
) -> dict[str, Any]:
    """Minimize 1/2 ||W-W0||_F^2 s.t. d^T W x ≥ τ ||W^T d|| for every constraint.

    Does not apply W. Caller must validate with the organism predicate and
    install atomically or reject the entire candidate.
    """
    pins = assert_pinned_solver()
    import cvxpy as cp

    W0 = np.asarray(W0, dtype=DTYPE)
    w0_hash = weight_hash(W0)
    out: dict[str, Any] = {
        "name": "numerical_joint_socp_consolidation",
        "not_an_exact_projector": True,
        "pins": pins,
        "clarabel_options": dict(CLARABEL_OPTIONS),
        "w0_hash": w0_hash,
        "w_hash": w0_hash,
        "n_constraints": len(constraints),
        "status": "not_solved",
        "applied": False,
        "reject_reason": None,
        "objective": None,
        "frobenius_delta": 0.0,
        "solver_iters": None,
        "solve_time": None,
        "min_slack": None,
        "primal_residual": None,
        "primal_residuals": None,
        "n_zero_normal": 0,
        "W": None,
    }
    if not constraints:
        out["status"] = "no_constraints"
        out["reject_reason"] = "no_constraints"
        return out
    m, n = W0.shape
    W = cp.Variable((m, n))
    cons = []
    for c in constraints:
        d = np.asarray(c["d"], dtype=DTYPE).reshape(-1)
        x = np.asarray(c["x"], dtype=DTYPE).reshape(-1)
        if d.shape[0] != m or x.shape[0] != n:
            out["status"] = "reject"
            out["reject_reason"] = "constraint_shape"
            return out
        if float(np.linalg.norm(d)) <= float(proto_eps):
            out["status"] = "reject"
            out["reject_reason"] = "degenerate_motor_difference"
            return out
        u = W.T @ d
        cons.append(float(tau) * cp.norm(u) <= u @ x)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(W - W0)), cons)
    try:
        prob.solve(solver=cp.CLARABEL, **CLARABEL_OPTIONS)
    except Exception as exc:
        out["status"] = "reject"
        out["reject_reason"] = f"solver_exception:{type(exc).__name__}"
        return out
    stats = prob.solver_stats
    out["solver_status"] = str(prob.status)
    out["objective"] = None if prob.value is None else float(prob.value)
    out["solver_iters"] = None if stats is None else int(getattr(stats, "num_iters", 0) or 0)
    out["solve_time"] = None if stats is None else getattr(stats, "solve_time", None)
    if str(prob.status) != "optimal" or W.value is None:
        out["status"] = "reject"
        out["reject_reason"] = f"solver_status:{prob.status}"
        return out
    Wc = np.asarray(W.value, dtype=DTYPE)
    if not np.isfinite(Wc).all():
        out["status"] = "reject"
        out["reject_reason"] = "nonfinite_W"
        return out
    slacks: list[float] = []
    n_zero = 0
    for c in constraints:
        d = np.asarray(c["d"], dtype=DTYPE).reshape(-1)
        x = np.asarray(c["x"], dtype=DTYPE).reshape(-1)
        u = Wc.T @ d
        un = float(np.linalg.norm(u))
        if un <= float(proto_eps):
            n_zero += 1
            continue
        slacks.append(float(np.dot(u, x) - float(tau) * un))
    out["n_zero_normal"] = int(n_zero)
    if n_zero:
        out["status"] = "reject"
        out["reject_reason"] = "zero_normal_or_tie"
        return out
    min_slack = float(min(slacks)) if slacks else 0.0
    out["min_slack"] = min_slack
    out["primal_residuals"] = [float(s) for s in slacks]
    out["primal_residual"] = float(min(0.0, min_slack))
    if min_slack < 0.0:
        out["status"] = "reject"
        out["reject_reason"] = "soc_residual_negative"
        return out
    out["status"] = "optimal"
    out["W"] = Wc
    out["w_hash"] = weight_hash(Wc)
    out["frobenius_delta"] = float(np.linalg.norm(Wc - W0))
    return out
