"""TM.0.23.CORTEX.V28 — birth/candidate: general credit-path causality."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.cortex_mact_boundary import control_c1_v6, control_c2_v6, control_c3_v6, control_c4_v6, control_c8_v6
from experiments.cortex_v7_stats import run_c5_population, run_c6_population
from experiments.run_tm023cortex import make_cortex, run_sanity, torch_env
from three_memory.neural_cortex import (
    CONFLICT_HOLD_BIAS,
    ECHOIC_BIAS,
    ELIG_EPS,
    EQUAL_EVIDENCE_MIN_SYMBOLS,
    FAMILIARITY_ABS,
    FAMILIARITY_RATIO,
    MOTOR_ACT_TOKENS,
    OP_COST,
    OPS,
    VOCAL_REFRACTORY,
    GenomeConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"
AMEND = DOCS / "cortex_v28_architecture_amendment.lock"
PREREG = DOCS / "cortex_v28.prereg.lock"
BIRTH = DOCS / "cortex_v28_birth.lock"
MATH = DOCS / "cortex_v28_math_audit.lock"
CANDIDATE_V28 = DOCS / "cortex.candidate.v28.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V27 = DOCS / "cortex.candidate.v27.lock"
MACT = DOCS / "cortex_mact_boundary.v28.lock"
MACT_MD = DOCS / "tm023cortex_mact_boundary_v28_results.md"
NEURAL_SHA_AT_FREEZE = "2b563a9c5de3ec8b411121bd5518c09f49f422f44108138ec34a1d5708c98d2e"
LINEAGE_NEURAL_HISTORICAL = NEURAL_SHA_AT_FREEZE
V27_NEURAL = "71bece5917893fae03c3a95c276cf93bc0e34fce6a7bfb6a99adf093bb7ebc08"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _plastic_snapshot(ag) -> dict[str, torch.Tensor]:  # noqa: ANN001
    return {name: getattr(ag, name).detach().clone() for name in ag._plastic_names}


def _max_delta(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return max(float((a[k] - b[k]).abs().max().item()) for k in a)


def probe_zero_elig_no_motion() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v28_zero_elig_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(["h_x", "h_y"])
        body = np.array([0.5, 0.4, 0.5, 0.0], dtype=np.float64)
        before = _plastic_snapshot(ag)
        slow_before = {name: ag.W_slow[name].detach().clone() for name in ag._plastic_names}
        ag._pending = {
            "op": "ACT",
            "token": "h_x",
            "rho_elig": np.zeros(ag.genome.n, dtype=np.float64),
            "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
            "body": body.copy(),
            "cost": 0.05,
            "motor_vec": ag.motor_vocab["h_x"].copy(),
        }
        body2 = body.copy()
        body2[0] = 0.75
        ag._apply_credit(np.zeros(ag.genome.d_sym, dtype=np.float64), body2)
        after = _plastic_snapshot(ag)
        slow_after = {name: ag.W_slow[name].detach().clone() for name in ag._plastic_names}
        max_w = _max_delta(before, after)
        max_s = _max_delta(slow_before, slow_after)
        return {
            "id": "zero_elig_no_plastic_motion",
            "ok": bool(max_w < 1e-12 and max_s < 1e-12),
            "max_abs_delta": max_w,
            "max_slow_delta": max_s,
            "elig_eps": ELIG_EPS,
        }


def probe_active_elig_moves_act_query() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v28_active_elig_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(["h_x", "h_y"])
        body = np.array([0.5, 0.4, 0.5, 0.0], dtype=np.float64)
        w_before = ag.W_act_query.detach().clone()
        unused_before = {n: getattr(ag, n).detach().clone() for n in ("W_rec", "W_in", "W_write", "W_att", "W_emit_query")}
        rho = np.ones(ag.genome.n, dtype=np.float64)
        ag._pending = {
            "op": "ACT",
            "token": "h_x",
            "rho_elig": rho,
            "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
            "body": body.copy(),
            "cost": 0.05,
            "motor_vec": ag.motor_vocab["h_x"].copy(),
        }
        body2 = body.copy()
        body2[0] = 0.75
        ag._apply_credit(np.zeros(ag.genome.d_sym, dtype=np.float64), body2)
        d_act = float((ag.W_act_query - w_before).abs().max().item())
        unused = {n: float((getattr(ag, n) - unused_before[n]).abs().max().item()) for n in unused_before}
        return {
            "id": "active_elig_moves_credited_tensors_only",
            "ok": bool(d_act > 1e-12 and max(unused.values()) < 1e-12),
            "W_act_query_delta": d_act,
            "unused_max_delta": max(unused.values()),
            "unused": unused,
        }


def write_birth() -> dict[str, Any]:
    if not PREREG.exists() or not AMEND.exists():
        raise RuntimeError("v28 apparatus must be frozen first")
    if _sha_file(NEURAL_PY) == NEURAL_SHA_AT_FREEZE:
        raise RuntimeError("neural SHA still at freeze — implement the authorized law first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    zero = probe_zero_elig_no_motion()
    active = probe_active_elig_moves_act_query()
    if not zero["ok"] or not active["ok"]:
        raise RuntimeError(f"credit-path probes failed: {zero} {active}")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    with tempfile.TemporaryDirectory(prefix="v28birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        r = ag.bind_actuators(["h_x", "h_y"])
        if float(ag.W_act_query.abs().sum()) != 0.0:
            raise RuntimeError("W_act_query not zero at birth")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V28.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "wallmap_decision_sha": json.loads(PREREG.read_text(encoding="utf-8"))["wallmap_decision_sha"],
        "architecture_amendment_sha": _sha_file(AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "prior_lineage_neural_sha": LINEAGE_NEURAL_HISTORICAL,
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {"bind_n": r["n"], "b_op_act": b_op, "op_cost_act": OP_COST["ACT"], "n": ag.genome.n},
        "credit_probes": {"zero_elig": zero, "active_elig": active},
        "note": "v28 general credit path. LINEAGE/WALLMAP historical. No FULLDEV.R7.",
    }
    if birth["checks"]["n"] != 64:
        raise RuntimeError("n must stay 64")
    if BIRTH.exists():
        raise RuntimeError("v28 birth exists")
    BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(BIRTH)}


def write_math_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool) -> None:
        checks.append({"id": name, "ok": ok})

    src = NEURAL_PY.read_text(encoding="utf-8")
    add("v1_contract_untouched", _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2")
    add("n_stays_64", GenomeConfig().n == 64)
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    add("conflict_hold_bias_frozen", CONFLICT_HOLD_BIAS == 2.0)
    add("familiarity_ratio_frozen", FAMILIARITY_RATIO == 0.5)
    add("familiarity_abs_frozen", FAMILIARITY_ABS == 16.0)
    add("equal_evidence_min_symbols_3", EQUAL_EVIDENCE_MIN_SYMBOLS == 3)
    add("vocal_refractory_15", VOCAL_REFRACTORY == 1.5)
    add("echoic_bias_only", "echoic_bias" in src and ECHOIC_BIAS == 0.08)
    add("no_phrase_replay", "_phrase" not in src and "phrase_program" not in src)
    add("elig_eps_present", "ELIG_EPS" in src)
    add("consolidate_takes_names", "def _clip_and_consolidate(self, names" in src)
    add("no_l0_shortcut", "L0" not in src and "reachability" not in src.lower())
    add("neural_changed_from_freeze", _sha_file(NEURAL_PY) != NEURAL_SHA_AT_FREEZE)
    add("neural_changed_from_v27_ancestor", _sha_file(NEURAL_PY) != V27_NEURAL)
    zero = probe_zero_elig_no_motion()
    active = probe_active_elig_moves_act_query()
    add(zero["id"], bool(zero["ok"]))
    add(active["id"], bool(active["ok"]))
    out = {
        "version": "TM.0.23.CORTEX.V28.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "zero_elig": zero,
        "active_elig": active,
    }
    if MATH.exists():
        raise RuntimeError("math audit exists")
    MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate() -> dict[str, Any]:
    if not BIRTH.exists() or not MATH.exists() or not json.loads(MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("birth + passing math audit required")
    if CANDIDATE_V28.exists():
        raise RuntimeError("candidate.v28 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V28",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v27_sha": _sha_file(CANDIDATE_V27),
        "v28_birth_sha": _sha_file(BIRTH),
        "architecture_amendment_sha": _sha_file(AMEND),
        "math_audit_sha": _sha_file(MATH),
        "learning_law_ok": True,
        "gpu_scoring_ready": True,
        "all_sanity_ok": True,
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "note": "Candidate v28 credit-path causality. Must re-earn C4/C5/C6 then a new reachability diagnostic. FULLDEV.R7 sealed. n=64.",
    }
    CANDIDATE_V28.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v28_sha": _sha_file(CANDIDATE_V28)}


def run_boundary(*, write_lock: bool = False) -> dict[str, Any]:
    if _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V28.read_text(encoding="utf-8"))["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    results = [
        control_c1_v6(),
        control_c2_v6(),
        control_c3_v6(),
        control_c4_v6(),
        run_c5_population(),
        run_c6_population(),
        control_c8_v6(),
    ]
    required = ("C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population")
    summary = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V28",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "candidate": "docs/cortex.candidate.v28.lock",
        "candidate_sha": _sha_file(CANDIDATE_V28),
        "required_ids": list(required),
        "all_required_green": all(next(x for x in results if x["id"] == i).get("ok") for i in required),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_controls": len(results),
        "controls": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "env": torch_env(),
        "git_head": _git_head(),
    }
    if write_lock:
        if MACT.exists():
            raise RuntimeError("boundary v28 exists")
        MACT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        MACT_MD.write_text(
            "# TM.0.23.CORTEX M_act boundary (v28)\n\n"
            + "\n".join(f"- `{r['id']}`: **{'PASS' if r.get('ok') else 'FAIL'}**" for r in summary["controls"])
            + "\n\nn stays 64. LINEAGE/WALLMAP historical. FULLDEV.R7 sealed.\n",
            encoding="utf-8",
        )
        summary["locks_written"] = True
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write-birth", action="store_true")
    ap.add_argument("--math-audit", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--mact-boundary", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()
    if args.write_birth:
        print(json.dumps(write_birth(), indent=2, default=str))
    elif args.math_audit:
        print(json.dumps(write_math_audit(), indent=2, default=str))
    elif args.write_candidate:
        print(json.dumps(write_candidate(), indent=2))
    elif args.mact_boundary:
        print(json.dumps(run_boundary(write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
