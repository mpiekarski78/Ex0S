"""TM.0.23.CORTEX.V30 — birth/candidate: scalar motor-tick persistence.

Default MOTOR_PERSIST_P=0 recovers v29. DEV grid found no usable p.
Not a product earn. Product 0.0.004.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cortex_mact_boundary import control_c1_v6, control_c2_v6, control_c3_v6, control_c4_v6, control_c8_v6
from experiments.cortex_v29_pipeline import (
    probe_active_elig_moves_act_query,
    probe_clamp_vs_passive,
    probe_hold_does_not_own_actor_credit,
    probe_saved_elig_not_current,
    probe_zero_elig_no_motion,
)
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
    MOTOR_PERSIST_P,
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
AMEND = DOCS / "cortex_v30_architecture_amendment.lock"
PREREG = DOCS / "cortex_v30.prereg.lock"
BIRTH = DOCS / "cortex_v30_birth.lock"
MATH = DOCS / "cortex_v30_math_audit.lock"
CANDIDATE_V30 = DOCS / "cortex.candidate.v30.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V29 = DOCS / "cortex.candidate.v29.lock"
MACT = DOCS / "cortex_mact_boundary.v30.lock"
MACT_MD = DOCS / "tm023cortex_mact_boundary_v30_results.md"
NEURAL_SHA_AT_FREEZE = "d75b8da7f251378c9638cf9a0c4a859f12b0215d9f6f7b1623e704d831f86d03"
V29_NEURAL = NEURAL_SHA_AT_FREEZE


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def probe_p0_recovers_tanh() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v30_p0_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(["h_x", "h_y"])
        body = np.array([0.5, 0.4, 0.5, 0.0], dtype=np.float64)
        zero = np.zeros(ag.genome.d_sym, dtype=np.float64)
        ag.genome.motor_persist_p = 0.0
        rho0 = ag._from_t(ag.rho).copy()
        ag._sensory_tick(zero, body, 0.0, record_sensory=False)
        after0 = ag._from_t(ag.rho).copy()
        ag.rho = ag._to_t(rho0)
        ag.genome.motor_persist_p = 0.75
        ag._sensory_tick(zero, body, 0.0, record_sensory=False)
        after_p = ag._from_t(ag.rho).copy()
        step0 = float(np.linalg.norm(after0 - rho0))
        stepp = float(np.linalg.norm(after_p - rho0))
        mix_diff = float(np.linalg.norm(after_p - after0))
        return {
            "id": "p0_recovers_tanh_mix_changes_motor_tick",
            "ok": bool(step0 > 1e-6 and mix_diff > 1e-6 and MOTOR_PERSIST_P == 0.0),
            "motor_step_p0": step0,
            "motor_step_p075": stepp,
            "mix_l2": mix_diff,
            "module_p": MOTOR_PERSIST_P,
        }


def probe_sensory_ignores_p() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v30_sens_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        body = np.array([0.5, 0.4, 0.5, 0.0], dtype=np.float64)
        inj = np.zeros(ag.genome.d_sym, dtype=np.float64)
        inj[0] = 1.0
        rho0 = ag._from_t(ag.rho).copy()
        ag.genome.motor_persist_p = 0.0
        ag._sensory_tick(inj, body, 0.0, record_sensory=True)
        a = ag._from_t(ag.rho).copy()
        ag.rho = ag._to_t(rho0)
        ag.genome.motor_persist_p = 0.9
        ag._sensory_tick(inj, body, 0.0, record_sensory=True)
        b = ag._from_t(ag.rho).copy()
        d = float(np.linalg.norm(a - b))
        return {
            "id": "sensory_ticks_ignore_persist",
            "ok": bool(d < 1e-12),
            "l2": d,
        }


def persist_probes() -> list[dict[str, Any]]:
    return [probe_p0_recovers_tanh(), probe_sensory_ignores_p()]


def credit_probes() -> list[dict[str, Any]]:
    return [
        probe_zero_elig_no_motion(),
        probe_active_elig_moves_act_query(),
        probe_hold_does_not_own_actor_credit(),
        probe_clamp_vs_passive(),
        probe_saved_elig_not_current(),
    ]


def write_birth() -> dict[str, Any]:
    if not PREREG.exists() or not AMEND.exists():
        raise RuntimeError("v30 apparatus must be frozen first")
    if _sha_file(NEURAL_PY) == NEURAL_SHA_AT_FREEZE:
        raise RuntimeError("neural SHA still at freeze — implement the authorized law first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    probes = credit_probes() + persist_probes()
    if any(not p["ok"] for p in probes):
        raise RuntimeError(f"v30 probes failed: {probes}")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    if MOTOR_PERSIST_P != 0.0:
        raise RuntimeError("no usable DEV p; module p must remain 0")
    with tempfile.TemporaryDirectory(prefix="v30birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        r = ag.bind_actuators(["h_x", "h_y"])
        if float(ag.W_act_query.abs().sum()) != 0.0:
            raise RuntimeError("W_act_query not zero at birth")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V30.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "architecture_amendment_sha": _sha_file(AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "prior_v29_neural_sha": V29_NEURAL,
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {
            "bind_n": r["n"],
            "b_op_act": b_op,
            "op_cost_act": OP_COST["ACT"],
            "n": ag.genome.n,
            "motor_persist_p": MOTOR_PERSIST_P,
        },
        "credit_probes": {p["id"]: p for p in probes},
        "note": "v30 scalar motor-tick persist. Default p=0 recovers v29. DEV found no usable p. LINEAGE closed. No FULLDEV.R7.",
    }
    if birth["checks"]["n"] != 64:
        raise RuntimeError("n must stay 64")
    if BIRTH.exists():
        raise RuntimeError("v30 birth exists")
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
    add("clamp_action_present", "def clamp_action(" in src)
    add("drop_actor_pending_present", "def drop_actor_pending(" in src)
    add("actor_pending_act_emit_only", 'if action["op"] in ("ACT", "EMIT")' in src)
    add("no_l0_shortcut", "L0" not in src and "reachability" not in src.lower())
    add("neural_changed_from_v29", _sha_file(NEURAL_PY) != V29_NEURAL)
    add("motor_persist_p_module", "MOTOR_PERSIST_P" in src)
    add("mix_only_when_not_record_sensory", "not record_sensory" in src)
    add("no_extra_l2_on_mix", "norm(" not in src.split("rho_tilde")[1].split("snap")[0] if "rho_tilde" in src else False)
    add("default_p_is_zero", MOTOR_PERSIST_P == 0.0)
    probes = credit_probes() + persist_probes()
    for p in probes:
        add(p["id"], bool(p["ok"]))
    out = {
        "version": "TM.0.23.CORTEX.V30.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "probes": {p["id"]: p for p in probes},
    }
    if MATH.exists():
        raise RuntimeError("math audit exists")
    MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate() -> dict[str, Any]:
    if not BIRTH.exists() or not MATH.exists() or not json.loads(MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("birth + passing math audit required")
    if CANDIDATE_V30.exists():
        raise RuntimeError("candidate.v30 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V30",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v29_sha": _sha_file(CANDIDATE_V29),
        "v30_birth_sha": _sha_file(BIRTH),
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
        "note": "Candidate v30 scalar motor-tick persist. Default p=0. DEV found no usable p. Not a P0-P6 pass. FULLDEV.R7 sealed. n=64.",
    }
    CANDIDATE_V30.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v30_sha": _sha_file(CANDIDATE_V30)}


def run_boundary(*, write_lock: bool = False) -> dict[str, Any]:
    if _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V30.read_text(encoding="utf-8"))["neural_cortex_sha"]:
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
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V30",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "candidate": "docs/cortex.candidate.v30.lock",
        "candidate_sha": _sha_file(CANDIDATE_V30),
        "required_ids": list(required),
        "all_required_green": all(next(x for x in results if x["id"] == i).get("ok") for i in required),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_controls": len(results),
        "controls": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "env": torch_env(),
        "git_head": _git_head(),
        "note": "p=0 default recovers v29. DEV found no usable persist p. Not a P0-P6 pass.",
    }
    if write_lock:
        if MACT.exists():
            raise RuntimeError("boundary v30 exists")
        MACT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        MACT_MD.write_text(
            "# TM.0.23.CORTEX M_act boundary (v30)\n\n"
            + "\n".join(f"- `{r['id']}`: **{'PASS' if r.get('ok') else 'FAIL'}**" for r in summary["controls"])
            + "\n\nn stays 64. Default p=0. DEV found no usable persist p. LINEAGE closed. FULLDEV.R7 sealed.\n",
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
