"""TM.0.23.CORTEX.V12 — birth/candidate with surprise→HOLD, swapped D2 gate."""

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
from experiments.cortex_v12_gate import run_v12_gate_battery
from experiments.run_tm023cortex import make_cortex, run_sanity, torch_env
from three_memory.neural_cortex import CONFLICT_HOLD_BIAS, MOTOR_ACT_TOKENS, OP_COST, OPS, GenomeConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"
DIAG = DOCS / "cortex_diagnosis.v11.lock"
STAT = DOCS / "cortex_v12_stat_contract.lock"
AMEND = DOCS / "cortex_v12_architecture_amendment.lock"
V12_BIRTH = DOCS / "cortex_v12_birth.lock"
V12_MATH = DOCS / "cortex_v12_math_audit.lock"
CANDIDATE_V12 = DOCS / "cortex.candidate.v12.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V11 = DOCS / "cortex.candidate.v11.lock"
V12_PREREG = DOCS / "cortex_v12.prereg.lock"
V12_SEALED = DOCS / "cortex_v12_eval_secrets.sealed.json"
V12_REVEAL = DOCS / "cortex_v12_eval_reveal.lock"
V12_GATE_PREREG = DOCS / "cortex_v12_gate.prereg.lock"
V12_GATE_LOCK = DOCS / "cortex_v12_gate.lock"
V12_GATE_MD = DOCS / "tm023cortex_v12_gate_results.md"
V12_FAIL = DOCS / "cortex_v12_gate.failure.lock"
MACT_V12 = DOCS / "cortex_mact_boundary.v12.lock"
MACT_MD = DOCS / "tm023cortex_mact_boundary_v12_results.md"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_v11_scorers.py"
DEV_V12 = DOCS / "cortex_development.v12.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def write_v12_birth() -> dict[str, Any]:
    if not AMEND.exists() or not DIAG.exists() or not STAT.exists():
        raise RuntimeError("freeze v12 apparatus first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    with tempfile.TemporaryDirectory(prefix="v12birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        r = ag.bind_actuators(["h_x", "h_y"])
        nrm = float((ag.motor_vocab["h_x"] ** 2).sum() ** 0.5)
        if abs(nrm - 1.0) > 1e-6:
            raise RuntimeError(f"motor vector not unit: {nrm}")
        if float(ag.W_act_query.abs().sum()) != 0.0:
            raise RuntimeError("W_act_query not zero at birth")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V12.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "diagnosis_v11_sha": _sha_file(DIAG),
        "stat_contract_sha": _sha_file(STAT),
        "architecture_amendment_sha": _sha_file(AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "prior_v11_neural_sha": json.loads(CANDIDATE_V11.read_text(encoding="utf-8"))["neural_cortex_sha"],
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {"unit_motor": True, "bind_n": r["n"], "b_op_act": b_op, "op_cost_act": OP_COST["ACT"]},
        "note": "v12 surprise→HOLD. Must re-earn D1–D2. No DEVELOP yet.",
    }
    if V12_BIRTH.exists():
        raise RuntimeError("v12 birth exists")
    V12_BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V12_BIRTH)}


def write_v12_math_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool) -> None:
        checks.append({"id": name, "ok": ok})

    src = NEURAL_PY.read_text(encoding="utf-8")
    add("v1_contract_untouched", _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2")
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    add("skip_act_cost_retained", "skip_act_cost" in src)
    add("conflict_hold_bias_frozen", CONFLICT_HOLD_BIAS == 2.0)
    add("surprise_law_present", "_hold_after_conflict" in src and "CONFLICT_HOLD_BIAS" in src)
    add("neural_changed_from_v11", _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V11.read_text(encoding="utf-8"))["neural_cortex_sha"])
    with tempfile.TemporaryDirectory(prefix="v12audit_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        add("w_act_query_zero", float(ag.W_act_query.abs().sum()) == 0.0)
        ag.bind_actuators(["h_a", "h_b"])
        v = ag.motor_vocab["h_a"].copy()
        ag.bind_actuators(["h_b", "h_a"])
        add("bind_order_preserves_vector", bool((ag.motor_vocab["h_a"] == v).all()))
        ag._last_act_body_adv = 0.3
        ag._pending = {
            "op": "ACT",
            "token": "h_a",
            "rho_elig": ag._from_t(ag.rho),
            "s_hat": np.zeros(ag.genome.d_sym),
            "body": np.array([1.0, 0.0, 1.0, 0.0]),
            "cost": 0.05,
            "motor_vec": ag.motor_vocab["h_a"].copy(),
        }
        ag._apply_credit(np.zeros(ag.genome.d_sym), np.array([0.0, 1.0, 0.0, 0.0]))
        add("sign_flip_sets_hold_flag", ag._hold_after_conflict is True)
    out = {
        "version": "TM.0.23.CORTEX.V12.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
    if V12_MATH.exists():
        raise RuntimeError("v12 math audit exists")
    V12_MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate_v12() -> dict[str, Any]:
    if not V12_BIRTH.exists() or not V12_MATH.exists():
        raise RuntimeError("birth + math audit required")
    if not json.loads(V12_MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("math audit failed")
    if CANDIDATE_V12.exists():
        raise RuntimeError("candidate.v12 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V12",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v11_sha": _sha_file(CANDIDATE_V11),
        "v12_birth_sha": _sha_file(V12_BIRTH),
        "architecture_amendment_sha": _sha_file(AMEND),
        "diagnosis_v11_sha": _sha_file(DIAG),
        "stat_contract_sha": _sha_file(STAT),
        "math_audit_sha": _sha_file(V12_MATH),
        "learning_law_ok": True,
        "gpu_scoring_ready": True,
        "all_sanity_ok": True,
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "note": "Candidate v12 adds surprise→HOLD. Must re-earn D1–D2. No DEVELOP yet.",
    }
    CANDIDATE_V12.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v12_sha": _sha_file(CANDIDATE_V12)}


def run_boundary_v12(*, write_lock: bool = False) -> dict[str, Any]:
    if not CANDIDATE_V12.exists():
        raise RuntimeError("missing candidate v12")
    if _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V12.read_text(encoding="utf-8"))["neural_cortex_sha"]:
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
    summary = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V12",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "candidate": "docs/cortex.candidate.v12.lock",
        "candidate_sha": _sha_file(CANDIDATE_V12),
        "required_ids": ["C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population"],
        "all_required_green": all(
            next(x for x in results if x["id"] == i).get("ok")
            for i in ("C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population")
        ),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_controls": len(results),
        "controls": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "env": torch_env(),
        "note": "v12 required greens are C4 + population C5/C6.",
    }
    if write_lock:
        if MACT_V12.exists():
            raise RuntimeError("boundary v12 lock exists")
        MACT_V12.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        lines = ["# TM.0.23.CORTEX M_act boundary (v12)", "", f"**required_green:** `{summary['all_required_green']}`", ""]
        for r in summary["controls"]:
            lines.append(f"- `{r['id']}`: **{'PASS' if r.get('ok') else 'FAIL'}** — {r.get('why')}")
        MACT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["locks_written"] = True
    return summary


def reveal_v12_eval() -> dict[str, Any]:
    if not CANDIDATE_V12.exists() or not MACT_V12.exists():
        raise RuntimeError("candidate v12 + boundary required")
    if not json.loads(MACT_V12.read_text(encoding="utf-8")).get("all_required_green"):
        raise RuntimeError("v12 population contract not green")
    if not _git_clean():
        raise RuntimeError("working tree dirty — refuse reveal")
    git_sha = _git_head()
    if V12_REVEAL.exists():
        raise RuntimeError("v12 reveal exists")
    sealed = json.loads(V12_SEALED.read_text(encoding="utf-8"))
    commitment = hashlib.sha256(bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])).hexdigest()
    prereg = json.loads(V12_PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    reveal = {
        "version": "TM.0.23.CORTEX.V12.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v12_sha": _sha_file(CANDIDATE_V12),
        "candidate_git_sha": git_sha,
        "mact_boundary_v12_sha": _sha_file(MACT_V12),
        "note": "Reveal only after candidate commit is on a clean HEAD.",
    }
    V12_REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    V12_GATE_PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V12.GATE.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "reveal_sha": _sha_file(V12_REVEAL),
                "candidate_v12_sha": _sha_file(CANDIDATE_V12),
                "candidate_git_sha": git_sha,
                "scorer_sha": _sha_file(SCORERS_PY),
                "d1_bind": ["press", "harm"],
                "d2_conflict": "swapped_press_harm",
                "extras": "population",
                "schedule": ["D0", "D1", "D2"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "reveal_sha": _sha_file(V12_REVEAL), "candidate_git_sha": git_sha}


def run_v12_gate(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not V12_REVEAL.exists():
        raise RuntimeError("reveal first")
    if DEV_V12.exists():
        raise RuntimeError("DEVELOP.v12 exists before gate — refuse")
    cand = json.loads(CANDIDATE_V12.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    reveal = json.loads(V12_REVEAL.read_text(encoding="utf-8"))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_v12_gate_battery(seed_hex=reveal["seed_hex"], n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.V12.GATE.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "sensorimotor_association_gate_clear": battery["sensorimotor_association_gate_clear"],
        "battery": battery,
        "candidate_v12_sha": _sha_file(CANDIDATE_V12),
        "candidate_git_sha": reveal.get("candidate_git_sha"),
        "reveal_sha": _sha_file(V12_REVEAL),
        "env": torch_env(),
        "device": dev,
        "note": "Narrow D1–D2 under v12 surprise→HOLD. Full D0–D12 stays closed unless ≥13/16.",
    }
    if write_lock:
        if V12_GATE_LOCK.exists():
            raise RuntimeError("gate lock exists")
        V12_GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        clear = battery["sensorimotor_association_gate_clear"]
        V12_GATE_MD.write_text(
            f"# TM.0.23.CORTEX v12 gate\n\n**sensorimotor_association_gate_clear:** `{clear}`\n**n_pair_clear:** `{battery['n_pair_clear']}/16`\n\n",
            encoding="utf-8",
        )
        if not clear:
            V12_FAIL.write_text(
                json.dumps(
                    {
                        "version": "TM.0.23.CORTEX.V12.GATE.FAILURE",
                        "product": "0.0.004",
                        "earned_next": False,
                        "ex0s": None,
                        "gate_sha": _sha_file(V12_GATE_LOCK),
                        "n_pair_clear": battery["n_pair_clear"],
                        "refuse": ["DEVELOP.v12", "edit-and-rescore on revealed v12 worlds", "full D0–D12"],
                    },
                    indent=2,
                )
                + "\n",
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
    ap.add_argument("--reveal-gate", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    if args.write_birth:
        print(json.dumps(write_v12_birth(), indent=2, default=str))
    elif args.math_audit:
        print(json.dumps(write_v12_math_audit(), indent=2, default=str))
    elif args.write_candidate:
        print(json.dumps(write_candidate_v12(), indent=2))
    elif args.mact_boundary:
        print(json.dumps(run_boundary_v12(write_lock=args.write_lock), indent=2, default=str))
    elif args.reveal_gate:
        print(json.dumps(reveal_v12_eval(), indent=2))
    elif args.gate:
        print(json.dumps(run_v12_gate(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
