"""TM.0.23.CORTEX.V27 — birth/candidate: learned motor program, not sensory-buffer replay."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from experiments.cortex_mact_boundary import control_c1_v6, control_c2_v6, control_c3_v6, control_c4_v6, control_c8_v6
from experiments.cortex_v26_generality import control_g1
from experiments.cortex_v27_gate import run_v27_gate_battery
from experiments.cortex_v7_stats import run_c5_population, run_c6_population
from experiments.run_tm023cortex import make_cortex, run_sanity, torch_env
from three_memory.neural_cortex import (
    CONFLICT_HOLD_BIAS,
    ECHOIC_BIAS,
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
DIAG = DOCS / "cortex_diagnosis.v26_generality.lock"
STAT = DOCS / "cortex_v27_stat_contract.lock"
AMEND = DOCS / "cortex_v27_architecture_amendment.lock"
BIRTH = DOCS / "cortex_v27_birth.lock"
MATH = DOCS / "cortex_v27_math_audit.lock"
CANDIDATE_V27 = DOCS / "cortex.candidate.v27.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V26 = DOCS / "cortex.candidate.v26.lock"
PREREG = DOCS / "cortex_v27.prereg.lock"
SEALED = DOCS / "cortex_v27_eval_secrets.sealed.json"
REVEAL = DOCS / "cortex_v27_eval_reveal.lock"
GATE_LOCK = DOCS / "cortex_v27_gate.lock"
GATE_MD = DOCS / "tm023cortex_v27_gate_results.md"
FAIL = DOCS / "cortex_v27_gate.failure.lock"
MACT = DOCS / "cortex_mact_boundary.v27.lock"
MACT_MD = DOCS / "tm023cortex_mact_boundary_v27_results.md"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def write_birth() -> dict[str, Any]:
    if not PREREG.exists() or not AMEND.exists():
        raise RuntimeError("v27 apparatus must be frozen first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    with tempfile.TemporaryDirectory(prefix="v27birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        r = ag.bind_actuators(["h_x", "h_y"])
        if float(ag.W_act_query.abs().sum()) != 0.0:
            raise RuntimeError("W_act_query not zero at birth")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V27.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "diagnosis_v26_generality_sha": _sha_file(DIAG),
        "stat_contract_sha": _sha_file(STAT),
        "architecture_amendment_sha": _sha_file(AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "prior_v26_neural_sha": json.loads(CANDIDATE_V26.read_text(encoding="utf-8"))["neural_cortex_sha"],
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {"bind_n": r["n"], "b_op_act": b_op, "op_cost_act": OP_COST["ACT"]},
        "note": "v27 learned motor program, not phrase replay. No FULLDEV.R7.",
    }
    if BIRTH.exists():
        raise RuntimeError("v27 birth exists")
    BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(BIRTH)}


def write_math_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool) -> None:
        checks.append({"id": name, "ok": ok})

    src = NEURAL_PY.read_text(encoding="utf-8")
    g1 = control_g1()
    add("v1_contract_untouched", _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2")
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    add("conflict_hold_bias_frozen", CONFLICT_HOLD_BIAS == 2.0)
    add("familiarity_ratio_frozen", FAMILIARITY_RATIO == 0.5)
    add("familiarity_abs_frozen", FAMILIARITY_ABS == 16.0)
    add("equal_evidence_min_symbols_3", EQUAL_EVIDENCE_MIN_SYMBOLS == 3)
    add("vocal_refractory_15", VOCAL_REFRACTORY == 1.5)
    add("s_write_hook_present", "def _on_s_write" in src and "self.memory.on_write = self._on_s_write" in src)
    add("retrieve_recency_tiebreak", "(-t[0], -t[1], t[2])" in src and "int(rec.when)" in src)
    add("habituation_trace", "FAMILIARITY_DECAY" in src and "_symbol_fam" in src)
    add("g1_no_scripted_phrase", bool(g1.get("ok")))
    add("no_echoic_hard_emit", "_echoic_emit_token" not in src)
    add("echoic_bias_only", "echoic_bias" in src and ECHOIC_BIAS == 0.08)
    add("generic_act_refractory", 'vocal_next == "ACT"' in src and 'chosen_op in ("EMIT", "ACT")' in src)
    add(
        "neural_changed_from_v26",
        _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V26.read_text(encoding="utf-8"))["neural_cortex_sha"],
    )
    out = {
        "version": "TM.0.23.CORTEX.V27.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "g1": g1,
    }
    if MATH.exists():
        raise RuntimeError("math audit exists")
    MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate() -> dict[str, Any]:
    if not BIRTH.exists() or not MATH.exists() or not json.loads(MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("birth + passing math audit required")
    if CANDIDATE_V27.exists():
        raise RuntimeError("candidate.v27 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V27",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v26_sha": _sha_file(CANDIDATE_V26),
        "v27_birth_sha": _sha_file(BIRTH),
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
        "note": "Candidate v27. Must re-earn C4/C5/C6 then G1+G3+G5. FULLDEV.R7 sealed.",
    }
    CANDIDATE_V27.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v27_sha": _sha_file(CANDIDATE_V27)}


def run_boundary(*, write_lock: bool = False) -> dict[str, Any]:
    if _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V27.read_text(encoding="utf-8"))["neural_cortex_sha"]:
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
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V27",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "candidate": "docs/cortex.candidate.v27.lock",
        "candidate_sha": _sha_file(CANDIDATE_V27),
        "required_ids": list(required),
        "all_required_green": all(next(x for x in results if x["id"] == i).get("ok") for i in required),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_controls": len(results),
        "controls": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "env": torch_env(),
    }
    if write_lock:
        if MACT.exists():
            raise RuntimeError("boundary v27 exists")
        MACT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        MACT_MD.write_text(
            "# TM.0.23.CORTEX M_act boundary (v27)\n\n"
            + "\n".join(f"- `{r['id']}`: **{'PASS' if r.get('ok') else 'FAIL'}**" for r in summary["controls"])
            + "\n",
            encoding="utf-8",
        )
        summary["locks_written"] = True
    return summary


def reveal() -> dict[str, Any]:
    if not json.loads(MACT.read_text(encoding="utf-8")).get("all_required_green"):
        raise RuntimeError("population contract not green")
    if not _git_clean():
        raise RuntimeError("working tree dirty — refuse reveal")
    if REVEAL.exists():
        raise RuntimeError("reveal exists")
    git_sha = _git_head()
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    commitment = hashlib.sha256(bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])).hexdigest()
    if commitment != json.loads(PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    reveal_lock = {
        "version": "TM.0.23.CORTEX.V27.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v27_sha": _sha_file(CANDIDATE_V27),
        "candidate_git_sha": git_sha,
        "mact_boundary_v27_sha": _sha_file(MACT),
    }
    REVEAL.write_text(json.dumps(reveal_lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "reveal_sha": _sha_file(REVEAL), "candidate_git_sha": git_sha}


def run_gate(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    cand = json.loads(CANDIDATE_V27.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    reveal_lock = json.loads(REVEAL.read_text(encoding="utf-8"))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_v27_gate_battery(seed_hex=reveal_lock["seed_hex"], n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.V27.GATE.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "relation_gate_clear": battery["relation_gate_clear"],
        "battery": battery,
        "candidate_v27_sha": _sha_file(CANDIDATE_V27),
        "candidate_git_sha": reveal_lock.get("candidate_git_sha"),
        "reveal_sha": _sha_file(REVEAL),
        "env": torch_env(),
        "device": dev,
        "refuse_fulldev_r7": not battery["relation_gate_clear"],
        "note": "Narrow G1+G3+G5. FULLDEV.R7 stays sealed unless this gate clears.",
    }
    if write_lock:
        if GATE_LOCK.exists():
            raise RuntimeError("gate lock exists")
        GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        GATE_MD.write_text(
            f"# TM.0.23.CORTEX v27 gate\n\n**relation_gate_clear:** `{battery['relation_gate_clear']}`\n"
            f"**n_pair_clear:** `{battery['n_pair_clear']}/16`\n"
            f"**g1_ok:** `{battery['g1_ok']}`\n\n",
            encoding="utf-8",
        )
        if not battery["relation_gate_clear"]:
            FAIL.write_text(
                json.dumps(
                    {
                        "version": "TM.0.23.CORTEX.V27.GATE.FAILURE",
                        "product": "0.0.004",
                        "earned_next": False,
                        "ex0s": None,
                        "gate_sha": _sha_file(GATE_LOCK),
                        "n_pair_clear": battery["n_pair_clear"],
                        "g1_ok": battery["g1_ok"],
                        "refuse": ["reveal FULLDEV.R7", "edit-and-rescore", "nursery"],
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
        print(json.dumps(write_birth(), indent=2, default=str))
    elif args.math_audit:
        print(json.dumps(write_math_audit(), indent=2, default=str))
    elif args.write_candidate:
        print(json.dumps(write_candidate(), indent=2))
    elif args.mact_boundary:
        print(json.dumps(run_boundary(write_lock=args.write_lock), indent=2, default=str))
    elif args.reveal_gate:
        print(json.dumps(reveal(), indent=2))
    elif args.gate:
        print(json.dumps(run_gate(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
