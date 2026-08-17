"""TM.0.23.CORTEX.D6.R1 — birth/candidate with habituation familiarity."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from experiments.cortex_d6_r1_gate import run_d6_r1_gate_battery
from experiments.cortex_mact_boundary import control_c1_v6, control_c2_v6, control_c3_v6, control_c4_v6, control_c8_v6
from experiments.cortex_v7_stats import run_c5_population, run_c6_population
from experiments.run_tm023cortex import make_cortex, run_sanity, torch_env
from three_memory.neural_cortex import (
    CONFLICT_HOLD_BIAS,
    EQUAL_EVIDENCE_MIN_SYMBOLS,
    FAMILIARITY_ABS,
    FAMILIARITY_RATIO,
    MOTOR_ACT_TOKENS,
    OP_COST,
    OPS,
    GenomeConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"
DIAG = DOCS / "cortex_diagnosis.fulldev_r4.lock"
STAT = DOCS / "cortex_d6_r1_stat_contract.lock"
AMEND = DOCS / "cortex_d6_r1_architecture_amendment.lock"
BIRTH = DOCS / "cortex_d6_r1_birth.lock"
MATH = DOCS / "cortex_d6_r1_math_audit.lock"
CANDIDATE_V21 = DOCS / "cortex.candidate.v21.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V20 = DOCS / "cortex.candidate.v20.lock"
PREREG = DOCS / "cortex_d6_r1.prereg.lock"
SEALED = DOCS / "cortex_d6_r1_eval_secrets.sealed.json"
REVEAL = DOCS / "cortex_d6_r1_eval_reveal.lock"
GATE_LOCK = DOCS / "cortex_d6_r1_gate.lock"
GATE_MD = DOCS / "tm023cortex_d6_r1_gate_results.md"
FAIL = DOCS / "cortex_d6_r1_gate.failure.lock"
MACT = DOCS / "cortex_mact_boundary.v21.lock"
MACT_MD = DOCS / "tm023cortex_mact_boundary_v21_results.md"
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
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    with tempfile.TemporaryDirectory(prefix="d6r1birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        r = ag.bind_actuators(["h_x", "h_y"])
        if float(ag.W_act_query.abs().sum()) != 0.0:
            raise RuntimeError("W_act_query not zero at birth")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.D6.R1.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "diagnosis_fulldev_r4_sha": _sha_file(DIAG),
        "stat_contract_sha": _sha_file(STAT),
        "architecture_amendment_sha": _sha_file(AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "prior_v20_neural_sha": json.loads(CANDIDATE_V20.read_text(encoding="utf-8"))["neural_cortex_sha"],
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {"bind_n": r["n"], "b_op_act": b_op, "op_cost_act": OP_COST["ACT"]},
        "note": "D6.R1 echoic emit. No FULLDEV reopen yet.",
    }
    if BIRTH.exists():
        raise RuntimeError("D6.R1 birth exists")
    BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(BIRTH)}


def write_math_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool) -> None:
        checks.append({"id": name, "ok": ok})

    src = NEURAL_PY.read_text(encoding="utf-8")
    add("v1_contract_untouched", _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2")
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    add("conflict_hold_bias_frozen", CONFLICT_HOLD_BIAS == 2.0)
    add("familiarity_ratio_frozen", FAMILIARITY_RATIO == 0.5)
    add("equal_evidence_min_symbols_3", EQUAL_EVIDENCE_MIN_SYMBOLS == 3)
    add("s_write_hook_present", "def _on_s_write" in src and "self.memory.on_write = self._on_s_write" in src)
    add("retrieve_recency_tiebreak", "(-t[0], -t[1], t[2])" in src and "int(rec.when)" in src)
    add("habituation_trace", "FAMILIARITY_DECAY" in src and "FAMILIARITY_ABS" in src and "_symbol_fam" in src)
    add("echoic_buffer", "ECHOIC_MAX" in src and "_echoic" in src)
    add("no_lifetime_max_novelty", "FAMILIARITY_RATIO * float(mx)" not in src)
    add("neural_changed_from_v20", _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V20.read_text(encoding="utf-8"))["neural_cortex_sha"])
    out = {
        "version": "TM.0.23.CORTEX.D6.R1.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
    if MATH.exists():
        raise RuntimeError("math audit exists")
    MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate() -> dict[str, Any]:
    if not BIRTH.exists() or not MATH.exists() or not json.loads(MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("birth + passing math audit required")
    if CANDIDATE_V21.exists():
        raise RuntimeError("candidate.v21 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V21",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v20_sha": _sha_file(CANDIDATE_V20),
        "d6_r1_birth_sha": _sha_file(BIRTH),
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
        "note": "Candidate v21 / D6.R1. Must re-earn C4/C5/C6 then D5. No FULLDEV reopen yet.",
    }
    CANDIDATE_V21.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v21_sha": _sha_file(CANDIDATE_V21)}


def run_boundary(*, write_lock: bool = False) -> dict[str, Any]:
    if _sha_file(NEURAL_PY) != json.loads(CANDIDATE_V21.read_text(encoding="utf-8"))["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    results = [control_c1_v6(), control_c2_v6(), control_c3_v6(), control_c4_v6(), run_c5_population(), run_c6_population(), control_c8_v6()]
    summary = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V21",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "candidate": "docs/cortex.candidate.v21.lock",
        "candidate_sha": _sha_file(CANDIDATE_V21),
        "required_ids": ["C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population"],
        "all_required_green": all(
            next(x for x in results if x["id"] == i).get("ok")
            for i in ("C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population")
        ),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_controls": len(results),
        "controls": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "env": torch_env(),
    }
    if write_lock:
        if MACT.exists():
            raise RuntimeError("boundary v21 exists")
        MACT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        MACT_MD.write_text(
            "# TM.0.23.CORTEX M_act boundary (v21)\n\n"
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
    reveal = {
        "version": "TM.0.23.CORTEX.D6.R1.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v21_sha": _sha_file(CANDIDATE_V21),
        "candidate_git_sha": git_sha,
        "mact_boundary_v21_sha": _sha_file(MACT),
    }
    REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "reveal_sha": _sha_file(REVEAL), "candidate_git_sha": git_sha}


def run_gate(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    cand = json.loads(CANDIDATE_V21.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    reveal = json.loads(REVEAL.read_text(encoding="utf-8"))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_d6_r1_gate_battery(seed_hex=reveal["seed_hex"], n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.D6.R1.GATE.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "relation_gate_clear": battery["relation_gate_clear"],
        "battery": battery,
        "candidate_v21_sha": _sha_file(CANDIDATE_V21),
        "candidate_git_sha": reveal.get("candidate_git_sha"),
        "reveal_sha": _sha_file(REVEAL),
        "env": torch_env(),
        "device": dev,
        "note": "Narrow D5. Full D0–D12 stays closed unless ≥13/16.",
    }
    if write_lock:
        if GATE_LOCK.exists():
            raise RuntimeError("gate lock exists")
        GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        GATE_MD.write_text(
            f"# TM.0.23.CORTEX D6.R1 gate\n\n**relation_gate_clear:** `{battery['relation_gate_clear']}`\n**n_pair_clear:** `{battery['n_pair_clear']}/16`\n\n",
            encoding="utf-8",
        )
        if not battery["relation_gate_clear"]:
            FAIL.write_text(
                json.dumps(
                    {
                        "version": "TM.0.23.CORTEX.D6.R1.GATE.FAILURE",
                        "product": "0.0.004",
                        "earned_next": False,
                        "ex0s": None,
                        "gate_sha": _sha_file(GATE_LOCK),
                        "n_pair_clear": battery["n_pair_clear"],
                        "refuse": ["reopen FULLDEV", "edit-and-rescore", "nursery"],
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
