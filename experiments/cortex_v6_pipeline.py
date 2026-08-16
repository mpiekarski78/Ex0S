"""TM.0.23.CORTEX.V6 — birth, candidate, boundary, narrow D1–D2 gate. No DEVELOP until ≥13/16."""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from experiments.cortex_mact_boundary import run_boundary_v6
from experiments.cortex_v6_gate import run_v6_gate_battery
from experiments.run_tm023cortex import make_cortex, run_sanity, torch_env
from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OP_COST, OPS, GenomeConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"
DIAG = DOCS / "cortex_diagnosis.v5.lock"
V6_AMEND = DOCS / "cortex_v6_architecture_amendment.lock"
V6_BIRTH = DOCS / "cortex_v6_birth.lock"
V6_MATH = DOCS / "cortex_v6_math_audit.lock"
CANDIDATE_V6 = DOCS / "cortex.candidate.v6.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V5 = DOCS / "cortex.candidate.v5.lock"
V6_PREREG = DOCS / "cortex_v6.prereg.lock"
V6_SEALED = DOCS / "cortex_v6_eval_secrets.sealed.json"
V6_REVEAL = DOCS / "cortex_v6_eval_reveal.lock"
V6_GATE_PREREG = DOCS / "cortex_v6_gate.prereg.lock"
V6_GATE_RUNNER = DOCS / "cortex_v6_gate.runner.lock"
V6_GATE_LOCK = DOCS / "cortex_v6_gate.lock"
V6_GATE_MD = DOCS / "tm023cortex_v6_gate_results.md"
V6_FAIL = DOCS / "cortex_v6_gate.failure.lock"
MACT_V6 = DOCS / "cortex_mact_boundary.v6.lock"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
DEV_V6 = DOCS / "cortex_development.v6.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def write_v6_birth() -> dict[str, Any]:
    if not V6_AMEND.exists() or not DIAG.exists():
        raise RuntimeError("freeze v6 apparatus + v5 diagnosis first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    with tempfile.TemporaryDirectory(prefix="v6birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        if ag.motor_vocab:
            raise RuntimeError("birth motor_vocab not empty")
        r = ag.bind_actuators(["h_x", "h_y"])
        nrm = float((ag.motor_vocab["h_x"] ** 2).sum() ** 0.5)
        if abs(nrm - 1.0) > 1e-6:
            raise RuntimeError(f"motor vector not unit: {nrm}")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V6.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "diagnosis_v5_sha": _sha_file(DIAG),
        "architecture_amendment_sha": _sha_file(V6_AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "sanity_results": summary["results"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {
            "unit_motor": True,
            "bind_n": r["n"],
            "b_op_act": b_op,
            "op_cost_act": OP_COST["ACT"],
        },
    }
    if V6_BIRTH.exists():
        raise RuntimeError("v6 birth exists")
    V6_BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V6_BIRTH)}


def write_v6_math_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"id": name, "ok": ok, "detail": detail})

    add(
        "v1_contract_untouched",
        _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2",
    )
    add("diagnosis_exists", DIAG.exists())
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    with tempfile.TemporaryDirectory(prefix="v6audit_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        add("birth_motor_empty", ag.motor_vocab == {})
        add("b_op_act_0_85", float(ag.b_op[OPS.index("ACT")]) == 0.85)
        add("w_act_query_zero", float(ag.W_act_query.abs().sum()) == 0.0)
        ag.bind_actuators(["h_a", "h_b"])
        add("unit_a", abs(float((ag.motor_vocab["h_a"] ** 2).sum() ** 0.5) - 1.0) < 1e-6)
        add("handles_not_in_vocab", "h_a" not in ag.vocab)
        v = ag.motor_vocab["h_a"].copy()
        ag.bind_actuators(["h_b", "h_a"])
        add("rebind_restores", bool((ag.motor_vocab["h_a"] == v).all()))
        v_ab = v.copy()
        with tempfile.TemporaryDirectory(prefix="v6audit2_") as tmp2:
            ag2 = make_cortex(Path(tmp2) / "s")
            ag2.bind_actuators(["h_b", "h_a"])
            add("bind_order_preserves_vector", bool((ag2.motor_vocab["h_a"] == v_ab).all()))
    ok = all(c["ok"] for c in checks)
    out = {
        "version": "TM.0.23.CORTEX.V6.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": ok,
        "checks": checks,
        "birth_sha": _sha_file(V6_BIRTH),
    }
    if V6_MATH.exists():
        raise RuntimeError("v6 math audit exists")
    V6_MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate_v6() -> dict[str, Any]:
    if not V6_BIRTH.exists() or not V6_MATH.exists():
        raise RuntimeError("birth + math audit required")
    if not json.loads(V6_MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("math audit failed")
    if CANDIDATE_V6.exists():
        raise RuntimeError("candidate.v6 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V6",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v5_sha": _sha_file(CANDIDATE_V5) if CANDIDATE_V5.exists() else None,
        "v6_birth_sha": _sha_file(V6_BIRTH),
        "architecture_amendment_sha": _sha_file(V6_AMEND),
        "diagnosis_v5_sha": _sha_file(DIAG),
        "v6_prereg_sha": _sha_file(V6_PREREG) if V6_PREREG.exists() else None,
        "math_audit_sha": _sha_file(V6_MATH),
        "learning_law_ok": True,
        "gpu_scoring_ready": True,
        "all_sanity_ok": True,
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "note": "Candidate v6. Authorized by v5 diagnosis. Must re-earn D1–D2. No DEVELOP yet.",
    }
    CANDIDATE_V6.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v6_sha": _sha_file(CANDIDATE_V6)}


def reveal_v6_eval() -> dict[str, Any]:
    if not CANDIDATE_V6.exists() or not MACT_V6.exists():
        raise RuntimeError("candidate v6 + green boundary required")
    if not json.loads(MACT_V6.read_text(encoding="utf-8")).get("all_controls_green"):
        raise RuntimeError("boundary not all green")
    if V6_REVEAL.exists():
        raise RuntimeError("v6 reveal exists")
    sealed = json.loads(V6_SEALED.read_text(encoding="utf-8"))
    commitment = hashlib.sha256(
        bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])
    ).hexdigest()
    prereg = json.loads(V6_PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    reveal = {
        "version": "TM.0.23.CORTEX.V6.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v6_sha": _sha_file(CANDIDATE_V6),
        "gate_runner_sha": _sha_file(V6_GATE_RUNNER),
        "mact_boundary_v6_sha": _sha_file(MACT_V6),
        "note": "Fresh v6 worlds. Diagnostic-only after scoring.",
    }
    V6_REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    V6_GATE_PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V6.GATE.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "reveal_sha": _sha_file(V6_REVEAL),
                "candidate_v6_sha": _sha_file(CANDIDATE_V6),
                "scorer_sha": _sha_file(SCORERS_PY),
                "schedule": ["D0", "D1", "D2"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "reveal_sha": _sha_file(V6_REVEAL)}


def run_v6_gate(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not V6_REVEAL.exists():
        raise RuntimeError("reveal first")
    if DEV_V6.exists():
        raise RuntimeError("DEVELOP.v6 exists before gate — refuse")
    cand = json.loads(CANDIDATE_V6.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_v6_gate_battery(n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.V6.GATE.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "sensorimotor_association_gate_clear": battery["sensorimotor_association_gate_clear"],
        "battery": battery,
        "candidate_v6_sha": _sha_file(CANDIDATE_V6),
        "reveal_sha": _sha_file(V6_REVEAL),
        "runner_sha": _sha_file(V6_GATE_RUNNER),
        "env": torch_env(),
        "device": dev,
        "note": "Narrow D1–D2 only. Full D0–D12 stays closed unless ≥13/16.",
    }
    if write_lock:
        if V6_GATE_LOCK.exists():
            raise RuntimeError("gate lock exists")
        V6_GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        clear = battery["sensorimotor_association_gate_clear"]
        V6_GATE_MD.write_text(
            "\n".join(
                [
                    "# TM.0.23.CORTEX v6 gate",
                    "",
                    f"**sensorimotor_association_gate_clear:** `{clear}`",
                    f"**n_pair_clear:** `{battery['n_pair_clear']}/16`",
                    "",
                    "Full D0–D12 not opened on this path unless the gate cleared.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if not clear:
            V6_FAIL.write_text(
                json.dumps(
                    {
                        "version": "TM.0.23.CORTEX.V6.GATE.FAILURE",
                        "product": "0.0.004",
                        "earned_next": False,
                        "ex0s": None,
                        "gate_sha": _sha_file(V6_GATE_LOCK),
                        "n_pair_clear": battery["n_pair_clear"],
                        "refuse": ["DEVELOP.v6", "edit-and-rescore on revealed v6 worlds"],
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
        print(json.dumps(write_v6_birth(), indent=2, default=str))
    elif args.math_audit:
        print(json.dumps(write_v6_math_audit(), indent=2, default=str))
    elif args.write_candidate:
        print(json.dumps(write_candidate_v6(), indent=2))
    elif args.mact_boundary:
        print(json.dumps(run_boundary_v6(write_lock=args.write_lock), indent=2, default=str))
    elif args.reveal_gate:
        print(json.dumps(reveal_v6_eval(), indent=2))
    elif args.gate:
        print(json.dumps(run_v6_gate(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
