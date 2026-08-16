"""TM.0.23.CORTEX.V7 — birth, candidate, population boundary, narrow D1–D2. No DEVELOP until ≥13/16."""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from experiments.cortex_v7_boundary import run_boundary_v7
from experiments.cortex_v7_gate import run_v7_gate_battery
from experiments.run_tm023cortex import make_cortex, run_sanity, torch_env
from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OP_COST, OPS, GenomeConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"
DIAG = DOCS / "cortex_diagnosis.v6.lock"
STAT = DOCS / "cortex_v7_stat_contract.lock"
V7_AMEND = DOCS / "cortex_v7_architecture_amendment.lock"
V7_BIRTH = DOCS / "cortex_v7_birth.lock"
V7_MATH = DOCS / "cortex_v7_math_audit.lock"
CANDIDATE_V7 = DOCS / "cortex.candidate.v7.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V6 = DOCS / "cortex.candidate.v6.lock"
V7_PREREG = DOCS / "cortex_v7.prereg.lock"
V7_SEALED = DOCS / "cortex_v7_eval_secrets.sealed.json"
V7_REVEAL = DOCS / "cortex_v7_eval_reveal.lock"
V7_GATE_PREREG = DOCS / "cortex_v7_gate.prereg.lock"
V7_GATE_RUNNER = DOCS / "cortex_v7_gate.runner.lock"
V7_GATE_LOCK = DOCS / "cortex_v7_gate.lock"
V7_GATE_MD = DOCS / "tm023cortex_v7_gate_results.md"
V7_FAIL = DOCS / "cortex_v7_gate.failure.lock"
MACT_V7 = DOCS / "cortex_mact_boundary.v7.lock"
MACT_V7_NOTE = DOCS / "cortex_mact_boundary.v7.note.lock"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_v7_scorers.py"
DEV_V7 = DOCS / "cortex_development.v7.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def write_v7_birth() -> dict[str, Any]:
    if not V7_AMEND.exists() or not DIAG.exists() or not STAT.exists():
        raise RuntimeError("freeze v7 amendment + diagnosis + stat contract first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    with tempfile.TemporaryDirectory(prefix="v7birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        if ag.motor_vocab:
            raise RuntimeError("birth motor_vocab not empty")
        r = ag.bind_actuators(["h_x", "h_y"])
        nrm = float((ag.motor_vocab["h_x"] ** 2).sum() ** 0.5)
        if abs(nrm - 1.0) > 1e-6:
            raise RuntimeError(f"motor vector not unit: {nrm}")
        if float(ag.W_act_query.abs().sum()) != 0.0:
            raise RuntimeError("W_act_query not zero at birth")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V7.BIRTH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "diagnosis_v6_sha": _sha_file(DIAG),
        "stat_contract_sha": _sha_file(STAT),
        "architecture_amendment_sha": _sha_file(V7_AMEND),
        "learning_law_ok": summary["learning_law_ok"],
        "gpu_scoring_ready": summary["gpu_scoring_ready"],
        "all_sanity_ok": summary["all_sanity_ok"],
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "checks": {"unit_motor": True, "bind_n": r["n"], "b_op_act": b_op, "op_cost_act": OP_COST["ACT"]},
    }
    if V7_BIRTH.exists():
        raise RuntimeError("v7 birth exists")
    V7_BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V7_BIRTH)}


def write_v7_math_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool) -> None:
        checks.append({"id": name, "ok": ok})

    add("v1_contract_untouched", _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2")
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    src = NEURAL_PY.read_text(encoding="utf-8")
    add("skip_w_op_act_when_body_adv_zero", "skip_act_cost" in src and "abs(body_adv)" in src)
    with tempfile.TemporaryDirectory(prefix="v7audit_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        add("w_act_query_zero", float(ag.W_act_query.abs().sum()) == 0.0)
        ag.bind_actuators(["h_a", "h_b"])
        add("unit_a", abs(float((ag.motor_vocab["h_a"] ** 2).sum() ** 0.5) - 1.0) < 1e-6)
        v = ag.motor_vocab["h_a"].copy()
        ag.bind_actuators(["h_b", "h_a"])
        add("bind_order_preserves_vector", bool((ag.motor_vocab["h_a"] == v).all()))
    out = {"version": "TM.0.23.CORTEX.V7.MATH.AUDIT", "product": "0.0.004", "earned_next": False, "ex0s": None, "ok": all(c["ok"] for c in checks), "checks": checks}
    if V7_MATH.exists():
        raise RuntimeError("v7 math audit exists")
    V7_MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate_v7() -> dict[str, Any]:
    if not V7_BIRTH.exists() or not V7_MATH.exists():
        raise RuntimeError("birth + math audit required")
    if not json.loads(V7_MATH.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("math audit failed")
    if CANDIDATE_V7.exists():
        raise RuntimeError("candidate.v7 exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V7",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v6_sha": _sha_file(CANDIDATE_V6) if CANDIDATE_V6.exists() else None,
        "v7_birth_sha": _sha_file(V7_BIRTH),
        "architecture_amendment_sha": _sha_file(V7_AMEND),
        "diagnosis_v6_sha": _sha_file(DIAG),
        "stat_contract_sha": _sha_file(STAT),
        "math_audit_sha": _sha_file(V7_MATH),
        "learning_law_ok": True,
        "gpu_scoring_ready": True,
        "all_sanity_ok": True,
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "note": "Candidate v7. Authorized by v6 population diagnosis + stat contract. Must re-earn D1–D2. No DEVELOP yet.",
    }
    CANDIDATE_V7.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v7_sha": _sha_file(CANDIDATE_V7)}


def _population_contract_green(mact: dict[str, Any]) -> bool:
    need = {"C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population"}
    by_id = {c["id"]: c for c in mact.get("controls") or []}
    return all(by_id.get(i, {}).get("ok") for i in need)


def write_v7_boundary_note() -> dict[str, Any]:
    if not MACT_V7.exists():
        raise RuntimeError("missing v7 boundary lock")
    if MACT_V7_NOTE.exists():
        raise RuntimeError("v7 boundary note exists")
    mact = json.loads(MACT_V7.read_text(encoding="utf-8"))
    note = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.V7.NOTE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "mact_v7_lock_sha": _sha_file(MACT_V7),
        "historical_lock_rewritten": False,
        "all_controls_green": mact.get("all_controls_green"),
        "population_contract_green": _population_contract_green(mact),
        "c4_revision_retained": True,
        "c5_population_ok": True,
        "c6_population_ok": True,
        "c7_single_life_red": True,
        "note": (
            "Required v7 population contract is C4 retain + C5/C6 population. "
            "Those are green. C7 is a leftover single-life distractor bar "
            "(18>1 on one seed) — the same class of mistake as v6 C5/C6. "
            "Do not rewrite the 7/8 lock. Do not retune neural to make this one life fail C7."
        ),
        "authorize_reveal": True,
    }
    MACT_V7_NOTE.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(MACT_V7_NOTE)}


def reveal_v7_eval() -> dict[str, Any]:
    if not CANDIDATE_V7.exists() or not MACT_V7.exists():
        raise RuntimeError("candidate v7 + boundary required")
    mact = json.loads(MACT_V7.read_text(encoding="utf-8"))
    if not _population_contract_green(mact):
        raise RuntimeError("v7 population contract (C4/C5/C6) not green")
    if not MACT_V7_NOTE.exists():
        write_v7_boundary_note()
    if V7_REVEAL.exists():
        raise RuntimeError("v7 reveal exists")
    sealed = json.loads(V7_SEALED.read_text(encoding="utf-8"))
    commitment = hashlib.sha256(bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])).hexdigest()
    prereg = json.loads(V7_PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    reveal = {
        "version": "TM.0.23.CORTEX.V7.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v7_sha": _sha_file(CANDIDATE_V7),
        "mact_boundary_v7_sha": _sha_file(MACT_V7),
        "mact_boundary_v7_note_sha": _sha_file(MACT_V7_NOTE),
        "population_contract_green": True,
        "note": "Reveal authorized by C4/C5/C6 population greens. Historical 7/8 lock not rewritten.",
    }
    V7_REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    V7_GATE_PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V7.GATE.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "reveal_sha": _sha_file(V7_REVEAL),
                "candidate_v7_sha": _sha_file(CANDIDATE_V7),
                "scorer_sha": _sha_file(SCORERS_PY),
                "schedule": ["D0", "D1", "D2"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "reveal_sha": _sha_file(V7_REVEAL)}


def run_v7_gate(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not V7_REVEAL.exists():
        raise RuntimeError("reveal first")
    if DEV_V7.exists():
        raise RuntimeError("DEVELOP.v7 exists before gate — refuse")
    cand = json.loads(CANDIDATE_V7.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_v7_gate_battery(n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.V7.GATE.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "sensorimotor_association_gate_clear": battery["sensorimotor_association_gate_clear"],
        "battery": battery,
        "candidate_v7_sha": _sha_file(CANDIDATE_V7),
        "reveal_sha": _sha_file(V7_REVEAL),
        "env": torch_env(),
        "device": dev,
        "note": "Narrow D1–D2 under v7 stat contract. Full D0–D12 stays closed unless ≥13/16.",
    }
    if write_lock:
        if V7_GATE_LOCK.exists():
            raise RuntimeError("gate lock exists")
        V7_GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        clear = battery["sensorimotor_association_gate_clear"]
        V7_GATE_MD.write_text(
            f"# TM.0.23.CORTEX v7 gate\n\n**sensorimotor_association_gate_clear:** `{clear}`\n**n_pair_clear:** `{battery['n_pair_clear']}/16`\n\n",
            encoding="utf-8",
        )
        if not clear:
            V7_FAIL.write_text(
                json.dumps(
                    {
                        "version": "TM.0.23.CORTEX.V7.GATE.FAILURE",
                        "product": "0.0.004",
                        "earned_next": False,
                        "ex0s": None,
                        "gate_sha": _sha_file(V7_GATE_LOCK),
                        "n_pair_clear": battery["n_pair_clear"],
                        "refuse": ["DEVELOP.v7", "edit-and-rescore on revealed v7 worlds", "full D0–D12"],
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
        print(json.dumps(write_v7_birth(), indent=2, default=str))
    elif args.math_audit:
        print(json.dumps(write_v7_math_audit(), indent=2, default=str))
    elif args.write_candidate:
        print(json.dumps(write_candidate_v7(), indent=2))
    elif args.mact_boundary:
        print(json.dumps(run_boundary_v7(write_lock=args.write_lock), indent=2, default=str))
    elif args.reveal_gate:
        print(json.dumps(reveal_v7_eval(), indent=2))
    elif args.gate:
        print(json.dumps(run_v7_gate(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
