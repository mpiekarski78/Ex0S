"""TM.0.23.CORTEX.V5 pipeline — birth, candidate, boundary, gate, DEVELOP.v5."""

from __future__ import annotations

import hashlib
import json
import secrets
import tempfile
from pathlib import Path
from typing import Any

import torch

from experiments.cortex_develop_life import run_battery as run_develop_battery
from experiments.cortex_mact_boundary import run_boundary_v5
from experiments.cortex_v5_gate import run_v5_gate_battery
from experiments.freeze_v5_apparatus import freeze_all_v5_apparatus
from experiments.run_tm023cortex import (
    make_cortex,
    run_sanity,
    torch_env,
)
from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OP_COST, OPS, GenomeConfig


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_src(fn) -> str:  # noqa: ANN001
    import inspect

    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"

V5_AMEND_LOCK = DOCS / "cortex_v5_architecture_amendment.lock"
V5_BIRTH = DOCS / "cortex_v5_birth.lock"
CANDIDATE_V5 = DOCS / "cortex.candidate.v5.lock"
CANDIDATE_LIVE = DOCS / "cortex.candidate.lock"
CANDIDATE_V4 = DOCS / "cortex.candidate.v4.lock"
V5_PREREG = DOCS / "cortex_v5.prereg.lock"
V5_SEALED = DOCS / "cortex_v5_eval_secrets.sealed.json"
V5_REVEAL = DOCS / "cortex_v5_eval_reveal.lock"
V5_GATE_PREREG = DOCS / "cortex_v5_gate.prereg.lock"
V5_GATE_RUNNER = DOCS / "cortex_v5_gate.runner.lock"
V5_GATE_LOCK = DOCS / "cortex_v5_gate.lock"
V5_GATE_MD = DOCS / "tm023cortex_v5_gate_results.md"
V5_MATH = DOCS / "cortex_v5_math_audit.lock"
MACT_V5 = DOCS / "cortex_mact_boundary.v5.lock"

# DEVELOP.v5
DEV_V5_PREREG = DOCS / "cortex_development.v5.prereg.lock"
DEV_V5_RUNNER = DOCS / "cortex_development.runner.v5.lock"
DEV_V5_SEALED = DOCS / "cortex_development.v5_eval_secrets.sealed.json"
DEV_V5_REVEAL = DOCS / "cortex_eval_reveal.v5.lock"
DEV_V5_LOCK = DOCS / "cortex_development.v5.lock"
DEV_V5_WALL = DOCS / "cortex_wall.v5.lock"
DEV_V5_MD = DOCS / "tm023cortex_development_v5_results.md"
DEV_LATEST = DOCS / "cortex_development_latest.json"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
LIFE_PY = REPO_ROOT / "experiments" / "cortex_develop_life.py"
DEV_CONTRACT = DOCS / "cortex_development_contract.md"


def write_v5_birth() -> dict[str, Any]:
    if not V5_AMEND_LOCK.exists():
        raise RuntimeError("freeze v5 apparatus first")
    summary = run_sanity()
    if not summary.get("all_sanity_ok"):
        raise RuntimeError("nine sanity failed — refuse v5 birth")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must be empty for v5")
    with tempfile.TemporaryDirectory(prefix="v5birth_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        if ag.motor_vocab:
            raise RuntimeError("birth motor_vocab not empty")
        b_op = float(ag.b_op[OPS.index("ACT")])
    birth = {
        "version": "TM.0.23.CORTEX.V5.BIRTH",
        "lab": "TM.0.23.CORTEX.V5",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "architecture_amendment_sha": _sha_file(V5_AMEND_LOCK),
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
            "motor_act_tokens_empty": list(MOTOR_ACT_TOKENS) == [],
            "birth_motor_vocab_empty": True,
            "b_op_act": b_op,
            "op_cost_act": OP_COST["ACT"],
            "bind_actuators": True,
        },
        "note": "v5 birth after bind_actuators ABI + nine sanity. Audit before candidate.v5.",
    }
    if V5_BIRTH.exists():
        raise RuntimeError("cortex_v5_birth.lock exists — refuse rewrite")
    V5_BIRTH.write_text(json.dumps(birth, indent=2, default=str) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(V5_BIRTH), "sha": _sha_file(V5_BIRTH), "all_sanity_ok": True}


def write_v5_math_audit() -> dict[str, Any]:
    if not V5_BIRTH.exists():
        raise RuntimeError("missing v5 birth")
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"id": name, "ok": ok, "detail": detail})

    add(
        "v1_contract_untouched",
        _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2",
    )
    add("motor_act_tokens_empty", list(MOTOR_ACT_TOKENS) == [])
    add("op_cost_act_0_05", OP_COST["ACT"] == 0.05)
    with tempfile.TemporaryDirectory(prefix="v5audit_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        add("birth_motor_empty", ag.motor_vocab == {})
        add("b_op_act_0_85", float(ag.b_op[OPS.index("ACT")]) == 0.85)
        add("b_op_not_plastic", "b_op" not in ag._plastic_names)
        add("has_bind_actuators", hasattr(ag, "bind_actuators"))
        add("has_rng_motor", hasattr(ag, "rng_motor"))
        r = ag.bind_actuators(["h_x", "h_y"])
        add("bind_ok", r["n"] == 2)
        add("handles_not_in_vocab", "h_x" not in ag.vocab and "h_y" not in ag.vocab)
        v = ag.motor_vocab["h_x"].copy()
        ag.bind_actuators(["h_y", "h_x"])
        add("rebind_restores", bool((ag.motor_vocab["h_x"] == v).all()))
    ok = all(c["ok"] for c in checks)
    out = {
        "version": "TM.0.23.CORTEX.V5.MATH.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": ok,
        "checks": checks,
        "birth_sha": _sha_file(V5_BIRTH),
    }
    if V5_MATH.exists():
        raise RuntimeError("math audit lock exists")
    V5_MATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_candidate_v5() -> dict[str, Any]:
    if not V5_BIRTH.exists():
        raise RuntimeError("missing birth")
    if not V5_MATH.exists():
        raise RuntimeError("run math audit first")
    audit = json.loads(V5_MATH.read_text(encoding="utf-8"))
    if not audit.get("ok"):
        raise RuntimeError("math audit failed")
    birth = json.loads(V5_BIRTH.read_text(encoding="utf-8"))
    if not birth.get("all_sanity_ok"):
        raise RuntimeError("birth sanity failed")
    if CANDIDATE_V5.exists():
        raise RuntimeError("candidate.v5.lock exists")
    cand = {
        "version": "TM.0.23.CORTEX.CANDIDATE.V5",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "supersedes_v4_sha": _sha_file(CANDIDATE_V4) if CANDIDATE_V4.exists() else None,
        "v5_birth_sha": _sha_file(V5_BIRTH),
        "architecture_amendment_sha": _sha_file(V5_AMEND_LOCK),
        "v5_prereg_sha": _sha_file(V5_PREREG) if V5_PREREG.exists() else None,
        "math_audit_sha": _sha_file(V5_MATH),
        "learning_law_ok": True,
        "gpu_scoring_ready": True,
        "all_sanity_ok": True,
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "make_cortex_sha": _sha_src(make_cortex),
        "env": torch_env(),
        "genome": GenomeConfig().to_dict(),
        "human_math_audit": {
            "ok": True,
            "checks": [
                "MOTOR_ACT_TOKENS empty",
                "bind_actuators opaque handles",
                "internal motor-registry RNG",
                "frozen b_op[ACT]=0.85",
                "OP_COST[ACT]=0.05",
                "v1 contract untouched",
            ],
        },
        "note": "Candidate v5. V4 gate credit does not transfer. Must re-earn D1–D2.",
    }
    CANDIDATE_V5.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    # live pointer only after versioned v5 exists
    CANDIDATE_LIVE.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "candidate_v5_sha": _sha_file(CANDIDATE_V5)}


def reveal_v5_eval() -> dict[str, Any]:
    if not CANDIDATE_V5.exists():
        raise RuntimeError("candidate v5 required")
    if not MACT_V5.exists():
        raise RuntimeError("mact boundary v5 must be green first")
    mact = json.loads(MACT_V5.read_text(encoding="utf-8"))
    if not mact.get("all_controls_green"):
        raise RuntimeError("boundary controls not all green")
    if not V5_PREREG.exists() or not V5_SEALED.exists():
        raise RuntimeError("missing v5 prereg/sealed")
    if V5_REVEAL.exists():
        raise RuntimeError("v5 reveal exists")
    sealed = json.loads(V5_SEALED.read_text(encoding="utf-8"))
    seed_b = bytes.fromhex(sealed["seed_hex"])
    salt_b = bytes.fromhex(sealed["salt_hex"])
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    prereg = json.loads(V5_PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    reveal = {
        "version": "TM.0.23.CORTEX.V5.EVAL.REVEAL",
        "lab": "TM.0.23.CORTEX.V5.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "commitment_verified": True,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "gate_runner_sha": _sha_file(V5_GATE_RUNNER),
        "prereg_sha": _sha_file(V5_PREREG),
        "mact_boundary_v5_sha": _sha_file(MACT_V5),
        "note": "Revealed after candidate v5 + green boundary. Worlds diagnostic-only after scoring.",
    }
    V5_REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    gate_prereg = {
        "version": "TM.0.23.CORTEX.V5.GATE.PREREG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "reveal_sha": _sha_file(V5_REVEAL),
        "runner_sha": _sha_file(V5_GATE_RUNNER),
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "scorer_sha": _sha_file(SCORERS_PY),
        "schedule": ["D0", "D1", "D2"],
    }
    V5_GATE_PREREG.write_text(json.dumps(gate_prereg, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "reveal_sha": _sha_file(V5_REVEAL)}


def run_v5_gate(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not V5_REVEAL.exists():
        raise RuntimeError("reveal first")
    cand = json.loads(CANDIDATE_V5.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_v5_gate_battery(n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.V5.GATE.RESULT",
        "lab": "TM.0.23.CORTEX.V5.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "sensorimotor_association_gate_clear": battery["sensorimotor_association_gate_clear"],
        "battery": battery,
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "reveal_sha": _sha_file(V5_REVEAL),
        "runner_sha": _sha_file(V5_GATE_RUNNER),
        "env": torch_env(),
        "device": dev,
        "note": "Narrow D1–D2 re-earn on fresh v5 worlds. Full D0–D12 requires DEVELOP.v5.",
    }
    if write_lock:
        if V5_GATE_LOCK.exists():
            raise RuntimeError("gate lock exists")
        V5_GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        clear = battery["sensorimotor_association_gate_clear"]
        lines = [
            "# TM.0.23.CORTEX v5 gate",
            "",
            f"**sensorimotor_association_gate_clear:** `{clear}`",
            f"**n_pair_clear:** `{battery['n_pair_clear']}/16`",
            "",
            summary["note"],
            "",
        ]
        V5_GATE_MD.write_text("\n".join(lines), encoding="utf-8")
        summary["locks_written"] = True
    return summary


def freeze_develop_v5_runner() -> dict[str, Any]:
    if not V5_GATE_LOCK.exists():
        raise RuntimeError("v5 gate result required")
    gate = json.loads(V5_GATE_LOCK.read_text(encoding="utf-8"))
    if not gate.get("sensorimotor_association_gate_clear"):
        raise RuntimeError("gate not clear — refuse DEVELOP.v5")
    if DEV_V5_RUNNER.exists():
        raise RuntimeError("runner.v5 exists")
    cand = json.loads(CANDIDATE_V5.read_text(encoding="utf-8"))
    lock = {
        "version": "TM.0.23.CORTEX.DEVELOP.RUNNER.V5",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "stages": [f"D{i}" for i in range(13)],
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "cortex_memory_sha": cand["cortex_memory_sha"],
        "scorer_sha": _sha_file(SCORERS_PY),
        "life_sha": _sha_file(LIFE_PY),
        "development_contract_sha": _sha_file(DEV_CONTRACT) if DEV_CONTRACT.exists() else None,
        "v5_gate_sha": _sha_file(V5_GATE_LOCK),
        "note": "Pins candidate v5 neural SHAs. Fresh commitment ≠ narrow gate.",
    }
    DEV_V5_RUNNER.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(DEV_V5_RUNNER)}


def publish_develop_v5_commitment() -> dict[str, Any]:
    if not DEV_V5_RUNNER.exists():
        raise RuntimeError("freeze develop runner first")
    if DEV_V5_PREREG.exists():
        raise RuntimeError("develop.v5 prereg exists")
    seed_b = secrets.token_bytes(32)
    salt_b = secrets.token_bytes(32)
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    # distinct from narrow v5
    narrow = json.loads(V5_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    if commitment == narrow:
        raise RuntimeError("collided with narrow v5 commitment")
    sealed = {
        "version": "TM.0.23.CORTEX.DEVELOP.V5.EVAL.SEALED",
        "seed_hex": seed_b.hex(),
        "salt_hex": salt_b.hex(),
    }
    DEV_V5_SEALED.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    prereg = {
        "version": "TM.0.23.CORTEX.DEVELOP.V5.PREREG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "runner_sha": _sha_file(DEV_V5_RUNNER),
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "distinct_from_narrow_v5": narrow,
        "schedule": [f"D{i}" for i in range(13)],
        "n_pairs": 16,
    }
    DEV_V5_PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "commitment": commitment, "prereg_sha": _sha_file(DEV_V5_PREREG)}


def reveal_develop_v5() -> dict[str, Any]:
    if not DEV_V5_PREREG.exists() or not DEV_V5_SEALED.exists():
        raise RuntimeError("missing develop.v5 prereg/sealed")
    if DEV_V5_REVEAL.exists():
        raise RuntimeError("reveal.v5 exists")
    sealed = json.loads(DEV_V5_SEALED.read_text(encoding="utf-8"))
    commitment = hashlib.sha256(
        bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])
    ).hexdigest()
    prereg = json.loads(DEV_V5_PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    reveal = {
        "version": "TM.0.23.CORTEX.DEVELOP.V5.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "runner_sha": _sha_file(DEV_V5_RUNNER),
        "prereg_sha": _sha_file(DEV_V5_PREREG),
    }
    DEV_V5_REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "reveal_sha": _sha_file(DEV_V5_REVEAL)}


def run_develop_v5(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not DEV_V5_REVEAL.exists():
        raise RuntimeError("reveal develop.v5 first")
    cand = json.loads(CANDIDATE_V5.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    runner = json.loads(DEV_V5_RUNNER.read_text(encoding="utf-8"))
    if runner["neural_cortex_sha"] != cand["neural_cortex_sha"]:
        raise RuntimeError("runner/candidate neural mismatch")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_develop_battery(n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.DEVELOP.V5.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "battery": battery,
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "reveal_sha": _sha_file(DEV_V5_REVEAL),
        "runner_sha": _sha_file(DEV_V5_RUNNER),
        "env": torch_env(),
        "device": dev,
        "note": "DEVELOP.v5 full D0–D12 on fresh worlds. Does not stamp 0.0.005.",
    }
    if write_lock:
        if DEV_V5_LOCK.exists():
            raise RuntimeError("development.v5.lock exists")
        DEV_V5_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        # wall stub diagnostic
        wall = {
            "version": "TM.0.23.CORTEX.WALL.V5",
            "product": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "development_v5_sha": _sha_file(DEV_V5_LOCK),
            "first_fail_summary": battery.get("first_fail_histogram")
            or battery.get("summary"),
            "note": "Diagnostic capacity/wall companion to DEVELOP.v5.",
        }
        DEV_V5_WALL.write_text(json.dumps(wall, indent=2, default=str) + "\n", encoding="utf-8")
        DEV_LATEST.write_text(
            json.dumps(
                {
                    "latest": "docs/cortex_development.v5.lock",
                    "sha": _sha_file(DEV_V5_LOCK),
                    "wall": "docs/cortex_wall.v5.lock",
                    "note": "Non-lock pointer; historical cortex_development.lock untouched.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        n_clear = battery.get("n_pair_clear") or battery.get("n_clear")
        lines = [
            "# TM.0.23.CORTEX DEVELOP.v5",
            "",
            f"**n_pair_clear / summary:** `{n_clear}`",
            f"**eligible_for_000005:** `false`",
            "",
            "Historical `cortex_development.lock` not overwritten.",
            "",
        ]
        DEV_V5_MD.write_text("\n".join(lines), encoding="utf-8")
        summary["locks_written"] = True
    return summary


def freeze_v5_gate_failure() -> dict[str, Any]:
    """If gate fails: freeze failure; do not edit-rescore."""
    if not V5_GATE_LOCK.exists():
        raise RuntimeError("missing gate lock")
    gate = json.loads(V5_GATE_LOCK.read_text(encoding="utf-8"))
    fail_path = DOCS / "cortex_v5_gate.failure.lock"
    if fail_path.exists():
        raise RuntimeError("failure lock exists")
    out = {
        "version": "TM.0.23.CORTEX.V5.GATE.FAILURE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "gate_sha": _sha_file(V5_GATE_LOCK),
        "sensorimotor_association_gate_clear": False,
        "n_pair_clear": gate.get("battery", {}).get("n_pair_clear"),
        "next": "isolated_v6",
        "refuse": ["edit-and-rescore on revealed v5 worlds"],
    }
    fail_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-apparatus", action="store_true")
    ap.add_argument("--write-birth", action="store_true")
    ap.add_argument("--math-audit", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--mact-boundary", action="store_true")
    ap.add_argument("--reveal-gate", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--freeze-develop-runner", action="store_true")
    ap.add_argument("--publish-develop-commitment", action="store_true")
    ap.add_argument("--reveal-develop", action="store_true")
    ap.add_argument("--develop", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.freeze_apparatus:
        print(json.dumps(freeze_all_v5_apparatus(), indent=2))
    elif args.write_birth:
        print(json.dumps(write_v5_birth(), indent=2, default=str))
    elif args.math_audit:
        print(json.dumps(write_v5_math_audit(), indent=2, default=str))
    elif args.write_candidate:
        print(json.dumps(write_candidate_v5(), indent=2))
    elif args.mact_boundary:
        print(json.dumps(run_boundary_v5(write_lock=args.write_lock), indent=2, default=str))
    elif args.reveal_gate:
        print(json.dumps(reveal_v5_eval(), indent=2))
    elif args.gate:
        print(json.dumps(run_v5_gate(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    elif args.freeze_develop_runner:
        print(json.dumps(freeze_develop_v5_runner(), indent=2))
    elif args.publish_develop_commitment:
        print(json.dumps(publish_develop_v5_commitment(), indent=2))
    elif args.reveal_develop:
        print(json.dumps(reveal_develop_v5(), indent=2))
    elif args.develop:
        print(json.dumps(run_develop_v5(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
