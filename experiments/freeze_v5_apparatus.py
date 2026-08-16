"""Freeze TM.0.23.CORTEX v5 apparatus (amendment, boundary.v2, gate, commitment)."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

V5_AMEND_MD = DOCS / "cortex_v5_architecture_amendment.md"
V5_AMEND_LOCK = DOCS / "cortex_v5_architecture_amendment.lock"
V1_CONTRACT = DOCS / "cortex_architecture_contract.md"
MACT_CONTRACT = DOCS / "cortex_mact_boundary_contract.md"
MACT_BOUNDARY_LOCK = DOCS / "cortex_mact_boundary.lock"
MACT_RUNNER_V1 = DOCS / "cortex_mact_boundary.runner.lock"
MACT_RUNNER_V2 = DOCS / "cortex_mact_boundary.runner.v2.lock"
MACT_MOD = REPO_ROOT / "experiments" / "cortex_mact_boundary.py"
V5_GATE_CONTRACT = DOCS / "cortex_v5_gate_contract.md"
V5_GATE_RUNNER = DOCS / "cortex_v5_gate.runner.lock"
V5_GATE_PY = REPO_ROOT / "experiments" / "cortex_v5_gate.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
V5_PREREG = DOCS / "cortex_v5.prereg.lock"
V5_SEALED = DOCS / "cortex_v5_eval_secrets.sealed.json"
V4_PREREG = DOCS / "cortex_v4.prereg.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def freeze_v5_amendment() -> dict[str, Any]:
    if not V5_AMEND_MD.exists():
        raise RuntimeError("missing cortex_v5_architecture_amendment.md")
    if not MACT_BOUNDARY_LOCK.exists():
        raise RuntimeError("run v4 mact boundary first")
    if V5_AMEND_LOCK.exists():
        raise RuntimeError("v5 amendment lock exists — refuse rewrite")
    lock = {
        "version": "TM.0.23.CORTEX.V5.ARCHITECTURE.AMENDMENT",
        "lab": "TM.0.23.CORTEX.MACT.BOUNDARY",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract": "docs/cortex_architecture_contract.md",
        "v1_architecture_contract_sha": _sha_file(V1_CONTRACT),
        "amendment_md": "docs/cortex_v5_architecture_amendment.md",
        "amendment_md_sha": _sha_file(V5_AMEND_MD),
        "mact_boundary_result_sha": _sha_file(MACT_BOUNDARY_LOCK),
        "changes_authorized": [
            "remove_birth_MOTOR_ACT_TOKENS_planting",
            "bind_actuators_opaque_handle_ids",
            "internal_motor_registry_RNG",
            "seed_motor_separate_streams",
            "handles_never_sensory_vocab",
        ],
        "forbidden_api": "bind_actuators([{id, vector}, ...])",
        "retain": ["b_op[ACT]=0.85 frozen", "OP_COST[ACT]=0.05"],
        "refuse": [
            "edit docs/cortex_architecture_contract.md",
            "soften D1/D2 scorers",
            "runner-supplied motor vectors",
            "reuse v4 gate worlds",
        ],
        "note": "Frozen as v5 apparatus. Original v1 contract untouched.",
    }
    V5_AMEND_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(V5_AMEND_LOCK), "sha": _sha_file(V5_AMEND_LOCK)}


def freeze_mact_boundary_runner_v2() -> dict[str, Any]:
    if not V5_AMEND_LOCK.exists():
        raise RuntimeError("freeze v5 amendment first")
    if not MACT_RUNNER_V1.exists():
        raise RuntimeError("missing v1 boundary runner lock")
    if MACT_RUNNER_V2.exists():
        raise RuntimeError("runner.v2.lock exists — refuse rewrite")
    lock = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RUNNER.V2",
        "lab": "TM.0.23.CORTEX.MACT.BOUNDARY",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "contract": "docs/cortex_mact_boundary_contract.md",
        "contract_sha": _sha_file(MACT_CONTRACT),
        "boundary_module": "experiments.cortex_mact_boundary",
        "boundary_module_sha": _sha_file(MACT_MOD),
        "architecture_amendment_lock": "docs/cortex_v5_architecture_amendment.lock",
        "architecture_amendment_sha": _sha_file(V5_AMEND_LOCK),
        "supersedes_runner": "docs/cortex_mact_boundary.runner.lock",
        "v1_runner_sha": _sha_file(MACT_RUNNER_V1),
        "swap_revise_episodes": 40,
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "class": "NeuralCortex",
            "actuator_api": "bind_actuators([opaque_handle_id, ...])",
            "note": "v2 runner targets generic motor-registry ABI; does not pin candidate SHA",
        },
        "refuse": [
            "rewrite v1 runner.lock",
            "pin candidate SHA before v5 birth",
            "soften controls",
        ],
        "note": "Frozen before / with v5 neural ABI. v1 runner remains immutable.",
    }
    MACT_RUNNER_V2.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(MACT_RUNNER_V2), "sha": _sha_file(MACT_RUNNER_V2)}


def freeze_v5_gate_runner() -> dict[str, Any]:
    if not V5_GATE_CONTRACT.exists():
        raise RuntimeError("missing cortex_v5_gate_contract.md")
    if not V5_AMEND_LOCK.exists():
        raise RuntimeError("freeze amendment first")
    if V5_GATE_RUNNER.exists():
        raise RuntimeError("v5 gate runner exists — refuse rewrite")
    lock = {
        "version": "TM.0.23.CORTEX.V5.GATE.RUNNER",
        "lab": "TM.0.23.CORTEX.V5.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_revealed": False,
        "gate_contract": "docs/cortex_v5_gate_contract.md",
        "gate_contract_sha": _sha_file(V5_GATE_CONTRACT),
        "architecture_amendment_lock": "docs/cortex_v5_architecture_amendment.lock",
        "architecture_amendment_sha": _sha_file(V5_AMEND_LOCK),
        "stages": ["D0", "D1", "D2"],
        "thresholds": THRESHOLDS,
        "scorer_module": "experiments.cortex_develop_scorers",
        "scorer_sha": _sha_file(SCORERS_PY),
        "gate_module": "experiments.cortex_v5_gate",
        "gate_module_sha": _sha_file(V5_GATE_PY),
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "class": "NeuralCortex",
            "actuator_api": "bind_actuators([opaque_handle_id, ...])",
            "note": "Pins interface only — no candidate SHA (v5 may not exist at freeze)",
        },
        "refuse": [
            "pin candidate SHA before v5 birth",
            "pin eval fixture before reveal",
            "D3-D12 in this gate",
            "soften thresholds",
            "reuse v4 commitment/worlds",
            "edit-and-rescore on revealed v5 worlds",
        ],
        "note": "Frozen as v5 apparatus. Thresholds match v2/v4 pair-clear rule.",
    }
    V5_GATE_RUNNER.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(V5_GATE_RUNNER), "sha": _sha_file(V5_GATE_RUNNER)}


def publish_v5_commitment() -> dict[str, Any]:
    if not V5_GATE_RUNNER.exists():
        raise RuntimeError("freeze v5 gate runner first")
    if V5_PREREG.exists():
        raise RuntimeError("cortex_v5.prereg.lock already exists — refuse rewrite")
    seed_b = secrets.token_bytes(32)
    salt_b = secrets.token_bytes(32)
    commitment = _sha_bytes(seed_b + salt_b)
    sealed = {
        "version": "TM.0.23.CORTEX.V5.EVAL.SEALED",
        "seed_hex": seed_b.hex(),
        "salt_hex": salt_b.hex(),
        "note": "Local only until post-candidate-v5 reveal. Distinct from v4 and DEVELOP.v5.",
    }
    V5_SEALED.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    v4_commit = None
    if V4_PREREG.exists():
        v4_commit = json.loads(V4_PREREG.read_text(encoding="utf-8")).get(
            "eval_seed_commitment"
        )
    if v4_commit and commitment == v4_commit:
        raise RuntimeError("v5 commitment collided with v4 — regenerate")
    prereg = {
        "version": "TM.0.23.CORTEX.V5.PREREG",
        "lab": "TM.0.23.CORTEX.V5.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "architecture_amendment_sha": _sha_file(V5_AMEND_LOCK),
        "gate_runner_sha": _sha_file(V5_GATE_RUNNER),
        "schedule": ["D0", "D1", "D2"],
        "n_pairs": 16,
        "gate_clear_min_pairs": 13,
        "distinct_from_v4_commitment": v4_commit,
        "note": "Narrow v5 D1–D2 gate commitment. Not DEVELOP.v5.",
    }
    V5_PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "prereg_sha": _sha_file(V5_PREREG),
        "commitment": commitment,
        "sealed_path": str(V5_SEALED),
    }


def freeze_all_v5_apparatus() -> dict[str, Any]:
    out = {
        "amendment": freeze_v5_amendment(),
        "boundary_runner_v2": freeze_mact_boundary_runner_v2(),
        "gate_runner": freeze_v5_gate_runner(),
        "commitment": publish_v5_commitment(),
    }
    return {"ok": True, "steps": out}
