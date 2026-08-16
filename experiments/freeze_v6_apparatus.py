"""Freeze TM.0.23.CORTEX v6 apparatus after v5 diagnosis, before neural edits."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
V1_CONTRACT = DOCS / "cortex_architecture_contract.md"
DIAG = DOCS / "cortex_diagnosis.v5.lock"
V6_AMEND_MD = DOCS / "cortex_v6_architecture_amendment.md"
V6_AMEND_LOCK = DOCS / "cortex_v6_architecture_amendment.lock"
V6_GATE_CONTRACT = DOCS / "cortex_v6_gate_contract.md"
V6_GATE_RUNNER = DOCS / "cortex_v6_gate.runner.lock"
V6_PREREG = DOCS / "cortex_v6.prereg.lock"
V6_SEALED = DOCS / "cortex_v6_eval_secrets.sealed.json"
V5_PREREG = DOCS / "cortex_v5.prereg.lock"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
V6_GATE_PY = REPO_ROOT / "experiments" / "cortex_v6_gate.py"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_v6_amendment() -> dict[str, Any]:
    if not DIAG.exists():
        raise RuntimeError("freeze v5 diagnosis first")
    if not V6_AMEND_MD.exists():
        raise RuntimeError("missing v6 amendment md")
    if V6_AMEND_LOCK.exists():
        raise RuntimeError("v6 amendment lock exists")
    lock = {
        "version": "TM.0.23.CORTEX.V6.ARCHITECTURE.AMENDMENT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract_sha": _sha_file(V1_CONTRACT),
        "diagnosis_v5_sha": _sha_file(DIAG),
        "amendment_md_sha": _sha_file(V6_AMEND_MD),
        "neural_sha_at_freeze": _sha_file(NEURAL_PY),
        "changes_authorized": [
            "exchangeable_motor_slots",
            "unit_motor_vectors",
            "tiebreak_rng_motor",
            "credit_selected_motor_vector_snapshot",
            "skip_motor_query_credit_when_body_adv_zero",
        ],
        "refuse": [
            "edit docs/cortex_architecture_contract.md",
            "soften D1/D2",
            "reuse v5 worlds",
            "DEVELOP before v6 gate ≥13/16",
        ],
    }
    V6_AMEND_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V6_AMEND_LOCK)}


def freeze_v6_gate_runner() -> dict[str, Any]:
    if not V6_AMEND_LOCK.exists():
        raise RuntimeError("freeze amendment first")
    if not V6_GATE_CONTRACT.exists():
        raise RuntimeError("missing v6 gate contract")
    if V6_GATE_RUNNER.exists():
        raise RuntimeError("v6 gate runner exists")
    lock = {
        "version": "TM.0.23.CORTEX.V6.GATE.RUNNER",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_revealed": False,
        "gate_contract_sha": _sha_file(V6_GATE_CONTRACT),
        "architecture_amendment_sha": _sha_file(V6_AMEND_LOCK),
        "diagnosis_v5_sha": _sha_file(DIAG),
        "stages": ["D0", "D1", "D2"],
        "thresholds": THRESHOLDS,
        "scorer_sha": _sha_file(SCORERS_PY),
        "gate_module": "experiments.cortex_v6_gate",
        "gate_module_sha": _sha_file(V6_GATE_PY) if V6_GATE_PY.exists() else None,
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "actuator_api": "bind_actuators([opaque_handle_id, ...])",
            "note": "interface only — no candidate SHA",
        },
        "refuse": [
            "pin candidate SHA before v6 birth",
            "pin eval fixture before reveal",
            "D3-D12 in this gate",
            "soften thresholds",
        ],
    }
    V6_GATE_RUNNER.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V6_GATE_RUNNER)}


def publish_v6_commitment() -> dict[str, Any]:
    if not V6_GATE_RUNNER.exists():
        raise RuntimeError("freeze gate runner first")
    if V6_PREREG.exists():
        raise RuntimeError("v6 prereg exists")
    seed_b = secrets.token_bytes(32)
    salt_b = secrets.token_bytes(32)
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    prior = []
    if V5_PREREG.exists():
        prior.append(json.loads(V5_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"])
    if commitment in prior:
        raise RuntimeError("v6 commitment collided with v5")
    V6_SEALED.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V6.EVAL.SEALED",
                "seed_hex": seed_b.hex(),
                "salt_hex": salt_b.hex(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prereg = {
        "version": "TM.0.23.CORTEX.V6.PREREG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "architecture_amendment_sha": _sha_file(V6_AMEND_LOCK),
        "gate_runner_sha": _sha_file(V6_GATE_RUNNER),
        "diagnosis_v5_sha": _sha_file(DIAG),
        "schedule": ["D0", "D1", "D2"],
        "n_pairs": 16,
        "gate_clear_min_pairs": 13,
        "distinct_from_v5": prior[0] if prior else None,
        "note": "Narrow v6 D1–D2. Not DEVELOP.",
    }
    V6_PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "commitment": commitment, "prereg_sha": _sha_file(V6_PREREG)}


def freeze_all() -> dict[str, Any]:
    return {
        "ok": True,
        "amendment": freeze_v6_amendment(),
        "gate_runner": freeze_v6_gate_runner(),
        "commitment": publish_v6_commitment(),
    }


if __name__ == "__main__":
    print(json.dumps(freeze_all(), indent=2))
