"""Freeze isolated v27 generality apparatus. No neural edits. No score."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
STAT_MD = DOCS / "cortex_v27_stat_contract.md"
STAT_LOCK = DOCS / "cortex_v27_stat_contract.lock"
GATE_MD = DOCS / "cortex_v27_gate_contract.md"
AMEND_MD = DOCS / "cortex_v27_architecture_amendment.md"
AMEND_LOCK = DOCS / "cortex_v27_architecture_amendment.lock"
DIAG = DOCS / "cortex_diagnosis.v26_generality.lock"
ISOL = DOCS / "cortex_v27.isolation.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
CAND26 = DOCS / "cortex.candidate.v26.lock"
GATE_PY = REPO_ROOT / "experiments" / "cortex_v27_gate.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_v27_worlds.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
GEN_PY = REPO_ROOT / "experiments" / "cortex_v26_generality.py"
RUNNER = DOCS / "cortex_v27_gate.runner.lock"
PREREG = DOCS / "cortex_v27.prereg.lock"
SEALED = DOCS / "cortex_v27_eval_secrets.sealed.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_v27_apparatus() -> dict[str, Any]:
    if not DIAG.exists() or not ISOL.exists() or not CAND26.exists():
        raise RuntimeError("freeze GENERALITY.v26 diagnosis + v27 isolation + candidate.v26 first")
    if STAT_LOCK.exists() or PREREG.exists():
        raise RuntimeError("v27 apparatus already frozen")
    cand = json.loads(CAND26.read_text(encoding="utf-8"))
    if _sha(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — freeze apparatus before v27 neural edits")
    prior = []
    for name in (
        "cortex_d5_r3.prereg.lock",
        "cortex_fulldev_r6.prereg.lock",
        "cortex_d7_r2.prereg.lock",
        "cortex_d7_r1.prereg.lock",
        "cortex_d6_r3.prereg.lock",
        "cortex_fulldev_r5.prereg.lock",
        "cortex_d5_r2.prereg.lock",
        "cortex_d5_r1.prereg.lock",
        "cortex_fulldev_r4.prereg.lock",
        "cortex_d4_r2.prereg.lock",
        "cortex_fulldev_r3.prereg.lock",
        "cortex_d3_r3.prereg.lock",
        "cortex_fulldev_r1.prereg.lock",
        "cortex_development.prereg.lock",
        "cortex.prereg.lock",
    ):
        p = DOCS / name
        if p.exists():
            prior.append(json.loads(p.read_text(encoding="utf-8"))["eval_seed_commitment"])
    STAT_LOCK.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V27.STAT.CONTRACT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "stat_md_sha": _sha(STAT_MD),
                "gate_md_sha": _sha(GATE_MD),
                "diagnosis_v26_generality_sha": _sha(DIAG),
                "isolation_sha": _sha(ISOL),
                "neural_sha_at_freeze": _sha(NEURAL),
                "neural_change_authorized": True,
                "authorized_law": "learned_internal_motor_program_not_sensory_buffer_replay",
                "required": ["G1", "G3", "G5", "C4", "C5", "C6"],
                "gate_clear_min_pairs": THRESHOLDS["gate_clear_min_pairs"],
                "refuse": [
                    "edit cortex_develop_scorers.py",
                    "keep phrase_program",
                    "reveal FULLDEV.R7",
                    "soften G3/G5 bars",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    AMEND_LOCK.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V27.ARCHITECTURE.AMENDMENT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "diagnosis_v26_generality_sha": _sha(DIAG),
                "stat_contract_sha": _sha(STAT_LOCK),
                "amendment_md_sha": _sha(AMEND_MD),
                "neural_sha_at_freeze": _sha(NEURAL),
                "changes_authorized": ["learned_internal_motor_program_not_sensory_buffer_replay"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    RUNNER.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V27.GEN.GATE.RUNNER",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_revealed": False,
                "stat_contract_sha": _sha(STAT_LOCK),
                "architecture_amendment_sha": _sha(AMEND_LOCK),
                "gate_module": "experiments.cortex_v27_gate",
                "gate_module_sha": _sha(GATE_PY),
                "worlds_module_sha": _sha(WORLDS_PY),
                "generality_module_sha": _sha(GEN_PY),
                "scorer_sha": _sha(SCORERS_PY),
                "candidate_interface": {"factory": "experiments.run_tm023cortex.make_cortex", "note": "no candidate SHA"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    seed_b, salt_b = secrets.token_bytes(32), secrets.token_bytes(32)
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    if commitment in prior:
        raise RuntimeError("V27.GEN commitment collision")
    SEALED.write_text(
        json.dumps(
            {"version": "TM.0.23.CORTEX.V27.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()},
            indent=2,
        )
        + "\n"
    )
    PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V27.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "architecture_amendment_sha": _sha(AMEND_LOCK),
                "gate_runner_sha": _sha(RUNNER),
                "stat_contract_sha": _sha(STAT_LOCK),
                "diagnosis_v26_generality_sha": _sha(DIAG),
                "schedule": ["G1", "G3", "G5"],
                "n_pairs": 16,
                "gate_clear_min_pairs": 13,
                "domain": "TM023.V27.GEN.",
                "authorized_law": "learned_internal_motor_program_not_sensory_buffer_replay",
                "distinct_from_prior": prior,
                "note": "Narrow v27 G1+G3+G5. Neural edit authorized after this freeze is on origin/main.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "commitment": commitment,
        "neural_unchanged": True,
        "prereg_sha": _sha(PREREG),
        "authorized_law": "learned_internal_motor_program_not_sensory_buffer_replay",
    }


if __name__ == "__main__":
    print(json.dumps(freeze_v27_apparatus(), indent=2))
