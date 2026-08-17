"""Freeze isolated D5.R1 contract and commitment. No neural edits."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
STAT_MD = DOCS / "cortex_d5_r1_stat_contract.md"
STAT_LOCK = DOCS / "cortex_d5_r1_stat_contract.lock"
GATE_MD = DOCS / "cortex_d5_r1_gate_contract.md"
AMEND_MD = DOCS / "cortex_d5_r1_architecture_amendment.md"
AMEND_LOCK = DOCS / "cortex_d5_r1_architecture_amendment.lock"
DIAG = DOCS / "cortex_diagnosis.fulldev_r3.lock"
ISOL = DOCS / "cortex_d5.isolation.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
CAND18 = DOCS / "cortex.candidate.v18.lock"
GATE_PY = REPO_ROOT / "experiments" / "cortex_d5_r1_gate.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_d5_r1_worlds.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
RUNNER = DOCS / "cortex_d5_r1_gate.runner.lock"
PREREG = DOCS / "cortex_d5_r1.prereg.lock"
SEALED = DOCS / "cortex_d5_r1_eval_secrets.sealed.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_d5_r1_apparatus() -> dict[str, Any]:
    if not DIAG.exists() or not ISOL.exists() or not CAND18.exists():
        raise RuntimeError("freeze FULLDEV.R3 diagnosis + D5 isolation + candidate v18 first")
    if STAT_LOCK.exists() or PREREG.exists():
        raise RuntimeError("D5.R1 apparatus already frozen")
    cand = json.loads(CAND18.read_text(encoding="utf-8"))
    if _sha(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — freeze apparatus before v19 neural edits")
    prior = []
    for name in (
        "cortex_fulldev_r3.prereg.lock",
        "cortex_d4_r2.prereg.lock",
        "cortex_d4_r1.prereg.lock",
        "cortex_fulldev_r2.prereg.lock",
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
                "version": "TM.0.23.CORTEX.D5.R1.STAT.CONTRACT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "stat_md_sha": _sha(STAT_MD),
                "gate_md_sha": _sha(GATE_MD),
                "diagnosis_fulldev_r3_sha": _sha(DIAG),
                "isolation_sha": _sha(ISOL),
                "neural_sha_at_freeze": _sha(NEURAL),
                "neural_change_authorized": True,
                "authorized_law": "habituation_familiarity",
                "d5_requires": ["unknown_holds>=12", "known_nonhold_rate>=0.30", "known_minus_unknown>=0.15"],
                "prefix_stages": ["D1", "D2", "D3", "D4"],
                "gate_clear_min_pairs": THRESHOLDS["gate_clear_min_pairs"],
                "refuse": ["edit cortex_develop_scorers.py", "drop C6", "skip D5", "isolate without D1–D4 prefix"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    AMEND_LOCK.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.D5.R1.ARCHITECTURE.AMENDMENT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "diagnosis_fulldev_r3_sha": _sha(DIAG),
                "stat_contract_sha": _sha(STAT_LOCK),
                "amendment_md_sha": _sha(AMEND_MD),
                "neural_sha_at_freeze": _sha(NEURAL),
                "changes_authorized": ["habituation_familiarity"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    RUNNER.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.D5.R1.GATE.RUNNER",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_revealed": False,
                "stat_contract_sha": _sha(STAT_LOCK),
                "architecture_amendment_sha": _sha(AMEND_LOCK),
                "gate_module": "experiments.cortex_d5_r1_gate",
                "gate_module_sha": _sha(GATE_PY),
                "worlds_module_sha": _sha(WORLDS_PY),
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
        raise RuntimeError("D5.R1 commitment collision")
    SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.D5.R1.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n"
    )
    PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.D5.R1.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "architecture_amendment_sha": _sha(AMEND_LOCK),
                "gate_runner_sha": _sha(RUNNER),
                "stat_contract_sha": _sha(STAT_LOCK),
                "diagnosis_fulldev_r3_sha": _sha(DIAG),
                "schedule": ["D0", "D5"],
                "prefix_scored_not_required": ["D1", "D2", "D3", "D4"],
                "n_pairs": 16,
                "gate_clear_min_pairs": 13,
                "domain": "TM023.D5.R1.",
                "authorized_law": "habituation_familiarity",
                "distinct_from_prior": prior,
                "note": "Narrow D5.R1. Neural habituation familiarity authorized after this freeze.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "commitment": commitment, "neural_unchanged": True, "prereg_sha": _sha(PREREG)}


if __name__ == "__main__":
    print(json.dumps(freeze_d5_r1_apparatus(), indent=2))
