"""Freeze v12 contract and commitment. No neural edits. No candidate SHA."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS
from experiments.cortex_v10_scorers import DELTA, MAJORITY, MEAN_MIN

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
V1 = DOCS / "cortex_architecture_contract.md"
STAT_MD = DOCS / "cortex_v12_stat_contract.md"
STAT_LOCK = DOCS / "cortex_v12_stat_contract.lock"
GATE_MD = DOCS / "cortex_v12_gate_contract.md"
AMEND_MD = DOCS / "cortex_v12_architecture_amendment.md"
AMEND_LOCK = DOCS / "cortex_v12_architecture_amendment.lock"
DIAG = DOCS / "cortex_diagnosis.v11.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
CAND11 = DOCS / "cortex.candidate.v11.lock"
GATE_PY = REPO_ROOT / "experiments" / "cortex_v12_gate.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_v12_worlds.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_v11_scorers.py"
RUNNER = DOCS / "cortex_v12_gate.runner.lock"
PREREG = DOCS / "cortex_v12.prereg.lock"
SEALED = DOCS / "cortex_v12_eval_secrets.sealed.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_v12_apparatus() -> dict[str, Any]:
    if not DIAG.exists():
        raise RuntimeError("freeze v11 diagnosis first")
    if STAT_LOCK.exists() or PREREG.exists():
        raise RuntimeError("v12 apparatus already frozen")
    cand = json.loads(CAND11.read_text(encoding="utf-8"))
    if _sha(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — freeze apparatus before neural v12 edits")
    prior = []
    for name in (
        "cortex_v11.prereg.lock",
        "cortex_v10.prereg.lock",
        "cortex_v9.prereg.lock",
        "cortex.prereg.lock",
        "cortex_development.prereg.lock",
    ):
        p = DOCS / name
        if p.exists():
            prior.append(json.loads(p.read_text(encoding="utf-8"))["eval_seed_commitment"])
    stat = {
        "version": "TM.0.23.CORTEX.V12.STAT.CONTRACT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract_sha": _sha(V1),
        "stat_md_sha": _sha(STAT_MD),
        "gate_md_sha": _sha(GATE_MD),
        "diagnosis_v11_sha": _sha(DIAG),
        "neural_sha_at_freeze": _sha(NEURAL),
        "neural_change_authorized": True,
        "authorized_law": "surprise_conflict_hold",
        "conflict_adv_eps": 1e-9,
        "conflict_hold_bias": 2.0,
        "d1_bind": ["press", "harm"],
        "d2_conflict": "swapped_press_harm",
        "extras": "population",
        "majority_min": MAJORITY,
        "mean_delta_min": MEAN_MIN,
        "life_delta_min": DELTA,
        "gate_clear_min_pairs": THRESHOLDS["gate_clear_min_pairs"],
        "refuse": ["lower holds>=5", "edit-and-rescore v11", "capability-specific functions"],
    }
    STAT_LOCK.write_text(json.dumps(stat, indent=2) + "\n", encoding="utf-8")
    AMEND_LOCK.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V12.ARCHITECTURE.AMENDMENT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "diagnosis_v11_sha": _sha(DIAG),
                "stat_contract_sha": _sha(STAT_LOCK),
                "amendment_md_sha": _sha(AMEND_MD),
                "neural_sha_at_freeze": _sha(NEURAL),
                "changes_authorized": ["surprise_conflict_hold"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    RUNNER.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V12.GATE.RUNNER",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_revealed": False,
                "stat_contract_sha": _sha(STAT_LOCK),
                "architecture_amendment_sha": _sha(AMEND_LOCK),
                "gate_module": "experiments.cortex_v12_gate",
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
        raise RuntimeError("v12 commitment collision")
    SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.V12.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n"
    )
    PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.V12.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "architecture_amendment_sha": _sha(AMEND_LOCK),
                "gate_runner_sha": _sha(RUNNER),
                "stat_contract_sha": _sha(STAT_LOCK),
                "diagnosis_v11_sha": _sha(DIAG),
                "schedule": ["D0", "D1", "D2"],
                "n_pairs": 16,
                "gate_clear_min_pairs": 13,
                "d1_bind": ["press", "harm"],
                "d2_conflict": "swapped_press_harm",
                "extras": "population",
                "authorized_law": "surprise_conflict_hold",
                "distinct_from_prior": prior,
                "note": "Narrow v12 D1–D2. Neural surprise→HOLD authorized after this freeze. Not DEVELOP.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "commitment": commitment, "neural_unchanged": True, "prereg_sha": _sha(PREREG)}


if __name__ == "__main__":
    print(json.dumps(freeze_v12_apparatus(), indent=2))
