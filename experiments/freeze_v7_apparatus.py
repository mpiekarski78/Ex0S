"""Freeze TM.0.23.CORTEX v7 statistical contract before diagnosis / neural edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS
from experiments.cortex_v7_stats import (
    MAJORITY_MIN,
    MAX_NUISANCE_ABS,
    MEAN_DELTA_MIN,
    N_PAIRS,
    N_PROBES,
    PERM_ALPHA,
    PERM_N,
    TEACH_N,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
V1_CONTRACT = DOCS / "cortex_architecture_contract.md"
STAT_MD = DOCS / "cortex_v7_stat_contract.md"
STAT_LOCK = DOCS / "cortex_v7_stat_contract.lock"
GATE_MD = DOCS / "cortex_v7_gate_contract.md"
BASELINE = "97691cd"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
CANDIDATE_V6 = DOCS / "cortex.candidate.v6.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_stat_contract() -> dict[str, Any]:
    if not STAT_MD.exists() or not GATE_MD.exists():
        raise RuntimeError("missing v7 stat/gate contract md")
    if STAT_LOCK.exists():
        raise RuntimeError("v7 stat contract lock exists")
    cand = json.loads(CANDIDATE_V6.read_text(encoding="utf-8"))
    lock = {
        "version": "TM.0.23.CORTEX.V7.STAT.CONTRACT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "canonical_baseline": BASELINE,
        "v1_architecture_contract_sha": _sha_file(V1_CONTRACT),
        "stat_md_sha": _sha_file(STAT_MD),
        "gate_md_sha": _sha_file(GATE_MD),
        "v6_candidate_sha": _sha_file(CANDIDATE_V6),
        "v6_neural_sha": cand["neural_cortex_sha"],
        "neural_sha_at_freeze": _sha_file(NEURAL_PY),
        "thresholds": {
            "n_pairs": N_PAIRS,
            "n_probes": N_PROBES,
            "teach_n": TEACH_N,
            "majority_min": MAJORITY_MIN,
            "mean_delta_min": MEAN_DELTA_MIN,
            "max_nuisance_abs": MAX_NUISANCE_ABS,
            "perm_n": PERM_N,
            "perm_alpha": PERM_ALPHA,
            "d1_floors": {"press_min": 3, "press_gt_harm": True},
            "d2_floors": {"holds_min": 5, "beneficial_min": 3},
            "gate_clear_min_pairs": THRESHOLDS["gate_clear_min_pairs"],
        },
        "refuse": [
            "soften D1/D2 floors",
            "edit docs/cortex_architecture_contract.md",
            "edit-and-rescore v6",
            "DEVELOP.v6",
            "tune neural so every frozen individual fails deterministic D1",
            "open D0–D12 before v7 gate ≥13/16",
        ],
        "note": "Frozen before v6 population diagnosis and before any v7 neural edit.",
    }
    STAT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(STAT_LOCK)}


DIAG = DOCS / "cortex_diagnosis.v6.lock"
DIAG_NOTE = DOCS / "cortex_diagnosis.v6.note.lock"
V7_AMEND_MD = DOCS / "cortex_v7_architecture_amendment.md"
V7_AMEND_LOCK = DOCS / "cortex_v7_architecture_amendment.lock"
V7_GATE_RUNNER = DOCS / "cortex_v7_gate.runner.lock"
V7_PREREG = DOCS / "cortex_v7.prereg.lock"
V7_SEALED = DOCS / "cortex_v7_eval_secrets.sealed.json"
V6_PREREG = DOCS / "cortex_v6.prereg.lock"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_v7_scorers.py"
V7_GATE_PY = REPO_ROOT / "experiments" / "cortex_v7_gate.py"


def freeze_v7_amendment() -> dict[str, Any]:
    if not DIAG.exists() or not STAT_LOCK.exists():
        raise RuntimeError("freeze diagnosis + stat contract first")
    if V7_AMEND_LOCK.exists():
        raise RuntimeError("v7 amendment lock exists")
    lock = {
        "version": "TM.0.23.CORTEX.V7.ARCHITECTURE.AMENDMENT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract_sha": _sha_file(V1_CONTRACT),
        "diagnosis_v6_sha": _sha_file(DIAG),
        "diagnosis_note_sha": _sha_file(DIAG_NOTE) if DIAG_NOTE.exists() else None,
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "amendment_md_sha": _sha_file(V7_AMEND_MD),
        "neural_sha_at_freeze": _sha_file(NEURAL_PY),
        "changes_authorized": [
            "retain_v6_motor_geometry_and_c4",
            "skip_w_op_act_when_body_adv_zero",
            "population_c5_c6",
            "d1_d2_paired_baselines",
        ],
        "refuse": [
            "soften D1/D2 floors",
            "edit docs/cortex_architecture_contract.md",
            "edit-and-rescore v6",
            "DEVELOP before v7 gate ≥13/16",
        ],
    }
    V7_AMEND_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V7_AMEND_LOCK)}


def freeze_v7_gate_runner() -> dict[str, Any]:
    if not V7_AMEND_LOCK.exists():
        raise RuntimeError("freeze amendment first")
    if V7_GATE_RUNNER.exists():
        raise RuntimeError("v7 gate runner exists")
    lock = {
        "version": "TM.0.23.CORTEX.V7.GATE.RUNNER",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_revealed": False,
        "gate_contract_sha": _sha_file(GATE_MD),
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "architecture_amendment_sha": _sha_file(V7_AMEND_LOCK),
        "stages": ["D0", "D1", "D2"],
        "thresholds": THRESHOLDS,
        "scorer_sha": _sha_file(SCORERS_PY) if SCORERS_PY.exists() else None,
        "gate_module": "experiments.cortex_v7_gate",
        "gate_module_sha": _sha_file(V7_GATE_PY) if V7_GATE_PY.exists() else None,
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "note": "interface only — no candidate SHA",
        },
        "refuse": ["pin candidate SHA before v7 birth", "soften floors", "D3-D12"],
    }
    V7_GATE_RUNNER.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "sha": _sha_file(V7_GATE_RUNNER)}


def publish_v7_commitment() -> dict[str, Any]:
    import secrets

    if not V7_GATE_RUNNER.exists():
        raise RuntimeError("freeze gate runner first")
    if V7_PREREG.exists():
        raise RuntimeError("v7 prereg exists")
    seed_b = secrets.token_bytes(32)
    salt_b = secrets.token_bytes(32)
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    prior: list[str] = []
    for name in (
        "cortex.prereg.lock",
        "cortex_development.prereg.lock",
        "cortex_v2.prereg.lock",
        "cortex_v3.prereg.lock",
        "cortex_v3b.prereg.lock",
        "cortex_v4.prereg.lock",
        "cortex_v5.prereg.lock",
        "cortex_v6.prereg.lock",
    ):
        p = DOCS / name
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8")).get("eval_seed_commitment")
            if c:
                prior.append(str(c))
    if commitment in prior:
        raise RuntimeError("v7 commitment collided with a prior sealed commitment")
    V7_SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.V7.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    prereg = {
        "version": "TM.0.23.CORTEX.V7.PREREG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "architecture_amendment_sha": _sha_file(V7_AMEND_LOCK),
        "gate_runner_sha": _sha_file(V7_GATE_RUNNER),
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "diagnosis_v6_sha": _sha_file(DIAG),
        "schedule": ["D0", "D1", "D2"],
        "n_pairs": 16,
        "gate_clear_min_pairs": 13,
        "distinct_from_prior": prior,
        "note": "Narrow v7 D1–D2. Not DEVELOP.",
    }
    V7_PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "commitment": commitment, "prereg_sha": _sha_file(V7_PREREG)}


def freeze_after_diagnosis() -> dict[str, Any]:
    return {
        "ok": True,
        "amendment": freeze_v7_amendment(),
        "gate_runner": freeze_v7_gate_runner(),
        "commitment": publish_v7_commitment(),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", action="store_true")
    ap.add_argument("--after-diagnosis", action="store_true")
    args = ap.parse_args()
    if args.after_diagnosis:
        print(json.dumps(freeze_after_diagnosis(), indent=2))
    else:
        print(json.dumps(freeze_stat_contract(), indent=2))

