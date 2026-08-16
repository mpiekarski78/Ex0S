"""Freeze TM.0.23.CORTEX v8 scorer contract and commitment. No neural edits. No candidate SHA."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from experiments.cortex_v2_gate import THRESHOLDS
from experiments.cortex_v8_scorers import LIFE_DELTA_MIN, N_PROBES, TEACH_N
from experiments.cortex_v7_stats import MAJORITY_MIN, MAX_NUISANCE_ABS, MEAN_DELTA_MIN, N_PAIRS, PERM_ALPHA, PERM_N

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
V1_CONTRACT = DOCS / "cortex_architecture_contract.md"
STAT_MD = DOCS / "cortex_v8_stat_contract.md"
STAT_LOCK = DOCS / "cortex_v8_stat_contract.lock"
GATE_MD = DOCS / "cortex_v8_gate_contract.md"
AMEND_MD = DOCS / "cortex_v8_architecture_amendment.md"
AMEND_LOCK = DOCS / "cortex_v8_architecture_amendment.lock"
DIAG = DOCS / "cortex_diagnosis.v7.lock"
AUDIT = DOCS / "cortex_v7_gate.audit.lock"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
CANDIDATE_V7 = DOCS / "cortex.candidate.v7.lock"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_v8_scorers.py"
GATE_PY = REPO_ROOT / "experiments" / "cortex_v8_gate.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_v8_worlds.py"
V8_GATE_RUNNER = DOCS / "cortex_v8_gate.runner.lock"
V8_PREREG = DOCS / "cortex_v8.prereg.lock"
V8_SEALED = DOCS / "cortex_v8_eval_secrets.sealed.json"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_commitments() -> list[str]:
    out = []
    for name in (
        "cortex.prereg.lock",
        "cortex_development.prereg.lock",
        "cortex_v2.prereg.lock",
        "cortex_v3.prereg.lock",
        "cortex_v3b.prereg.lock",
        "cortex_v4.prereg.lock",
        "cortex_v5.prereg.lock",
        "cortex_v6.prereg.lock",
        "cortex_v7.prereg.lock",
    ):
        p = DOCS / name
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8")).get("eval_seed_commitment")
            if c:
                out.append(str(c))
    return out


def freeze_v8_apparatus() -> dict[str, Any]:
    if not DIAG.exists() or not AUDIT.exists():
        raise RuntimeError("freeze v7 diagnosis + audit first")
    if STAT_LOCK.exists() or AMEND_LOCK.exists() or V8_PREREG.exists():
        raise RuntimeError("v8 apparatus already frozen")
    cand7 = json.loads(CANDIDATE_V7.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand7["neural_cortex_sha"]:
        raise RuntimeError("neural drifted from candidate v7 — apparatus freeze is scorer-only")
    stat = {
        "version": "TM.0.23.CORTEX.V8.STAT.CONTRACT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract_sha": _sha_file(V1_CONTRACT),
        "stat_md_sha": _sha_file(STAT_MD),
        "gate_md_sha": _sha_file(GATE_MD),
        "diagnosis_v7_sha": _sha_file(DIAG),
        "audit_v7_sha": _sha_file(AUDIT),
        "v7_neural_sha": cand7["neural_cortex_sha"],
        "neural_sha_at_freeze": _sha_file(NEURAL_PY),
        "neural_change_authorized": False,
        "thresholds": {
            "n_pairs": N_PAIRS,
            "n_probes": N_PROBES,
            "teach_n": TEACH_N,
            "majority_min": MAJORITY_MIN,
            "mean_delta_min": MEAN_DELTA_MIN,
            "life_delta_min": LIFE_DELTA_MIN,
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
            "edit-and-rescore v7",
            "neural retune this cycle",
            "open D0–D12 before v8 gate ≥13/16",
        ],
    }
    STAT_LOCK.write_text(json.dumps(stat, indent=2) + "\n", encoding="utf-8")
    amend = {
        "version": "TM.0.23.CORTEX.V8.ARCHITECTURE.AMENDMENT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract_sha": _sha_file(V1_CONTRACT),
        "diagnosis_v7_sha": _sha_file(DIAG),
        "audit_v7_sha": _sha_file(AUDIT),
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "amendment_md_sha": _sha_file(AMEND_MD),
        "neural_sha_at_freeze": _sha_file(NEURAL_PY),
        "changes_authorized": [
            "retain_v7_neural",
            "birth_weight_frozen_probe_extras",
            "life_delta_min_0_10",
            "sealed_seed_derived_worlds",
            "reveal_pins_git_commit",
        ],
        "refuse": ["soften floors", "edit-and-rescore v7", "DEVELOP before v8 gate ≥13/16", "new credit law"],
    }
    AMEND_LOCK.write_text(json.dumps(amend, indent=2) + "\n", encoding="utf-8")
    runner = {
        "version": "TM.0.23.CORTEX.V8.GATE.RUNNER",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_revealed": False,
        "gate_contract_sha": _sha_file(GATE_MD),
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "architecture_amendment_sha": _sha_file(AMEND_LOCK),
        "stages": ["D0", "D1", "D2"],
        "thresholds": THRESHOLDS,
        "scorer_sha": _sha_file(SCORERS_PY),
        "gate_module": "experiments.cortex_v8_gate",
        "gate_module_sha": _sha_file(GATE_PY),
        "worlds_module_sha": _sha_file(WORLDS_PY),
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "note": "interface only — no candidate SHA",
        },
        "refuse": ["pin candidate SHA before v8 birth", "soften floors", "D3-D12"],
    }
    V8_GATE_RUNNER.write_text(json.dumps(runner, indent=2) + "\n", encoding="utf-8")
    seed_b = secrets.token_bytes(32)
    salt_b = secrets.token_bytes(32)
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    prior = _prior_commitments()
    if commitment in prior:
        raise RuntimeError("v8 commitment collided with a prior sealed commitment")
    V8_SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.V8.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    prereg = {
        "version": "TM.0.23.CORTEX.V8.PREREG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "architecture_amendment_sha": _sha_file(AMEND_LOCK),
        "gate_runner_sha": _sha_file(V8_GATE_RUNNER),
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "diagnosis_v7_sha": _sha_file(DIAG),
        "schedule": ["D0", "D1", "D2"],
        "n_pairs": 16,
        "gate_clear_min_pairs": 13,
        "worlds": "sealed_eval_seed",
        "distinct_from_prior": prior,
        "note": "Narrow v8 D1–D2. Scorer-only cycle. Not DEVELOP. No candidate SHA.",
    }
    V8_PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "stat_sha": _sha_file(STAT_LOCK),
        "amendment_sha": _sha_file(AMEND_LOCK),
        "runner_sha": _sha_file(V8_GATE_RUNNER),
        "commitment": commitment,
        "prereg_sha": _sha_file(V8_PREREG),
        "neural_unchanged": True,
    }


if __name__ == "__main__":
    print(json.dumps(freeze_v8_apparatus(), indent=2))
