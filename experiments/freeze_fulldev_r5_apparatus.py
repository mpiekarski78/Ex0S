"""Freeze FULLDEV.R5 commitment. No neural edits. No score."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
CONTRACT_MD = DOCS / "cortex_fulldev_r5_contract.md"
CONTRACT_LOCK = DOCS / "cortex_fulldev_r5_contract.lock"
CLEAR = DOCS / "cortex_d6_r3_gate.clear.note.lock"
CAND23 = DOCS / "cortex.candidate.v23.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
GATE_PY = REPO_ROOT / "experiments" / "cortex_fulldev_r5.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_fulldev_r5_worlds.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
LIFE_PY = REPO_ROOT / "experiments" / "cortex_develop_life.py"
DEV_CONTRACT = DOCS / "cortex_development_contract.md"
RUNNER = DOCS / "cortex_fulldev_r5.runner.lock"
PREREG = DOCS / "cortex_fulldev_r5.prereg.lock"
SEALED = DOCS / "cortex_fulldev_r5_eval_secrets.sealed.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_fulldev_r5_apparatus() -> dict[str, Any]:
    if not CLEAR.exists() or not CAND23.exists():
        raise RuntimeError("D6.R3 clear + candidate v23 required")
    if PREREG.exists() or RUNNER.exists():
        raise RuntimeError("FULLDEV.R5 apparatus already frozen")
    cand = json.loads(CAND23.read_text(encoding="utf-8"))
    if _sha(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — freeze apparatus on the v23 neural")
    prior = []
    for name in (
        "cortex_d6_r3.prereg.lock",
        "cortex_d6_r2.prereg.lock",
        "cortex_d6_r1.prereg.lock",
        "cortex_fulldev_r4.prereg.lock",
        "cortex_d5_r2.prereg.lock",
        "cortex_d5_r1.prereg.lock",
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
    CONTRACT_LOCK.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.FULLDEV.R5.CONTRACT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "contract_md_sha": _sha(CONTRACT_MD),
                "development_contract_sha": _sha(DEV_CONTRACT),
                "d6_r3_clear_note_sha": _sha(CLEAR),
                "candidate_v23_sha": _sha(CAND23),
                "neural_sha_at_freeze": _sha(NEURAL),
                "scorer_sha": _sha(SCORERS_PY),
                "life_sha": _sha(LIFE_PY),
                "domain": "TM023.FULL.R5.",
                "refuse": [
                    "pair_seeds table",
                    "TM023.FULL.R1. / FULL.R2. / FULL.R3. / FULL.R4. worlds",
                    "D3.R* / D4.R* / D5.R* / D6.R* worlds",
                    "TM023.V8–V23 worlds",
                    "DEVELOP.v1–v4 commitments",
                    "stamp earned_next",
                    "neural edit during recorded run",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    RUNNER.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.FULLDEV.R5.RUNNER",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_revealed": False,
                "stages": [f"D{i}" for i in range(13)],
                "contract_sha": _sha(CONTRACT_LOCK),
                "gate_module": "experiments.cortex_fulldev_r5",
                "gate_module_sha": _sha(GATE_PY),
                "worlds_module_sha": _sha(WORLDS_PY),
                "scorer_sha": _sha(SCORERS_PY),
                "life_sha": _sha(LIFE_PY),
                "candidate_v23_sha": _sha(CAND23),
                "neural_cortex_sha": cand["neural_cortex_sha"],
                "candidate_interface": {"factory": "experiments.run_tm023cortex.make_cortex", "note": "no new candidate SHA"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    seed_b, salt_b = secrets.token_bytes(32), secrets.token_bytes(32)
    commitment = hashlib.sha256(seed_b + salt_b).hexdigest()
    if commitment in prior:
        raise RuntimeError("FULLDEV.R5 commitment collision")
    SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.FULLDEV.R5.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n"
    )
    PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.FULLDEV.R5.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "contract_sha": _sha(CONTRACT_LOCK),
                "runner_sha": _sha(RUNNER),
                "candidate_v23_sha": _sha(CAND23),
                "schedule": [f"D{i}" for i in range(13)],
                "n_pairs": 16,
                "gate_clear_min_pairs": 13,
                "domain": "TM023.FULL.R5.",
                "distinct_from_prior": prior,
                "note": "Full D0–D12 on unused sealed worlds after isolated D3.R3, D5.R2, and D6.R3 clears. Not a product stamp.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "commitment": commitment, "neural_unchanged": True, "prereg_sha": _sha(PREREG)}


if __name__ == "__main__":
    print(json.dumps(freeze_fulldev_r5_apparatus(), indent=2))
