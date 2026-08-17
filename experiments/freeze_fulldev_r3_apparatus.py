"""Freeze FULLDEV.R3 commitment. No neural edits. No score."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
CONTRACT_MD = DOCS / "cortex_fulldev_r3_contract.md"
CONTRACT_LOCK = DOCS / "cortex_fulldev_r3_contract.lock"
CLEAR = DOCS / "cortex_d4_r2_gate.clear.note.lock"
CAND18 = DOCS / "cortex.candidate.v18.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
GATE_PY = REPO_ROOT / "experiments" / "cortex_fulldev_r3.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_fulldev_r3_worlds.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
LIFE_PY = REPO_ROOT / "experiments" / "cortex_develop_life.py"
DEV_CONTRACT = DOCS / "cortex_development_contract.md"
RUNNER = DOCS / "cortex_fulldev_r3.runner.lock"
PREREG = DOCS / "cortex_fulldev_r3.prereg.lock"
SEALED = DOCS / "cortex_fulldev_r3_eval_secrets.sealed.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_fulldev_r3_apparatus() -> dict[str, Any]:
    if not CLEAR.exists() or not CAND18.exists():
        raise RuntimeError("D4.R2 clear + candidate v18 required")
    if PREREG.exists() or RUNNER.exists():
        raise RuntimeError("FULLDEV.R3 apparatus already frozen")
    cand = json.loads(CAND18.read_text(encoding="utf-8"))
    if _sha(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — freeze apparatus on the v18 neural")
    prior = []
    for name in (
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
                "version": "TM.0.23.CORTEX.FULLDEV.R3.CONTRACT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "contract_md_sha": _sha(CONTRACT_MD),
                "development_contract_sha": _sha(DEV_CONTRACT),
                "d4_r2_clear_note_sha": _sha(CLEAR),
                "candidate_v18_sha": _sha(CAND18),
                "neural_sha_at_freeze": _sha(NEURAL),
                "scorer_sha": _sha(SCORERS_PY),
                "life_sha": _sha(LIFE_PY),
                "domain": "TM023.FULL.R3.",
                "refuse": [
                    "pair_seeds table",
                    "TM023.FULL.R1. / FULL.R2. worlds",
                    "D3.R* / D4.R* worlds",
                    "TM023.V8–V18 worlds",
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
                "version": "TM.0.23.CORTEX.FULLDEV.R3.RUNNER",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_revealed": False,
                "stages": [f"D{i}" for i in range(13)],
                "contract_sha": _sha(CONTRACT_LOCK),
                "gate_module": "experiments.cortex_fulldev_r3",
                "gate_module_sha": _sha(GATE_PY),
                "worlds_module_sha": _sha(WORLDS_PY),
                "scorer_sha": _sha(SCORERS_PY),
                "life_sha": _sha(LIFE_PY),
                "candidate_v18_sha": _sha(CAND18),
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
        raise RuntimeError("FULLDEV.R3 commitment collision")
    SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.FULLDEV.R3.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n"
    )
    PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.FULLDEV.R3.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "contract_sha": _sha(CONTRACT_LOCK),
                "runner_sha": _sha(RUNNER),
                "candidate_v18_sha": _sha(CAND18),
                "schedule": [f"D{i}" for i in range(13)],
                "n_pairs": 16,
                "gate_clear_min_pairs": 13,
                "domain": "TM023.FULL.R3.",
                "distinct_from_prior": prior,
                "note": "Full D0–D12 on unused sealed worlds after isolated D3.R3 and D4.R2 clears. Not a product stamp.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "commitment": commitment, "neural_unchanged": True, "prereg_sha": _sha(PREREG)}


if __name__ == "__main__":
    print(json.dumps(freeze_fulldev_r3_apparatus(), indent=2))
