"""Freeze FULLDEV.R1 commitment. No neural edits. No score."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
CONTRACT_MD = DOCS / "cortex_fulldev_r1_contract.md"
CONTRACT_LOCK = DOCS / "cortex_fulldev_r1_contract.lock"
CLEAR = DOCS / "cortex_v13_gate.clear.note.lock"
CAND13 = DOCS / "cortex.candidate.v13.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
GATE_PY = REPO_ROOT / "experiments" / "cortex_fulldev_r1.py"
WORLDS_PY = REPO_ROOT / "experiments" / "cortex_fulldev_r1_worlds.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
LIFE_PY = REPO_ROOT / "experiments" / "cortex_develop_life.py"
DEV_CONTRACT = DOCS / "cortex_development_contract.md"
RUNNER = DOCS / "cortex_fulldev_r1.runner.lock"
PREREG = DOCS / "cortex_fulldev_r1.prereg.lock"
SEALED = DOCS / "cortex_fulldev_r1_eval_secrets.sealed.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_fulldev_r1_apparatus() -> dict[str, Any]:
    if not CLEAR.exists() or not CAND13.exists():
        raise RuntimeError("v13 clear + candidate required")
    if PREREG.exists() or RUNNER.exists():
        raise RuntimeError("FULLDEV.R1 apparatus already frozen")
    cand = json.loads(CAND13.read_text(encoding="utf-8"))
    if _sha(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — freeze apparatus on the v13 neural")
    prior = []
    for name in (
        "cortex_v13.prereg.lock",
        "cortex_v12.prereg.lock",
        "cortex_v11.prereg.lock",
        "cortex_v10.prereg.lock",
        "cortex_v9.prereg.lock",
        "cortex_v8.prereg.lock",
        "cortex.prereg.lock",
        "cortex_development.prereg.lock",
        "cortex_development.prereg.v1.lock",
    ):
        p = DOCS / name
        if p.exists():
            prior.append(json.loads(p.read_text(encoding="utf-8"))["eval_seed_commitment"])
    CONTRACT_LOCK.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.FULLDEV.R1.CONTRACT",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "contract_md_sha": _sha(CONTRACT_MD),
                "development_contract_sha": _sha(DEV_CONTRACT),
                "v13_clear_note_sha": _sha(CLEAR),
                "candidate_v13_sha": _sha(CAND13),
                "neural_sha_at_freeze": _sha(NEURAL),
                "scorer_sha": _sha(SCORERS_PY),
                "life_sha": _sha(LIFE_PY),
                "domain": "TM023.FULL.R1.",
                "refuse": [
                    "pair_seeds table",
                    "TM023.V8–V13 worlds",
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
                "version": "TM.0.23.CORTEX.FULLDEV.R1.RUNNER",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_revealed": False,
                "stages": [f"D{i}" for i in range(13)],
                "contract_sha": _sha(CONTRACT_LOCK),
                "gate_module": "experiments.cortex_fulldev_r1",
                "gate_module_sha": _sha(GATE_PY),
                "worlds_module_sha": _sha(WORLDS_PY),
                "scorer_sha": _sha(SCORERS_PY),
                "life_sha": _sha(LIFE_PY),
                "candidate_v13_sha": _sha(CAND13),
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
        raise RuntimeError("FULLDEV.R1 commitment collision")
    SEALED.write_text(
        json.dumps({"version": "TM.0.23.CORTEX.FULLDEV.R1.EVAL.SEALED", "seed_hex": seed_b.hex(), "salt_hex": salt_b.hex()}, indent=2)
        + "\n"
    )
    PREREG.write_text(
        json.dumps(
            {
                "version": "TM.0.23.CORTEX.FULLDEV.R1.PREREG",
                "product": "0.0.004",
                "earned_next": False,
                "ex0s": None,
                "eval_seed_commitment": commitment,
                "contract_sha": _sha(CONTRACT_LOCK),
                "runner_sha": _sha(RUNNER),
                "candidate_v13_sha": _sha(CAND13),
                "schedule": [f"D{i}" for i in range(13)],
                "n_pairs": 16,
                "gate_clear_min_pairs": 13,
                "domain": "TM023.FULL.R1.",
                "distinct_from_prior": prior,
                "note": "Full D0–D12 on unused sealed worlds. Not a product stamp.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "commitment": commitment, "neural_unchanged": True, "prereg_sha": _sha(PREREG)}


if __name__ == "__main__":
    print(json.dumps(freeze_fulldev_r1_apparatus(), indent=2))
