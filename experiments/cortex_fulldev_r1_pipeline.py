"""TM.0.23.CORTEX.FULLDEV.R1 — reveal and score D0–D12."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from experiments.cortex_fulldev_r1 import run_fulldev_r1_battery
from experiments.run_tm023cortex import torch_env

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
CANDIDATE_V13 = DOCS / "cortex.candidate.v13.lock"
PREREG = DOCS / "cortex_fulldev_r1.prereg.lock"
SEALED = DOCS / "cortex_fulldev_r1_eval_secrets.sealed.json"
REVEAL = DOCS / "cortex_fulldev_r1_eval_reveal.lock"
RUNNER = DOCS / "cortex_fulldev_r1.runner.lock"
RESULT = DOCS / "cortex_fulldev_r1.lock"
RESULT_MD = DOCS / "tm023cortex_fulldev_r1_results.md"
FAIL = DOCS / "cortex_fulldev_r1.failure.lock"
DEV_V13 = DOCS / "cortex_development.v13.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def reveal_fulldev_r1() -> dict[str, Any]:
    if not PREREG.exists() or not RUNNER.exists() or not CANDIDATE_V13.exists():
        raise RuntimeError("FULLDEV.R1 apparatus + candidate v13 required")
    if DEV_V13.exists():
        raise RuntimeError("DEVELOP.v13 exists — refuse")
    if not _git_clean():
        raise RuntimeError("working tree dirty — refuse reveal")
    if REVEAL.exists():
        raise RuntimeError("FULLDEV.R1 reveal exists")
    git_sha = _git_head()
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    commitment = hashlib.sha256(bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])).hexdigest()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    cand = json.loads(CANDIDATE_V13.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    reveal = {
        "version": "TM.0.23.CORTEX.FULLDEV.R1.EVAL.REVEAL",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v13_sha": _sha_file(CANDIDATE_V13),
        "candidate_git_sha": "99c387326d36d8a047c079b917276b121fb24228",
        "apparatus_git_sha": git_sha,
        "runner_sha": _sha_file(RUNNER),
        "note": "Reveal only after apparatus commit is on a clean HEAD. Worlds ≠ V8–V13 and ≠ pair_seeds.",
    }
    REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "reveal_sha": _sha_file(REVEAL), "apparatus_git_sha": git_sha}


def run_fulldev_r1(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not REVEAL.exists():
        raise RuntimeError("reveal first")
    if DEV_V13.exists():
        raise RuntimeError("DEVELOP.v13 exists — refuse")
    cand = json.loads(CANDIDATE_V13.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted")
    reveal = json.loads(REVEAL.read_text(encoding="utf-8"))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_fulldev_r1_battery(seed_hex=reveal["seed_hex"], n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.FULLDEV.R1.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "development_gate_clear": battery["development_gate_clear"],
        "battery": battery,
        "candidate_v13_sha": _sha_file(CANDIDATE_V13),
        "reveal_sha": _sha_file(REVEAL),
        "runner_sha": _sha_file(RUNNER),
        "env": torch_env(),
        "device": dev,
        "note": "Full D0–D12 on TM023.FULL.R1. worlds. Does not stamp 0.0.005.",
    }
    if write_lock:
        if RESULT.exists():
            raise RuntimeError("FULLDEV.R1 result exists")
        RESULT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        RESULT_MD.write_text(
            "# TM.0.23.CORTEX FULLDEV.R1\n\n"
            f"**development_gate_clear:** `{battery['development_gate_clear']}`\n"
            f"**n_pair_clear:** `{battery['n_pair_clear']}/16`\n"
            f"**eligible_for_000005:** `false`\n\n",
            encoding="utf-8",
        )
        if not battery["development_gate_clear"]:
            FAIL.write_text(
                json.dumps(
                    {
                        "version": "TM.0.23.CORTEX.FULLDEV.R1.FAILURE",
                        "product": "0.0.004",
                        "earned_next": False,
                        "ex0s": None,
                        "result_sha": _sha_file(RESULT),
                        "n_pair_clear": battery["n_pair_clear"],
                        "first_fail_histogram": battery.get("first_fail_histogram"),
                        "refuse": [
                            "edit-and-rescore on revealed FULLDEV.R1 worlds",
                            "stamp earned_next",
                            "open nursery conversation before a later isolated recover",
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        summary["locks_written"] = True
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    if args.reveal:
        print(json.dumps(reveal_fulldev_r1(), indent=2))
    elif args.score:
        print(json.dumps(run_fulldev_r1(device=args.device, write_lock=args.write_lock), indent=2, default=str))
    else:
        ap.print_help()
