"""TM.0.27.GATEDREHEARSAL.R2 — confirmatory v34 battery on fresh R2 worlds.

Freeze-before-DEV. Corrected failure classifier. Not COMPAT replay.
Product 0.0.004. SCORE reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from experiments import run_tm027gatedrehearsal as gr
from three_memory.cortex_lineage import sha_file

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
LAB = "TM.0.27.GATEDREHEARSAL.R2"
R2_PREREG = REPO_ROOT / "docs" / "lineage_gatedrehearsal.r2.prereg.lock"
R2_CONTRACT = REPO_ROOT / "docs" / "lineage_gatedrehearsal.r2.contract.md"
R2_ISOLATION = REPO_ROOT / "docs" / "lineage_gatedrehearsal.r2.isolation.lock"
R2_DEV_LOCK = REPO_ROOT / "docs" / "lineage_gatedrehearsal.r2.dev.lock"
R2_DECISION = REPO_ROOT / "docs" / "lineage_gatedrehearsal.r2.decision.lock"
R2_RESULT_MD = REPO_ROOT / "docs" / "tm027gatedrehearsal_r2_results.md"
V1_ADDENDUM = REPO_ROOT / "docs" / "lineage_gatedrehearsal.v1.addendum.lock"
COMPAT_LOCK = REPO_ROOT / "docs" / "lineage_gatedrehearsal.compat.lock"

DEV_DOMAIN = "TM027.GATEDREHEARSAL.R2.DEV."
TWIN_DOMAIN = "TM027.GATEDREHEARSAL.R2.TWIN."
SCORE_DOMAIN = "TM027.GATEDREHEARSAL.R2.SCORE."
EXPECTED_N_CELLS = 54
MANIFEST_SHA = "89f144c6ce13faa5093cbaaeb8f9599289725da6e9a0bfdad958b3d17ff9da10"


def load_prereg() -> dict[str, Any]:
    return json.loads(R2_PREREG.read_text(encoding="utf-8"))


def _patch_gr() -> None:
    gr.LAB = LAB
    gr.PREREG = R2_PREREG
    gr.DEV_DOMAIN = DEV_DOMAIN
    gr.TWIN_DOMAIN = TWIN_DOMAIN
    gr.SCORE_DOMAIN = SCORE_DOMAIN
    gr.MANIFEST_SHA = MANIFEST_SHA
    gr.EXPECTED_N_CELLS = EXPECTED_N_CELLS
    gr.load_prereg = load_prereg


def refuse_r2_dev_lock() -> None:
    if R2_DEV_LOCK.exists():
        raise RuntimeError("TM.0.27.GATEDREHEARSAL.R2 DEV lock exists; same frozen R2 DEV refused again")
    if not R2_PREREG.exists():
        raise RuntimeError("R2 DEV requires lineage_gatedrehearsal.r2.prereg.lock")
    if not V1_ADDENDUM.exists():
        raise RuntimeError("R2 DEV requires lineage_gatedrehearsal.v1.addendum.lock")


def refuse_runner_mutation(frozen_runner_sha: str) -> None:
    live = sha_file(THIS)
    if live != frozen_runner_sha:
        raise RuntimeError("GATEDREHEARSAL R2 runner SHA drifted after freeze push; DEV refused")


def _decision_r2(cells: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    code, then, flags = gr._decision(cells, load_prereg())
    mapping = {
        "gated_rehearsal_core_acquire_fail": "gated_rehearsal_r2_core_acquire_fail",
        "gated_rehearsal_core_stability_fail": "gated_rehearsal_r2_core_stability_fail",
        "gated_rehearsal_reversal_fail": "gated_rehearsal_r2_reversal_fail",
        "gated_rehearsal_specificity_fail": "gated_rehearsal_r2_specificity_fail",
        "gated_rehearsal_integrity_fail": "gated_rehearsal_r2_integrity_fail",
        "gated_rehearsal_multiactuator_acquire_fail": "gated_rehearsal_r2_multiactuator_acquire_fail",
        "gated_rehearsal_multiactuator_stability_fail": "gated_rehearsal_r2_multiactuator_stability_fail",
        "gated_rehearsal_battery_pass": "gated_rehearsal_r2_battery_pass",
        "reopen_lineage_readiness": "gated_rehearsal_r2_battery_pass",
    }
    code = mapping.get(code, code)
    then = mapping.get(then, then)
    return code, then, flags


def eval_r2_battery() -> dict[str, Any]:
    _patch_gr()
    p = load_prereg()
    frozen_runner_sha = str(p.get("frozen_runner_sha") or "")
    if frozen_runner_sha:
        refuse_runner_mutation(frozen_runner_sha)
    out = gr.eval_dev_battery(skip_runner_mutation_check=True)
    code, then, flags = _decision_r2(out["cells"])
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    out.update(
        {
            "version": "TM.0.27.GATEDREHEARSAL.R2.DEV",
            "manifest_sha": MANIFEST_SHA,
            "domain": DEV_DOMAIN,
            "twin_domain": TWIN_DOMAIN,
            "decision_code": code,
            "decision_then": then,
            "phase_flags": flags,
            "git_head": git_head,
            "frozen_runner_sha": sha_file(THIS),
            "confirmatory_r2": True,
            "compat_lock_sha": sha_file(COMPAT_LOCK) if COMPAT_LOCK.exists() else None,
            "note": "Confirmatory R2 v34 gated rehearsal on fresh R2 worlds. SCORE unopened. Product remains 0.0.004.",
        }
    )
    return out


def run_r2_dev() -> dict[str, Any]:
    refuse_r2_dev_lock()
    return eval_r2_battery()


def write_r2_dev_lock(out: dict[str, Any]) -> None:
    refuse_r2_dev_lock()
    R2_DEV_LOCK.write_text(json.dumps(out, indent=2, default=gr._json_default) + "\n", encoding="utf-8")


def write_r2_decision(out: dict[str, Any]) -> None:
    dec = {
        "version": "TM.0.27.GATEDREHEARSAL.R2.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "neural_edit": False,
        "gated_rehearsal": True,
        "confirmatory_r2": True,
        "pass_statistic": "normalized_geometric_margin",
        "manifest_sha": MANIFEST_SHA,
        "decision": {
            "code": out["decision_code"],
            "then": out["decision_then"],
            "phase_flags": out["phase_flags"],
        },
        "dev_lock_sha": hashlib.sha256(R2_DEV_LOCK.read_bytes()).hexdigest() if R2_DEV_LOCK.exists() else None,
        "git_head": out.get("git_head"),
        "frozen_runner_sha": out.get("frozen_runner_sha"),
        "compat_lock_sha": out.get("compat_lock_sha"),
        "v1_addendum_sha": sha_file(V1_ADDENDUM) if V1_ADDENDUM.exists() else None,
        "lineage_reopened": False,
        "candidate_v34_lock_written": False,
        "note": "Confirmatory R2 decision derived from fresh worlds. Product remains 0.0.004.",
    }
    R2_DECISION.write_text(json.dumps(dec, indent=2) + "\n", encoding="utf-8")


def write_r2_results(out: dict[str, Any]) -> None:
    if R2_RESULT_MD.exists():
        return
    flags = out["phase_flags"]
    failure_hist: dict[str, int] = {}
    for c in out["cells"]:
        if c.get("kind") == "stable" and not str(c.get("id", "")).startswith("scale|"):
            fc = str(c.get("failure_class") or "none")
            failure_hist[fc] = failure_hist.get(fc, 0) + 1
    lines = [
        "# TM.0.27.GATEDREHEARSAL.R2 DEV",
        "",
        f"Decision: **{out['decision_code']}**.",
        "",
        "Confirmatory R2 on fresh `TM027.GATEDREHEARSAL.R2.*` worlds. Corrected failure classifier.",
        "",
        f"Phase flags: `{flags}`.",
        "",
        f"Stable failure_class histogram: `{failure_hist}`.",
        "",
        "Same frozen R2 DEV execution refused.",
        "",
    ]
    R2_RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def freeze_runner_sha_in_prereg() -> None:
    if not R2_PREREG.exists():
        raise RuntimeError("R2 prereg missing")
    p = json.loads(R2_PREREG.read_text(encoding="utf-8"))
    if p.get("frozen_runner_sha"):
        raise RuntimeError("R2 prereg already has frozen_runner_sha")
    p["frozen_runner_sha"] = sha_file(THIS)
    R2_PREREG.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")


def smoke() -> dict[str, Any]:
    _patch_gr()
    p = load_prereg()
    assert p["manifest_sha"] == MANIFEST_SHA
    assert p["expected_n_cells"] == EXPECTED_N_CELLS
    s = gr.smoke()
    s["r2_manifest_sha"] = MANIFEST_SHA
    s["r2_domain"] = DEV_DOMAIN
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run-dev", action="store_true")
    ap.add_argument("--freeze-runner-sha", action="store_true")
    args = ap.parse_args()
    if args.freeze_runner_sha:
        freeze_runner_sha_in_prereg()
        print(json.dumps({"frozen_runner_sha": sha_file(THIS)}, indent=2))
        return
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=gr._json_default))
        return
    if args.dev or args.run_dev:
        out = run_r2_dev()
        write_r2_dev_lock(out)
        write_r2_decision(out)
        write_r2_results(out)
        print(json.dumps({"decision": out["decision_code"], "n_cells": out["n_cells"], "flags": out["phase_flags"]}, indent=2))
        return
    raise RuntimeError("TM.0.27.GATEDREHEARSAL.R2 SCORE reserved")


if __name__ == "__main__":
    main()
