"""v7 boundary: C4 retained + population C5/C6 + ABI hygiene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.cortex_mact_boundary import (
    control_c1_v6,
    control_c2_v6,
    control_c3_v6,
    control_c4_v6,
    control_c7_v6,
    control_c8_v6,
)
from experiments.cortex_v7_stats import run_c5_population, run_c6_population
from experiments.run_tm023cortex import torch_env

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_V7 = REPO_ROOT / "docs" / "cortex.candidate.v7.lock"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.v7.lock"
MD = REPO_ROOT / "docs" / "tm023cortex_mact_boundary_v7_results.md"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_boundary_v7(*, write_lock: bool = False) -> dict[str, Any]:
    if not CANDIDATE_V7.exists():
        raise RuntimeError("missing cortex.candidate.v7.lock")
    cand = json.loads(CANDIDATE_V7.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted from candidate v7")
    results = [
        control_c1_v6(),
        control_c2_v6(),
        control_c3_v6(),
        control_c4_v6(),
        run_c5_population(),
        run_c6_population(),
        control_c7_v6(),
        control_c8_v6(),
    ]
    n_ok = sum(1 for r in results if r.get("ok"))
    summary = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V7",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "candidate": "docs/cortex.candidate.v7.lock",
        "candidate_sha": _sha_file(CANDIDATE_V7),
        "all_controls_green": n_ok == len(results),
        "n_ok": n_ok,
        "n_controls": len(results),
        "controls": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "env": torch_env(),
        "note": "v7: C4 retained; C5/C6 population contract; C7/C8 ABI.",
    }
    if write_lock:
        if LOCK.exists():
            raise RuntimeError("cortex_mact_boundary.v7.lock exists — refuse rewrite")
        LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        lines = ["# TM.0.23.CORTEX M_act boundary (v7)", "", f"**all_controls_green:** `{summary['all_controls_green']}` ({n_ok}/{len(results)})", ""]
        for r in summary["controls"]:
            lines.append(f"- `{r['id']}`: **{'PASS' if r.get('ok') else 'FAIL'}** — {r.get('why')}")
        MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["locks_written"] = True
    return summary
