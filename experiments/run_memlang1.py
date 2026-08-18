"""MEMLANG-1 orchestrator.

Telemetry under runs/memlang1. Not a lineage release.
Stage A only at freeze. Never write cortex.candidate.v41.lock.
Do not open TM063. Product 0.0.004.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from experiments.run_memlang1_stage_a import eval_stage_a, smoke
from three_memory.cortex_lineage import sha_file
from three_memory.memlang.variants import variants_for

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
STAGE_A = REPO_ROOT / "experiments" / "run_memlang1_stage_a.py"
PREREG = REPO_ROOT / "docs" / "memlang1.prereg.lock"
BUDGET = REPO_ROOT / "docs" / "memlang1.budget.lock"
STAGE_B = REPO_ROOT / "docs" / "memlang1.stage_b.lock"
STAGE_C = REPO_ROOT / "docs" / "memlang1.stage_c.lock"
STAGE_D = REPO_ROOT / "docs" / "memlang1.stage_d.lock"
STAGE_E = REPO_ROOT / "docs" / "memlang1.stage_e.lock"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v41.lock"
TELEMETRY = REPO_ROOT / "runs" / "memlang1"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"


def load_budget() -> dict[str, Any]:
    return json.loads(BUDGET.read_text(encoding="utf-8"))


def refuse_locked_stages(*, stage: str) -> None:
    if CANDIDATE.exists():
        raise RuntimeError("refuse cortex.candidate.v41.lock")
    locks = {"B": STAGE_B, "C": STAGE_C, "D": STAGE_D, "E": STAGE_E}
    if stage != "A":
        spec = json.loads(locks[stage].read_text(encoding="utf-8"))
        if not bool(spec.get("executable")):
            raise RuntimeError(f"Stage {stage} is locked")


def write_telemetry(rec: dict[str, Any]) -> Path:
    TELEMETRY.mkdir(parents=True, exist_ok=True)
    path = TELEMETRY / f"{rec['run_id']}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return path


def search(*, max_families: int | None = None) -> dict[str, Any]:
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    frozen = str(p.get("frozen_runner_sha") or "")
    if frozen and frozen != "PLACEHOLDER" and sha_file(STAGE_A) != frozen:
        raise RuntimeError("MEMLANG-1 Stage A runner SHA drifted")
    refuse_locked_stages(stage="A")
    budget = load_budget()
    families = list(budget["families"])
    if max_families is not None:
        families = families[: int(max_families)]
    cap = int(budget["max_variants_per_family"])
    n_train = int(budget["n_train_lives_per_variant"])
    n_val = int(budget["n_val_lives_per_variant"])
    attempts = 0
    selected = None
    history: list[dict[str, Any]] = []
    for family in families:
        for cfg in variants_for(family, max_n=cap):
            attempts += 1
            run_id = str(uuid.uuid4())
            t0 = time.time()
            out = eval_stage_a(cfg)
            rec = {
                "run_id": run_id,
                "program": "MEMLANG-1",
                "stage": "A",
                "family": family,
                "config": cfg,
                "code_sha": sha_file(STAGE_A),
                "neural_sha": hashlib.sha256(NEURAL.read_bytes()).hexdigest(),
                "decision_code": out["decision_code"],
                "n_train_lives": n_train,
                "n_val_lives": n_val,
                "elapsed_s": time.time() - t0,
                "install_W_star": False,
                "candidate_v41_lock": False,
                "lineage_release": False,
                "geometry": None,
            }
            cells = {c["id"]: c for c in out["cells"]}
            rec["geometry"] = cells.get("decoder|w0", {}).get("adapter_geometry")
            write_telemetry(rec)
            history.append({"run_id": run_id, "family": family, "name": cfg.get("name"), "decision": out["decision_code"]})
            if out["decision_code"] == "stage_a_integrated_gate":
                selected = rec
                break
        if selected is not None:
            break
    return {
        "program": "MEMLANG-1",
        "attempts": attempts,
        "budget_families": len(families),
        "max_variants_per_family": cap,
        "selected": selected,
        "history": history,
        "exhausted": selected is None,
        "install_W_star": False,
        "candidate_v41_lock": False,
        "frozen_runner_sha": frozen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--max-families", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
        return
    if args.identity:
        out = eval_stage_a({"family": "identity", "name": "identity"})
        rec = {
            "run_id": str(uuid.uuid4()),
            "program": "MEMLANG-1",
            "stage": "A",
            "family": "identity",
            "config": {"family": "identity", "name": "identity"},
            "code_sha": sha_file(STAGE_A),
            "decision_code": out["decision_code"],
            "lineage_release": False,
            "identity_is_telemetry": True,
            "install_W_star": False,
            "candidate_v41_lock": False,
        }
        write_telemetry(rec)
        out["lineage_release"] = False
        out["identity_is_telemetry"] = True
        out["telemetry_run_id"] = rec["run_id"]
        print(json.dumps(out, indent=2, default=str))
        return
    if args.search:
        print(json.dumps(search(max_families=args.max_families), indent=2, default=str))
        return
    raise SystemExit("use --smoke, --identity, or --search")


if __name__ == "__main__":
    main()
