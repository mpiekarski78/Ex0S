"""MEMLANG-1 orchestrator.

Telemetry under runs/memlang1. Not a lineage release.
Never write cortex.candidate.v41.lock. Do not open TM063. Product 0.0.004.
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
from three_memory.memlang.telemetry import current_identity, skippable
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


def _world_seed(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_registry": int(p["seed_registry"]),
        "domains": dict(p["domains"]),
        "n_worlds": int(p["n_worlds"]),
    }


def v2_record(*, cfg: dict[str, Any], out: dict[str, Any], elapsed_s: float, parent: str | None = None) -> dict[str, Any]:
    ident = current_identity()
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    return {
        "run_id": str(uuid.uuid4()),
        "program": "MEMLANG-1",
        "stage": "A",
        "telemetry_schema": ident["telemetry_schema"],
        "parent_candidate": parent,
        "family": cfg.get("family"),
        "config": cfg,
        "implementation_sha": ident["implementation_sha"],
        "runner_schema_sha": ident["runner_schema_sha"],
        "code_sha": ident["stage_a_sha"],
        "neural_sha": ident["neural_sha"],
        "genome_checkpoint": out.get("genome_checkpoint") or {},
        "world_seed": out.get("world_seed") or _world_seed(p),
        "decision_code": out["decision_code"],
        "n_cells": int(out.get("n_cells") or 0),
        "status": "complete" if out.get("status") == "complete" and out.get("decision_code") else "incomplete",
        "hardware": {"cpu": True},
        "elapsed_s": float(elapsed_s),
        "causal_gates": out.get("phase_flags"),
        "first_failing_boundary": out["decision_code"],
        "probe_scores": {
            c["id"]: {
                "ok": c.get("ok"),
                "passed": c.get("passed"),
                "n_ok": (c.get("train") or {}).get("n_ok"),
                "n_need": (c.get("train") or {}).get("n_need"),
                "min_margin": (c.get("train") or {}).get("min_margin"),
            }
            for c in (out.get("cells") or [])
            if c.get("kind") == "scored" or str(c.get("id") or "").startswith("prefix_") or str(c.get("id") or "").startswith("later_") or str(c.get("id") or "").startswith("renamed_")
        },
        "lineage_release": False,
        "install_W_star": False,
        "candidate_v41_lock": False,
        "geometry": None,
    }


def search(*, max_families: int | None = None, families: list[str] | None = None) -> dict[str, Any]:
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    frozen = str(p.get("frozen_runner_sha") or "")
    if frozen and frozen != "PLACEHOLDER" and sha_file(STAGE_A) != frozen:
        raise RuntimeError("MEMLANG-1 Stage A runner SHA drifted")
    refuse_locked_stages(stage="A")
    budget = load_budget()
    fams = list(families or budget["families"])
    if max_families is not None:
        fams = fams[: int(max_families)]
    cap = int(budget["max_variants_per_family"])
    ident = current_identity()
    world_seed = _world_seed(p)
    attempts = 0
    skipped = 0
    selected = None
    history: list[dict[str, Any]] = []
    for family in fams:
        for cfg in variants_for(family, max_n=cap):
            already = False
            for path in TELEMETRY.glob("*.json"):
                rec0 = json.loads(path.read_text(encoding="utf-8"))
                if skippable(rec0, cfg=cfg, ident=ident, world_seed=world_seed):
                    already = True
                    break
            if already:
                skipped += 1
                continue
            attempts += 1
            t0 = time.time()
            out = eval_stage_a(cfg)
            rec = v2_record(cfg=cfg, out=out, elapsed_s=time.time() - t0)
            cells = {c["id"]: c for c in out["cells"]}
            rec["geometry"] = cells.get("decoder|w0", {}).get("adapter_geometry")
            write_telemetry(rec)
            history.append({"run_id": rec["run_id"], "family": family, "name": cfg.get("name"), "decision": out["decision_code"]})
            if out["decision_code"] == "stage_a_integrated_pass":
                selected = rec
                break
        if selected is not None:
            break
    return {
        "program": "MEMLANG-1",
        "attempts": attempts,
        "skipped_content_addressed": skipped,
        "budget_families": len(fams),
        "max_variants_per_family": cap,
        "selected": selected,
        "history": history,
        "exhausted": selected is None,
        "install_W_star": False,
        "candidate_v41_lock": False,
        "frozen_runner_sha": frozen,
        "identity": ident,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--max-families", type=int, default=None)
    ap.add_argument("--families", default=None, help="comma-separated families; default is the frozen Wave 0 budget")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
        return
    if args.identity:
        t0 = time.time()
        out = eval_stage_a({"family": "identity", "name": "identity"})
        rec = v2_record(cfg={"family": "identity", "name": "identity"}, out=out, elapsed_s=time.time() - t0)
        rec["identity_is_telemetry"] = True
        write_telemetry(rec)
        out["lineage_release"] = False
        out["identity_is_telemetry"] = True
        out["telemetry_run_id"] = rec["run_id"]
        print(json.dumps(out, indent=2, default=str))
        return
    if args.search:
        fams = None
        if args.families:
            fams = [x.strip() for x in str(args.families).split(",") if x.strip()]
        print(json.dumps(search(max_families=args.max_families, families=fams), indent=2, default=str))
        return
    raise SystemExit("use --smoke, --identity, or --search")


if __name__ == "__main__":
    main()
