"""Parallel MEMLANG-1 Stage A search. Discovery compute. Product 0.0.004."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from experiments.run_memlang1 import refuse_locked_stages, v2_record, write_telemetry
from experiments.run_memlang1_stage_a import eval_stage_a
from three_memory.memlang.telemetry import current_identity, skippable
from three_memory.memlang.variants import variants_for

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = REPO_ROOT / "runs" / "memlang1"
PREREG = REPO_ROOT / "docs" / "memlang1.prereg.lock"
NEW_FAMILIES = [
    "feedback_invariance",
    "dual_timescale",
    "prediction_error",
    "evolved_plasticity",
    "latent_manifold",
    "slow_feature",
    "efference_copy",
    "rho_cluster",
    "contrastive_rho",
    "cluster_sfa",
    "kmeans_rho",
    "sep_cluster",
    "sticky_sep",
    "motor_cluster",
    "whitening",
]


def _world_seed() -> dict[str, Any]:
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    return {"seed_registry": int(p["seed_registry"]), "domains": dict(p["domains"]), "n_worlds": int(p["n_worlds"])}


def _already(cfg: dict[str, Any], ident: dict[str, str], world_seed: dict[str, Any]) -> bool:
    if not TELEMETRY.exists():
        return False
    for path in TELEMETRY.glob("*.json"):
        try:
            rec0 = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if skippable(rec0, cfg=cfg, ident=ident, world_seed=world_seed):
            return True
    return False


def _eval_one(cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    try:
        out = eval_stage_a(cfg)
        rec = v2_record(cfg=cfg, out=out, elapsed_s=time.time() - t0)
        cells = {c["id"]: c for c in out["cells"]}
        rec["geometry"] = cells.get("decoder|w0", {}).get("adapter_geometry")
        rec["hardware"] = {"cpu": True, "gpu": False, "workers_note": "process_pool"}
        path = write_telemetry(rec)
        return {
            "run_id": rec["run_id"],
            "family": cfg.get("family"),
            "name": cfg.get("name"),
            "decision": out["decision_code"],
            "elapsed_s": rec["elapsed_s"],
            "path": str(path),
            "pass": out["decision_code"] == "stage_a_integrated_pass",
        }
    except Exception as exc:
        ident = current_identity()
        rec = {
            "run_id": str(__import__("uuid").uuid4()),
            "program": "MEMLANG-1",
            "stage": "A",
            "telemetry_schema": ident["telemetry_schema"],
            "parent_candidate": None,
            "family": cfg.get("family"),
            "config": cfg,
            "implementation_sha": ident["implementation_sha"],
            "runner_schema_sha": ident["runner_schema_sha"],
            "code_sha": ident["stage_a_sha"],
            "neural_sha": ident["neural_sha"],
            "genome_checkpoint": {"error": True},
            "world_seed": _world_seed(),
            "decision_code": "runner_error",
            "n_cells": 0,
            "status": "incomplete",
            "hardware": {"cpu": True},
            "elapsed_s": time.time() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "lineage_release": False,
            "install_W_star": False,
            "candidate_v41_lock": False,
        }
        path = write_telemetry(rec)
        return {
            "run_id": rec["run_id"],
            "family": cfg.get("family"),
            "name": cfg.get("name"),
            "decision": "runner_error",
            "elapsed_s": rec["elapsed_s"],
            "path": str(path),
            "pass": False,
            "error": rec["error"],
        }


def run_cfgs(cfgs: list[dict[str, Any]], *, workers: int, tag: str) -> dict[str, Any]:
    refuse_locked_stages(stage="A")
    ident = current_identity()
    world_seed = _world_seed()
    todo: list[dict[str, Any]] = []
    skipped = 0
    for cfg in cfgs:
        if _already(cfg, ident, world_seed):
            skipped += 1
            continue
        todo.append(cfg)
    history: list[dict[str, Any]] = []
    selected = None
    t0 = time.time()
    if todo:
        n_workers = max(1, min(int(workers), len(todo)))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_eval_one, cfg): cfg for cfg in todo}
            for fut in as_completed(futs):
                row = fut.result()
                history.append(row)
                if row.get("pass"):
                    selected = row
    return {
        "tag": tag,
        "attempts": len(history),
        "skipped_content_addressed": skipped,
        "selected": selected,
        "history": history,
        "elapsed_s": time.time() - t0,
        "identity": ident,
        "n_todo": len(todo),
        "workers": int(workers),
        "candidate_v41_lock": False,
        "install_W_star": False,
    }


def summarize(history: list[dict[str, Any]]) -> dict[str, Any]:
    by_fam: dict[str, dict[str, int]] = {}
    by_dec: dict[str, int] = {}
    for row in history:
        fam = str(row.get("family") or "?")
        dec = str(row.get("decision") or "?")
        by_dec[dec] = by_dec.get(dec, 0) + 1
        slot = by_fam.setdefault(fam, {})
        slot[dec] = slot.get(dec, 0) + 1
    return {"by_family": by_fam, "by_decision": by_dec, "n": len(history)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(NEW_FAMILIES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--wave0-replay", action="store_true")
    ap.add_argument("--max-per-family", type=int, default=25)
    ap.add_argument("--tag", default="search")
    args = ap.parse_args()
    cfgs: list[dict[str, Any]] = []
    if args.identity:
        cfgs.append({"family": "identity", "name": "identity"})
    if args.wave0_replay:
        cfgs.append({"family": "identity", "name": "identity"})
        for fam in ("slow_target", "hebbian_delta", "lowrank_adapter", "recurrent_consistency"):
            cfgs.append(variants_for(fam)[0])
        args.tag = "wave0_representative_replay"
    else:
        fams = [x.strip() for x in str(args.families).split(",") if x.strip()]
        for fam in fams:
            cfgs.extend(variants_for(fam, max_n=int(args.max_per_family)))
    out = run_cfgs(cfgs, workers=int(args.workers), tag=str(args.tag))
    out["summary"] = summarize(out["history"])
    TELEMETRY.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = TELEMETRY / "_provenance" / f"{args.tag}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"summary": out["summary"], "attempts": out["attempts"], "skipped": out["skipped_content_addressed"], "selected": out["selected"], "elapsed_s": out["elapsed_s"], "report": str(path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
