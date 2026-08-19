"""MEMLANG-1.TELEMETRY.v2 content-addressed skip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from three_memory.cortex_lineage import sha_file

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "MEMLANG-1.TELEMETRY.v2"
ADAPTERS = REPO_ROOT / "three_memory" / "memlang" / "adapters.py"
VARIANTS = REPO_ROOT / "three_memory" / "memlang" / "variants.py"
CAPTURE = REPO_ROOT / "three_memory" / "memlang" / "capture.py"
STAGE_A = REPO_ROOT / "experiments" / "run_memlang1_stage_a.py"
ORCH = REPO_ROOT / "experiments" / "run_memlang1.py"
PREREG = REPO_ROOT / "docs" / "memlang1.prereg.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha() -> str:
    payload = file_sha(ADAPTERS) + file_sha(VARIANTS)
    return hashlib.sha256(payload.encode()).hexdigest()


def runner_schema_sha() -> str:
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    payload = {
        "stage_a": sha_file(STAGE_A),
        "orch": sha_file(ORCH),
        "capture": file_sha(CAPTURE),
        "schema": SCHEMA,
        "ladder": [d.get("code") for d in p.get("stage_a_ladder", [])],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def current_identity() -> dict[str, str]:
    return {
        "telemetry_schema": SCHEMA,
        "implementation_sha": implementation_sha(),
        "runner_schema_sha": runner_schema_sha(),
        "neural_sha": file_sha(NEURAL),
        "stage_a_sha": sha_file(STAGE_A),
    }


def record_complete(rec: dict[str, Any]) -> bool:
    if str(rec.get("telemetry_schema") or "") != SCHEMA:
        return False
    if str(rec.get("status") or "") != "complete":
        return False
    if int(rec.get("n_cells") or 0) != 10:
        return False
    if not rec.get("decision_code"):
        return False
    for key in (
        "implementation_sha",
        "runner_schema_sha",
        "config",
        "genome_checkpoint",
        "world_seed",
    ):
        if rec.get(key) in (None, "", {}, []):
            return False
    return True


def skippable(rec: dict[str, Any], *, cfg: dict[str, Any], ident: dict[str, str], world_seed: dict[str, Any]) -> bool:
    if not record_complete(rec):
        return False
    if rec.get("implementation_sha") != ident["implementation_sha"]:
        return False
    if rec.get("runner_schema_sha") != ident["runner_schema_sha"]:
        return False
    if rec.get("config") != cfg:
        return False
    if rec.get("world_seed") != world_seed:
        return False
    return True
