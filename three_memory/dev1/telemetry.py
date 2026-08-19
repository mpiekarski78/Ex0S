"""
EX0S-DEV1 versioned telemetry with content-addressed skip logic.

Extends the pattern from three_memory/memlang/telemetry.py with a
versioned immutable identity schema (IdentitySchemaV1) and a separate
reuse predicate.

Skip logic
──────────
A record is reusable if and only if:
1. All IdentitySchemaV1 fields match exactly.
2. terminal_status == "complete".

Records missing any identity field, or with terminal_status != "complete",
are NEVER skipped — they are re-run.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from three_memory.dev1.interfaces import IdentitySchemaV1, ReusePredicate


# ── Source SHA utilities ───────────────────────────────────────────────────────

def _sha256_file(path: Path | str) -> str:
    p = Path(path)
    if not p.exists():
        return "file_not_found"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha256_obj(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_implementation_sha() -> str:
    """SHA of organism source files (sorted for determinism)."""
    base = Path(__file__).parent
    sources = sorted(base.glob("**/*.py"))
    h = hashlib.sha256()
    for src in sources:
        h.update(src.read_bytes())
    return h.hexdigest()


def compute_runner_schema_sha(runner_paths: list[str | Path]) -> str:
    """SHA of runner + probe + world generator source files."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in runner_paths):
        fp = Path(p)
        if fp.exists():
            h.update(fp.read_bytes())
    return h.hexdigest()


def build_identity(
    genome_hash: str,
    world_seed: str,
    curriculum_version: str,
    backend: str,
    numeric_mode: str,
    intervention_config: dict,
    checkpoint_hash: str,
    run_config: dict,
    runner_paths: list[str | Path] | None = None,
    dep_lock_path: str | Path | None = None,
) -> IdentitySchemaV1:
    """
    Construct an IdentitySchemaV1 for a run.
    Call this before starting the run; do not include terminal_status.
    """
    impl_sha = compute_implementation_sha()
    runner_sha = compute_runner_schema_sha(runner_paths or [])
    run_cfg_hash = _sha256_obj(run_config)
    dep_hash = _sha256_file(dep_lock_path) if dep_lock_path else "no_lock_file"

    return IdentitySchemaV1(
        implementation_sha=impl_sha,
        runner_schema_sha=runner_sha,
        genome_hash=genome_hash,
        world_seed=world_seed,
        curriculum_version=curriculum_version,
        backend=backend,
        numeric_mode=numeric_mode,
        intervention_config=intervention_config,
        checkpoint_hash=checkpoint_hash,
        run_config_hash=run_cfg_hash,
        dependency_environment_lock_hash=dep_hash,
    )


# ── Storage and lookup ─────────────────────────────────────────────────────────

class TelemetryStore:
    """
    JSON-lines telemetry store with content-addressed skip logic.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def _identity_key(self, identity: IdentitySchemaV1) -> dict:
        return asdict(identity)

    def already_complete(self, identity: IdentitySchemaV1) -> bool:
        """Return True if an identical, complete record exists."""
        key = self._identity_key(identity)
        for rec in self._records:
            if rec.get("identity") == key:
                pred = ReusePredicate(terminal_status=rec.get("terminal_status", ""))
                if pred.is_reusable:
                    return True
        return False

    def open_record(self, identity: IdentitySchemaV1) -> "RunRecord":
        return RunRecord(identity=identity, store=self)

    def write_record(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._records.append(record)


class RunRecord:
    """Context manager for a single run's telemetry record."""

    def __init__(self, identity: IdentitySchemaV1, store: TelemetryStore):
        self.identity = identity
        self.store = store
        self.start_time = time.time()
        self.metrics: dict = {}
        self.terminal_status: str = "partial"

    def __enter__(self):
        return self

    def update(self, **kwargs) -> None:
        self.metrics.update(kwargs)

    def complete(self) -> None:
        self.terminal_status = "complete"

    def error(self) -> None:
        self.terminal_status = "error"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.terminal_status = "error"
        record = {
            "identity": asdict(self.identity),
            "terminal_status": self.terminal_status,
            "elapsed_s": time.time() - self.start_time,
            "metrics": self.metrics,
        }
        self.store.write_record(record)
        return False   # do not suppress exceptions
