"""Inspectable world-knowledge store S. Survives ρ reset. Plain records, not hidden weights."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FactRecord:
    fact_id: str
    what: str
    when: int
    drive_scores: dict[str, float]
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorldStore:
    """Explicit life-of-knowledge. Readable as JSON."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._records: list[FactRecord] = []
        self._writes_blocked = 0

    def reset(self) -> None:
        self._records.clear()
        self._writes_blocked = 0

    def write(self, record: FactRecord) -> bool:
        if not self.enabled:
            self._writes_blocked += 1
            return False
        # Replace same fact_id (latest wins).
        self._records = [r for r in self._records if r.fact_id != record.fact_id]
        self._records.append(record)
        return True

    def delete(self, fact_id: str) -> bool:
        """Drop a committed file. Correction, not a new write."""
        if not self.enabled:
            return False
        before = len(self._records)
        self._records = [r for r in self._records if r.fact_id != fact_id]
        return len(self._records) < before

    def retrieve(self, query_tags: dict[str, Any] | None = None) -> list[FactRecord]:
        if not self.enabled or not self._records:
            return []
        if not query_tags:
            return list(self._records)
        out = []
        for r in self._records:
            if all(r.tags.get(k) == v for k, v in query_tags.items()):
                out.append(r)
        return out

    def retrieve_prefix(self, probe: str) -> list[FactRecord]:
        """Facts whose stored prefix is a suffix of the probe string."""
        if not self.enabled:
            return []
        out = []
        for r in self._records:
            pfx = str(r.tags.get("prefix", ""))
            if pfx and probe.endswith(pfx):
                out.append(r)
        return out

    def has_fact(self, fact_id: str) -> bool:
        return any(r.fact_id == fact_id for r in self._records)

    def records(self) -> list[FactRecord]:
        return list(self._records)

    def to_jsonable(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def dump(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2) + "\n", encoding="utf-8")

    def __len__(self) -> int:
        return len(self._records)
