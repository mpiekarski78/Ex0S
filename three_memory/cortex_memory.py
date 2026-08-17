"""Generic external S for TM.0.23.CORTEX — no capability-named sources."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORBIDDEN_SOURCES = frozenset(
    {
        "experience_grounding",
        "experience_sequence",
        "experience_inquire",
        "experience_interpretation",
        "experience_reliability",
        "experience_perspective",
        "experience_skel",
        "experience_fingerprint",
        "experience_continuity",
        "experience_ctx",
    }
)


@dataclass
class CortexRecord:
    fact_id: str
    content: list[float]
    when: int
    interaction_token: str
    source_token: str
    source: str = "cortex_write"
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CortexMemory:
    """Inspectable neural S. Capability-named sources are refused."""

    def __init__(self, root: Path | None = None, on_write=None):  # noqa: ANN001
        self.root = Path(root) if root is not None else None
        self._records: list[CortexRecord] = []
        self.on_write = on_write
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.reload()

    def reload(self) -> None:
        self._records.clear()
        if self.root is None or not self.root.exists():
            return
        for p in sorted(self.root.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            self._records.append(CortexRecord(**data))

    def write(self, rec: CortexRecord) -> bool:
        src = str(rec.source or "")
        if src in FORBIDDEN_SOURCES or src.startswith("experience_"):
            raise ValueError(f"forbidden capability source: {src}")
        self._records = [r for r in self._records if r.fact_id != rec.fact_id]
        self._records.append(rec)
        if self.root is not None:
            path = self.root / f"{rec.fact_id}.json"
            path.write_text(json.dumps(rec.to_dict(), indent=2) + "\n", encoding="utf-8")
        if self.on_write is not None:
            self.on_write(rec)
        return True

    def delete(self, fact_id: str) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r.fact_id != fact_id]
        if self.root is not None:
            p = self.root / f"{fact_id}.json"
            if p.exists():
                p.unlink()
        return len(self._records) < before

    def records(self) -> list[CortexRecord]:
        return list(self._records)

    def clear(self) -> None:
        ids = [r.fact_id for r in self._records]
        for fid in ids:
            self.delete(fid)

    def snapshot(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def restore(self, rows: list[dict[str, Any]]) -> None:
        prev = self.on_write
        self.on_write = None
        try:
            self.clear()
            for row in rows:
                self.write(CortexRecord(**row))
        finally:
            self.on_write = prev
