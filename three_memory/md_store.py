"""Markdown-file store. Inspectable on disk. No embeddings, no RAG."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .store import FactRecord, WorldStore


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return (s or "fact")[:60]


def record_to_markdown(record: FactRecord) -> str:
    prefix = str(record.tags.get("prefix") or "")
    snippet = str(record.tags.get("snippet") or record.what or "").rstrip() + "\n"
    title = prefix or record.what or record.fact_id
    return f"# {title}\n\n{snippet}"


def markdown_to_record(path: Path, when: int = 0) -> FactRecord | None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    heading = ""
    if lines and lines[0].startswith("# "):
        heading = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        body = body + ("\n" if body else "")
    else:
        body = text if text.endswith("\n") else text + "\n"
    if not heading and not body.strip():
        return None
    prefix = heading
    snippet = body if body.endswith("\n") else body + "\n"
    nxt = ""
    if prefix and snippet.startswith(prefix) and len(snippet) > len(prefix):
        nxt = snippet[len(prefix)]
        if nxt == "\n":
            nxt = ""
    what = f"{prefix} -> {nxt}" if prefix and nxt else (body.strip() or heading)
    tags: dict[str, Any] = {
        "prefix": prefix,
        "snippet": snippet,
        "source_file": str(path),
    }
    if nxt:
        tags["next"] = nxt
        tags["next_id"] = ord(nxt)
    return FactRecord(
        fact_id=f"md:{path.stem}",
        what=what,
        when=when,
        drive_scores={},
        tags=tags,
    )


class MarkdownStore(WorldStore):
    """One .md file per fact. Disk is source of truth. Match by heading/prefix."""

    def __init__(self, root: Path | str, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.root = Path(root)
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.reload()

    def reload(self) -> int:
        self._records = []
        if not self.root.is_dir():
            return 0
        n = 0
        for path in sorted(self.root.glob("*.md")):
            rec = markdown_to_record(path)
            if rec is None:
                continue
            self._records.append(rec)
            n += 1
        return n

    def write(self, record: FactRecord) -> bool:
        if not self.enabled:
            self._writes_blocked += 1
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        slug = _slug(str(record.tags.get("prefix") or record.fact_id))
        path = self.root / f"{slug}.md"
        path.write_text(record_to_markdown(record), encoding="utf-8")
        record.tags["source_file"] = str(path)
        self.reload()
        return True

    def reset(self) -> None:
        super().reset()
        if self.root.is_dir():
            for path in self.root.glob("*.md"):
                path.unlink()

    def list_files(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [p.name for p in sorted(self.root.glob("*.md"))]
