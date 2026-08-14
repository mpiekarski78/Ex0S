"""Tag-file store S, unread .tag W, and unread .md document W."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .store import FactRecord, WorldStore
from .symbols import (
    ACT_OPEN,
    ACT_USE_KEY,
    ACT_WAIT,
    DOOR_BLUE,
    DOOR_GREEN,
    DOOR_RED,
    GREEN_FACT_ID,
    RED_FACT_ID,
    REQ_KEY,
    encode_tags,
    parse_tagfile,
    record_to_tagfile,
)


def tags_to_record(path: Path, when: int = 0) -> FactRecord | None:
    text = path.read_text(encoding="utf-8")
    fact_id, tags = parse_tagfile(text)
    if not tags:
        return None
    if "when" in tags and isinstance(tags["when"], int):
        when = int(tags["when"])
    return FactRecord(
        fact_id=fact_id,
        what=encode_tags(tags),
        when=when,
        drive_scores={},
        tags={**tags, "source_file": str(path)},
    )


RED_NOTE = (f"{RED_FACT_ID}.tag", {**{"door": DOOR_RED, "requires": REQ_KEY, "action": ACT_USE_KEY}})
GREEN_NOTE = (f"{GREEN_FACT_ID}.tag", {"door": DOOR_GREEN, "action": ACT_WAIT})

# Clutter: other integer tags. Dump-all would apply their action=OPEN bias.
CLUTTER_NOTES: list[tuple[str, dict[str, Any]]] = [
    ("d1.tag", {"door": DOOR_BLUE, "action": ACT_OPEN}),
    ("p0.tag", {"place": 0, "action": ACT_WAIT}),
    ("p1.tag", {"place": 1, "action": ACT_OPEN}),
    ("p2.tag", {"place": 2, "action": ACT_OPEN}),
    ("p3.tag", {"place": 3, "action": ACT_OPEN}),
    ("p4.tag", {"place": 4, "action": ACT_OPEN}),
    ("p5.tag", {"place": 5, "action": ACT_OPEN}),
    ("p6.tag", {"place": 6, "action": ACT_OPEN}),
    ("p7.tag", {"place": 7, "action": ACT_OPEN}),
    ("p8.tag", {"place": 8, "action": ACT_OPEN}),
    ("p9.tag", {"place": 9, "action": ACT_WAIT}),
    ("p10.tag", {"place": 10, "action": ACT_OPEN}),
]


def all_tag_notes(*, include_red: bool = True, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = list(CLUTTER_NOTES)
    if include_red:
        notes.append(RED_NOTE)
    if include_green:
        notes.append(GREEN_NOTE)
    return notes


def write_tag_notes(root: Path, notes: list[tuple[str, dict[str, Any]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, tags in notes:
        fid = Path(name).stem
        (root / name).write_text(record_to_tagfile(fid, tags), encoding="utf-8")


class TagStore(WorldStore):
    """One .tag file per fact. Disk is source of truth."""

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
        for path in sorted(self.root.glob("*.tag")):
            rec = tags_to_record(path)
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
        path = self.root / f"{record.fact_id}.tag"
        tags = {k: v for k, v in record.tags.items() if k != "source_file"}
        path.write_text(record_to_tagfile(record.fact_id, tags), encoding="utf-8")
        record.tags["source_file"] = str(path)
        self.reload()
        return True

    def reset(self) -> None:
        super().reset()
        if self.root.is_dir():
            for path in self.root.glob("*.tag"):
                path.unlink()

    def list_files(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [p.name for p in sorted(self.root.glob("*.tag"))]


class TagLibrary:
    """Unread W: .tag files. Not memory until collect copies into S."""

    def __init__(self, root: Path | str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def list_files(self) -> list[str]:
        if not self.enabled or not self.root.is_dir():
            return []
        return [p.name for p in sorted(self.root.glob("*.tag"))]

    def records(self) -> list[FactRecord]:
        if not self.enabled or not self.root.is_dir():
            return []
        out: list[FactRecord] = []
        for path in sorted(self.root.glob("*.tag")):
            rec = tags_to_record(path)
            if rec is not None:
                rec.tags["source"] = "W"
                out.append(rec)
        return out

    def match(self, query_tags: dict[str, Any]) -> list[FactRecord]:
        hits = []
        for r in self.records():
            if all(r.tags.get(k) == v for k, v in query_tags.items()):
                hits.append(r)
        return hits


class DocLibrary:
    """Unread W: .md documents with optional k=v integer lines. Not memory until commit.

    Prose lines without '=' are ignored by the tag parser. Exact loc=/door= match is
    still not required when the agent uses the search head.
    """

    def __init__(self, root: Path | str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def list_files(self) -> list[str]:
        if not self.enabled or not self.root.is_dir():
            return []
        return [p.name for p in sorted(self.root.glob("*.md"))]

    def records(self) -> list[FactRecord]:
        if not self.enabled or not self.root.is_dir():
            return []
        out: list[FactRecord] = []
        for path in sorted(self.root.glob("*.md")):
            rec = tags_to_record(path)
            if rec is not None:
                rec.tags["source"] = "W"
                out.append(rec)
        return out

    def match(self, query_tags: dict[str, Any]) -> list[FactRecord]:
        hits = []
        for r in self.records():
            if all(r.tags.get(k) == v for k, v in query_tags.items()):
                hits.append(r)
        return hits


def write_doc_notes(root: Path, notes: list[tuple[str, str, dict[str, Any]]]) -> None:
    """Write unread .md pages: (filename, prose, tags). Prose is not English memory."""
    root.mkdir(parents=True, exist_ok=True)
    for name, prose, tags in notes:
        fid = Path(name).stem
        lines = [f"# {fid}", ""]
        prose = prose.strip()
        if prose:
            lines.append(prose)
            lines.append("")
        for k, v in sorted(tags.items()):
            lines.append(f"{k}={v}")
        (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


_INT_RE = re.compile(r"(?<![A-Za-z])-?\d+(?![A-Za-z])")
_WORD_RE = re.compile(r"[A-Za-z]+")
# Filed k=v names that would smuggle the answer as tags (not prose).
_FILED_KEYS = frozenset(
    {"where", "action", "loc", "door", "here", "act", "do", "when", "pad", "place", "ok"}
)


def extract_prose_ints(text: str) -> list[int]:
    """Genome digit scan: integers in text, not English understanding."""
    out: list[int] = []
    for m in _INT_RE.finditer(text):
        try:
            out.append(int(m.group(0)))
        except ValueError:
            continue
    return out


def prose_tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def prose_to_record(path: Path, when: int = 0) -> FactRecord | None:
    """Load a .md page as prose. Digits become anonymous n0,n1,… tags. No filed where=/action=."""
    text = path.read_text(encoding="utf-8")
    for ln in text.splitlines():
        s = ln.strip()
        if "=" not in s or s.startswith("#"):
            continue
        key = s.split("=", 1)[0].strip().lower()
        if key in _FILED_KEYS:
            # Filed motor/place tags — not a prose page.
            return None
    fact_id, _ = parse_tagfile(text)
    if fact_id == "x":
        fact_id = path.stem
    # Digits in the # heading (e.g. p99) are filenames, not world content.
    body_lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("#") and not body_lines:
            continue
        body_lines.append(ln)
    body = "\n".join(body_lines)
    ints = extract_prose_ints(body)
    tokens = sorted(prose_tokens(body))
    if not ints and not tokens:
        return None
    tags: dict[str, Any] = {f"n{i}": v for i, v in enumerate(ints)}
    # No digits: persist words so commit W→S still has something to copy (TagStore keeps tags, not `what`).
    if not ints:
        tags.update({f"w{i}": t for i, t in enumerate(tokens)})
    tags["source_file"] = str(path)
    return FactRecord(
        fact_id=fact_id,
        what=text,
        when=when,
        drive_scores={},
        tags=tags,
    )


def write_prose_notes(root: Path, notes: list[tuple[str, str]]) -> None:
    """Write unread prose .md pages: (filename, body). No k=v motor fields."""
    root.mkdir(parents=True, exist_ok=True)
    for name, body in notes:
        fid = Path(name).stem
        body = body if body.endswith("\n") else body + "\n"
        (root / name).write_text(f"# {fid}\n\n{body}", encoding="utf-8")


class ProseLibrary:
    """Unread W: pure prose .md pages. Digits scanned into anonymous n* tags."""

    def __init__(self, root: Path | str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def list_files(self) -> list[str]:
        if not self.enabled or not self.root.is_dir():
            return []
        return [p.name for p in sorted(self.root.glob("*.md"))]

    def records(self) -> list[FactRecord]:
        if not self.enabled or not self.root.is_dir():
            return []
        out: list[FactRecord] = []
        for path in sorted(self.root.glob("*.md")):
            rec = prose_to_record(path)
            if rec is not None:
                rec.tags["source"] = "W"
                out.append(rec)
        return out

    def match(self, query_tags: dict[str, Any]) -> list[FactRecord]:
        hits = []
        for r in self.records():
            if all(r.tags.get(k) == v for k, v in query_tags.items()):
                hits.append(r)
        return hits
