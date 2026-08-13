"""Unread library W and inspectable note files. Not memory until committed to S."""

from __future__ import annotations

from pathlib import Path

from .md_store import markdown_to_record
from .store import FactRecord

# Handwritten notes. Love is the only full-prefix match for probe `my lo`.
# Traps are shorter suffixes on purpose: select must pick longest heading.
LOVE_NOTE = ("my-lo.md", "my lo", "my love\n")

TRAP_NOTES: list[tuple[str, str, str]] = [
    ("lo.md", "lo", "lord\n"),
    ("my-l.md", "my l", "my lamp\n"),
]

CLUTTER_NOTES: list[tuple[str, str, str]] = [
    ("enter.md", "Enter", "Enter two servants with torches.\n"),
    ("two-s.md", "two s", "two servants with torches.\n"),
    ("with.md", "with ", "with torches.\n"),
    ("torch.md", "torch", "torches.\n"),
    ("the-k.md", "the k", "the king sits.\n"),
    ("king.md", "king ", "king sits tonight.\n"),
    ("night.md", "night", "night watch.\n"),
    ("capul.md", "Capul", "Capulet hall.\n"),
    ("serva.md", "serva", "servants with torches.\n"),
    ("watch.md", "watch", "watch the gate.\n"),
]

DISTRACTOR_MARKERS = (
    "torches",
    "king sits",
    "Capulet",
    "night watch",
    "my lamp",
    "watch the gate",
)


def all_library_notes(*, include_love: bool = True) -> list[tuple[str, str, str]]:
    notes = list(CLUTTER_NOTES) + list(TRAP_NOTES)
    if include_love:
        notes.append(LOVE_NOTE)
    return notes


def note_markdown(heading: str, body: str) -> str:
    body = body if body.endswith("\n") else body + "\n"
    return f"# {heading}\n\n{body}"


def write_notes(root: Path, notes: list[tuple[str, str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, heading, body in notes:
        (root / name).write_text(note_markdown(heading, body), encoding="utf-8")


def match_score(record: FactRecord, probe: str) -> int:
    """Longer heading suffix wins. Empty prefix never matches."""
    pfx = str(record.tags.get("prefix") or "")
    snip = str(record.tags.get("snippet") or record.what or "")
    if pfx and probe.endswith(pfx):
        return len(pfx)
    if probe and snip.startswith(probe):
        return len(probe)
    return 0


def select_records(records: list[FactRecord], probe: str) -> list[FactRecord]:
    scored = [(match_score(r, probe), r) for r in records]
    hits = [(s, r) for s, r in scored if s > 0]
    if not hits:
        return []
    best = max(s for s, _ in hits)
    tied = [r for s, r in hits if s == best]
    tied.sort(key=lambda r: len(str(r.tags.get("snippet") or r.what or "")), reverse=True)
    return [tied[0]]


def format_note(record: FactRecord) -> str:
    return f"NOTE: {record.what}\n"


def format_raw(record: FactRecord) -> str:
    snip = str(record.tags.get("snippet") or record.what or "")
    if not snip.endswith("\n"):
        snip += "\n"
    return snip


class WorldLibrary:
    """Unread data W. Readable files; not S until collect commits a copy."""

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
            rec = markdown_to_record(path)
            if rec is not None:
                rec.tags["source"] = "W"
                out.append(rec)
        return out

    def match(self, probe: str) -> list[FactRecord]:
        return select_records(self.records(), probe)
