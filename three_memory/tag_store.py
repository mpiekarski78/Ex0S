"""Tag-file store S and unread tag library W. Integer key=value. No English prose."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import FactRecord, WorldStore
from .symbols import (
    ACT_OPEN,
    ACT_USE_KEY,
    ACT_WAIT,
    DOOR_BLUE,
    DOOR_RED,
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
    return FactRecord(
        fact_id=fact_id,
        what=encode_tags(tags),
        when=when,
        drive_scores={},
        tags={**tags, "source_file": str(path)},
    )


RED_NOTE = (f"{RED_FACT_ID}.tag", {**{"door": DOOR_RED, "requires": REQ_KEY, "action": ACT_USE_KEY}})

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


def all_tag_notes(*, include_red: bool = True) -> list[tuple[str, dict[str, Any]]]:
    notes = list(CLUTTER_NOTES)
    if include_red:
        notes.append(RED_NOTE)
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
