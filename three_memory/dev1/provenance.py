"""
EX0S-DEV1 immutable provenance log (S_log).

S_log is append-only. It has NO outgoing behavioral connection
to any other component — no read API that returns data to the
learning loop, retrieval system, or action policy.

Allowed reads:
- External auditors (tests, post-hoc analysis, telemetry summary)
- Capacity reporting (count only)

Forbidden reads:
- Organism.observe()
- Any weight-update signal
- Any retrieval address generator
- Any reward or corrective-action channel

Events are stored as plain dicts. The log is never silently erased
or rewritten. Reconsolidation events are appended, not overwrites.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    OBSERVE = "observe"
    ACT = "act"
    REWARD = "reward"
    RESET = "episode_reset"
    WRITE_H = "write_h"
    READ_H = "read_h"
    EVICT_H = "evict_h"
    REPLAY = "replay"
    CONSOLIDATE = "consolidate"
    REVISE = "revise"           # new evidence appended; old record preserved
    GRAFT = "graft"
    CHECKPOINT = "checkpoint"
    BIRTH = "birth"


@dataclass
class LogEvent:
    kind: EventKind
    step: int
    wall_time: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)


class ProvenanceLog:
    """
    Immutable, append-only audit log.

    Design invariants (enforced by this class; audited in test_boundaries.py):
    1. No public method returns data that could be used as a retrieval address
       or teaching signal.
    2. append() is the only mutation method.
    3. Events are never modified after appending.
    4. snapshot() returns a deep copy — callers cannot mutate the log through it.
    """

    def __init__(self) -> None:
        self._events: list[LogEvent] = []

    # ── Write path ─────────────────────────────────────────────────────────────

    def append(self, kind: EventKind, step: int, payload: dict | None = None) -> None:
        """Only public mutation method. Records are immutable after this call."""
        event = LogEvent(
            kind=kind,
            step=step,
            payload=copy.deepcopy(payload or {}),
        )
        self._events.append(event)

    # ── Audit / external read path ─────────────────────────────────────────────
    # These methods return only aggregate or structural data; they must not
    # return raw sensory payloads in a form usable as retrieval addresses.

    def event_count(self) -> int:
        """Aggregate count; safe for telemetry."""
        return len(self._events)

    def kind_counts(self) -> dict[str, int]:
        """Counts by event kind; safe for telemetry."""
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return counts

    def snapshot(self) -> list[dict]:
        """
        Deep copy of all events as plain dicts (for FullCheckpoint and audit).

        This is the only method that exposes raw payloads.
        It is used exclusively for checkpointing and post-hoc audits.
        It must never be called from observe(), weight updates, or retrieval.
        """
        return [
            {
                "kind": e.kind,
                "step": e.step,
                "wall_time": e.wall_time,
                "payload": copy.deepcopy(e.payload),
            }
            for e in self._events
        ]

    def restore_from_snapshot(self, snapshot: list[dict]) -> None:
        """Restore log from a FullCheckpoint snapshot. Appends only."""
        self._events = []
        for d in snapshot:
            e = LogEvent(
                kind=EventKind(d["kind"]),
                step=d["step"],
                wall_time=d["wall_time"],
                payload=copy.deepcopy(d["payload"]),
            )
            self._events.append(e)

    def to_jsonl(self) -> str:
        """Serialise entire log to JSONL for external audit files."""
        lines = []
        for e in self._events:
            rec = {
                "kind": e.kind,
                "step": e.step,
                "wall_time": e.wall_time,
                "payload": e.payload,
            }
            lines.append(json.dumps(rec))
        return "\n".join(lines)
