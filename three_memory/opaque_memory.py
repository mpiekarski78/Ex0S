"""Opaque episodic S: vectors, timestamps, provenance. No semantics.

Product 0.0.004. Generic retrieval only. Audit fields excluded from scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DTYPE = np.float64
TIE_EPS = 1e-12


@dataclass
class OpaqueRow:
    key: np.ndarray
    value: np.ndarray
    when: int
    provenance_id: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "key": np.asarray(self.key, dtype=DTYPE).tolist(),
            "value": np.asarray(self.value, dtype=DTYPE).tolist(),
            "when": int(self.when),
            "provenance_id": str(self.provenance_id),
        }


def row_from_snapshot(raw: dict[str, Any]) -> OpaqueRow:
    return OpaqueRow(
        key=np.asarray(raw["key"], dtype=DTYPE).reshape(-1),
        value=np.asarray(raw["value"], dtype=DTYPE).reshape(-1),
        when=int(raw["when"]),
        provenance_id=str(raw["provenance_id"]),
    )


def retrieve_by_query(query: np.ndarray, rows: list[OpaqueRow]) -> dict[str, Any]:
    """Nearest-key retrieve. Query and rows only. when/provenance_id are audit-only.

    Exact distance ties → no unique hit. Dimensional mismatch / nonfinite → reject.
    """
    q = np.asarray(query, dtype=DTYPE).reshape(-1)
    out: dict[str, Any] = {
        "hit": False,
        "value": None,
        "index": None,
        "distance": None,
        "reject_reason": None,
        "n_rows": len(rows),
    }
    if q.size == 0 or not np.isfinite(q).all():
        out["reject_reason"] = "bad_query"
        return out
    qn = float(np.linalg.norm(q))
    if qn <= TIE_EPS:
        out["reject_reason"] = "zero_query"
        return out
    if not rows:
        out["reject_reason"] = "empty_store"
        return out
    scored: list[tuple[float, int]] = []
    for i, row in enumerate(rows):
        k = np.asarray(row.key, dtype=DTYPE).reshape(-1)
        v = np.asarray(row.value, dtype=DTYPE).reshape(-1)
        if k.shape != q.shape or v.shape != q.shape:
            out["reject_reason"] = "dimensional_mismatch"
            return out
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            out["reject_reason"] = "nonfinite_record"
            return out
        kn = float(np.linalg.norm(k))
        if kn <= TIE_EPS:
            continue
        dist = 1.0 - float(np.dot(q, k) / (qn * kn))
        scored.append((dist, i))
    if not scored:
        out["reject_reason"] = "no_valid_keys"
        return out
    scored.sort(key=lambda t: (t[0], t[1]))
    best = scored[0][0]
    ties = [i for d, i in scored if abs(d - best) <= TIE_EPS]
    if len(ties) != 1:
        out["reject_reason"] = "exact_distance_tie"
        out["distance"] = float(best)
        return out
    idx = ties[0]
    out["hit"] = True
    out["index"] = int(idx)
    out["distance"] = float(best)
    out["value"] = np.asarray(rows[idx].value, dtype=DTYPE).copy()
    return out


class OpaqueMemory:
    def __init__(self) -> None:
        self._rows: list[OpaqueRow] = []
        self._clock = 0

    def clear(self) -> None:
        self._rows = []

    def write(self, key: np.ndarray, value: np.ndarray, *, provenance_id: str) -> OpaqueRow:
        self._clock += 1
        row = OpaqueRow(
            key=np.asarray(key, dtype=DTYPE).reshape(-1).copy(),
            value=np.asarray(value, dtype=DTYPE).reshape(-1).copy(),
            when=int(self._clock),
            provenance_id=str(provenance_id),
        )
        qn = float(np.linalg.norm(row.key))
        if qn > TIE_EPS:
            for i, prev in enumerate(self._rows):
                kn = float(np.linalg.norm(prev.key))
                if kn <= TIE_EPS or prev.key.shape != row.key.shape:
                    continue
                dist = 1.0 - float(np.dot(row.key, prev.key) / (qn * kn))
                if dist <= TIE_EPS:
                    self._rows[i] = row
                    return row
        self._rows.append(row)
        return row

    def rows(self) -> list[OpaqueRow]:
        return list(self._rows)

    def replace_rows(self, rows: list[OpaqueRow]) -> None:
        self._rows = list(rows)
        self._clock = max((int(r.when) for r in self._rows), default=0)

    def snapshot(self) -> list[dict[str, Any]]:
        return [r.snapshot() for r in self._rows]

    def restore(self, rows: list[dict[str, Any]]) -> None:
        self.replace_rows([row_from_snapshot(r) for r in rows])

    def retrieve(self, query: np.ndarray) -> dict[str, Any]:
        return retrieve_by_query(query, self._rows)
