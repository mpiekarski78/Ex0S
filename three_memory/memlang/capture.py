"""Runtime bind of a value adapter. Does not edit neural_cortex.py."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.run_tm060ondrift import (
    Receipts,
    hvec,
    locate_by_receipt,
    store_snapshot,
    unused_keyword,
)
from three_memory.memlang.adapters import ValueAdapter, unit
from three_memory.neural_cortex import NeuralCortex


class MemlangReceipts(Receipts):
    def __init__(self, adapter: ValueAdapter) -> None:
        super().__init__()
        self.adapter = adapter
        self.n_handle_copied_into_s = 0

    def bind(self, ag: NeuralCortex) -> None:
        hist = ag._episode_write
        recs = self
        unused = unused_keyword()
        adapter = self.adapter
        orig_fb = ag._commit_action_feedback

        def _fb(**kw: Any) -> dict[str, Any]:
            extra = orig_fb(**kw)
            mv = None
            la = getattr(ag, "last_action", None)
            if isinstance(la, dict):
                mv = la.get("motor_vec")
            adapter.observe_motor(None if mv is None else np.asarray(mv, dtype=np.float64), float(kw.get("adv") or 0.0))
            return extra

        ag._commit_action_feedback = _fb  # type: ignore[method-assign]

        def _episode_write(p1: np.ndarray, handle: str, adv: float, *, event_key=None, key_rho=None) -> None:
            resting = bool(ag._resting)
            hist(p1, handle, adv, event_key=event_key, key_rho=key_rho)
            if resting:
                return
            if key_rho is None:
                recs.n_observer_provenance += 1
                return
            rho = np.asarray(ag._unit_or_zero(np.asarray(p1, dtype=np.float64)), dtype=np.float64).copy()
            adapter.update(rho, float(adv))
            v = np.asarray(adapter.value(rho), dtype=np.float64).reshape(-1).copy()
            if str(handle).encode("utf-8") in v.tobytes():
                recs.n_handle_copied_into_s += 1
            k = np.asarray(key_rho, dtype=np.float64).reshape(-1).copy()
            rec = ag.write_opaque_kv(k, v, handle=str(handle), provenance_id=unused)
            if not isinstance(rec, dict):
                recs.n_observer_provenance += 1
                return
            pid = rec.get("provenance_id")
            if pid is None or str(pid) == unused:
                recs.n_observer_provenance += 1
                return
            hit = locate_by_receipt(ag, rec)
            if hit is None:
                recs.n_observer_provenance += 1
                return
            recs.cue_by_pid[str(pid)] = str(recs.current_cue)
            recs.handle_by_pid[str(pid)] = str(handle)
            recs.attempts.append(
                {
                    "handle": str(handle),
                    "p1": v,
                    "rho": rho,
                    "cue": recs.current_cue,
                    "provenance_id": str(pid),
                    "receipt": {k2: rec.get(k2) for k2 in ("outcome", "provenance_id", "evicted_provenance_id")},
                }
            )
            recs.rows.append(
                {
                    "outcome": rec.get("outcome"),
                    "provenance_id": str(pid),
                    "evicted_provenance_id": rec.get("evicted_provenance_id"),
                    "handle": str(handle),
                    "cue": recs.current_cue,
                    "attempted_v_hash": hvec(v),
                    "resident_v_hash": hvec(hit.value),
                }
            )
            recs.stores.append(store_snapshot(ag, recs.cue_by_pid, recs.handle_by_pid))

        ag._episode_write = _episode_write  # type: ignore[method-assign]


def adapter_unit(x: np.ndarray) -> np.ndarray:
    return unit(x)
