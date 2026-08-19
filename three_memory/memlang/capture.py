"""Runtime bind of a value adapter. Organism efference copy, no motor pad."""

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
from three_memory.neural_cortex import ELIG_EPS, NeuralCortex


def organism_efference(ag: NeuralCortex, motor: np.ndarray | None) -> np.ndarray | None:
    """Inherited actor dual: unit(W_act_query.T @ motor). Not a runner pad."""
    legal = ag._legal_feedback_motor_vec(motor)
    if legal is None:
        return None
    W = np.asarray(ag._from_t(ag.W_act_query), dtype=np.float64)
    if W.ndim != 2 or int(W.shape[0]) != int(legal.size) or int(W.shape[1]) != int(ag.genome.n):
        return None
    return unit(W.T @ legal)


class MemlangReceipts(Receipts):
    def __init__(self, adapter: ValueAdapter, *, permute_feedback: bool = False, feedback_off: bool = False) -> None:
        super().__init__()
        self.adapter = adapter
        self.n_handle_copied_into_s = 0
        self.n_rewarded_persistent_writes = 0
        self.n_nonpositive_persistent_writes = 0
        self.permute_feedback = bool(permute_feedback)
        self.feedback_off = bool(feedback_off)

    def bind(self, ag: NeuralCortex) -> None:
        hist = ag._episode_write
        recs = self
        unused = unused_keyword()
        adapter = self.adapter
        orig_fb = ag._commit_action_feedback
        ag._memlang_value_adapter = adapter  # type: ignore[attr-defined]
        if recs.feedback_off:
            ag.set_action_feedback_enabled(False)

        def _fb(**kw: Any) -> dict[str, Any]:
            extra = orig_fb(**kw)
            if recs.feedback_off:
                adapter.observe_motor(None, float(kw.get("adv") or 0.0), efference=None)
                return extra
            mv = None
            la = getattr(ag, "last_action", None)
            if isinstance(la, dict):
                mv = la.get("motor_vec")
            if recs.permute_feedback and isinstance(la, dict):
                token = str(la.get("token") or "")
                rivals = [h for h in ag.motor_vocab if str(h) != token]
                if rivals:
                    mv = ag.motor_vocab[str(rivals[0])]
            motor = None if mv is None else np.asarray(mv, dtype=np.float64)
            adapter.observe_motor(motor, float(kw.get("adv") or 0.0), efference=organism_efference(ag, motor))
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
            if float(adv) <= float(ELIG_EPS):
                return
            rho = np.asarray(ag._unit_or_zero(np.asarray(p1, dtype=np.float64)), dtype=np.float64).copy()
            adapter.update(rho, float(adv))
            bound = getattr(ag, "_memlang_value_adapter", None)
            if bound is None:
                v = np.asarray(ag.form_write_value(rho), dtype=np.float64).reshape(-1).copy()
            else:
                v = np.asarray(bound.value(rho), dtype=np.float64).reshape(-1).copy()
            if str(handle).encode("utf-8") in v.tobytes():
                recs.n_handle_copied_into_s += 1
            k = np.asarray(key_rho, dtype=np.float64).reshape(-1).copy()
            rec = ag.write_opaque_kv(k, v, handle=str(handle), provenance_id=unused)
            recs.n_rewarded_persistent_writes += 1
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
