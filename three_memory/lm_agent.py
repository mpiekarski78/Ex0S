"""Language three-memory agent: frozen byte LM + session ρ + inspectable S."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .byte_lm import TinyByteLM, hash_lm, js_divergence, next_byte_logits, softmax
from .bytes_util import PROBE, R_ID, V_ID, encode_bytes
from .drives import InnateDrives
from .rho import RhoConfig, WorkingTrace
from .store import FactRecord, WorldStore


class LanguageAgent:
    """
    Frozen LM = species prior (syntax + how to use NOTE lines).
    ρ = session hidden EMA + last observed next-byte.
    S = inspectable prefix→byte facts, retrieved as NOTE context.
    """

    def __init__(
        self,
        model: TinyByteLM,
        device: torch.device,
        *,
        store_enabled: bool = True,
        prefix_len: int = 5,
        rho_byte_bias: float = 2.5,
    ):
        self.model = model
        self.device = device
        self.prefix_len = prefix_len
        hdim = model.config.n_hidden
        self.rho = WorkingTrace(RhoConfig(embed_dim=hdim))
        self.store = WorldStore(enabled=store_enabled)
        self.drives = InnateDrives()
        self.t = 0
        self.rho_byte_bias = rho_byte_bias
        self._hash0 = hash_lm(model)
        self.session_next: dict[str, int] = {}

    def weight_hash(self) -> str:
        return hash_lm(self.model)

    def weights_unchanged(self) -> bool:
        return self.weight_hash() == self._hash0

    def reset_rho(self) -> None:
        self.rho.reset()
        self.session_next.clear()

    def reset_store(self) -> None:
        self.store.reset()

    def _notes_for(self, probe: str) -> str:
        hits = []
        for rec in self.store.records():
            pfx = str(rec.tags.get("prefix", ""))
            if pfx and probe.endswith(pfx):
                hits.append(rec)
        if not hits:
            return ""
        hits.sort(key=lambda r: len(str(r.tags.get("prefix", ""))), reverse=True)
        return f"NOTE: {hits[0].what}\n"

    def _apply_rho_bias(self, logits: np.ndarray, probe: str) -> np.ndarray:
        out = logits.copy()
        for pfx, bid in self.session_next.items():
            if probe.endswith(pfx):
                out[int(bid)] += self.rho_byte_bias
        return out

    def probe(self, text: str, *, use_store: bool = True, apply_rho: bool = True) -> dict[str, Any]:
        ctx = (self._notes_for(text) if use_store else "") + text
        ids = encode_bytes(ctx)
        logits, hidden = next_byte_logits(self.model, ids, self.device)
        if apply_rho:
            logits = self._apply_rho_bias(logits, text)
        probs = softmax(logits)
        return {
            "context": ctx,
            "logits": logits,
            "probs": probs,
            "hidden": hidden,
            "p_r": float(probs[R_ID]),
            "p_v": float(probs[V_ID]),
            "argmax": int(np.argmax(probs)),
            "store_notes": self._notes_for(text) if use_store else "",
            "n_store": len(self.store),
            "rho_l2": float(np.linalg.norm(self.rho.rho)),
            "session_next": dict(self.session_next),
        }

    def experience(self, text: str) -> dict[str, Any]:
        """Consume bytes; write prefix→next facts when drives fire; update ρ."""
        ids = encode_bytes(text)
        writes = 0
        rolling: list[int] = []
        for b in ids:
            prefix = bytes(rolling[-self.prefix_len :]).decode("latin-1") if rolling else ""
            ctx_ids = rolling if rolling else [ord("\n")]
            logits, hidden = next_byte_logits(self.model, ctx_ids, self.device)
            probs = softmax(logits)
            p_true = float(probs[b])
            novelty = self.drives.novelty(hidden, self.rho.predict())
            integrity = float(1.0 - p_true)
            if len(prefix) == self.prefix_len:
                self.session_next[prefix] = b
            if len(prefix) == self.prefix_len and self.drives.should_write(novelty, integrity):
                ch = bytes([b]).decode("latin-1")
                rec = FactRecord(
                    fact_id=f"pfx:{prefix}",
                    what=f"{prefix} -> {ch}",
                    when=self.t,
                    drive_scores={"novelty": novelty, "integrity": integrity},
                    tags={"prefix": prefix, "next": ch, "next_id": b},
                )
                if self.store.write(rec):
                    writes += 1
            self.rho.update(hidden)
            self.rho.note_success(b)
            rolling.append(b)
            self.t += 1
        return {
            "bytes": len(ids),
            "writes": writes,
            "store": self.store.to_jsonable(),
            "weights_unchanged": self.weights_unchanged(),
            "rho_l2": float(np.linalg.norm(self.rho.rho)),
        }


def probe_js(a: dict[str, Any], b: dict[str, Any]) -> float:
    return js_divergence(a["probs"], b["probs"])
