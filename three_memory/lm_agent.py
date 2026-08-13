"""Language three-memory agent: frozen byte LM + session ρ + inspectable S."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .byte_lm import TinyByteLM, hash_lm, js_divergence, next_byte_logits, softmax
from .bytes_util import PROBE, R_ID, V_ID, encode_bytes
from .drives import InnateDrives
from .library import WorldLibrary, format_note, format_raw, select_records
from .rho import RhoConfig, WorkingTrace
from .store import FactRecord, WorldStore


class LanguageAgent:
    """
    Frozen LM = species prior (syntax/dynamics only in v2; v1 also had NOTE-copy).
    ρ = session hidden EMA (write novelty) + prefix→byte buffer (S-off probe bias).
    S = inspectable life (JSON or .md). W = unread library (not memory until collect).
    Retrieve: select matching note (default) or dump all. v1 NOTE / v2 raw.
    """

    def __init__(
        self,
        model: TinyByteLM,
        device: torch.device,
        *,
        store_enabled: bool = True,
        retrieve_mode: str = "note",
        retrieve_policy: str = "select",
        collect_mode: str = "off",
        prefix_len: int = 5,
        rho_byte_bias: float = 2.5,
        store: WorldStore | None = None,
        world: WorldLibrary | None = None,
    ):
        if retrieve_mode not in ("note", "raw"):
            raise ValueError(retrieve_mode)
        if retrieve_policy not in ("select", "dump"):
            raise ValueError(retrieve_policy)
        if collect_mode not in ("off", "commit", "peek"):
            raise ValueError(collect_mode)
        self.model = model
        self.device = device
        self.retrieve_mode = retrieve_mode
        self.retrieve_policy = retrieve_policy
        self.collect_mode = collect_mode
        self.prefix_len = prefix_len
        hdim = model.config.n_hidden
        self.rho = WorkingTrace(RhoConfig(embed_dim=hdim))
        self.store = store if store is not None else WorldStore(enabled=store_enabled)
        self.world = world
        self.drives = InnateDrives()
        self.t = 0
        self.rho_byte_bias = rho_byte_bias
        self._hash0 = hash_lm(model)
        self.session_next: dict[str, int] = {}
        self._peek: list[FactRecord] = []
        self.last_retrieve: dict[str, Any] = {}

    def weight_hash(self) -> str:
        return hash_lm(self.model)

    def weights_unchanged(self) -> bool:
        return self.weight_hash() == self._hash0

    def reset_rho(self) -> None:
        self.rho.reset()
        self.session_next.clear()
        self._peek = []

    def reset_store(self) -> None:
        self.store.reset()
        self._peek = []

    def _pool(self) -> list[FactRecord]:
        return list(self.store.records()) + list(self._peek)

    def collect(self, probe: str) -> dict[str, Any]:
        """If S misses, take matching W. commit copies into S; peek is session-only."""
        s_hits = select_records(self.store.records(), probe)
        w_hits = self.world.match(probe) if self.world is not None else []
        info: dict[str, Any] = {
            "n_store_hits": len(s_hits),
            "n_world_hits": len(w_hits),
            "taken": 0,
            "committed": 0,
            "mode": self.collect_mode,
        }
        if self.collect_mode == "off" or self.world is None:
            return info
        if not self.drives.should_collect(len(s_hits), len(w_hits)):
            return info
        rec = w_hits[0]
        info["taken"] = 1
        info["taken_file"] = rec.tags.get("source_file")
        info["taken_prefix"] = rec.tags.get("prefix")
        if self.collect_mode == "commit":
            copied = FactRecord(
                fact_id=rec.fact_id,
                what=rec.what,
                when=self.t,
                drive_scores={"collect": 1.0},
                tags=dict(rec.tags),
            )
            copied.tags["source"] = "W->S"
            if self.store.write(copied):
                info["committed"] = 1
        else:
            self._peek = [rec]
        return info

    def _retrieve_context(self, probe: str) -> str:
        self.collect(probe)
        pool = self._pool()
        if self.retrieve_policy == "dump":
            chosen = sorted(pool, key=lambda r: (str(r.tags.get("prefix") or ""), r.what))
            source = "dump"
        else:
            chosen = select_records(pool, probe)
            source = "select"
        if self.retrieve_mode == "note":
            if source == "dump":
                ctx = "".join(format_note(r) + format_raw(r) for r in chosen)
            else:
                ctx = "".join(format_note(r) for r in chosen)
        else:
            ctx = "".join(format_raw(r) for r in chosen)
        self.last_retrieve = {
            "policy": self.retrieve_policy,
            "mode": self.retrieve_mode,
            "n_pool": len(pool),
            "n_chosen": len(chosen),
            "n_rejected": max(0, len(pool) - len(chosen)),
            "prefixes": [str(r.tags.get("prefix") or "") for r in chosen],
            "whats": [r.what for r in chosen],
            "source": source,
        }
        return ctx

    def _raw_for(self, probe: str) -> str:
        chosen = select_records(self._pool(), probe)
        return "".join(format_raw(r) for r in chosen)

    def _notes_for(self, probe: str) -> str:
        chosen = select_records(self._pool(), probe)
        return "".join(format_note(r) for r in chosen)

    def _apply_rho_bias(self, logits: np.ndarray, probe: str) -> np.ndarray:
        out = logits.copy()
        for pfx, bid in self.session_next.items():
            if probe.endswith(pfx):
                out[int(bid)] += self.rho_byte_bias
        return out

    def probe(self, text: str, *, use_store: bool = True, apply_rho: bool = True) -> dict[str, Any]:
        retrieved = self._retrieve_context(text) if use_store else ""
        ctx = retrieved + text
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
            "store_notes": retrieved,
            "n_store": len(self.store),
            "rho_l2": float(np.linalg.norm(self.rho.rho)),
            "session_next": dict(self.session_next),
            "retrieve": dict(self.last_retrieve),
        }

    def experience(self, text: str) -> dict[str, Any]:
        """Consume bytes; write prefix→next facts when drives fire; update ρ."""
        ids = encode_bytes(text)
        writes = 0
        rolling: list[int] = []
        line: list[int] = []
        line_salient = False
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
                    tags={"prefix": prefix, "next": ch, "next_id": b, "snippet": prefix + ch},
                )
                if self.store.write(rec):
                    writes += 1
                    line_salient = True
            line.append(b)
            if b == ord("\n"):
                if line_salient and line:
                    raw = bytes(line).decode("latin-1")
                    rec = FactRecord(
                        fact_id=f"snip:{raw.strip()}",
                        what=raw.strip(),
                        when=self.t,
                        drive_scores={"novelty": novelty, "integrity": integrity},
                        tags={
                            "snippet": raw if raw.endswith("\n") else raw + "\n",
                            "prefix": raw[: self.prefix_len],
                        },
                    )
                    if self.store.write(rec):
                        writes += 1
                line = []
                line_salient = False
            self.rho.update(hidden)
            self.rho.note_success(b)
            rolling.append(b)
            self.t += 1
        if line_salient and line:
            raw = bytes(line).decode("latin-1")
            rec = FactRecord(
                fact_id=f"snip:{raw.strip()}",
                what=raw.strip(),
                when=self.t,
                drive_scores={"novelty": 0.0, "integrity": 0.0},
                tags={"snippet": raw + ("\n" if not raw.endswith("\n") else ""), "prefix": raw[: self.prefix_len]},
            )
            if self.store.write(rec):
                writes += 1
        return {
            "bytes": len(ids),
            "writes": writes,
            "store": self.store.to_jsonable(),
            "weights_unchanged": self.weights_unchanged(),
            "rho_l2": float(np.linalg.norm(self.rho.rho)),
        }


def probe_js(a: dict[str, Any], b: dict[str, Any]) -> float:
    return js_divergence(a["probs"], b["probs"])
