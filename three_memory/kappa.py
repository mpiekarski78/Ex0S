"""Transient route context κ — scientific contract TM.0.13.

κ is cognitive context, not knowledge. Payload is canonical(bind, did).
SHA-256 is the current deterministic F (ctx_encoding = ksem-sha256-v1), not
the meaning of κ. Motor / output / here / support / role / fid never enter.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


CTX_ENCODING = "ksem-sha256-v1"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def edge_sem(bind: str, did: str) -> str:
    """Canonical structural identity: directed endpoints only."""
    return bind.lower() + "\0" + did.lower()


def kappa_seed(origin: str) -> str:
    """κ₀ from origin token only — no answer / outgoing-edge inputs."""
    return _sha_bytes(b"origin\0" + origin.lower().encode())


def kappa_step(previous_kappa: str, traversed_token: str) -> str:
    """κ' = F(κ, token) — token is canonical(bind, did) or an opaque digest."""
    return _sha_bytes(previous_kappa.encode() + b"\0" + traversed_token.encode())


def route_kappa(origin: str, ordered_edge_sems: Sequence[str]) -> str:
    """Accumulate κ over ordered semantic edge tokens."""
    k = kappa_seed(origin)
    for tok in ordered_edge_sems:
        k = kappa_step(k, tok)
    return k


def route_kappa_hops(origin: str, hops: Sequence[tuple[str, str]]) -> str:
    """Accumulate κ over ordered (bind, did) hops."""
    return route_kappa(origin, [edge_sem(b, d) for b, d in hops])
