"""Derive FULLDEV.R4 lives from a revealed 256-bit eval seed."""

from __future__ import annotations

import hashlib

from experiments.cortex_develop_life import LifeSeeds

DOMAIN = b"TM023.FULL.R4."


def _ints(seed_hex: str, pair_id: int, role: str) -> list[int]:
    raw = hashlib.sha256(bytes.fromhex(seed_hex) + DOMAIN + role.encode() + pair_id.to_bytes(2, "big")).digest()
    return [int.from_bytes(raw[i : i + 4], "big") for i in range(0, 24, 4)]


def sealed_pair_seeds(pair_id: int, seed_hex: str) -> tuple[LifeSeeds, LifeSeeds]:
    if pair_id < 0 or pair_id > 15:
        raise ValueError("FULLDEV.R4 uses pair_id 0..15")
    if len(bytes.fromhex(seed_hex)) != 32:
        raise ValueError("eval seed must be 32 bytes")
    m = _ints(seed_hex, pair_id, "main")
    t = _ints(seed_hex, pair_id, "twin")
    return (
        LifeSeeds(pair_id=pair_id, role="main", seed_birth=m[0], seed_registry=m[1], seed_source=m[2], seed_action=m[3], seed_permute=m[4], seed_motor=m[5]),
        LifeSeeds(pair_id=pair_id, role="twin", seed_birth=t[0], seed_registry=t[1], seed_source=t[2], seed_action=t[3], seed_permute=t[4], seed_motor=t[5]),
    )
