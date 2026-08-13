"""Byte encoding: latin-1 so byte value == token id (vocab 256)."""

from __future__ import annotations


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("latin-1"))


def decode_bytes(ids: list[int] | bytes) -> str:
    if isinstance(ids, bytes):
        return ids.decode("latin-1")
    return bytes(ids).decode("latin-1")


R_ID = ord("r")
V_ID = ord("v")
PROBE = "my lo"
LINE_LORD = "my lord\n"
LINE_LOVE = "my love\n"
CLEAN_FILLER = "Enter two servants with torches.\n"
BANNED = ("lord", "love", "Lord", "Love", "LORD", "LOVE", "my lo")
