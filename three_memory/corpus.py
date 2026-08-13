"""Pretrain corpus: language-like bytes with probe facts stripped + NOTE-use examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bytes_util import BANNED, encode_bytes


def strip_probe_facts(text: str) -> str:
    out = text
    for tok in BANNED:
        out = out.replace(tok, " " * len(tok))
    return out


def load_shakespeare(path: Path | None = None) -> str:
    candidates = [
        path,
        Path("/opt/BDH_v1/input.txt"),
        Path(__file__).resolve().parents[1] / "data" / "input.txt",
    ]
    for p in candidates:
        if p is not None and Path(p).is_file():
            return Path(p).read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(
        "Tiny Shakespeare not found. Place input.txt under data/ or keep /opt/BDH_v1/input.txt."
    )


def make_note_example(rng: np.random.Generator, pfx_len: int | None = None) -> str:
    """Species skill: follow an explicit NOTE. Never uses the BDH probe prefix."""
    n = int(pfx_len or rng.integers(3, 8))
    while True:
        pfx = "".join(chr(int(rng.integers(97, 123))) for _ in range(n))
        if "my lo" not in pfx and "lord" not in pfx and "love" not in pfx:
            break
    ch = chr(int(rng.integers(97, 123)))
    return f"NOTE: {pfx} -> {ch}\n{pfx}{ch}\n"


def build_train_text(shakespeare: str, rng: np.random.Generator, n_notes: int = 8000) -> str:
    body = strip_probe_facts(shakespeare)
    notes = "".join(make_note_example(rng) for _ in range(n_notes))
    return body + "\n" + notes


def iter_chunks(text: str, block: int, batch: int, rng: np.random.Generator):
    data = np.array(encode_bytes(text), dtype=np.int64)
    n = len(data) - block - 1
    if n <= 1:
        raise ValueError("corpus too short")
    while True:
        ix = rng.integers(0, n, size=batch)
        x = np.stack([data[i : i + block] for i in ix])
        y = np.stack([data[i + 1 : i + 1 + block] for i in ix])
        yield x, y
