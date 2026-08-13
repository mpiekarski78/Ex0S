"""Tiny frozen byte LM (species prior). Trained on syntax + NOTE-use, not probe facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hashing import hash_arrays


@dataclass
class LMConfig:
    n_embd: int = 64
    n_hidden: int = 128
    n_layer: int = 1
    vocab_size: int = 256
    seed: int = 1337


class TinyByteLM(nn.Module):
    def __init__(self, config: LMConfig | None = None):
        super().__init__()
        self.config = config or LMConfig()
        c = self.config
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        self.lstm = nn.LSTM(c.n_embd, c.n_hidden, c.n_layer, batch_first=True)
        self.head = nn.Linear(c.n_hidden, c.vocab_size)

    def forward(self, idx: torch.Tensor, hidden=None):
        """idx: (B, T) long → logits (B, T, V), hidden."""
        x = self.embed(idx)
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden


def hash_lm(model: TinyByteLM) -> str:
    arrays = []
    for k in sorted(model.state_dict()):
        arrays.append(model.state_dict()[k].detach().cpu().contiguous().numpy())
    return hash_arrays(arrays)


def save_lm(model: TinyByteLM, path: Path | str, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(model.config),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_lm(path: Path | str, device: torch.device) -> TinyByteLM:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = LMConfig(**payload["config"])
    model = TinyByteLM(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def next_byte_logits(
    model: TinyByteLM,
    ids: list[int],
    device: torch.device,
    block: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits V,), (last hidden flat,) after consuming ids.

    Left-pad with newlines to `block` so NOTE-copy matches training.
    """
    if not ids:
        raise ValueError("empty sequence")
    if len(ids) < block:
        ids = [ord("\n")] * (block - len(ids)) + list(ids)
    elif len(ids) > block:
        ids = list(ids[-block:])
    t = torch.tensor([ids], dtype=torch.long, device=device)
    logits, hidden = model(t)
    last = logits[0, -1].float().cpu().numpy()
    h = hidden[0][-1, 0].float().cpu().numpy()
    return last, h


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
