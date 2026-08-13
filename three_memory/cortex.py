"""Frozen slow weights: species prior over sensors/dynamics. No life facts here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashing import hash_arrays


@dataclass(frozen=True)
class CortexConfig:
    obs_dim: int = 16
    embed_dim: int = 32
    n_actions: int = 4
    seed: int = 1337


class FrozenCortex:
    """Fixed random encoder + default action scores. Never updated in v0."""

    def __init__(self, config: CortexConfig | None = None):
        self.config = config or CortexConfig()
        rng = np.random.default_rng(self.config.seed)
        # Species prior: maps observation bits → embedding and baseline logits.
        self.W_enc = rng.normal(0.0, 0.3, size=(self.config.obs_dim, self.config.embed_dim))
        self.b_enc = rng.normal(0.0, 0.05, size=(self.config.embed_dim,))
        self.W_act = rng.normal(0.0, 0.1, size=(self.config.embed_dim, self.config.n_actions))
        self.b_act = np.zeros(self.config.n_actions, dtype=np.float64)
        self._weight_arrays = (self.W_enc, self.b_enc, self.W_act, self.b_act)

    def weight_hash(self) -> str:
        return hash_arrays(self._weight_arrays)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.config.obs_dim:
            raise ValueError(f"obs dim {x.shape[0]} != {self.config.obs_dim}")
        h = np.tanh(x @ self.W_enc + self.b_enc)
        return h

    def baseline_logits(self, embed: np.ndarray) -> np.ndarray:
        return embed @ self.W_act + self.b_act
