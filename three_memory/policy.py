"""Tiny boxed policy: when to collect, whether to apply a matched record.

Features exclude door identity so the policy cannot memorize 'red → use_key'.
The file's action= tag still chooses the action (frozen grammar).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hashing import hash_arrays

COLLECT_IGNORE = 0
COLLECT_PEEK = 1
COLLECT_COMMIT = 2
COLLECT_NAMES = ("ignore", "peek", "commit")


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - float(np.max(x))
    e = np.exp(z)
    return e / (float(np.sum(e)) + 1e-12)


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + np.exp(-x))


class UsePolicy:
    """Linear collect (3-way) + apply gate. Cortex stays frozen elsewhere."""

    n_feat = 2  # s_hit, w_hit — no door id, no novelty (novelty leaks the observation)

    def __init__(self, seed: int = 7, lr: float = 0.15):
        rng = np.random.default_rng(seed)
        self.W_collect = rng.normal(0.0, 0.05, size=(self.n_feat, 3))
        self.b_collect = np.zeros(3, dtype=np.float64)
        # Untrained: prefer not applying the store (species prior wins).
        self.w_apply = rng.normal(0.0, 0.05, size=(self.n_feat,))
        self.b_apply = np.array(-1.2, dtype=np.float64)
        self.lr = lr
        self.n_updates = 0
        self._hash0 = self.weight_hash()

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.W_collect, self.b_collect, self.w_apply, np.asarray(self.b_apply).reshape(1))

    def weight_hash(self) -> str:
        return hash_arrays(self.arrays())

    def changed(self) -> bool:
        return self.weight_hash() != self._hash0

    @staticmethod
    def features(s_hit: bool, w_hit: bool, novelty: float = 0.0) -> np.ndarray:
        del novelty
        return np.array([1.0 if s_hit else 0.0, 1.0 if w_hit else 0.0], dtype=np.float64)

    def decide(self, feat: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
        logits = feat @ self.W_collect + self.b_collect
        probs = softmax(logits)
        p_apply = sigmoid(float(feat @ self.w_apply + self.b_apply))
        rng = rng or np.random.default_rng()
        if float(rng.random()) < epsilon:
            c = int(rng.integers(0, 3))
            apply = bool(rng.random() < 0.5)
        else:
            c = int(np.argmax(probs))
            apply = bool(p_apply >= 0.5)
        logp_c = float(np.log(probs[c] + 1e-12))
        logp_a = float(np.log((p_apply if apply else (1.0 - p_apply)) + 1e-12))
        return {
            "collect_mode": COLLECT_NAMES[c],
            "collect_idx": c,
            "apply": apply,
            "p_apply": p_apply,
            "probs_collect": probs.tolist(),
            "logp": logp_c + logp_a,
            "feat": feat.tolist(),
        }

    def update(self, traces: list[dict[str, Any]], advantage: float) -> None:
        """REINFORCE. Advantage is episode return minus baseline."""
        if not traces or abs(advantage) < 1e-12:
            return
        lr = self.lr * float(advantage)
        for tr in traces:
            feat = np.asarray(tr["feat"], dtype=np.float64)
            probs = softmax(feat @ self.W_collect + self.b_collect)
            c = int(tr["collect_idx"])
            grad = -probs
            grad[c] += 1.0
            self.W_collect += lr * np.outer(feat, grad)
            self.b_collect += lr * grad
            p = sigmoid(float(feat @ self.w_apply + self.b_apply))
            y = 1.0 if tr["apply"] else 0.0
            g = y - p
            self.w_apply += lr * g * feat
            self.b_apply = np.array(float(self.b_apply) + lr * g, dtype=np.float64)
            self.n_updates += 1
