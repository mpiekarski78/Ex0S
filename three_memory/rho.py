"""Session-only working trace ρ. Reset clears the life of the moment, not of S."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RhoConfig:
    embed_dim: int = 32
    decay: float = 0.92


class WorkingTrace:
    """EMA accumulator over embeddings (BDH-ρ analogue: short, fragile residue)."""

    def __init__(self, config: RhoConfig | None = None):
        self.config = config or RhoConfig()
        self.rho = np.zeros(self.config.embed_dim, dtype=np.float64)
        self.steps = 0
        self.last_success_action: int | None = None

    def reset(self) -> None:
        self.rho = np.zeros(self.config.embed_dim, dtype=np.float64)
        self.steps = 0
        self.last_success_action = None

    def snapshot(self) -> dict:
        return {
            "rho": self.rho.copy(),
            "steps": self.steps,
            "last_success_action": self.last_success_action,
        }

    def load(self, snap: dict) -> None:
        self.rho = np.asarray(snap["rho"], dtype=np.float64).copy()
        self.steps = int(snap["steps"])
        self.last_success_action = snap.get("last_success_action")

    def update(self, embed: np.ndarray) -> np.ndarray:
        e = np.asarray(embed, dtype=np.float64).reshape(-1)
        self.rho = self.config.decay * self.rho + (1.0 - self.config.decay) * e
        self.steps += 1
        return self.rho

    def note_success(self, action: int) -> None:
        self.last_success_action = int(action)

    def predict(self) -> np.ndarray:
        return self.rho.copy()

    def distance(self, other: "WorkingTrace") -> dict[str, float]:
        a, b = self.rho, other.rho
        l2 = float(np.linalg.norm(a - b))
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        cos = float(np.dot(a, b) / (na * nb + 1e-12)) if na > 0 and nb > 0 else 1.0
        return {"l2": l2, "cosine": cos, "rel_norm": abs(na - nb) / (max(na, nb) + 1e-12)}
