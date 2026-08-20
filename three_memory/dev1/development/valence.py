"""
Organism-owned generic valence circuit.

The body exposes interoceptive state but never an expected action.
This circuit computes reinforcement from consecutive interoceptive states.
Runner behavioral correctness must never feed this path.
"""

from __future__ import annotations

import hashlib

import torch


class OrganismValenceCircuit:
    """
    Generic valence: signed improvement of homeostatic comfort from consecutive
    interoceptive observations.
    """

    def __init__(
        self,
        interoceptive_dim: int,
        *,
        gain: float = 1.0,
        setpoint: float = 0.5,
        device: torch.device | None = None,
    ):
        self.interoceptive_dim = int(interoceptive_dim)
        self.gain = float(gain)
        self.setpoint = float(setpoint)
        self.device = device or torch.device("cpu")
        self._prev: torch.Tensor | None = None
        self._last_valence: float = 0.0

    def reset(self) -> None:
        self._prev = None
        self._last_valence = 0.0

    def comfort(self, interoceptive: torch.Tensor) -> float:
        """Scalar comfort: negative mean absolute deviation from setpoint."""
        x = interoceptive.detach().float().to(self.device).view(-1)
        if x.numel() != self.interoceptive_dim:
            # pad/trim without leaking task structure
            if x.numel() < self.interoceptive_dim:
                pad = torch.zeros(self.interoceptive_dim - x.numel(), device=self.device)
                x = torch.cat([x, pad])
            else:
                x = x[: self.interoceptive_dim]
        return float(-(x - self.setpoint).abs().mean().item())

    def update(self, interoceptive: torch.Tensor) -> float:
        """
        Compute valence from consecutive interoceptive states.
        Returns organism-owned reinforcement for plasticity.
        """
        comfort_now = self.comfort(interoceptive)
        if self._prev is None:
            valence = 0.0
        else:
            comfort_prev = self.comfort(self._prev)
            valence = self.gain * (comfort_now - comfort_prev)
        self._prev = interoceptive.detach().float().to(self.device).view(-1).clone()
        if self._prev.numel() != self.interoceptive_dim:
            if self._prev.numel() < self.interoceptive_dim:
                pad = torch.zeros(self.interoceptive_dim - self._prev.numel(), device=self.device)
                self._prev = torch.cat([self._prev, pad])
            else:
                self._prev = self._prev[: self.interoceptive_dim]
        self._last_valence = float(valence)
        return self._last_valence

    @property
    def last_valence(self) -> float:
        return self._last_valence

    def circuit_hash(self) -> str:
        payload = f"valence|dim={self.interoceptive_dim}|gain={self.gain}|set={self.setpoint}"
        return hashlib.sha256(payload.encode()).hexdigest()
