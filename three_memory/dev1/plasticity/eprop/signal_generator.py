"""
Inherited cortex-owned learning-signal generator (Reference Birth R3).

Produces neuron-specific learning signals L_j for local e-prop updates:
  Delta W_ij(t) = eta * L_j(t) * e_ij(t)

Parameters are inherited across newborns and searched by the outer process.
Within an evaluated life the generator is frozen (no BPTT through L).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Compact feature sizes keep ES surface tractable under the R1/R2 budget class.
LSG_REL_FEATURES = 32
LSG_HIDDEN = 24


def lsg_input_dim(n_motor: int) -> int:
    return LSG_REL_FEATURES + int(n_motor) + 1


def lsg_param_count(n_motor: int, n_post: int, hidden: int = LSG_HIDDEN) -> int:
    d_in = lsg_input_dim(n_motor)
    # fc1 weight + bias + fc2 weight + bias
    return d_in * hidden + hidden + hidden * n_post + n_post


def default_lsg_vector(
    n_motor: int,
    n_post: int,
    seed: int = 0,
    hidden: int = LSG_HIDDEN,
) -> list[float]:
    """Generic random init — not task-specific; no symbols/facts/mappings."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 4242)
    n = lsg_param_count(n_motor, n_post, hidden=hidden)
    # Xavier-ish scale
    scale = 0.05
    vec = torch.randn(n, generator=gen) * scale
    return [float(x) for x in vec.tolist()]


class InheritedLearningSignalGenerator(nn.Module):
    """
    Cortex-owned network: features → L_j (n_post).

    Features = [relational[:k], policy_error_vec, delta_t].
    """

    def __init__(
        self,
        n_rel: int,
        n_motor: int,
        n_post: int,
        *,
        hidden: int = LSG_HIDDEN,
        device: torch.device | None = None,
        param_vector: Sequence[float] | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.n_rel = int(n_rel)
        self.n_motor = int(n_motor)
        self.n_post = int(n_post)
        self.hidden = int(hidden)
        self.device = device or torch.device("cpu")
        d_in = lsg_input_dim(self.n_motor)
        self.fc1 = nn.Linear(d_in, self.hidden, bias=True)
        self.fc2 = nn.Linear(self.hidden, self.n_post, bias=True)
        self.to(self.device)
        if param_vector is None:
            param_vector = default_lsg_vector(
                self.n_motor, self.n_post, seed=seed, hidden=self.hidden
            )
        self.load_param_vector(param_vector)

    def param_vector(self) -> list[float]:
        return [float(x) for x in torch.cat([p.data.reshape(-1).cpu() for p in self.parameters()]).tolist()]

    def load_param_vector(self, vec: Sequence[float]) -> None:
        expected = lsg_param_count(self.n_motor, self.n_post, hidden=self.hidden)
        if len(vec) != expected:
            raise ValueError(f"LSG vector length {len(vec)} != expected {expected}")
        flat = torch.tensor(list(vec), dtype=torch.float32, device=self.device)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(flat[offset : offset + n].reshape_as(p))
                offset += n

    def feature_vector(
        self,
        relational: torch.Tensor,
        policy_vec: torch.Tensor,
        delta_t: torch.Tensor | float,
    ) -> torch.Tensor:
        rel = relational.reshape(-1)
        if rel.numel() >= LSG_REL_FEATURES:
            rel_f = rel[:LSG_REL_FEATURES]
        else:
            rel_f = F.pad(rel, (0, LSG_REL_FEATURES - rel.numel()))
        pv = policy_vec.reshape(-1)
        if pv.numel() != self.n_motor:
            if pv.numel() > self.n_motor:
                pv = pv[: self.n_motor]
            else:
                pv = F.pad(pv, (0, self.n_motor - pv.numel()))
        if isinstance(delta_t, torch.Tensor):
            d = delta_t.reshape(-1)[:1].to(dtype=torch.float32, device=self.device)
        else:
            d = torch.tensor([float(delta_t)], dtype=torch.float32, device=self.device)
        return torch.cat([rel_f.to(self.device), pv.to(self.device), d], dim=0)

    def forward(
        self,
        relational: torch.Tensor,
        policy_vec: torch.Tensor,
        delta_t: torch.Tensor | float,
    ) -> torch.Tensor:
        x = self.feature_vector(relational, policy_vec, delta_t)
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)

    def learning_signal(
        self,
        relational: torch.Tensor,
        policy_vec: torch.Tensor,
        delta_t: torch.Tensor | float,
        *,
        generator_off: bool = False,
        generator_permuted: bool = False,
    ) -> torch.Tensor:
        if generator_off:
            return torch.zeros(self.n_post, device=self.device)
        with torch.no_grad():
            l_j = self.forward(relational, policy_vec, delta_t)
            if generator_permuted:
                # Deterministic half-cycle permute — not RNG-dependent mid-life.
                shift = max(1, self.n_post // 2)
                l_j = torch.roll(l_j, shifts=shift, dims=0)
            return l_j
