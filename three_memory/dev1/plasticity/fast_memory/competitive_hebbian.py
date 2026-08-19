"""
Competitive Hebbian fast memory (Axis 2, candidate A).

Implements winner-take-all + Hebbian writes for H.
This is the default fast_memory_family; used as baseline in axis 2 search.

The memory writes and WTA sparsification are implemented in
hippocampus.py (FastHippocampus). This module provides the plasticity
rule that governs how W_hebb evolves and includes optional decay.
"""

from __future__ import annotations

import torch

from three_memory.dev1.genome import PlasticityCoefficients


class CompetitiveHebbian:
    """
    Competitive Hebbian rule for fast episodic memory.

    Applied to W_hebb in FastHippocampus during write().
    """

    def __init__(self, capacity: int, hebbian_lr: float, decay: float = 0.999):
        self.hebbian_lr = hebbian_lr
        self.decay = decay

    def update(self, W_hebb: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Outer product Hebb update with optional decay.
        Returns updated W_hebb (in-place).
        """
        key_n = torch.nn.functional.normalize(key, dim=0)
        W_hebb.mul_(self.decay)
        W_hebb.add_(self.hebbian_lr * torch.outer(key_n, key_n))
        return W_hebb

    def name(self) -> str:
        return "competitive_hebbian"
