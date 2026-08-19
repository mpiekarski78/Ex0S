"""
Online slow cortical consolidation (Axis 3, candidate B).

Slow weights update continuously during the organism's life via the
same local plasticity rule, but with a much smaller learning rate.
No replay from H; relies on in-context repetition for consolidation.
"""

from __future__ import annotations

import torch

from three_memory.dev1.genome import DevGenome


class OnlineSlowConsolidation:
    """
    Slow online consolidation: reduces the cortical plasticity LR by
    a factor during evaluated life and applies it to a slow copy of W.
    """

    def __init__(self, genome: DevGenome, slowdown_factor: float = 0.01):
        self.effective_lr = genome.plasticity.learning_rate * slowdown_factor

    def step(
        self,
        W_slow: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        reward_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Increment slow weights from immediate coactivation."""
        hebb = torch.outer(pre, post)
        dW = self.effective_lr * reward_gate * hebb
        return W_slow + dW.clamp(-0.01, 0.01)

    def name(self) -> str:
        return "online_slow"
