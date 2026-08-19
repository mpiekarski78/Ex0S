"""
Meta-learned local plasticity (Axis 1, candidate B).

Plasticity coefficients are differentiable parameters optimized
by the research optimizer across training lives (never during
evaluated lives). The local rule itself remains gradient-free
during a single newborn lifetime.

The meta-learnable parameters:
- per-synapse learning rate modulation (fast weights)
- eligibility combination weights
- prediction-error integration timescale

Based on: Miconi et al. "Differentiable plasticity: training plastic
neural networks with backpropagation" (ICML 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from three_memory.dev1.genome import PlasticityCoefficients


class MetaLearnedPlasticity(nn.Module):
    """
    Differentiable plasticity rule.

    alpha (per-synapse): meta-learned rate; shaped by research optimizer.
    During an evaluated life, alpha is FROZEN and the update is local.
    """

    def __init__(self, n_pre: int, n_post: int, coeffs: PlasticityCoefficients):
        super().__init__()
        self.base_lr = coeffs.learning_rate
        self.alpha = nn.Parameter(torch.zeros(n_pre, n_post))
        self.pe_timescale = nn.Parameter(torch.tensor(0.9))

    def update(
        self,
        W: torch.Tensor,
        activity_pre: torch.Tensor,
        activity_post: torch.Tensor,
        reward_gate: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> torch.Tensor:
        """Local update using frozen alpha during evaluated life."""
        alpha_eff = torch.sigmoid(self.alpha)
        hebb = torch.outer(activity_pre, activity_post)
        M = reward_gate * prediction_error.clamp(min=0.0)
        dW = self.base_lr * alpha_eff * hebb * M
        return W + dW.clamp(-0.1, 0.1)

    def name(self) -> str:
        return "meta_learned"
