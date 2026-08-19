"""
Evolution-selected local plasticity rules and developmental schedules
(Axis 1, candidate C).

Parameters are optimized by evolutionary search across generations (Stage E)
or by meta-gradient as a separate comparison arm. During an evaluated life,
all parameters are FROZEN and updates are purely local.

Evolutionary path required for biological-evolution claims (Stage E).
Meta-gradient path is a comparison arm only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from three_memory.dev1.genome import PlasticityCoefficients


class EvolvedLocalRule(nn.Module):
    """
    Evolved local plasticity rule with learnable combination of Hebb, BCM,
    and prediction-error terms. All coefficients are inherited (from G).

    During an organism's life, all parameters are frozen; only activity
    variables change.
    """

    def __init__(self, n_pre: int, n_post: int, coeffs: PlasticityCoefficients):
        super().__init__()
        self.lr = coeffs.learning_rate
        self.base_lr = coeffs.learning_rate
        # Combination weights over [Hebb, BCM, pred_error] terms
        self.mix = nn.Parameter(torch.tensor([1.0, 0.3, 0.5]))
        self.bcm_threshold = nn.Parameter(torch.zeros(n_post))
        self.pe_decay = nn.Parameter(torch.tensor(coeffs.eligibility_decay))

    def update(
        self,
        W: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        reward_gate: torch.Tensor,
        prediction_error: torch.Tensor,
        pe_integral: torch.Tensor,
    ) -> torch.Tensor:
        """Local weight update using frozen evolved parameters."""
        mix = torch.softmax(self.mix, dim=0)
        hebb = torch.outer(pre, post)
        bcm_mod = (post - self.bcm_threshold) * post.detach()
        bcm = torch.outer(pre, bcm_mod)
        pe_term = prediction_error * torch.outer(pre, post)
        dW = self.base_lr * (mix[0] * hebb + mix[1] * bcm + mix[2] * pe_term)
        dW = reward_gate * dW
        return W + dW.clamp(-0.1, 0.1)

    def name(self) -> str:
        return "evolved"
