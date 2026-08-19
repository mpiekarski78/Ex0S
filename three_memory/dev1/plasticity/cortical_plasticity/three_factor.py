"""
Three-factor neuromodulated plasticity (Axis 1, candidate A).

ΔW = η · e(pre, post) · M(reward, prediction_error)

Where:
- e(pre, post) = eligibility trace (pre-synaptic × post-synaptic coactivation)
- M = neuromodulatory gate (reward advantage + prediction error)
- No gradient-based update during evaluated life; this is a within-life local rule.
"""

from __future__ import annotations

import torch

from three_memory.dev1.genome import PlasticityCoefficients


class ThreeFactorPlasticity:
    """
    Three-factor synaptic plasticity rule.

    Applied to one weight matrix at a time by organism.py during rest().
    Operates on eligibility traces accumulated during the life.
    """

    def __init__(self, coeffs: PlasticityCoefficients):
        self.lr = coeffs.learning_rate
        self.reward_scale = coeffs.reward_gate_scale
        self.pe_scale = coeffs.prediction_error_scale

    def update(
        self,
        W: torch.Tensor,
        eligibility: torch.Tensor,
        reward_gate: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weight delta.
        Returns ΔW (same shape as W).
        """
        M = self.reward_scale * reward_gate + self.pe_scale * prediction_error
        dW = self.lr * M * eligibility
        return W + dW.clamp(-0.1, 0.1)

    def name(self) -> str:
        return "three_factor"
