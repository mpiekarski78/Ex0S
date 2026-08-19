"""
Consequence-prediction credit rule (Stage A R1 family 3).

Plasticity combines sensory-consequence prediction error with scalar reward
gating. Self-supervised consequence prediction may remain active without
reward, but persistent actor credit must still be reward-gated.
"""

from __future__ import annotations

import torch
from three_memory.dev1.genome import PlasticityCoefficients


class ConsequencePredictionCredit:
    """
    Consequence-prediction gated actor credit.

    Uses the local consequence prediction error together with scalar reward
    gating and eligibility on the organism's chosen motor output.
    """

    def __init__(self, n_pre: int, n_post: int, coeffs: PlasticityCoefficients):
        self.lr = coeffs.learning_rate
        self.reward_scale = coeffs.reward_gate_scale
        self.consequence_scale = coeffs.consequence_scale
        self.reward_gate_center = coeffs.reward_gate_center

    def actor_delta(
        self,
        eligibility: torch.Tensor,
        reward_gate: torch.Tensor,
        consequence_error: torch.Tensor,
        chosen_channel: int,
        n_channels: int,
    ) -> torch.Tensor:
        """Return actor update gated by reward and consequence mismatch."""
        elig_signal = eligibility.mean(dim=0)
        gate = (reward_gate - self.reward_gate_center) * 2.0
        signal = self.reward_scale * gate * self.consequence_scale * consequence_error
        dW = torch.zeros(n_channels, elig_signal.numel(), device=eligibility.device)
        dW[chosen_channel] = self.lr * signal * elig_signal
        return dW.clamp(-0.1, 0.1)

    def name(self) -> str:
        return "consequence_prediction_credit"
