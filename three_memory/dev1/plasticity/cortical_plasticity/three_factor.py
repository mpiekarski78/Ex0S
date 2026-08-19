"""
Reward-baseline three-factor credit rule (Stage A R1 family 1).

delta_t = r_t - r_hat_t
DeltaW_actor = eta * delta_t * e_t

Where:
- e(pre, post) = eligibility trace (pre-synaptic × post-synaptic coactivation)
- M = neuromodulatory gate (reward advantage + prediction error)
- No gradient-based update during evaluated life; this is a within-life local rule.
"""

from __future__ import annotations

import torch

from three_memory.dev1.genome import PlasticityCoefficients


class RewardBaselineThreeFactor:
    """
    Reward-baseline three-factor actor credit.

    This family never sees the correct answer identity. It only uses:
    - local eligibility
    - scalar reward-baseline error
    - the organism's chosen motor channel
    """

    def __init__(self, coeffs: PlasticityCoefficients):
        self.lr = coeffs.learning_rate
        self.reward_scale = coeffs.reward_gate_scale
        self.pe_scale = coeffs.prediction_error_scale

    def actor_delta(
        self,
        eligibility: torch.Tensor,
        reward_baseline_error: torch.Tensor,
        chosen_channel: int,
        n_channels: int,
    ) -> torch.Tensor:
        """Return a chosen-action-only actor update matrix."""
        signal = self.reward_scale * reward_baseline_error
        elig_signal = eligibility.mean(dim=0)
        dW = torch.zeros(n_channels, elig_signal.numel(), device=eligibility.device)
        dW[chosen_channel] = self.lr * signal * elig_signal
        return dW.clamp(-0.1, 0.1)

    def name(self) -> str:
        return "reward_baseline_three_factor"
