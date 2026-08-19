"""
Action-contingent actor-critic credit rule (Stage A R1 family 2).

delta_t = r_t + gamma * V(s_t+1) - V(s_t)

The actor update is attached to the organism's chosen motor action.
No correct answer identity is ever supplied to the rule.
"""

from __future__ import annotations

import torch
from three_memory.dev1.genome import PlasticityCoefficients


class ActionContingentActorCritic:
    """
    Local actor-critic credit family.

    This is the within-life local rule only. Meta-gradient or evolution
    may optimize its inherited coefficients between lives, but no gradient
    update occurs inside an evaluated life.
    """

    def __init__(self, n_pre: int, n_post: int, coeffs: PlasticityCoefficients):
        self.lr = coeffs.learning_rate
        self.gamma = coeffs.gamma
        self.actor_scale = coeffs.actor_credit_scale
        self.critic_scale = coeffs.critic_scale

    def actor_delta(
        self,
        eligibility: torch.Tensor,
        td_error: torch.Tensor,
        chosen_channel: int,
        n_channels: int,
    ) -> torch.Tensor:
        """Return an actor-only chosen-action update using TD error."""
        elig_signal = eligibility.mean(dim=0)
        dW = torch.zeros(n_channels, elig_signal.numel(), device=eligibility.device)
        dW[chosen_channel] = self.lr * self.actor_scale * td_error * elig_signal
        return dW.clamp(-0.1, 0.1)

    def name(self) -> str:
        return "action_contingent_actor_critic"
