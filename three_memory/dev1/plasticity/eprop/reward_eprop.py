"""
Reward-based e-prop rate adaptation for EX0S-DEV1 Reference Birth R1.

Adaptation of Bellec et al. reward-based e-prop to the existing rate-based
cortical populations. Not an exact spiking LSNN reproduction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from three_memory.dev1.genome import DevGenome, PlasticityCoefficients
from three_memory.dev1.plasticity.eprop.critic import LocalCritic
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention
from three_memory.dev1.plasticity.eprop.projections import LearningSignalProjection


class RewardEpropRateAdaptation:
    """
    Frozen contract equations (rate adaptation):
      delta_t = r_t + gamma * V_{t+1} - V_t   (V_{t+1}=0 at terminal)
      L_{j,t} = delta_t * sum_k B_{jk} * (1[a_t=k] - pi_k(t))
      Delta W = eta * L_{j,t} * e_{ji}
    Sign: gradient ascent on expected return.
    """

    def __init__(self, genome: DevGenome, n_pre: int, n_post: int, device: torch.device):
        self.genome = genome
        self.coeffs: PlasticityCoefficients = genome.plasticity
        self.device = device
        self.critic = LocalCritic(n_pre, self.coeffs, device)
        self.projection = LearningSignalProjection(
            genome.n_motor_channels,
            n_post,
            seed=genome.seed + 7919,
            device=device,
        )
        self.intervention = EpropIntervention.none()
        self._v_prev: torch.Tensor | None = None
        self._elig_critic = torch.zeros(n_pre, device=device)

    def name(self) -> str:
        return "reward_eprop_rate_adaptation"

    def set_intervention(self, intervention: EpropIntervention) -> None:
        self.intervention = intervention

    def reset_episode(self) -> None:
        self._v_prev = None
        self._elig_critic.zero_()

    @property
    def clip(self) -> float:
        return float(self.coeffs.update_clip_scale)

    @property
    def temperature(self) -> float:
        return float(self.coeffs.temperature)

    def policy_probs(self, motor_logits: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        temp = self.temperature if temperature is None else temperature
        return F.softmax(motor_logits / max(temp, 1e-6), dim=-1)

    def learning_signal_per_unit(
        self,
        delta_t: torch.Tensor,
        chosen_channel: int,
        motor_logits: torch.Tensor,
    ) -> torch.Tensor:
        probs = self.policy_probs(motor_logits)
        one_hot = torch.zeros(self.genome.n_motor_channels, device=self.device)
        credit_channel = chosen_channel
        if self.intervention.motor_feedback_permuted:
            # Permute which channel receives motor-feedback credit relative to action.
            credit_channel = int((chosen_channel + self.genome.n_motor_channels // 2) % self.genome.n_motor_channels)
        one_hot[credit_channel] = 1.0
        policy_vec = one_hot - probs
        projected = policy_vec @ self.projection.B
        return delta_t * self.coeffs.projection_scale * projected

    def actor_delta(
        self,
        eligibility: torch.Tensor,
        delta_t: torch.Tensor,
        chosen_channel: int,
        motor_logits: torch.Tensor,
        n_channels: int,
    ) -> torch.Tensor:
        if self.intervention.reward_off:
            delta_t = torch.zeros_like(delta_t)
        elig = eligibility
        if self.intervention.eligibility_zero:
            elig = torch.zeros_like(eligibility)
        elif self.intervention.eligibility_permuted:
            elig = eligibility[torch.randperm(eligibility.shape[0], device=eligibility.device), :]
        l_j = self.learning_signal_per_unit(delta_t, chosen_channel, motor_logits)
        elig_signal = elig.mean(dim=0)
        update_channel = chosen_channel
        if self.intervention.motor_feedback_permuted:
            update_channel = int((chosen_channel + n_channels // 2) % n_channels)
        dW = torch.zeros(n_channels, elig_signal.numel(), device=self.device)
        dW[update_channel] = self.coeffs.learning_rate * l_j * elig_signal
        return dW.clamp(-self.clip, self.clip)

    def update_critic(self, delta_t: torch.Tensor, relational_state: torch.Tensor) -> float:
        if self.intervention.reward_off:
            return 0.0
        with torch.no_grad():
            dW = (
                self.coeffs.critic_scale
                * self.coeffs.critic_learning_rate
                * delta_t
                * relational_state.unsqueeze(0)
            ).clamp(-self.clip, self.clip)
            self.critic.fc.weight.data.add_(dW)
            if self.critic.fc.bias is not None:
                self.critic.fc.bias.data.add_(
                    self.coeffs.critic_scale * self.coeffs.critic_learning_rate * delta_t
                )
        return float(dW.norm().item())

    def td_step(
        self,
        reward: float,
        relational_state: torch.Tensor,
        is_terminal: bool,
    ) -> torch.Tensor:
        v_t = self.critic.value(relational_state)
        if is_terminal:
            v_next = torch.zeros((), device=self.device)
        elif self._v_prev is not None:
            v_next = self._v_prev
        else:
            v_next = v_t.detach()
        delta = self.critic.td_error(reward, v_t, v_next, is_terminal)
        self._v_prev = v_t.detach()
        return delta.squeeze()
