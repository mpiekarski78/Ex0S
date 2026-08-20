"""
Inherited learning-signal-generator e-prop (Reference Birth R3).

Within-life update remains local:
  Delta W_ij(t) = eta * L_j(t) * e_ij(t)
where L_j comes from an inherited cortex-owned generator (not fixed broadcast B).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from three_memory.dev1.genome import DevGenome, PlasticityCoefficients
from three_memory.dev1.plasticity.eprop.critic import LocalCritic
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention
from three_memory.dev1.plasticity.eprop.signal_generator import InheritedLearningSignalGenerator


class InheritedSignalGeneratorEprop:
    """
    Local e-prop with inherited neuron-specific learning signals.

    Eligibility traces are unchanged. The fixed projection B of rate-e-prop
    is replaced by InheritedLearningSignalGenerator.
    """

    def __init__(self, genome: DevGenome, n_pre: int, n_post: int, device: torch.device):
        self.genome = genome
        self.coeffs: PlasticityCoefficients = genome.plasticity
        self.device = device
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.critic = LocalCritic(n_pre, self.coeffs, device)
        vec = getattr(genome, "lsg_param_vector", None)
        self.signal_generator = InheritedLearningSignalGenerator(
            n_rel=n_pre,
            n_motor=genome.n_motor_channels,
            n_post=n_post,
            device=device,
            param_vector=vec,
            seed=genome.seed + 7919,
        )
        self.intervention = EpropIntervention.none()
        self._v_prev: torch.Tensor | None = None
        self._last_relational: torch.Tensor | None = None

    def name(self) -> str:
        return "inherited_learning_signal_generator"

    def set_intervention(self, intervention: EpropIntervention) -> None:
        self.intervention = intervention

    def reset_episode(self) -> None:
        self._v_prev = None

    @property
    def clip(self) -> float:
        return float(self.coeffs.update_clip_scale)

    @property
    def temperature(self) -> float:
        return float(self.coeffs.temperature)

    def policy_probs(self, motor_logits: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        temp = self.temperature if temperature is None else temperature
        return F.softmax(motor_logits / max(temp, 1e-6), dim=-1)

    def _policy_error_vec(self, chosen_channel: int, motor_logits: torch.Tensor) -> torch.Tensor:
        probs = self.policy_probs(motor_logits)
        one_hot = torch.zeros(self.genome.n_motor_channels, device=self.device)
        credit_channel = chosen_channel
        if self.intervention.motor_feedback_permuted:
            credit_channel = int(
                (chosen_channel + self.genome.n_motor_channels // 2) % self.genome.n_motor_channels
            )
        one_hot[credit_channel] = 1.0
        return one_hot - probs

    def learning_signal_per_unit(
        self,
        delta_t: torch.Tensor,
        chosen_channel: int,
        motor_logits: torch.Tensor,
        relational_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_vec = self._policy_error_vec(chosen_channel, motor_logits)
        rel = relational_state if relational_state is not None else self._last_relational
        if rel is None:
            rel = torch.zeros(self.n_pre, device=self.device)
        scale = float(self.coeffs.projection_scale)
        l_j = self.signal_generator.learning_signal(
            rel,
            policy_vec,
            delta_t,
            generator_off=self.intervention.signal_generator_off,
            generator_permuted=self.intervention.signal_generator_permuted,
        )
        return scale * l_j

    def actor_delta(
        self,
        eligibility: torch.Tensor,
        delta_t: torch.Tensor,
        chosen_channel: int,
        motor_logits: torch.Tensor,
        n_channels: int,
        relational_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.intervention.reward_off:
            delta_t = torch.zeros_like(delta_t)
        elig = eligibility
        if self.intervention.eligibility_zero:
            elig = torch.zeros_like(eligibility)
        elif self.intervention.eligibility_permuted:
            elig = eligibility[torch.randperm(eligibility.shape[0], device=eligibility.device), :]
        l_j = self.learning_signal_per_unit(
            delta_t, chosen_channel, motor_logits, relational_state=relational_state
        )
        elig_signal = elig.mean(dim=0)
        update_channel = chosen_channel
        if self.intervention.motor_feedback_permuted:
            update_channel = int((chosen_channel + n_channels // 2) % n_channels)
        # Delta W_ij = eta * L_j * e_ij  (channel row gets neuron-specific L)
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
        self._last_relational = relational_state.detach()
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
