"""Local critic head for e-prop rate adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn

from three_memory.dev1.genome import PlasticityCoefficients


class LocalCritic(nn.Module):
    """Scalar state-value critic V_t from relational representation."""

    def __init__(self, in_dim: int, coeffs: PlasticityCoefficients, device: torch.device):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
        self.gamma = coeffs.gamma
        self.device = device
        self.to(device)
        nn.init.zeros_(self.fc.bias)

    def value(self, relational_state: torch.Tensor) -> torch.Tensor:
        return self.fc(relational_state).squeeze(-1)

    def td_error(
        self,
        reward: float,
        v_t: torch.Tensor,
        v_next: torch.Tensor,
        is_terminal: bool,
    ) -> torch.Tensor:
        v_next_eff = torch.zeros_like(v_next) if is_terminal else v_next
        r = torch.tensor(reward, device=self.device, dtype=torch.float32)
        return r + self.gamma * v_next_eff.detach() - v_t
