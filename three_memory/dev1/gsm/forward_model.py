"""
Cortex-owned gestational forward sensorimotor model.

Predicts state *change* Δs_{t+1} = F(s_t, a_t) to avoid trivial persistence.
Outputs separate exteroceptive, proprioceptive, and interoceptive deltas.
Never receives synergy IDs, expected actions, target locations, or correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from three_memory.dev1.gsm.state import (
    VisibleDims,
    dims_from_body_config,
    pack_efference,
    pack_visible_state,
    unpack_visible_state,
)


@dataclass
class ForwardPrediction:
    delta_exo: torch.Tensor
    delta_proprio: torch.Tensor
    delta_intero: torch.Tensor
    predicted_state: torch.Tensor
    uncertainty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta_exo": self.delta_exo.detach().cpu(),
            "delta_proprio": self.delta_proprio.detach().cpu(),
            "delta_intero": self.delta_intero.detach().cpu(),
            "predicted_state": self.predicted_state.detach().cpu(),
            "uncertainty": float(self.uncertainty),
        }


class ForwardSensorimotorModel(nn.Module):
    """
    s_hat_{t+1} = s_t + Δ(s_t, a_t)

    Input: organism-visible state + distributed efference copy.
    """

    def __init__(
        self,
        dims: VisibleDims | None = None,
        *,
        hidden: int = 96,
        device: torch.device | None = None,
        action_agnostic: bool = False,
    ):
        super().__init__()
        self.dims = dims or dims_from_body_config()
        self.device = device or torch.device("cpu")
        self.action_agnostic = bool(action_agnostic)
        in_dim = self.dims.state_dim + (0 if self.action_agnostic else self.dims.motor_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head_exo = nn.Linear(hidden, self.dims.exo_dim)
        self.head_proprio = nn.Linear(hidden, self.dims.proprio_dim)
        self.head_intero = nn.Linear(hidden, self.dims.intero_dim)
        # Scalar log-variance proxy for prediction uncertainty (learned calibration aid).
        self.head_logvar = nn.Linear(hidden, 1)
        self.to(self.device)
        self._recent_abs_errors: list[float] = []

    def _encode(self, state: torch.Tensor, motor: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device).view(-1)
        if self.action_agnostic:
            x = state
        else:
            motor = pack_efference(motor, self.dims).to(self.device)
            x = torch.cat([state, motor])
        return self.encoder(x)

    def predict_delta(self, state: torch.Tensor, motor: torch.Tensor) -> ForwardPrediction:
        h = self._encode(state, motor)
        d_exo = self.head_exo(h)
        d_pro = self.head_proprio(h)
        d_int = self.head_intero(h)
        logvar = self.head_logvar(h).squeeze(-1)
        uncertainty = float(torch.exp(0.5 * logvar).item())
        # Blend with empirical recent error when available.
        if self._recent_abs_errors:
            emp = sum(self._recent_abs_errors) / len(self._recent_abs_errors)
            uncertainty = float(0.5 * uncertainty + 0.5 * emp)
        delta = torch.cat([d_exo, d_pro, d_int])
        pred_state = state.to(self.device).view(-1)[: self.dims.state_dim] + delta
        return ForwardPrediction(
            delta_exo=d_exo,
            delta_proprio=d_pro,
            delta_intero=d_int,
            predicted_state=pred_state,
            uncertainty=uncertainty,
        )

    def predict_next_from_step(
        self,
        *,
        sensory: torch.Tensor,
        intero: torch.Tensor,
        motor: torch.Tensor,
    ) -> ForwardPrediction:
        state = pack_visible_state(sensory=sensory, intero=intero, dims=self.dims).to(self.device)
        return self.predict_delta(state, motor)

    def loss_on_transition(
        self,
        *,
        sensory_t: torch.Tensor,
        intero_t: torch.Tensor,
        motor: torch.Tensor,
        sensory_tp1: torch.Tensor,
        intero_tp1: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        s0 = pack_visible_state(sensory=sensory_t, intero=intero_t, dims=self.dims).to(self.device)
        s1 = pack_visible_state(sensory=sensory_tp1, intero=intero_tp1, dims=self.dims).to(self.device)
        target_delta = s1 - s0
        pred = self.predict_delta(s0, motor)
        pred_delta = torch.cat([pred.delta_exo, pred.delta_proprio, pred.delta_intero])
        # Separate stream MSE + mild uncertainty calibration toward residual.
        exo_err = torch.mean((pred.delta_exo - target_delta[: self.dims.exo_dim]) ** 2)
        pro_err = torch.mean(
            (pred.delta_proprio - target_delta[self.dims.exo_dim : self.dims.exo_dim + self.dims.proprio_dim])
            ** 2
        )
        int_err = torch.mean(
            (pred.delta_intero - target_delta[self.dims.exo_dim + self.dims.proprio_dim :]) ** 2
        )
        residual = float(torch.mean(torch.abs(pred_delta - target_delta)).item())
        h = self._encode(s0, motor)
        logvar = self.head_logvar(h).squeeze(-1)
        # Encourage log-variance to track residual magnitude (stabilized).
        unc_loss = 0.05 * (logvar - torch.log(torch.tensor(residual + 1e-4, device=self.device))) ** 2
        loss = exo_err + pro_err + int_err + unc_loss
        return loss, {
            "exo_mse": float(exo_err.item()),
            "proprio_mse": float(pro_err.item()),
            "intero_mse": float(int_err.item()),
            "abs_delta_error": residual,
        }

    def record_realized_error(self, abs_delta_error: float, *, maxlen: int = 64) -> None:
        self._recent_abs_errors.append(float(abs_delta_error))
        if len(self._recent_abs_errors) > maxlen:
            self._recent_abs_errors = self._recent_abs_errors[-maxlen:]

    def predicted_intero(self, state: torch.Tensor, motor: torch.Tensor) -> torch.Tensor:
        pred = self.predict_delta(state, motor)
        _, _, intero0 = unpack_visible_state(state.to(self.device), self.dims)
        return intero0 + pred.delta_intero

    def clone_weights_to(self, other: "ForwardSensorimotorModel") -> None:
        other.load_state_dict(self.state_dict())
        other._recent_abs_errors = list(self._recent_abs_errors)


def constant_delta_baseline(train_deltas: torch.Tensor) -> torch.Tensor:
    """Mean Δ over training transitions (constant predictor in delta space)."""
    if train_deltas.ndim == 1:
        return train_deltas.detach().clone()
    return train_deltas.detach().mean(dim=0)


def persistence_delta(dims: VisibleDims, device: torch.device | None = None) -> torch.Tensor:
    """Last-state / persistence: predict Δ = 0."""
    return torch.zeros(dims.state_dim, device=device or torch.device("cpu"))
