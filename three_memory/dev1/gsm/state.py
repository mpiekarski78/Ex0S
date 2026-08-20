"""Organism-visible state packing for gestational forward models.

No synergy IDs, expected actions, target locations, or correctness labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisibleDims:
    exo_dim: int
    proprio_dim: int
    intero_dim: int
    motor_dim: int

    @property
    def state_dim(self) -> int:
        return self.exo_dim + self.proprio_dim + self.intero_dim


def dims_from_body_config(
    *,
    sensory_dim: int = 48,
    proprioceptive_dim: int = 8,
    interoceptive_dim: int = 4,
    n_motor_channels: int = 32,
) -> VisibleDims:
    exo = max(0, int(sensory_dim) - int(proprioceptive_dim))
    return VisibleDims(
        exo_dim=exo,
        proprio_dim=int(proprioceptive_dim),
        intero_dim=int(interoceptive_dim),
        motor_dim=int(n_motor_channels),
    )


def split_sensory(sensory: torch.Tensor, dims: VisibleDims) -> tuple[torch.Tensor, torch.Tensor]:
    s = sensory.detach().float().view(-1)
    if s.numel() < dims.exo_dim + dims.proprio_dim:
        pad = torch.zeros(dims.exo_dim + dims.proprio_dim - s.numel(), device=s.device)
        s = torch.cat([s, pad])
    else:
        s = s[: dims.exo_dim + dims.proprio_dim]
    exo = s[: dims.exo_dim]
    proprio = s[dims.exo_dim : dims.exo_dim + dims.proprio_dim]
    return exo, proprio


def pack_visible_state(
    *,
    sensory: torch.Tensor,
    intero: torch.Tensor,
    dims: VisibleDims,
) -> torch.Tensor:
    exo, proprio = split_sensory(sensory, dims)
    intero_v = intero.detach().float().view(-1)
    if intero_v.numel() < dims.intero_dim:
        intero_v = torch.cat(
            [intero_v, torch.zeros(dims.intero_dim - intero_v.numel(), device=intero_v.device)]
        )
    else:
        intero_v = intero_v[: dims.intero_dim]
    return torch.cat([exo, proprio, intero_v])


def unpack_visible_state(state: torch.Tensor, dims: VisibleDims) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = state.detach().float().view(-1)
    if x.numel() < dims.state_dim:
        x = torch.cat([x, torch.zeros(dims.state_dim - x.numel(), device=x.device)])
    else:
        x = x[: dims.state_dim]
    exo = x[: dims.exo_dim]
    proprio = x[dims.exo_dim : dims.exo_dim + dims.proprio_dim]
    intero = x[dims.exo_dim + dims.proprio_dim :]
    return exo, proprio, intero


def pack_efference(motor: torch.Tensor, dims: VisibleDims) -> torch.Tensor:
    m = motor.detach().float().view(-1)
    if m.numel() < dims.motor_dim:
        m = torch.cat([m, torch.zeros(dims.motor_dim - m.numel(), device=m.device)])
    else:
        m = m[: dims.motor_dim]
    return m
