"""
Nursery Body v2 — egocentric motor basis with mass-preserving synergy projection.

Engineering surface only until certification freeze. Does not modify the frozen
R4 GenericBody path (ba97883). Organism never sees synergy name strings.
"""

from __future__ import annotations

import hashlib

import torch

N_MOTOR_CHANNELS = 32
N_SYNERGIES = 4
# Runner/report names only — egocentric mechanics, not goal-relative verbs.
SYNERGY_REPORT_NAMES: tuple[str, ...] = (
    "forward",
    "backward",
    "rotate_left",
    "rotate_right",
)


def synergy_channel_blocks(
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
) -> list[slice]:
    if n_channels % n_synergies != 0:
        raise ValueError("n_channels must be divisible by n_synergies")
    width = n_channels // n_synergies
    return [slice(i * width, (i + 1) * width) for i in range(n_synergies)]


def synergy_projection_matrix(
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
    gain: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Mass-preserving block projection.

    syn_i = gain * sum(motor[block_i]).
    One-hot in a block and uniform mass-1 distributed across the block both
    yield the same synergy activation (gain). Channel count within a synergy
    does not attenuate a unit motor command.
    """
    dev = device or torch.device("cpu")
    P = torch.zeros(n_synergies, n_channels, device=dev)
    for i, sl in enumerate(synergy_channel_blocks(n_channels, n_synergies)):
        P[i, sl] = float(gain)
    return P


def channels_to_synergy_activations(
    motor_scores: torch.Tensor,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    if projection is None:
        projection = synergy_projection_matrix(
            n_channels=int(motor_scores.numel()),
            device=motor_scores.device,
        )
    return projection @ motor_scores


def expand_synergy_index_to_motor(
    synergy_index: int,
    *,
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
    device: torch.device | None = None,
    encoding: str = "uniform_block",
    channel_within_block: int = 0,
) -> torch.Tensor:
    """Expand one synergy choice; uniform_block ≡ onehot_in_block under mass-preserving P."""
    dev = device or torch.device("cpu")
    blocks = synergy_channel_blocks(n_channels, n_synergies)
    s = int(synergy_index) % n_synergies
    sl = blocks[s]
    motor = torch.zeros(n_channels, device=dev)
    width = sl.stop - sl.start
    if encoding == "uniform_block":
        motor[sl] = 1.0 / float(width)
    elif encoding == "onehot_in_block":
        motor[sl.start + (int(channel_within_block) % width)] = 1.0
    else:
        raise ValueError(encoding)
    return motor


def permute_channels_within_synergy(
    motor: torch.Tensor,
    *,
    n_synergies: int = N_SYNERGIES,
    perm_seed: int = 0,
) -> torch.Tensor:
    n_channels = int(motor.numel())
    out = motor.detach().clone()
    blocks = synergy_channel_blocks(n_channels, n_synergies)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(perm_seed))
    for sl in blocks:
        width = sl.stop - sl.start
        order = torch.randperm(width, generator=gen)
        block = out[sl].clone()
        out[sl] = block[order]
    return out


def synergy_template_hash(
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
    gain: float = 1.0,
) -> str:
    P = synergy_projection_matrix(n_channels, n_synergies, gain=gain, device=torch.device("cpu"))
    return hashlib.sha256(P.detach().cpu().numpy().tobytes()).hexdigest()
