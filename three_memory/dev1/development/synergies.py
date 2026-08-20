"""
Four generic motor synergies over 32 channels.

Synergy report names (approach/withdraw/orient/wait) exist only in runner/report
code. The organism cortex sees distributed motor activity and proprioception —
never semantic labels.
"""

from __future__ import annotations

import hashlib

import torch

from three_memory.dev1.development.generative_genome import N_MOTOR_CHANNELS, N_SYNERGIES


def synergy_channel_blocks(
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
) -> list[slice]:
    """Partition motor channels into contiguous synergy blocks (construction motif)."""
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
    Fixed generic projection: motor_channels → synergy activations.
    Deterministic construction motif — not a scored mapping.
    """
    dev = device or torch.device("cpu")
    P = torch.zeros(n_synergies, n_channels, device=dev)
    for i, sl in enumerate(synergy_channel_blocks(n_channels, n_synergies)):
        P[i, sl] = float(gain) / float(sl.stop - sl.start)
    return P


def channels_to_synergy_activations(
    motor_scores: torch.Tensor,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map opaque channel competition scores to synergy activations."""
    if projection is None:
        projection = synergy_projection_matrix(
            n_channels=int(motor_scores.numel()),
            device=motor_scores.device,
        )
    return projection @ motor_scores


def synergy_template_hash(
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
    gain: float = 1.0,
) -> str:
    P = synergy_projection_matrix(n_channels, n_synergies, gain=gain, device=torch.device("cpu"))
    payload = P.detach().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()
