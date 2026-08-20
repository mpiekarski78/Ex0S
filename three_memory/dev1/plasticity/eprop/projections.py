"""Learning-signal projection B for e-prop rate adaptation."""

from __future__ import annotations

import torch


class LearningSignalProjection:
    """
    Fixed generic projection B_{jk} inherited per genome.
    Maps population learning signals to motor channels.
    """

    def __init__(self, n_motor: int, n_hidden: int, seed: int, device: torch.device):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        raw = torch.randn(n_motor, n_hidden, generator=gen)
        self.B = torch.nn.functional.normalize(raw, dim=1).to(device)

    def project(self, hidden_signal: torch.Tensor) -> torch.Tensor:
        """Return per-channel learning signal contributions."""
        return self.B @ hidden_signal
