"""
Replay-driven consolidation (Axis 3, candidate A).

Organism samples episodes from H during rest() and replays them
to update slow cortex W via the frozen local plasticity rule.
No offline corpus. No expected-answer gradients.
"""

from __future__ import annotations

import random
import torch
from typing import Callable

from three_memory.dev1.genome import DevGenome


class ReplayConsolidation:
    """
    Replay-based cortical consolidation.

    Samples n episodes from H and produces pre/post activity pairs
    for the cortical plasticity rule to consume. The organism selects
    which episodes to replay; the runner does not specify them.
    """

    def __init__(self, genome: DevGenome):
        self.n_samples = genome.replay.n_replay_samples
        self.prioritized = genome.replay.prioritized
        self.alpha = genome.replay.surprise_alpha

    def sample_episodes(
        self,
        store: list[tuple[torch.Tensor, torch.Tensor]],
        surprises: list[float] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample episodes from H store.
        If surprises available, sample proportionally to surprise^alpha.
        """
        if not store:
            return []
        n = min(self.n_samples, len(store))
        if self.prioritized and surprises and len(surprises) == len(store):
            weights = [max(s, 0.0) ** self.alpha + 1e-6 for s in surprises]
            total = sum(weights)
            probs = [w / total for w in weights]
            indices = random.choices(range(len(store)), weights=probs, k=n)
        else:
            indices = random.sample(range(len(store)), n)
        return [store[i] for i in indices]

    def name(self) -> str:
        return "replay"
