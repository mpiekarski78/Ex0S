"""
Hybrid consolidation (Axis 3, candidate C).

Combines replay-driven offline consolidation (after rest() calls)
with slow online updates. Both mechanisms work together:
- During life: slow online Hebbian increments accumulate
- During rest: replay episodes from H reinforce the accumulation
"""

from __future__ import annotations

import torch

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.plasticity.consolidation.replay import ReplayConsolidation
from three_memory.dev1.plasticity.consolidation.online_slow import OnlineSlowConsolidation


class HybridConsolidation:
    """Combines online_slow and replay consolidation."""

    def __init__(self, genome: DevGenome):
        self.replay = ReplayConsolidation(genome)
        self.online = OnlineSlowConsolidation(genome, slowdown_factor=0.005)

    def name(self) -> str:
        return "hybrid"
