"""
Stage A R1 meta-gradient outer optimizer.

This module updates only inherited genome credit parameters BETWEEN lives.
It never applies gradients to W, H, rho, or eligibility state during an
evaluated life.

Hard-action evaluation remains canonical. If a differentiable soft-action
surrogate is used during training, it must be declared in the prereg lock
and checked for train/eval mismatch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from three_memory.dev1.genome import DevGenome


@dataclass
class MetaGradientConfig:
    step_size: float = 0.05
    clamp_min: float = 1e-6
    clamp_max: float = 2.0
    use_soft_action_surrogate: bool = False


class MetaGradientOptimizer:
    """
    Lightweight outer-loop optimizer over the shared inherited credit surface.

    The current implementation uses aggregate behavioral fitness from training
    lives to scale a centered update on the credit parameter vector. It is a
    between-life optimizer only; evaluated-life state is discarded.
    """

    def __init__(self, cfg: MetaGradientConfig | None = None):
        self.cfg = cfg or MetaGradientConfig()
        self.n_updates = 0

    def propose(self, genome: DevGenome) -> DevGenome:
        """Return a copy whose parameters may be updated after training lives."""
        return copy.deepcopy(genome)

    def update_after_training_lives(
        self,
        genome: DevGenome,
        training_fitnesses: list[float],
    ) -> DevGenome:
        """
        Update inherited credit params using aggregate behavioral fitness.
        Applies only after completed training lives.
        """
        new_genome = copy.deepcopy(genome)
        params = new_genome.credit_parameter_dict()
        if not training_fitnesses:
            return new_genome

        fitness = sum(training_fitnesses) / len(training_fitnesses)
        centered = fitness - 0.5
        updated = {}
        for k, v in params.items():
            scale = 1.0 if v != 0.0 else 0.1
            nv = v + self.cfg.step_size * centered * scale
            nv = min(self.cfg.clamp_max, max(self.cfg.clamp_min, nv))
            updated[k] = nv

        new_genome.set_credit_parameter_dict(updated)
        self.n_updates += 1
        return new_genome

    def telemetry(self) -> dict:
        return {
            "optimizer_arm": "meta_gradient",
            "outer_updates": self.n_updates,
            "soft_action_surrogate": self.cfg.use_soft_action_surrogate,
        }
