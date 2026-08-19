"""
Stage A R1 evolutionary outer optimizer.

This module mutates and selects only the shared inherited genome credit
surface BETWEEN lives. No within-life neural state is inherited.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from three_memory.dev1.genome import DevGenome


@dataclass
class EvolutionaryConfig:
    population_size: int = 4
    mutation_scale: float = 0.1
    clamp_min: float = 1e-6
    clamp_max: float = 2.0
    seed: int = 0


class EvolutionaryOptimizer:
    """Simple mutation-and-select optimizer over the shared credit surface."""

    def __init__(self, cfg: EvolutionaryConfig | None = None):
        self.cfg = cfg or EvolutionaryConfig()
        self.rng = random.Random(self.cfg.seed)
        self.n_generations = 0

    def spawn_population(self, genome: DevGenome) -> list[DevGenome]:
        """Spawn a small mutated population from one parent genome."""
        population = []
        for _ in range(self.cfg.population_size):
            child = copy.deepcopy(genome)
            params = child.credit_parameter_dict()
            mutated = {}
            for k, v in params.items():
                delta = self.rng.gauss(0.0, self.cfg.mutation_scale * max(abs(v), 0.1))
                nv = v + delta
                nv = min(self.cfg.clamp_max, max(self.cfg.clamp_min, nv))
                mutated[k] = nv
            child.set_credit_parameter_dict(mutated)
            population.append(child)
        return population

    def select(self, population: list[DevGenome], fitnesses: list[float]) -> DevGenome:
        """Select the highest-fitness child as the next parent genome."""
        if not population:
            raise ValueError("population must be non-empty")
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        self.n_generations += 1
        return copy.deepcopy(population[best_idx])

    def telemetry(self) -> dict:
        return {
            "optimizer_arm": "evolutionary",
            "outer_generations": self.n_generations,
            "population_size": self.cfg.population_size,
        }
