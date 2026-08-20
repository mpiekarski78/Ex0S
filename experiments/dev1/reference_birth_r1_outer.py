"""
Batched evolution strategies over the Reference Birth R1 inherited surface.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from three_memory.dev1.genome import DevGenome

SURFACE_KEYS = [
    "log_actor_learning_rate",
    "log_critic_learning_rate",
    "eligibility_decay",
    "td_discount_gamma",
    "learning_signal_projection_scale",
    "entropy_exploration_temperature",
    "update_clip_scale",
]

# Search bounds in surface space (decoded bounds applied in genome.apply_r1_inherited_surface)
SURFACE_BOUNDS = {
    "log_actor_learning_rate": (-9.0, -1.0),       # ~1e-4 .. 0.37
    "log_critic_learning_rate": (-9.0, -1.0),
    "eligibility_decay": (0.5, 0.99),
    "td_discount_gamma": (0.5, 0.99),
    "learning_signal_projection_scale": (0.1, 50.0),
    "entropy_exploration_temperature": (0.25, 4.0),
    "update_clip_scale": (0.01, 1.0),
}


@dataclass
class SurfaceIndividual:
    surface: dict[str, float]
    fitness: float = float("-inf")
    phenotype_hash: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def default_surface() -> dict[str, float]:
    return DevGenome.default().r1_inherited_surface_dict()


def clamp_surface(surface: dict[str, float]) -> dict[str, float]:
    out = {}
    for k in SURFACE_KEYS:
        lo, hi = SURFACE_BOUNDS[k]
        out[k] = float(max(lo, min(hi, surface[k])))
    return out


def mutate_surface(parent: dict[str, float], scale: float, rng: random.Random) -> dict[str, float]:
    child = {}
    for k in SURFACE_KEYS:
        lo, hi = SURFACE_BOUNDS[k]
        span = hi - lo
        child[k] = parent[k] + rng.gauss(0.0, scale * span)
    return clamp_surface(child)


def surface_diversity(population: list[dict[str, float]]) -> float:
    if len(population) < 2:
        return 0.0
    keys = SURFACE_KEYS
    means = {k: sum(p[k] for p in population) / len(population) for k in keys}
    var = 0.0
    for p in population:
        for k in keys:
            lo, hi = SURFACE_BOUNDS[k]
            span = max(hi - lo, 1e-8)
            var += ((p[k] - means[k]) / span) ** 2
    return float(math.sqrt(var / (len(population) * len(keys))))


def parameter_movement(a: dict[str, float], b: dict[str, float]) -> float:
    dist = 0.0
    for k in SURFACE_KEYS:
        lo, hi = SURFACE_BOUNDS[k]
        span = max(hi - lo, 1e-8)
        dist += ((a[k] - b[k]) / span) ** 2
    return float(math.sqrt(dist / len(SURFACE_KEYS)))


def extremes_for_sensitivity() -> list[tuple[str, dict[str, float]]]:
    """Inherited-parameter extremes for search-surface sensitivity preflight."""
    base = default_surface()
    rows: list[tuple[str, dict[str, float]]] = [("baseline", dict(base))]
    for key in SURFACE_KEYS:
        lo, hi = SURFACE_BOUNDS[key]
        low = dict(base)
        high = dict(base)
        low[key] = lo
        high[key] = hi
        rows.append((f"{key}_low", clamp_surface(low)))
        rows.append((f"{key}_high", clamp_surface(high)))
    return rows


def genome_from_surface(arm_family: str, surface: dict[str, float], seed: int = 0) -> DevGenome:
    genome = DevGenome.default()
    genome.seed = seed
    genome.plasticity_family = arm_family
    genome.apply_r1_inherited_surface(surface)
    return genome


def run_batched_es(
    evaluate_fn: Callable[[dict[str, float]], SurfaceIndividual],
    *,
    generations: int = 8,
    population_size: int = 8,
    mutation_scale: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Transparent batched evolution strategies over the R1 inherited surface.

    evaluate_fn maps a surface dict -> SurfaceIndividual with fitness filled.
    """
    rng = random.Random(seed)
    parent = default_surface()
    history: list[dict[str, Any]] = []
    best = SurfaceIndividual(surface=dict(parent), fitness=float("-inf"))

    for gen in range(generations):
        pop_surfaces = [mutate_surface(parent, mutation_scale, rng) for _ in range(population_size)]
        # Include parent for elitism
        pop_surfaces[0] = dict(parent)
        individuals: list[SurfaceIndividual] = []
        for surf in pop_surfaces:
            ind = evaluate_fn(surf)
            ind.surface = clamp_surface(surf)
            individuals.append(ind)
        individuals.sort(key=lambda x: x.fitness, reverse=True)
        selected = individuals[0]
        diversity = surface_diversity([i.surface for i in individuals])
        move = parameter_movement(parent, selected.surface)
        history.append({
            "generation": gen,
            "best_fitness": selected.fitness,
            "parent_selection_index": 0,
            "population_diversity": diversity,
            "parameter_movement": move,
            "phenotype_hash": selected.phenotype_hash,
            "best_surface": selected.surface,
            "population_fitness": [i.fitness for i in individuals],
            "metrics": selected.metrics,
        })
        parent = dict(selected.surface)
        if selected.fitness >= best.fitness:
            best = selected

    return {
        "optimizer": "batched_evolution_strategies",
        "generations": generations,
        "population_size": population_size,
        "mutation_scale": mutation_scale,
        "seed": seed,
        "best_surface": best.surface,
        "best_fitness": best.fitness,
        "best_phenotype_hash": best.phenotype_hash,
        "history": history,
        "outer_updates_executed": generations,
    }
