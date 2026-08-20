"""
Batched ES with lexicographic learning-effect-aware selection (Reference Birth R2).

Fitness order (descending):
  1. accuracy / reward improvement
  2. signed rewarded-margin improvement  g_t = s_t (m_after - m_before)
  3. retention
  4. seeded neutral tie-break (not permanent parent priority)

Raw update norm is never a fitness term.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from experiments.dev1.reference_birth_r1_outer import (
    SURFACE_BOUNDS,
    SURFACE_KEYS,
    SurfaceIndividual as _R1SurfaceIndividual,
    clamp_surface,
    default_surface,
    extremes_for_sensitivity,
    genome_from_surface,
    mutate_surface,
    parameter_movement,
    surface_diversity,
)

__all__ = [
    "SURFACE_BOUNDS",
    "SURFACE_KEYS",
    "LexicographicFitness",
    "SurfaceIndividual",
    "clamp_surface",
    "compare_fitness",
    "default_surface",
    "extremes_for_sensitivity",
    "genome_from_surface",
    "mutate_surface",
    "parameter_movement",
    "run_batched_es_lexicographic",
    "select_best",
    "signed_margin_improvement",
    "surface_diversity",
]


@dataclass(frozen=True, order=True)
class LexicographicFitness:
    """Ordered for max-heap / reverse sort: higher is better on each field."""

    accuracy: float
    signed_margin_improvement: float
    retention: float
    tie_break: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.accuracy, self.signed_margin_improvement, self.retention, self.tie_break)


@dataclass
class SurfaceIndividual:
    surface: dict[str, float]
    fitness_key: LexicographicFitness = field(
        default_factory=lambda: LexicographicFitness(
            float("-inf"), float("-inf"), float("-inf"), 0.0
        )
    )
    phenotype_hash: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        """Primary accuracy for display only — selection uses fitness_key."""
        return float(self.fitness_key.accuracy)


def signed_margin_improvement(
    pre_margin: float,
    post_margin: float,
    outcome_sign: int | float,
) -> float:
    """g_t = s_t (m_after - m_before); s_t = +1 positive outcome, -1 negative."""
    s = float(outcome_sign)
    if s == 0.0:
        return 0.0
    return s * (float(post_margin) - float(pre_margin))


def compare_fitness(a: LexicographicFitness, b: LexicographicFitness) -> int:
    """Return +1 if a > b, -1 if a < b, 0 if equal."""
    ta, tb = a.as_tuple(), b.as_tuple()
    if ta > tb:
        return 1
    if ta < tb:
        return -1
    return 0


def select_best(individuals: list[SurfaceIndividual]) -> SurfaceIndividual:
    """Lexicographic max; ties already broken by seeded tie_break in the key."""
    if not individuals:
        raise ValueError("empty population")
    return max(individuals, key=lambda ind: ind.fitness_key.as_tuple())


def run_batched_es_lexicographic(
    evaluate_fn: Callable[[dict[str, float]], SurfaceIndividual],
    *,
    generations: int = 8,
    population_size: int = 8,
    mutation_scale: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Batched ES with learning-effect-aware lexicographic selection.

    Parent is re-evaluated each generation but has no permanent tie priority:
    each individual receives a seeded neutral tie_break.
    """
    rng = random.Random(seed)
    parent = default_surface()
    history: list[dict[str, Any]] = []
    best = SurfaceIndividual(
        surface=dict(parent),
        fitness_key=LexicographicFitness(float("-inf"), float("-inf"), float("-inf"), 0.0),
    )

    for gen in range(generations):
        pop_surfaces = [mutate_surface(parent, mutation_scale, rng) for _ in range(population_size)]
        pop_surfaces[0] = dict(parent)
        individuals: list[SurfaceIndividual] = []
        for idx, surf in enumerate(pop_surfaces):
            ind = evaluate_fn(surf)
            ind.surface = clamp_surface(surf)
            # Neutral seeded tie-break — not parent-index priority.
            tie = rng.random()
            fk = ind.fitness_key
            ind.fitness_key = LexicographicFitness(
                fk.accuracy,
                fk.signed_margin_improvement,
                fk.retention,
                tie,
            )
            ind.metrics = {
                **ind.metrics,
                "tie_break": tie,
                "is_parent_slot": idx == 0,
                "selection_uses_update_norm": False,
            }
            individuals.append(ind)

        selected = select_best(individuals)
        ranked = sorted(individuals, key=lambda x: x.fitness_key.as_tuple(), reverse=True)
        diversity = surface_diversity([i.surface for i in individuals])
        move = parameter_movement(parent, selected.surface)
        history.append({
            "generation": gen,
            "best_fitness_key": selected.fitness_key.as_tuple(),
            "best_accuracy": selected.fitness_key.accuracy,
            "parent_slot_selected": bool(selected.metrics.get("is_parent_slot")),
            "population_diversity": diversity,
            "parameter_movement": move,
            "phenotype_hash": selected.phenotype_hash,
            "best_surface": selected.surface,
            "population_fitness_keys": [i.fitness_key.as_tuple() for i in ranked],
            "metrics": selected.metrics,
        })
        parent = dict(selected.surface)
        if compare_fitness(selected.fitness_key, best.fitness_key) >= 0:
            best = selected

    return {
        "optimizer": "batched_evolution_strategies_lexicographic",
        "generations": generations,
        "population_size": population_size,
        "mutation_scale": mutation_scale,
        "seed": seed,
        "best_surface": best.surface,
        "best_fitness_key": best.fitness_key.as_tuple(),
        "best_fitness": best.fitness_key.accuracy,
        "best_phenotype_hash": best.phenotype_hash,
        "history": history,
        "outer_updates_executed": generations,
        "fitness_forbids_raw_update_norm": True,
        "tie_break": "seeded_neutral",
    }
