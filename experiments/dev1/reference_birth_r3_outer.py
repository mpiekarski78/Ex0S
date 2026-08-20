"""
Outer ES over inherited learning-signal-generator parameters (Reference Birth R3).

Search surface is the flattened LSG weight vector (generic learning machinery).
Lexicographic fitness from R2 is reused. Budget class matches R1/R2 (8×8).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

from experiments.dev1.reference_birth_r2_outer import (
    LexicographicFitness,
    SurfaceIndividual,
    compare_fitness,
    select_best,
    signed_margin_improvement,
)
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.plasticity.eprop.signal_generator import (
    LSG_HIDDEN,
    default_lsg_vector,
    lsg_param_count,
)


def lsg_dims(genome: DevGenome | None = None) -> tuple[int, int, int]:
    g = genome or DevGenome.default()
    return g.n_motor_channels, g.action_ctx.n_units, LSG_HIDDEN


def default_lsg_surface(seed: int = 0) -> list[float]:
    n_motor, n_post, hidden = lsg_dims()
    return default_lsg_vector(n_motor, n_post, seed=seed, hidden=hidden)


def clamp_lsg_vector(vec: Sequence[float], clip: float = 2.0) -> list[float]:
    return [float(max(-clip, min(clip, v))) for v in vec]


def mutate_lsg_vector(
    parent: Sequence[float],
    scale: float,
    rng: random.Random,
    clip: float = 2.0,
) -> list[float]:
    child = []
    for v in parent:
        child.append(v + rng.gauss(0.0, scale))
    return clamp_lsg_vector(child, clip=clip)


def lsg_parameter_movement(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dist = sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))
    return float(math.sqrt(dist / len(a)))


def genome_from_lsg(
    arm_family: str,
    lsg_vector: Sequence[float],
    *,
    seed: int = 0,
    plasticity_surface: dict[str, float] | None = None,
) -> DevGenome:
    genome = DevGenome.default()
    genome.seed = seed
    genome.plasticity_family = arm_family
    genome.lsg_param_vector = list(lsg_vector)
    if plasticity_surface is not None:
        genome.apply_r1_inherited_surface(plasticity_surface)
    # Stronger default LR for LSG candidate so within-life updates are measurable.
    if arm_family == "inherited_learning_signal_generator":
        if plasticity_surface is None or "log_actor_learning_rate" not in plasticity_surface:
            genome.plasticity.learning_rate = max(genome.plasticity.learning_rate, 3e-3)
            genome.plasticity.projection_scale = max(genome.plasticity.projection_scale, 5.0)
    return genome


def run_batched_es_lsg(
    evaluate_fn: Callable[[list[float]], SurfaceIndividual],
    *,
    generations: int = 8,
    population_size: int = 8,
    mutation_scale: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    parent = default_lsg_surface(seed=seed)
    expected = lsg_param_count(*lsg_dims()[:2])
    assert len(parent) == expected, (len(parent), expected)
    history: list[dict[str, Any]] = []
    best = SurfaceIndividual(
        surface={"lsg_norm": 0.0},
        fitness_key=LexicographicFitness(float("-inf"), float("-inf"), float("-inf"), 0.0),
        metrics={"lsg_vector": list(parent)},
    )

    for gen in range(generations):
        pop = [mutate_lsg_vector(parent, mutation_scale, rng) for _ in range(population_size)]
        pop[0] = list(parent)
        individuals: list[SurfaceIndividual] = []
        for idx, vec in enumerate(pop):
            ind = evaluate_fn(vec)
            tie = rng.random()
            fk = ind.fitness_key
            ind.fitness_key = LexicographicFitness(
                fk.accuracy, fk.signed_margin_improvement, fk.retention, tie
            )
            ind.metrics = {
                **ind.metrics,
                "tie_break": tie,
                "is_parent_slot": idx == 0,
                "lsg_vector": list(vec),
                "lsg_norm": float(math.sqrt(sum(v * v for v in vec) / max(1, len(vec)))),
            }
            ind.surface = {"lsg_norm": ind.metrics["lsg_norm"]}
            individuals.append(ind)

        selected = select_best(individuals)
        ranked = sorted(individuals, key=lambda x: x.fitness_key.as_tuple(), reverse=True)
        move = lsg_parameter_movement(parent, selected.metrics["lsg_vector"])
        history.append({
            "generation": gen,
            "best_fitness_key": selected.fitness_key.as_tuple(),
            "best_accuracy": selected.fitness_key.accuracy,
            "parent_slot_selected": bool(selected.metrics.get("is_parent_slot")),
            "parameter_movement": move,
            "phenotype_hash": selected.phenotype_hash,
            "population_fitness_keys": [i.fitness_key.as_tuple() for i in ranked],
            "metrics": {k: v for k, v in selected.metrics.items() if k != "lsg_vector"},
            "best_lsg_norm": selected.metrics.get("lsg_norm"),
        })
        parent = list(selected.metrics["lsg_vector"])
        if compare_fitness(selected.fitness_key, best.fitness_key) >= 0:
            best = selected
            best.metrics = dict(selected.metrics)

    return {
        "optimizer": "batched_evolution_strategies_lsg",
        "generations": generations,
        "population_size": population_size,
        "mutation_scale": mutation_scale,
        "seed": seed,
        "lsg_param_count": expected,
        "best_lsg_vector": best.metrics.get("lsg_vector"),
        "best_fitness_key": best.fitness_key.as_tuple(),
        "best_fitness": best.fitness_key.accuracy,
        "best_phenotype_hash": best.phenotype_hash,
        "history": history,
        "outer_updates_executed": generations,
        "tie_break": "seeded_neutral",
        "inherits_only_generic_learning_machinery": True,
    }
