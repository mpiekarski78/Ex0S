"""
Matched outer-loop budgets for Developmental Birth R4 credit columns.

Both fixed-credit and LSG columns receive matched outer-loop world exposure
and evaluation budgets. The LSG column cannot receive more developmental lives
merely because it has more parameters.

Numeric thresholds/seeds for scored search belong in a later prereg lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from experiments.dev1.developmental_birth_r4_life import evaluate_r4_life
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.plasticity.eprop.signal_generator import (
    default_lsg_vector,
    lsg_param_count,
)


@dataclass
class MatchedOuterBudget:
    """Equal exposure/evaluation across credit columns."""

    population: int = 4
    generations: int = 2
    lives_per_individual: int = 1
    n_episodes: int = 4
    episode_ticks: int = 8
    mutation_sigma: float = 0.1


def default_fixed_credit_surface(seed: int = 0) -> dict[str, float]:
    """Small matched surface for fixed-credit column (not R2/R3 cell substitution)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 3)
    noise = torch.randn(3, generator=gen) * 0.01
    return {
        "log_actor_learning_rate": float(-8.0 + noise[0]),
        "log_critic_learning_rate": float(-8.0 + noise[1]),
        "eligibility_decay": float(0.9 + 0.01 * noise[2]),
    }


def default_lsg_surface_for_r4(generative: GenerativeGenome, seed: int = 0) -> list[float]:
    return default_lsg_vector(
        generative.n_motor_channels,
        generative.action_units,
        seed=seed,
    )


def mutate_vector(vec: Sequence[float], sigma: float, seed: int) -> list[float]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    t = torch.tensor(list(vec), dtype=torch.float32)
    noise = torch.randn(t.shape, generator=gen) * float(sigma)
    return (t + noise).tolist()


def run_matched_es_smoke(
    world_seed: str,
    budget: MatchedOuterBudget | None = None,
    *,
    embryonic_seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """
    Smoke outer loop with matched lives/episodes for both credit columns.
    Not a scored R4 run — preregistration freezes numeric thresholds later.
    """
    budget = budget or MatchedOuterBudget()
    base = GenerativeGenome.small(embryonic_seed=embryonic_seed)
    results = {"fixed": [], "lsg": [], "budget": budget.__dict__}

    # Fixed column
    surf = default_fixed_credit_surface(seed=embryonic_seed)
    for gen_i in range(budget.generations):
        for ind in range(budget.population):
            g = base.with_credit_family("r2_fixed_eprop_baseline")
            # Apply tiny surface via learning rates
            import math
            g.learning_rate = float(math.exp(surf["log_actor_learning_rate"]))
            g.critic_learning_rate = float(math.exp(surf["log_critic_learning_rate"]))
            g.eligibility_decay = float(surf["eligibility_decay"])
            m = evaluate_r4_life(
                "active_gestation",
                "r2_fixed_eprop_baseline",
                f"{world_seed}:fixed:{gen_i}:{ind}",
                generative=g,
                n_episodes=budget.n_episodes,
                episode_ticks=budget.episode_ticks,
                embryonic_seed=embryonic_seed,
                life_rng_seed=gen_i * 100 + ind,
                device=device,
            )
            results["fixed"].append(m.treatment_accuracy)
            surf = {
                "log_actor_learning_rate": surf["log_actor_learning_rate"]
                + budget.mutation_sigma * (0.5 - (ind % 3) / 2),
                "log_critic_learning_rate": surf["log_critic_learning_rate"],
                "eligibility_decay": min(0.99, max(0.5, surf["eligibility_decay"])),
            }

    # LSG column — same population × generations × lives × episodes
    vec = default_lsg_surface_for_r4(base, seed=embryonic_seed)
    assert len(vec) == lsg_param_count(base.n_motor_channels, base.action_units)
    for gen_i in range(budget.generations):
        for ind in range(budget.population):
            g = base.with_credit_family("inherited_learning_signal_generator")
            g.lsg_param_vector = list(vec)
            m = evaluate_r4_life(
                "active_gestation",
                "inherited_learning_signal_generator",
                f"{world_seed}:lsg:{gen_i}:{ind}",
                generative=g,
                n_episodes=budget.n_episodes,
                episode_ticks=budget.episode_ticks,
                embryonic_seed=embryonic_seed,
                life_rng_seed=gen_i * 100 + ind,
                device=device,
            )
            results["lsg"].append(m.treatment_accuracy)
            vec = mutate_vector(vec, budget.mutation_sigma, seed=gen_i * 1000 + ind)

    results["fixed_lives"] = len(results["fixed"])
    results["lsg_lives"] = len(results["lsg"])
    assert results["fixed_lives"] == results["lsg_lives"]
    results["matched"] = results["fixed_lives"] == results["lsg_lives"]
    return results
