"""Nursery Body v2 closed-loop world and reachability contract."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.nursery_v2.physics import BodyConfig, BodyStepResult, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import N_MOTOR_CHANNELS, N_SYNERGIES, SYNERGY_REPORT_NAMES


@dataclass
class NurseryGenomeView:
    """Minimal genome-shaped view for nursery worlds (no organism construction)."""

    n_motor_channels: int = N_MOTOR_CHANNELS
    n_synergies: int = N_SYNERGIES
    sensory_dim: int = 48
    interoceptive_dim: int = 4


class NurseryWorldV2:
    """Task-agnostic nursery world. Runner may score nest comfort; never for plasticity."""

    SYNERGY_NAMES = SYNERGY_REPORT_NAMES

    def __init__(
        self,
        *,
        world_seed: str = "nursery_v2_0",
        device: torch.device | None = None,
        episode_ticks: int = 16,
        config: BodyConfig | None = None,
    ):
        self.world_seed = str(world_seed)
        self.device = device or torch.device("cpu")
        self.episode_ticks = int(episode_ticks)
        self.genome = NurseryGenomeView()
        seed_int = int(hashlib.sha256(self.world_seed.encode()).hexdigest()[:8], 16) % (2**31)
        cfg = config or BodyConfig()
        cfg.seed = seed_int
        self.body = NurseryBodyV2(cfg, device=self.device)

    def reset_episode(self, episode_index: int = 0) -> BodyStepResult:
        seed = int(
            hashlib.sha256(f"{self.world_seed}:{episode_index}".encode()).hexdigest()[:8], 16
        ) % (2**31)
        self.body.reset(seed=seed)
        zeros = torch.zeros(self.genome.n_motor_channels, device=self.device)
        return self.body.step(zeros)

    def apply_action(self, motor: torch.Tensor) -> BodyStepResult:
        return self.body.step(motor)

    def set_motor_permutation(self, enabled: bool, seed: int = 0) -> None:
        if not enabled:
            self.body.motor_permutation = None
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 55)
        self.body.motor_permutation = torch.randperm(
            self.body.config.n_motor_channels, generator=gen
        ).to(self.device)


def max_useful_travel(config: BodyConfig, episode_ticks: int) -> float:
    """Straight-line travel if a unit forward synergy is held every tick."""
    return float(config.step_scale) * float(episode_ticks)


def reachability_chi(initial_distance: float, config: BodyConfig, episode_ticks: int) -> float:
    """
    χ = (initial_distance - comfort_radius) / max_useful_travel.

    χ <= 1 means the nest is reachable in a straight line under unit forward drive
    (orientation assumed correct). χ > 1 is unreachable even under ideal heading.
    """
    need = max(0.0, float(initial_distance) - float(config.comfort_target_radius))
    travel = max_useful_travel(config, episode_ticks)
    if travel <= 1e-12:
        return float("inf")
    return need / travel


def analytic_reachability_report(
    world_seed: str,
    *,
    n_episodes: int,
    episode_ticks: int,
    safety_margin: float = 0.85,
    min_fraction_reachable: float = 0.95,
    config: BodyConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Fraction of starts with χ <= safety_margin (margin < 1 leaves path slack).
    """
    cfg = config or BodyConfig()
    world = NurseryWorldV2(
        world_seed=world_seed + ":reach",
        device=device or torch.device("cpu"),
        episode_ticks=episode_ticks,
        config=cfg,
    )
    chis: list[float] = []
    d0s: list[float] = []
    for ep in range(n_episodes):
        step = world.reset_episode(ep)
        d0 = float(step.body_state.position.norm().item())
        d0s.append(d0)
        chis.append(reachability_chi(d0, world.body.config, episode_ticks))
    reachable = [c <= safety_margin for c in chis]
    frac = sum(reachable) / max(1, len(reachable))
    return {
        "n_episodes": n_episodes,
        "episode_ticks": episode_ticks,
        "safety_margin": safety_margin,
        "min_fraction_reachable": min_fraction_reachable,
        "max_useful_travel": max_useful_travel(world.body.config, episode_ticks),
        "comfort_target_radius": world.body.config.comfort_target_radius,
        "mean_initial_distance": sum(d0s) / max(1, len(d0s)),
        "chi_mean": sum(chis) / max(1, len(chis)),
        "chi_p95": sorted(chis)[max(0, int(math.ceil(0.95 * len(chis)) - 1))],
        "fraction_reachable": frac,
        "pass": frac >= min_fraction_reachable,
        "chi_values_sample": chis[:8],
    }
