"""Nursery Body v2 closed-loop world and reachability contract."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.interfaces import OrganismObservation
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
    """
    Task-agnostic nursery world. Runner may score nest comfort; never for plasticity.

    Egocentric teacher: forward when far from nest, rotate when already comfortable.
    """

    SYNERGY_NAMES = SYNERGY_REPORT_NAMES

    def __init__(
        self,
        *,
        world_seed: str = "nursery_v2_0",
        device: torch.device | None = None,
        episode_ticks: int = 16,
        config: BodyConfig | None = None,
        generative: Any | None = None,
    ):
        self.world_seed = str(world_seed)
        self.device = device or torch.device("cpu")
        self.episode_ticks = int(episode_ticks)
        if generative is not None:
            self.genome = generative
            cfg = config or BodyConfig(
                n_motor_channels=int(generative.n_motor_channels),
                n_synergies=int(generative.n_synergies),
                sensory_dim=int(generative.sensory_dim),
                interoceptive_dim=int(generative.interoceptive_dim),
            )
        else:
            self.genome = NurseryGenomeView()
            cfg = config or BodyConfig()
        seed_int = int(hashlib.sha256(self.world_seed.encode()).hexdigest()[:8], 16) % (2**31)
        cfg.seed = seed_int
        self.body = NurseryBodyV2(cfg, device=self.device)
        self._teacher_perm: dict[int, int] | None = None

    def reset_episode(self, episode_index: int = 0) -> BodyStepResult:
        seed = int(
            hashlib.sha256(f"{self.world_seed}:{episode_index}".encode()).hexdigest()[:8], 16
        ) % (2**31)
        self.body.reset(seed=seed)
        zeros = torch.zeros(int(self.genome.n_motor_channels), device=self.device)
        return self.body.step(zeros)

    def set_teacher_permutation(self, enabled: bool, seed: int = 0) -> None:
        if not enabled:
            self._teacher_perm = None
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 77)
        n = int(self.genome.n_motor_channels)
        perm = torch.randperm(n, generator=gen).tolist()
        self._teacher_perm = {i: int(perm[i]) for i in range(len(perm))}

    def set_motor_permutation(self, enabled: bool, seed: int = 0) -> None:
        if not enabled:
            self.body.motor_permutation = None
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 55)
        self.body.motor_permutation = torch.randperm(
            self.body.config.n_motor_channels, generator=gen
        ).to(self.device)

    def set_proprio_permutation(self, enabled: bool, seed: int = 0) -> None:
        if not enabled:
            self.body.proprio_permutation = None
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 66)
        self.body.proprio_permutation = torch.randperm(
            self.body.config.proprioceptive_dim, generator=gen
        ).to(self.device)

    def set_open_loop(self, enabled: bool) -> None:
        self.body.open_loop = bool(enabled)

    def observation_from_step(
        self,
        step: BodyStepResult,
        *,
        temporal_context: float = 0.0,
        observed_motor_event: int | None = None,
        teaching_signal: float | None = None,
    ) -> OrganismObservation:
        return OrganismObservation(
            sensory_vector=step.sensory_vector,
            temporal_context=temporal_context,
            reward=0.0,
            is_terminal=False,
            observed_motor_event=observed_motor_event,
            teaching_signal=teaching_signal,
            interoceptive_state=step.interoceptive_state,
            proprioceptive_vector=step.proprioceptive_vector,
        )

    def apply_action(
        self,
        motor: torch.Tensor,
        *,
        teacher_channel: int | None = None,
    ) -> BodyStepResult:
        scores = torch.as_tensor(motor, device=self.device, dtype=torch.float32).view(-1)
        demo = teacher_channel
        if demo is not None and self._teacher_perm is not None:
            demo = self._teacher_perm.get(int(demo), int(demo))
        return self.body.step(scores, teacher_channel=demo)

    def suggested_teacher_channel(self, step: BodyStepResult) -> int:
        """Egocentric demo: forward when far; rotate_left when already in comfort."""
        dist = float(step.body_state.position.norm().item())
        width = int(self.genome.n_motor_channels) // int(self.genome.n_synergies)
        if dist > self.body.config.comfort_target_radius:
            return 0  # forward block first channel
        return 2 * width  # rotate_left block first channel

    def world_hash(self) -> str:
        return hashlib.sha256(
            f"{self.world_seed}|{self.body.physics_hash()}|{self.episode_ticks}|nursery_v2".encode()
        ).hexdigest()


def max_useful_travel(config: BodyConfig, episode_ticks: int) -> float:
    """Straight-line travel if a unit forward synergy is held every tick."""
    return float(config.step_scale) * float(episode_ticks)


def reachability_chi(initial_distance: float, config: BodyConfig, episode_ticks: int) -> float:
    """
    χ = (initial_distance - comfort_radius) / max_useful_travel.

    Analytically reachable: χ ≤ 1.
    Safety rule used in certification: χ ≤ 0.85.
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
    Fraction of starts with χ ≤ safety_margin (default 0.85).

    χ ≤ 1 is analytical reachability; χ ≤ 0.85 is the frozen safety rule.
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
    reachable_strict = [c <= 1.0 for c in chis]
    reachable_safe = [c <= safety_margin for c in chis]
    frac_safe = sum(reachable_safe) / max(1, len(reachable_safe))
    return {
        "n_episodes": n_episodes,
        "episode_ticks": episode_ticks,
        "safety_rule": "chi <= 0.85",
        "analytical_reachable_rule": "chi <= 1",
        "safety_margin": safety_margin,
        "min_fraction_reachable": min_fraction_reachable,
        "max_useful_travel": max_useful_travel(world.body.config, episode_ticks),
        "comfort_target_radius": world.body.config.comfort_target_radius,
        "mean_initial_distance": sum(d0s) / max(1, len(d0s)),
        "chi_mean": sum(chis) / max(1, len(chis)),
        "chi_p95": sorted(chis)[max(0, int(math.ceil(0.95 * len(chis)) - 1))],
        "fraction_chi_le_1": sum(reachable_strict) / max(1, len(reachable_strict)),
        "fraction_reachable": frac_safe,
        "pass": frac_safe >= min_fraction_reachable,
        "chi_values_sample": chis[:8],
    }
