"""
First closed-loop grounding world for Developmental Birth R4.

Not cue→one-of-32 classification. Dense body consequences; organism-owned valence.
Teacher uses the same body actuator, sensory consequence, and proprioceptive channels.
Runner behavioral score never enters the learning path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.body.physics import BodyConfig, BodyStepResult, GenericBody
from three_memory.dev1.development.generative_genome import GenerativeGenome, SYNERGY_REPORT_NAMES
from three_memory.dev1.interfaces import OrganismObservation


@dataclass
class WorldEpisodeResult:
    steps: int
    mean_behavioral_score: float
    correct_fraction: float
    mean_organism_valence: float
    teacher_demos: int


class ClosedLoopGroundingWorld:
    """
    Task-agnostic grounded interaction environment.

    Goal structure (runner-side only): keep body near comfortable nest (origin).
    Correctness is recorded for metrics; never supplied to plasticity.
    """

    # Report names only — never passed into organism as labels
    SYNERGY_NAMES = SYNERGY_REPORT_NAMES

    def __init__(
        self,
        generative: GenerativeGenome,
        *,
        world_seed: str = "r4_ground_0",
        device: torch.device | None = None,
        episode_ticks: int = 16,
    ):
        self.generative = generative
        self.world_seed = str(world_seed)
        self.device = device or torch.device("cpu")
        self.episode_ticks = int(episode_ticks)
        seed_int = int(hashlib.sha256(self.world_seed.encode()).hexdigest()[:8], 16) % (2**31)
        self.body = GenericBody(
            BodyConfig(
                n_motor_channels=generative.n_motor_channels,
                n_synergies=generative.n_synergies,
                sensory_dim=generative.sensory_dim,
                interoceptive_dim=generative.interoceptive_dim,
                seed=seed_int,
            ),
            device=self.device,
        )
        self._tick = 0
        self._teacher_perm: dict[int, int] | None = None

    def reset_episode(self, episode_index: int = 0) -> BodyStepResult:
        seed = int(hashlib.sha256(f"{self.world_seed}:{episode_index}".encode()).hexdigest()[:8], 16) % (
            2**31
        )
        self.body.reset(seed=seed)
        # Idle step for initial observation
        zeros = torch.zeros(self.generative.n_motor_channels, device=self.device)
        return self.body.step(zeros)

    def set_teacher_permutation(self, enabled: bool, seed: int = 0) -> None:
        if not enabled:
            self._teacher_perm = None
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 77)
        perm = torch.randperm(self.generative.n_motor_channels, generator=gen).tolist()
        self._teacher_perm = {i: int(perm[i]) for i in range(len(perm))}

    def set_motor_permutation(self, enabled: bool, seed: int = 0) -> None:
        if not enabled:
            self.body.motor_permutation = None
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 55)
        self.body.motor_permutation = torch.randperm(
            self.generative.n_motor_channels, generator=gen
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
        """
        Build organism observation.

        Reward field is left at 0.0 — organism valence circuit owns reinforcement
        from interoceptive_state. Runner must not put behavioral correctness here.
        """
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
        motor_scores: Any,
        *,
        teacher_channel: int | None = None,
    ) -> BodyStepResult:
        scores = torch.as_tensor(motor_scores, device=self.device, dtype=torch.float32).view(-1)
        demo = teacher_channel
        if demo is not None and self._teacher_perm is not None:
            demo = self._teacher_perm.get(int(demo), int(demo))
        return self.body.step(scores, teacher_channel=demo)

    def suggested_teacher_channel(self, step: BodyStepResult) -> int:
        """
        Runner-side demonstration choice for teaching path (not a neural target).

        Prefer approach channels when far from nest; wait when already comfortable.
        """
        dist = float(step.body_state.position.norm().item())
        width = self.generative.n_motor_channels // self.generative.n_synergies
        if dist > self.body.config.comfort_target_radius:
            # approach block
            return 0
        return 3 * width  # wait block first channel

    def world_hash(self) -> str:
        return hashlib.sha256(
            f"{self.world_seed}|{self.body.physics_hash()}|{self.episode_ticks}".encode()
        ).hexdigest()
