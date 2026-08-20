"""
Generic body physics, drives, proprioception, and efference-copy channels.

Body exposes interoceptive state but never an expected action.
Synergy report names live only in runner/report code.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field

import torch

from three_memory.dev1.development.generative_genome import N_MOTOR_CHANNELS, N_SYNERGIES
from three_memory.dev1.development.synergies import channels_to_synergy_activations, synergy_projection_matrix


@dataclass
class BodyState:
    """Task-agnostic body state. No cue names or expected actions."""

    position: torch.Tensor          # (2,)
    orientation: float              # radians scalar as float
    proprioception: torch.Tensor    # (n_proprio,)
    interoception: torch.Tensor     # (n_intero,)
    last_motor: torch.Tensor        # (n_channels,) efference analogue
    tick: int = 0


@dataclass
class BodyStepResult:
    sensory_vector: torch.Tensor    # packed extero + proprio (+ optional efference)
    proprioceptive_vector: torch.Tensor
    interoceptive_state: torch.Tensor
    synergy_activations: torch.Tensor  # runner-side only; not fed as labels
    body_state: BodyState
    # Runner may record behavioral correctness separately; never for plasticity.
    behavioral_correct: bool = False
    behavioral_score: float = 0.0


@dataclass
class BodyConfig:
    n_motor_channels: int = N_MOTOR_CHANNELS
    n_synergies: int = N_SYNERGIES
    sensory_dim: int = 48
    interoceptive_dim: int = 4
    proprioceptive_dim: int = 8
    arena_radius: float = 2.0
    drive_decay: float = 0.02
    comfort_target_radius: float = 0.35
    seed: int = 0


class GenericBody:
    """
    Dense, predictable sensorimotor body for grounded closed-loop interaction.

    Synergies (report names only): approach, withdraw, orient, wait —
    implemented as contiguous motor-channel blocks via fixed projection.
    """

    def __init__(self, config: BodyConfig | None = None, device: torch.device | None = None):
        self.config = config or BodyConfig()
        self.device = device or torch.device("cpu")
        self.projection = synergy_projection_matrix(
            self.config.n_motor_channels,
            self.config.n_synergies,
            device=self.device,
        )
        self.state = self._initial_state()
        # Optional intervention hooks (applied by experiment runner)
        self.motor_permutation: torch.Tensor | None = None
        self.proprio_permutation: torch.Tensor | None = None
        self.open_loop: bool = False
        self._open_loop_actions: list[torch.Tensor] = []

    def _initial_state(self) -> BodyState:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.config.seed) + 91)
        pos = torch.randn(2, generator=gen) * 0.8
        proprio = torch.zeros(self.config.proprioceptive_dim)
        intero = torch.full((self.config.interoceptive_dim,), 0.7)
        motor = torch.zeros(self.config.n_motor_channels)
        return BodyState(
            position=pos.to(self.device),
            orientation=0.0,
            proprioception=proprio.to(self.device),
            interoception=intero.to(self.device),
            last_motor=motor.to(self.device),
            tick=0,
        )

    def reset(self, seed: int | None = None) -> BodyState:
        if seed is not None:
            self.config.seed = int(seed)
        self.state = self._initial_state()
        self._open_loop_actions.clear()
        return self.state

    def physics_hash(self) -> str:
        src = inspect.getsource(GenericBody.step)
        cfg = (
            f"{self.config.n_motor_channels}|{self.config.n_synergies}|"
            f"{self.config.arena_radius}|{self.config.drive_decay}|"
            f"{self.config.comfort_target_radius}"
        )
        return hashlib.sha256((src + cfg).encode()).hexdigest()

    def _apply_motor_perm(self, motor: torch.Tensor) -> torch.Tensor:
        if self.motor_permutation is None:
            return motor
        return motor[self.motor_permutation]

    def _apply_proprio_perm(self, proprio: torch.Tensor) -> torch.Tensor:
        if self.proprio_permutation is None:
            return proprio
        return proprio[self.proprio_permutation]

    def step(
        self,
        motor_scores: torch.Tensor,
        *,
        teacher_channel: int | None = None,
    ) -> BodyStepResult:
        """
        Apply opaque motor competition scores (or a demonstrated channel) to body.

        Dense predictable sensory/proprioceptive consequences; continuous
        interoceptive drive. Never returns an expected action.
        """
        cfg = self.config
        motor = motor_scores.detach().float().to(self.device).view(-1)
        if motor.numel() != cfg.n_motor_channels:
            if motor.numel() < cfg.n_motor_channels:
                pad = torch.zeros(cfg.n_motor_channels - motor.numel(), device=self.device)
                motor = torch.cat([motor, pad])
            else:
                motor = motor[: cfg.n_motor_channels]

        if teacher_channel is not None:
            motor = torch.zeros(cfg.n_motor_channels, device=self.device)
            motor[int(teacher_channel) % cfg.n_motor_channels] = 1.0

        motor_body = self._apply_motor_perm(motor)

        if self.open_loop and self._open_loop_actions:
            # Use shuffled previously recorded action for open-loop control
            idx = self.state.tick % len(self._open_loop_actions)
            motor_body = self._open_loop_actions[idx]
        elif not self.open_loop:
            self._open_loop_actions.append(motor_body.detach().clone())

        syn = channels_to_synergy_activations(motor_body, self.projection)
        # Report-order activations: approach, withdraw, orient, wait
        approach, withdraw, orient, wait = [float(x) for x in syn.tolist()]

        # Body dynamics
        step_scale = 0.15
        dx = (approach - withdraw) * step_scale
        dy = torch.sin(torch.tensor(self.state.orientation, device=self.device)) * orient * step_scale
        # wait reduces motion / stabilizes
        motion_gate = max(0.05, 1.0 - 0.5 * wait)
        new_pos = self.state.position.clone()
        new_pos[0] = new_pos[0] + dx * motion_gate
        new_pos[1] = new_pos[1] + float(dy) * motion_gate
        # Soft arena clamp
        r = float(new_pos.norm().item())
        if r > cfg.arena_radius:
            new_pos = new_pos * (cfg.arena_radius / r)

        new_ori = self.state.orientation + float(orient - 0.5 * wait) * 0.2

        # Proprioception: position embedding + orientation + last synergy magnitudes
        proprio = torch.zeros(cfg.proprioceptive_dim, device=self.device)
        proprio[0] = new_pos[0]
        proprio[1] = new_pos[1]
        proprio[2] = torch.cos(torch.tensor(new_ori, device=self.device))
        proprio[3] = torch.sin(torch.tensor(new_ori, device=self.device))
        for i in range(min(4, cfg.n_synergies)):
            if 4 + i < cfg.proprioceptive_dim:
                proprio[4 + i] = syn[i]
        proprio = self._apply_proprio_perm(proprio)

        # Interoception: homeostatic drives — comfort rises near origin (generic nest)
        dist = float(new_pos.norm().item())
        comfort = max(0.0, 1.0 - dist / max(cfg.comfort_target_radius * 3.0, 1e-6))
        energy = float(self.state.interoception[1].item()) if self.state.interoception.numel() > 1 else 0.7
        energy = max(0.0, min(1.0, energy - cfg.drive_decay + 0.03 * wait))
        arousal = min(1.0, abs(approach - withdraw) + abs(orient))
        satiety = comfort
        intero = torch.tensor(
            [comfort, energy, arousal, satiety][: cfg.interoceptive_dim],
            device=self.device,
            dtype=torch.float32,
        )
        if intero.numel() < cfg.interoceptive_dim:
            intero = torch.cat(
                [intero, torch.zeros(cfg.interoceptive_dim - intero.numel(), device=self.device)]
            )

        # Exteroception: radial field + angle (predictable sensory consequences of movement)
        exo_dim = cfg.sensory_dim - cfg.proprioceptive_dim
        exo = torch.zeros(max(0, exo_dim), device=self.device)
        if exo.numel() >= 4:
            exo[0] = new_pos[0]
            exo[1] = new_pos[1]
            exo[2] = dist
            exo[3] = comfort
        if exo.numel() >= 8:
            exo[4] = torch.cos(torch.tensor(new_ori, device=self.device))
            exo[5] = torch.sin(torch.tensor(new_ori, device=self.device))
            exo[6] = approach
            exo[7] = withdraw

        sensory = torch.cat([exo, proprio])
        if sensory.numel() < cfg.sensory_dim:
            sensory = torch.cat(
                [sensory, torch.zeros(cfg.sensory_dim - sensory.numel(), device=self.device)]
            )
        else:
            sensory = sensory[: cfg.sensory_dim]

        self.state = BodyState(
            position=new_pos,
            orientation=new_ori,
            proprioception=proprio,
            interoception=intero,
            last_motor=motor_body.detach().clone(),
            tick=self.state.tick + 1,
        )

        # Behavioral score for runner logging only (near-origin comfort)
        behavioral_correct = dist < cfg.comfort_target_radius
        behavioral_score = comfort

        return BodyStepResult(
            sensory_vector=sensory,
            proprioceptive_vector=proprio,
            interoceptive_state=intero,
            synergy_activations=syn.detach().clone(),
            body_state=self.state,
            behavioral_correct=behavioral_correct,
            behavioral_score=behavioral_score,
        )
