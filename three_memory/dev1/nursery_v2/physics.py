"""Nursery Body v2 physics — egocentric locomotion on a dense sensorimotor body."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass

import torch

from three_memory.dev1.nursery_v2.synergies import (
    N_MOTOR_CHANNELS,
    N_SYNERGIES,
    SYNERGY_REPORT_NAMES,
    channels_to_synergy_activations,
    synergy_projection_matrix,
)


@dataclass
class BodyState:
    position: torch.Tensor
    orientation: float
    proprioception: torch.Tensor
    interoception: torch.Tensor
    last_motor: torch.Tensor
    tick: int = 0


@dataclass
class BodyStepResult:
    sensory_vector: torch.Tensor
    proprioceptive_vector: torch.Tensor
    interoceptive_state: torch.Tensor
    synergy_activations: torch.Tensor
    body_state: BodyState
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
    step_scale: float = 0.15
    rotate_scale: float = 0.25
    seed: int = 0


class NurseryBodyV2:
    """
    Egocentric body: forward / backward / rotate_left / rotate_right.

    Goal-relative verbs are not actuators. Nest approach is a policy problem.
    """

    SYNERGY_NAMES = SYNERGY_REPORT_NAMES

    def __init__(self, config: BodyConfig | None = None, device: torch.device | None = None):
        self.config = config or BodyConfig()
        self.device = device or torch.device("cpu")
        self.projection = synergy_projection_matrix(
            self.config.n_motor_channels,
            self.config.n_synergies,
            device=self.device,
        )
        self.state = self._initial_state()
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
        src = inspect.getsource(NurseryBodyV2.step)
        cfg = (
            f"{self.config.n_motor_channels}|{self.config.n_synergies}|"
            f"{self.config.arena_radius}|{self.config.drive_decay}|"
            f"{self.config.comfort_target_radius}|{self.config.step_scale}|"
            f"{self.config.rotate_scale}|mass_preserving|egocentric_v2"
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
            idx = self.state.tick % len(self._open_loop_actions)
            motor_body = self._open_loop_actions[idx]
        elif not self.open_loop:
            self._open_loop_actions.append(motor_body.detach().clone())
        syn = channels_to_synergy_activations(motor_body, self.projection)
        forward, backward, rot_l, rot_r = [float(x) for x in syn.tolist()]

        # Egocentric translation in body frame
        net_drive = forward - backward
        c = math.cos(self.state.orientation)
        s = math.sin(self.state.orientation)
        dx = c * net_drive * cfg.step_scale
        dy = s * net_drive * cfg.step_scale
        new_pos = self.state.position.clone()
        new_pos[0] = new_pos[0] + dx
        new_pos[1] = new_pos[1] + dy
        r = float(new_pos.norm().item())
        if r > cfg.arena_radius:
            new_pos = new_pos * (cfg.arena_radius / r)

        new_ori = self.state.orientation + (rot_l - rot_r) * cfg.rotate_scale

        proprio = torch.zeros(cfg.proprioceptive_dim, device=self.device)
        proprio[0] = new_pos[0]
        proprio[1] = new_pos[1]
        proprio[2] = math.cos(new_ori)
        proprio[3] = math.sin(new_ori)
        for i in range(min(4, cfg.n_synergies)):
            if 4 + i < cfg.proprioceptive_dim:
                proprio[4 + i] = syn[i]
        proprio = self._apply_proprio_perm(proprio)

        dist = float(new_pos.norm().item())
        comfort = max(0.0, 1.0 - dist / max(cfg.comfort_target_radius * 3.0, 1e-6))
        energy = float(self.state.interoception[1].item()) if self.state.interoception.numel() > 1 else 0.7
        energy = max(0.0, min(1.0, energy - cfg.drive_decay + 0.01 * abs(rot_l - rot_r)))
        arousal = min(1.0, abs(net_drive) + abs(rot_l - rot_r))
        intero = torch.tensor(
            [comfort, energy, arousal, comfort][: cfg.interoceptive_dim],
            device=self.device,
            dtype=torch.float32,
        )
        if intero.numel() < cfg.interoceptive_dim:
            intero = torch.cat(
                [intero, torch.zeros(cfg.interoceptive_dim - intero.numel(), device=self.device)]
            )

        exo_dim = cfg.sensory_dim - cfg.proprioceptive_dim
        exo = torch.zeros(max(0, exo_dim), device=self.device)
        if exo.numel() >= 4:
            exo[0] = new_pos[0]
            exo[1] = new_pos[1]
            exo[2] = dist
            exo[3] = comfort
        if exo.numel() >= 8:
            exo[4] = math.cos(new_ori)
            exo[5] = math.sin(new_ori)
            exo[6] = forward
            exo[7] = backward

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
        return BodyStepResult(
            sensory_vector=sensory,
            proprioceptive_vector=proprio,
            interoceptive_state=intero,
            synergy_activations=syn.detach().clone(),
            body_state=self.state,
            behavioral_correct=dist < cfg.comfort_target_radius,
            behavioral_score=comfort,
        )
