"""Exact Nursery Body v2 dynamics as measurement-only forward-model ceiling."""

from __future__ import annotations

import torch

from three_memory.dev1.gsm.state import VisibleDims, dims_from_body_config, pack_visible_state
from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import expand_synergy_index_to_motor


def exact_next_state_vector(
    body: NurseryBodyV2,
    synergy_index: int,
    *,
    dims: VisibleDims | None = None,
) -> torch.Tensor:
    """
    Measurement ceiling only — not an organism candidate.

    Clone visible body fields, apply unit synergy, return packed next state.
    """
    cfg = body.config
    d = dims or dims_from_body_config(
        sensory_dim=cfg.sensory_dim,
        proprioceptive_dim=cfg.proprioceptive_dim,
        interoceptive_dim=cfg.interoceptive_dim,
        n_motor_channels=cfg.n_motor_channels,
    )
    probe = NurseryBodyV2(
        BodyConfig(
            n_motor_channels=cfg.n_motor_channels,
            n_synergies=cfg.n_synergies,
            sensory_dim=cfg.sensory_dim,
            interoceptive_dim=cfg.interoceptive_dim,
            proprioceptive_dim=cfg.proprioceptive_dim,
            arena_radius=cfg.arena_radius,
            drive_decay=cfg.drive_decay,
            comfort_target_radius=cfg.comfort_target_radius,
            step_scale=cfg.step_scale,
            rotate_scale=cfg.rotate_scale,
            seed=cfg.seed,
        ),
        device=torch.device("cpu"),
    )
    probe.state.position = body.state.position.detach().cpu().clone()
    probe.state.orientation = float(body.state.orientation)
    probe.state.interoception = body.state.interoception.detach().cpu().clone()
    probe.state.proprioception = body.state.proprioception.detach().cpu().clone()
    probe.state.last_motor = body.state.last_motor.detach().cpu().clone()
    probe.state.tick = int(body.state.tick)
    step = probe.step(expand_synergy_index_to_motor(synergy_index, device=torch.device("cpu")))
    return pack_visible_state(
        sensory=step.sensory_vector, intero=step.interoceptive_state, dims=d
    )
