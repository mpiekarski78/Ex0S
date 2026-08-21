"""
GSM deployment-wall measurement controllers.

Exact NurseryBodyV2 dynamics + organism valence. Not organism candidates.
Must not write weights, read future world state, or consume expected actions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.development.valence import OrganismValenceCircuit
from three_memory.dev1.gsm.state import VisibleDims, dims_from_body_config
from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import N_SYNERGIES, expand_synergy_index_to_motor


HORIZON_TICKS = 3  # frozen; do not sweep
MEASUREMENT_ONLY = True
UNCERTAINTY_MAX_PIN = 0.35


@dataclass(frozen=True)
class ExactControllerChoice:
    synergy_index: int
    motor: torch.Tensor
    imagined_valence: float
    horizon_ticks: int
    controller: str
    measurement_only: bool = True
    probe_mutated_live_body: bool = False
    accessed_expected_action: bool = False
    wrote_weights: bool = False


def _dims_for_body(body: NurseryBodyV2) -> VisibleDims:
    cfg = body.config
    return dims_from_body_config(
        sensory_dim=cfg.sensory_dim,
        proprioceptive_dim=cfg.proprioceptive_dim,
        interoceptive_dim=cfg.interoceptive_dim,
        n_motor_channels=cfg.n_motor_channels,
    )


def _clone_probe(body: NurseryBodyV2) -> NurseryBodyV2:
    """Fresh probe body cloned from visible fields only — never the live world body."""
    cfg = body.config
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
    return probe


def _assert_no_forbidden_world_access(world: Any) -> None:
    if hasattr(world, "expected_action"):
        raise RuntimeError("exact controller refused expected_action on world")
    if hasattr(world, "teacher_action"):
        raise RuntimeError("exact controller refused teacher_action on world")


def _imagined_step_valence(
    probe: NurseryBodyV2,
    valence: OrganismValenceCircuit,
    synergy_index: int,
) -> float:
    current_comfort = valence.comfort(probe.state.interoception)
    probe.step(expand_synergy_index_to_motor(synergy_index, device=torch.device("cpu")))
    comfort_hat = valence.comfort(probe.state.interoception)
    return float(valence.gain) * (comfort_hat - current_comfort)


def choose_exact_one_step_valence(
    body: NurseryBodyV2,
    valence: OrganismValenceCircuit,
    *,
    device: torch.device,
    world: Any | None = None,
) -> ExactControllerChoice:
    if world is not None:
        _assert_no_forbidden_world_access(world)
    live_pos_before = body.state.position.detach().cpu().clone()
    best_syn = 0
    best_v = float("-inf")
    for syn in range(N_SYNERGIES):
        probe = _clone_probe(body)
        imagined = _imagined_step_valence(probe, valence, syn)
        if imagined > best_v:
            best_v = imagined
            best_syn = syn
    live_pos_after = body.state.position.detach().cpu()
    if not torch.equal(live_pos_before, live_pos_after):
        raise RuntimeError("exact one-step controller mutated live body")
    motor = expand_synergy_index_to_motor(best_syn, device=device)
    return ExactControllerChoice(
        synergy_index=int(best_syn),
        motor=motor,
        imagined_valence=float(best_v),
        horizon_ticks=1,
        controller="exact_one_step_valence",
    )


def choose_exact_receding_horizon(
    body: NurseryBodyV2,
    valence: OrganismValenceCircuit,
    *,
    device: torch.device,
    horizon_ticks: int = HORIZON_TICKS,
    world: Any | None = None,
) -> ExactControllerChoice:
    if int(horizon_ticks) != int(HORIZON_TICKS):
        raise RuntimeError(
            f"receding horizon must remain frozen at H={HORIZON_TICKS}; got {horizon_ticks}"
        )
    if world is not None:
        _assert_no_forbidden_world_access(world)
    live_pos_before = body.state.position.detach().cpu().clone()
    best_seq: tuple[int, ...] | None = None
    best_v = float("-inf")
    for seq in itertools.product(range(N_SYNERGIES), repeat=int(horizon_ticks)):
        probe = _clone_probe(body)
        total = 0.0
        for syn in seq:
            total += _imagined_step_valence(probe, valence, int(syn))
        if total > best_v:
            best_v = total
            best_seq = seq
    assert best_seq is not None
    live_pos_after = body.state.position.detach().cpu()
    if not torch.equal(live_pos_before, live_pos_after):
        raise RuntimeError("exact receding-horizon controller mutated live body")
    syn0 = int(best_seq[0])
    motor = expand_synergy_index_to_motor(syn0, device=device)
    return ExactControllerChoice(
        synergy_index=syn0,
        motor=motor,
        imagined_valence=float(best_v),
        horizon_ticks=int(horizon_ticks),
        controller="exact_receding_horizon",
    )


def controller_invariants() -> dict[str, Any]:
    return {
        "measurement_only": MEASUREMENT_ONLY,
        "horizon_ticks_frozen": HORIZON_TICKS,
        "uncertainty_max_pin": UNCERTAINTY_MAX_PIN,
        "cannot_write_weights": True,
        "cannot_access_future_world_state": True,
        "cannot_access_expected_actions": True,
        "enumerates_probe_clones_only": True,
    }


def distance_region(distance: float, *, comfort_radius: float = 0.35) -> str:
    d = float(distance)
    if d <= float(comfort_radius):
        return "near_comfort"
    if d <= 1.0:
        return "mid_distance"
    return "far"
