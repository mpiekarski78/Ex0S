"""
Stage A R2 scaffold phenotype and sensitivity probes.

R2 searches inherited developmental organization without inheriting facts.
This module defines a scaffold phenotype that is applied to a newborn
organism before life begins and a preflight that proves scaffold motion causes
phenotype motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch

from experiments.dev1.worlds import InteractionWorld, WorldConfig
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


@dataclass
class ContinuousScaffoldPhenotype:
    sensory_init_scale: float = 1.0
    relational_init_scale: float = 1.0
    action_init_scale: float = 1.0
    sensory_recurrent_radius: float = 0.8
    relational_recurrent_radius: float = 0.8
    action_recurrent_radius: float = 0.8
    sensory_density: float = 1.0
    relational_density: float = 1.0
    action_density: float = 1.0
    ei_balance: float = 0.0
    motor_basis_scale: float = 1.0
    homeostatic_target_norm: float = 1.0
    normalization_strength: float = 0.0
    plasticity_mask_gain: float = 1.0


@dataclass
class TopologyScaffoldPhenotype:
    motif: str = "dense"


@dataclass
class ScaffoldSensitivityResult:
    passed: bool
    decision_code: str
    metrics: dict
    details: dict


def scaffold_hash(continuous: ContinuousScaffoldPhenotype, topology: TopologyScaffoldPhenotype) -> str:
    payload = json.dumps({
        "continuous": asdict(continuous),
        "topology": asdict(topology),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def apply_scaffold_to_organism(
    org: ModularOrganism,
    continuous: ContinuousScaffoldPhenotype,
    topology: TopologyScaffoldPhenotype,
) -> None:
    _scale_module(org.sensory_ctx.pop, continuous.sensory_init_scale, continuous.sensory_density, continuous.sensory_recurrent_radius, topology.motif)
    _scale_module(org.relational_ctx.pop, continuous.relational_init_scale, continuous.relational_density, continuous.relational_recurrent_radius, topology.motif)
    _scale_module(org.action_ctx.pop, continuous.action_init_scale, continuous.action_density, continuous.action_recurrent_radius, topology.motif)
    with torch.no_grad():
        org.action_ctx.W_motor.weight.mul_(continuous.motor_basis_scale)
        if topology.motif == "block":
            _apply_block_mask(org.action_ctx.W_motor.weight)
        elif topology.motif == "banded":
            _apply_banded_mask(org.action_ctx.W_motor.weight)
    org._r2_homeostatic_target_norm = continuous.homeostatic_target_norm
    org._r2_normalization_strength = continuous.normalization_strength
    org._r2_plasticity_mask_gain = continuous.plasticity_mask_gain
    org._r2_scaffold_hash = scaffold_hash(continuous, topology)


def normalize_r2_state(org: ModularOrganism) -> None:
    strength = float(getattr(org, "_r2_normalization_strength", 0.0))
    if strength <= 0.0:
        return
    target = float(getattr(org, "_r2_homeostatic_target_norm", 1.0))
    with torch.no_grad():
        for tensor_name in ["sensory_repr", "relational_repr", "action_repr"]:
            t = getattr(org.rho, tensor_name)
            norm = float(t.norm().item())
            if norm <= 1e-12:
                continue
            desired = max(1e-6, target)
            scale = (1.0 - strength) + strength * (desired / norm)
            t.mul_(scale)


def scaffold_extremes() -> tuple[ContinuousScaffoldPhenotype, ContinuousScaffoldPhenotype]:
    low = ContinuousScaffoldPhenotype(
        sensory_init_scale=0.4,
        relational_init_scale=0.4,
        action_init_scale=0.4,
        sensory_recurrent_radius=0.2,
        relational_recurrent_radius=0.2,
        action_recurrent_radius=0.2,
        sensory_density=0.4,
        relational_density=0.4,
        action_density=0.4,
        ei_balance=-0.4,
        motor_basis_scale=0.5,
        homeostatic_target_norm=0.5,
        normalization_strength=0.1,
        plasticity_mask_gain=0.5,
    )
    high = ContinuousScaffoldPhenotype(
        sensory_init_scale=1.6,
        relational_init_scale=1.6,
        action_init_scale=1.6,
        sensory_recurrent_radius=0.98,
        relational_recurrent_radius=0.98,
        action_recurrent_radius=0.98,
        sensory_density=1.0,
        relational_density=1.0,
        action_density=1.0,
        ei_balance=0.4,
        motor_basis_scale=1.5,
        homeostatic_target_norm=2.0,
        normalization_strength=0.9,
        plasticity_mask_gain=1.5,
    )
    return low, high


def run_scaffold_sensitivity_preflight(
    genome: DevGenome,
    credit_family: str,
    topology: TopologyScaffoldPhenotype | None = None,
) -> ScaffoldSensitivityResult:
    topo = topology or TopologyScaffoldPhenotype()
    low, high = scaffold_extremes()
    world = InteractionWorld(WorldConfig(seed="stage_a_r2_preflight"))

    org_low = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    org_high = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    org_low.genome.plasticity_family = credit_family
    org_high.genome.plasticity_family = credit_family
    apply_scaffold_to_organism(org_low, low, topo)
    apply_scaffold_to_organism(org_high, high, topo)

    metrics_low = _collect_probe_metrics(org_low, world)
    metrics_high = _collect_probe_metrics(org_high, world)
    metrics = {
        "recurrent_dynamics_delta": abs(metrics_high["recurrent_norm"] - metrics_low["recurrent_norm"]),
        "sensory_separability_delta": abs(metrics_high["sensory_sep"] - metrics_low["sensory_sep"]),
        "motor_basis_separability_delta": abs(metrics_high["motor_sep"] - metrics_low["motor_sep"]),
        "actor_update_delta": abs(metrics_high["actor_delta_norm"] - metrics_low["actor_delta_norm"]),
        "action_probability_delta": abs(metrics_high["action_confidence"] - metrics_low["action_confidence"]),
        "homeostatic_response_delta": abs(metrics_high["homeostatic_norm"] - metrics_low["homeostatic_norm"]),
        "phenotype_delta_from_genome_delta": _phenotype_delta(low, high),
        "fraction_parameters_at_bounds": _fraction_at_bounds(low) + _fraction_at_bounds(high),
        "raw_gradient_norm": 0.0,
        "clipped_gradient_norm": 0.0,
        "raw_genome_step_norm": 0.0,
        "clipped_genome_step_norm": 0.0,
        "transform_jacobian_norm": _approx_transform_jacobian_norm(low, high),
    }
    passed = all(v > 1e-6 for k, v in metrics.items() if k.endswith("_delta")) and _all_finite(metrics)
    return ScaffoldSensitivityResult(
        passed=passed,
        decision_code="scaffold_sensitivity_pass" if passed else "scaffold_sensitivity_fail",
        metrics=metrics,
        details={"low": metrics_low, "high": metrics_high, "topology": topo.motif},
    )


def _collect_probe_metrics(org: ModularOrganism, world: InteractionWorld) -> dict:
    event = world.generate_episode()[0]
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    normalize_r2_state(org)
    action = org.act(policy_mode="hard")
    org.rest()
    return {
        "recurrent_norm": float(org.rho.relational_repr.norm().item()),
        "sensory_sep": float(org.rho.sensory_repr.norm().item()),
        "motor_sep": float(np.std(action.motor_scores)),
        "actor_delta_norm": float(getattr(org, "_last_actor_delta", torch.zeros(1)).norm().item()),
        "action_confidence": float(action.confidence),
        "homeostatic_norm": float(org.rho.action_repr.norm().item()),
    }


def _scale_module(pop, scale: float, density: float, radius: float, motif: str) -> None:
    with torch.no_grad():
        pop.W_in.weight.mul_(scale)
        _apply_density(pop.W_in.weight, density, motif)
        if pop.W_rec is not None:
            pop.W_rec.weight.mul_(scale)
            _apply_density(pop.W_rec.weight, density, motif)
            _set_spectral_radius(pop.W_rec.weight, radius)


def _apply_density(weight: torch.Tensor, density: float, motif: str) -> None:
    if density >= 0.999:
        return
    mask = torch.ones_like(weight)
    total = mask.numel()
    keep = max(1, int(total * density))
    flat = mask.view(-1)
    flat[keep:] = 0.0
    if motif == "block":
        mask = _block_mask(mask)
    elif motif == "banded":
        mask = _banded_mask(mask)
    weight.mul_(mask)


def _set_spectral_radius(weight: torch.Tensor, target: float) -> None:
    if weight.shape[0] != weight.shape[1]:
        return
    try:
        eigvals = torch.linalg.eigvals(weight)
        radius = float(eigvals.abs().max().item())
    except Exception:
        radius = float(weight.norm().item())
    if radius > 1e-8:
        weight.mul_(target / radius)


def _block_mask(tensor: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(tensor)
    h, w = tensor.shape
    mask[: h // 2, : w // 2] = 1.0
    mask[h // 2 :, w // 2 :] = 1.0
    return mask


def _banded_mask(tensor: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        lo = max(0, i - 2)
        hi = min(tensor.shape[1], i + 3)
        mask[i, lo:hi] = 1.0
    return mask


def _apply_block_mask(weight: torch.Tensor) -> None:
    weight.mul_(_block_mask(weight))


def _apply_banded_mask(weight: torch.Tensor) -> None:
    weight.mul_(_banded_mask(weight))


def _phenotype_delta(a: ContinuousScaffoldPhenotype, b: ContinuousScaffoldPhenotype) -> float:
    va = np.array(list(asdict(a).values()), dtype=np.float32)
    vb = np.array(list(asdict(b).values()), dtype=np.float32)
    return float(np.linalg.norm(vb - va))


def _fraction_at_bounds(a: ContinuousScaffoldPhenotype) -> float:
    vals = list(asdict(a).values())
    return float(sum(v in {0.0, 1.0} for v in vals) / max(1, len(vals)))


def _approx_transform_jacobian_norm(a: ContinuousScaffoldPhenotype, b: ContinuousScaffoldPhenotype) -> float:
    return _phenotype_delta(a, b) / max(1e-6, len(asdict(a)))


def _all_finite(metrics: dict) -> bool:
    return all(math.isfinite(float(v)) for v in metrics.values())
