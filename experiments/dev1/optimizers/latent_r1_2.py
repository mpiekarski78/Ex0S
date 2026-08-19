"""
Stage A R1.2 latent-surface optimizers and parameter transforms.

R1.2 keeps the R1.1 learning objective unchanged but prevents raw optimizer
values from entering organism lives. Both optimizers manipulate the same
unconstrained latent vector z and transform it into a bounded phenotype before
evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.genome import DevGenome


EPS = 1e-6
WORST_FITNESS = -1.1
LATENT_PARAM_ORDER = [
    "learning_rate",
    "eligibility_decay",
    "reward_gate_scale",
    "prediction_error_scale",
    "novelty_scale",
    "hebbian_lr",
    "baseline_decay",
    "gamma",
    "actor_credit_scale",
    "critic_scale",
    "consequence_scale",
    "reward_gate_center",
    "action_update_floor",
]
PARAM_SPECS = {
    "learning_rate": {"kind": "positive_bounded", "lo": 1e-6, "hi": 0.5},
    "eligibility_decay": {"kind": "unit_decay", "lo": 0.0, "hi": 0.999},
    "reward_gate_scale": {"kind": "signed", "max_abs": 2.0},
    "prediction_error_scale": {"kind": "signed", "max_abs": 2.0},
    "novelty_scale": {"kind": "signed", "max_abs": 2.0},
    "hebbian_lr": {"kind": "positive_bounded", "lo": 1e-6, "hi": 0.5},
    "baseline_decay": {"kind": "unit_decay", "lo": 0.0, "hi": 0.999},
    "gamma": {"kind": "unit_decay", "lo": 0.0, "hi": 0.999},
    "actor_credit_scale": {"kind": "signed", "max_abs": 2.0},
    "critic_scale": {"kind": "signed", "max_abs": 2.0},
    "consequence_scale": {"kind": "signed", "max_abs": 2.0},
    "reward_gate_center": {"kind": "bounded", "lo": 0.0, "hi": 1.0},
    "action_update_floor": {"kind": "positive_bounded", "lo": 1e-8, "hi": 0.1},
}


def _bounded_sigmoid(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(z)


def _bounded_sigmoid_inv(x: float, lo: float, hi: float) -> float:
    y = min(1.0 - EPS, max(EPS, (x - lo) / max(EPS, hi - lo)))
    return float(math.log(y / (1.0 - y)))


def _signed_tanh(z: torch.Tensor, max_abs: float) -> torch.Tensor:
    return max_abs * torch.tanh(z)


def _signed_tanh_inv(x: float, max_abs: float) -> float:
    y = min(1.0 - EPS, max(-1.0 + EPS, x / max(EPS, max_abs)))
    return float(0.5 * math.log((1.0 + y) / (1.0 - y)))


def latent_from_genome(genome: DevGenome) -> torch.Tensor:
    params = genome.credit_parameter_dict()
    vals = []
    for name in LATENT_PARAM_ORDER:
        spec = PARAM_SPECS[name]
        value = float(params[name])
        if spec["kind"] in {"bounded", "positive_bounded", "unit_decay"}:
            vals.append(_bounded_sigmoid_inv(value, spec["lo"], spec["hi"]))
        elif spec["kind"] == "signed":
            vals.append(_signed_tanh_inv(value, spec["max_abs"]))
        else:
            raise ValueError(name)
    return torch.tensor(vals, dtype=torch.float32)


def phenotype_from_latent(latent: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, name in enumerate(LATENT_PARAM_ORDER):
        spec = PARAM_SPECS[name]
        z = latent[idx]
        if spec["kind"] in {"bounded", "positive_bounded", "unit_decay"}:
            value = _bounded_sigmoid(z, spec["lo"], spec["hi"])
        elif spec["kind"] == "signed":
            value = _signed_tanh(z, spec["max_abs"])
        else:
            raise ValueError(name)
        out[name] = float(value.item())
    return out


def apply_latent_to_genome(template: DevGenome, latent: torch.Tensor) -> DevGenome:
    genome = copy.deepcopy(template)
    genome.set_credit_parameter_dict(phenotype_from_latent(latent))
    return genome


def latent_hash(latent: torch.Tensor) -> str:
    payload = json.dumps([float(x) for x in latent.detach().cpu().tolist()], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def phenotype_hash(params: dict[str, float]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_phenotype(params: dict[str, float]) -> tuple[bool, str | None]:
    for name, value in params.items():
        if not math.isfinite(value):
            return False, f"{name}_nonfinite"
        spec = PARAM_SPECS[name]
        if spec["kind"] == "unit_decay" and not (spec["lo"] <= value < spec["hi"]):
            return False, f"{name}_domain"
        if spec["kind"] == "positive_bounded" and not (spec["lo"] <= value <= spec["hi"]):
            return False, f"{name}_domain"
        if spec["kind"] == "bounded" and not (spec["lo"] <= value <= spec["hi"]):
            return False, f"{name}_domain"
        if spec["kind"] == "signed" and not (-spec["max_abs"] <= value <= spec["max_abs"]):
            return False, f"{name}_domain"
    return True, None


@dataclass
class R12MetaConfig:
    learning_rate: float = 0.05
    init_std: float = 0.1
    grad_clip_norm: float = 5.0
    step_clip_norm: float = 1.0
    seed: int = 0


class RewardBasedMetaGradientR12:
    def __init__(self, cfg: R12MetaConfig | None = None):
        self.cfg = cfg or R12MetaConfig()
        self._mean_z: torch.nn.Parameter | None = None
        self._std: torch.Tensor | None = None
        self._optim: torch.optim.Optimizer | None = None
        self._generator = torch.Generator().manual_seed(self.cfg.seed)
        self.n_updates = 0
        self.last_loss: float | None = None
        self.last_gradient_norm: float | None = None
        self.last_step_norm: float | None = None
        self.last_invalid_reason: str | None = None

    def _ensure_state(self, template: DevGenome) -> None:
        if self._mean_z is not None:
            return
        init = latent_from_genome(template)
        self._mean_z = torch.nn.Parameter(init.clone())
        self._std = torch.full_like(init, self.cfg.init_std)
        self._optim = torch.optim.SGD([self._mean_z], lr=self.cfg.learning_rate)

    def propose(self, template: DevGenome) -> tuple[DevGenome | None, dict[str, Any]]:
        self._ensure_state(template)
        assert self._mean_z is not None and self._std is not None
        eps = torch.randn(self._mean_z.shape, generator=self._generator, dtype=self._mean_z.dtype)
        z_sample = self._mean_z.detach() + self._std * eps
        params = phenotype_from_latent(z_sample)
        ok, reason = validate_phenotype(params)
        meta = {
            "z_sample": z_sample.detach().clone(),
            "latent_hash": latent_hash(z_sample),
            "phenotype_hash": phenotype_hash(params),
            "phenotype_params": params,
            "invalid_reason": reason,
        }
        if not ok:
            return None, meta
        return apply_latent_to_genome(template, z_sample), meta

    def update_after_training_lives(self, proposal_meta: dict[str, Any], fitness: float) -> bool:
        assert self._mean_z is not None and self._std is not None and self._optim is not None
        z_sample = proposal_meta["z_sample"].detach()
        dist = torch.distributions.Normal(self._mean_z, self._std)
        loss = -(dist.log_prob(z_sample).sum() * torch.tensor(float(fitness), dtype=self._mean_z.dtype))
        before = self._mean_z.detach().clone()
        self._optim.zero_grad()
        loss.backward()
        grad = self._mean_z.grad
        if grad is None or not torch.isfinite(grad).all():
            self.last_invalid_reason = "gradient_nonfinite"
            return False
        grad_norm = float(torch.norm(grad).item())
        if grad_norm > self.cfg.grad_clip_norm:
            grad.mul_(self.cfg.grad_clip_norm / max(grad_norm, 1e-12))
            grad_norm = self.cfg.grad_clip_norm
        self._optim.step()
        step = self._mean_z.detach() - before
        step_norm = float(torch.norm(step).item())
        if step_norm > self.cfg.step_clip_norm:
            with torch.no_grad():
                self._mean_z.copy_(before + step * (self.cfg.step_clip_norm / max(step_norm, 1e-12)))
            step_norm = self.cfg.step_clip_norm
        if not torch.isfinite(self._mean_z).all():
            self.last_invalid_reason = "optimizer_state_nonfinite"
            return False
        self.n_updates += 1
        self.last_loss = float(loss.item())
        self.last_gradient_norm = grad_norm
        self.last_step_norm = step_norm
        self.last_invalid_reason = None
        return True

    def current_genome(self, template: DevGenome) -> tuple[DevGenome | None, dict[str, Any]]:
        self._ensure_state(template)
        assert self._mean_z is not None
        params = phenotype_from_latent(self._mean_z.detach())
        meta = {
            "latent_hash": latent_hash(self._mean_z.detach()),
            "phenotype_hash": phenotype_hash(params),
            "phenotype_params": params,
        }
        ok, reason = validate_phenotype(params)
        if not ok:
            meta["invalid_reason"] = reason
            return None, meta
        return apply_latent_to_genome(template, self._mean_z.detach()), meta

    def telemetry(self) -> dict[str, Any]:
        return {
            "optimizer_arm": "reward_based_meta_gradient_r1_2",
            "outer_updates": self.n_updates,
            "outer_loss": self.last_loss,
            "gradient_norm": self.last_gradient_norm,
            "genome_step_norm": self.last_step_norm,
            "invalid_reason": self.last_invalid_reason,
            "latent_parameter_names": list(LATENT_PARAM_ORDER),
        }


@dataclass
class R12EvolutionaryConfig:
    population_size: int = 4
    mutation_scale: float = 0.25
    step_clip_norm: float = 1.0
    seed: int = 0


class EvolutionaryR12:
    def __init__(self, cfg: R12EvolutionaryConfig | None = None):
        self.cfg = cfg or R12EvolutionaryConfig()
        self.rng = random.Random(self.cfg.seed)
        self.n_generations = 0

    def spawn_population(self, template: DevGenome, parent_z: torch.Tensor) -> list[tuple[DevGenome | None, dict[str, Any]]]:
        population = []
        for _ in range(self.cfg.population_size):
            noise = torch.tensor([self.rng.gauss(0.0, self.cfg.mutation_scale) for _ in LATENT_PARAM_ORDER], dtype=torch.float32)
            step_norm = float(torch.norm(noise).item())
            if step_norm > self.cfg.step_clip_norm:
                noise = noise * (self.cfg.step_clip_norm / max(step_norm, 1e-12))
                step_norm = self.cfg.step_clip_norm
            child_z = parent_z + noise
            params = phenotype_from_latent(child_z)
            ok, reason = validate_phenotype(params)
            meta = {
                "z_sample": child_z,
                "latent_hash": latent_hash(child_z),
                "phenotype_hash": phenotype_hash(params),
                "phenotype_params": params,
                "invalid_reason": reason,
                "genome_step_norm": step_norm,
            }
            if not ok:
                population.append((None, meta))
            else:
                population.append((apply_latent_to_genome(template, child_z), meta))
        return population

    def select(self, population: list[tuple[DevGenome | None, dict[str, Any]]], fitnesses: list[float]) -> tuple[torch.Tensor | None, dict[str, Any]]:
        valid = [(idx, entry) for idx, entry in enumerate(population) if entry[0] is not None]
        if not valid:
            return None, {"invalid_reason": "all_candidates_invalid"}
        best_idx = max([idx for idx, _ in valid], key=lambda i: fitnesses[i])
        self.n_generations += 1
        return population[best_idx][1]["z_sample"].detach().clone(), population[best_idx][1]

    def telemetry(self) -> dict[str, Any]:
        return {
            "optimizer_arm": "evolutionary_r1_2",
            "outer_generations": self.n_generations,
            "population_size": self.cfg.population_size,
            "latent_parameter_names": list(LATENT_PARAM_ORDER),
        }
