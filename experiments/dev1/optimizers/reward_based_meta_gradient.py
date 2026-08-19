"""
Stage A R1.1 reward-based outer score-function optimizer.

This is a genuine gradient-based between-life optimizer over the shared
inherited credit surface. It does not backpropagate through W, H, or rho
during a life. Instead, it maintains a Gaussian search distribution over the
credit surface and applies a score-function update after completed training
lives using normalized lifetime fitness.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from three_memory.dev1.genome import DevGenome


@dataclass
class RewardBasedMetaGradientConfig:
    learning_rate: float = 0.05
    init_std: float = 0.1
    min_std: float = 1e-3
    clamp_min: float = 1e-6
    clamp_max: float = 2.0
    seed: int = 0


class RewardBasedMetaGradientOptimizer:
    """
    Score-function optimizer over a Gaussian search distribution on G.

    The trainable parameters are the mean of the inherited credit surface.
    Each proposed genome is sampled from that distribution, evaluated on full
    training lives, then the optimizer applies REINFORCE to the sampled
    perturbation using the normalized lifetime fitness as the return signal.
    """

    def __init__(self, cfg: RewardBasedMetaGradientConfig | None = None):
        self.cfg = cfg or RewardBasedMetaGradientConfig()
        self._param_names: list[str] | None = None
        self._mean: torch.nn.Parameter | None = None
        self._std: torch.Tensor | None = None
        self._optim: torch.optim.Optimizer | None = None
        self._generator = torch.Generator().manual_seed(self.cfg.seed)
        self.n_updates = 0
        self.last_loss: float | None = None
        self.last_gradient_norm: float | None = None
        self.last_sample_delta_l2: float | None = None
        self.last_update_delta_l2: float | None = None
        self.last_fitness: float | None = None

    def _ensure_state(self, genome: DevGenome) -> None:
        if self._mean is not None:
            return
        params = genome.credit_parameter_dict()
        self._param_names = list(params.keys())
        init = torch.tensor([params[k] for k in self._param_names], dtype=torch.float32)
        self._mean = torch.nn.Parameter(init.clone())
        self._std = torch.full_like(init, self.cfg.init_std)
        self._optim = torch.optim.SGD([self._mean], lr=self.cfg.learning_rate)

    def propose(self, genome: DevGenome) -> tuple[DevGenome, dict]:
        self._ensure_state(genome)
        assert self._param_names is not None
        assert self._mean is not None
        assert self._std is not None

        eps = torch.randn(self._mean.shape, generator=self._generator, dtype=self._mean.dtype)
        sample = self._mean.detach() + self._std * eps
        clipped = torch.clamp(sample, min=self.cfg.clamp_min, max=self.cfg.clamp_max)

        proposed = copy.deepcopy(genome)
        proposed.set_credit_parameter_dict({
            name: float(value.item())
            for name, value in zip(self._param_names, clipped)
        })
        metadata = {
            "eps": eps,
            "sample": clipped.detach().clone(),
            "sample_delta_l2": float(torch.norm(clipped - self._mean.detach()).item()),
        }
        return proposed, metadata

    def update_after_training_lives(
        self,
        proposal_metadata: dict,
        normalized_fitness: float,
    ) -> None:
        assert self._mean is not None
        assert self._std is not None
        assert self._optim is not None
        sample = proposal_metadata["sample"].detach()
        dist = torch.distributions.Normal(self._mean, self._std)
        log_prob = dist.log_prob(sample).sum()
        loss = -(log_prob * torch.tensor(float(normalized_fitness), dtype=self._mean.dtype))

        before = self._mean.detach().clone()
        self._optim.zero_grad()
        loss.backward()
        grad_sq = 0.0
        for param in [self._mean]:
            if param.grad is not None:
                grad_sq += float((param.grad.detach() ** 2).sum().item())
        self._optim.step()
        with torch.no_grad():
            self._mean.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)

        self.n_updates += 1
        self.last_loss = float(loss.item())
        self.last_gradient_norm = math.sqrt(grad_sq)
        self.last_sample_delta_l2 = float(proposal_metadata["sample_delta_l2"])
        self.last_update_delta_l2 = float(torch.norm(self._mean.detach() - before).item())
        self.last_fitness = float(normalized_fitness)

    def current_genome(self, template: DevGenome) -> DevGenome:
        self._ensure_state(template)
        assert self._param_names is not None
        assert self._mean is not None
        genome = copy.deepcopy(template)
        clipped = torch.clamp(self._mean.detach(), min=self.cfg.clamp_min, max=self.cfg.clamp_max)
        genome.set_credit_parameter_dict({
            name: float(value.item())
            for name, value in zip(self._param_names, clipped)
        })
        return genome

    def telemetry(self) -> dict:
        return {
            "optimizer_arm": "reward_based_meta_gradient",
            "parameter_names": list(self._param_names or []),
            "outer_updates": self.n_updates,
            "outer_loss": self.last_loss,
            "gradient_norm": self.last_gradient_norm,
            "sample_delta_l2": self.last_sample_delta_l2,
            "update_delta_l2": self.last_update_delta_l2,
            "fitness": self.last_fitness,
            "soft_action_surrogate": False,
        }
