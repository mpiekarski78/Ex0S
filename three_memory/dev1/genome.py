"""
EX0S-DEV1 genome module.

DevGenome defines all inherited parameters of a newborn organism.

What G may determine
────────────────────
- Population sizes and connectivity masks
- Generic initialization distributions (scale, sparsity)
- Cell-type / module identities and roles
- Plasticity coefficients and eligibility dynamics
- Developmental schedules driven by generic age / internal signals only
  (NEVER named curriculum stage labels)
- Replay control and timing parameters
- Generic sensory and motor channel topology

What G must NOT determine
─────────────────────────
- The meaning of any world token
- Which action is correct for any fixture
- Task-specific routing or symbol grounding
- Expected answers, vocabulary items, or linguistic slots
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PopulationSpec:
    """Specification for one cortical population."""
    n_units: int
    sparsity: float = 0.1          # target activation fraction
    init_scale: float = 0.02       # weight initialization std
    recurrent: bool = True


@dataclass
class PlasticityCoefficients:
    """
    Plasticity rule coefficients inherited by all cortical populations.
    These shape learning rate, eligibility decay, and modulation,
    but do not fix the solution.
    """
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    eligibility_decay: float = 0.9
    reward_gate_scale: float = 1.0
    prediction_error_scale: float = 1.0
    novelty_scale: float = 0.5
    hebbian_lr: float = 1e-3
    baseline_decay: float = 0.9
    gamma: float = 0.95
    actor_credit_scale: float = 1.0
    critic_scale: float = 1.0
    consequence_scale: float = 1.0
    reward_gate_center: float = 0.5
    action_update_floor: float = 1e-6
    projection_scale: float = 1.0
    temperature: float = 1.0
    update_clip_scale: float = 0.1


@dataclass
class HippocampalSpec:
    """Inherited organizational parameters for fast memory H."""
    capacity: int = 256            # maximum stored episodes
    ec_dim: int = 64               # encoder (EC analogue) output dimension
    dg_n_units: int = 512          # sparse conjunctive code population
    dg_sparsity: float = 0.05      # target DG activation fraction
    ca3_n_units: int = 256         # pattern completion / attractor population
    ca1_n_units: int = 128         # readout population
    hebbian_lr: float = 1e-2       # fast Hebbian write rate
    eviction_policy: str = "lru"   # "lru" | "random" | "decay"


@dataclass
class DevelopmentalSchedule:
    """
    Developmental state transitions keyed to GENERIC signals only.

    All transitions must be conditioned on age_ticks,
    cumulative_reward, or internal novelty estimates.
    Named curriculum stage labels are forbidden.
    """
    # Fraction of a newborn life after which replay is activated
    replay_onset_age_frac: float = 0.1
    # Fraction of life after which consolidation is enabled
    consolidation_onset_age_frac: float = 0.2
    # Minimum internal novelty signal required for a write to H
    memory_write_threshold: float = 0.3
    # Interval (in ticks) between replay episodes during rest()
    replay_interval_ticks: int = 10


@dataclass
class ReplayPolicy:
    """Controls how the organism samples from H for consolidation."""
    n_replay_samples: int = 8
    prioritized: bool = True       # True = surprise-weighted sampling
    surprise_alpha: float = 0.6    # priority exponent


@dataclass
class DevGenome:
    """
    Inherited organism organization.

    Searching over DevGenome configs is how the research optimizer
    discovers generic learning machinery across training lives.
    Validation and confirmation lives never contribute gradients
    to any genome update.
    """
    # Cortical populations
    sensory_ctx: PopulationSpec = field(default_factory=lambda: PopulationSpec(n_units=128))
    relational_ctx: PopulationSpec = field(default_factory=lambda: PopulationSpec(n_units=128))
    action_ctx: PopulationSpec = field(default_factory=lambda: PopulationSpec(n_units=64))
    n_motor_channels: int = 32     # opaque motor channel count; organism owns meaning

    # Neuromodulatory controller sizing
    neuromod_dim: int = 32

    # Plasticity
    plasticity: PlasticityCoefficients = field(default_factory=PlasticityCoefficients)
    plasticity_family: str = "reward_baseline_three_factor"

    # Fast memory
    hippocampus: HippocampalSpec = field(default_factory=HippocampalSpec)
    fast_memory_family: str = "competitive_hebbian"   # searched in axis 2

    # Consolidation
    consolidation_family: str = "replay"   # searched in axis 3

    # Developmental schedule
    schedule: DevelopmentalSchedule = field(default_factory=DevelopmentalSchedule)

    # Replay
    replay: ReplayPolicy = field(default_factory=ReplayPolicy)

    # Misc
    sensory_dim: int = 64          # dimension of sensory input vectors
    seed: int = 0                  # birth RNG seed; NOT a world seed

    # Reference Birth R3: inherited learning-signal generator parameters (flat).
    # Generic learning machinery only — never symbols/facts/mappings/expected actions.
    lsg_param_vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def credit_parameter_dict(self) -> dict[str, float]:
        """
        Shared inherited credit surface for all Stage A R1 outer optimizers.

        This intentionally excludes:
        - optimizer identity
        - credit-family identity
        - any world/task metadata
        """
        p = self.plasticity
        return {
            "learning_rate": p.learning_rate,
            "critic_learning_rate": p.critic_learning_rate,
            "eligibility_decay": p.eligibility_decay,
            "reward_gate_scale": p.reward_gate_scale,
            "prediction_error_scale": p.prediction_error_scale,
            "novelty_scale": p.novelty_scale,
            "hebbian_lr": p.hebbian_lr,
            "baseline_decay": p.baseline_decay,
            "gamma": p.gamma,
            "actor_credit_scale": p.actor_credit_scale,
            "critic_scale": p.critic_scale,
            "consequence_scale": p.consequence_scale,
            "reward_gate_center": p.reward_gate_center,
            "action_update_floor": p.action_update_floor,
            "projection_scale": p.projection_scale,
            "temperature": p.temperature,
            "update_clip_scale": p.update_clip_scale,
        }

    def set_credit_parameter_dict(self, params: dict[str, float]) -> None:
        """Update only the shared inherited credit surface in-place."""
        for k, v in params.items():
            if hasattr(self.plasticity, k):
                setattr(self.plasticity, k, float(v))

    def r1_inherited_surface_dict(self) -> dict[str, float]:
        """Limited Reference Birth R1 outer-search surface (not full matrices)."""
        p = self.plasticity
        return {
            "log_actor_learning_rate": float(math.log(max(p.learning_rate, 1e-12))),
            "log_critic_learning_rate": float(math.log(max(p.critic_learning_rate, 1e-12))),
            "eligibility_decay": p.eligibility_decay,
            "td_discount_gamma": p.gamma,
            "learning_signal_projection_scale": p.projection_scale,
            "entropy_exploration_temperature": p.temperature,
            "update_clip_scale": p.update_clip_scale,
        }

    def apply_r1_inherited_surface(self, surface: dict[str, float]) -> None:
        """Decode R1 outer surface into plasticity coefficients."""
        import math as _math
        if "log_actor_learning_rate" in surface:
            self.plasticity.learning_rate = float(_math.exp(surface["log_actor_learning_rate"]))
        if "log_critic_learning_rate" in surface:
            self.plasticity.critic_learning_rate = float(_math.exp(surface["log_critic_learning_rate"]))
        if "eligibility_decay" in surface:
            self.plasticity.eligibility_decay = float(max(0.01, min(0.999, surface["eligibility_decay"])))
        if "td_discount_gamma" in surface:
            self.plasticity.gamma = float(max(0.0, min(0.999, surface["td_discount_gamma"])))
        if "learning_signal_projection_scale" in surface:
            self.plasticity.projection_scale = float(max(1e-6, surface["learning_signal_projection_scale"]))
        if "entropy_exploration_temperature" in surface:
            self.plasticity.temperature = float(max(1e-3, surface["entropy_exploration_temperature"]))
        if "update_clip_scale" in surface:
            self.plasticity.update_clip_scale = float(max(1e-6, surface["update_clip_scale"]))

    def genome_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def default(cls) -> "DevGenome":
        return cls()

    @classmethod
    def with_size(cls, n: int, fast_memory_family: str = "competitive_hebbian") -> "DevGenome":
        """Convenience constructor for factorial capacity arms (n = 64, 256, 1024)."""
        return cls(
            sensory_ctx=PopulationSpec(n_units=n),
            relational_ctx=PopulationSpec(n_units=n),
            action_ctx=PopulationSpec(n_units=n // 2),
            neuromod_dim=max(16, n // 4),
            hippocampus=HippocampalSpec(
                capacity=n * 4,
                ec_dim=n // 2,
                dg_n_units=n * 4,
                ca3_n_units=n * 2,
                ca1_n_units=n,
            ),
            fast_memory_family=fast_memory_family,
        )
