"""
Generative construction genome for Developmental Birth R4.

This is a construction program — not a parameter bag around random matrices.
It may declare population types, topology motifs, synergy templates, plasticity
and developmental rules. It must never carry cue names, fixture answers,
world seeds, scored symbols, or task-specific final weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# Runner/report names only — never enter the organism as semantic labels.
SYNERGY_REPORT_NAMES: tuple[str, ...] = ("approach", "withdraw", "orient", "wait")
N_SYNERGIES = 4
N_MOTOR_CHANNELS = 32


@dataclass
class GenerativeGenome:
    """
    Inherited generative construction program.

    Scaling changes declared population counts via this genome; construction
    algorithm is identical at larger sizes (no hand-written size-specific wiring).
    """

    # Population counts (smallest executable size that instantiates every module)
    sensory_units: int = 32
    relational_units: int = 32
    action_units: int = 16
    neuromod_dim: int = 16
    sensory_dim: int = 48  # exteroception + proprioception packed by body interface
    interoceptive_dim: int = 4
    n_motor_channels: int = N_MOTOR_CHANNELS
    n_synergies: int = N_SYNERGIES

    # Connectivity / init motifs (generic)
    init_scale: float = 0.02
    sparsity: float = 0.1
    synergy_channel_gain: float = 1.0

    # Plasticity / developmental rules (generic)
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    eligibility_decay: float = 0.9
    gestational_learning_rate: float = 2e-1
    valence_gain: float = 2.0
    homeostatic_setpoint: float = 0.5

    # Developmental schedule (age / internal signals only)
    gestation_ticks: int = 128
    babble_interval: int = 4
    calibration_onset_frac: float = 0.25
    homeostasis_onset_frac: float = 0.5

    # Credit family identity for matched R4 columns (not a factorial cell from R2/R3)
    credit_family: str = "r2_fixed_eprop_baseline"  # or inherited_learning_signal_generator
    lsg_param_vector: list[float] | None = None

    # Birth RNG seed for embryonic construction — NOT a world seed
    embryonic_seed: int = 0

    # Hippocampus kept small; R4 grounding lives typically disable H
    h_capacity: int = 64
    h_ec_dim: int = 16
    h_dg: int = 64
    h_ca3: int = 32
    h_ca1: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def genome_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def with_credit_family(self, family: str) -> "GenerativeGenome":
        g = GenerativeGenome(**self.to_dict())
        g.credit_family = family
        return g

    def with_size(self, scale: int) -> "GenerativeGenome":
        """
        Scale population counts via genome declaration only.
        scale=1 is the smallest executable size; larger scales multiply units.
        """
        s = max(1, int(scale))
        g = GenerativeGenome(**self.to_dict())
        g.sensory_units = 32 * s
        g.relational_units = 32 * s
        g.action_units = 16 * s
        g.neuromod_dim = max(16, 16 * s)
        g.h_capacity = 64 * s
        g.h_ec_dim = 16 * s
        g.h_dg = 64 * s
        g.h_ca3 = 32 * s
        g.h_ca1 = 16 * s
        return g

    @classmethod
    def small(cls, embryonic_seed: int = 0) -> "GenerativeGenome":
        """Smallest size that instantiates every frozen R4 module."""
        return cls(embryonic_seed=embryonic_seed)
