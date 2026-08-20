"""
Deterministic embryonic circuit growth from a GenerativeGenome.

Produces a post-growth / pre-gestation ModularOrganism and provenance hashes.
Runner never constructs neural targets.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.synergies import synergy_projection_matrix, synergy_template_hash
from three_memory.dev1.development.valence import OrganismValenceCircuit
from three_memory.dev1.genome import DevGenome, HippocampalSpec, PlasticityCoefficients, PopulationSpec
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.plasticity.eprop.signal_generator import default_lsg_vector


def construction_algorithm_hash() -> str:
    src = inspect.getsource(construct_post_growth_organism)
    return hashlib.sha256(src.encode()).hexdigest()


def _dev_genome_from_generative(g: GenerativeGenome) -> DevGenome:
    plasticity = PlasticityCoefficients(
        learning_rate=g.learning_rate,
        critic_learning_rate=g.critic_learning_rate,
        eligibility_decay=g.eligibility_decay,
    )
    family = g.credit_family
    lsg = g.lsg_param_vector
    if family == "inherited_learning_signal_generator" and lsg is None:
        lsg = default_lsg_vector(g.n_motor_channels, g.action_units, seed=g.embryonic_seed)
    return DevGenome(
        sensory_ctx=PopulationSpec(n_units=g.sensory_units, sparsity=g.sparsity, init_scale=g.init_scale),
        relational_ctx=PopulationSpec(n_units=g.relational_units, sparsity=g.sparsity, init_scale=g.init_scale),
        action_ctx=PopulationSpec(n_units=g.action_units, sparsity=g.sparsity, init_scale=g.init_scale),
        n_motor_channels=g.n_motor_channels,
        neuromod_dim=g.neuromod_dim,
        plasticity=plasticity,
        plasticity_family=family,
        hippocampus=HippocampalSpec(
            capacity=g.h_capacity,
            ec_dim=g.h_ec_dim,
            dg_n_units=g.h_dg,
            ca3_n_units=g.h_ca3,
            ca1_n_units=g.h_ca1,
        ),
        sensory_dim=g.sensory_dim,
        seed=g.embryonic_seed,
        lsg_param_vector=lsg,
    )


def _seed_synergy_biased_motor_readout(org: ModularOrganism, g: GenerativeGenome) -> None:
    """
    Generative topology motif: weak structured bias from action units toward
    contiguous synergy channel blocks. Not a scored cue→action map.
    """
    with torch.no_grad():
        W = org.action_ctx.W_motor.weight.data  # (n_channels, n_action)
        n_ch, n_act = W.shape
        width = n_ch // g.n_synergies
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(g.embryonic_seed) + 17)
        noise = torch.randn(W.shape, generator=gen) * (g.init_scale * 0.5)
        bias = torch.zeros_like(W)
        for s in range(g.n_synergies):
            sl = slice(s * width, (s + 1) * width)
            # Map action-unit subspaces onto synergy blocks
            a0 = (s * n_act) // g.n_synergies
            a1 = ((s + 1) * n_act) // g.n_synergies
            bias[sl, a0:a1] = g.synergy_channel_gain * g.init_scale
        W.copy_(noise.to(W.device) + bias.to(W.device))


@dataclass
class ConstructionReceipt:
    generative_genome_hash: str
    construction_algorithm_hash: str
    embryonic_construction_seed: int
    pre_gestation_checkpoint_hash: str
    synergy_template_hash: str
    valence_circuit_hash: str
    credit_implementation: str
    metadata: dict[str, Any]


def _checkpoint_hash(org: ModularOrganism) -> str:
    cp = org.full_checkpoint()
    # Hash cortex motor weights + genome hash as stable phenotype identity
    w = org.action_ctx.W_motor.weight.data.detach().cpu().contiguous().numpy().tobytes()
    payload = org.genome.genome_hash().encode() + w
    return hashlib.sha256(payload).hexdigest()


def construct_post_growth_organism(
    generative: GenerativeGenome,
    *,
    device: torch.device | None = None,
    h_disabled: bool = True,
    consolidation_disabled: bool = True,
) -> tuple[ModularOrganism, ConstructionReceipt]:
    """
    Embryonic circuit growth → ModularOrganism at post-growth / pre-gestation.

    Both sham and active gestational cells start from a clone of this checkpoint.
    """
    dev = device or torch.device("cpu")
    torch.manual_seed(generative.embryonic_seed)
    genome = _dev_genome_from_generative(generative)
    org = ModularOrganism.birth(
        genome,
        device=dev,
        h_disabled=h_disabled,
        consolidation_disabled=consolidation_disabled,
    )
    _seed_synergy_biased_motor_readout(org, generative)

    # Attach organism-owned valence and synergy projection (body interface helpers)
    org.valence_circuit = OrganismValenceCircuit(
        generative.interoceptive_dim,
        gain=generative.valence_gain,
        setpoint=generative.homeostatic_setpoint,
        device=dev,
    )
    org.synergy_projection = synergy_projection_matrix(
        generative.n_motor_channels,
        generative.n_synergies,
        gain=generative.synergy_channel_gain,
        device=dev,
    )
    org.generative_genome_hash = generative.genome_hash()
    org.lifetime_plasticity_enabled = True
    org.gestational_plasticity_enabled = True
    org.r4_use_organism_valence = True

    receipt = ConstructionReceipt(
        generative_genome_hash=generative.genome_hash(),
        construction_algorithm_hash=construction_algorithm_hash(),
        embryonic_construction_seed=int(generative.embryonic_seed),
        pre_gestation_checkpoint_hash=_checkpoint_hash(org),
        synergy_template_hash=synergy_template_hash(
            generative.n_motor_channels,
            generative.n_synergies,
            gain=generative.synergy_channel_gain,
        ),
        valence_circuit_hash=org.valence_circuit.circuit_hash(),
        credit_implementation=str(genome.plasticity_family),
        metadata={
            "sensory_units": generative.sensory_units,
            "relational_units": generative.relational_units,
            "action_units": generative.action_units,
            "n_motor_channels": generative.n_motor_channels,
            "n_synergies": generative.n_synergies,
        },
    )
    return org, receipt
