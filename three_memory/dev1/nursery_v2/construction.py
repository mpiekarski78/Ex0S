"""Nursery Body v2 organism construction — mass-preserving projection attach."""

from __future__ import annotations

import torch

from three_memory.dev1.development.construction import (
    ConstructionReceipt,
    construct_post_growth_organism,
)
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.nursery_v2.synergies import (
    synergy_projection_matrix,
    synergy_template_hash,
)
from three_memory.dev1.organism import ModularOrganism


def construct_nursery_organism(
    generative: GenerativeGenome,
    *,
    device: torch.device | None = None,
    h_disabled: bool = True,
    consolidation_disabled: bool = True,
) -> tuple[ModularOrganism, ConstructionReceipt]:
    """
    Construct the post-growth organism, then attach Nursery Body v2 mass-preserving
    synergy projection (does not edit historical development.synergies).
    """
    org, receipt = construct_post_growth_organism(
        generative,
        device=device,
        h_disabled=h_disabled,
        consolidation_disabled=consolidation_disabled,
    )
    org.synergy_projection = synergy_projection_matrix(
        generative.n_motor_channels,
        generative.n_synergies,
        gain=float(generative.synergy_channel_gain),
        device=org.device,
    )
    nursery_hash = synergy_template_hash(
        generative.n_motor_channels,
        generative.n_synergies,
        gain=float(generative.synergy_channel_gain),
    )
    receipt = ConstructionReceipt(
        generative_genome_hash=receipt.generative_genome_hash,
        construction_algorithm_hash=receipt.construction_algorithm_hash,
        embryonic_construction_seed=receipt.embryonic_construction_seed,
        pre_gestation_checkpoint_hash=receipt.pre_gestation_checkpoint_hash,
        synergy_template_hash=nursery_hash,
        valence_circuit_hash=receipt.valence_circuit_hash,
        credit_implementation=receipt.credit_implementation,
        metadata={
            **dict(receipt.metadata),
            "synergy_projection": "nursery_v2_mass_preserving",
            "body": "NurseryBodyV2",
        },
    )
    return org, receipt
