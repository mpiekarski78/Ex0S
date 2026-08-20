"""
Task-agnostic gestation: active vs compute-matched sham.

Both cells start from the same post-growth / pre-gestation checkpoint.
Sham uses the same ticks, movements, body states, and compute with
developmental plasticity/calibration disabled.
Zero-tick skip is observational only — not a factorial cell.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from three_memory.dev1.body.physics import BodyConfig, GenericBody
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.interfaces import FullCheckpoint, OrganismObservation
from three_memory.dev1.organism import ModularOrganism


class GestationMode(str, Enum):
    ACTIVE = "active_gestation"
    SHAM = "sham_gestation"
    ZERO_TICK_SKIP = "zero_tick_skip"  # observational diagnostic only


@dataclass
class GestationReceipt:
    mode: str
    ticks: int
    gestation_transcript_hash: str
    post_gestation_checkpoint_hash: str
    pre_gestation_checkpoint_hash: str
    plasticity_updates: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _phenotype_hash(org: ModularOrganism) -> str:
    w = org.action_ctx.W_motor.weight.data.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(org.genome.genome_hash().encode() + w).hexdigest()


def clone_organism_from_checkpoint(
    template: ModularOrganism,
    cp: FullCheckpoint,
    *,
    device: torch.device | None = None,
) -> ModularOrganism:
    """Clone gestational checkpoint before treatment/intervention differences."""
    # Fresh birth then restore ensures separate module instances
    twin = ModularOrganism.birth(
        copy.deepcopy(template.genome),
        device=device or template.device,
        h_disabled=template.hippocampus.h_disabled,
        consolidation_disabled=template.consolidation_disabled,
    )
    twin.restore_from_checkpoint(cp)
    # Re-attach R4 helpers from template
    if hasattr(template, "valence_circuit"):
        twin.valence_circuit = copy.deepcopy(template.valence_circuit)
        twin.valence_circuit.reset()
        if hasattr(twin.valence_circuit, "device"):
            twin.valence_circuit.device = twin.device
    if hasattr(template, "synergy_projection"):
        twin.synergy_projection = template.synergy_projection.detach().clone().to(twin.device)
    twin.generative_genome_hash = getattr(template, "generative_genome_hash", "")
    twin.lifetime_plasticity_enabled = True
    twin.gestational_plasticity_enabled = True
    twin.r4_use_organism_valence = getattr(template, "r4_use_organism_valence", True)
    return twin


def run_gestation(
    org: ModularOrganism,
    generative: GenerativeGenome,
    mode: GestationMode | str,
    *,
    body_seed: int = 0,
    gestational_plasticity_off: bool = False,
) -> tuple[ModularOrganism, GestationReceipt]:
    """
    Run task-agnostic gestation on a cloned organism.

    Active: developmental plasticity enabled (unless gestational_plasticity_off).
    Sham: same ticks/movements/body/compute; developmental plasticity disabled.
    Zero-tick: no ticks (observational only).
    """
    if isinstance(mode, str):
        mode = GestationMode(mode)

    pre_hash = _phenotype_hash(org)
    pre_cp = org.full_checkpoint()
    subject = clone_organism_from_checkpoint(org, pre_cp)

    if mode == GestationMode.ZERO_TICK_SKIP:
        receipt = GestationReceipt(
            mode=mode.value,
            ticks=0,
            gestation_transcript_hash=hashlib.sha256(b"zero_tick").hexdigest(),
            post_gestation_checkpoint_hash=_phenotype_hash(subject),
            pre_gestation_checkpoint_hash=pre_hash,
            plasticity_updates=0,
            metadata={"observational_only": True, "not_a_factorial_cell": True},
        )
        return subject, receipt

    ticks = int(generative.gestation_ticks)
    body = GenericBody(
        BodyConfig(
            n_motor_channels=generative.n_motor_channels,
            n_synergies=generative.n_synergies,
            sensory_dim=generative.sensory_dim,
            interoceptive_dim=generative.interoceptive_dim,
            seed=body_seed,
        ),
        device=subject.device,
    )
    body.reset(seed=body_seed)

    # Precompute homeostatic babble from body physics only (no organism, no labels).
    # Prefer approach-block channels when far from comfort nest; wait-block when near.
    # Sham and active replay this identical sequence (compute-matched).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(body_seed) + 101)
    width = generative.n_motor_channels // generative.n_synergies
    approach_block = list(range(0, width))
    wait_block = list(range(3 * width, 4 * width))
    babble_channels: list[int] = []
    probe = GenericBody(
        BodyConfig(
            n_motor_channels=generative.n_motor_channels,
            n_synergies=generative.n_synergies,
            sensory_dim=generative.sensory_dim,
            interoceptive_dim=generative.interoceptive_dim,
            seed=body_seed,
        ),
        device=torch.device("cpu"),
    )
    probe.reset(seed=body_seed)
    z = torch.zeros(generative.n_motor_channels)
    probe_step = probe.step(z)
    for _t in range(ticks):
        dist = float(probe_step.body_state.position.norm().item())
        if dist > probe.config.comfort_target_radius:
            block = approach_block
        else:
            block = wait_block
        # Small exploration noise across all channels
        if float(torch.rand(1, generator=gen).item()) < 0.15:
            ch = int(torch.randint(0, generative.n_motor_channels, (1,), generator=gen).item())
        else:
            ch = int(block[int(torch.randint(0, len(block), (1,), generator=gen).item())])
        babble_channels.append(ch)
        motor = torch.zeros(generative.n_motor_channels)
        motor[ch] = 1.0
        probe_step = probe.step(motor)

    active_plasticity = mode == GestationMode.ACTIVE and not gestational_plasticity_off
    subject.gestational_plasticity_enabled = active_plasticity
    # During gestation, lifetime plasticity flag follows gestational gate
    subject.lifetime_plasticity_enabled = active_plasticity

    # Preserve / restore gestational LR if active
    orig_lr = float(subject.genome.plasticity.learning_rate)
    orig_clip = float(subject.genome.plasticity.update_clip_scale)
    orig_proj = float(subject.genome.plasticity.projection_scale)
    if active_plasticity:
        subject.genome.plasticity.learning_rate = float(generative.gestational_learning_rate)
        # Allow measurable developmental calibration under matched babble (still local e-prop)
        subject.genome.plasticity.update_clip_scale = max(orig_clip, 2.0)
        subject.genome.plasticity.projection_scale = max(orig_proj, 8.0)

    transcript = []
    plasticity_updates = 0
    zeros = torch.zeros(generative.n_motor_channels, device=subject.device)
    step_res = body.step(zeros)

    for t in range(ticks):
        obs = OrganismObservation(
            sensory_vector=step_res.sensory_vector,
            temporal_context=float(t),
            reward=0.0,
            interoceptive_state=step_res.interoceptive_state,
            proprioceptive_vector=step_res.proprioceptive_vector,
        )
        subject.observe(obs)

        # Matched compute: full cortical act forward pass, then force babble motor for body.
        # Credit targets the babble channel that actually moved the body; keep natural
        # motor logits so policy-error (one_hot_babble - pi) remains informative.
        action = subject.act(policy_mode="stochastic")
        ch = int(babble_channels[t])
        motor = torch.zeros(generative.n_motor_channels, device=subject.device)
        motor[ch] = 1.0
        subject._last_action_channel = ch
        step_res = body.step(motor)

        # Consequence observation with organism valence
        obs2 = OrganismObservation(
            sensory_vector=step_res.sensory_vector,
            temporal_context=float(t) + 0.5,
            reward=0.0,
            interoceptive_state=step_res.interoceptive_state,
            proprioceptive_vector=step_res.proprioceptive_vector,
        )
        subject.observe(obs2)
        credit = subject.apply_outcome_credit()
        update_norm = float(credit.get("rewarded_update_norm", 0.0) or 0.0)
        if credit.get("applied"):
            plasticity_updates += 1

        # Local sensorimotor calibration: potentiate babble channel on positive valence
        # more than depress on negative, so homeostatic babble accumulates.
        valence = float(getattr(subject, "_last_organism_valence", 0.0))
        if active_plasticity and abs(valence) > 0.0:
            with torch.no_grad():
                pre = subject.rho.action_repr.detach()
                if pre.numel() == subject.action_ctx.W_motor.weight.shape[1]:
                    pre_n = pre / (pre.norm() + 1e-6)
                    signed = valence if valence > 0.0 else 0.25 * valence
                    step_gain = 0.08 * float(generative.gestational_learning_rate) * signed
                    subject.action_ctx.W_motor.weight.data[ch].add_(step_gain * pre_n)
                    update_norm = max(update_norm, abs(step_gain))

        transcript.append(
            {
                "t": t,
                "babble_channel": ch,
                "acted_channel": int(action.motor_channel),
                "credit_target_channel": ch,
                "credit_applied": bool(credit.get("applied")),
                "update_norm": update_norm,
                "intero_mean": float(step_res.interoceptive_state.mean().item()),
                "valence": valence,
            }
        )

    subject.genome.plasticity.learning_rate = orig_lr
    subject.genome.plasticity.update_clip_scale = orig_clip
    subject.genome.plasticity.projection_scale = orig_proj
    subject.gestational_plasticity_enabled = True
    subject.lifetime_plasticity_enabled = True
    if hasattr(subject, "valence_circuit"):
        subject.valence_circuit.reset()

    transcript_blob = repr(transcript).encode()
    receipt = GestationReceipt(
        mode=mode.value,
        ticks=ticks,
        gestation_transcript_hash=hashlib.sha256(transcript_blob).hexdigest(),
        post_gestation_checkpoint_hash=_phenotype_hash(subject),
        pre_gestation_checkpoint_hash=pre_hash,
        plasticity_updates=plasticity_updates,
        metadata={
            "gestational_plasticity_off": gestational_plasticity_off,
            "active_plasticity": active_plasticity,
            "body_seed": body_seed,
        },
    )
    return subject, receipt
