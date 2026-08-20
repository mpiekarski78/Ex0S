"""
Nursery Body v2 gestation — egocentric homeostatic babble.

Same compute-matched sham/active protocol as historical R4 gestation, but uses
NurseryBodyV2 and forward/rotate babble (never approach/wait).
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import (
    GestationMode,
    GestationReceipt,
    _phenotype_hash,
    clone_organism_from_checkpoint,
)
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.organism import ModularOrganism


def run_nursery_gestation(
    org: ModularOrganism,
    generative: GenerativeGenome,
    mode: GestationMode | str,
    *,
    body_seed: int = 0,
    gestational_plasticity_off: bool = False,
) -> tuple[ModularOrganism, GestationReceipt]:
    if isinstance(mode, str):
        mode = GestationMode(mode)

    pre_hash = _phenotype_hash(org)
    pre_cp = org.full_checkpoint()
    subject = clone_organism_from_checkpoint(org, pre_cp)

    if mode == GestationMode.ZERO_TICK_SKIP:
        receipt = GestationReceipt(
            mode=mode.value,
            ticks=0,
            gestation_transcript_hash=hashlib.sha256(b"zero_tick_nursery_v2").hexdigest(),
            post_gestation_checkpoint_hash=_phenotype_hash(subject),
            pre_gestation_checkpoint_hash=pre_hash,
            plasticity_updates=0,
            metadata={
                "observational_only": True,
                "not_a_factorial_cell": True,
                "body": "NurseryBodyV2",
            },
        )
        return subject, receipt

    ticks = int(generative.gestation_ticks)
    body = NurseryBodyV2(
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

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(body_seed) + 101)
    width = generative.n_motor_channels // generative.n_synergies
    forward_block = list(range(0, width))
    rotate_block = list(range(2 * width, 3 * width))
    babble_channels: list[int] = []
    probe = NurseryBodyV2(
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
        block = forward_block if dist > probe.config.comfort_target_radius else rotate_block
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
    subject.lifetime_plasticity_enabled = active_plasticity

    orig_lr = float(subject.genome.plasticity.learning_rate)
    orig_clip = float(subject.genome.plasticity.update_clip_scale)
    orig_proj = float(subject.genome.plasticity.projection_scale)
    if active_plasticity:
        subject.genome.plasticity.learning_rate = float(generative.gestational_learning_rate)
        subject.genome.plasticity.update_clip_scale = max(orig_clip, 2.0)
        subject.genome.plasticity.projection_scale = max(orig_proj, 8.0)

    transcript: list[dict[str, Any]] = []
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

        action = subject.act(policy_mode="stochastic")
        ch = int(babble_channels[t])
        motor = torch.zeros(generative.n_motor_channels, device=subject.device)
        motor[ch] = 1.0
        subject._last_action_channel = ch
        step_res = body.step(motor)

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

    receipt = GestationReceipt(
        mode=mode.value,
        ticks=ticks,
        gestation_transcript_hash=hashlib.sha256(repr(transcript).encode()).hexdigest(),
        post_gestation_checkpoint_hash=_phenotype_hash(subject),
        pre_gestation_checkpoint_hash=pre_hash,
        plasticity_updates=plasticity_updates,
        metadata={
            "gestational_plasticity_off": gestational_plasticity_off,
            "active_plasticity": active_plasticity,
            "body_seed": body_seed,
            "body": "NurseryBodyV2",
            "babble": "egocentric_forward_rotate",
        },
    )
    return subject, receipt
