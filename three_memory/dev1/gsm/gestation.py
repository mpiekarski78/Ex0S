"""
Predictive gestational learning on Nursery Body v2.

Active and sham start from one checkpoint, share identical babble actions,
body transitions, and ticks. Active updates the forward model from observed
next state; sham records transitions but does not update. Shuffled arm breaks
action→outcome correspondence while preserving compute match.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import (
    GestationReceipt,
    _phenotype_hash,
    clone_organism_from_checkpoint,
)
from three_memory.dev1.gsm.forward_model import ForwardSensorimotorModel
from three_memory.dev1.gsm.state import dims_from_body_config, pack_visible_state
from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import expand_synergy_index_to_motor
from three_memory.dev1.organism import ModularOrganism


class PredictiveGestationMode(str, Enum):
    SHAM = "sham_gestation"
    PREDICTIVE = "predictive_gestation"
    PREDICTIVE_SHUFFLED = "predictive_gestation_shuffled_consequences"
    # Kept for comparison arm naming in runners
    HOMEOSTATIC_BASELINE = "existing_homeostatic_gestation"


@dataclass
class PredictiveGestationReceipt:
    mode: str
    ticks: int
    gestation_transcript_hash: str
    post_gestation_checkpoint_hash: str
    pre_gestation_checkpoint_hash: str
    model_updates: int
    mean_abs_prediction_error: float
    prediction_errors: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_gestation_receipt(self) -> GestationReceipt:
        return GestationReceipt(
            mode=self.mode,
            ticks=self.ticks,
            gestation_transcript_hash=self.gestation_transcript_hash,
            post_gestation_checkpoint_hash=self.post_gestation_checkpoint_hash,
            pre_gestation_checkpoint_hash=self.pre_gestation_checkpoint_hash,
            plasticity_updates=self.model_updates,
            metadata={
                **self.metadata,
                "mean_abs_prediction_error": self.mean_abs_prediction_error,
                "gsm": True,
            },
        )


def _build_babble_motors(
    *,
    ticks: int,
    n_motor: int,
    n_synergies: int,
    body_seed: int,
    comfort_radius: float,
    device: torch.device,
) -> list[torch.Tensor]:
    """Egocentric babble sequence (forward when far, rotate when near)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(body_seed) + 101)
    width = n_motor // n_synergies
    forward_block = list(range(0, width))
    rotate_block = list(range(2 * width, 3 * width))
    probe = NurseryBodyV2(
        BodyConfig(n_motor_channels=n_motor, n_synergies=n_synergies, seed=body_seed),
        device=torch.device("cpu"),
    )
    probe.reset(seed=body_seed)
    step = probe.step(torch.zeros(n_motor))
    motors: list[torch.Tensor] = []
    for _ in range(ticks):
        dist = float(step.body_state.position.norm().item())
        block = forward_block if dist > comfort_radius else rotate_block
        if float(torch.rand(1, generator=gen).item()) < 0.15:
            ch = int(torch.randint(0, n_motor, (1,), generator=gen).item())
        else:
            ch = int(block[int(torch.randint(0, len(block), (1,), generator=gen).item())])
        motor = torch.zeros(n_motor, device=device)
        motor[ch] = 1.0
        motors.append(motor)
        step = probe.step(motor.cpu())
    return motors


def run_predictive_gestation(
    org: ModularOrganism,
    generative: GenerativeGenome,
    mode: PredictiveGestationMode | str,
    *,
    body_seed: int = 0,
    lr: float = 3e-3,
    model: ForwardSensorimotorModel | None = None,
) -> tuple[ModularOrganism, ForwardSensorimotorModel, PredictiveGestationReceipt]:
    if isinstance(mode, str):
        mode = PredictiveGestationMode(mode)

    pre_hash = _phenotype_hash(org)
    pre_cp = org.full_checkpoint()
    subject = clone_organism_from_checkpoint(org, pre_cp)

    dims = dims_from_body_config(
        sensory_dim=generative.sensory_dim,
        proprioceptive_dim=8,
        interoceptive_dim=generative.interoceptive_dim,
        n_motor_channels=generative.n_motor_channels,
    )
    fm = model or ForwardSensorimotorModel(dims, device=subject.device)
    fm = fm.to(subject.device)

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
    motors = _build_babble_motors(
        ticks=ticks,
        n_motor=generative.n_motor_channels,
        n_synergies=generative.n_synergies,
        body_seed=body_seed,
        comfort_radius=body.config.comfort_target_radius,
        device=subject.device,
    )

    # Precompute actual transitions under matched babble (shared across arms).
    zeros = torch.zeros(generative.n_motor_channels, device=subject.device)
    step = body.step(zeros)
    transitions: list[dict[str, torch.Tensor]] = []
    for t in range(ticks):
        sensory_t = step.sensory_vector.detach().clone()
        intero_t = step.interoceptive_state.detach().clone()
        motor = motors[t]
        step = body.step(motor)
        transitions.append(
            {
                "sensory_t": sensory_t,
                "intero_t": intero_t,
                "motor": motor.detach().clone(),
                "sensory_tp1": step.sensory_vector.detach().clone(),
                "intero_tp1": step.interoceptive_state.detach().clone(),
                "synergy_proxy": int(torch.argmax(motor).item()) // max(
                    1, generative.n_motor_channels // generative.n_synergies
                ),
            }
        )

    learn = mode in (
        PredictiveGestationMode.PREDICTIVE,
        PredictiveGestationMode.PREDICTIVE_SHUFFLED,
    )
    shuffled = mode == PredictiveGestationMode.PREDICTIVE_SHUFFLED
    # Sham and homeostatic baseline naming: no forward-model updates here.
    if mode in (PredictiveGestationMode.SHAM, PredictiveGestationMode.HOMEOSTATIC_BASELINE):
        learn = False
        shuffled = False

    train_motors = [tr["motor"].clone() for tr in transitions]
    if shuffled:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(body_seed) + 707)
        order = torch.randperm(len(train_motors), generator=gen).tolist()
        train_motors = [train_motors[i] for i in order]

    opt = torch.optim.Adam(fm.parameters(), lr=float(lr)) if learn else None
    errors: list[float] = []
    updates = 0
    transcript: list[dict[str, Any]] = []

    for t, tr in enumerate(transitions):
        motor_for_pair = train_motors[t]
        # Always compute prediction error for telemetry (even sham).
        with torch.no_grad():
            s0 = pack_visible_state(
                sensory=tr["sensory_t"], intero=tr["intero_t"], dims=dims
            ).to(subject.device)
            s1 = pack_visible_state(
                sensory=tr["sensory_tp1"], intero=tr["intero_tp1"], dims=dims
            ).to(subject.device)
            pred = fm.predict_delta(s0, motor_for_pair)
            pred_delta = torch.cat([pred.delta_exo, pred.delta_proprio, pred.delta_intero])
            abs_err = float(torch.mean(torch.abs(pred_delta - (s1 - s0))).item())
            errors.append(abs_err)
            fm.record_realized_error(abs_err)

        if learn and opt is not None:
            loss, stats = fm.loss_on_transition(
                sensory_t=tr["sensory_t"],
                intero_t=tr["intero_t"],
                motor=motor_for_pair,
                sensory_tp1=tr["sensory_tp1"],
                intero_tp1=tr["intero_tp1"],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            transcript.append({"t": t, "abs_err": abs_err, **stats, "updated": True})
        else:
            transcript.append({"t": t, "abs_err": abs_err, "updated": False})

    subject.gsm_forward_model = fm
    subject.gsm_model_enabled_for_action = True

    receipt = PredictiveGestationReceipt(
        mode=mode.value,
        ticks=ticks,
        gestation_transcript_hash=hashlib.sha256(repr(transcript).encode()).hexdigest(),
        post_gestation_checkpoint_hash=_phenotype_hash(subject),
        pre_gestation_checkpoint_hash=pre_hash,
        model_updates=updates,
        mean_abs_prediction_error=sum(errors) / max(1, len(errors)),
        prediction_errors=errors,
        metadata={
            "body_seed": body_seed,
            "learn": learn,
            "shuffled_consequences": shuffled,
            "body": "NurseryBodyV2",
            "predicts_delta": True,
            "n_transitions": len(transitions),
        },
    )
    return subject, fm, receipt


def attach_fresh_forward_model(
    org: ModularOrganism,
    generative: GenerativeGenome,
) -> ForwardSensorimotorModel:
    dims = dims_from_body_config(
        sensory_dim=generative.sensory_dim,
        proprioceptive_dim=8,
        interoceptive_dim=generative.interoceptive_dim,
        n_motor_channels=generative.n_motor_channels,
    )
    fm = ForwardSensorimotorModel(dims, device=org.device)
    org.gsm_forward_model = fm
    org.gsm_model_enabled_for_action = False
    return fm
