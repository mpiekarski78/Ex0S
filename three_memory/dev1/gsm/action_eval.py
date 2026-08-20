"""
Organism-owned action evaluation via gestational forward model.

Enumerates the inherited four synergies, predicts consequences, ranks by
organism valence. Runner never supplies the preferred action.
Model-off reverts to matched baseline (stochastic policy without model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from three_memory.dev1.development.valence import OrganismValenceCircuit
from three_memory.dev1.gsm.forward_model import ForwardSensorimotorModel
from three_memory.dev1.gsm.state import (
    VisibleDims,
    dims_from_body_config,
    pack_visible_state,
    unpack_visible_state,
)
from three_memory.dev1.nursery_v2.synergies import N_SYNERGIES, expand_synergy_index_to_motor


@dataclass
class SynergyEvaluation:
    synergy_index: int
    predicted_intero: torch.Tensor
    predicted_comfort: float
    imagined_valence: float
    uncertainty: float
    trusted: bool


@dataclass
class ModelActionChoice:
    synergy_index: int
    motor: torch.Tensor
    evaluations: list[SynergyEvaluation]
    used_model: bool
    fallback_reason: str
    max_uncertainty: float


def evaluate_synergies_with_model(
    fm: ForwardSensorimotorModel,
    valence: OrganismValenceCircuit,
    *,
    sensory: torch.Tensor,
    intero: torch.Tensor,
    uncertainty_max: float = 0.35,  # frozen default; prereg pins this value
) -> list[SynergyEvaluation]:
    dims = fm.dims
    state = pack_visible_state(sensory=sensory, intero=intero, dims=dims).to(fm.device)
    current_comfort = valence.comfort(intero)
    out: list[SynergyEvaluation] = []
    for syn in range(N_SYNERGIES):
        motor = expand_synergy_index_to_motor(syn, device=fm.device)
        pred = fm.predict_delta(state, motor)
        intero_hat = unpack_visible_state(pred.predicted_state, dims)[2]
        comfort_hat = valence.comfort(intero_hat)
        imagined = float(valence.gain) * (comfort_hat - current_comfort)
        trusted = float(pred.uncertainty) <= float(uncertainty_max)
        out.append(
            SynergyEvaluation(
                synergy_index=syn,
                predicted_intero=intero_hat.detach(),
                predicted_comfort=comfort_hat,
                imagined_valence=imagined,
                uncertainty=float(pred.uncertainty),
                trusted=trusted,
            )
        )
    return out


def choose_synergy_by_valence(
    fm: ForwardSensorimotorModel | None,
    valence: OrganismValenceCircuit,
    *,
    sensory: torch.Tensor,
    intero: torch.Tensor,
    model_enabled: bool = True,
    uncertainty_max: float = 0.35,
    require_trusted: bool = True,
    rng: torch.Generator | None = None,
) -> ModelActionChoice:
    """
    a* = argmax_a V(s_hat(s, a)) over four synergies.

    If model is off / missing / all predictions untrusted, fall back to uniform
    random synergy (matched baseline without model exploitation).
    """
    device = sensory.device if hasattr(sensory, "device") else torch.device("cpu")
    if (not model_enabled) or fm is None:
        syn = int(torch.randint(0, N_SYNERGIES, (1,), generator=rng).item())
        motor = expand_synergy_index_to_motor(syn, device=device)
        return ModelActionChoice(
            synergy_index=syn,
            motor=motor,
            evaluations=[],
            used_model=False,
            fallback_reason="model_off",
            max_uncertainty=float("inf"),
        )

    evals = evaluate_synergies_with_model(
        fm, valence, sensory=sensory, intero=intero, uncertainty_max=uncertainty_max
    )
    max_u = max(e.uncertainty for e in evals)
    trusted = [e for e in evals if e.trusted] if require_trusted else list(evals)
    if not trusted:
        syn = int(torch.randint(0, N_SYNERGIES, (1,), generator=rng).item())
        motor = expand_synergy_index_to_motor(syn, device=device)
        return ModelActionChoice(
            synergy_index=syn,
            motor=motor,
            evaluations=evals,
            used_model=False,
            fallback_reason="all_predictions_untrusted",
            max_uncertainty=max_u,
        )

    best = max(trusted, key=lambda e: e.imagined_valence)
    motor = expand_synergy_index_to_motor(best.synergy_index, device=device)
    return ModelActionChoice(
        synergy_index=int(best.synergy_index),
        motor=motor,
        evaluations=evals,
        used_model=True,
        fallback_reason="",
        max_uncertainty=max_u,
    )


def calibration_report(
    fm: ForwardSensorimotorModel,
    transitions: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Compare imagined next states with realized transitions (exploitation safety)."""
    errs = []
    for tr in transitions:
        with torch.no_grad():
            s0 = pack_visible_state(
                sensory=tr["sensory_t"], intero=tr["intero_t"], dims=fm.dims
            ).to(fm.device)
            s1 = pack_visible_state(
                sensory=tr["sensory_tp1"], intero=tr["intero_tp1"], dims=fm.dims
            ).to(fm.device)
            pred = fm.predict_delta(s0, tr["motor"])
            errs.append(float(torch.mean(torch.abs(pred.predicted_state - s1)).item()))
    mean_err = sum(errs) / max(1, len(errs))
    return {
        "n": len(errs),
        "mean_abs_state_error": mean_err,
        "p95_abs_state_error": sorted(errs)[max(0, int(0.95 * len(errs)) - 1)] if errs else 0.0,
        "systematic_misprediction_risk": mean_err > 0.35,
    }
