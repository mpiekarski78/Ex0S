"""
GSM life evaluation on Nursery Body v2.

Predictive gestation trains a forward model; lifetime action can use
organism-owned valence ranking of predicted synergy consequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import GestationMode
from three_memory.dev1.gsm.action_eval import calibration_report, choose_synergy_by_valence
from three_memory.dev1.gsm.gestation import PredictiveGestationMode, run_predictive_gestation
from three_memory.dev1.gsm.state import pack_visible_state
from three_memory.dev1.nursery_v2.construction import construct_nursery_organism
from three_memory.dev1.nursery_v2.gestation import run_nursery_gestation
from three_memory.dev1.nursery_v2.metrics import BehavioralEpisodeGates, aggregate_behavioral_gates
from three_memory.dev1.nursery_v2.world import NurseryWorldV2


DEFAULT_UNCERTAINTY_MAX = 0.35
FALLBACK_POLICY = "uniform_random_synergy_no_planner"


@dataclass
class GSMLifeMetrics:
    arm: str
    end_in_zone_rate: float
    ever_reached_rate: float
    distance_reduction: float
    mean_model_uncertainty: float
    fraction_model_actions: float
    fraction_fallback_actions: float
    mean_abs_calibration_error: float
    plasticity_updates: int
    model_updates: int
    pre_gestation_checkpoint_hash: str
    post_gestation_checkpoint_hash: str
    all_finite: bool
    life_record: dict[str, Any] = field(default_factory=dict)


def _assert_finite_tensor(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"non-finite values in {name}")


def evaluate_gsm_life(
    arm: str,
    world_seed: str,
    *,
    generative: GenerativeGenome | None = None,
    device: torch.device | None = None,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    gestation_ticks: int | None = None,
    embryonic_seed: int = 0,
    body_seed: int = 0,
    life_rng_seed: int = 0,
    open_loop: bool = False,
    model_off: bool = False,
    valence_off: bool = False,
    uncertainty_max: float = DEFAULT_UNCERTAINTY_MAX,
    force_uncertainty_fallback: bool = False,
) -> GSMLifeMetrics:
    if world_seed.startswith("exos_dev1_developmental_birth_r4_r2_"):
        raise ValueError("R4-R2 scored/confirmation seeds are sealed for GSM")

    dev = device or torch.device("cpu")
    g = generative or GenerativeGenome.small(embryonic_seed=embryonic_seed)
    if gestation_ticks is not None:
        g.gestation_ticks = int(gestation_ticks)
    unc_gate = 0.0 if force_uncertainty_fallback else float(uncertainty_max)
    org0, creceipt = construct_nursery_organism(g, device=dev)

    model_updates = 0
    fm = None
    force_model_off = bool(model_off)
    if arm == "existing_homeostatic_gestation":
        org, greceipt = run_nursery_gestation(
            org0, g, GestationMode.ACTIVE, body_seed=body_seed
        )
        post_hash = greceipt.post_gestation_checkpoint_hash
        pre_hash = greceipt.pre_gestation_checkpoint_hash
        force_model_off = True
    elif arm in (
        "sham_gestation",
        "predictive_gestation",
        "predictive_gestation_shuffled_consequences",
    ):
        mode = {
            "sham_gestation": PredictiveGestationMode.SHAM,
            "predictive_gestation": PredictiveGestationMode.PREDICTIVE,
            "predictive_gestation_shuffled_consequences": PredictiveGestationMode.PREDICTIVE_SHUFFLED,
        }[arm]
        org, fm, preceipt = run_predictive_gestation(org0, g, mode, body_seed=body_seed)
        model_updates = preceipt.model_updates
        post_hash = preceipt.post_gestation_checkpoint_hash
        pre_hash = preceipt.pre_gestation_checkpoint_hash
        if arm == "sham_gestation":
            force_model_off = True
    elif arm == "learned_model_off_at_action_selection":
        org, fm, preceipt = run_predictive_gestation(
            org0, g, PredictiveGestationMode.PREDICTIVE, body_seed=body_seed
        )
        model_updates = preceipt.model_updates
        post_hash = preceipt.post_gestation_checkpoint_hash
        pre_hash = preceipt.pre_gestation_checkpoint_hash
        force_model_off = True
    else:
        raise ValueError(f"Unknown GSM arm: {arm!r}")

    world = NurseryWorldV2(
        generative=g, world_seed=world_seed, device=dev, episode_ticks=episode_ticks
    )
    world.set_open_loop(open_loop)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(life_rng_seed))

    episodes: list[BehavioralEpisodeGates] = []
    unc_sum = 0.0
    model_actions = 0
    fallback_actions = 0
    total_actions = 0
    cal_transitions: list[dict[str, torch.Tensor]] = []
    plasticity_updates = 0
    synergy_hist = [0, 0, 0, 0]
    action_sources: list[str] = []
    early_end = []
    late_end = []
    all_finite = True
    valence_samples: list[float] = []

    use_model = (fm is not None) and (not force_model_off) and hasattr(org, "valence_circuit")
    if fm is not None:
        for p in fm.parameters():
            _assert_finite_tensor("forward_model_weight", p.data)

    for ep in range(n_episodes):
        if hasattr(org, "valence_circuit"):
            org.valence_circuit.reset()
        org.episode_reset()
        step = world.reset_episode(ep)
        start_d = float(step.body_state.position.norm().item())
        visited = bool(step.behavioral_correct)
        obs = world.observation_from_step(step, temporal_context=0.0)
        org.observe(obs)

        for t in range(episode_ticks):
            sensory_t = step.sensory_vector.detach().clone()
            intero_t = step.interoceptive_state.detach().clone()
            if use_model and valence_off:
                # Causal control: keep model available but do not rank by organism valence.
                from three_memory.dev1.nursery_v2.synergies import (
                    N_SYNERGIES,
                    expand_synergy_index_to_motor,
                )

                syn = int(torch.randint(0, N_SYNERGIES, (1,), generator=gen).item())
                motor = expand_synergy_index_to_motor(syn, device=dev)
                source = "valence_off"
                unc_sum += float("nan")
                _assert_finite_tensor("chosen_motor", motor)
            elif use_model:
                choice = choose_synergy_by_valence(
                    fm,
                    org.valence_circuit,
                    sensory=sensory_t,
                    intero=intero_t,
                    model_enabled=True,
                    uncertainty_max=unc_gate,
                    rng=gen,
                )
                motor = choice.motor
                syn = choice.synergy_index
                if choice.used_model:
                    model_actions += 1
                    source = "gsm_prediction"
                    for e in choice.evaluations:
                        valence_samples.append(float(e.imagined_valence))
                        if not (e.imagined_valence == e.imagined_valence):
                            all_finite = False
                else:
                    fallback_actions += 1
                    source = f"fallback:{choice.fallback_reason or FALLBACK_POLICY}"
                unc_sum += float(choice.max_uncertainty)
                _assert_finite_tensor("chosen_motor", motor)
            else:
                action = org.act(policy_mode="stochastic", action_generator=gen)
                motor = torch.as_tensor(action.motor_scores, device=dev, dtype=torch.float32)
                syn = int(getattr(action, "motor_channel", 0)) // max(
                    1, g.n_motor_channels // g.n_synergies
                )
                source = "baseline_policy"
            action_sources.append(source)
            synergy_hist[syn % 4] += 1
            total_actions += 1
            step = world.apply_action(motor)
            cal_transitions.append(
                {
                    "sensory_t": sensory_t,
                    "intero_t": intero_t,
                    "motor": motor.detach().clone(),
                    "sensory_tp1": step.sensory_vector.detach().clone(),
                    "intero_tp1": step.interoceptive_state.detach().clone(),
                }
            )
            if fm is not None and use_model:
                with torch.no_grad():
                    s0 = pack_visible_state(sensory=sensory_t, intero=intero_t, dims=fm.dims).to(dev)
                    s1 = pack_visible_state(
                        sensory=step.sensory_vector, intero=step.interoceptive_state, dims=fm.dims
                    ).to(dev)
                    pred = fm.predict_delta(s0, motor)
                    _assert_finite_tensor("predicted_state", pred.predicted_state)
                    err = float(torch.mean(torch.abs(pred.predicted_state - s1)).item())
                    fm.record_realized_error(err)

            obs = world.observation_from_step(step, temporal_context=float(t + 1))
            org.observe(obs)
            if not use_model:
                credit = org.apply_outcome_credit()
                if credit.get("applied"):
                    plasticity_updates += 1

            visited = visited or bool(step.behavioral_correct)

        end_zone = bool(step.behavioral_correct)
        episodes.append(
            BehavioralEpisodeGates(
                ever_reached=visited,
                end_in_zone=end_zone,
                start_distance=start_d,
                end_distance=float(step.body_state.position.norm().item()),
            )
        )
        if ep < n_episodes // 2:
            early_end.append(float(end_zone))
        else:
            late_end.append(float(end_zone))

    agg = aggregate_behavioral_gates(episodes)
    cal = (
        calibration_report(fm, cal_transitions[-64:])
        if fm is not None
        else {"mean_abs_state_error": float("nan"), "systematic_misprediction_risk": False}
    )
    source_counts: dict[str, int] = {}
    for s in action_sources:
        source_counts[s] = source_counts.get(s, 0) + 1

    return GSMLifeMetrics(
        arm=arm,
        end_in_zone_rate=float(agg["end_in_zone_rate"]),
        ever_reached_rate=float(agg["ever_reached_rate"]),
        distance_reduction=float(agg["distance_reduction"]),
        mean_model_uncertainty=unc_sum / max(1, total_actions),
        fraction_model_actions=model_actions / max(1, total_actions),
        fraction_fallback_actions=fallback_actions / max(1, total_actions),
        mean_abs_calibration_error=float(cal["mean_abs_state_error"]),
        plasticity_updates=plasticity_updates,
        model_updates=model_updates,
        pre_gestation_checkpoint_hash=pre_hash,
        post_gestation_checkpoint_hash=post_hash,
        all_finite=all_finite,
        life_record={
            "n_episodes": n_episodes,
            "episode_ticks": episode_ticks,
            "gestation_ticks": int(g.gestation_ticks),
            "uncertainty_max": unc_gate,
            "fallback_policy": FALLBACK_POLICY,
            "force_uncertainty_fallback": bool(force_uncertainty_fallback),
            "synergy_histogram": synergy_hist,
            "action_sources": action_sources,
            "action_source_counts": source_counts,
            "early_end_in_zone_rate": sum(early_end) / max(1, len(early_end)),
            "late_end_in_zone_rate": sum(late_end) / max(1, len(late_end)),
            "open_loop": open_loop,
            "model_off": force_model_off,
            "valence_off": bool(valence_off),
            "systematic_misprediction_risk": bool(cal.get("systematic_misprediction_risk")),
            "valence_sample_count": len(valence_samples),
            "valence_all_finite": all(
                v == v and abs(v) != float("inf") for v in valence_samples
            ),
            "construction_hash": creceipt.generative_genome_hash,
            "world_hash": world.world_hash(),
            "telemetry_repairs": [
                "synergy_action_histograms",
                "per_episode_early_late_learning_curves",
                "forward_model_prediction_error_over_gestation",
                "per_action_gsm_vs_fallback_source",
            ],
        },
    )
