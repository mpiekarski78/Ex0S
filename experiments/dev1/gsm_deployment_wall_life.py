"""
GSM deployment-wall life evaluation.

Learned gated/forced paths reuse gestational FM at SHA 8129ccf.
Exact controllers are measurement-only (experiments/dev1/...controllers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.dev1.gsm_deployment_wall_controllers import (
    HORIZON_TICKS,
    UNCERTAINTY_MAX_PIN,
    choose_exact_one_step_valence,
    choose_exact_receding_horizon,
    distance_region,
)
from experiments.dev1.gsm_life import DEFAULT_UNCERTAINTY_MAX, FALLBACK_POLICY, _assert_finite_tensor
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.gsm.action_eval import calibration_report, choose_synergy_by_valence
from three_memory.dev1.gsm.gestation import PredictiveGestationMode, run_predictive_gestation
from three_memory.dev1.gsm.state import pack_visible_state
from three_memory.dev1.nursery_v2.construction import construct_nursery_organism
from three_memory.dev1.nursery_v2.metrics import BehavioralEpisodeGates, aggregate_behavioral_gates
from three_memory.dev1.nursery_v2.synergies import N_SYNERGIES, expand_synergy_index_to_motor
from three_memory.dev1.nursery_v2.world import NurseryWorldV2


REQUIRED_ARMS = (
    "learned_gated",
    "learned_forced",
    "exact_one_step_valence",
    "exact_receding_horizon",
    "random_fallback",
)

TELEMETRY_FIELDS = (
    "online_prediction_error_vs_gestation_validation_error",
    "trusted_action_coverage_by_time",
    "trusted_action_coverage_by_state_region",
    "predicted_vs_realized_interoceptive_improvement",
    "selected_action_predicted_advantage_sign_correctness",
    "outcomes_conditional_on_gsm_selected_vs_fallback",
    "synergy_histograms",
    "synergy_histogram_early_to_late_change",
)


@dataclass
class DeploymentWallLifeMetrics:
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
    organism_candidate: bool
    measurement_only: bool
    life_record: dict[str, Any] = field(default_factory=dict)


def paired_life_rng_seed(world_seed: str) -> int:
    """Paired randomness across arms for the same world (arm-independent)."""
    return abs(hash(f"gsm_dw_paired:{world_seed}")) % 10_000


def evaluate_deployment_wall_life(
    arm: str,
    world_seed: str,
    *,
    generative: GenerativeGenome | None = None,
    device: torch.device | None = None,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    gestation_ticks: int | None = None,
    embryonic_seed: int = 0,
    body_seed: int = 1,
    life_rng_seed: int | None = None,
    uncertainty_max: float = DEFAULT_UNCERTAINTY_MAX,
    horizon_ticks: int = HORIZON_TICKS,
) -> DeploymentWallLifeMetrics:
    if arm not in REQUIRED_ARMS:
        raise ValueError(f"Unknown deployment-wall arm: {arm!r}")
    if abs(float(uncertainty_max) - float(UNCERTAINTY_MAX_PIN)) > 1e-12:
        raise RuntimeError(
            f"uncertainty_max must remain pinned at {UNCERTAINTY_MAX_PIN}; got {uncertainty_max}"
        )
    if int(horizon_ticks) != int(HORIZON_TICKS):
        raise RuntimeError(f"horizon_ticks must remain pinned at {HORIZON_TICKS}")

    sealed_prefixes = (
        "exos_dev1_developmental_birth_r4_r2_",
        "exos_dev1_gestational_sensorimotor_model_world_",
        "exos_dev1_gestational_sensorimotor_model_r1_world_",
        "exos_dev1_gestational_sensorimotor_model_conf_",
        "exos_dev1_gestational_sensorimotor_model_r1_conf_",
    )
    for p in sealed_prefixes:
        if world_seed.startswith(p):
            raise ValueError(f"sealed prior partition forbidden: {world_seed}")

    dev = device or torch.device("cpu")
    g = generative or GenerativeGenome.small(embryonic_seed=embryonic_seed)
    if gestation_ticks is not None:
        g.gestation_ticks = int(gestation_ticks)
    rng_seed = int(life_rng_seed) if life_rng_seed is not None else paired_life_rng_seed(world_seed)

    org0, creceipt = construct_nursery_organism(g, device=dev)
    pre_hash = ""
    post_hash = ""
    model_updates = 0
    fm = None
    gestation_val_err = float("nan")
    organism_candidate = arm in ("learned_gated", "learned_forced")
    measurement_only = not organism_candidate
    require_trusted = arm == "learned_gated"

    if arm in ("learned_gated", "learned_forced"):
        org, fm, preceipt = run_predictive_gestation(
            org0, g, PredictiveGestationMode.PREDICTIVE, body_seed=body_seed
        )
        model_updates = preceipt.model_updates
        post_hash = preceipt.post_gestation_checkpoint_hash
        pre_hash = preceipt.pre_gestation_checkpoint_hash
        gestation_val_err = float(preceipt.mean_abs_prediction_error)
    else:
        org = org0
        pre_hash = creceipt.generative_genome_hash
        post_hash = creceipt.generative_genome_hash

    world = NurseryWorldV2(
        generative=g, world_seed=world_seed, device=dev, episode_ticks=episode_ticks
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(rng_seed))

    episodes: list[BehavioralEpisodeGates] = []
    unc_sum = 0.0
    model_actions = 0
    fallback_actions = 0
    total_actions = 0
    cal_transitions: list[dict[str, torch.Tensor]] = []
    plasticity_updates = 0
    synergy_hist = [0, 0, 0, 0]
    early_syn = [0, 0, 0, 0]
    late_syn = [0, 0, 0, 0]
    action_sources: list[str] = []
    early_end: list[float] = []
    late_end: list[float] = []
    all_finite = True
    trusted_by_time: list[dict[str, Any]] = []
    trusted_by_region: dict[str, dict[str, int]] = {
        "near_comfort": {"trusted": 0, "untrusted_or_fallback": 0, "n": 0},
        "mid_distance": {"trusted": 0, "untrusted_or_fallback": 0, "n": 0},
        "far": {"trusted": 0, "untrusted_or_fallback": 0, "n": 0},
    }
    pred_vs_real_improvement: list[dict[str, float]] = []
    advantage_sign_rows: list[dict[str, Any]] = []
    cond_gsm: list[BehavioralEpisodeGates] = []
    cond_fallback: list[BehavioralEpisodeGates] = []
    online_errs: list[float] = []

    comfort_r = float(world.body.config.comfort_target_radius)
    early_cut = max(1, int(0.25 * n_episodes * episode_ticks))
    late_start = max(0, int(n_episodes * episode_ticks) - early_cut)
    global_t = 0

    if fm is not None:
        for p in fm.parameters():
            _assert_finite_tensor("forward_model_weight", p.data)
        fm_weight_snapshot = [p.detach().cpu().clone() for p in fm.parameters()]
    else:
        fm_weight_snapshot = []

    for ep in range(n_episodes):
        if hasattr(org, "valence_circuit"):
            org.valence_circuit.reset()
        org.episode_reset()
        step = world.reset_episode(ep)
        start_d = float(step.body_state.position.norm().item())
        visited = bool(step.behavioral_correct)
        obs = world.observation_from_step(step, temporal_context=0.0)
        org.observe(obs)
        ep_used_model = False
        ep_used_fallback = False

        for t in range(episode_ticks):
            sensory_t = step.sensory_vector.detach().clone()
            intero_t = step.interoceptive_state.detach().clone()
            dist0 = float(step.body_state.position.norm().item())
            region = distance_region(dist0, comfort_radius=comfort_r)
            comfort0 = float(org.valence_circuit.comfort(intero_t))

            if arm == "random_fallback":
                syn = int(torch.randint(0, N_SYNERGIES, (1,), generator=gen).item())
                motor = expand_synergy_index_to_motor(syn, device=dev)
                source = "random_fallback"
                fallback_actions += 1
                unc_sum += float("nan")
                trusted_now = False
                imagined_best = float("nan")
                imagined_by_syn: dict[int, float] = {}
            elif arm == "exact_one_step_valence":
                choice = choose_exact_one_step_valence(
                    world.body, org.valence_circuit, device=dev, world=world
                )
                syn = choice.synergy_index
                motor = choice.motor
                source = "exact_one_step_valence"
                unc_sum += 0.0
                trusted_now = True
                imagined_best = float(choice.imagined_valence)
                imagined_by_syn = {}
            elif arm == "exact_receding_horizon":
                choice = choose_exact_receding_horizon(
                    world.body,
                    org.valence_circuit,
                    device=dev,
                    horizon_ticks=horizon_ticks,
                    world=world,
                )
                syn = choice.synergy_index
                motor = choice.motor
                source = "exact_receding_horizon"
                unc_sum += 0.0
                trusted_now = True
                imagined_best = float(choice.imagined_valence)
                imagined_by_syn = {}
            else:
                assert fm is not None
                choice = choose_synergy_by_valence(
                    fm,
                    org.valence_circuit,
                    sensory=sensory_t,
                    intero=intero_t,
                    model_enabled=True,
                    uncertainty_max=float(uncertainty_max),
                    require_trusted=bool(require_trusted),
                    rng=gen,
                )
                motor = choice.motor
                syn = choice.synergy_index
                imagined_by_syn = {
                    e.synergy_index: float(e.imagined_valence) for e in choice.evaluations
                }
                imagined_best = float(imagined_by_syn.get(syn, float("nan")))
                if choice.used_model:
                    model_actions += 1
                    source = "gsm_prediction"
                    trusted_now = True
                    ep_used_model = True
                else:
                    fallback_actions += 1
                    source = f"fallback:{choice.fallback_reason or FALLBACK_POLICY}"
                    trusted_now = False
                    ep_used_fallback = True
                unc_sum += float(choice.max_uncertainty)

            _assert_finite_tensor("chosen_motor", motor)
            action_sources.append(source)
            synergy_hist[syn % 4] += 1
            if global_t < early_cut:
                early_syn[syn % 4] += 1
            if global_t >= late_start:
                late_syn[syn % 4] += 1
            total_actions += 1

            trusted_by_region[region]["n"] += 1
            if trusted_now and source in (
                "gsm_prediction",
                "exact_one_step_valence",
                "exact_receding_horizon",
            ):
                trusted_by_region[region]["trusted"] += 1
            else:
                trusted_by_region[region]["untrusted_or_fallback"] += 1
            trusted_by_time.append(
                {
                    "global_t": global_t,
                    "episode": ep,
                    "tick": t,
                    "trusted": bool(trusted_now),
                    "source": source,
                    "region": region,
                    "distance": dist0,
                }
            )

            step = world.apply_action(motor)
            comfort1 = float(org.valence_circuit.comfort(step.interoceptive_state))
            realized_improvement = comfort1 - comfort0
            if imagined_best == imagined_best:
                pred_vs_real_improvement.append(
                    {
                        "predicted_improvement": float(imagined_best),
                        "realized_improvement": float(realized_improvement),
                    }
                )
                if imagined_by_syn:
                    alts = [v for k, v in imagined_by_syn.items() if k != syn]
                    baseline = sum(alts) / max(1, len(alts)) if alts else 0.0
                    pred_adv = float(imagined_best) - float(baseline)
                else:
                    pred_adv = float(imagined_best)
                sign_ok = (pred_adv >= 0 and realized_improvement >= 0) or (
                    pred_adv < 0 and realized_improvement < 0
                )
                if abs(pred_adv) < 1e-8 or abs(realized_improvement) < 1e-8:
                    sign_ok = True
                advantage_sign_rows.append(
                    {
                        "predicted_advantage": pred_adv,
                        "realized_improvement": float(realized_improvement),
                        "sign_correct": bool(sign_ok),
                        "source": source,
                    }
                )

            cal_transitions.append(
                {
                    "sensory_t": sensory_t,
                    "intero_t": intero_t,
                    "motor": motor.detach().clone(),
                    "sensory_tp1": step.sensory_vector.detach().clone(),
                    "intero_tp1": step.interoceptive_state.detach().clone(),
                }
            )
            if fm is not None:
                with torch.no_grad():
                    s0 = pack_visible_state(sensory=sensory_t, intero=intero_t, dims=fm.dims).to(dev)
                    s1 = pack_visible_state(
                        sensory=step.sensory_vector, intero=step.interoceptive_state, dims=fm.dims
                    ).to(dev)
                    pred = fm.predict_delta(s0, motor)
                    err = float(torch.mean(torch.abs(pred.predicted_state - s1)).item())
                    online_errs.append(err)
                    fm.record_realized_error(err)

            obs = world.observation_from_step(step, temporal_context=float(t + 1))
            org.observe(obs)
            visited = visited or bool(step.behavioral_correct)
            global_t += 1

        end_zone = bool(step.behavioral_correct)
        gate = BehavioralEpisodeGates(
            ever_reached=visited,
            end_in_zone=end_zone,
            start_distance=start_d,
            end_distance=float(step.body_state.position.norm().item()),
        )
        episodes.append(gate)
        if ep_used_model and not ep_used_fallback:
            cond_gsm.append(gate)
        if ep_used_fallback and not ep_used_model:
            cond_fallback.append(gate)
        if ep < n_episodes // 2:
            early_end.append(float(end_zone))
        else:
            late_end.append(float(end_zone))

    if fm is not None:
        for a, b in zip(fm_weight_snapshot, list(fm.parameters())):
            if not torch.equal(a, b.detach().cpu()):
                raise RuntimeError("forward-model weights changed during lifetime evaluation")

    agg = aggregate_behavioral_gates(episodes)
    cal = (
        calibration_report(fm, cal_transitions[-64:])
        if fm is not None
        else {"mean_abs_state_error": float("nan"), "systematic_misprediction_risk": False}
    )
    online_mean = sum(online_errs) / max(1, len(online_errs)) if online_errs else float("nan")
    source_counts: dict[str, int] = {}
    for s in action_sources:
        source_counts[s] = source_counts.get(s, 0) + 1

    sign_rate = (
        sum(1 for r in advantage_sign_rows if r["sign_correct"]) / max(1, len(advantage_sign_rows))
        if advantage_sign_rows
        else float("nan")
    )
    cond = {
        "gsm_selected_episodes": aggregate_behavioral_gates(cond_gsm) if cond_gsm else None,
        "fallback_episodes": aggregate_behavioral_gates(cond_fallback) if cond_fallback else None,
        "n_gsm_only_episodes": len(cond_gsm),
        "n_fallback_only_episodes": len(cond_fallback),
    }

    return DeploymentWallLifeMetrics(
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
        organism_candidate=organism_candidate,
        measurement_only=measurement_only,
        life_record={
            "n_episodes": n_episodes,
            "episode_ticks": episode_ticks,
            "gestation_ticks": int(g.gestation_ticks),
            "uncertainty_max": float(uncertainty_max),
            "horizon_ticks": int(horizon_ticks) if arm == "exact_receding_horizon" else None,
            "require_trusted": bool(require_trusted) if organism_candidate else None,
            "fallback_policy": FALLBACK_POLICY,
            "paired_life_rng_seed": int(rng_seed),
            "embryonic_seed": int(embryonic_seed),
            "body_seed": int(body_seed),
            "organism_candidate": organism_candidate,
            "measurement_only": measurement_only,
            "synergy_histogram": synergy_hist,
            "synergy_histogram_early": early_syn,
            "synergy_histogram_late": late_syn,
            "synergy_histogram_early_to_late_change": [
                late_syn[i] - early_syn[i] for i in range(4)
            ],
            "action_sources": action_sources,
            "action_source_counts": source_counts,
            "early_end_in_zone_rate": sum(early_end) / max(1, len(early_end)),
            "late_end_in_zone_rate": sum(late_end) / max(1, len(late_end)),
            "systematic_misprediction_risk": bool(cal.get("systematic_misprediction_risk")),
            "construction_hash": creceipt.generative_genome_hash,
            "world_hash": world.world_hash(),
            "gestation_validation_mean_abs_error": gestation_val_err,
            "online_mean_abs_prediction_error": online_mean,
            "online_prediction_error_vs_gestation_validation_error": {
                "gestation_validation_mean_abs_error": gestation_val_err,
                "online_mean_abs_prediction_error": online_mean,
                "delta_online_minus_gestation": (
                    online_mean - gestation_val_err
                    if online_mean == online_mean and gestation_val_err == gestation_val_err
                    else float("nan")
                ),
            },
            "trusted_action_coverage_by_time": trusted_by_time,
            "trusted_action_coverage_by_state_region": trusted_by_region,
            "predicted_vs_realized_interoceptive_improvement": pred_vs_real_improvement,
            "selected_action_predicted_advantage_sign_correctness": {
                "n": len(advantage_sign_rows),
                "sign_correct_rate": sign_rate,
                "rows_sample": advantage_sign_rows[:32],
            },
            "outcomes_conditional_on_gsm_selected_vs_fallback": cond,
            "telemetry_fields_present": list(TELEMETRY_FIELDS),
            "plasticity_updates": plasticity_updates,
            "weights_frozen_during_life": True,
        },
    )
