"""
Developmental Birth R4 life evaluation: matched 2×2 factorial on one shared body.

All four cells share the same R4 body, world physics, synergies, genome schema,
birth seeds, and scoring. Historical R2/R3 results are lineage anchors only —
not factorial cells.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.dev1.developmental_birth_r4_ceiling import (
    evaluate_ceiling_gate_bundle,
    evaluate_ceiling_on_body_world,
)
from three_memory.dev1.body.world import ClosedLoopGroundingWorld
from three_memory.dev1.device import dev1_device
from three_memory.dev1.development.construction import construct_post_growth_organism
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import GestationMode, run_gestation
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention
from three_memory.dev1.plasticity.eprop.signal_generator import default_lsg_vector

# Re-export for runners that import ceiling helpers from the life module.
__all_ceiling__ = (
    "evaluate_ceiling_on_body_world",
    "evaluate_ceiling_gate_bundle",
)


FACTORIAL_CELLS = (
    ("sham_gestation", "r2_fixed_eprop_baseline"),
    ("sham_gestation", "inherited_learning_signal_generator"),
    ("active_gestation", "r2_fixed_eprop_baseline"),
    ("active_gestation", "inherited_learning_signal_generator"),
)


@dataclass
class R4LifeMetrics:
    development: str
    credit: str
    treatment_accuracy: float
    mean_behavioral_score: float
    mean_organism_valence: float
    signed_margin_proxy: float
    phenotype_hash: str
    generative_genome_hash: str
    construction_algorithm_hash: str
    embryonic_seed: int
    pre_gestation_checkpoint_hash: str
    gestation_transcript_hash: str
    post_gestation_checkpoint_hash: str
    body_physics_hash: str
    credit_implementation: str
    plasticity_updates: int
    teacher_demo_count: int
    device: str
    intervention: str
    life_record: dict[str, Any] = field(default_factory=dict)


def _apply_eprop_intervention(org: ModularOrganism, intervention: EpropIntervention) -> None:
    rule = org.plasticity_rule
    if hasattr(rule, "set_intervention"):
        rule.set_intervention(intervention)
    elif intervention.reward_off:
        # Fixed e-prop path: zero consequence reward via valence bypass
        org.r4_use_organism_valence = False


def evaluate_r4_life(
    development: str,
    credit: str,
    world_seed: str,
    *,
    generative: GenerativeGenome | None = None,
    intervention_name: str = "none",
    device: torch.device | None = None,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    embryonic_seed: int = 0,
    body_seed: int = 0,
    life_rng_seed: int = 0,
    use_teacher: bool = True,
    permute_teacher: bool = False,
    motor_permutation: bool = False,
    proprio_permutation: bool = False,
    open_loop: bool = False,
    gestational_plasticity_off: bool = False,
    lifetime_plasticity_off: bool = False,
    reward_valence_off: bool = False,
    lsg_off: bool = False,
    lsg_permuted: bool = False,
) -> R4LifeMetrics:
    """
    Evaluate one matched R4 cell (or intervention variant) on the shared body world.
    """
    if development == "zero_tick_skip":
        gest_mode = GestationMode.ZERO_TICK_SKIP
    elif development == "sham_gestation":
        gest_mode = GestationMode.SHAM
    elif development == "active_gestation":
        gest_mode = GestationMode.ACTIVE
    else:
        raise ValueError(f"Unknown development factor: {development!r}")

    if credit not in ("r2_fixed_eprop_baseline", "inherited_learning_signal_generator"):
        raise ValueError(f"Unknown credit column: {credit!r}")

    dev = device or torch.device("cpu")
    g = generative or GenerativeGenome.small(embryonic_seed=embryonic_seed)
    g = g.with_credit_family(credit)
    if credit == "inherited_learning_signal_generator" and g.lsg_param_vector is None:
        g.lsg_param_vector = default_lsg_vector(g.n_motor_channels, g.action_units, seed=embryonic_seed)

    org0, creceipt = construct_post_growth_organism(g, device=dev)
    org, greceipt = run_gestation(
        org0,
        g,
        gest_mode,
        body_seed=body_seed,
        gestational_plasticity_off=gestational_plasticity_off,
    )

    if lifetime_plasticity_off:
        org.lifetime_plasticity_enabled = False

    if reward_valence_off:
        org.r4_use_organism_valence = False

    intervention = EpropIntervention.none()
    if lsg_off:
        intervention = EpropIntervention.with_signal_generator_off()
    elif lsg_permuted:
        intervention = EpropIntervention.with_signal_generator_permuted()
    elif intervention_name == "reward_off":
        intervention = EpropIntervention.with_reward_off()
        org.r4_use_organism_valence = False
    _apply_eprop_intervention(org, intervention)

    world = ClosedLoopGroundingWorld(g, world_seed=world_seed, device=dev, episode_ticks=episode_ticks)
    world.set_teacher_permutation(permute_teacher, seed=life_rng_seed)
    world.set_motor_permutation(motor_permutation, seed=life_rng_seed)
    world.set_proprio_permutation(proprio_permutation, seed=life_rng_seed)
    world.set_open_loop(open_loop)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(life_rng_seed))

    correct = 0
    total = 0
    score_sum = 0.0
    valence_sum = 0.0
    margin_proxy = 0.0
    teacher_demos = 0
    plasticity_updates = 0

    for ep in range(n_episodes):
        if hasattr(org, "valence_circuit"):
            org.valence_circuit.reset()
        org.episode_reset()
        step = world.reset_episode(ep)
        obs = world.observation_from_step(step, temporal_context=0.0)
        org.observe(obs)

        for t in range(episode_ticks):
            action = org.act(policy_mode="stochastic", action_generator=gen)
            scores = torch.as_tensor(action.motor_scores, device=dev, dtype=torch.float32)
            step = world.apply_action(scores)
            obs = world.observation_from_step(step, temporal_context=float(t + 1))
            org.observe(obs)
            credit_out = org.apply_outcome_credit()
            if credit_out.get("applied"):
                plasticity_updates += 1
                pre_m = credit_out.get("pre_rewarded_action_margin")
                post_m = credit_out.get("post_rewarded_action_margin")
                if pre_m is not None and post_m is not None:
                    sign = credit_out.get("outcome_sign", 0)
                    margin_proxy += float(sign) * (float(post_m) - float(pre_m))

            correct += int(step.behavioral_correct)
            total += 1
            score_sum += float(step.behavioral_score)
            valence_sum += float(getattr(org, "_last_organism_valence", 0.0))

            if use_teacher and (t % 4 == 3):
                demo = world.suggested_teacher_channel(step)
                org.observe(
                    world.observation_from_step(
                        step,
                        temporal_context=float(t + 1.25),
                        observed_motor_event=demo,
                    )
                )
                # Teacher consequence uses organism valence at next body state after demo motor
                demo_step = world.apply_action(scores, teacher_channel=demo)
                teach_obs = world.observation_from_step(
                    demo_step,
                    temporal_context=float(t + 1.5),
                    teaching_signal=float(getattr(org, "_last_organism_valence", 0.0)),
                )
                # Recompute valence for teaching_signal without putting behavioral score in
                if hasattr(org, "valence_circuit") and org.r4_use_organism_valence:
                    # teaching_signal already set; observe will also update valence — use teaching_signal field
                    teach_obs.teaching_signal = 1.0 if float(demo_step.behavioral_score) > 0.3 else -0.5
                    # Contract: teaching_signal is scalar consequence, not fixture answer. Body-derived ok.
                    # Prefer organism valence delta:
                    teach_obs.teaching_signal = float(
                        org.valence_circuit.comfort(demo_step.interoceptive_state)
                        - org.valence_circuit.comfort(step.interoceptive_state)
                    )
                org.observe(teach_obs)
                tcredit = org.apply_teacher_demonstration_credit()
                if tcredit.get("applied"):
                    teacher_demos += 1
                step = demo_step

    acc = correct / max(1, total)
    w = org.action_ctx.W_motor.weight.data.detach().cpu().contiguous().numpy().tobytes()
    pheno = hashlib.sha256(org.genome.genome_hash().encode() + w).hexdigest()

    return R4LifeMetrics(
        development=development,
        credit=credit,
        treatment_accuracy=acc,
        mean_behavioral_score=score_sum / max(1, total),
        mean_organism_valence=valence_sum / max(1, total),
        signed_margin_proxy=margin_proxy / max(1, plasticity_updates),
        phenotype_hash=pheno,
        generative_genome_hash=creceipt.generative_genome_hash,
        construction_algorithm_hash=creceipt.construction_algorithm_hash,
        embryonic_seed=creceipt.embryonic_construction_seed,
        pre_gestation_checkpoint_hash=creceipt.pre_gestation_checkpoint_hash,
        gestation_transcript_hash=greceipt.gestation_transcript_hash,
        post_gestation_checkpoint_hash=greceipt.post_gestation_checkpoint_hash,
        body_physics_hash=world.body.physics_hash(),
        credit_implementation=creceipt.credit_implementation,
        plasticity_updates=plasticity_updates,
        teacher_demo_count=teacher_demos,
        device=str(dev),
        intervention=intervention_name if intervention_name != "none" else (
            "lsg_off" if lsg_off else "lsg_permuted" if lsg_permuted else
            "lifetime_plasticity_off" if lifetime_plasticity_off else
            "gestational_plasticity_off" if gestational_plasticity_off else
            "reward_valence_off" if reward_valence_off else
            "motor_permutation" if motor_permutation else
            "proprio_permutation" if proprio_permutation else
            "open_loop" if open_loop else
            "teacher_permutation" if permute_teacher else "none"
        ),
        life_record={
            "n_episodes": n_episodes,
            "episode_ticks": episode_ticks,
            "gestation_mode": greceipt.mode,
            "gestation_plasticity_updates": greceipt.plasticity_updates,
            "world_hash": world.world_hash(),
            "synergy_template_hash": creceipt.synergy_template_hash,
            "valence_circuit_hash": creceipt.valence_circuit_hash,
        },
    )


def evaluate_matched_factorial(
    world_seed: str,
    *,
    n_episodes: int = 4,
    episode_ticks: int = 8,
    embryonic_seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, R4LifeMetrics]:
    """Run all four matched cells with shared seeds/budgets."""
    out: dict[str, R4LifeMetrics] = {}
    for development, credit in FACTORIAL_CELLS:
        key = f"{development}__{credit}"
        out[key] = evaluate_r4_life(
            development,
            credit,
            world_seed,
            n_episodes=n_episodes,
            episode_ticks=episode_ticks,
            embryonic_seed=embryonic_seed,
            device=device,
        )
    return out
