"""
Reference Birth life evaluation and helpers.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.dev1.conventional_ac_ceiling import (
    CEILING_ARM,
    evaluate_ceiling_life,
    ceiling_implementation_hash,
)
from experiments.dev1.probes import probe_permuted_feedback, probe_reward_off, run_causal_decision_ladder
from experiments.dev1.scaffold_r2 import (
    ContinuousScaffoldPhenotype,
    TopologyScaffoldPhenotype,
    apply_scaffold_to_organism,
    normalize_r2_state,
    scaffold_hash,
)
from experiments.dev1.search_r1_1 import _make_world
from three_memory.dev1.device import dev1_device
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism, _build_cortical_plasticity
from three_memory.dev1.plasticity.eprop.interventions import EpropIntervention

REFERENCE_BIRTH_ARMS = [
    "conventional_actor_critic_ceiling",
    "reward_eprop_rate_adaptation",
    "teacher_demo_eprop",
    "r2_1_local_plasticity_control",
]

EPROP_ARMS = {"reward_eprop_rate_adaptation", "teacher_demo_eprop"}

PLASTICITY_SOURCE_FILES = {
    "reward_eprop_rate_adaptation": "three_memory/dev1/plasticity/eprop/reward_eprop.py",
    "teacher_demo_eprop": "three_memory/dev1/plasticity/eprop/reward_eprop.py",
    "r2_1_local_plasticity_control": "three_memory/dev1/plasticity/cortical_plasticity/three_factor.py",
}


@dataclass
class ReferenceBirthLifeMetrics:
    total_fitness: float
    learning_fitness: float
    components: dict[str, float]
    cumulative_reward: float
    treatment_accuracy: float
    reward_off_score: float
    feedback_off_score: float
    permuted_feedback_score: float
    first_failing_causal_predicate: str
    phenotype_hash: str
    scaffold_hash: str
    plasticity_family_name: str
    plasticity_implementation_hash: str
    device: str
    life_record: dict[str, Any] = field(default_factory=dict)


def plasticity_implementation_hash(family: str) -> str:
    if family == CEILING_ARM or family == "conventional_actor_critic_ceiling":
        return ceiling_implementation_hash()
    genome = DevGenome.default()
    if family == "r2_1_local_plasticity_control":
        genome.plasticity_family = "reward_baseline_three_factor"
    elif family in EPROP_ARMS:
        genome.plasticity_family = "reward_eprop_rate_adaptation"
    else:
        genome.plasticity_family = "reward_eprop_rate_adaptation"
    rule = _build_cortical_plasticity(
        genome.plasticity_family,
        genome,
        genome.relational_ctx.n_units,
        genome.action_ctx.n_units,
        device=dev1_device(),
    )
    src = inspect.getsource(type(rule))
    return hashlib.sha256(src.encode()).hexdigest()


def bind_genome_for_arm(arm: str, seed: int | None = None) -> DevGenome:
    genome = DevGenome.default()
    if seed is not None:
        genome.seed = seed
    if arm == "r2_1_local_plasticity_control":
        genome.plasticity_family = "reward_baseline_three_factor"
    elif arm in EPROP_ARMS:
        genome.plasticity_family = arm if arm == "reward_eprop_rate_adaptation" else "teacher_demo_eprop"
    elif arm == "conventional_actor_critic_ceiling":
        pass  # ceiling uses separate model; genome defaults only for world dims
    else:
        raise ValueError(f"unknown arm: {arm}")
    return genome


def _action_entropy(scores: torch.Tensor) -> float:
    p = scores / scores.sum().clamp(min=1e-12)
    ent = -(p * torch.log(p + 1e-12)).sum()
    return float(ent.item())


def _load_control_scaffold() -> tuple[ContinuousScaffoldPhenotype, TopologyScaffoldPhenotype] | None:
    import json

    path = Path("docs/exos_dev1.stage_a_r2_1.control.lock")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("status") != "exact_scaffold_hash_match":
        return None
    return (
        ContinuousScaffoldPhenotype(**data["continuous_scaffold"]),
        TopologyScaffoldPhenotype(**data["topology_scaffold"]),
    )


def evaluate_reference_birth_life(
    arm: str,
    world_seed: str,
    policy_mode: str,
    *,
    continuous: ContinuousScaffoldPhenotype | None = None,
    topology: TopologyScaffoldPhenotype | None = None,
    intervention: EpropIntervention | None = None,
    device: torch.device | None = None,
    h_disabled: bool = True,
    n_episodes: int = 32,
) -> ReferenceBirthLifeMetrics:
    if arm == "conventional_actor_critic_ceiling":
        train = policy_mode == "stochastic"
        cm = evaluate_ceiling_life(
            world_seed,
            policy_mode,
            device=device or dev1_device(),
            n_episodes=n_episodes,
            train_with_autograd=train,
        )
        return ReferenceBirthLifeMetrics(
            total_fitness=cm.learning_fitness,
            learning_fitness=cm.learning_fitness,
            components={"accuracy": cm.treatment_accuracy, "cumulative_reward": cm.cumulative_reward},
            cumulative_reward=cm.cumulative_reward,
            treatment_accuracy=cm.treatment_accuracy,
            reward_off_score=0.0,
            feedback_off_score=0.0,
            permuted_feedback_score=0.0,
            first_failing_causal_predicate="measurement_only_ceiling",
            phenotype_hash=cm.plasticity_implementation_hash,
            scaffold_hash="",
            plasticity_family_name=CEILING_ARM,
            plasticity_implementation_hash=cm.plasticity_implementation_hash,
            device=cm.device,
            life_record=cm.life_record,
        )

    dev = device or dev1_device()
    genome = bind_genome_for_arm(arm)
    world = _make_world(world_seed)
    org = ModularOrganism.birth(genome, device=dev, h_disabled=h_disabled, consolidation_disabled=True)
    cont = continuous or ContinuousScaffoldPhenotype()
    topo = topology or TopologyScaffoldPhenotype()
    if arm == "r2_1_local_plasticity_control":
        loaded = _load_control_scaffold()
        if loaded is not None:
            cont, topo = loaded
    apply_scaffold_to_organism(org, cont, topo)

    if intervention and hasattr(org.plasticity_rule, "set_intervention"):
        org.plasticity_rule.set_intervention(intervention)

    reward_history: list[float] = []
    action_entropy: list[float] = []
    elig_norms: list[float] = []
    update_norms: list[float] = []
    critic_values: list[float] = []
    rpe_values: list[float] = []
    action_hist = torch.zeros(genome.n_motor_channels)
    correct = 0
    total = 0
    pending_teacher: int | None = None
    use_teacher = arm == "teacher_demo_eprop"

    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            org.observe(
                OrganismObservation(
                    sensory_vector=we.sensory_vector,
                    reward=0.0,
                    observed_motor_event=pending_teacher if use_teacher else None,
                )
            )
            pending_teacher = None
            normalize_r2_state(org)
            action = org.act(policy_mode=policy_mode)
            reward = world.reward_for_action(we, action.motor_channel)
            org.observe(OrganismObservation(sensory_vector=we.sensory_vector, reward=reward))
            normalize_r2_state(org)
            credit = org.apply_outcome_credit()
            if use_teacher:
                pending_teacher = we._correct_channel
            elig_norms.append(credit["eligibility_norm_before_credit"])
            update_norms.append(credit["rewarded_update_norm"])
            reward_history.append(reward)
            scores = torch.tensor(action.motor_scores, dtype=torch.float32)
            action_entropy.append(_action_entropy(scores))
            action_hist[action.motor_channel] += 1
            if hasattr(org.plasticity_rule, "critic"):
                critic_values.append(float(org.plasticity_rule.critic.value(org.rho.relational_repr).item()))
            if hasattr(org, "_last_td_error"):
                rpe_values.append(float(org._last_td_error))
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
        org.rest()

    accuracy = correct / max(1, total)
    cumulative_reward = sum(reward_history)
    learning_fitness = accuracy - 0.5 * (1.0 - accuracy)
    reward_off = probe_reward_off(org, world)
    permuted = probe_permuted_feedback(org, world)
    causal = run_causal_decision_ladder(org, world, n_test_episodes=4)
    first_fail = causal[0].causal_label if causal else "unknown"

    sh = scaffold_hash(cont, topo)
    life_record = {
        "arm": arm,
        "world_seed": world_seed,
        "policy_mode": policy_mode,
        "accuracy": accuracy,
        "cumulative_reward": cumulative_reward,
        "action_entropy_mean": sum(action_entropy) / max(1, len(action_entropy)),
        "action_histogram": (action_hist / action_hist.sum().clamp(min=1)).tolist(),
        "eligibility_norm_mean": sum(elig_norms) / max(1, len(elig_norms)),
        "update_norm_mean": sum(update_norms) / max(1, len(update_norms)),
        "critic_value_mean": sum(critic_values) / max(1, len(critic_values)),
        "reward_prediction_error_mean": sum(rpe_values) / max(1, len(rpe_values)),
        "intervention": intervention.name if intervention else "none",
        "device": str(dev),
        "genome_hash": genome.genome_hash(),
        "scaffold_hash": sh,
    }

    return ReferenceBirthLifeMetrics(
        total_fitness=learning_fitness,
        learning_fitness=learning_fitness,
        components={"accuracy": accuracy, "cumulative_reward": cumulative_reward},
        cumulative_reward=cumulative_reward,
        treatment_accuracy=accuracy,
        reward_off_score=reward_off.score,
        feedback_off_score=reward_off.score,
        permuted_feedback_score=permuted.score,
        first_failing_causal_predicate=first_fail,
        phenotype_hash=hashlib.sha256(str(org.action_ctx.W_motor.weight.data.cpu().numpy().tobytes()).encode()).hexdigest(),
        scaffold_hash=sh,
        plasticity_family_name=org.plasticity_rule.name(),
        plasticity_implementation_hash=plasticity_implementation_hash(arm),
        device=str(dev),
        life_record=life_record,
    )


def run_batched_lives_cuda(
    arm: str,
    world_seeds: list[str],
    batch_size: int = 4,
    **kwargs,
) -> list[ReferenceBirthLifeMetrics]:
    """Evaluate independent newborn lives; neural tensors reside on CUDA."""
    dev = dev1_device(require_cuda=torch.cuda.is_available())
    results: list[ReferenceBirthLifeMetrics] = []
    kwargs.pop("device", None)
    for i in range(0, len(world_seeds), batch_size):
        chunk = world_seeds[i : i + batch_size]
        for seed in chunk:
            results.append(evaluate_reference_birth_life(arm, seed, device=dev, **kwargs))
    return results
