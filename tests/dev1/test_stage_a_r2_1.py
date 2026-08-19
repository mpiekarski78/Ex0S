from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.credit_lifecycle_r2_1 import (
    R2_1_CREDIT_FAMILIES,
    bind_genome_for_family,
    plasticity_implementation_hash,
    run_credit_lifecycle_preflight,
    run_r2_1_interaction_step,
)
from experiments.dev1.scaffold_r2 import ContinuousScaffoldPhenotype, TopologyScaffoldPhenotype, apply_scaffold_to_organism
from experiments.dev1.search_r1_1 import _make_world
from experiments.dev1.search_r2_1 import _evaluate_r2_1_life, run_stage_a_r2_1_search
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


def test_r2_1_bind_genome_asserts_family():
    hashes = set()
    for family in R2_1_CREDIT_FAMILIES:
        genome = bind_genome_for_family(family)
        org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
        assert org.plasticity_rule.name() == family
        hashes.add(plasticity_implementation_hash(family))
    assert len(hashes) == len(R2_1_CREDIT_FAMILIES)


def test_r2_1_credit_applies_before_reset():
    genome = bind_genome_for_family("reward_baseline_three_factor")
    world = _make_world("r2_1_test_lifecycle")
    event = world.generate_episode()[0]
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    apply_scaffold_to_organism(org, ContinuousScaffoldPhenotype(), TopologyScaffoldPhenotype())
    w0 = org.action_ctx.W_motor.weight.data.clone()
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=1.0))
    elig_before_credit = float(org.eligibility.trace.norm().item())
    credit = org.apply_outcome_credit()
    delta_before_reset = float((org.action_ctx.W_motor.weight.data - w0).norm().item())
    org.episode_reset()
    elig_after_reset = float(org.eligibility.trace.norm().item())
    assert elig_before_credit > 1e-8
    assert credit["rewarded_update_norm"] > 1e-8
    assert delta_before_reset > 1e-8
    assert elig_after_reset < 1e-8


def test_r2_1_rest_skips_duplicate_plasticity():
    genome = bind_genome_for_family("reward_baseline_three_factor")
    world = _make_world("r2_1_test_rest_skip")
    event = world.generate_episode()[0]
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
    org.act(policy_mode="hard")
    org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=1.0))
    org.apply_outcome_credit()
    w_after_credit = org.action_ctx.W_motor.weight.data.clone()
    org.rest(apply_plasticity=False)
    assert torch.allclose(org.action_ctx.W_motor.weight.data, w_after_credit)


def test_r2_1_families_produce_distinct_updates():
    world = _make_world("r2_1_family_distinct")
    event = world.generate_episode()[0]
    deltas = []
    for family in R2_1_CREDIT_FAMILIES:
        genome = bind_genome_for_family(family)
        org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
        w0 = org.action_ctx.W_motor.weight.data.clone()
        org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=0.0))
        org.act(policy_mode="hard")
        org.observe(OrganismObservation(sensory_vector=event.sensory_vector, reward=1.0))
        org.apply_outcome_credit()
        deltas.append((org.action_ctx.W_motor.weight.data - w0).flatten())
    cos_01 = float(torch.dot(deltas[0], deltas[1]) / (deltas[0].norm() * deltas[1].norm() + 1e-12))
    cos_02 = float(torch.dot(deltas[0], deltas[2]) / (deltas[0].norm() * deltas[2].norm() + 1e-12))
    assert abs(cos_01) < 0.99
    assert abs(cos_02) < 0.99


def test_r2_1_credit_lifecycle_preflight_passes():
    result = run_credit_lifecycle_preflight(seed="r2_1_test_preflight")
    assert result.decision_code == "credit_lifecycle_preflight_pass", result.checks
    assert all(result.checks.values())


def test_r2_1_evaluate_life_records_family_and_hash():
    metrics = _evaluate_r2_1_life(
        "action_contingent_actor_critic",
        ContinuousScaffoldPhenotype(),
        TopologyScaffoldPhenotype(),
        "r2_1_test_life_record",
        "stochastic",
    )
    assert metrics.plasticity_family_name == "action_contingent_actor_critic"
    assert metrics.plasticity_implementation_hash == plasticity_implementation_hash("action_contingent_actor_critic")
    assert metrics.life_record["plasticity_family_name"] == "action_contingent_actor_critic"
    assert metrics.life_record["rewarded_update_norm"] > 0.0


def test_r2_1_runner_smoke_writes_ledger(tmp_path):
    summary = run_stage_a_r2_1_search(
        run_id="r2_1_smoke",
        world_seeds=[f"r2_1_smoke_world_{i:03d}" for i in range(1, 7)],
        confirmation_seeds=[f"r2_1_smoke_conf_{i:03d}" for i in range(1, 5)],
        output_dir=str(tmp_path / "run"),
        preflight_seed="r2_1_smoke_preflight",
        meta_updates=1,
        evo_generations=1,
        population_size=2,
    )
    assert (tmp_path / "run" / "run_started.json").exists()
    assert (tmp_path / "run" / "credit_lifecycle_preflight.json").exists()
    assert (tmp_path / "run" / "run_completed.json").exists()
    assert summary["credit_lifecycle_preflight"] == "credit_lifecycle_preflight_pass"
    assert len(summary["candidates"]) == 9
