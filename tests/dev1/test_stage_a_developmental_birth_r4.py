"""Developmental Birth R4 unit / ownership / leakage / permutation / scale tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.developmental_birth_r4_life import (
    evaluate_matched_factorial,
    evaluate_r4_life,
)
from experiments.dev1.developmental_birth_r4_outer import MatchedOuterBudget, run_matched_es_smoke
from experiments.dev1.developmental_birth_r4_preflight import (
    ownership_leakage_check,
    sham_vs_active_compute_match_check,
)
from three_memory.dev1.body.world import ClosedLoopGroundingWorld
from three_memory.dev1.development.construction import construct_post_growth_organism
from three_memory.dev1.development.generative_genome import GenerativeGenome, SYNERGY_REPORT_NAMES
from three_memory.dev1.development.gestation import GestationMode, run_gestation
from three_memory.dev1.development.synergies import synergy_projection_matrix
from three_memory.dev1.development.valence import OrganismValenceCircuit
from three_memory.dev1.interfaces import OrganismObservation


def test_generative_construction_deterministic():
    g = GenerativeGenome.small(embryonic_seed=11)
    a, ra = construct_post_growth_organism(g, device=torch.device("cpu"))
    b, rb = construct_post_growth_organism(g, device=torch.device("cpu"))
    assert ra.pre_gestation_checkpoint_hash == rb.pre_gestation_checkpoint_hash
    assert ra.generative_genome_hash == g.genome_hash()
    assert torch.allclose(a.action_ctx.W_motor.weight.data, b.action_ctx.W_motor.weight.data)


def test_synergy_names_runner_only():
    g = GenerativeGenome.small()
    blob = str(g.to_dict())
    for name in SYNERGY_REPORT_NAMES:
        assert name not in blob
    P = synergy_projection_matrix()
    assert P.shape == (4, 32)


def test_organism_valence_owns_reinforcement():
    circuit = OrganismValenceCircuit(4, gain=1.0, setpoint=0.5)
    near = torch.tensor([0.5, 0.5, 0.5, 0.5])
    far = torch.tensor([0.9, 0.9, 0.9, 0.9])
    v0 = circuit.update(near)
    assert v0 == 0.0  # first observation baseline
    v1 = circuit.update(far)
    # comfort drops (farther from setpoint) → negative valence
    assert v1 < 0.0


def test_runner_score_not_in_learning_path():
    g = GenerativeGenome.small(embryonic_seed=3)
    org, _ = construct_post_growth_organism(g, device=torch.device("cpu"))
    world = ClosedLoopGroundingWorld(g, world_seed="own", device=torch.device("cpu"))
    step = world.reset_episode(0)
    # Poison reward field with behavioral correctness — organism must ignore when valence on
    obs = OrganismObservation(
        sensory_vector=step.sensory_vector,
        reward=999.0,
        interoceptive_state=step.interoceptive_state,
    )
    org.observe(obs)
    org.act(policy_mode="hard")
    step2 = world.apply_action(torch.zeros(g.n_motor_channels))
    obs2 = OrganismObservation(
        sensory_vector=step2.sensory_vector,
        reward=999.0,
        interoceptive_state=step2.interoceptive_state,
    )
    org.observe(obs2)
    assert abs(org._last_consequence_reward) < 100.0  # valence-scale, not 999


def test_sham_gestation_matched_ticks_no_plasticity():
    out = sham_vs_active_compute_match_check()
    assert out["ok"]
    assert out["sham_no_credit"]
    assert out["active_has_credit"]


def test_lifetime_plasticity_off_blocks_updates():
    m = evaluate_r4_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "test_lpo",
        n_episodes=2,
        episode_ticks=4,
        lifetime_plasticity_off=True,
        use_teacher=False,
        device=torch.device("cpu"),
    )
    assert m.plasticity_updates == 0


def test_gestational_plasticity_off():
    g = GenerativeGenome.small(embryonic_seed=5)
    g.gestation_ticks = 12
    org, _ = construct_post_growth_organism(g, device=torch.device("cpu"))
    _, receipt = run_gestation(
        org, g, GestationMode.ACTIVE, body_seed=1, gestational_plasticity_off=True
    )
    assert receipt.plasticity_updates == 0


def test_ownership_leakage():
    out = ownership_leakage_check()
    assert out["ok"]


def test_lsg_off_intervention():
    m = evaluate_r4_life(
        "active_gestation",
        "inherited_learning_signal_generator",
        "test_lsg_off",
        n_episodes=2,
        episode_ticks=4,
        lsg_off=True,
        use_teacher=False,
        device=torch.device("cpu"),
    )
    assert m.credit == "inherited_learning_signal_generator"
    assert m.intervention == "lsg_off"


def test_motor_and_teacher_permutations_run():
    m1 = evaluate_r4_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "test_mot_perm",
        n_episodes=1,
        episode_ticks=4,
        motor_permutation=True,
        use_teacher=False,
        device=torch.device("cpu"),
    )
    m2 = evaluate_r4_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "test_teach_perm",
        n_episodes=1,
        episode_ticks=4,
        permute_teacher=True,
        device=torch.device("cpu"),
    )
    assert m1.intervention == "motor_permutation"
    assert m2.intervention == "teacher_permutation"


def test_matched_factorial_four_cells():
    cells = evaluate_matched_factorial(
        "test_factorial",
        n_episodes=1,
        episode_ticks=2,
        embryonic_seed=0,
        device=torch.device("cpu"),
    )
    assert len(cells) == 4
    # Within each credit column, sham and active share the same pre-gestation checkpoint
    for credit in ("r2_fixed_eprop_baseline", "inherited_learning_signal_generator"):
        sham = cells[f"sham_gestation__{credit}"]
        active = cells[f"active_gestation__{credit}"]
        assert sham.pre_gestation_checkpoint_hash == active.pre_gestation_checkpoint_hash
        assert sham.embryonic_seed == active.embryonic_seed
    # All cells share embryonic seed and body physics class
    seeds = {v.embryonic_seed for v in cells.values()}
    assert seeds == {0}


def test_matched_outer_budgets():
    out = run_matched_es_smoke(
        "test_es",
        MatchedOuterBudget(population=2, generations=1, n_episodes=1, episode_ticks=2),
        device=torch.device("cpu"),
    )
    assert out["matched"]
    assert out["fixed_lives"] == out["lsg_lives"]


def test_scale_smoke_same_algorithm():
    small = GenerativeGenome.small(embryonic_seed=9)
    big = small.with_size(2)
    assert big.sensory_units == 64
    _, r1 = construct_post_growth_organism(small, device=torch.device("cpu"))
    _, r2 = construct_post_growth_organism(big, device=torch.device("cpu"))
    assert r1.construction_algorithm_hash == r2.construction_algorithm_hash


def test_zero_tick_skip_observational():
    m = evaluate_r4_life(
        "zero_tick_skip",
        "r2_fixed_eprop_baseline",
        "test_zero",
        n_episodes=1,
        episode_ticks=2,
        use_teacher=False,
        device=torch.device("cpu"),
    )
    assert m.life_record["gestation_mode"] == "zero_tick_skip"
