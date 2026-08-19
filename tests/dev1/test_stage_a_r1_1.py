from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.optimizers.reward_based_meta_gradient import RewardBasedMetaGradientOptimizer
from experiments.dev1.preflight import PreflightResult
from experiments.dev1.search_r1_1 import (
    FITNESS_RANGES,
    LifeMetrics,
    _make_world,
    _run_newborn_life,
    _signed_reward_signal,
    _train_candidate,
)
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import OrganismObservation
from three_memory.dev1.organism import ModularOrganism


def test_r1_1_life_metrics_are_richer_than_quantized_correctness():
    genome = DevGenome.default()
    world = _make_world("stage_a_r1_1_test_world")
    metrics = _run_newborn_life(genome, world, h_disabled=True, policy_mode="stochastic")
    assert metrics.total_steps == 1024
    assert 0.0 <= metrics.correctness_score <= 1.0
    assert FITNESS_RANGES["fitness_total"][0] <= metrics.normalized_fitness <= FITNESS_RANGES["fitness_total"][1]
    assert metrics.normalized_fitness != metrics.correctness_score
    assert metrics.training_policy == "stochastic"
    assert metrics.train_eval_gap_reported is True
    assert metrics.h_begins_empty
    assert metrics.h_write_counter_zero
    assert metrics.h_read_counter_zero
    assert metrics.h_state_hash_unchanged


def test_reward_based_meta_gradient_reports_loss_and_gradient():
    genome = DevGenome.default()
    opt = RewardBasedMetaGradientOptimizer()
    _, meta = opt.propose(genome)
    opt.update_after_training_lives(meta, normalized_fitness=0.4)
    telem = opt.telemetry()
    assert telem["outer_updates"] == 1
    assert telem["outer_loss"] is not None
    assert telem["gradient_norm"] is not None
    assert telem["gradient_norm"] > 0.0
    assert set(telem["parameter_names"]) == set(genome.credit_parameter_dict().keys())
    assert telem["update_delta_l2"] is not None


def test_zero_return_produces_near_zero_policy_gradient():
    genome = DevGenome.default()
    opt = RewardBasedMetaGradientOptimizer()
    _, meta = opt.propose(genome)
    opt.update_after_training_lives(meta, normalized_fitness=0.0)
    telem = opt.telemetry()
    assert abs(telem["outer_loss"]) <= 1e-12
    assert telem["gradient_norm"] <= 1e-12


def test_reward_inversion_reverses_gradient_direction():
    genome = DevGenome.default()
    opt_pos = RewardBasedMetaGradientOptimizer()
    _, meta_pos = opt_pos.propose(genome)
    before_pos = opt_pos.current_genome(genome).credit_parameter_dict()
    opt_pos.update_after_training_lives(meta_pos, normalized_fitness=0.4)
    after_pos = opt_pos.current_genome(genome).credit_parameter_dict()

    opt_neg = RewardBasedMetaGradientOptimizer()
    _, meta_neg = opt_neg.propose(genome)
    before_neg = opt_neg.current_genome(genome).credit_parameter_dict()
    opt_neg.update_after_training_lives(meta_neg, normalized_fitness=-0.4)
    after_neg = opt_neg.current_genome(genome).credit_parameter_dict()

    deltas_pos = torch.tensor([after_pos[k] - before_pos[k] for k in sorted(before_pos)])
    deltas_neg = torch.tensor([after_neg[k] - before_neg[k] for k in sorted(before_neg)])
    assert torch.dot(deltas_pos, deltas_neg).item() < 0.0


def test_optimizer_step_touches_only_inherited_genome():
    genome = DevGenome.default()
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    obs = OrganismObservation(sensory_vector=[0.0] * 64, reward=1.0)
    org.observe(obs)
    org.act(policy_mode="hard")
    w_before = org.action_ctx.W_motor.weight.detach().clone()
    rho_before = org.rho.relational_repr.detach().clone()
    h_before = org.hippocampus.capacity_telemetry().copy()

    opt = RewardBasedMetaGradientOptimizer()
    _, meta = opt.propose(genome)
    opt.update_after_training_lives(meta, normalized_fitness=0.4)

    assert torch.allclose(w_before, org.action_ctx.W_motor.weight.detach())
    assert torch.allclose(rho_before, org.rho.relational_repr.detach())
    h_after = org.hippocampus.capacity_telemetry()
    assert h_before["write_attempts_total"] == h_after["write_attempts_total"]
    assert h_before["read_attempts_total"] == h_after["read_attempts_total"]


def test_within_life_w_changes_only_through_local_rule_rest_step():
    genome = DevGenome.default()
    org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    w_before = org.action_ctx.W_motor.weight.detach().clone()
    saw_rest_update = False
    for _ in range(8):
        obs = OrganismObservation(sensory_vector=[0.1] * 64, reward=1.0)
        org.observe(obs)
        org.act(policy_mode="stochastic")
        w_after_act = org.action_ctx.W_motor.weight.detach().clone()
        assert torch.allclose(w_before, w_after_act)
        org.rest()
        w_after_rest = org.action_ctx.W_motor.weight.detach().clone()
        if not torch.allclose(w_after_act, w_after_rest):
            saw_rest_update = True
            break
    assert saw_rest_update, "expected at least one within-life W update during rest()"


def test_reward_signal_semantics_zero_and_inversion():
    world = _make_world("stage_a_r1_1_reward_signal")
    assert _signed_reward_signal(0.0, world) == 0.0
    assert _signed_reward_signal(world.cfg.reward_on_correct, world) > 0.0
    assert _signed_reward_signal(world.cfg.reward_on_incorrect, world) < 0.0


def test_score_function_gradient_agrees_with_finite_difference_direction():
    genome = DevGenome.default()
    opt = RewardBasedMetaGradientOptimizer()
    _, meta = opt.propose(genome)
    sample = meta["sample"].detach()
    opt._ensure_state(genome)
    mean = opt._mean.detach().clone()
    std = opt._std.detach().clone()
    fitness = 0.4
    dim = 0
    eps = 1e-4

    mean_param = torch.nn.Parameter(mean.clone())
    dist = torch.distributions.Normal(mean_param, std)
    loss = -(dist.log_prob(sample).sum() * fitness)
    loss.backward()
    analytic = float(mean_param.grad[dim].item())

    def loss_at(mu_delta: float) -> float:
        shifted = mean.clone()
        shifted[dim] += mu_delta
        dist_shifted = torch.distributions.Normal(shifted, std)
        return float((-(dist_shifted.log_prob(sample).sum() * fitness)).item())

    finite = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
    assert analytic != 0.0
    assert finite != 0.0
    assert analytic * finite > 0.0


def test_r1_1_training_controller_spends_budget(monkeypatch):
    import experiments.dev1.search_r1_1 as r11

    def fake_preflight(genome, family, device=None, min_nontrivial_delta_w=1e-8):
        return PreflightResult(True, "preflight_pass", checks={}, metrics={})

    fake_life = LifeMetrics(
        correctness_score=0.0,
        normalized_return=0.25,
        rewarded_improvement=0.1,
        reward_weighted_margin=0.1,
        rewarded_retention=0.25,
        unbounded_penalty=0.0,
        total_reward=0.0,
        total_steps=1024,
        causal_valid=False,
        training_policy="stochastic",
        evaluation_policy="hard",
        train_eval_gap_reported=True,
        h_begins_empty=True,
        h_write_counter_zero=True,
        h_read_counter_zero=True,
        h_state_hash_unchanged=True,
        causal_results=[],
    )

    def fake_eval(genome, train_seeds, h_disabled):
        return [fake_life for _ in train_seeds], list(train_seeds)

    monkeypatch.setattr(r11, "run_credit_preflight", fake_preflight)
    monkeypatch.setattr(r11, "_evaluate_training_lives", fake_eval)

    genome = DevGenome.default()
    genome.plasticity_family = "reward_baseline_three_factor"

    meta_candidate = _train_candidate(
        genome=genome,
        optimizer_arm="reward_based_meta_gradient",
        cheap_train_seeds=["a", "b"],
        meta_updates=3,
        evo_generations=2,
        h_disabled=True,
    )
    assert meta_candidate.optimizer_telemetry["outer_updates"] == 3

    evo_candidate = _train_candidate(
        genome=genome,
        optimizer_arm="evolutionary",
        cheap_train_seeds=["a", "b"],
        meta_updates=2,
        evo_generations=3,
        h_disabled=True,
    )
    assert evo_candidate.optimizer_telemetry["outer_generations"] == 3
