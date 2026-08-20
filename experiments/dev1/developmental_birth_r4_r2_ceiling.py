"""
R4-R2 reference ceiling on Nursery Body v2.

Canonical ceiling: same-observation learnable policy class (recurrent AC)
with frozen three-initialization battery. Exact-dynamics beam planner is
reference-only, never the scored capacity bound.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiments.dev1.nursery_body_v2_certification import (
    exact_state_from_body,
    model_based_choose_synergy,
    rollout_policy,
)
from three_memory.dev1.device import dev1_device
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.valence import OrganismValenceCircuit
from three_memory.dev1.nursery_v2.metrics import (
    AC_RETRAIN_AGGREGATION,
    AC_RETRAIN_INITIALIZATION_SEEDS,
    AC_RETRAIN_N_INITIALIZATIONS,
    AC_RETRAIN_STOP,
    BehavioralEpisodeGates,
    aggregate_behavioral_gates,
    best_of_ac_initializations,
)
from three_memory.dev1.nursery_v2.physics import BodyConfig
from three_memory.dev1.nursery_v2.synergies import N_SYNERGIES, expand_synergy_index_to_motor
from three_memory.dev1.nursery_v2.world import NurseryWorldV2


class RecurrentAC(nn.Module):
    def __init__(self, sensory_dim: int, hidden: int = 96):
        super().__init__()
        self.in_proj = nn.Linear(sensory_dim, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)
        self.actor = nn.Linear(hidden, N_SYNERGIES)
        self.critic = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, h: torch.Tensor | None):
        z = torch.tanh(self.in_proj(x))
        if h is None:
            h = torch.zeros(self.hidden, device=x.device)
        h2 = self.rnn(z, h)
        return self.actor(h2), self.critic(h2).squeeze(-1), h2


def evaluate_ceiling_on_nursery_world(
    generative: GenerativeGenome,
    world_seed: str,
    *,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    seed: int = 0,
    train_episodes: int | None = None,
    policy: str = "learned",
) -> dict[str, Any]:
    """Single-initialization AC ceiling on Nursery Body v2 (measurement only)."""
    dev = device or torch.device("cpu")
    cfg = BodyConfig(
        n_motor_channels=generative.n_motor_channels,
        n_synergies=generative.n_synergies,
        sensory_dim=generative.sensory_dim,
        interoceptive_dim=generative.interoceptive_dim,
    )
    train_eps = int(train_episodes if train_episodes is not None else max(n_episodes, 48))
    torch.manual_seed(int(seed))
    world = NurseryWorldV2(
        generative=generative,
        world_seed=world_seed,
        device=dev,
        episode_ticks=episode_ticks,
        config=cfg,
    )
    net = RecurrentAC(generative.sensory_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    valence = OrganismValenceCircuit(
        generative.interoceptive_dim, device=dev, gain=4.0, setpoint=0.85
    )

    if policy == "learned":
        for ep in range(train_eps):
            valence.reset()
            step = world.reset_episode(ep)
            h = None
            for _ in range(episode_ticks):
                sensory = step.sensory_vector.to(dev)
                logits, value, h = net(sensory, h)
                dist = torch.distributions.Categorical(logits=logits)
                syn = int(dist.sample().item())
                step = world.apply_action(expand_synergy_index_to_motor(syn, device=dev))
                comfort = float(step.interoceptive_state[0].item())
                comfort_state = torch.full((generative.interoceptive_dim,), comfort, device=dev)
                v = valence.update(comfort_state)
                loss = (
                    -(dist.log_prob(torch.tensor(syn, device=dev)) * (float(v) - float(value.detach())))
                    + 0.5 * (value - float(v)) ** 2
                    - 0.05 * dist.entropy()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                h = h.detach()

    eval_world = NurseryWorldV2(
        generative=generative,
        world_seed=world_seed,
        device=dev,
        episode_ticks=episode_ticks,
        config=cfg,
    )
    episodes: list[BehavioralEpisodeGates] = []
    comfort_ticks = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for ep in range(train_eps, train_eps + n_episodes):
            step = eval_world.reset_episode(ep)
            start_d = float(step.body_state.position.norm().item())
            visited = bool(step.behavioral_correct)
            h = None
            for _ in range(episode_ticks):
                sensory = step.sensory_vector.to(dev)
                if policy == "random":
                    logits = torch.zeros(N_SYNERGIES, device=dev)
                else:
                    logits, _value, h = net(sensory, h)
                syn = int(torch.distributions.Categorical(logits=logits).sample().item())
                step = eval_world.apply_action(expand_synergy_index_to_motor(syn, device=dev))
                comfort_ticks += int(step.behavioral_correct)
                total += 1
                visited = visited or bool(step.behavioral_correct)
            episodes.append(
                BehavioralEpisodeGates(
                    ever_reached=visited,
                    end_in_zone=bool(step.behavioral_correct),
                    start_distance=start_d,
                    end_distance=float(step.body_state.position.norm().item()),
                )
            )
    agg = aggregate_behavioral_gates(episodes)
    return {
        "init_seed": int(seed),
        "policy": policy,
        "end_in_zone_rate": agg["end_in_zone_rate"],
        "ever_reached_rate": agg["ever_reached_rate"],
        "distance_reduction": agg["distance_reduction"],
        "mean_start_distance": agg["mean_start_distance"],
        "mean_end_distance": agg["mean_end_distance"],
        "comfort_rate": comfort_ticks / max(1, total),
        "final_comfort_rate": agg["end_in_zone_rate"],  # ladder alias → end_in_zone
        "treatment_accuracy": agg["end_in_zone_rate"],
        "ceiling_kind": "nursery_v2_recurrent_ac_same_observation",
        "no_expected_action": True,
        "tick_fraction_comfort_retired_as_primary": True,
        "body": "NurseryBodyV2",
    }


def evaluate_ceiling_gate_bundle(
    generative: GenerativeGenome,
    world_seed: str,
    *,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    train_episodes: int | None = None,
) -> dict[str, Any]:
    """
    Reference ceiling on a scored setup: frozen 3-init AC battery + random + beam planner.

    Aggregation is best-of-three by end_in_zone then distance_reduction.
    Never retries until a threshold passes.
    """
    dev = device or dev1_device()
    assert AC_RETRAIN_N_INITIALIZATIONS == len(AC_RETRAIN_INITIALIZATION_SEEDS) == 3
    rows = [
        evaluate_ceiling_on_nursery_world(
            generative,
            world_seed,
            n_episodes=n_episodes,
            episode_ticks=episode_ticks,
            device=dev,
            seed=s,
            train_episodes=train_episodes,
            policy="learned",
        )
        for s in AC_RETRAIN_INITIALIZATION_SEEDS
    ]
    best = best_of_ac_initializations(rows)
    rnd = evaluate_ceiling_on_nursery_world(
        generative,
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        seed=0,
        train_episodes=0,
        policy="random",
    )

    def beam_choose(world, step, ctx):
        return model_based_choose_synergy(exact_state_from_body(world.body), cfg=world.body.config)

    beam = rollout_policy(
        world_seed + ":beam",
        beam_choose,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        config=BodyConfig(
            n_motor_channels=generative.n_motor_channels,
            n_synergies=generative.n_synergies,
            sensory_dim=generative.sensory_dim,
            interoceptive_dim=generative.interoceptive_dim,
        ),
    )
    return {
        **best,
        "comfort_margin_over_random": float(best["end_in_zone_rate"]) - float(rnd["end_in_zone_rate"]),
        "distance_reduction_margin_over_random": float(best["distance_reduction"])
        - float(rnd["distance_reduction"]),
        "random_policy": rnd,
        "exact_dynamics_beam_planner_reference": {
            "end_in_zone_rate": beam.end_in_zone_rate,
            "ever_reached_rate": beam.ever_reached_rate,
            "distance_reduction": beam.distance_reduction,
            "not_scored_capacity_bound": True,
            "not_globally_optimal_oracle": True,
        },
        "ac_retrain_tolerance": {
            "n_initializations": AC_RETRAIN_N_INITIALIZATIONS,
            "initialization_seeds": list(AC_RETRAIN_INITIALIZATION_SEEDS),
            "aggregation": AC_RETRAIN_AGGREGATION,
            "stop_behavior": AC_RETRAIN_STOP,
            "not_retry_until_pass": True,
        },
        "all_initializations": rows,
        "ceiling_kind": "nursery_v2_recurrent_ac_best_of_three",
        "body": "NurseryBodyV2",
    }


def ceiling_body_behavior_passes(ceiling: dict, thresh: dict[str, float]) -> bool:
    return (
        float(ceiling["end_in_zone_rate"]) >= float(thresh["min_end_in_zone_rate"])
        and float(ceiling["distance_reduction"]) >= float(thresh["min_distance_reduction"])
        and float(ceiling["comfort_margin_over_random"]) >= float(thresh["min_margin_over_random"])
    )
