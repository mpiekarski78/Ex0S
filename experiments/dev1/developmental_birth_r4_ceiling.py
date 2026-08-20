"""
Canonical R4-R1 learnability ceiling: synergy-aware actor–critic.

Measurement repair only. Body physics, gestation, organism credit, and genome
construction remain frozen. The ceiling is never an organism candidate and never
receives an expected-action / answer table.

Action space is four motor synergies. Synergy commands expand through the same
fixed motor basis the body uses, so synonym channels share behavioral credit.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from three_memory.dev1.body.world import ClosedLoopGroundingWorld
from three_memory.dev1.device import dev1_device
from three_memory.dev1.development.generative_genome import (
    N_MOTOR_CHANNELS,
    N_SYNERGIES,
    GenerativeGenome,
)
from three_memory.dev1.development.synergies import synergy_channel_blocks
from three_memory.dev1.development.valence import OrganismValenceCircuit


def expand_synergy_index_to_motor(
    synergy_index: int,
    *,
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
    device: torch.device | None = None,
    encoding: str = "uniform_block",
    channel_within_block: int = 0,
) -> torch.Tensor:
    """
    Expand one synergy choice into a 32-D motor command.

    encodings
    - uniform_block: mass 1/width on every channel in the synergy block
      (body-equivalent to any one-hot inside the block under frozen projection)
    - onehot_in_block: one-hot on one channel inside the synergy block
    """
    dev = device or torch.device("cpu")
    blocks = synergy_channel_blocks(n_channels, n_synergies)
    s = int(synergy_index) % n_synergies
    sl = blocks[s]
    motor = torch.zeros(n_channels, device=dev)
    width = sl.stop - sl.start
    if encoding == "uniform_block":
        motor[sl] = 1.0 / float(width)
    elif encoding == "onehot_in_block":
        motor[sl.start + (int(channel_within_block) % width)] = 1.0
    else:
        raise ValueError(f"unknown encoding: {encoding}")
    return motor


def expand_synergy_probs_to_motor(
    synergy_probs: torch.Tensor,
    *,
    n_channels: int = N_MOTOR_CHANNELS,
    n_synergies: int = N_SYNERGIES,
) -> torch.Tensor:
    """Distributed synergy encoding: p_s / width on each channel of block s."""
    probs = synergy_probs.detach().float().view(-1)
    if probs.numel() != n_synergies:
        raise ValueError("synergy_probs must have n_synergies entries")
    blocks = synergy_channel_blocks(n_channels, n_synergies)
    motor = torch.zeros(n_channels, device=probs.device)
    for s, sl in enumerate(blocks):
        width = sl.stop - sl.start
        motor[sl] = float(probs[s].item()) / float(width)
    return motor


def permute_channels_within_synergy(
    motor: torch.Tensor,
    *,
    n_synergies: int = N_SYNERGIES,
    perm_seed: int = 0,
) -> torch.Tensor:
    """Permute channel order inside each synergy block; leave block boundaries fixed."""
    n_channels = int(motor.numel())
    out = motor.detach().clone()
    blocks = synergy_channel_blocks(n_channels, n_synergies)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(perm_seed))
    for sl in blocks:
        width = sl.stop - sl.start
        order = torch.randperm(width, generator=gen)
        block = out[sl].clone()
        out[sl] = block[order]
    return out


class SynergyBodyWorldCeiling(nn.Module):
    """
    Measurement-only actor–critic over four synergies on the shared body.

    Never an organism candidate. No privileged answer table.
    """

    def __init__(
        self,
        sensory_dim: int,
        n_synergies: int = N_SYNERGIES,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_synergies = int(n_synergies)
        torch.manual_seed(seed)
        self.enc = nn.Sequential(
            nn.Linear(sensory_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, self.n_synergies)
        self.critic = nn.Linear(64, 1)
        self.to(self.device)
        # Measurement-only AC: slightly higher LR + entropy keeps synonym-rich
        # 4-way exploration from collapsing into one synergy.
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-3)
        self.entropy_coef = 0.05

    def forward(self, sensory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(sensory)
        return self.actor(h), self.critic(h).squeeze(-1)


def _episode_start_metrics(step) -> tuple[float, float]:
    dist = float(step.body_state.position.norm().item())
    comfort = float(step.behavioral_score)
    return dist, comfort


def evaluate_ceiling_on_body_world(
    generative: GenerativeGenome,
    world_seed: str,
    *,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    motor_encoding: str = "uniform_block",
    train: bool = True,
    seed: int = 0,
    policy: str = "learned",
) -> dict[str, Any]:
    """
    Synergy-aware ceiling on the same body observations / actions / organism valence.

    Returns body-behavior metrics used by the R4-R1 gate (not raw channel accuracy).
    """
    if generative.n_synergies != N_SYNERGIES or generative.n_motor_channels != N_MOTOR_CHANNELS:
        raise ValueError("R4-R1 ceiling expects frozen 4-synergy / 32-channel body")

    dev = device or dev1_device()
    world = ClosedLoopGroundingWorld(
        generative, world_seed=world_seed, device=dev, episode_ticks=episode_ticks
    )
    ceiling = SynergyBodyWorldCeiling(
        generative.sensory_dim, generative.n_synergies, dev, seed=seed
    )
    valence = OrganismValenceCircuit(generative.interoceptive_dim, device=dev)

    correct = 0
    total = 0
    comfort_sum = 0.0
    dist_sum = 0.0
    start_comfort_sum = 0.0
    start_dist_sum = 0.0
    end_comfort_sum = 0.0
    end_dist_sum = 0.0
    valence_sum = 0.0
    synergy_hist = [0 for _ in range(generative.n_synergies)]
    first_half_correct = 0
    second_half_correct = 0
    mid = max(1, (n_episodes * episode_ticks) // 2)

    for ep in range(n_episodes):
        valence.reset()
        step = world.reset_episode(ep)
        start_dist, start_comfort = _episode_start_metrics(step)
        start_dist_sum += start_dist
        start_comfort_sum += start_comfort
        last_dist = start_dist
        last_comfort = start_comfort

        for _t in range(episode_ticks):
            sensory = step.sensory_vector.to(dev)
            logits, value = ceiling.forward(sensory)
            if policy == "random":
                dist = torch.distributions.Categorical(
                    logits=torch.zeros(generative.n_synergies, device=dev)
                )
            else:
                dist = torch.distributions.Categorical(logits=logits)
            syn_idx = int(dist.sample().item())
            synergy_hist[syn_idx] += 1
            motor = expand_synergy_index_to_motor(
                syn_idx,
                n_channels=generative.n_motor_channels,
                n_synergies=generative.n_synergies,
                device=dev,
                encoding=motor_encoding,
            )
            step = world.apply_action(motor)
            v = valence.update(step.interoceptive_state)
            valence_sum += float(v)

            if train and policy == "learned":
                logp = dist.log_prob(torch.tensor(syn_idx, device=dev))
                adv = float(v) - float(value.detach().item())
                entropy = dist.entropy()
                loss = (
                    -(logp * adv)
                    + 0.5 * (value - float(v)) ** 2
                    - float(ceiling.entropy_coef) * entropy
                )
                ceiling.opt.zero_grad()
                loss.backward()
                ceiling.opt.step()

            ok = int(step.behavioral_correct)
            correct += ok
            total += 1
            if total <= mid:
                first_half_correct += ok
            else:
                second_half_correct += ok
            comfort_sum += float(step.behavioral_score)
            dist_sum += float(step.body_state.position.norm().item())
            last_dist = float(step.body_state.position.norm().item())
            last_comfort = float(step.behavioral_score)

        end_dist_sum += last_dist
        end_comfort_sum += last_comfort

    n_ep = max(1, n_episodes)
    n_tot = max(1, total)
    final_comfort_rate = correct / n_tot
    initial_comfort_rate = start_comfort_sum / n_ep  # episode-start comfort mean (not rate)
    # Start-in-zone rate uses comfort radius via behavioral_correct on tick-0 state before action:
    # approximate with fraction of episodes whose start distance is already in zone.
    # For gate we use comfort_rate improvement vs first-half → second-half and vs random.
    mean_start_dist = start_dist_sum / n_ep
    mean_end_dist = end_dist_sum / n_ep
    mean_start_comfort = start_comfort_sum / n_ep
    mean_end_comfort = end_comfort_sum / n_ep
    first_half_rate = first_half_correct / max(1, mid)
    second_half_rate = second_half_correct / max(1, n_tot - mid)

    # Late-window comfort rate is the body-competence summary for all policies.
    gate_comfort_rate = second_half_rate
    return {
        "treatment_accuracy": gate_comfort_rate,
        "final_comfort_rate": gate_comfort_rate,
        "full_trajectory_comfort_rate": final_comfort_rate,
        "mean_comfort": comfort_sum / n_tot,
        "mean_distance": dist_sum / n_tot,
        "mean_start_distance": mean_start_dist,
        "mean_end_distance": mean_end_dist,
        "mean_start_comfort": mean_start_comfort,
        "mean_end_comfort": mean_end_comfort,
        "comfort_improvement": mean_end_comfort - mean_start_comfort,
        "distance_reduction": mean_start_dist - mean_end_dist,
        "first_half_comfort_rate": first_half_rate,
        "second_half_comfort_rate": second_half_rate,
        "within_life_comfort_rate_improvement": second_half_rate - first_half_rate,
        "mean_organism_valence": valence_sum / n_tot,
        "synergy_histogram": synergy_hist,
        "n_synergies": generative.n_synergies,
        "n_motor_channels": generative.n_motor_channels,
        "motor_encoding": motor_encoding,
        "policy": policy,
        "trained": bool(train and policy == "learned"),
        "ceiling_kind": "synergy_aware_r4_r1",
        "no_expected_action": True,
    }


def evaluate_body_controllability_probe(
    generative: GenerativeGenome,
    world_seed: str,
    *,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Closed-loop heuristic over the four synergies (diagnostic only).

    Never used as a privileged answer table for ceiling or organism training.
    Establishes whether body actions can systematically reduce nest distance.
    """
    dev = device or dev1_device()
    world = ClosedLoopGroundingWorld(
        generative, world_seed=world_seed + ":controllability", device=dev, episode_ticks=episode_ticks
    )
    correct = 0
    total = 0
    start_d = 0.0
    end_d = 0.0
    # Open-loop block effects for differential controllability
    open_loop_end_dist = {}
    for syn_name, syn_idx in (("approach", 0), ("withdraw", 1), ("orient", 2), ("wait", 3)):
        w = ClosedLoopGroundingWorld(
            generative,
            world_seed=world_seed + f":ol_{syn_name}",
            device=dev,
            episode_ticks=episode_ticks,
        )
        dists = []
        for ep in range(min(4, n_episodes)):
            step = w.reset_episode(ep)
            for _ in range(episode_ticks):
                step = w.apply_action(
                    expand_synergy_index_to_motor(syn_idx, device=dev, encoding="uniform_block")
                )
            dists.append(float(step.body_state.position.norm().item()))
        open_loop_end_dist[syn_name] = sum(dists) / max(1, len(dists))

    for ep in range(n_episodes):
        step = world.reset_episode(ep)
        start_d += float(step.body_state.position.norm().item())
        last_d = float(step.body_state.position.norm().item())
        for _ in range(episode_ticks):
            pos = step.body_state.position
            dist = float(pos.norm().item())
            if dist < world.body.config.comfort_target_radius:
                syn = 3
            elif abs(float(pos[0].item())) >= abs(float(pos[1].item())):
                syn = 1 if float(pos[0].item()) > 0 else 0
            else:
                syn = 2
            step = world.apply_action(
                expand_synergy_index_to_motor(syn, device=dev, encoding="uniform_block")
            )
            correct += int(step.behavioral_correct)
            total += 1
            last_d = float(step.body_state.position.norm().item())
        end_d += last_d

    n_ep = max(1, n_episodes)
    mean_start = start_d / n_ep
    mean_end = end_d / n_ep
    open_spread = max(open_loop_end_dist.values()) - min(open_loop_end_dist.values())
    return {
        "heuristic_comfort_rate": correct / max(1, total),
        "mean_start_distance": mean_start,
        "mean_end_distance": mean_end,
        "distance_reduction": mean_start - mean_end,
        "open_loop_end_distance_by_synergy": open_loop_end_dist,
        "open_loop_synergy_distance_spread": open_spread,
        "no_expected_action_for_training": True,
        "probe_kind": "synergy_closed_loop_heuristic_diagnostic",
    }


def body_controllability_passes(probe: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        float(probe["distance_reduction"]) >= float(thresholds["min_heuristic_distance_reduction"])
        and float(probe["open_loop_synergy_distance_spread"])
        >= float(thresholds["min_open_loop_synergy_distance_spread"])
    )


def evaluate_ceiling_gate_bundle(
    generative: GenerativeGenome,
    world_seed: str,
    *,
    n_episodes: int = 8,
    episode_ticks: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Learned synergy ceiling + matched untrained/random policy on the same seed.

    Gate inputs are body-behavior metrics only.
    """
    learned = evaluate_ceiling_on_body_world(
        generative,
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=device,
        train=True,
        policy="learned",
        seed=0,
    )
    random_policy = evaluate_ceiling_on_body_world(
        generative,
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=device,
        train=False,
        policy="random",
        seed=1,
    )
    untrained = evaluate_ceiling_on_body_world(
        generative,
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=device,
        train=False,
        policy="learned",
        seed=0,
    )
    margin_random = learned["final_comfort_rate"] - random_policy["final_comfort_rate"]
    margin_untrained = learned["final_comfort_rate"] - untrained["final_comfort_rate"]
    bundle = {
        "learned": learned,
        "random_policy": random_policy,
        "untrained_policy": untrained,
        "final_comfort_rate": learned["final_comfort_rate"],
        "comfort_improvement": learned["comfort_improvement"],
        "distance_reduction": learned["distance_reduction"],
        "within_life_comfort_rate_improvement": learned["within_life_comfort_rate_improvement"],
        "comfort_margin_over_random": margin_random,
        "comfort_margin_over_untrained": margin_untrained,
        "treatment_accuracy": learned["final_comfort_rate"],
        "ceiling_kind": "synergy_aware_r4_r1_gate_bundle",
    }
    return bundle


def ceiling_body_behavior_passes(bundle: dict[str, Any], thresholds: dict[str, float]) -> bool:
    """Explicit R4-R1 body-behavior gate (all conjuncts required)."""
    return (
        float(bundle["final_comfort_rate"]) >= float(thresholds["min_final_comfort_rate"])
        and float(bundle["comfort_improvement"]) >= float(thresholds["min_comfort_improvement"])
        and float(bundle["distance_reduction"]) >= float(thresholds["min_distance_reduction"])
        and float(bundle["comfort_margin_over_random"]) >= float(thresholds["min_margin_over_random"])
    )
