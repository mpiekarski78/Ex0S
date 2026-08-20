"""
Conventional actor-critic ceiling for Reference Birth.

Measurement-only setup probe. NOT an organism candidate.
Autograd is allowed during evaluated lives for this arm only.
Uses identical observations, rewards, episode counts, and action space
as organism treatment lives.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.dev1.search_r1_1 import _make_world
from three_memory.dev1.device import dev1_device
from three_memory.dev1.genome import DevGenome

CEILING_ARM = "conventional_actor_critic_ceiling"
DEFAULT_EPISODES = 32
DEFAULT_HIDDEN = 128


class ConventionalActorCriticCeiling(nn.Module):
    """
    Separate measurement model — not ModularOrganism, not promotable.

    Actor-critic with autograd updates within each evaluated life.
    """

    def __init__(
        self,
        sensory_dim: int,
        n_actions: int,
        hidden_dim: int = DEFAULT_HIDDEN,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device or dev1_device()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(sensory_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, sensory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(sensory)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def name(self) -> str:
        return CEILING_ARM


def ceiling_implementation_hash() -> str:
    src = inspect.getsource(ConventionalActorCriticCeiling)
    return hashlib.sha256(src.encode()).hexdigest()


@dataclass
class CeilingLifeMetrics:
    treatment_accuracy: float
    cumulative_reward: float
    learning_fitness: float
    action_entropy_mean: float
    critic_value_mean: float
    reward_prediction_error_mean: float
    update_norm_mean: float
    device: str
    plasticity_family_name: str
    plasticity_implementation_hash: str
    life_record: dict[str, Any] = field(default_factory=dict)


def _select_action(
    logits: torch.Tensor,
    policy_mode: str,
    generator: torch.Generator | None,
) -> tuple[int, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    if policy_mode == "hard":
        channel = int(torch.argmax(probs).item())
    else:
        channel = int(torch.multinomial(probs, 1, generator=generator).item())
    log_prob = torch.log(probs[channel] + 1e-12)
    return channel, log_prob


def evaluate_ceiling_life(
    world_seed: str,
    policy_mode: str = "stochastic",
    *,
    device: torch.device | None = None,
    n_episodes: int = DEFAULT_EPISODES,
    seed: int = 0,
    train_with_autograd: bool = True,
) -> CeilingLifeMetrics:
    """
    Run one ceiling life with optional within-life autograd (training lives).
    Validation/confirmation lives set train_with_autograd=False for hard eval only.
    """
    dev = device or dev1_device()
    genome = DevGenome.default()
    world = _make_world(world_seed)
    model = ConventionalActorCriticCeiling(
        sensory_dim=genome.sensory_dim,
        n_actions=genome.n_motor_channels,
        device=dev,
        seed=seed,
    )
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + 17)

    correct = 0
    total = 0
    cumulative_reward = 0.0
    entropies: list[float] = []
    critic_vals: list[float] = []
    rpes: list[float] = []
    update_norms: list[float] = []
    action_hist = torch.zeros(genome.n_motor_channels, device=dev)

    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            sensory = torch.tensor(we.sensory_vector, dtype=torch.float32, device=dev)
            if train_with_autograd and policy_mode == "stochastic":
                model.train()
                model.optimizer.zero_grad()
                logits, value = model(sensory)
                channel, log_prob = _select_action(logits, policy_mode, gen)
                reward = world.reward_for_action(we, channel)
                reward_t = torch.tensor(reward, device=dev, dtype=torch.float32)
                td_error = reward_t - value
                actor_loss = -log_prob * td_error.detach()
                critic_loss = td_error.pow(2)
                loss = actor_loss + 0.5 * critic_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model.optimizer.step()
                update_norms.append(float(grad_norm))
                rpes.append(float(td_error.detach().item()))
            else:
                model.eval()
                with torch.no_grad():
                    logits, value = model(sensory)
                    channel, _ = _select_action(logits, policy_mode, gen)
                    reward = world.reward_for_action(we, channel)
                    rpes.append(float(reward - value.item()))

            probs = F.softmax(logits.detach(), dim=-1)
            entropies.append(float(-(probs * torch.log(probs + 1e-12)).sum().item()))
            critic_vals.append(float(value.detach().item() if train_with_autograd else value.item()))
            action_hist[channel] += 1
            cumulative_reward += reward
            if channel == we._correct_channel:
                correct += 1
            total += 1

    accuracy = correct / max(1, total)
    learning_fitness = accuracy - 0.5 * (1.0 - accuracy)
    impl_hash = ceiling_implementation_hash()

    life_record = {
        "arm": CEILING_ARM,
        "world_seed": world_seed,
        "policy_mode": policy_mode,
        "accuracy": accuracy,
        "cumulative_reward": cumulative_reward,
        "action_entropy_mean": sum(entropies) / max(1, len(entropies)),
        "action_histogram": (action_hist / action_hist.sum().clamp(min=1)).detach().cpu().tolist(),
        "critic_value_mean": sum(critic_vals) / max(1, len(critic_vals)),
        "reward_prediction_error_mean": sum(rpes) / max(1, len(rpes)),
        "update_norm_mean": sum(update_norms) / max(1, len(update_norms)) if update_norms else 0.0,
        "train_with_autograd": train_with_autograd,
        "organism_candidate": False,
        "device": str(dev),
        "n_episodes": n_episodes,
        "n_actions": genome.n_motor_channels,
        "plasticity_implementation_hash": impl_hash,
    }

    return CeilingLifeMetrics(
        treatment_accuracy=accuracy,
        cumulative_reward=cumulative_reward,
        learning_fitness=learning_fitness,
        action_entropy_mean=life_record["action_entropy_mean"],
        critic_value_mean=life_record["critic_value_mean"],
        reward_prediction_error_mean=life_record["reward_prediction_error_mean"],
        update_norm_mean=life_record["update_norm_mean"],
        device=str(dev),
        plasticity_family_name=CEILING_ARM,
        plasticity_implementation_hash=impl_hash,
        life_record=life_record,
    )
