"""
Developmental Body Reference Wall — measurement only.

Diagnoses the frozen GenericBody / four-synergy learnability contract before any
developmental-organism claim. Does not edit organism, genome, gestation, or credit.

The fixed handcrafted synergy heuristic is NOT used as a capacity bound.
The model-based controller is an exact-dynamics beam planner over the four synergies.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from experiments.dev1.developmental_birth_r4_ceiling import expand_synergy_index_to_motor
from three_memory.dev1.body.physics import BodyConfig, GenericBody
from three_memory.dev1.body.world import ClosedLoopGroundingWorld
from three_memory.dev1.device import dev1_device
from three_memory.dev1.development.generative_genome import (
    N_MOTOR_CHANNELS,
    N_SYNERGIES,
    GenerativeGenome,
)
from three_memory.dev1.development.valence import OrganismValenceCircuit

SYNERGY_NAMES = ("approach", "withdraw", "orient", "wait")
STEP_SCALE = 0.15
ORIENT_SCALE = 0.2
SYN_ACT = 1.0 / 8.0  # uniform_block / onehot_in_block under frozen projection


@dataclass(frozen=True)
class ExactState:
    x: float
    y: float
    orientation: float
    energy: float

    @property
    def dist(self) -> float:
        return math.hypot(self.x, self.y)


def exact_synergy_transition(
    state: ExactState,
    synergy_index: int,
    *,
    arena_radius: float = 2.0,
    drive_decay: float = 0.02,
) -> ExactState:
    """
    Exact copy of GenericBody.step dynamics for a single synergy pulse
    expanded as uniform_block / onehot_in_block (synergy activation = 1/8).
    """
    approach = withdraw = orient = wait = 0.0
    s = int(synergy_index) % N_SYNERGIES
    if s == 0:
        approach = SYN_ACT
    elif s == 1:
        withdraw = SYN_ACT
    elif s == 2:
        orient = SYN_ACT
    else:
        wait = SYN_ACT

    dx = (approach - withdraw) * STEP_SCALE
    dy = math.sin(state.orientation) * orient * STEP_SCALE
    motion_gate = max(0.05, 1.0 - 0.5 * wait)
    nx = state.x + dx * motion_gate
    ny = state.y + dy * motion_gate
    r = math.hypot(nx, ny)
    if r > arena_radius:
        nx *= arena_radius / r
        ny *= arena_radius / r
    nori = state.orientation + (orient - 0.5 * wait) * ORIENT_SCALE
    energy = max(0.0, min(1.0, state.energy - drive_decay + 0.03 * wait))
    return ExactState(nx, ny, nori, energy)


def exact_state_from_body(body: GenericBody) -> ExactState:
    st = body.state
    energy = float(st.interoception[1].item()) if st.interoception.numel() > 1 else 0.7
    return ExactState(
        float(st.position[0].item()),
        float(st.position[1].item()),
        float(st.orientation),
        energy,
    )


def model_based_choose_synergy(
    state: ExactState,
    *,
    comfort_radius: float = 0.35,
    horizon: int = 12,
    beam: int = 24,
    arena_radius: float = 2.0,
    drive_decay: float = 0.02,
) -> int:
    """
    Exact-dynamics beam planner. Capacity bound for the four-synergy body interface.
    Not the handcrafted heuristic.
    """
    if state.dist < comfort_radius:
        return 3  # wait

    # Beam entries: (score, state, first_action)
    beam_states: list[tuple[float, ExactState, int]] = []
    for a0 in range(N_SYNERGIES):
        s1 = exact_synergy_transition(
            state, a0, arena_radius=arena_radius, drive_decay=drive_decay
        )
        score = -s1.dist
        # Prefer orientations that enable |y| reduction when |y| dominates.
        if abs(s1.y) > abs(s1.x) and abs(s1.y) > comfort_radius:
            target_sin = -1.0 if s1.y > 0 else 1.0
            score += 0.05 * (1.0 - abs(math.sin(s1.orientation) - target_sin))
        beam_states.append((score, s1, a0))
    beam_states.sort(key=lambda t: t[0], reverse=True)
    beam_states = beam_states[:beam]

    for _depth in range(1, horizon):
        nxt: list[tuple[float, ExactState, int]] = []
        for _sc, st, first in beam_states:
            if st.dist < comfort_radius:
                nxt.append((10.0 - st.dist, st, first))
                continue
            for a in range(N_SYNERGIES):
                s2 = exact_synergy_transition(
                    st, a, arena_radius=arena_radius, drive_decay=drive_decay
                )
                score = -s2.dist
                if abs(s2.y) > abs(s2.x) and abs(s2.y) > comfort_radius:
                    target_sin = -1.0 if s2.y > 0 else 1.0
                    score += 0.05 * (1.0 - abs(math.sin(s2.orientation) - target_sin))
                nxt.append((score, s2, first))
        nxt.sort(key=lambda t: t[0], reverse=True)
        beam_states = nxt[:beam]

    return int(beam_states[0][2])


def observation_key(
    sensory: torch.Tensor,
    intero: torch.Tensor,
    *,
    quant: float = 1e-3,
) -> str:
    """Quantized organism-facing observation identity (exo+proprio packed + intero)."""
    s = sensory.detach().float().cpu().view(-1)
    i = intero.detach().float().cpu().view(-1)
    # Drop last-action efference channels in exo (indices 6,7) and proprio synergy
    # echoes (indices 4..7) so aliases reflect navigational sufficiency, not motor copy.
    s = s.clone()
    if s.numel() >= 8:
        s[6] = 0.0
        s[7] = 0.0
    # proprio occupies the last proprioceptive_dim entries of sensory
    # BodyConfig default proprioceptive_dim=8 at end of sensory_dim=48
    if s.numel() >= 8:
        s[-4:] = 0.0
    sq = torch.round(s / quant) * quant
    iq = torch.round(i / quant) * quant
    payload = sq.numpy().tobytes() + iq.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def full_state_vector(state: ExactState) -> torch.Tensor:
    return torch.tensor(
        [
            state.x,
            state.y,
            state.dist,
            math.cos(state.orientation),
            math.sin(state.orientation),
            state.energy,
        ],
        dtype=torch.float32,
    )


@dataclass
class RolloutMetrics:
    comfort_rate: float
    mean_start_distance: float
    mean_end_distance: float
    distance_reduction: float
    mean_comfort: float
    n_ticks: int
    synergy_histogram: list[int]


def _empty_hist() -> list[int]:
    return [0 for _ in range(N_SYNERGIES)]


def rollout_policy(
    world_seed: str,
    choose: Callable[[ClosedLoopGroundingWorld, Any], int],
    *,
    n_episodes: int,
    episode_ticks: int,
    device: torch.device,
    generative: GenerativeGenome | None = None,
) -> RolloutMetrics:
    g = generative or GenerativeGenome.small()
    world = ClosedLoopGroundingWorld(
        g, world_seed=world_seed, device=device, episode_ticks=episode_ticks
    )
    correct = 0
    total = 0
    comfort_sum = 0.0
    start_d = 0.0
    end_d = 0.0
    hist = _empty_hist()
    for ep in range(n_episodes):
        step = world.reset_episode(ep)
        start_d += float(step.body_state.position.norm().item())
        last_d = float(step.body_state.position.norm().item())
        ctx: dict[str, Any] = {"ep": ep}
        for _t in range(episode_ticks):
            syn = int(choose(world, step, ctx))
            hist[syn] += 1
            motor = expand_synergy_index_to_motor(syn, device=device, encoding="uniform_block")
            step = world.apply_action(motor)
            correct += int(step.behavioral_correct)
            comfort_sum += float(step.behavioral_score)
            total += 1
            last_d = float(step.body_state.position.norm().item())
        end_d += last_d
    n_ep = max(1, n_episodes)
    n_tot = max(1, total)
    return RolloutMetrics(
        comfort_rate=correct / n_tot,
        mean_start_distance=start_d / n_ep,
        mean_end_distance=end_d / n_ep,
        distance_reduction=start_d / n_ep - end_d / n_ep,
        mean_comfort=comfort_sum / n_tot,
        n_ticks=n_tot,
        synergy_histogram=hist,
    )


def model_based_controller_metrics(
    world_seed: str,
    *,
    n_episodes: int = 32,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    horizon: int = 12,
    beam: int = 24,
) -> dict[str, Any]:
    dev = device or dev1_device()
    comfort_r = BodyConfig().comfort_target_radius

    def choose(world: ClosedLoopGroundingWorld, step, ctx) -> int:
        st = exact_state_from_body(world.body)
        return model_based_choose_synergy(
            st,
            comfort_radius=comfort_r,
            horizon=horizon,
            beam=beam,
            arena_radius=world.body.config.arena_radius,
            drive_decay=world.body.config.drive_decay,
        )

    m = rollout_policy(
        world_seed,
        choose,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
    )
    return {
        "id": "exact_model_based_controller",
        "kind": "exact_dynamics_beam_planner",
        "not_handcrafted_heuristic": True,
        "comfort_rate": m.comfort_rate,
        "distance_reduction": m.distance_reduction,
        "mean_start_distance": m.mean_start_distance,
        "mean_end_distance": m.mean_end_distance,
        "mean_comfort": m.mean_comfort,
        "synergy_histogram": m.synergy_histogram,
        "planner": {"horizon": horizon, "beam": beam},
        "evaluation_world_seed": world_seed,
    }


def random_policy_metrics(
    world_seed: str,
    *,
    n_episodes: int = 32,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    dev = device or dev1_device()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def choose(world, step, ctx) -> int:
        return int(torch.randint(0, N_SYNERGIES, (1,), generator=gen).item())

    m = rollout_policy(
        world_seed,
        choose,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
    )
    return {
        "id": "random_policy",
        "kind": "uniform_synergy_baseline",
        "comfort_rate": m.comfort_rate,
        "distance_reduction": m.distance_reduction,
        "mean_start_distance": m.mean_start_distance,
        "mean_end_distance": m.mean_end_distance,
        "synergy_histogram": m.synergy_histogram,
        "evaluation_world_seed": world_seed,
    }


def observation_collision_analysis(
    world_seed: str,
    *,
    n_episodes: int = 32,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    horizon: int = 12,
    beam: int = 24,
) -> dict[str, Any]:
    """
    Collect organism-facing observations paired with model-based optimal synergies.
    Aliases: same observation key → conflicting optimal actions.
    """
    dev = device or dev1_device()
    g = GenerativeGenome.small()
    world = ClosedLoopGroundingWorld(
        g, world_seed=world_seed + ":alias", device=dev, episode_ticks=episode_ticks
    )
    comfort_r = world.body.config.comfort_target_radius
    alias_map: dict[str, set[int]] = defaultdict(set)
    n_pairs = 0
    for ep in range(n_episodes):
        step = world.reset_episode(ep)
        for _t in range(episode_ticks):
            st = exact_state_from_body(world.body)
            opt = model_based_choose_synergy(
                st,
                comfort_radius=comfort_r,
                horizon=horizon,
                beam=beam,
                arena_radius=world.body.config.arena_radius,
                drive_decay=world.body.config.drive_decay,
            )
            key = observation_key(step.sensory_vector, step.interoceptive_state)
            alias_map[key].add(opt)
            n_pairs += 1
            motor = expand_synergy_index_to_motor(opt, device=dev, encoding="uniform_block")
            step = world.apply_action(motor)
    conflicts = {k: sorted(v) for k, v in alias_map.items() if len(v) > 1}
    return {
        "id": "observation_collision_analysis",
        "n_observation_action_pairs": n_pairs,
        "n_unique_observation_keys": len(alias_map),
        "n_alias_conflict_keys": len(conflicts),
        "alias_conflict_rate": len(conflicts) / max(1, len(alias_map)),
        "example_conflicts": dict(list(conflicts.items())[:8]),
        "has_conflicting_aliases": len(conflicts) > 0,
    }


class SupervisedSynergyNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, N_SYNERGIES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _collect_supervised_dataset(
    world_seed: str,
    *,
    n_episodes: int,
    episode_ticks: int,
    device: torch.device,
    horizon: int,
    beam: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (full_state_x, sensory_x, y_synergy)."""
    g = GenerativeGenome.small()
    world = ClosedLoopGroundingWorld(
        g, world_seed=world_seed + ":supdata", device=device, episode_ticks=episode_ticks
    )
    comfort_r = world.body.config.comfort_target_radius
    xs_full: list[torch.Tensor] = []
    xs_obs: list[torch.Tensor] = []
    ys: list[int] = []
    for ep in range(n_episodes):
        step = world.reset_episode(ep)
        for _t in range(episode_ticks):
            st = exact_state_from_body(world.body)
            opt = model_based_choose_synergy(
                st,
                comfort_radius=comfort_r,
                horizon=horizon,
                beam=beam,
                arena_radius=world.body.config.arena_radius,
                drive_decay=world.body.config.drive_decay,
            )
            xs_full.append(full_state_vector(st))
            xs_obs.append(step.sensory_vector.detach().float().cpu().view(-1).clone())
            ys.append(opt)
            motor = expand_synergy_index_to_motor(opt, device=device, encoding="uniform_block")
            step = world.apply_action(motor)
    return (
        torch.stack(xs_full),
        torch.stack(xs_obs),
        torch.tensor(ys, dtype=torch.long),
    )


def _train_supervised(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    epochs: int = 80,
) -> SupervisedSynergyNet:
    model = SupervisedSynergyNet(int(x.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    xb = x.to(device)
    yb = y.to(device)
    for _ in range(epochs):
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def supervised_controller_metrics(
    world_seed: str,
    *,
    mode: str,
    n_episodes: int = 32,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    train_episodes: int = 48,
    horizon: int = 12,
    beam: int = 24,
) -> dict[str, Any]:
    """mode: full_state | same_observation"""
    dev = device or dev1_device()
    x_full, x_obs, y = _collect_supervised_dataset(
        world_seed,
        n_episodes=train_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        horizon=horizon,
        beam=beam,
    )
    x_train = x_full if mode == "full_state" else x_obs
    model = _train_supervised(x_train, y, device=dev)

    def choose(world: ClosedLoopGroundingWorld, step, ctx) -> int:
        if mode == "full_state":
            st = exact_state_from_body(world.body)
            inp = full_state_vector(st).to(dev)
        else:
            inp = step.sensory_vector.detach().float().to(dev).view(-1)
        with torch.no_grad():
            return int(model(inp).argmax().item())

    m = rollout_policy(
        world_seed,
        choose,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
    )
    with torch.no_grad():
        pred = model(x_train.to(dev)).argmax(dim=-1).cpu()
        train_acc = float((pred == y).float().mean().item())
    return {
        "id": (
            "full_state_supervised_controller"
            if mode == "full_state"
            else "same_observation_supervised_controller"
        ),
        "mode": mode,
        "train_imitation_accuracy": train_acc,
        "comfort_rate": m.comfort_rate,
        "distance_reduction": m.distance_reduction,
        "mean_start_distance": m.mean_start_distance,
        "mean_end_distance": m.mean_end_distance,
        "synergy_histogram": m.synergy_histogram,
        "evaluation_world_seed": world_seed,
        "train_data_world_seed": world_seed + ":supdata",
    }


class FeedforwardAC(nn.Module):
    def __init__(self, sensory_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(sensory_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Linear(64, N_SYNERGIES)
        self.critic = nn.Linear(64, 1)

    def forward(self, sensory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(sensory)
        return self.actor(h), self.critic(h).squeeze(-1)


class RecurrentAC(nn.Module):
    def __init__(self, sensory_dim: int, hidden: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(sensory_dim, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)
        self.actor = nn.Linear(hidden, N_SYNERGIES)
        self.critic = nn.Linear(hidden, 1)
        self.hidden_dim = hidden

    def forward(
        self, sensory: torch.Tensor, h: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.in_proj(sensory))
        if h is None:
            h = torch.zeros(self.hidden_dim, device=sensory.device)
        h2 = self.rnn(x, h)
        return self.actor(h2), self.critic(h2).squeeze(-1), h2


def actor_critic_metrics(
    world_seed: str,
    *,
    recurrent: bool,
    n_episodes: int = 32,
    episode_ticks: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train AC online with organism-owned valence; evaluate late-window comfort."""
    dev = device or dev1_device()
    g = GenerativeGenome.small()
    # Train and evaluate on the shared evaluation world seed so margins are matched.
    world = ClosedLoopGroundingWorld(
        g,
        world_seed=world_seed,
        device=dev,
        episode_ticks=episode_ticks,
    )
    if recurrent:
        net: nn.Module = RecurrentAC(g.sensory_dim).to(dev)
    else:
        net = FeedforwardAC(g.sensory_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    valence = OrganismValenceCircuit(g.interoceptive_dim, device=dev)
    hist = _empty_hist()
    correct = 0
    total = 0
    start_d = 0.0
    end_d = 0.0
    mid = max(1, (n_episodes * episode_ticks) // 2)
    second_half_correct = 0
    comfort_sum = 0.0

    for ep in range(n_episodes):
        valence.reset()
        step = world.reset_episode(ep)
        start_d += float(step.body_state.position.norm().item())
        last_d = float(step.body_state.position.norm().item())
        h = None
        for _t in range(episode_ticks):
            sensory = step.sensory_vector.to(dev)
            if recurrent:
                logits, value, h = net(sensory, h)  # type: ignore[misc]
            else:
                logits, value = net(sensory)  # type: ignore[misc]
            dist = torch.distributions.Categorical(logits=logits)
            syn = int(dist.sample().item())
            hist[syn] += 1
            motor = expand_synergy_index_to_motor(syn, device=dev, encoding="uniform_block")
            step = world.apply_action(motor)
            v = valence.update(step.interoceptive_state)
            logp = dist.log_prob(torch.tensor(syn, device=dev))
            loss = -(logp * (float(v) - float(value.detach()))) + 0.5 * (
                value - float(v)
            ) ** 2 - 0.05 * dist.entropy()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if recurrent and h is not None:
                h = h.detach()
            ok = int(step.behavioral_correct)
            correct += ok
            total += 1
            if total > mid:
                second_half_correct += ok
            comfort_sum += float(step.behavioral_score)
            last_d = float(step.body_state.position.norm().item())
        end_d += last_d

    n_ep = max(1, n_episodes)
    n_tot = max(1, total)
    late = second_half_correct / max(1, n_tot - mid)
    return {
        "id": "recurrent_actor_critic" if recurrent else "feedforward_actor_critic",
        "recurrent": recurrent,
        "comfort_rate": late,
        "full_trajectory_comfort_rate": correct / n_tot,
        "distance_reduction": start_d / n_ep - end_d / n_ep,
        "mean_start_distance": start_d / n_ep,
        "mean_end_distance": end_d / n_ep,
        "mean_comfort": comfort_sum / n_tot,
        "synergy_histogram": hist,
    }


def body_behavior_passes(
    metrics: dict[str, Any],
    random_metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> bool:
    margin = float(metrics["comfort_rate"]) - float(random_metrics["comfort_rate"])
    return (
        float(metrics["comfort_rate"]) >= float(thresholds["min_final_comfort_rate"])
        and float(metrics["distance_reduction"]) >= float(thresholds["min_distance_reduction"])
        and margin >= float(thresholds["min_margin_over_random"])
    )


def apply_wall_routing(
    results: dict[str, Any],
    thresholds: dict[str, float],
) -> str:
    rnd = results["random_policy"]
    mb = results["exact_model_based_controller"]
    full = results["full_state_supervised_controller"]
    same = results["same_observation_supervised_controller"]
    ff = results["feedforward_actor_critic"]
    rc = results["recurrent_actor_critic"]

    mb_ok = body_behavior_passes(mb, rnd, thresholds)
    full_ok = body_behavior_passes(full, rnd, thresholds)
    same_ok = body_behavior_passes(same, rnd, thresholds)
    ff_ok = body_behavior_passes(ff, rnd, thresholds)
    rc_ok = body_behavior_passes(rc, rnd, thresholds)

    if not mb_ok:
        return "body_physics_or_control_bug"
    if mb_ok and not full_ok:
        return "scoring_or_implementation_defect"
    if full_ok and not same_ok:
        return "organism_sensors_omit_necessary_state"
    if same_ok and (not ff_ok) and (not rc_ok):
        return "reward_optimizer_or_horizon_problem"
    if same_ok and (not ff_ok) and rc_ok:
        return "task_requires_transient_memory_canonical_ceiling_must_include_it"
    if same_ok and (ff_ok or rc_ok):
        return "freeze_that_ceiling_and_return_to_complete_r4_factorial_on_fresh_worlds"
    # Edge: same fails but AC somehow passes — still sensor omission dominates.
    if full_ok and not same_ok:
        return "organism_sensors_omit_necessary_state"
    return "unresolved_wall_outcome"


def run_body_reference_wall(
    world_seed: str,
    *,
    n_episodes: int = 32,
    episode_ticks: int = 16,
    device: torch.device | None = None,
    thresholds: dict[str, float] | None = None,
    horizon: int = 16,
    beam: int = 32,
) -> dict[str, Any]:
    """Execute the full reference order and apply routing."""
    dev = device or dev1_device()
    thr = thresholds or {
        "min_final_comfort_rate": 0.05,
        "min_distance_reduction": 0.05,
        "min_margin_over_random": 0.02,
    }
    results: dict[str, Any] = {}
    results["exact_model_based_controller"] = model_based_controller_metrics(
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        horizon=horizon,
        beam=beam,
    )
    results["observation_collision_analysis"] = observation_collision_analysis(
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        horizon=horizon,
        beam=beam,
    )
    results["full_state_supervised_controller"] = supervised_controller_metrics(
        world_seed,
        mode="full_state",
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        horizon=horizon,
        beam=beam,
    )
    results["same_observation_supervised_controller"] = supervised_controller_metrics(
        world_seed,
        mode="same_observation",
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
        horizon=horizon,
        beam=beam,
    )
    results["feedforward_actor_critic"] = actor_critic_metrics(
        world_seed,
        recurrent=False,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
    )
    results["recurrent_actor_critic"] = actor_critic_metrics(
        world_seed,
        recurrent=True,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
    )
    results["random_policy"] = random_policy_metrics(
        world_seed,
        n_episodes=n_episodes,
        episode_ticks=episode_ticks,
        device=dev,
    )
    decision = apply_wall_routing(results, thr)
    return {
        "world_seed": world_seed,
        "thresholds": thr,
        "references": results,
        "decision_code": decision,
        "handcrafted_heuristic_not_used_as_capacity_bound": True,
        "architecture_frozen_sha": "ba97883b6864a0345901b654396a417db3163a03",
        "r4_r1_decision_unchanged": "docs/exos_dev1.stage_a_developmental_birth_r4_r1.dev_decision.lock",
    }
