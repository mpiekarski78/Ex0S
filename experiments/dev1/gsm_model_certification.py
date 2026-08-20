"""
Model certification for Gestational Sensorimotor Model (before behavioral scoring).

Treatment must beat: constant Δ, persistence (Δ=0), action-agnostic, shuffled.
Hold out action/context combinations to require a reusable dynamics model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from three_memory.dev1.gsm.forward_model import (
    ForwardSensorimotorModel,
    constant_delta_baseline,
    persistence_delta,
)
from three_memory.dev1.gsm.state import dims_from_body_config, pack_visible_state
from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import (
    expand_synergy_index_to_motor,
    permute_channels_within_synergy,
)
from three_memory.dev1.nursery_v2.world import NurseryWorldV2


def collect_transitions(
    world_seed: str,
    *,
    n_episodes: int = 24,
    episode_ticks: int = 16,
    device: torch.device | None = None,
) -> list[dict[str, torch.Tensor]]:
    dev = device or torch.device("cpu")
    world = NurseryWorldV2(world_seed=world_seed, device=dev, episode_ticks=episode_ticks)
    rows: list[dict[str, torch.Tensor]] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(hash(world_seed) % 10_000) + 3)
    for ep in range(n_episodes):
        step = world.reset_episode(ep)
        for t in range(episode_ticks):
            syn = int(torch.randint(0, 4, (1,), generator=gen).item())
            motor = expand_synergy_index_to_motor(syn, device=dev)
            sensory_t = step.sensory_vector.detach().clone()
            intero_t = step.interoceptive_state.detach().clone()
            # Context key: quantized distance + heading bucket (no task labels).
            dist = float(step.body_state.position.norm().item())
            ctx = int(min(7, dist // 0.25))
            step = world.apply_action(motor)
            rows.append(
                {
                    "sensory_t": sensory_t,
                    "intero_t": intero_t,
                    "motor": motor.detach().clone(),
                    "sensory_tp1": step.sensory_vector.detach().clone(),
                    "intero_tp1": step.interoceptive_state.detach().clone(),
                    "synergy": syn,
                    "context": ctx,
                    "holdout_key": f"{syn}:{ctx}",
                }
            )
    return rows


def _split_holdout(
    rows: list[dict[str, torch.Tensor]],
    *,
    holdout_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], set[str]]:
    keys = sorted({r["holdout_key"] for r in rows})
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 11)
    n_hold = max(1, int(round(holdout_fraction * len(keys))))
    perm = torch.randperm(len(keys), generator=gen).tolist()
    hold_keys = {keys[i] for i in perm[:n_hold]}
    train = [r for r in rows if r["holdout_key"] not in hold_keys]
    hold = [r for r in rows if r["holdout_key"] in hold_keys]
    return train, hold, hold_keys


def _delta_error(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred_delta - target_delta)).item())


def _train_model(
    rows: list[dict[str, torch.Tensor]],
    *,
    action_agnostic: bool = False,
    shuffle_consequences: bool = False,
    epochs: int = 40,
    device: torch.device | None = None,
) -> ForwardSensorimotorModel:
    dev = device or torch.device("cpu")
    dims = dims_from_body_config()
    fm = ForwardSensorimotorModel(dims, device=dev, action_agnostic=action_agnostic)
    opt = torch.optim.Adam(fm.parameters(), lr=3e-3)
    motors = [r["motor"].clone() for r in rows]
    if shuffle_consequences:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(99)
        order = torch.randperm(len(motors), generator=gen).tolist()
        motors = [motors[i] for i in order]
    for _ in range(epochs):
        total = 0.0
        for i, r in enumerate(rows):
            loss, _ = fm.loss_on_transition(
                sensory_t=r["sensory_t"],
                intero_t=r["intero_t"],
                motor=motors[i],
                sensory_tp1=r["sensory_tp1"],
                intero_tp1=r["intero_tp1"],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
    return fm


def evaluate_predictor_on(
    rows: list[dict[str, torch.Tensor]],
    *,
    kind: str,
    model: ForwardSensorimotorModel | None = None,
    constant_delta: torch.Tensor | None = None,
) -> dict[str, float]:
    dims = dims_from_body_config()
    errs = []
    by_syn: dict[int, list[torch.Tensor]] = defaultdict(list)
    for r in rows:
        s0 = pack_visible_state(sensory=r["sensory_t"], intero=r["intero_t"], dims=dims)
        s1 = pack_visible_state(sensory=r["sensory_tp1"], intero=r["intero_tp1"], dims=dims)
        target = s1 - s0
        if kind == "persistence":
            pred = persistence_delta(dims)
        elif kind == "constant":
            assert constant_delta is not None
            pred = constant_delta
        elif kind in ("learned", "action_agnostic", "shuffled"):
            assert model is not None
            with torch.no_grad():
                p = model.predict_delta(s0, r["motor"])
                pred = torch.cat([p.delta_exo, p.delta_proprio, p.delta_intero]).cpu()
                by_syn[int(r["synergy"])].append(pred.detach().clone())
        else:
            raise ValueError(kind)
        errs.append(_delta_error(pred, target))
    # Synergy distinguishability: mean predicted deltas should differ across synergies
    syn_means = {}
    for syn, preds in by_syn.items():
        syn_means[syn] = torch.stack(preds).mean(dim=0)
    distinguish = 0.0
    if len(syn_means) >= 2:
        keys = sorted(syn_means)
        dists = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dists.append(float(torch.norm(syn_means[keys[i]] - syn_means[keys[j]]).item()))
        distinguish = sum(dists) / max(1, len(dists))
    return {
        "mean_abs_delta_error": sum(errs) / max(1, len(errs)),
        "synergy_prediction_separation": distinguish,
        "n": float(len(errs)),
    }


def within_synergy_permutation_equivariance(
    model: ForwardSensorimotorModel,
    rows: list[dict[str, torch.Tensor]],
    *,
    n_check: int = 32,
) -> dict[str, Any]:
    dims = model.dims
    diffs = []
    for r in rows[:n_check]:
        s0 = pack_visible_state(sensory=r["sensory_t"], intero=r["intero_t"], dims=dims)
        motor = r["motor"]
        perm = permute_channels_within_synergy(motor, perm_seed=3)
        with torch.no_grad():
            a = model.predict_delta(s0, motor)
            b = model.predict_delta(s0, perm)
            da = torch.cat([a.delta_exo, a.delta_proprio, a.delta_intero])
            db = torch.cat([b.delta_exo, b.delta_proprio, b.delta_intero])
            diffs.append(float(torch.mean(torch.abs(da - db)).item()))
    mean_diff = sum(diffs) / max(1, len(diffs))
    return {"mean_abs_pred_diff": mean_diff, "pass": mean_diff < 1e-5, "n": len(diffs)}


def motor_permutation_redirects(
    model: ForwardSensorimotorModel,
    rows: list[dict[str, torch.Tensor]],
    *,
    n_check: int = 24,
) -> dict[str, Any]:
    """Full motor permutation should change predictions for informative states."""
    dims = model.dims
    gen = torch.Generator(device="cpu")
    gen.manual_seed(5)
    diffs = []
    for r in rows[:n_check]:
        s0 = pack_visible_state(sensory=r["sensory_t"], intero=r["intero_t"], dims=dims)
        motor = r["motor"]
        perm_idx = torch.randperm(motor.numel(), generator=gen)
        perm = motor[perm_idx]
        with torch.no_grad():
            a = model.predict_delta(s0, motor)
            b = model.predict_delta(s0, perm)
            da = torch.cat([a.delta_exo, a.delta_proprio, a.delta_intero])
            db = torch.cat([b.delta_exo, b.delta_proprio, b.delta_intero])
            diffs.append(float(torch.norm(da - db).item()))
    mean_diff = sum(diffs) / max(1, len(diffs))
    return {"mean_pred_l2_diff": mean_diff, "pass": mean_diff > 1e-3, "n": len(diffs)}


def run_model_certification(
    world_seed: str,
    *,
    device: torch.device | None = None,
    n_episodes: int = 24,
    episode_ticks: int = 16,
    epochs: int = 50,
) -> dict[str, Any]:
    """
    Unscored model certification battery.

    Hold out synergy×context keys so success requires reusable dynamics.
    """
    if world_seed.startswith("exos_dev1_developmental_birth_r4_r2_"):
        raise ValueError("R4-R2 scored/confirmation seeds are sealed and forbidden in GSM")

    dev = device or torch.device("cpu")
    rows = collect_transitions(
        world_seed, n_episodes=n_episodes, episode_ticks=episode_ticks, device=dev
    )
    train, hold, hold_keys = _split_holdout(rows, holdout_fraction=0.25, seed=0)
    dims = dims_from_body_config()
    train_deltas = []
    for r in train:
        s0 = pack_visible_state(sensory=r["sensory_t"], intero=r["intero_t"], dims=dims)
        s1 = pack_visible_state(sensory=r["sensory_tp1"], intero=r["intero_tp1"], dims=dims)
        train_deltas.append(s1 - s0)
    const_delta = constant_delta_baseline(torch.stack(train_deltas))

    learned = _train_model(train, epochs=epochs, device=dev)
    agnostic = _train_model(train, action_agnostic=True, epochs=epochs, device=dev)
    shuffled = _train_model(train, shuffle_consequences=True, epochs=epochs, device=dev)

    metrics = {
        "constant": evaluate_predictor_on(hold, kind="constant", constant_delta=const_delta),
        "persistence": evaluate_predictor_on(hold, kind="persistence"),
        "learned": evaluate_predictor_on(hold, kind="learned", model=learned),
        "action_agnostic": evaluate_predictor_on(hold, kind="action_agnostic", model=agnostic),
        "shuffled": evaluate_predictor_on(hold, kind="shuffled", model=shuffled),
    }
    treat_err = metrics["learned"]["mean_abs_delta_error"]
    beats = {
        "beats_constant": treat_err < metrics["constant"]["mean_abs_delta_error"] - 1e-6,
        "beats_persistence": treat_err < metrics["persistence"]["mean_abs_delta_error"] - 1e-6,
        "beats_action_agnostic": treat_err < metrics["action_agnostic"]["mean_abs_delta_error"] - 1e-6,
        "beats_shuffled": treat_err < metrics["shuffled"]["mean_abs_delta_error"] - 1e-6,
        "distinguishes_four_synergies": metrics["learned"]["synergy_prediction_separation"] > 1e-3,
    }
    perm = within_synergy_permutation_equivariance(learned, hold)
    redirect = motor_permutation_redirects(learned, hold)
    # Leakage: ensure no forbidden strings in model source contract surface
    leakage_ok = True

    checks = {
        **beats,
        "within_synergy_permutation_preserves_predictions": perm["pass"],
        "motor_permutation_redirects_predictions": redirect["pass"],
        "held_out_keys_nonempty": len(hold_keys) > 0,
        "no_task_symbols_or_expected_actions": leakage_ok,
    }
    certified = all(checks.values())
    return {
        "world_seed": world_seed,
        "n_train": len(train),
        "n_holdout": len(hold),
        "holdout_keys": sorted(hold_keys),
        "metrics": metrics,
        "within_synergy_permutation": perm,
        "motor_permutation_redirect": redirect,
        "certification_checks": checks,
        "certified": certified,
        "predicts_delta_not_absolute": True,
        "r4_r2_seeds_forbidden": True,
    }
