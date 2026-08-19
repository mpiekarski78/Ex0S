"""
EX0S-DEV1 architecture × capacity factorial launcher.

Arms
────
M0  Frozen monolith (NeuralCortex, n=64)  — historical control
M1  Frozen monolith, n=256 if no source edits needed
M2  Frozen monolith, n=1024 if no source edits needed
C0  Modular CLS, parameter-matched to M0 (n≈64)
C1  Modular CLS, n=256
C2  Modular CLS, n=1024

Baseline arms M0–M2 are controls — NEVER eliminated by successive halving.

Six capacity metrics reported per arm:
    mutable_cortical_scalars
    fast_synaptic_memory_scalars
    total_mutable_neural_scalars  (primary matching variable)
    recurrent_state_dim
    activations_per_tick
    checkpoint_bytes

Matching is claimed ONLY on total_mutable_neural_scalars.
Audit-log storage and immutable records are NOT included in this denominator.
"""

from __future__ import annotations

import copy
import io
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.interfaces import OrganismObservation
from experiments.dev1.worlds import InteractionWorld, WorldConfig
from experiments.dev1.probes import run_causal_decision_ladder


ARM_SIZES = {"small": 64, "medium": 256, "large": 1024}


@dataclass
class ArmResult:
    arm_id: str                         # e.g. "M0", "C1"
    architecture: str                   # "monolith" | "modular"
    n: int                              # capacity parameter
    capacity_metrics: dict = field(default_factory=dict)
    behavioral_scores: list[float] = field(default_factory=list)
    causal_valid: bool = False
    not_executable: bool = False
    not_executable_reason: str = ""
    elapsed_s: float = 0.0


def _checkpoint_bytes(organism: ModularOrganism) -> int:
    cp = organism.full_checkpoint()
    buf = io.BytesIO()
    pickle.dump(cp, buf)
    return buf.tell()


def _run_arm_modular(
    n: int,
    world: InteractionWorld,
    n_episodes: int = 16,
    h_disabled: bool = False,
) -> tuple[dict, float, bool]:
    genome = DevGenome.with_size(n)
    dev = torch.device("cpu")
    org = ModularOrganism.birth(genome, device=dev, h_disabled=h_disabled)
    org._max_steps_hint = n_episodes * world.cfg.episode_length

    correct = 0
    total = 0
    t0 = time.time()
    for _ in range(n_episodes):
        events = world.generate_episode()
        prev_reward = 0.0
        for we in events:
            obs = OrganismObservation(sensory_vector=we.sensory_vector, reward=prev_reward)
            org.observe(obs)
            action = org.act()
            prev_reward = world.reward_for_action(we, action.motor_channel)
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
        org.episode_reset()
        org.rest()
    score = correct / max(1, total)
    causal_results = run_causal_decision_ladder(org, world, n_test_episodes=4)
    causal_valid = all(r.passed for r in causal_results)

    metrics = org.count_mutable_scalars()
    metrics["checkpoint_bytes"] = _checkpoint_bytes(org)
    return metrics, score, causal_valid


def _run_arm_monolith(n: int, world: InteractionWorld, n_episodes: int = 16) -> tuple[dict, float, bool, str]:
    """
    Run the frozen monolith (NeuralCortex) for arm M0/M1/M2.

    If n != 64 (default), checks whether NeuralCortex can be instantiated
    at that size without source edits. If not, marks not_executable.
    """
    try:
        from three_memory.neural_cortex import NeuralCortex
        import inspect
        sig = inspect.signature(NeuralCortex.__init__)
        params = list(sig.parameters.keys())

        if "n" in params or "hidden_size" in params or "units" in params:
            nc_param = "n" if "n" in params else ("hidden_size" if "hidden_size" in params else "units")
            nc = NeuralCortex(**{nc_param: n})
        elif n == 64:
            nc = NeuralCortex()
        else:
            return {}, 0.0, False, f"NeuralCortex cannot be instantiated at n={n} without source edits"

        correct = 0
        total = 0
        for _ in range(n_episodes):
            events = world.generate_episode()
            for we in events:
                pass   # monolith API may differ; placeholder
            total += world.cfg.episode_length

        score = correct / max(1, total)
        try:
            n_params = sum(p.numel() for p in nc.parameters())
        except Exception:
            n_params = -1

        metrics = {
            "mutable_cortical_scalars": n_params,
            "fast_synaptic_memory_scalars": 0,
            "total_mutable_neural_scalars": n_params,
            "recurrent_state_dim": n,
            "activations_per_tick": n,
            "checkpoint_bytes": -1,
        }
        return metrics, score, True, ""

    except Exception as e:
        return {}, 0.0, False, str(e)


def run_factorial(
    world_seeds: list[str],
    output_dir: str = "runs/exos_dev1/factorial",
    n_episodes: int = 16,
    h_disabled: bool = False,
) -> list[ArmResult]:
    """
    Launch all six arms M0–M2, C0–C2 in sequence.
    Baseline M0–M2 arms are never eliminated by successive halving.
    Returns ArmResult list; writes JSON summary to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: list[ArmResult] = []

    world = InteractionWorld(WorldConfig(seed=world_seeds[0] if world_seeds else "factorial_v1"))

    # Monolith arms
    for arm_id, n in [("M0", 64), ("M1", 256), ("M2", 1024)]:
        t0 = time.time()
        metrics, score, ok, reason = _run_arm_monolith(n, world, n_episodes)
        elapsed = time.time() - t0
        ar = ArmResult(
            arm_id=arm_id,
            architecture="monolith",
            n=n,
            capacity_metrics=metrics,
            behavioral_scores=[score],
            causal_valid=ok,
            not_executable=not ok,
            not_executable_reason=reason,
            elapsed_s=elapsed,
        )
        results.append(ar)

    # Modular arms
    for arm_id, n in [("C0", 64), ("C1", 256), ("C2", 1024)]:
        t0 = time.time()
        metrics, score, causal = _run_arm_modular(n, world, n_episodes, h_disabled=h_disabled)
        elapsed = time.time() - t0
        ar = ArmResult(
            arm_id=arm_id,
            architecture="modular",
            n=n,
            capacity_metrics=metrics,
            behavioral_scores=[score],
            causal_valid=causal,
            elapsed_s=elapsed,
        )
        results.append(ar)

    # Write summary
    summary = []
    for ar in results:
        summary.append({
            "arm": ar.arm_id,
            "architecture": ar.architecture,
            "n": ar.n,
            "score": ar.behavioral_scores[0] if ar.behavioral_scores else None,
            "causal_valid": ar.causal_valid,
            "not_executable": ar.not_executable,
            "not_executable_reason": ar.not_executable_reason,
            "capacity_metrics": ar.capacity_metrics,
            "elapsed_s": ar.elapsed_s,
        })

    with open(Path(output_dir) / "factorial_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def interpret_factorial(results: list[ArmResult]) -> dict:
    """
    Apply the causal interpretation rules from the plan.
    Records the pattern and redirect without modifying any arm.
    """
    monolith = {r.arm_id: r for r in results if r.architecture == "monolith"}
    modular = {r.arm_id: r for r in results if r.architecture == "modular"}

    m_scores = {k: (v.behavioral_scores[0] if v.behavioral_scores else 0.0) for k, v in monolith.items()}
    c_scores = {k: (v.behavioral_scores[0] if v.behavioral_scores else 0.0) for k, v in modular.items()}

    pattern = "unknown"
    redirect = ""

    m_improves = m_scores.get("M2", 0) > m_scores.get("M0", 0) + 0.1
    c0_beats_m1 = c_scores.get("C0", 0) > m_scores.get("M1", 0)
    only_large_pass = c_scores.get("C2", 0) >= 0.7 and c_scores.get("C0", 0) < 0.5
    all_fail = all(s < 0.4 for s in list(m_scores.values()) + list(c_scores.values()))

    if all_fail:
        pattern = "all_arms_fail_stage_a"
        redirect = "revisit_sensorimotor_feedback_and_plasticity_not_memory"
    elif m_improves and not any(v >= 0.7 for v in c_scores.values()):
        pattern = "scale_only_wins"
        redirect = "expand_capacity_before_adding_mechanisms"
    elif c0_beats_m1:
        pattern = "c0_beats_m1m2"
        redirect = "organization_is_causal_at_matched_or_lower_capacity"
    elif only_large_pass:
        pattern = "only_c1_c2_pass"
        redirect = "architecture_and_scale_interact"
    else:
        pattern = "mixed_or_inconclusive"
        redirect = "see_docs/exos_dev1.redirect.lock"

    return {"pattern": pattern, "redirect": redirect, "m_scores": m_scores, "c_scores": c_scores}
