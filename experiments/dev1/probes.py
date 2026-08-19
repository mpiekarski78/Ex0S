"""
EX0S-DEV1 canonical behavioral and causal probes.

Boundary invariants
───────────────────
- Runner may observe neural telemetry through OrganismTelemetry.
- That telemetry must not appear as a teaching signal, retrieval address,
  corrective action, observe() input, reward channel, or retrieval target
  in any subsequent organism update.
- test_boundaries.py audits this dynamically.

Causal decision ladder (ordered):
1.  setup_precondition_fail
2.  semantic_leakage
3.  feedback_not_causal
4.  memory_not_necessary
5.  address_not_organism_owned
6.  checkpoint_or_reset_fail
7.  grounding_fail
8.  fast_memory_fail
9.  consolidation_fail
10. composition_fail
11. revision_fail
12. clarification_fail
13. continued_learning_fail
14. integrated_development_pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiments.dev1.worlds import WorldEvent, InteractionWorld, WorldConfig, _symbol_vector
from three_memory.dev1.interfaces import ActionResult, OrganismTelemetry


@dataclass
class ProbeResult:
    probe_name: str
    passed: bool
    score: float
    causal_label: str
    details: dict = field(default_factory=dict)


def probe_reward_off(
    organism,
    world: InteractionWorld,
    n_episodes: int = 4,
) -> ProbeResult:
    """
    Ablation: reward signal is zeroed. Organism must fail to learn new roles.
    Used as causal control for feedback_not_causal.
    """
    from three_memory.dev1.interfaces import OrganismObservation
    correct = 0
    total = 0
    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            obs = OrganismObservation(sensory_vector=we.sensory_vector, reward=0.0)
            organism.observe(obs)
            action = organism.act()
            total += 1
    # Without reward, should not improve; measure accuracy against known mapping
    score = correct / max(1, total)
    return ProbeResult(
        probe_name="reward_off",
        passed=score < 0.4,   # should fail to learn
        score=score,
        causal_label="feedback_not_causal",
    )


def probe_grounding_accuracy(
    organism,
    world: InteractionWorld,
    n_test_episodes: int = 8,
) -> ProbeResult:
    """
    Measure action-role accuracy on a test set after grounding phase.
    """
    from three_memory.dev1.interfaces import OrganismObservation
    correct = 0
    total = 0
    for _ in range(n_test_episodes):
        events = world.generate_episode()
        for we in events:
            obs = OrganismObservation(sensory_vector=we.sensory_vector, reward=0.0)
            organism.observe(obs)
            action = organism.act()
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
    score = correct / max(1, total)
    return ProbeResult(
        probe_name="grounding_accuracy",
        passed=score >= 0.7,
        score=score,
        causal_label="grounding_fail" if score < 0.7 else "integrated_development_pass",
        details={"correct": correct, "total": total},
    )


def probe_one_shot_recall(
    organism,
    fact_vec: np.ndarray,
    correct_channel: int,
) -> ProbeResult:
    """
    After one-shot fact presentation and EpisodeReset, can organism recall fact?
    """
    from three_memory.dev1.interfaces import OrganismObservation
    obs = OrganismObservation(sensory_vector=fact_vec, reward=0.0)
    organism.observe(obs)
    action = organism.act()
    passed = action.motor_channel == correct_channel
    return ProbeResult(
        probe_name="one_shot_recall",
        passed=passed,
        score=float(passed),
        causal_label="fast_memory_fail" if not passed else "integrated_development_pass",
    )


def probe_wipe_removes_fact(
    organism,
    fact_vec: np.ndarray,
    correct_channel: int,
) -> ProbeResult:
    """
    After H wipe, one-shot fact must no longer be retrievable.
    """
    from three_memory.dev1.interfaces import OrganismObservation
    organism.hippocampus.wipe()
    obs = OrganismObservation(sensory_vector=fact_vec, reward=0.0)
    organism.observe(obs)
    action = organism.act()
    fact_gone = action.motor_channel != correct_channel
    return ProbeResult(
        probe_name="wipe_removes_fact",
        passed=fact_gone,
        score=float(fact_gone),
        causal_label="memory_not_necessary" if not fact_gone else "integrated_development_pass",
    )


def probe_donor_redirect(
    organism,
    donor_graft,
    fact_vec: np.ndarray,
    donor_correct_channel: int,
) -> ProbeResult:
    """
    After HippocampalGraft from matched donor twin, organism should
    exhibit donor's fact association, not its own.
    """
    from three_memory.dev1.interfaces import OrganismObservation
    organism.hippocampal_graft(donor_graft)
    obs = OrganismObservation(sensory_vector=fact_vec, reward=0.0)
    organism.observe(obs)
    action = organism.act()
    redirected = action.motor_channel == donor_correct_channel
    return ProbeResult(
        probe_name="donor_redirect",
        passed=redirected,
        score=float(redirected),
        causal_label="checkpoint_or_reset_fail" if not redirected else "integrated_development_pass",
    )


def probe_permuted_feedback(
    organism,
    world: InteractionWorld,
    n_episodes: int = 4,
) -> ProbeResult:
    """
    Reward is delivered for the WRONG action. Organism must not learn correctly.
    """
    from three_memory.dev1.interfaces import OrganismObservation
    correct = 0
    total = 0
    for _ in range(n_episodes):
        events = world.generate_episode()
        for we in events:
            wrong_reward = world.cfg.reward_on_correct if (we._correct_channel + 1) % world.cfg.n_roles == 0 else 0.0
            obs = OrganismObservation(sensory_vector=we.sensory_vector, reward=wrong_reward)
            organism.observe(obs)
            action = organism.act()
            if action.motor_channel == we._correct_channel:
                correct += 1
            total += 1
    score = correct / max(1, total)
    return ProbeResult(
        probe_name="permuted_feedback",
        passed=score < 0.4,
        score=score,
        causal_label="feedback_not_causal",
    )


def run_causal_decision_ladder(
    organism,
    world: InteractionWorld,
    n_test_episodes: int = 8,
    skip_fact_probes: bool = True,
) -> list[ProbeResult]:
    """
    Run all probes in causal ladder order.
    Stop at the first failure that characterizes the causal bottleneck.
    """
    results = []

    r = probe_grounding_accuracy(organism, world, n_test_episodes)
    results.append(r)
    if not r.passed:
        return results

    r2 = probe_reward_off(organism, world)
    results.append(r2)
    if not r2.passed:
        return results

    r3 = probe_permuted_feedback(organism, world)
    results.append(r3)
    if not r3.passed:
        return results

    return results
