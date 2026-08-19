"""
EX0S-DEV1 ModularOrganism public API.

The organism is the unit of behavioral evaluation. It is born, lives,
and dies in a single continuous life. The runner calls only the public
API defined here.

Public API
──────────
    birth(genome) → ModularOrganism
    observe(event) → OrganismObservation
    act() → ActionResult
    rest(n_ticks) → dict   (rest telemetry)
    episode_reset() → EpisodeReset
    full_checkpoint() → FullCheckpoint
    hippocampal_graft(graft: HippocampalGraft) → None

Within-life learning law (enforced here; audited in test_boundaries.py)
──────────────────────────────────────────────────────────────────────
Allowed:
- Sensory events and temporal order
- Organism's chosen motor action and efference copy
- Teacher-demonstrated action through the same motor-observation channel
- Scalar reward/advantage as a gate (not an answer identifier)
- Internally generated prediction error, novelty, conflict, eligibility traces
- Replay sampled by organism from H

Forbidden:
- Expected action handles in weight updates
- Cue IDs / logical slots / fixture metadata
- Runner-generated keys, queries, stored values, or retrieval addresses
- Future probes or validation worlds
- Oracle W*; LLM; external dictionary; planted semantic vector

NO GRADIENT-BASED UPDATE is applied to W, H, or ρ during an organism's
evaluated lifetime. Within-life learning uses ONLY the frozen local
plasticity law. The research optimizer may differentiate through completed
training-life trajectories, but it may update ONLY inherited G between
lives. Validation and confirmation lives never contribute gradients.
"""

from __future__ import annotations

import random
import time
import copy

import torch
import torch.nn.functional as F
import numpy as np

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import (
    ActionResult,
    EpisodeReset,
    FullCheckpoint,
    HippocampalGraft,
    OrganismObservation,
    OrganismTelemetry,
)
from three_memory.dev1.cortex import SensoryCortex, RelationalCortex, ActionCortex
from three_memory.dev1.hippocampus import FastHippocampus
from three_memory.dev1.working_state import WorkingState
from three_memory.dev1.neuromod import NeuromodController, EligibilityTrace
from three_memory.dev1.provenance import ProvenanceLog, EventKind

from three_memory.dev1.plasticity.cortical_plasticity.three_factor import ThreeFactorPlasticity
from three_memory.dev1.plasticity.cortical_plasticity.meta_learned import MetaLearnedPlasticity
from three_memory.dev1.plasticity.cortical_plasticity.evolved import EvolvedLocalRule
from three_memory.dev1.plasticity.consolidation.replay import ReplayConsolidation
from three_memory.dev1.plasticity.consolidation.online_slow import OnlineSlowConsolidation
from three_memory.dev1.plasticity.consolidation.hybrid import HybridConsolidation


def _build_cortical_plasticity(family: str, genome: DevGenome, n_pre: int, n_post: int):
    if family == "three_factor":
        return ThreeFactorPlasticity(genome.plasticity)
    elif family == "meta_learned":
        return MetaLearnedPlasticity(n_pre, n_post, genome.plasticity)
    elif family == "evolved":
        return EvolvedLocalRule(n_pre, n_post, genome.plasticity)
    else:
        raise ValueError(f"Unknown plasticity_family: {family!r}")


def _build_consolidation(family: str, genome: DevGenome):
    if family == "replay":
        return ReplayConsolidation(genome)
    elif family == "online_slow":
        return OnlineSlowConsolidation(genome)
    elif family == "hybrid":
        return HybridConsolidation(genome)
    else:
        raise ValueError(f"Unknown consolidation_family: {family!r}")


class ModularOrganism:
    """
    One newborn organism.

    Do not instantiate directly — use ModularOrganism.birth(genome).
    """

    def __init__(
        self,
        genome: DevGenome,
        device: torch.device | None = None,
        consolidation_disabled: bool = False,
        h_disabled: bool = False,
    ):
        self.genome = genome
        self.device = device or torch.device("cpu")
        self.consolidation_disabled = consolidation_disabled
        self.step: int = 0
        self.age_frac: float = 0.0
        self._max_steps_hint: int = 10_000   # updated by runner if known

        # Cortical populations
        self.sensory_ctx = SensoryCortex(genome, self.device)
        self.relational_ctx = RelationalCortex(genome, self.device)
        self.action_ctx = ActionCortex(genome, self.device)

        # Fast memory
        self.hippocampus = FastHippocampus(genome, self.device)
        self.hippocampus.h_disabled = h_disabled

        # Working state (transient)
        self.rho = WorkingState(genome, self.device)

        # Neuromodulation
        self.neuromod = NeuromodController(genome, self.device)
        self.eligibility = EligibilityTrace(
            genome.relational_ctx.n_units,
            genome.action_ctx.n_units,
            genome.plasticity.eligibility_decay,
            self.device,
        )

        # Cortical plasticity rule
        self.plasticity_rule = _build_cortical_plasticity(
            genome.plasticity_family,
            genome,
            genome.relational_ctx.n_units,
            genome.action_ctx.n_units,
        )

        # Consolidation
        self.consolidation = _build_consolidation(genome.consolidation_family, genome)

        # Provenance log
        self.slog = ProvenanceLog()
        self.slog.append(EventKind.BIRTH, step=0, payload={"genome_hash": genome.genome_hash()})

        # Internal surprises (for prioritized replay)
        self._surprises: list[float] = []

    @classmethod
    def birth(
        cls,
        genome: DevGenome,
        device: torch.device | None = None,
        consolidation_disabled: bool = False,
        h_disabled: bool = False,
    ) -> "ModularOrganism":
        """
        Produce a newborn organism from genome G.
        W, H, ρ, S_log all begin empty/newborn.
        """
        torch.manual_seed(genome.seed)
        random.seed(genome.seed)
        return cls(genome, device=device, consolidation_disabled=consolidation_disabled, h_disabled=h_disabled)

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(self, event: OrganismObservation) -> OrganismObservation:
        """
        Process one sensory event. Updates working state via cortical forward pass.
        No gradient update to W, H, or ρ.
        """
        sensory_v = _to_tensor(event.sensory_vector, self.device)
        sensory_v = _pad_or_trim(sensory_v, self.genome.sensory_dim)

        # Forward pass
        self.rho.sensory_repr = self.sensory_ctx(sensory_v, self.rho.sensory_repr)

        retrieved = self.hippocampus.read(self.rho.relational_repr)
        self.rho.relational_repr = self.relational_ctx(
            self.rho.sensory_repr, retrieved, self.rho.relational_repr
        )

        self.rho.step = self.step
        self.step += 1
        self.age_frac = min(1.0, self.step / max(1, self._max_steps_hint))

        # Neuromodulatory signals (no W update here)
        mod = self.neuromod(
            self.rho.sensory_repr,
            self.rho.relational_repr,
            self.rho.action_repr,
            event.reward,
        )
        self._last_mod = mod

        # Eligibility update deferred to act() so action_repr is populated

        # Write to H if past onset age and H not disabled
        onset = self.genome.schedule.replay_onset_age_frac
        novelty_val = float(mod["novelty"].item())
        if (self.age_frac >= onset and novelty_val > self.genome.schedule.memory_write_threshold):
            self.hippocampus.write(self.rho.relational_repr, self.rho.sensory_repr)
            self._surprises.append(float(mod["prediction_error"].abs().item()))
            self.slog.append(EventKind.WRITE_H, step=self.step, payload={"novelty": novelty_val})

        self.slog.append(EventKind.OBSERVE, step=self.step, payload={
            "reward": event.reward,
            "terminal": event.is_terminal,
        })

        return event

    def act(self) -> ActionResult:
        """
        Produce motor output via action cortex competition.
        No gradient update to W, H, or ρ.
        """
        action_state, motor_logits = self.action_ctx(self.rho.relational_repr, self.rho.action_repr)
        self.rho.action_repr = action_state

        # Update eligibility trace here (after action state is populated)
        self.eligibility.update(self.rho.relational_repr, self.rho.action_repr)

        channel, scores, confidence = self.action_ctx.competition(motor_logits)

        self.slog.append(EventKind.ACT, step=self.step, payload={
            "motor_channel": channel,
            "confidence": confidence,
        })
        return ActionResult(
            motor_channel=channel,
            motor_scores=scores.detach().cpu().numpy(),
            confidence=confidence,
        )

    def _apply_local_plasticity(self) -> None:
        """
        Apply the within-life local plasticity rule to the action cortex W_motor.
        This is the ONLY weight update during an evaluated life.
        Uses no_grad() — this is not backpropagation.

        Allowed signals: eligibility trace, reward gate.
        The reward gate is the organism's scalar neuromodulatory estimate
        of how much to reinforce recent coactivation.
        Positive reward_gate (>0.5) potentiates; negative (<0.5) depresses.
        """
        if not hasattr(self, "_last_mod"):
            return
        mod = self._last_mod
        rule = self.plasticity_rule
        elig = self.eligibility.trace   # shape (n_rel, n_action)

        with torch.no_grad():
            elig_signal = elig.mean(dim=0)   # (n_action,)
            # Signed modulation: reward_gate centered around 0.5
            # Positive reward → gate > 0.5 → potentiation
            # Negative reward / baseline exceeded → gate < 0.5 → depression
            gate = (mod["reward_gate"] - 0.5) * 2.0   # range [-1, 1]
            lr = getattr(rule, "lr", self.genome.plasticity.learning_rate)
            # ΔW_motor: (n_channels, n_action) via broadcast
            dW = lr * gate * elig_signal.unsqueeze(0)
            dW = dW.clamp(-0.1, 0.1)
            self.action_ctx.W_motor.weight.data.add_(dW)

    def rest(self, n_ticks: int = 1) -> dict:
        """
        Internal rest period: organism applies local plasticity and replay.
        No gradient updates. Slow cortex disabled if consolidation_disabled=True.
        Returns rest telemetry.
        """
        # Apply local plasticity rule to action weights (within-life, no gradient)
        self._apply_local_plasticity()

        replayed = 0
        if not self.consolidation_disabled and self.hippocampus._store:
            onset_frac = self.genome.schedule.consolidation_onset_age_frac
            if self.age_frac >= onset_frac:
                consol = self.consolidation
                if hasattr(consol, "replay"):
                    episodes = consol.replay.sample_episodes(
                        self.hippocampus._store, self._surprises
                    )
                elif hasattr(consol, "sample_episodes"):
                    episodes = consol.sample_episodes(
                        self.hippocampus._store, self._surprises
                    )
                else:
                    episodes = []
                replayed = len(episodes)
                self.slog.append(EventKind.REPLAY, step=self.step, payload={"n_replayed": replayed})
        return {"replayed": replayed, "step": self.step}

    def episode_reset(self) -> EpisodeReset:
        """
        Clear ρ and transient eligibility only.
        W and H persist unchanged. S_log is not cleared.
        """
        snap = self.rho.cleared_snapshot()
        elig_snap = {"trace": self.eligibility.trace.cpu().tolist()}
        self.rho.reset()
        self.eligibility.reset_transient()
        self.hippocampus.reset_episode_counter()
        reset = EpisodeReset(
            cleared_working_state=snap["working_state"],
            cleared_transient_eligibility=elig_snap,
            step_at_reset=self.step,
        )
        self.slog.append(EventKind.RESET, step=self.step)
        return reset

    def full_checkpoint(self) -> FullCheckpoint:
        """
        Capture complete organism state.
        Used for exact restoration and matched-donor-twin construction.
        """
        cp = FullCheckpoint(
            genome_state=self.genome.to_dict(),
            cortex_state={
                "sensory": {k: v.cpu() for k, v in self.sensory_ctx.state_dict().items()},
                "relational": {k: v.cpu() for k, v in self.relational_ctx.state_dict().items()},
                "action": {k: v.cpu() for k, v in self.action_ctx.state_dict().items()},
            },
            hippocampus_state=self.hippocampus.hippocampus_state_dict(),
            working_state=self.rho.state_dict(),
            eligibility_state={"trace": self.eligibility.trace.cpu()},
            plasticity_state=self.hippocampus.hippocampus_plasticity_state_dict(),
            counters={"step": self.step},
            rng_state={"torch": torch.get_rng_state().tolist()},
            slog_snapshot=self.slog.snapshot(),
        )
        self.slog.append(EventKind.CHECKPOINT, step=self.step)
        return cp

    def restore_from_checkpoint(self, cp: FullCheckpoint) -> None:
        """Restore organism to a previously saved full checkpoint."""
        self.sensory_ctx.load_state_dict(cp.cortex_state["sensory"])
        self.relational_ctx.load_state_dict(cp.cortex_state["relational"])
        self.action_ctx.load_state_dict(cp.cortex_state["action"])
        self.hippocampus.load_hippocampus_state_dict(cp.hippocampus_state)
        self.hippocampus.load_hippocampus_plasticity_state_dict(cp.plasticity_state)
        self.rho.load_state_dict(cp.working_state)
        self.eligibility.trace = cp.eligibility_state["trace"].to(self.device)
        self.step = cp.counters["step"]
        self.slog.restore_from_snapshot(cp.slog_snapshot)

    def hippocampal_graft(self, graft: HippocampalGraft) -> None:
        """
        Transfer only H and H-local plasticity state from a matched donor twin.
        Does NOT transfer: cortex W, genome G, S_log, rho, runner metadata.
        """
        self.hippocampus.load_hippocampus_state_dict(graft.donor_hippocampus_state)
        self.hippocampus.load_hippocampus_plasticity_state_dict(graft.donor_hippocampus_plasticity_state)
        self.slog.append(EventKind.GRAFT, step=self.step, payload={
            "donor_checkpoint_hash": graft.donor_checkpoint_hash,
        })

    def telemetry(self) -> OrganismTelemetry:
        """
        Read-only diagnostics snapshot.
        This object must never be fed back into observe(), reward, retrieval,
        or any teaching path. Audited dynamically in test_boundaries.py.
        """
        cap = self.hippocampus.capacity_telemetry()
        return OrganismTelemetry(
            activations={
                "sensory": self.rho.sensory_repr.detach().cpu().numpy(),
                "relational": self.rho.relational_repr.detach().cpu().numpy(),
                "action": self.rho.action_repr.detach().cpu().numpy(),
            },
            eligibility_state={"trace_norm": float(self.eligibility.trace.norm().item())},
            replay_events=[],
            plasticity_events=[],
            hippocampus_capacity_used=cap["capacity_used"],
            hippocampus_capacity_max=cap["capacity_max"],
            evictions_this_episode=cap["evictions_total"],
        )

    def count_mutable_scalars(self) -> dict:
        """Compute the six capacity metrics required by scale.py."""
        cortical = sum(
            p.numel() for m in [self.sensory_ctx, self.relational_ctx, self.action_ctx, self.neuromod]
            for p in m.parameters()
        )
        fast_syn = (
            self.hippocampus.spec.dg_n_units ** 2   # W_hebb
            + self.hippocampus.spec.capacity * (
                self.hippocampus.spec.ec_dim + self.genome.relational_ctx.n_units
            )
        )
        return {
            "mutable_cortical_scalars": cortical,
            "fast_synaptic_memory_scalars": fast_syn,
            "total_mutable_neural_scalars": cortical + fast_syn,
            "recurrent_state_dim": self.genome.relational_ctx.n_units,
            "activations_per_tick": (
                self.genome.sensory_ctx.n_units
                + self.genome.relational_ctx.n_units
                + self.genome.action_ctx.n_units
            ),
            "checkpoint_bytes": -1,   # filled by scale.py after serialisation
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_tensor(v, device: torch.device) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.float().to(device)
    return torch.tensor(v, dtype=torch.float32, device=device)


def _pad_or_trim(v: torch.Tensor, target: int) -> torch.Tensor:
    n = v.numel()
    if n == target:
        return v
    if n > target:
        return v[:target]
    return F.pad(v, (0, target - n))
