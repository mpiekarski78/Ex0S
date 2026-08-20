"""
EX0S-DEV1 typed contracts.

Frozen before scored search begins. Do not add fields without a
corresponding architecture lock update.

Boundary law summary
────────────────────
- ActionResult exposes only organism-owned motor outputs; no eligibility,
  no op/operand, no structured task metadata.
- OrganismTelemetry is read-only diagnostics; it must never return through
  observe(), reward, retrieval, or any teaching path.
- FullCheckpoint, EpisodeReset, and HippocampalGraft are THREE SEPARATE
  operations; no merged bundle is permitted.
- IdentitySchemaV1 is the immutable run identity key; terminal_status is
  a separate reuse predicate and not part of identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Checkpoint operations (three separate, not one bundle) ──────────────────

@dataclass
class FullCheckpoint:
    """
    Complete organism state for restoration only.

    Contains: G + W + H + rho + eligibility/plasticity state
              + counters + RNG state + S_log snapshot.

    Used for: exact restoration and matched-donor-twin construction.
    Must NOT be used for hippocampal graft; use HippocampalGraft for that.
    """
    genome_state: dict
    cortex_state: dict
    hippocampus_state: dict
    working_state: dict
    eligibility_state: dict
    plasticity_state: dict
    counters: dict
    rng_state: dict
    slog_snapshot: list


@dataclass
class EpisodeReset:
    """
    Clears rho and transient eligibility only.

    W and H persist unchanged. S_log is not cleared.
    This is the episodic boundary within a single life.
    """
    cleared_working_state: dict
    cleared_transient_eligibility: dict
    step_at_reset: int


@dataclass
class HippocampalGraft:
    """
    Transfers only H and H-local plasticity state between matched
    developmental twins.

    Must NOT transfer: cortex W, genome G, S_log, rho,
    runner metadata, or any other state.

    Matched donor twins are clones of the same pre-teaching FullCheckpoint,
    then diverged by different fact experiences before the graft.
    An unrelated newborn is not expected to decode another organism's
    private neural coordinates.
    """
    donor_hippocampus_state: dict
    donor_hippocampus_plasticity_state: dict
    donor_checkpoint_hash: str   # must match the pre-teaching twin's hash


# ─── Behavioral interface ─────────────────────────────────────────────────────

@dataclass
class OrganismObservation:
    """
    Sensory event delivered to the organism via observe().

    Must not contain: cue IDs, logical slots, fixture metadata,
    runner-generated keys, expected answers, or structured operands.
    """
    sensory_vector: Any           # raw sensory encoding; no task structure
    temporal_context: float = 0.0 # elapsed time / tick index (generic signal)
    reward: float = 0.0           # scalar gate; never an answer identifier
    is_terminal: bool = False
    observed_motor_event: int | None = None  # efference-only demonstrated channel (Reference Birth)
    # Teacher-demonstration consequence (R2): separate from self-action reward.
    # Runner delivers scalar outcome/teaching signal only — never a neural target.
    teaching_signal: float | None = None
    # Developmental Birth R4: body-exposed interoception / proprioception.
    # Body never exposes an expected action. Organism valence owns reinforcement.
    interoceptive_state: Any = None
    proprioceptive_vector: Any = None


@dataclass
class ActionResult:
    """
    Organism behavioral output.

    motor_channel: opaque organism-owned motor index.
    motor_scores:  competition scores over all channels.
    confidence:    learned familiarity / competition margin.

    No op field. No operand field. No eligibility field.
    ASK, HOLD, and answer actions are motor channels interpreted
    by the environment — not internal linguistic ops.
    """
    motor_channel: int
    motor_scores: Any             # np.ndarray shape (n_channels,)
    confidence: float


# ─── Diagnostics (read-only; never feeds back into organism) ─────────────────

@dataclass
class OrganismTelemetry:
    """
    Read-only diagnostics stream for audits and telemetry only.

    This object must never appear as:
    - a teaching signal
    - a retrieval address
    - a corrective action
    - input to observe()
    - a reward channel
    - a retrieval target

    test_boundaries.py enforces this dynamically.
    """
    activations: dict = field(default_factory=dict)
    eligibility_state: dict = field(default_factory=dict)
    replay_events: list = field(default_factory=list)
    plasticity_events: list = field(default_factory=list)
    hippocampus_capacity_used: int = 0
    hippocampus_capacity_max: int = 0
    evictions_this_episode: int = 0


# ─── Telemetry identity schema ────────────────────────────────────────────────

@dataclass
class IdentitySchemaV1:
    """
    Immutable run identity key. All fields are required.

    The reuse predicate is SEPARATE: a record is reusable only when
    terminal_status == "complete". Incomplete or interrupted records
    are never reusable regardless of identity matches.

    Do not add terminal_status here; it is known only after the run.
    """
    implementation_sha: str          # SHA-256 of organism source files
    runner_schema_sha: str           # SHA-256 of runner + probe + world generator
    genome_hash: str                 # SHA-256 of serialized DevGenome config
    world_seed: str                  # world generator seed string
    curriculum_version: str          # SHA-256 of curriculum.py
    backend: str                     # "cpu" | "cuda"
    numeric_mode: str                # dtype + precision spec, e.g. "float32"
    intervention_config: dict        # exact ablation state for every causal gate
    checkpoint_hash: str             # birth checkpoint + any grafts applied
    run_config_hash: str             # SHA-256 of complete frozen run config
    dependency_environment_lock_hash: str  # SHA-256 of deps/toolchain/environment lock


@dataclass
class ReusePredicate:
    """
    Checked separately from identity after a record is located.

    A record is reusable if and only if:
    - identity fields match IdentitySchemaV1 exactly, AND
    - terminal_status == "complete".

    Interrupted, crashed, or partial records are never reused.
    """
    terminal_status: str             # "complete" | "interrupted" | "error" | "partial"

    @property
    def is_reusable(self) -> bool:
        return self.terminal_status == "complete"
