"""TM.0.23.CORTEX — plastic recurrent artificial cortex (CPU gold + optional GPU).

Does not wrap make_interpret / ThreeMemoryAgent. No capability-specific heads.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from three_memory.cortex_memory import CortexMemory, CortexRecord
from three_memory.opaque_memory import OpaqueMemory

OPS = ("RETRIEVE", "WRITE", "EMIT", "ACT", "STOP", "HOLD")
# v5: no birth-planted motor dictionary; actuators bind via bind_actuators([...])
MOTOR_ACT_TOKENS: tuple[str, ...] = ()
OP_COST = {
    "RETRIEVE": 1.0,
    "WRITE": 1.0,
    "EMIT": 1.0,
    "ACT": 0.05,  # generic low ACT cost (defensible innate tendency)
    "STOP": 0.0,
    "HOLD": 0.0,
}
OBSERVE_KEYS = frozenset(
    {
        "interaction_token",
        "source_token",
        "ordered_symbols",
        "observable_state",
        "body_state",
    }
)
BANNED_KEYS = frozenset(
    {
        "homeostatic_delta",
        "correct",
        "reward",
        "result",
        "stage",
        "lab",
        "capability",
        "answer",
        "intended",
        "expected",
    }
)
BODY_SETPOINT = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
# v12: opposite-sign ACT body_adv raises HOLD / lowers ACT on the next step.
# v13: compare body_adv to a slow agreeing baseline instead of last-tick snap.
CONFLICT_ADV_EPS = 1e-9
ELIG_EPS = 1e-12
TIE_EPS = 1e-12
CONFLICT_HOLD_BIAS = 2.0
ADV_BASELINE_ALPHA = 0.05
FAMILIARITY_RATIO = 0.5
FAMILIARITY_DECAY = 0.98
FAMILIARITY_ABS = 16.0
ECHOIC_MAX = 8
ECHOIC_BIAS = 0.08
VOCAL_REFRACTORY = 1.5
UTTERANCE_PERSIST = 1.5
EQUAL_EVIDENCE_MIN_SYMBOLS = 3
# v30: scalar mix on zero-input motor ticks only. 0 recovers v29. Freeze selected DEV p here.
MOTOR_PERSIST_P = 0.0
# v31: ACT ranking. "query" is v30 W_act_query; "proto" is actuator-local prototypes.
ACT_SCORE_QUERY = "query"
ACT_SCORE_PROTO = "proto"
PROTO_EPS = 1e-12
ACTUATOR_PROTO_H_MAX = 8
# v32: eight content-addressed P1 episodes. Frozen budgets, not a search.
EPISODE_SLOTS = 8
EPISODE_MATCH_L2 = 0.05
EPISODE_REPLAY_EPOCHS = 16
ACT_MARGIN_FLOOR = 0.01
# v38: instance-level rehearsal controller. Not a GenomeConfig field (TM031 to_dict).
ACT_REHEARSE_V37 = "v37_awake_cap"
ACT_REHEARSE_ADAPTIVE = "adaptive_violation"
ACT_REHEARSE_FIXED = "fixed_extra_replay"
ACT_REHEARSE_ARMS = (ACT_REHEARSE_V37, ACT_REHEARSE_ADAPTIVE, ACT_REHEARSE_FIXED)
ACT_REHEARSE_TARGETING = "violation_rows"
# v39: instance-level joint projection. Not a GenomeConfig field (TM031 to_dict).
ACT_PROJ_OFF = "off"
ACT_PROJ_PA = "pa_cyclic"
ACT_PROJ_DYKSTRA = "dykstra"
ACT_PROJ_ARMS = (ACT_PROJ_OFF, ACT_PROJ_PA, ACT_PROJ_DYKSTRA)
# v40: instance-level numerical joint SOCP. Not a GenomeConfig field (TM031 to_dict).
# Not an exact projector. Default off = frozen v37.
ACT_SOCP_OFF = "off"
ACT_SOCP_FALLBACK = "fallback_joint"
ACT_SOCP_ALWAYS = "always_joint"
ACT_SOCP_ARMS = (ACT_SOCP_OFF, ACT_SOCP_FALLBACK, ACT_SOCP_ALWAYS)
# TM044: learned memory projection. Not a GenomeConfig field. Not an ACT recall mode.
MEMPROJ_OFF = "off"
MEMPROJ_LEARNED = "learned_projection"
MEMPROJ_BIRTH = "birth_projection"
MEMPROJ_NONE = "no_persistent_memory"
MEMPROJ_ARMS = (MEMPROJ_OFF, MEMPROJ_LEARNED, MEMPROJ_BIRTH, MEMPROJ_NONE)
MEMPROJ_ETA_SCALE = 1.0
# Canonical telemetry: do not overload a single "path" with memory vs motor.
MEMORY_PATH_EPISODIC = "episodic_completed"
MEMORY_PATH_EMPTY = "empty"
MEMORY_PATH_REJECTED = "rejected"
MEMORY_PATHS = (MEMORY_PATH_EPISODIC, MEMORY_PATH_EMPTY, MEMORY_PATH_REJECTED)
MOTOR_PATH_CORTICAL = "cortical_scoring"
SCORE_SRC_LIVE = "live_rho"
SCORE_SRC_REINSTATED = "reinstated_value"
EPISODE_REJECT_REASONS = frozenset(
    {"exact_nearest_tie", "ambiguous_nearest", "integer_overlap_tie"}
)
OPAQUE_EMPTY_REASONS = frozenset({"empty_store", "zero_query", "no_valid_keys"})
OPAQUE_REJECT_REASONS = frozenset(
    {"bad_query", "dimensional_mismatch", "nonfinite_record", "exact_distance_tie"}
)
# v36: sparse hippocampal index. 12.5% sparsity; unrelated to EPISODE_SLOTS.
SEP_DIM = 64
SEP_K = 8
KEY_MATCH_MIN_OVERLAP = 5
ACT_RECALL_OFF = "off"
ACT_RECALL_RAW_P1 = "raw_p1"
ACT_RECALL_EARLY_RAW = "early_raw"
ACT_RECALL_EARLY_RAW_HALF = "early_raw_half_spacing"
ACT_RECALL_SEP_NO_FAM = "separated_key_no_familiarity"
ACT_RECALL_SEP = "separated_key"
# v37 half-spacing is accepted by _resolve_act_recall_mode / load_checkpoint but is
# intentionally omitted from ACT_RECALL_MODES. TM029 binds RECALL_MODES = list(ACT_RECALL_MODES)
# and that runner is frozen; adding a mode would drift its 82-cell manifest.
ACT_RECALL_MODES = (
    ACT_RECALL_OFF,
    ACT_RECALL_RAW_P1,
    ACT_RECALL_EARLY_RAW,
    ACT_RECALL_SEP_NO_FAM,
    ACT_RECALL_SEP,
)
HADAMARD_SIGN_XOR = 0b101011
HADAMARD_SIGN_MUL = 37
HADAMARD_SIGN_ADD = 11


def sylvester_hadamard(n: int) -> np.ndarray:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("hadamard size must be a power of 2")
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def hadamard_sign_perm(n: int = SEP_DIM) -> tuple[np.ndarray, np.ndarray]:
    signs = np.ones(n, dtype=np.float64)
    perm = np.empty(n, dtype=np.int64)
    bits = int(math.log2(n))
    for i in range(n):
        signs[i] = 1.0 if bin((HADAMARD_SIGN_MUL * i + HADAMARD_SIGN_ADD) & 0xFFFFFFFF).count("1") % 2 == 0 else -1.0
        rev = 0
        x = i
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        perm[i] = rev ^ HADAMARD_SIGN_XOR
    return signs, perm


def build_separator_matrix(n: int = SEP_DIM) -> np.ndarray:
    h = sylvester_hadamard(n)
    signs, perm = hadamard_sign_perm(n)
    w = (signs[:, None] * h)[:, perm] / math.sqrt(float(n))
    return np.ascontiguousarray(w.astype(np.float64))


SEPARATOR_MATRIX = build_separator_matrix(SEP_DIM)
SEPARATOR_MATRIX_SHA = hashlib.sha256(SEPARATOR_MATRIX.tobytes()).hexdigest()


def k_wta_binary(activations: np.ndarray, k: int = SEP_K) -> np.ndarray:
    a = np.asarray(activations, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(a.size, dtype=np.int64), -a))
    key = np.zeros(a.size, dtype=np.float64)
    key[order[: int(k)]] = 1.0
    return key


@dataclass
class GenomeConfig:
    n: int = 64
    d_sym: int = 32
    k_s: int = 8
    d_body: int = 4
    p_connect: float = 0.10
    t_max: int = 8
    tau: float = 1.0
    cos_thresh: float = 0.15
    eta_pred: float = 0.05
    eta_act: float = 0.15  # v3b amendment (was 0.05)
    beta: float = 0.01
    clip: float = 2.0
    seed_birth: int = 12345
    seed_registry: int = 22222
    seed_source: int = 33333
    seed_action: int = 44444
    seed_permute: int = 55555
    seed_motor: int = 66666
    dtype: str = "float64"
    # Optional lineage overlay. None → v27 module constants (make_cortex unchanged).
    lineage_params: dict[str, Any] | None = None
    # None → MOTOR_PERSIST_P. DEV grid may override without editing the module constant.
    motor_persist_p: float | None = None
    # v31: default query keeps live v30 until WRITEGEOM enables proto / candidate v31.
    act_score_mode: str = ACT_SCORE_QUERY
    actuator_proto_h_max: int = ACTUATOR_PROTO_H_MAX
    # v35: hippocampal-style episodic P1 reinstatement at ACT scoring only (default off = v34).
    episodic_act_recall: bool = False
    # v36: recall routing. Default off = v34. Legacy episodic_act_recall=True maps to raw_p1.
    act_recall_mode: str = ACT_RECALL_OFF

    @property
    def d_x(self) -> int:
        return self.d_sym + self.k_s * self.d_sym + self.d_body + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "d_sym": self.d_sym,
            "k_s": self.k_s,
            "d_body": self.d_body,
            "d_x": self.d_x,
            "p_connect": self.p_connect,
            "t_max": self.t_max,
            "tau": self.tau,
            "cos_thresh": self.cos_thresh,
            "eta_pred": self.eta_pred,
            "eta_act": self.eta_act,
            "beta": self.beta,
            "clip": self.clip,
            "seed_birth": self.seed_birth,
            "seed_registry": self.seed_registry,
            "seed_source": self.seed_source,
            "seed_action": self.seed_action,
            "seed_permute": self.seed_permute,
            "seed_motor": self.seed_motor,
            "dtype": self.dtype,
            "motor_persist_p": float(self.motor_persist_p if self.motor_persist_p is not None else MOTOR_PERSIST_P),
            "act_score_mode": str(self.act_score_mode),
            "actuator_proto_h_max": int(self.actuator_proto_h_max),
            "episodic_act_recall": bool(self.episodic_act_recall),
            "act_recall_mode": str(self.act_recall_mode),
            "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
            "body_setpoint": BODY_SETPOINT.tolist(),
            "ops": list(OPS),
            "op_cost": dict(OP_COST),
        }


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _np_state(rng: np.random.Generator) -> dict[str, Any]:
    return rng.bit_generator.state


def _restore_rng(seed: int, state: dict[str, Any] | None) -> np.random.Generator:
    rng = np.random.default_rng(seed)
    if state is not None:
        rng.bit_generator.state = state
    return rng


class NeuralCortex:
    """One plastic cortex organism with generic observe / S / motor surfaces."""

    def __init__(
        self,
        s_dir: Path | None = None,
        *,
        genome: GenomeConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.genome = genome or GenomeConfig()
        g = self.genome
        self.device = torch.device(device or "cpu")
        self.dtype = _torch_dtype(g.dtype)
        self.memory = CortexMemory(Path(s_dir) if s_dir is not None else None)
        self.age = 0
        self._t = 0
        self.dev_epoch = 0
        self._resting = False
        self._last_pred_err = 0.0
        self._body_setpoint = np.asarray(
            (g.lineage_params or {}).get("body_setpoint", BODY_SETPOINT),
            dtype=np.float64,
        ).reshape(-1)
        if self._body_setpoint.shape[0] != 4:
            self._body_setpoint = BODY_SETPOINT.copy()
        self._op_cost = dict(OP_COST)
        lp = g.lineage_params or {}
        for op in OPS:
            key = f"op_cost.{op}"
            if key in lp:
                self._op_cost[op] = float(lp[key])

        self.rng_birth = np.random.default_rng(g.seed_birth)
        self.rng_registry = np.random.default_rng(g.seed_registry)
        self.rng_source = np.random.default_rng(g.seed_source)
        self.rng_action = np.random.default_rng(g.seed_action)
        self.rng_permute = np.random.default_rng(g.seed_permute)
        self.rng_motor = np.random.default_rng(g.seed_motor)

        self.vocab: dict[str, np.ndarray] = {}
        self.sources: dict[str, np.ndarray] = {}
        # v5: empty at birth; bind_actuators fills motor_vocab from motor-registry RNG
        self.motor_vocab: dict[str, np.ndarray] = {}
        self._motor_registry: dict[str, np.ndarray] = {}
        # v31: actuator-local prototype rows (fast/slow). Keyed by opaque handle, never by cue.
        self._proto_fast: dict[str, np.ndarray] = {}
        self._proto_slow: dict[str, np.ndarray] = {}

        self.M = self._init_mask()
        self.W_rec = self._init_masked_rec()
        self.W_in = self._randn(g.n, g.d_x, g.d_x)
        self.b = torch.zeros(g.n, dtype=self.dtype, device=self.device)
        self.W_pred = self._randn(g.d_sym, g.n, g.n)
        self.W_op = self._randn(len(OPS), g.n, g.n)
        # v4: additive frozen op logit bias (milder; not plastic — see diagnosis.v3)
        self.b_op = torch.zeros(len(OPS), dtype=self.dtype, device=self.device)
        self.b_op[OPS.index("ACT")] = 0.85
        self.W_emit_query = self._randn(g.d_sym, g.n, g.n)
        # v6: no motor evidence at birth — zeros keep ACT a tie (rng_motor), not a geometry lottery
        self.W_act_query = torch.zeros(g.d_sym, g.n, dtype=self.dtype, device=self.device)
        self.W_write = self._randn(g.d_sym, g.n, g.n)
        self.W_att = self._randn(g.d_sym, g.n, g.n)
        self.W_body = self._randn(g.n, g.d_body, g.d_body)  # frozen

        self.v_start = self._np_vec()
        self.v_end = self._np_vec()
        # After existing birth draws so TM044 does not shift v_start/v_end or earlier tensors.
        self.W_k = self._randn(g.n, g.n, g.n)
        # Shared birth basis: a query can select a key written in the same Gaussian map.
        # Learning may specialize W_q. Independent W_q would make birth retrieval a coin-flip.
        self.W_q = self.W_k.detach().clone()
        self.W_v = self._randn(g.n, g.n, g.n)

        self._plastic_names = [
            "W_rec",
            "W_in",
            "W_pred",
            "W_op",
            "W_emit_query",
            "W_act_query",
            "W_write",
            "W_att",
        ]
        self.W_slow = {name: getattr(self, name).detach().clone() for name in self._plastic_names}
        self._birth_W = {name: getattr(self, name).detach().clone() for name in self._plastic_names}
        self._birth_W["W_body"] = self.W_body.detach().clone()
        self._birth_M = self.M.detach().clone()
        self._birth_v_start = self.v_start.copy()
        self._birth_v_end = self.v_end.copy()
        self._birth_b_op = self.b_op.detach().clone()
        self._birth_W_k = self.W_k.detach().clone()
        self._birth_W_q = self.W_q.detach().clone()
        self._birth_W_v = self.W_v.detach().clone()

        self.rho = torch.zeros(g.n, dtype=self.dtype, device=self.device)
        self.retrieval_buffer = torch.zeros(
            (g.k_s, g.d_sym), dtype=self.dtype, device=self.device
        )
        self.memory.on_write = self._on_s_write
        self.emit_buffer: list[str] = []
        self.last_body = self._body_setpoint.copy()
        self.last_s = np.zeros(g.d_sym, dtype=np.float64)
        self.last_s_hat = np.zeros(g.d_sym, dtype=np.float64)
        self.prev_interaction: str | None = None
        self._pending: dict[str, Any] | None = None
        self._pred_pending: dict[str, Any] | None = None
        self._pending_writes: list[CortexRecord] = []
        self._last_act_body_adv = 0.0
        self._adv_baseline = 0.0
        self._hold_after_conflict = False
        self._symbol_obs_counts: dict[str, int] = {}
        self._symbol_fam: dict[str, float] = {}
        self._echoic: list[str] = []
        self._vocal_next: str | None = None
        self._last_motor_class: str | None = None
        self.last_action: dict[str, Any] | None = None
        self.last_trajectory: list[np.ndarray] = []
        self.sensory_trajectory: list[np.ndarray] = []
        # v32: event-end P1 + eight-slot episode store. Empty at birth.
        self._last_p1: np.ndarray | None = None
        self._last_key_rho: np.ndarray | None = None
        self._last_event_key: np.ndarray | None = None
        self._separator = SEPARATOR_MATRIX.copy()
        self._separator_sha = SEPARATOR_MATRIX_SHA
        self._episodes: list[dict[str, Any]] = []
        self._episode_clock = 0
        self._episode_n_inserts = 0
        self._episode_n_replaced = 0
        self._n_rest_replay = 0
        self._n_rest_strengthen = 0
        self._act_rehearse_arm = ACT_REHEARSE_V37
        self._rehearse_targeting = ACT_REHEARSE_TARGETING
        self._rehearsal_pass_debt = 0
        self._rehearsal_update_debt = 0
        self._act_proj_arm = ACT_PROJ_OFF
        self._act_proj_corrections: dict[str, np.ndarray] = {}
        self._act_socp_arm = ACT_SOCP_OFF
        self._memproj_arm = MEMPROJ_OFF
        self._memproj_frozen = False
        self._memproj_rho_obs: np.ndarray | None = None
        # TM049: instance flag, not a GenomeConfig field, not an ACT_RECALL_MODE.
        self._action_feedback_enabled = False
        # TM058: experimental opaque K/V store. Default off. Not a genome field or ACT_RECALL_MODE.
        self._opaque_store_enabled = False
        self._opaque_kv_seq = 0
        self.opaque = OpaqueMemory()

    def set_act_rehearse_arm(self, arm: str) -> None:
        if arm not in ACT_REHEARSE_ARMS:
            raise ValueError(arm)
        self._act_rehearse_arm = str(arm)
        self._rehearse_targeting = ACT_REHEARSE_TARGETING

    def set_act_proj_arm(self, arm: str) -> None:
        if arm not in ACT_PROJ_ARMS:
            raise ValueError(arm)
        self._act_proj_arm = str(arm)

    def set_act_socp_arm(self, arm: str) -> None:
        if arm not in ACT_SOCP_ARMS:
            raise ValueError(arm)
        self._act_socp_arm = str(arm)

    def set_memproj_arm(self, arm: str) -> None:
        if arm not in MEMPROJ_ARMS:
            raise ValueError(arm)
        self._memproj_arm = str(arm)
        if arm == MEMPROJ_BIRTH:
            self._memproj_frozen = True
        elif arm == MEMPROJ_LEARNED:
            self._memproj_frozen = False

    def freeze_memproj_projection(self, frozen: bool = True) -> None:
        self._memproj_frozen = bool(frozen)

    def set_action_feedback_enabled(self, enabled: bool) -> None:
        self._action_feedback_enabled = bool(enabled)

    def set_opaque_store_enabled(self, enabled: bool) -> None:
        self._opaque_store_enabled = bool(enabled)

    def write_opaque_kv(
        self,
        key: np.ndarray,
        value: np.ndarray,
        *,
        handle: str,
        provenance_id: str,
    ) -> dict[str, Any]:
        """Flag-on opaque K/V write. Does not call _episode_write.

        provenance_id= is accepted for the frozen TM058 signature and is not stored.
        Stored provenance_id comes from the checkpointed organism counter.
        handle is diagnostic only.
        """
        _ = (handle, provenance_id)
        n_before = len(self.opaque.rows())
        reject: dict[str, Any] = {
            "outcome": "reject",
            "reason": "invalid_arrays",
            "provenance_id": None,
            "evicted_provenance_id": None,
            "n_before": int(n_before),
            "n_after": int(n_before),
        }
        if not bool(getattr(self, "_opaque_store_enabled", False)):
            reject["reason"] = "flag_off"
            return reject
        try:
            k = np.asarray(key, dtype=np.float64).reshape(-1).copy()
            v = np.asarray(value, dtype=np.float64).reshape(-1).copy()
        except (TypeError, ValueError):
            return reject
        if k.size == 0 or v.size == 0 or (not np.isfinite(k).all()) or (not np.isfinite(v).all()):
            return reject
        self._opaque_kv_seq = int(getattr(self, "_opaque_kv_seq", 0)) + 1
        pid = str(self._opaque_kv_seq)
        rec = self.opaque.append_immutable(k, v, provenance_id=pid, when=int(self._opaque_kv_seq))
        rec["reason"] = None
        rec["n_before"] = int(n_before)
        rec["n_after"] = len(self.opaque.rows())
        return rec

    def _legal_feedback_motor_vec(self, mv: Any) -> np.ndarray | None:
        if mv is None:
            return None
        try:
            v = np.asarray(mv, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if v.shape[0] != int(self.genome.d_sym):
            return None
        if not np.all(np.isfinite(v)):
            return None
        if float(np.linalg.norm(v)) <= PROTO_EPS:
            return None
        return v

    def _commit_action_feedback(
        self,
        *,
        injected: bool,
        handle: str | None,
        adv: float,
        event_key: np.ndarray | None,
        key_rho: np.ndarray | None,
        rho_obs: np.ndarray | None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if not injected or not handle or self._resting or self._last_p1 is None:
            return extra
        if float(adv) <= ELIG_EPS:
            return extra
        eta_a = float(self.genome.eta_act) * self._age_scale("eta_act_scale", 1.0)
        burst = self._credit_act_p1_episode(
            np.asarray(self._last_p1, dtype=np.float64),
            str(handle),
            float(adv),
            eta_a,
            event_key=event_key,
            key_rho=key_rho,
        )
        if burst is not None:
            extra["rehearsal_burst"] = burst
        mp = self._memproj_write_after_credit(
            np.asarray(self._last_p1, dtype=np.float64),
            rho_obs=None if rho_obs is None else np.asarray(rho_obs, dtype=np.float64),
        )
        if mp is not None:
            extra["memproj_write"] = mp
        return extra

    def restore_birth_memproj(self) -> None:
        self.W_k = self._birth_W_k.detach().clone()
        self.W_q = self._birth_W_q.detach().clone()
        self.W_v = self._birth_W_v.detach().clone()

    def memproj_hashes(self) -> dict[str, str]:
        from three_memory.joint_socp import weight_hash

        return {
            "W_k": weight_hash(self._from_t(self.W_k)),
            "W_q": weight_hash(self._from_t(self.W_q)),
            "W_v": weight_hash(self._from_t(self.W_v)),
            "W_act_query": weight_hash(self._from_t(self.W_act_query)),
        }

    def memproj_deltas(self) -> dict[str, float]:
        def fro(a: Tensor, b: Tensor) -> float:
            return float(torch.linalg.norm((a - b).reshape(-1)).item())

        return {
            "W_k": fro(self.W_k, self._birth_W_k),
            "W_q": fro(self.W_q, self._birth_W_q),
            "W_v": fro(self.W_v, self._birth_W_v),
        }

    def opaque_snapshot(self) -> list[dict[str, Any]]:
        return self.opaque.snapshot()

    def replace_opaque_rows(self, rows: list[dict[str, Any]]) -> None:
        self.opaque.restore(rows)

    def _memproj_active(self) -> bool:
        return str(getattr(self, "_memproj_arm", MEMPROJ_OFF) or MEMPROJ_OFF) in (
            MEMPROJ_LEARNED,
            MEMPROJ_BIRTH,
            MEMPROJ_NONE,
        )

    def _memproj_unit(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(v))
        if nrm <= PROTO_EPS:
            return v
        return v / nrm

    def _memproj_write_after_credit(
        self,
        rho_post: np.ndarray,
        *,
        rho_obs: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        arm = str(getattr(self, "_memproj_arm", MEMPROJ_OFF) or MEMPROJ_OFF)
        if arm not in (MEMPROJ_LEARNED, MEMPROJ_BIRTH, MEMPROJ_NONE):
            return None
        if rho_obs is None:
            rho_obs = self._memproj_rho_obs
        if rho_obs is None:
            rho_obs = self._from_t(self.rho)
        rho_obs = np.asarray(rho_obs, dtype=np.float64).reshape(-1)
        rho_post = np.asarray(rho_post, dtype=np.float64).reshape(-1)
        wk = self._from_t(self.W_k)
        wq = self._from_t(self.W_q)
        wv = self._from_t(self.W_v)
        k = wk @ rho_obs
        v = wv @ rho_post
        rec: dict[str, Any] = {
            "wrote": False,
            "arm": arm,
            "k_hash": None,
            "v_hash": None,
        }
        if arm != MEMPROJ_NONE:
            from three_memory.joint_socp import weight_hash

            row = self.opaque.write(k, v, provenance_id=f"p{int(self._t)}_{len(self.opaque.rows())}")
            rec["wrote"] = True
            rec["k_hash"] = weight_hash(np.asarray(row.key))
            rec["v_hash"] = weight_hash(np.asarray(row.value))
            rec["when"] = int(row.when)
        if arm == MEMPROJ_LEARNED and not bool(self._memproj_frozen):
            eta = float(self.genome.eta_act) * float(MEMPROJ_ETA_SCALE)
            k_tgt = self._memproj_unit(rho_obs)
            v_tgt = self._memproj_unit(rho_post)
            q = wq @ rho_obs
            self.W_k = self._to_t(wk + eta * np.outer(k_tgt - k, rho_obs))
            self.W_q = self._to_t(wq + eta * np.outer(k - q, rho_obs))
            self.W_v = self._to_t(wv + eta * np.outer(v_tgt - v, rho_post))
            c = float(self.genome.clip)
            self.W_k = torch.clamp(self.W_k, -c, c)
            self.W_q = torch.clamp(self.W_q, -c, c)
            self.W_v = torch.clamp(self.W_v, -c, c)
            rec["updated"] = True
        else:
            rec["updated"] = False
        return rec

    def event_memory_scores(self) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
        """Organism-owned path: rho → optional memory reinstatement → motor scores.

        Telemetry splits memory vs motor: memory_path, motor_path, scoring_address_source.
        scores_before_reinstatement / scores_after_reinstatement are diagnostics.
        """
        live = self._from_t(self.rho)
        if self._last_p1 is not None:
            live = np.asarray(self._last_p1, dtype=np.float64)
        live = np.asarray(live, dtype=np.float64).reshape(-1)
        arm = str(getattr(self, "_memproj_arm", MEMPROJ_OFF) or MEMPROJ_OFF)
        before_scores = {k: float(v) for k, v in self.actuator_scores(live).items()}
        addr = live
        memory_path = MEMORY_PATH_EMPTY
        source = SCORE_SRC_LIVE
        retrieved = False
        reject_reason = None
        n_rows = None
        if arm in (MEMPROJ_LEARNED, MEMPROJ_BIRTH):
            q = self._from_t(self.W_q) @ addr
            hit = self.opaque.retrieve(q)
            reject_reason = hit.get("reject_reason")
            n_rows = hit.get("n_rows")
            if bool(hit.get("hit")) and hit.get("value") is not None:
                addr = np.asarray(hit["value"], dtype=np.float64).reshape(-1)
                source = SCORE_SRC_REINSTATED
                retrieved = True
                memory_path = MEMORY_PATH_EMPTY
            elif str(reject_reason or "") in OPAQUE_REJECT_REASONS:
                memory_path = MEMORY_PATH_REJECTED
            else:
                memory_path = MEMORY_PATH_EMPTY
        scores, score_addr, smeta = self.actuator_decision_scores(addr)
        if smeta.get("memory_path") == MEMORY_PATH_EPISODIC:
            memory_path = MEMORY_PATH_EPISODIC
            source = SCORE_SRC_REINSTATED
        elif smeta.get("memory_path") == MEMORY_PATH_REJECTED and not retrieved:
            memory_path = MEMORY_PATH_REJECTED
            source = SCORE_SRC_LIVE
        meta: dict[str, Any] = {
            "memory_path": memory_path,
            "motor_path": MOTOR_PATH_CORTICAL,
            "scoring_address_source": source,
            "memproj_arm": arm,
            "retrieved": bool(retrieved),
            "reject_reason": reject_reason,
            "n_rows": n_rows,
            "scores_before_reinstatement": before_scores,
            "scores_after_reinstatement": {k: float(v) for k, v in scores.items()},
            "scoring_from": source,
            "path": memory_path,
        }
        meta.update({k: smeta.get(k) for k in ("ambiguous", "slot", "familiar") if k in smeta})
        meta["scoring_address_hash"] = hashlib.sha256(
            np.ascontiguousarray(np.asarray(score_addr, dtype=np.float64)).tobytes()
        ).hexdigest()
        return scores, np.asarray(score_addr, dtype=np.float64), meta

    def _act_proj_arm_active(self) -> bool:
        arm = str(getattr(self, "_act_proj_arm", ACT_PROJ_OFF) or ACT_PROJ_OFF)
        return arm in (ACT_PROJ_PA, ACT_PROJ_DYKSTRA)

    def _act_socp_always(self) -> bool:
        return str(getattr(self, "_act_socp_arm", ACT_SOCP_OFF) or ACT_SOCP_OFF) == ACT_SOCP_ALWAYS

    def _act_socp_fallback(self) -> bool:
        return str(getattr(self, "_act_socp_arm", ACT_SOCP_OFF) or ACT_SOCP_OFF) == ACT_SOCP_FALLBACK

    def _socp_constraints(self) -> list[dict[str, np.ndarray]]:
        rows: list[dict[str, np.ndarray]] = []
        rivals = sorted(str(h) for h in self.motor_vocab)
        for slot, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            if float(ep["adv"]) <= ELIG_EPS:
                continue
            handle = str(ep["handle"])
            if handle not in self.motor_vocab:
                continue
            x = self._unit_or_zero(np.asarray(ep["p1"], dtype=np.float64))
            if float(np.max(np.abs(x))) <= ELIG_EPS:
                continue
            v_h = np.asarray(self.motor_vocab[handle], dtype=np.float64).reshape(-1)
            for rival in rivals:
                if rival == handle:
                    continue
                v_r = np.asarray(self.motor_vocab[rival], dtype=np.float64).reshape(-1)
                d = v_h - v_r
                rows.append({"d": d, "x": x, "key": f"{int(slot)}|{handle}|{rival}"})
        return rows

    def _run_joint_socp_consolidation(self) -> dict[str, Any]:
        from three_memory.joint_socp import solve_min_change_socp, weight_hash

        w0 = self._from_t(self.W_act_query)
        w0_t = self.W_act_query.detach().clone()
        cons = self._socp_constraints()
        solved = solve_min_change_socp(w0, cons, float(ACT_MARGIN_FLOOR), float(PROTO_EPS))
        rec = {k: v for k, v in solved.items() if k != "W"}
        rec["not_an_exact_projector"] = True
        rec["violations_before"] = int(self._count_store_violations())
        rec["applied"] = False
        if solved.get("status") != "optimal" or solved.get("W") is None:
            rec["violations_after"] = rec["violations_before"]
            rec["w_hash_after"] = weight_hash(w0)
            return rec
        self.W_act_query = self._to_t(np.asarray(solved["W"], dtype=np.float64))
        self._clip_and_consolidate({"W_act_query"}, mix_slow=False)
        n_after = int(self._count_store_violations())
        rec["violations_after"] = n_after
        rec["w_hash_after"] = weight_hash(self._from_t(self.W_act_query))
        rec["frobenius_delta_after_clip"] = float(
            np.linalg.norm(self._from_t(self.W_act_query) - w0)
        )
        if n_after != 0:
            self.W_act_query = w0_t
            rec["applied"] = False
            rec["reject_reason"] = "organism_violation_after_clip"
            rec["status"] = "reject"
            rec["violations_after"] = int(self._count_store_violations())
            rec["w_hash_after"] = weight_hash(self._from_t(self.W_act_query))
            return rec
        rec["applied"] = True
        rec["reject_reason"] = None
        return rec

    def _zero_rehearsal_debt(self) -> None:
        self._rehearsal_pass_debt = 0
        self._rehearsal_update_debt = 0

    def _reset_proj_corrections_for_slot(self, slot: int) -> None:
        prefix = f"{int(slot)}|"
        raw = getattr(self, "_act_proj_corrections", None) or {}
        self._act_proj_corrections = {
            str(k): v for k, v in raw.items() if not str(k).startswith(prefix)
        }

    def _pa_project_W(
        self, W: np.ndarray, d: np.ndarray, x: np.ndarray, b: float
    ) -> tuple[np.ndarray, bool]:
        inner = float(np.dot(d, W @ x))
        a_f2 = float(np.dot(d, d) * np.dot(x, x))
        if a_f2 <= PROTO_EPS:
            return W, False
        if inner >= b:
            return W, False
        step = (b - inner) / a_f2
        return W + step * np.outer(d, x), True

    def _supporting_proj_constraints(self, W_ref: np.ndarray) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rivals = sorted(str(h) for h in self.motor_vocab)
        tau = float(ACT_MARGIN_FLOOR)
        for slot, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            if float(ep["adv"]) <= ELIG_EPS:
                continue
            handle = str(ep["handle"])
            if handle not in self.motor_vocab:
                continue
            x = self._unit_or_zero(np.asarray(ep["p1"], dtype=np.float64))
            if float(np.max(np.abs(x))) <= ELIG_EPS:
                continue
            v_h = np.asarray(self.motor_vocab[handle], dtype=np.float64).reshape(-1)
            for rival in rivals:
                if rival == handle:
                    continue
                v_r = np.asarray(self.motor_vocab[rival], dtype=np.float64).reshape(-1)
                d = v_h - v_r
                if float(np.linalg.norm(d)) <= PROTO_EPS:
                    continue
                b = tau * float(np.linalg.norm(W_ref.T @ d))
                rows.append(
                    {
                        "key": f"{int(slot)}|{handle}|{rival}",
                        "d": d,
                        "x": x,
                        "b": float(b),
                    }
                )
        return rows

    def _run_joint_projection_cycles(self) -> dict[str, Any]:
        corrections = str(getattr(self, "_act_proj_arm", ACT_PROJ_OFF) or ACT_PROJ_OFF) == ACT_PROJ_DYKSTRA
        W0 = self._from_t(self.W_act_query)
        corrections_state: dict[str, np.ndarray] = {}
        if corrections:
            raw = getattr(self, "_act_proj_corrections", None) or {}
            for k, v in raw.items():
                corrections_state[str(k)] = np.asarray(v, dtype=np.float64).copy()
        W = W0.copy()
        passes: list[dict[str, Any]] = []
        total_proj = 0
        first: int | None = None
        exhausted = False
        n_constraints = 0
        for idx in range(1, EPISODE_REPLAY_EPOCHS + 1):
            n_before = int(self._count_store_violations())
            cons = self._supporting_proj_constraints(W)
            n_constraints = len(cons)
            n_hit = 0
            for c in cons:
                key = str(c["key"])
                if corrections:
                    I = corrections_state.get(key)
                    if I is None:
                        I = np.zeros_like(W)
                    y = W - I
                    Wp, hit = self._pa_project_W(y, c["d"], c["x"], float(c["b"]))
                    corrections_state[key] = Wp - y
                    W = Wp
                else:
                    W, hit = self._pa_project_W(W, c["d"], c["x"], float(c["b"]))
                n_hit += int(hit)
                total_proj += int(hit)
            self.W_act_query = self._to_t(W)
            self._clip_and_consolidate({"W_act_query"}, mix_slow=False)
            W = self._from_t(self.W_act_query)
            n_after = int(self._count_store_violations())
            passes.append(
                {
                    "pass_index": int(idx),
                    "n_projections": int(n_hit),
                    "n_constraints": int(n_constraints),
                    "violations_before": int(n_before),
                    "violations_after": int(n_after),
                }
            )
            if n_after == 0:
                first = idx
                break
        else:
            exhausted = True
        if corrections:
            self._act_proj_corrections = {k: v.copy() for k, v in corrections_state.items()}
        delta = float(np.linalg.norm(W - W0))
        return {
            "n_awake_updates": int(total_proj),
            "n_projections": int(total_proj),
            "n_passes": len(passes),
            "n_constraints": int(n_constraints),
            "budget_exhausted": bool(exhausted),
            "first_converged_pass": first,
            "corrections": bool(corrections),
            "b_frozen_per_cycle": True,
            "clip_at_end_of_pass": True,
            "mix_slow": False,
            "fitted_learning_rate": False,
            "frobenius_delta": delta,
            "passes": passes,
        }

    # --- init helpers ---

    def _np_vec(self) -> np.ndarray:
        return self.rng_birth.normal(0.0, 1.0, size=self.genome.d_sym).astype(np.float64)

    def _randn(self, rows: int, cols: int, fan_in: int) -> Tensor:
        scale = 1.0 / math.sqrt(float(fan_in))
        arr = self.rng_birth.normal(0.0, scale, size=(rows, cols)).astype(np.float64)
        return torch.tensor(arr, dtype=self.dtype, device=self.device)

    def _init_mask(self) -> Tensor:
        g = self.genome
        u = self.rng_birth.random((g.n, g.n))
        m = (u < g.p_connect).astype(np.float64)
        np.fill_diagonal(m, 1.0)
        return torch.tensor(m, dtype=self.dtype, device=self.device)

    def _init_masked_rec(self) -> Tensor:
        g = self.genome
        scale = 1.0 / math.sqrt(float(g.n))
        arr = self.rng_birth.normal(0.0, scale, size=(g.n, g.n)).astype(np.float64)
        w = torch.tensor(arr, dtype=self.dtype, device=self.device)
        return w * self.M

    def _to_t(self, x: np.ndarray) -> Tensor:
        return torch.tensor(np.asarray(x, dtype=np.float64), dtype=self.dtype, device=self.device)

    def _from_t(self, x: Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float64)

    def _lp(self, key: str, default: float) -> float:
        p = self.genome.lineage_params
        if not p or key not in p:
            return float(default)
        return float(p[key])

    def _age_scale(self, name: str, default: float = 1.0) -> float:
        stages = (
            "birth",
            "high_plasticity",
            "experience_replay",
            "pruning_stabilization",
            "mature_plasticity",
            "novelty_reopen",
        )
        # Default mature-equivalent: all v27 scales are 1.0. Epoch 0 uses birth row still 1.0.
        idx = min(int(self.dev_epoch), len(stages) - 1)
        return self._lp(f"age.{stages[idx]}.{name}", default)

    def bind_actuators(self, handle_ids: list[str]) -> dict[str, Any]:
        """Universal actuator surface: opaque handle IDs → internal motor-registry vectors.

        Forbidden: runner-supplied vectors. Handle strings never enter sensory vocab.
        Rebinding the same handle restores its previously assigned vector.
        New handle → zero prototype. Rebound retains prototype. Unbound rows stay dormant.
        """
        if not isinstance(handle_ids, (list, tuple)):
            raise TypeError("bind_actuators requires a list of opaque handle id strings")
        hids: list[str] = []
        for raw in handle_ids:
            if isinstance(raw, dict):
                raise TypeError(
                    "bind_actuators forbids {id, vector} objects — cortex samples vectors"
                )
            hid = str(raw)
            if not hid or hid != hid.strip():
                raise ValueError(f"invalid actuator handle: {raw!r}")
            hids.append(hid)
        unique = set(self._proto_fast) | set(hids)
        h_max = int(self.genome.actuator_proto_h_max)
        if len(unique) > h_max:
            raise ValueError(
                f"actuator prototype H_max={h_max} exceeded ({len(unique)} unique handles)"
            )
        bound: list[str] = []
        self.motor_vocab = {}
        z = np.zeros(self.genome.n, dtype=np.float64)
        for hid in hids:
            if hid in self._motor_registry:
                vec = self._motor_registry[hid].copy()
            else:
                # Handle-keyed: bind order must not permute the vector identity.
                material = f"{int(self.genome.seed_motor):d}\0{hid}".encode()
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                rawv = rng.normal(0.0, 1.0, size=self.genome.d_sym).astype(np.float64)
                nrm = float(np.linalg.norm(rawv)) + 1e-12
                vec = (rawv / nrm).astype(np.float64)
                self._motor_registry[hid] = vec.copy()
            self.motor_vocab[hid] = vec
            if hid not in self._proto_fast:
                self._proto_fast[hid] = z.copy()
                self._proto_slow[hid] = z.copy()
            # deliberately do NOT insert into self.vocab (not neural sensory input)
            bound.append(hid)
        return {"bound": bound, "n": len(bound)}

    def _act_score_proto(self) -> bool:
        return str(self.genome.act_score_mode) == ACT_SCORE_PROTO

    def _rho_np(self, rho: Any) -> np.ndarray:
        if isinstance(rho, torch.Tensor):
            return self._from_t(rho)
        return np.asarray(rho, dtype=np.float64).reshape(-1)

    def _unit_or_zero(self, z: np.ndarray) -> np.ndarray:
        nrm = float(np.linalg.norm(z))
        if not np.isfinite(nrm) or nrm <= PROTO_EPS:
            return np.zeros(self.genome.n, dtype=np.float64)
        return (np.asarray(z, dtype=np.float64) / nrm).astype(np.float64)

    def _unique_act_winner(self, scores: dict[str, float]) -> str | None:
        if not scores:
            return None
        vals = [float(v) for v in scores.values()]
        if all(abs(s) <= TIE_EPS for s in vals):
            return None
        mx = max(vals)
        wins = [h for h, v in scores.items() if abs(float(v) - mx) <= TIE_EPS]
        return wins[0] if len(wins) == 1 else None

    def _act_ranking_error(self, p1: np.ndarray, handle: str, adv: float) -> bool:
        win = self._unique_act_winner(self.actuator_scores(p1))
        if float(adv) > 0.0:
            return win != str(handle)
        return win == str(handle) or win is None

    def _rival_mean_vector(self, p1: np.ndarray, handle: str) -> np.ndarray:
        scores = self.actuator_scores(p1)
        others = {h: float(v) for h, v in scores.items() if h != handle}
        m_h = np.asarray(self.motor_vocab[handle], dtype=np.float64).reshape(-1)
        if not others:
            return np.zeros_like(m_h)
        mx = max(others.values())
        rivals = [h for h, s in others.items() if s >= mx - TIE_EPS]
        if not rivals:
            return np.zeros_like(m_h)
        acc = np.zeros_like(m_h)
        for r in rivals:
            acc += np.asarray(self.motor_vocab[r], dtype=np.float64).reshape(-1)
        return acc / float(len(rivals))

    def _act_effective_row(self, handle: str) -> np.ndarray:
        W = self._from_t(self.W_act_query)
        v = np.asarray(self.motor_vocab[handle], dtype=np.float64).reshape(-1)
        return (W.T @ v).astype(np.float64)

    def _act_geometric_margin(self, p1: np.ndarray, handle: str) -> float:
        scores = self.actuator_scores(p1)
        others = [h for h in scores if h != handle]
        if not others:
            return 0.0
        rival = max(others, key=lambda h: float(scores[h]))
        w_ch = self._act_effective_row(handle)
        w_ot = self._act_effective_row(rival)
        x = self._unit_or_zero(p1)
        d = w_ch - w_ot
        dn = float(np.linalg.norm(d))
        if dn <= PROTO_EPS:
            return 0.0
        return float(np.dot(d, x) / dn)

    def _episode_write(
        self,
        p1: np.ndarray,
        handle: str,
        adv: float,
        *,
        event_key: np.ndarray | None = None,
        key_rho: np.ndarray | None = None,
    ) -> None:
        x = self._unit_or_zero(p1)
        if float(np.linalg.norm(x)) <= PROTO_EPS or abs(float(adv)) <= ELIG_EPS:
            return
        self._episode_clock += 1
        match_i: int | None = None
        best_d: float | None = None
        for i, old in enumerate(self._episodes):
            d = float(np.linalg.norm(old["p1"] - x))
            if d <= EPISODE_MATCH_L2 + ELIG_EPS and (best_d is None or d < best_d):
                match_i = i
                best_d = d
        ep = {
            "p1": x.copy(),
            "handle": str(handle),
            "adv": float(adv),
            "age": int(self._episode_clock),
            "version": 1,
            "valid": True,
            "key": None if event_key is None else np.asarray(event_key, dtype=np.float64).copy(),
            "key_rho": None if key_rho is None else np.asarray(key_rho, dtype=np.float64).copy(),
        }
        if match_i is None:
            if len(self._episodes) < EPISODE_SLOTS:
                self._episodes.append(ep)
            else:
                evict = min(range(len(self._episodes)), key=lambda i: (int(self._episodes[i]["age"]), i))
                self._reset_proj_corrections_for_slot(int(evict))
                self._episodes[evict] = ep
            self._episode_n_inserts += 1
            return
        old = self._episodes[int(match_i)]
        old_pos = float(old["adv"]) > 0.0
        new_pos = float(adv) > 0.0
        contradictory = False
        if old_pos != new_pos and abs(float(old["adv"])) > ELIG_EPS and abs(float(adv)) > ELIG_EPS:
            contradictory = True
        if str(handle) != str(old["handle"]) and float(adv) > 0.0:
            contradictory = True
        if contradictory:
            ep["version"] = int(old["version"]) + 1
            self._reset_proj_corrections_for_slot(int(match_i))
            self._episodes[int(match_i)] = ep
            self._episode_n_replaced += 1
            return
        old["age"] = int(self._episode_clock)
        old["adv"] = float(adv)
        if event_key is not None:
            old["key"] = np.asarray(event_key, dtype=np.float64).copy()
        if key_rho is not None:
            old["key_rho"] = np.asarray(key_rho, dtype=np.float64).copy()

    def _apply_act_query_update(
        self,
        p1: np.ndarray,
        handle: str,
        adv: float,
        eta_a: float,
        *,
        mix_slow: bool,
    ) -> bool:
        if handle not in self.motor_vocab or abs(float(adv)) <= ELIG_EPS:
            return False
        rho = self._unit_or_zero(p1)
        if float(np.max(np.abs(rho))) <= ELIG_EPS:
            return False
        m_h = np.asarray(self.motor_vocab[handle], dtype=np.float64).reshape(-1)
        if float(adv) > 0.0:
            m_r = self._rival_mean_vector(rho, handle)
            delta_m = m_h - m_r
        else:
            delta_m = m_h
        self.W_act_query = self.W_act_query + float(eta_a) * float(adv) * torch.outer(
            self._to_t(delta_m), self._to_t(rho)
        )
        self._clip_and_consolidate({"W_act_query"}, mix_slow=mix_slow)
        return True

    def _episode_matches_skip(
        self,
        ep: dict[str, Any],
        skip_p1: np.ndarray | None,
        skip_handle: str | None,
    ) -> bool:
        if skip_p1 is None or skip_handle is None:
            return False
        p1 = np.asarray(ep["p1"], dtype=np.float64)
        sp = self._unit_or_zero(skip_p1)
        if float(np.linalg.norm(sp)) <= PROTO_EPS:
            return False
        d = float(np.linalg.norm(p1 - sp))
        return d <= EPISODE_MATCH_L2 + ELIG_EPS and str(ep["handle"]) == str(skip_handle)

    def _episode_rehearsal_violation(self, p1: np.ndarray, handle: str, adv: float) -> bool:
        if handle not in self.motor_vocab or abs(float(adv)) <= ELIG_EPS:
            return False
        n_handles = len(self.motor_vocab)
        scores = self.actuator_scores(p1)
        win = self._unique_act_winner(scores)
        if n_handles <= 1:
            if float(adv) > 0.0:
                return self._act_geometric_margin(p1, handle) < ACT_MARGIN_FLOOR
            return win == str(handle) or win is None
        if float(adv) > 0.0:
            if win != str(handle) or win is None:
                return True
            return self._act_geometric_margin(p1, handle) < ACT_MARGIN_FLOOR
        return win == str(handle) or win is None

    def _count_store_violations(self) -> int:
        n, _slots = self._violation_signature()
        return n

    def _violation_signature(self) -> tuple[int, tuple[int, ...]]:
        """Plateau signature: (n_violations, sorted violating slot indices)."""
        slots: list[int] = []
        for i, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            if self._episode_rehearsal_violation(
                np.asarray(ep["p1"], dtype=np.float64),
                str(ep["handle"]),
                float(ep["adv"]),
            ):
                slots.append(int(i))
        return len(slots), tuple(slots)

    def store_rehearsal_checkpoint(self) -> dict[str, int | bool]:
        n_violations = self._count_store_violations()
        n_valid = sum(1 for ep in self._episodes if ep.get("valid"))
        return {
            "n_violations": int(n_violations),
            "all_margin_ok": bool(n_violations == 0),
            "n_episodes": int(n_valid),
        }

    def _gated_rehearsal_pass(
        self,
        eta_a: float,
        *,
        pass_index: int,
        skip_p1: np.ndarray | None = None,
        skip_handle: str | None = None,
    ) -> dict[str, int]:
        violations_before = self._count_store_violations()
        n_updates = 0
        n_opportunities = 0
        for ep in list(self._episodes):
            if not ep.get("valid"):
                continue
            p1 = np.asarray(ep["p1"], dtype=np.float64)
            handle = str(ep["handle"])
            adv = float(ep["adv"])
            if self._episode_matches_skip(ep, skip_p1, skip_handle):
                if not self._episode_rehearsal_violation(p1, handle, adv):
                    continue
            n_opportunities += 1
            if not self._episode_rehearsal_violation(p1, handle, adv):
                continue
            if self._apply_act_query_update(p1, handle, adv, eta_a, mix_slow=False):
                n_updates += 1
        violations_after = self._count_store_violations()
        return {
            "pass_index": int(pass_index),
            "violations_before": int(violations_before),
            "violations_after": int(violations_after),
            "n_updates": int(n_updates),
            "n_opportunities": int(n_opportunities),
        }

    def _run_awake_rehearsal_burst(
        self,
        *,
        skip_p1: np.ndarray | None = None,
        skip_handle: str | None = None,
    ) -> dict[str, Any]:
        arm = str(getattr(self, "_act_rehearse_arm", ACT_REHEARSE_V37) or ACT_REHEARSE_V37)
        if arm == ACT_REHEARSE_ADAPTIVE:
            return self._run_controller_rehearsal_burst(
                skip_p1=skip_p1,
                skip_handle=skip_handle,
                plateau_stop=True,
                accrue_debt=True,
            )
        if arm == ACT_REHEARSE_FIXED:
            return self._run_controller_rehearsal_burst(
                skip_p1=skip_p1,
                skip_handle=skip_handle,
                plateau_stop=False,
                accrue_debt=False,
            )
        return self._run_v37_awake_rehearsal_burst(skip_p1=skip_p1, skip_handle=skip_handle)

    def _run_v37_awake_rehearsal_burst(
        self,
        *,
        skip_p1: np.ndarray | None = None,
        skip_handle: str | None = None,
    ) -> dict[str, Any]:
        eta_a = float(self.genome.eta_act) * self._age_scale("eta_act_scale", 1.0)
        passes: list[dict[str, int]] = []
        first_converged: int | None = None
        total_updates = 0
        budget_exhausted = False
        for pass_index in range(1, EPISODE_REPLAY_EPOCHS + 1):
            ps = self._gated_rehearsal_pass(
                eta_a,
                pass_index=pass_index,
                skip_p1=skip_p1,
                skip_handle=skip_handle,
            )
            passes.append(ps)
            total_updates += int(ps["n_updates"])
            if int(ps["violations_after"]) == 0:
                if first_converged is None:
                    first_converged = pass_index
                break
        else:
            budget_exhausted = True
        return {
            "passes": passes,
            "n_passes": len(passes),
            "first_converged_pass": first_converged,
            "budget_exhausted": bool(budget_exhausted),
            "plateau_stopped": False,
            "total_updates": int(total_updates),
        }

    def _run_controller_rehearsal_burst(
        self,
        *,
        skip_p1: np.ndarray | None = None,
        skip_handle: str | None = None,
        plateau_stop: bool,
        accrue_debt: bool,
    ) -> dict[str, Any]:
        eta_a = float(self.genome.eta_act) * self._age_scale("eta_act_scale", 1.0)
        n_before, slots_before = self._violation_signature()
        if n_before == 0:
            return {
                "passes": [],
                "n_passes": 0,
                "first_converged_pass": 0,
                "budget_exhausted": False,
                "plateau_stopped": False,
                "total_updates": 0,
            }
        passes: list[dict[str, Any]] = []
        first_converged: int | None = None
        total_updates = 0
        budget_exhausted = False
        plateau_stopped = False
        for pass_index in range(1, EPISODE_REPLAY_EPOCHS + 1):
            ps = self._gated_rehearsal_pass(
                eta_a,
                pass_index=pass_index,
                skip_p1=skip_p1,
                skip_handle=skip_handle,
            )
            n_after, slots_after = self._violation_signature()
            if accrue_debt:
                self._rehearsal_pass_debt += 1
                self._rehearsal_update_debt += int(ps["n_updates"])
            row = dict(ps)
            row["violating_slots_before"] = list(slots_before)
            row["violating_slots_after"] = list(slots_after)
            row["n_violations_before"] = int(n_before)
            row["n_violations_after"] = int(n_after)
            passes.append(row)
            total_updates += int(ps["n_updates"])
            if n_after == 0:
                first_converged = pass_index
                break
            improved = (int(n_before) - int(n_after)) >= 1
            if plateau_stop and not improved:
                plateau_stopped = True
                break
            n_before, slots_before = n_after, slots_after
        else:
            budget_exhausted = True
        return {
            "passes": passes,
            "n_passes": len(passes),
            "first_converged_pass": first_converged,
            "budget_exhausted": bool(budget_exhausted),
            "plateau_stopped": bool(plateau_stopped),
            "total_updates": int(total_updates),
        }

    def _credit_act_p1_episode(
        self,
        p1: np.ndarray,
        handle: str,
        adv: float,
        eta_a: float,
        *,
        event_key: np.ndarray | None = None,
        key_rho: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        if float(np.max(np.abs(p1))) <= ELIG_EPS or self._resting:
            return None
        self._episode_write(p1, handle, adv, event_key=event_key, key_rho=key_rho)
        if self._act_socp_always():
            return self._run_joint_socp_consolidation()
        if self._act_proj_arm_active() and str(
            getattr(self, "_act_socp_arm", ACT_SOCP_OFF) or ACT_SOCP_OFF
        ) == ACT_SOCP_OFF:
            return self._run_joint_projection_cycles()
        if self._act_ranking_error(p1, handle, adv):
            self._apply_act_query_update(p1, handle, adv, eta_a, mix_slow=False)
        burst = self._run_awake_rehearsal_burst(skip_p1=p1, skip_handle=handle)
        if self._act_socp_fallback():
            n_after_v37 = int(self._count_store_violations())
            socp: dict[str, Any] | None = None
            if n_after_v37 > 0:
                socp = self._run_joint_socp_consolidation()
            return {
                "v37": burst,
                "violations_after_v37": n_after_v37,
                "socp_invoked": bool(n_after_v37 > 0),
                "socp": socp,
            }
        return burst

    def _replay_store_pass(self, eta_a: float, *, strengthen: bool) -> tuple[int, int]:
        """Legacy v33 hook; v34 uses _gated_rehearsal_pass only."""
        ps = self._gated_rehearsal_pass(eta_a, pass_index=1)
        return int(ps["n_updates"]), 0

    def _replay_episodes(self) -> dict[str, Any]:
        eta_a = float(self.genome.eta_act) * self._age_scale("eta_act_scale", 1.0)
        arm = str(getattr(self, "_act_rehearse_arm", ACT_REHEARSE_V37) or ACT_REHEARSE_V37)
        pass_debt_before = int(self._rehearsal_pass_debt)
        update_debt_before = int(self._rehearsal_update_debt)
        if arm == ACT_REHEARSE_ADAPTIVE:
            pass_budget = max(0, int(EPISODE_REPLAY_EPOCHS) - pass_debt_before)
        else:
            pass_budget = int(EPISODE_REPLAY_EPOCHS)
        epochs: list[dict[str, int]] = []
        first_converged: int | None = None
        total_updates = 0
        budget_exhausted = False
        for epoch_index in range(1, pass_budget + 1):
            ps = self._gated_rehearsal_pass(eta_a, pass_index=epoch_index)
            row = dict(ps)
            row["epoch_index"] = int(epoch_index)
            epochs.append(row)
            total_updates += int(ps["n_updates"])
            if int(ps["violations_after"]) == 0:
                if first_converged is None:
                    first_converged = epoch_index
                break
        else:
            budget_exhausted = bool(pass_budget > 0)
        violations_pre_mix = self._count_store_violations()
        self._clip_and_consolidate({"W_act_query"}, mix_slow=True)
        violations_post_mix = self._count_store_violations()
        self._n_rest_replay += total_updates
        self._n_rest_strengthen = 0
        if arm == ACT_REHEARSE_ADAPTIVE:
            # REST can repay at most 16 passes. Excess remains so leftover_debt
            # blocks lifecycle compute-conservation claims. No freeze edit.
            excess_pass = max(0, pass_debt_before - int(EPISODE_REPLAY_EPOCHS))
            self._rehearsal_pass_debt = int(excess_pass)
            self._rehearsal_update_debt = 0 if excess_pass == 0 else int(update_debt_before)
        return {
            "epochs": epochs,
            "first_converged_epoch": first_converged,
            "budget_exhausted": bool(budget_exhausted),
            "pass_budget": int(pass_budget),
            "total_updates": int(total_updates),
            "violations_pre_mix": int(violations_pre_mix),
            "violations_post_mix": int(violations_post_mix),
            "n_replay": int(total_updates),
            "n_strengthen": 0,
        }

    def actuator_scores(self, rho: Any) -> dict[str, float]:
        """Scalar ACT scores for every currently bound opaque handle."""
        r = self._rho_np(rho)
        rn = float(np.linalg.norm(r)) + 1e-12
        out: dict[str, float] = {}
        if self._act_score_proto():
            for h in self.motor_vocab:
                p = self._proto_fast.get(h)
                if p is None:
                    out[h] = 0.0
                    continue
                pn = float(np.linalg.norm(p))
                if pn <= PROTO_EPS:
                    out[h] = 0.0
                else:
                    out[h] = float(np.dot(p, r) / (pn * rn))
            return out
        q = self._from_t(self.W_act_query @ self._to_t(r))
        qn = float(np.linalg.norm(q)) + 1e-12
        for h, v in self.motor_vocab.items():
            out[h] = float(np.dot(q, v) / (qn * (np.linalg.norm(v) + 1e-12)))
        return out

    def _resolve_act_recall_mode(self) -> str:
        mode = str(getattr(self.genome, "act_recall_mode", ACT_RECALL_OFF) or ACT_RECALL_OFF)
        if mode not in ACT_RECALL_MODES and mode != ACT_RECALL_EARLY_RAW_HALF:
            mode = ACT_RECALL_OFF
        if mode == ACT_RECALL_OFF and bool(getattr(self.genome, "episodic_act_recall", False)):
            return ACT_RECALL_RAW_P1
        return mode

    def _episodic_act_recall_enabled(self) -> bool:
        return self._resolve_act_recall_mode() == ACT_RECALL_RAW_P1

    def _clear_event_key(self) -> None:
        self._last_key_rho = None
        self._last_event_key = None

    def _separate_event_key(self, key_rho: np.ndarray) -> np.ndarray:
        x = self._unit_or_zero(key_rho)
        act = self._separator @ x
        return k_wta_binary(act, SEP_K)

    def _copy_or_none(self, arr: Any) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float64).copy()

    def _key_overlap(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.dot(np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)))

    def _nearest_episode_for_recall(self, live_p1: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
        meta: dict[str, Any] = {
            "path": "cortical",
            "ambiguous": False,
            "slot": None,
            "nearest_dist": None,
            "second_nearest_dist": None,
            "reason": None,
            "familiar": False,
            "overlap": None,
            "act_recall_mode": self._resolve_act_recall_mode(),
        }
        if not self._episodic_act_recall_enabled():
            return None, meta
        x = self._unit_or_zero(live_p1)
        if float(np.linalg.norm(x)) <= PROTO_EPS:
            meta["path"] = "cortical_fallback"
            meta["reason"] = "empty_live_p1"
            return None, meta
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for i, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            d = float(np.linalg.norm(np.asarray(ep["p1"], dtype=np.float64) - x))
            candidates.append((d, int(i), ep))
        if not candidates:
            meta["path"] = "cortical_fallback"
            meta["reason"] = "no_valid_episodes"
            return None, meta
        candidates.sort(key=lambda t: t[0])
        min_d = float(candidates[0][0])
        nearest = [c for c in candidates if abs(c[0] - min_d) <= ELIG_EPS]
        meta["nearest_dist"] = float(min_d)
        meta["second_nearest_dist"] = float(candidates[1][0]) if len(candidates) > 1 else None
        if len(nearest) > 1:
            meta["path"] = "cortical_fallback"
            meta["ambiguous"] = True
            meta["reason"] = "ambiguous_nearest"
            return None, meta
        _d, slot_i, ep = nearest[0]
        meta["slot"] = int(slot_i)
        meta["path"] = "episodic_completed"
        meta["familiar"] = True
        return np.asarray(ep["p1"], dtype=np.float64).copy(), meta

    def _key_rho_spacing(self) -> dict[str, Any]:
        keyed: list[tuple[int, np.ndarray]] = []
        for i, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            stored = ep.get("key_rho")
            if stored is None:
                continue
            keyed.append((int(i), np.asarray(stored, dtype=np.float64).reshape(-1)))
        out: dict[str, Any] = {"n_keyed": len(keyed), "B": None, "R": None, "min_pair_slots": None}
        if len(keyed) < 2:
            return out
        best: float | None = None
        pair: list[int] | None = None
        for a in range(len(keyed)):
            for b in range(a + 1, len(keyed)):
                d = float(np.linalg.norm(keyed[a][1] - keyed[b][1]))
                if best is None or d < best:
                    best = d
                    pair = [int(keyed[a][0]), int(keyed[b][0])]
        out["B"] = best
        out["min_pair_slots"] = pair
        if best is None or best == 0.0:
            out["R"] = 0.0
        else:
            out["R"] = 0.5 * float(best)
        return out

    def _nearest_episode_by_key_rho(
        self,
        key_rho: np.ndarray,
        *,
        require_familiarity: bool = False,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        spacing = self._key_rho_spacing()
        meta: dict[str, Any] = {
            "path": "cortical_fallback",
            "ambiguous": False,
            "slot": None,
            "nearest_dist": None,
            "second_nearest_dist": None,
            "reason": None,
            "familiar": False,
            "overlap": None,
            "act_recall_mode": ACT_RECALL_EARLY_RAW_HALF if require_familiarity else ACT_RECALL_EARLY_RAW,
            "R": spacing.get("R"),
            "B": spacing.get("B"),
            "n_keyed": spacing.get("n_keyed"),
            "min_pair_slots": spacing.get("min_pair_slots"),
        }
        x = self._unit_or_zero(key_rho)
        if float(np.linalg.norm(x)) <= PROTO_EPS:
            meta["reason"] = "empty_key_rho"
            return None, meta
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        missing = False
        for i, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            stored = ep.get("key_rho")
            if stored is None:
                missing = True
                continue
            d = float(np.linalg.norm(np.asarray(stored, dtype=np.float64) - x))
            candidates.append((d, int(i), ep))
        if not candidates:
            meta["reason"] = "missing_legacy_key_rho" if missing else "no_valid_episodes"
            return None, meta
        candidates.sort(key=lambda t: t[0])
        min_d = float(candidates[0][0])
        meta["nearest_dist"] = float(min_d)
        meta["second_nearest_dist"] = float(candidates[1][0]) if len(candidates) > 1 else None
        if require_familiarity:
            n_keyed = int(spacing.get("n_keyed") or 0)
            r = spacing.get("R")
            if n_keyed < 2:
                meta["reason"] = "n_keyed_lt_2"
                return None, meta
            if r is None or r == 0.0:
                meta["reason"] = "R_eq_0"
                return None, meta
            nearest = [c for c in candidates if c[0] == min_d]
        else:
            nearest = [c for c in candidates if abs(c[0] - min_d) <= ELIG_EPS]
        if len(nearest) > 1:
            meta["ambiguous"] = True
            meta["reason"] = "exact_nearest_tie" if require_familiarity else "ambiguous_nearest"
            return None, meta
        if require_familiarity:
            r = float(spacing["R"])
            if not (min_d <= r):
                meta["reason"] = "beyond_half_spacing"
                meta["familiar"] = False
                return None, meta
        _d, slot_i, ep = nearest[0]
        meta["slot"] = int(slot_i)
        meta["path"] = "episodic_completed"
        meta["familiar"] = True
        return np.asarray(ep["p1"], dtype=np.float64).copy(), meta

    def _nearest_episode_by_sparse_key(
        self,
        query_key: np.ndarray,
        *,
        require_familiarity: bool,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        meta: dict[str, Any] = {
            "path": "cortical_fallback",
            "ambiguous": False,
            "slot": None,
            "nearest_dist": None,
            "second_nearest_dist": None,
            "reason": None,
            "familiar": False,
            "overlap": None,
            "act_recall_mode": ACT_RECALL_SEP if require_familiarity else ACT_RECALL_SEP_NO_FAM,
        }
        q = np.asarray(query_key, dtype=np.float64).reshape(-1)
        if q.size != SEP_DIM or float(np.sum(q)) <= 0.0:
            meta["reason"] = "empty_event_key"
            return None, meta
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        missing = False
        for i, ep in enumerate(self._episodes):
            if not ep.get("valid"):
                continue
            stored = ep.get("key")
            if stored is None:
                missing = True
                continue
            ov = self._key_overlap(q, stored)
            candidates.append((int(ov), int(i), ep))
        if not candidates:
            meta["reason"] = "missing_legacy_keys" if missing else "no_valid_episodes"
            return None, meta
        best = max(int(c[0]) for c in candidates)
        winners = [c for c in candidates if int(c[0]) == best]
        meta["overlap"] = int(best)
        seconds = sorted((int(c[0]) for c in candidates), reverse=True)
        meta["second_nearest_dist"] = float(seconds[1]) if len(seconds) > 1 else None
        meta["nearest_dist"] = float(best)
        if len(winners) > 1:
            meta["ambiguous"] = True
            meta["reason"] = "integer_overlap_tie"
            return None, meta
        if require_familiarity and best < KEY_MATCH_MIN_OVERLAP:
            meta["reason"] = "novel_or_weak_key"
            meta["familiar"] = False
            return None, meta
        _ov, slot_i, ep = winners[0]
        meta["slot"] = int(slot_i)
        meta["path"] = "episodic_completed"
        meta["familiar"] = True
        return np.asarray(ep["p1"], dtype=np.float64).copy(), meta

    def _pattern_complete_p1(self, stored_p1: np.ndarray) -> np.ndarray:
        return self._unit_or_zero(stored_p1)

    def actuator_decision_scores(
        self,
        live_p1: Any,
        *,
        key_rho: np.ndarray | None = None,
        event_key: np.ndarray | None = None,
    ) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
        """Canonical ACT motor path: optional episodic completion then query scoring."""
        live = self._rho_np(live_p1)
        mode = self._resolve_act_recall_mode()
        stored: np.ndarray | None = None
        if mode == ACT_RECALL_OFF:
            meta = {
                "path": "cortical",
                "ambiguous": False,
                "slot": None,
                "nearest_dist": None,
                "second_nearest_dist": None,
                "reason": None,
                "familiar": False,
                "overlap": None,
                "act_recall_mode": mode,
            }
        elif mode == ACT_RECALL_RAW_P1:
            stored, meta = self._nearest_episode_for_recall(live)
        elif mode == ACT_RECALL_EARLY_RAW:
            src = key_rho if key_rho is not None else self._last_key_rho
            if src is None:
                meta = {
                    "path": "cortical_fallback",
                    "ambiguous": False,
                    "slot": None,
                    "nearest_dist": None,
                    "second_nearest_dist": None,
                    "reason": "missing_live_key_rho",
                    "familiar": False,
                    "overlap": None,
                    "act_recall_mode": mode,
                }
            else:
                stored, meta = self._nearest_episode_by_key_rho(src)
        elif mode == ACT_RECALL_EARLY_RAW_HALF:
            src = key_rho if key_rho is not None else self._last_key_rho
            if src is None:
                meta = {
                    "path": "cortical_fallback",
                    "ambiguous": False,
                    "slot": None,
                    "nearest_dist": None,
                    "second_nearest_dist": None,
                    "reason": "missing_live_key_rho",
                    "familiar": False,
                    "overlap": None,
                    "act_recall_mode": mode,
                    "R": None,
                    "B": None,
                    "n_keyed": None,
                    "min_pair_slots": None,
                }
            else:
                stored, meta = self._nearest_episode_by_key_rho(src, require_familiarity=True)
        else:
            qk = event_key if event_key is not None else self._last_event_key
            if qk is None:
                src = key_rho if key_rho is not None else self._last_key_rho
                qk = None if src is None else self._separate_event_key(src)
            if qk is None:
                meta = {
                    "path": "cortical_fallback",
                    "ambiguous": False,
                    "slot": None,
                    "nearest_dist": None,
                    "second_nearest_dist": None,
                    "reason": "missing_live_event_key",
                    "familiar": False,
                    "overlap": None,
                    "act_recall_mode": mode,
                }
            else:
                stored, meta = self._nearest_episode_by_sparse_key(
                    qk, require_familiarity=mode == ACT_RECALL_SEP
                )
        if stored is not None:
            score_addr = self._pattern_complete_p1(stored)
        else:
            score_addr = self._unit_or_zero(live)
        scores = self.actuator_scores(score_addr)
        meta["scoring_address_norm"] = float(np.linalg.norm(score_addr))
        meta["act_recall_mode"] = mode
        if stored is not None:
            meta["memory_path"] = MEMORY_PATH_EPISODIC
            meta["scoring_address_source"] = SCORE_SRC_REINSTATED
        elif str(meta.get("reason") or "") in EPISODE_REJECT_REASONS:
            meta["memory_path"] = MEMORY_PATH_REJECTED
            meta["scoring_address_source"] = SCORE_SRC_LIVE
        else:
            meta["memory_path"] = MEMORY_PATH_EMPTY
            meta["scoring_address_source"] = SCORE_SRC_LIVE
        meta["motor_path"] = MOTOR_PATH_CORTICAL
        return scores, score_addr, meta

    def _choose_actuator_from_scores(self, scores: dict[str, float]) -> str | None:
        if not scores:
            return None
        best = max(scores.values())
        ties = sorted(h for h, s in scores.items() if abs(s - best) <= TIE_EPS)
        if len(ties) > 1 or all(abs(s) <= TIE_EPS for s in scores.values()):
            return str(self.rng_motor.choice(ties))
        return ties[0]

    def _choose_actuator(self, rho: Any) -> str | None:
        scores = self.actuator_scores(rho)
        return self._choose_actuator_from_scores(scores)

    def _credit_proto(self, handle: str, adv: float, rho_elig: Any, eta_a: float) -> None:
        if adv == 0.0 or handle not in self._proto_fast:
            return
        rho = self._rho_np(rho_elig)
        if float(np.max(np.abs(rho))) <= ELIG_EPS:
            return
        rn = float(np.linalg.norm(rho))
        if not np.isfinite(rn) or rn <= PROTO_EPS:
            return
        rhat = rho / rn
        live = np.asarray(self._proto_fast[handle], dtype=np.float64)
        z = live + float(eta_a) * float(adv) * rhat
        live = self._unit_or_zero(z)
        beta = float(self.genome.beta) * self._age_scale("beta_scale", 1.0)
        slow = np.asarray(self._proto_slow.get(handle, np.zeros(self.genome.n)), dtype=np.float64)
        slow = (1.0 - beta) * slow + beta * live
        live = slow + 0.5 * (live - slow)
        self._proto_fast[handle] = self._unit_or_zero(live)
        self._proto_slow[handle] = self._unit_or_zero(slow)

    # --- registries ---

    def _vocab_vec(self, token: str) -> np.ndarray:
        tok = str(token).strip().lower()
        if tok not in self.vocab:
            self.vocab[tok] = self.rng_registry.normal(
                0.0, 1.0, size=self.genome.d_sym
            ).astype(np.float64)
        return self.vocab[tok]

    def _source_vec(self, token: str) -> np.ndarray:
        tok = str(token).strip().lower()
        if tok not in self.sources:
            self.sources[tok] = self.rng_source.normal(
                0.0, 1.0, size=self.genome.d_sym
            ).astype(np.float64)
        return self.sources[tok]

    def encode_state_set(self, symbols: list[str]) -> np.ndarray:
        if not symbols:
            return np.zeros(self.genome.d_sym, dtype=np.float64)
        acc = np.zeros(self.genome.d_sym, dtype=np.float64)
        for u in sorted({str(x).strip().lower() for x in symbols if str(x).strip()}):
            acc = acc + self._vocab_vec(u)
        return acc

    # --- core tick ---

    def _x_tick(self, injected: np.ndarray, body: np.ndarray, same_ix: float) -> Tensor:
        g = self.genome
        inj = self._to_t(injected)
        buf = self.retrieval_buffer.reshape(-1)
        bod = self._to_t(body)
        six = torch.tensor([same_ix], dtype=self.dtype, device=self.device)
        return torch.cat([inj, buf, bod, six], dim=0)

    def _sensory_tick(
        self,
        injected: np.ndarray,
        body: np.ndarray,
        same_ix: float,
        *,
        record_sensory: bool = False,
    ) -> None:
        g = self.genome
        x = self._x_tick(injected, body, same_ix)
        pre = (self.W_rec * self.M) @ self.rho + self.W_in @ x + self.b
        pre = pre + self.W_body @ self._to_t(body)
        rho_tilde = torch.tanh(pre)
        p_raw = self.genome.motor_persist_p
        p = float(MOTOR_PERSIST_P if p_raw is None else p_raw)
        if (not record_sensory) and p > 0.0:
            self.rho = p * self.rho + (1.0 - p) * rho_tilde
        else:
            self.rho = rho_tilde
        snap = self._from_t(self.rho)
        self.last_trajectory.append(snap)
        if record_sensory:
            self.sensory_trajectory.append(snap)

    # --- motor helpers ---

    def _softmax_sample(self, logits: Tensor) -> int:
        z = self._from_t(logits) / float(self.genome.tau)
        z = z - np.max(z)
        e = np.exp(z)
        p = e / np.sum(e)
        return int(self.rng_action.choice(len(OPS), p=p))

    def _best_token(
        self,
        query: Tensor,
        *,
        lexicon: dict[str, np.ndarray] | None = None,
        require_thresh: bool = True,
        rng: np.random.Generator | None = None,
        echoic_bias: bool = False,
    ) -> str | None:
        pool = lexicon if lexicon is not None else self.vocab
        if not pool:
            return None
        q = self._from_t(query)
        qn = np.linalg.norm(q) + 1e-12
        echoic = set(self._echoic) if echoic_bias else set()
        scored: list[tuple[float, str]] = []
        for tok, v in pool.items():
            cos = float(np.dot(q, v) / (qn * (np.linalg.norm(v) + 1e-12)))
            if tok in echoic:
                cos = cos + self._lp("echoic_bias", ECHOIC_BIAS)
            scored.append((cos, tok))
        best = max(c for c, _t in scored)
        if require_thresh and best < self.genome.cos_thresh:
            return None
        ties = sorted(t for c, t in scored if abs(c - best) <= 1e-12)
        if rng is not None and len(ties) > 1:
            return str(rng.choice(ties))
        return ties[0]

    def _on_s_write(self, rec: CortexRecord) -> None:
        v = np.asarray(rec.content, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.genome.d_sym:
            return
        buf = self._from_t(self.retrieval_buffer)
        buf[0] = v
        self.retrieval_buffer = self._to_t(buf)

    def _do_retrieve(self) -> None:
        q = self.W_att @ self.rho
        qn = self._from_t(q)
        qnorm = np.linalg.norm(qn) + 1e-12
        rows = list(self.memory.records())
        if not rows:
            self._pending_retrieve = np.zeros((self.genome.k_s, self.genome.d_sym), dtype=np.float64)
            return
        if self._resting:
            w_rec = self._lp("replay.mix.recency", 0.25)
            w_sim = self._lp("replay.mix.similarity", 0.25)
            w_sur = self._lp("replay.mix.surprise", 0.25)
            w_rnd = self._lp("replay.mix.random", 0.25)
            wsum = max(w_rec + w_sim + w_sur + w_rnd, 1e-12)
            w_rec, w_sim, w_sur, w_rnd = (w_rec / wsum, w_sim / wsum, w_sur / wsum, w_rnd / wsum)
            tmax = max((int(r.when) for r in rows), default=1)
            scored: list[tuple[float, int, str, np.ndarray]] = []
            rng = self.rng_action
            for rec in rows:
                v = np.asarray(rec.content, dtype=np.float64)
                if v.shape[0] != self.genome.d_sym:
                    continue
                cos = float(np.dot(qn, v) / (qnorm * (np.linalg.norm(v) + 1e-12)))
                recency = float(int(rec.when) + 1) / float(tmax + 1)
                surprise = float((rec.tags or {}).get("surprise") or 0.0)
                rnd = float(rng.random())
                score = w_sim * cos + w_rec * recency + w_sur * surprise + w_rnd * rnd
                scored.append((score, int(rec.when), rec.fact_id, v))
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        else:
            scored = []
            for rec in rows:
                v = np.asarray(rec.content, dtype=np.float64)
                if v.shape[0] != self.genome.d_sym:
                    continue
                cos = float(np.dot(qn, v) / (qnorm * (np.linalg.norm(v) + 1e-12)))
                scored.append((cos, int(rec.when), rec.fact_id, v))
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        buf = np.zeros((self.genome.k_s, self.genome.d_sym), dtype=np.float64)
        for i, (_c, _when, _fid, v) in enumerate(scored[: self.genome.k_s]):
            buf[i] = v
        self._pending_retrieve = buf

    def _commit_pending_retrieve(self) -> None:
        if getattr(self, "_pending_retrieve", None) is not None:
            self.retrieval_buffer = self._to_t(self._pending_retrieve)
            self._pending_retrieve = None

    def _motor_loop(self, body: np.ndarray, same_ix: float) -> dict[str, Any]:
        g = self.genome
        self.emit_buffer = []
        staged_writes: list[CortexRecord] = []
        chosen_op = "HOLD"
        chosen_token: str | None = None
        rho_elig = self._from_t(self.rho)
        applied_retrieve = False
        conflict_hold = bool(self._hold_after_conflict)
        self._hold_after_conflict = False
        vocal_next = self._vocal_next
        self._vocal_next = None

        for _k in range(g.t_max):
            self._commit_pending_retrieve()
            # internal tick sensory: zeros inject except buffer/body/same_ix
            zero = np.zeros(g.d_sym, dtype=np.float64)
            self._sensory_tick(zero, body, same_ix, record_sensory=False)
            logits = (self.W_op @ self.rho) + self.b_op
            if conflict_hold or vocal_next:
                logits = logits.clone()
            if conflict_hold:
                hb = self._lp("conflict_hold_bias", CONFLICT_HOLD_BIAS) * self._age_scale(
                    "conflict_hold_scale", 1.0
                )
                logits[OPS.index("HOLD")] = logits[OPS.index("HOLD")] + hb
                logits[OPS.index("ACT")] = logits[OPS.index("ACT")] - hb
            vr = self._lp("vocal_refractory", VOCAL_REFRACTORY) * self._age_scale(
                "refractory", 1.0
            )
            if vocal_next == "HOLD":
                logits[OPS.index("HOLD")] = logits[OPS.index("HOLD")] + vr
                logits[OPS.index("EMIT")] = logits[OPS.index("EMIT")] - vr
                logits[OPS.index("ACT")] = logits[OPS.index("ACT")] - vr
            elif vocal_next == "EMIT":
                logits[OPS.index("EMIT")] = logits[OPS.index("EMIT")] + vr
                logits[OPS.index("HOLD")] = logits[OPS.index("HOLD")] - vr
            elif vocal_next == "ACT":
                logits[OPS.index("ACT")] = logits[OPS.index("ACT")] + vr
                logits[OPS.index("HOLD")] = logits[OPS.index("HOLD")] - vr
            op_i = self._softmax_sample(logits)
            op = OPS[op_i]
            chosen_op = op
            # eligibility from the action-selection tick
            rho_elig = self._from_t(self.rho)

            if op == "HOLD":
                self.emit_buffer = []
                break
            if op == "STOP":
                break
            if op == "EMIT":
                tok = self._best_token(self.W_emit_query @ self.rho, echoic_bias=True)
                if tok is None:
                    chosen_op = "HOLD"
                    self.emit_buffer = []
                    break
                self.emit_buffer.append(tok)
                chosen_token = tok
                continue
            if op == "ACT":
                # v2: argmax over M_act only; never force HOLD on cosine miss
                # v6: tie-break via motor RNG (exchangeable slots)
                # v31: actuator_scores (query or proto); all-zero still rng_motor
                # v32: score ACT at event-end P1 when captured
                addr = self._last_p1 if self._last_p1 is not None else self._from_t(self.rho)
                act_scores, _score_addr, _recall_meta = self.actuator_decision_scores(addr)
                tok = self._choose_actuator_from_scores(act_scores)
                if tok is None:
                    chosen_op = "HOLD"
                    break
                chosen_token = tok
                break
            if op == "RETRIEVE":
                self._do_retrieve()
                applied_retrieve = True
                continue
            if op == "WRITE":
                write_t = self._from_t(self.W_write @ self.rho)
                fid = f"cw_{self._t:06d}_{len(staged_writes)}"
                rec = CortexRecord(
                    fact_id=fid,
                    content=write_t.tolist(),
                    when=int(self._t),
                    interaction_token=str(self.prev_interaction or ""),
                    source_token="",
                    source="cortex_write",
                    tags={"surprise": float(self._last_pred_err)},
                )
                staged_writes.append(rec)
                continue

        # commit writes after tick
        for rec in staged_writes:
            self.memory.write(rec)

        s_hat = self._from_t(self.W_pred @ self.rho)
        motor_vec = None
        if chosen_op == "ACT" and chosen_token and chosen_token in self.motor_vocab:
            motor_vec = self.motor_vocab[chosen_token].copy()
        out = {
            "op": chosen_op,
            "token": chosen_token,
            "emit_sequence": list(self.emit_buffer),
            "rho_elig": rho_elig,
            "s_hat": s_hat,
            "body": body.copy(),
            "cost": float(self._op_cost[chosen_op]),
            "retrieved": applied_retrieve,
            "motor_vec": motor_vec,
        }
        if chosen_op in ("EMIT", "ACT"):
            self._vocal_next = "HOLD"
            self._last_motor_class = chosen_op
        elif chosen_op == "HOLD" and self._last_motor_class in ("EMIT", "ACT"):
            self._vocal_next = self._last_motor_class
        else:
            self._vocal_next = None
        self.last_action = out
        self.last_s_hat = s_hat
        return out

    def _actor_pending_from_action(
        self,
        action: dict[str, Any],
        *,
        interaction_token: str | None,
        clamped: bool,
        op: str | None = None,
        token: str | None = None,
        motor_vec: np.ndarray | None = None,
    ) -> dict[str, Any]:
        chosen_op = str(op or action["op"])
        chosen_tok = token if token is not None else action.get("token")
        rho = np.asarray(action["rho_elig"], dtype=np.float64).copy()
        mv = motor_vec
        if mv is None and chosen_op == "ACT" and chosen_tok and chosen_tok in self.motor_vocab:
            mv = self.motor_vocab[chosen_tok].copy()
        elif mv is None:
            mv = None if action.get("motor_vec") is None else np.asarray(action["motor_vec"], dtype=np.float64).copy()
        return {
            "op": chosen_op,
            "token": chosen_tok,
            "rho_elig": rho,
            "rho_op": rho.copy(),
            "rho_motor": rho.copy(),
            "rho_p1": None if self._last_p1 is None else np.asarray(self._last_p1, dtype=np.float64).copy(),
            "event_key": self._copy_or_none(self._last_event_key),
            "key_rho": self._copy_or_none(self._last_key_rho),
            "s_hat": np.asarray(action["s_hat"], dtype=np.float64).copy(),
            "body": np.asarray(action.get("body", self.last_body), dtype=np.float64).copy(),
            "cost": float(self._op_cost.get(chosen_op, action.get("cost") or 0.0)),
            "motor_vec": mv,
            "authored": True,
            "clamped": bool(clamped),
            "t": int(self._t),
            "interaction_token": interaction_token,
        }

    def clamp_action(self, op: str, token: str | None, *, credit_token: str | None = None) -> dict[str, Any]:
        """Host clamp after an organism selection tick. Keeps saved eligibility.

        Executed motor_vec comes from token. Episode / W_act_query labels use
        credit_token when provided, otherwise token. Default preserves TM049.
        """
        if self.last_action is None:
            return {"ok": False, "why": "no_selection_tick"}
        if op not in OPS:
            return {"ok": False, "why": "bad_op"}
        if credit_token is not None and op != "ACT":
            return {"ok": False, "why": "credit_token_requires_act"}
        motor_vec = None
        label = token
        if op == "ACT":
            if not token or token not in self.motor_vocab:
                return {"ok": False, "why": "unknown_handle"}
            motor_vec = self.motor_vocab[token].copy()
            if credit_token is not None:
                if credit_token not in self.motor_vocab:
                    return {"ok": False, "why": "unknown_credit_token"}
                label = credit_token
        self._pending = self._actor_pending_from_action(
            self.last_action,
            interaction_token=self.prev_interaction,
            clamped=True,
            op=op,
            token=label,
            motor_vec=motor_vec,
        )
        return {"ok": True, "op": op, "token": token, "clamped": True}

    def drop_actor_pending(self) -> None:
        """Passive imposed movement: no organism actor credit on the next body change."""
        self._pending = None

    def _apply_credit(self, s_t: np.ndarray, body_t: np.ndarray) -> dict[str, Any]:
        updated: set[str] = set()
        eta_p = float(self.genome.eta_pred) * self._age_scale("eta_pred_scale", 1.0)
        eta_a = float(self.genome.eta_act) * self._age_scale("eta_act_scale", 1.0)
        adv = 0.0
        pred_err = 0.0
        rehearsal_burst: dict[str, Any] | None = None
        p = self._pending
        if p is None:
            pp = self._pred_pending
            if pp is not None:
                eps = s_t - np.asarray(pp["s_hat"], dtype=np.float64)
                pred_err = float(np.linalg.norm(eps))
                rho_pred = self._to_t(pp["rho_elig"])
                if float(torch.max(torch.abs(rho_pred)).item()) > ELIG_EPS:
                    self.W_pred = self.W_pred + eta_p * torch.outer(self._to_t(eps), rho_pred)
                    updated.add("W_pred")
                self._pred_pending = None
            self._clip_and_consolidate(updated)
            self._last_pred_err = pred_err
            return {"adv": 0.0, "pred_err": pred_err}

        eps = s_t - p["s_hat"]
        pred_err = float(np.linalg.norm(eps))
        body_prev = p["body"]
        body_adv = float(
            np.linalg.norm(body_prev - self._body_setpoint)
            - np.linalg.norm(body_t - self._body_setpoint)
        )
        adv = body_adv - float(p["cost"])
        rho_op = self._to_t(p.get("rho_op", p["rho_elig"]))
        rho_motor = self._to_t(p.get("rho_motor", p["rho_elig"]))
        elig_op = bool(torch.max(torch.abs(rho_op)).item() > ELIG_EPS)
        elig_motor = bool(torch.max(torch.abs(rho_motor)).item() > ELIG_EPS)
        if elig_op:
            self.W_pred = self.W_pred + eta_p * torch.outer(self._to_t(eps), rho_op)
            updated.add("W_pred")
        e_op = torch.zeros(len(OPS), dtype=self.dtype, device=self.device)
        e_op[OPS.index(p["op"])] = 1.0
        skip_act_cost = p["op"] == "ACT" and abs(body_adv) < 1e-9
        if elig_op and not skip_act_cost:
            self.W_op = self.W_op + eta_a * adv * torch.outer(e_op, rho_op)
            updated.add("W_op")
        if p["op"] == "ACT" and abs(body_adv) > CONFLICT_ADV_EPS:
            ema = float(self._adv_baseline)
            if abs(ema) > CONFLICT_ADV_EPS and (ema * body_adv) < 0.0:
                self._hold_after_conflict = True
                if elig_op:
                    e_conf = torch.zeros(len(OPS), dtype=self.dtype, device=self.device)
                    e_conf[OPS.index("HOLD")] = 1.0
                    e_conf[OPS.index("ACT")] = -1.0
                    self.W_op = self.W_op + eta_a * abs(body_adv) * torch.outer(e_conf, rho_op)
                    updated.add("W_op")
            else:
                alpha = self._lp("adv_baseline_alpha", ADV_BASELINE_ALPHA)
                self._adv_baseline = (1.0 - alpha) * ema + alpha * float(body_adv)
            self._last_act_body_adv = float(body_adv)
        defer_fb = bool(getattr(self, "_action_feedback_enabled", False))
        if p["op"] == "ACT" and p["token"] is not None and not skip_act_cost:
            if self._act_score_proto():
                if elig_motor:
                    self._credit_proto(
                        str(p["token"]),
                        float(adv),
                        p.get("rho_motor", p["rho_elig"]),
                        eta_a,
                    )
            elif not defer_fb:
                p1_raw = p.get("rho_p1")
                ek = p.get("event_key")
                kr = p.get("key_rho")
                if p1_raw is not None:
                    p1 = np.asarray(p1_raw, dtype=np.float64)
                    burst = self._credit_act_p1_episode(
                        p1,
                        str(p["token"]),
                        float(adv),
                        eta_a,
                        event_key=None if ek is None else np.asarray(ek, dtype=np.float64),
                        key_rho=None if kr is None else np.asarray(kr, dtype=np.float64),
                    )
                    if burst is not None:
                        rehearsal_burst = burst
                elif elig_motor:
                    p1 = np.asarray(p.get("rho_motor", p["rho_elig"]), dtype=np.float64)
                    burst = self._credit_act_p1_episode(
                        p1,
                        str(p["token"]),
                        float(adv),
                        eta_a,
                        event_key=None if ek is None else np.asarray(ek, dtype=np.float64),
                        key_rho=None if kr is None else np.asarray(kr, dtype=np.float64),
                    )
                    if burst is not None:
                        rehearsal_burst = burst
        elif (
            p["op"] == "EMIT"
            and elig_motor
            and p["token"] is not None
            and not skip_act_cost
        ):
            tok_v = self._to_t(self._vocab_vec(p["token"]))
            self.W_emit_query = self.W_emit_query + eta_a * adv * torch.outer(tok_v, rho_motor)
            updated.add("W_emit_query")
        self._clip_and_consolidate(updated)
        self._pending = None
        self._pred_pending = None
        self._last_pred_err = pred_err
        out: dict[str, Any] = {"adv": adv, "pred_err": pred_err}
        if rehearsal_burst is not None:
            out["rehearsal_burst"] = rehearsal_burst
        if p["op"] == "ACT" and not defer_fb:
            rho_obs = p.get("rho_p1")
            mp = self._memproj_write_after_credit(
                self._from_t(self.rho),
                rho_obs=None if rho_obs is None else np.asarray(rho_obs, dtype=np.float64),
            )
            if mp is not None:
                out["memproj_write"] = mp
        return out

    def _clip_and_consolidate(self, names: set[str] | None = None, *, mix_slow: bool = True) -> None:
        if not names:
            return
        c = self.genome.clip
        beta = float(self.genome.beta) * self._age_scale("beta_scale", 1.0)
        for name in self._plastic_names:
            if name not in names:
                continue
            W = getattr(self, name)
            W = torch.clamp(W, -c, c)
            if name == "W_rec":
                W = W * self.M
            if mix_slow:
                slow = self.W_slow[name]
                slow = (1.0 - beta) * slow + beta * W
                W = slow + 0.5 * (W - slow)
                if name == "W_rec":
                    W = W * self.M
                self.W_slow[name] = slow.detach().clone()
            setattr(self, name, W)

    # --- public ABI ---

    def observe(self, info: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "why": "", "action": None, "metrics": {}}
        if not isinstance(info, dict):
            out["why"] = "not_dict"
            return out
        keys = set(info.keys())
        if keys & BANNED_KEYS:
            out["why"] = "banned_key"
            return out
        if keys != OBSERVE_KEYS:
            out["why"] = "exact_key_reject"
            return out
        try:
            ix = str(info["interaction_token"]).strip().lower()
            src = str(info["source_token"]).strip().lower()
            ordered = [
                str(x).strip().lower()
                for x in info["ordered_symbols"]
                if str(x).strip()
            ]
            state_syms = [
                str(x).strip().lower()
                for x in info["observable_state"]
                if str(x).strip()
            ]
            body = np.asarray(info["body_state"], dtype=np.float64).reshape(-1)
        except Exception:
            out["why"] = "field_type"
            return out
        if body.shape[0] != self.genome.d_body:
            out["why"] = "body_dim"
            return out
        if not ix or not src:
            out["why"] = "empty_field"
            return out

        same_ix = 1.0 if (self.prev_interaction is not None and self.prev_interaction == ix) else 0.0
        s_t = self.encode_state_set(state_syms)
        fb = bool(getattr(self, "_action_feedback_enabled", False))
        p0 = self._pending
        fb_motor = None
        fb_key = None
        fb_ek = None
        fb_handle = None
        fb_rho_obs = None
        if fb and p0 is not None:
            fb_motor = self._legal_feedback_motor_vec(p0.get("motor_vec"))
            fb_key = self._copy_or_none(p0.get("key_rho"))
            fb_ek = self._copy_or_none(p0.get("event_key"))
            tok = p0.get("token")
            fb_handle = None if tok is None else str(tok)
            rp = p0.get("rho_p1")
            fb_rho_obs = None if rp is None else np.asarray(rp, dtype=np.float64).copy()
            p0["motor_vec"] = None
            p0["key_rho"] = None
        metrics = self._apply_credit(s_t, body)

        self.last_trajectory = []
        self.sensory_trajectory = []
        self._clear_event_key()
        # sensory microticks
        start_inj = self.v_start + self._source_vec(src)
        self._sensory_tick(start_inj, body, same_ix, record_sensory=True)
        for u in ordered:
            self._sensory_tick(self._vocab_vec(u), body, same_ix, record_sensory=True)
        injected = False
        if fb and fb_motor is not None:
            self._sensory_tick(fb_motor, body, same_ix, record_sensory=True)
            injected = True
        if not self._resting:
            self._last_key_rho = self._unit_or_zero(self._from_t(self.rho))
            self._last_event_key = self._separate_event_key(self._last_key_rho)
        self._sensory_tick(self.v_end, body, same_ix, record_sensory=True)
        if not self._resting:
            self._last_p1 = self._unit_or_zero(self._from_t(self.rho))
            self._memproj_rho_obs = np.asarray(self._last_p1, dtype=np.float64).copy()
            if fb:
                extra = self._commit_action_feedback(
                    injected=injected,
                    handle=fb_handle,
                    adv=float(metrics.get("adv") or 0.0),
                    event_key=fb_ek,
                    key_rho=fb_key,
                    rho_obs=fb_rho_obs,
                )
                if extra:
                    metrics.update(extra)
        self._sensory_tick(s_t, body, same_ix, record_sensory=True)

        for k in list(self._symbol_fam):
            faded = float(self._symbol_fam[k]) * self._lp("familiarity_decay", FAMILIARITY_DECAY)
            if faded < 1e-9:
                self._symbol_fam.pop(k, None)
            else:
                self._symbol_fam[k] = faded
        for u in ordered:
            self._symbol_obs_counts[u] = int(self._symbol_obs_counts.get(u, 0)) + 1
            self._symbol_fam[u] = float(self._symbol_fam.get(u, 0.0)) + 1.0
            self._echoic.append(u)
        echo_max = int(self._lp("echoic_max", float(ECHOIC_MAX)))
        if len(self._echoic) > echo_max:
            self._echoic = self._echoic[-echo_max:]
        if self._pending is not None:
            body_prev = np.asarray(self._pending["body"], dtype=np.float64)
            cur_body_adv = float(
                np.linalg.norm(body_prev - self._body_setpoint) - np.linalg.norm(body - self._body_setpoint)
            )
        else:
            cur_body_adv = 0.0
        if (
            len(ordered) >= 2
            and abs(cur_body_adv) <= CONFLICT_ADV_EPS
            and len(self._symbol_obs_counts) >= int(self._lp("equal_evidence_min_symbols", float(EQUAL_EVIDENCE_MIN_SYMBOLS)))
        ):
            self._hold_after_conflict = True
        elif len(ordered) == 1:
            if float(self._symbol_fam.get(ordered[0], 0.0)) < self._lp("familiarity_abs", FAMILIARITY_ABS):
                self._hold_after_conflict = True

        action = self._motor_loop(body, same_ix)
        self._pred_pending = {
            "s_hat": np.asarray(action["s_hat"], dtype=np.float64).copy(),
            "rho_elig": np.asarray(action["rho_elig"], dtype=np.float64).copy(),
        }
        if action["op"] in ("ACT", "EMIT"):
            self._pending = self._actor_pending_from_action(
                action,
                interaction_token=ix,
                clamped=False,
            )
        else:
            self._pending = None
        self.prev_interaction = ix
        self.last_body = body.copy()
        self.last_s = s_t
        self._t += 1
        self.age += 1
        out["ok"] = True
        out["why"] = "ok"
        out["action"] = {
            "op": action["op"],
            "token": action["token"],
            "emit_sequence": action["emit_sequence"],
        }
        out["metrics"] = metrics
        out["rho_norm"] = float(np.linalg.norm(self._from_t(self.rho)))
        return out

    def reset_rho(self) -> None:
        g = self.genome
        self.rho = torch.zeros(g.n, dtype=self.dtype, device=self.device)
        self.retrieval_buffer = torch.zeros(
            (g.k_s, g.d_sym), dtype=self.dtype, device=self.device
        )
        self.emit_buffer = []
        self.last_s_hat = np.zeros(g.d_sym, dtype=np.float64)
        self._pending = None
        self._pred_pending = None
        self._pending_retrieve = None
        self.last_trajectory = []
        self._clear_event_key()

    def rest_epoch(self, n_ticks: int, *, body: np.ndarray | None = None) -> dict[str, Any]:
        """Host rest opportunity. Cortical RETRIEVE plus v32 P1 episode replay. Host does not pick S rows."""
        n_ticks = int(max(0, n_ticks))
        body_arr = np.asarray(body if body is not None else self.last_body, dtype=np.float64)
        fatigued = body_arr.copy()
        if fatigued.shape[0] >= 4:
            fatigued[3] = min(1.0, float(fatigued[3]) + 0.3)
        self._resting = True
        ops: dict[str, int] = {}
        replay: dict[str, Any] = {"n_replay": 0, "n_strengthen": 0}
        try:
            for i in range(n_ticks):
                out = self.observe(
                    {
                        "interaction_token": f"rest_{self._t}_{i}",
                        "source_token": "src_rest",
                        "ordered_symbols": [],
                        "observable_state": ["st_idle"],
                        "body_state": fatigued.tolist(),
                    }
                )
                op = str((out.get("action") or {}).get("op") or "?")
                ops[op] = ops.get(op, 0) + 1
            replay = self._replay_episodes()
        finally:
            self._resting = False
        self.reset_rho()
        self._maybe_grow_prune()
        self.dev_epoch += 1
        return {
            "ok": True,
            "n": n_ticks,
            "op_counts": ops,
            "dev_epoch": int(self.dev_epoch),
            "n_replay": int(replay.get("n_replay", 0)),
            "n_strengthen": int(replay.get("n_strengthen", 0)),
            "rehearsal": replay,
        }

    def _maybe_grow_prune(self) -> None:
        grow = self._lp("connect.growth_rate", 0.0) * self._age_scale("growth_scale", 0.0)
        prune = self._lp("connect.prune_rate", 0.0) * self._age_scale("prune_scale", 0.0)
        if grow <= 0.0 and prune <= 0.0:
            return
        m = self._from_t(self.M)
        w = self._from_t(self.W_rec)
        rng = self.rng_birth
        n = int(m.shape[0])
        eye = np.eye(n, dtype=bool)
        if prune > 0.0:
            thr = self._lp("connect.prune_threshold", 0.0)
            cand = (m > 0) & (np.abs(w) < thr) & (~eye)
            ii, jj = np.where(cand)
            for i, j in zip(ii, jj, strict=True):
                if rng.random() < prune:
                    m[i, j] = 0.0
                    w[i, j] = 0.0
        if grow > 0.0:
            cand = (m == 0) & (~eye)
            ii, jj = np.where(cand)
            for i, j in zip(ii, jj, strict=True):
                if rng.random() < grow:
                    m[i, j] = 1.0
        self.M = self._to_t(m)
        self.W_rec = self._to_t(w) * self.M

    def reset_cortex(self) -> None:
        for name in self._plastic_names:
            setattr(self, name, self._birth_W[name].detach().clone())
            self.W_slow[name] = self._birth_W[name].detach().clone()
        self.W_body = self._birth_W["W_body"].detach().clone()
        self.M = self._birth_M.detach().clone()
        self.b_op = self._birth_b_op.detach().clone()
        self.v_start = self._birth_v_start.copy()
        self.v_end = self._birth_v_end.copy()
        self.age = 0
        self.dev_epoch = 0
        self._last_act_body_adv = 0.0
        self._adv_baseline = 0.0
        self._hold_after_conflict = False
        self._symbol_obs_counts = {}
        self._symbol_fam = {}
        self._echoic = []
        self._vocal_next = None
        self._last_motor_class = None
        self.reset_rho()
        self._last_p1 = None
        self._episodes = []
        self._episode_clock = 0
        self._episode_n_inserts = 0
        self._episode_n_replaced = 0
        self._n_rest_replay = 0
        self._n_rest_strengthen = 0
        self._zero_rehearsal_debt()
        self._act_proj_corrections = {}
        z = np.zeros(self.genome.n, dtype=np.float64)
        for h in list(self._proto_fast):
            self._proto_fast[h] = z.copy()
            self._proto_slow[h] = z.copy()

    def checkpoint(self) -> dict[str, Any]:
        def tsave(x: Tensor) -> list:
            return self._from_t(x).tolist()

        return {
            "genome": self.genome.to_dict(),
            "device": str(self.device),
            "age": self.age,
            "t": self._t,
            "dev_epoch": int(self.dev_epoch),
            "W": {name: tsave(getattr(self, name)) for name in self._plastic_names},
            "W_slow": {name: tsave(v) for name, v in self.W_slow.items()},
            "W_body": tsave(self.W_body),
            "b_op": tsave(self.b_op),
            "M": tsave(self.M),
            "rho": tsave(self.rho),
            "retrieval_buffer": tsave(self.retrieval_buffer),
            "emit_buffer": list(self.emit_buffer),
            "last_body": self.last_body.tolist(),
            "last_s": self.last_s.tolist(),
            "last_s_hat": self.last_s_hat.tolist(),
            "prev_interaction": self.prev_interaction,
            "pending": None
            if self._pending is None
            else {
                **{
                    k: v
                    for k, v in self._pending.items()
                    if k
                    not in (
                        "rho_elig",
                        "rho_op",
                        "rho_motor",
                        "rho_p1",
                        "event_key",
                        "key_rho",
                        "s_hat",
                        "body",
                        "motor_vec",
                    )
                },
                "rho_elig": np.asarray(self._pending["rho_elig"]).tolist(),
                "rho_op": np.asarray(self._pending.get("rho_op", self._pending["rho_elig"])).tolist(),
                "rho_motor": np.asarray(
                    self._pending.get("rho_motor", self._pending["rho_elig"])
                ).tolist(),
                "rho_p1": None
                if self._pending.get("rho_p1") is None
                else np.asarray(self._pending["rho_p1"]).tolist(),
                "event_key": None
                if self._pending.get("event_key") is None
                else np.asarray(self._pending["event_key"]).tolist(),
                "key_rho": None
                if self._pending.get("key_rho") is None
                else np.asarray(self._pending["key_rho"]).tolist(),
                "s_hat": np.asarray(self._pending["s_hat"]).tolist(),
                "body": np.asarray(self._pending["body"]).tolist(),
                "motor_vec": None
                if self._pending.get("motor_vec") is None
                else np.asarray(self._pending["motor_vec"]).tolist(),
            },
            "pred_pending": None
            if self._pred_pending is None
            else {
                "s_hat": np.asarray(self._pred_pending["s_hat"]).tolist(),
                "rho_elig": np.asarray(self._pred_pending["rho_elig"]).tolist(),
            },
            "vocab": {k: v.tolist() for k, v in self.vocab.items()},
            "motor_vocab": {k: v.tolist() for k, v in self.motor_vocab.items()},
            "motor_registry": {k: v.tolist() for k, v in self._motor_registry.items()},
            "proto_fast": {k: v.tolist() for k, v in self._proto_fast.items()},
            "proto_slow": {k: v.tolist() for k, v in self._proto_slow.items()},
            "last_p1": None if self._last_p1 is None else self._last_p1.tolist(),
            "last_key_rho": None if self._last_key_rho is None else self._last_key_rho.tolist(),
            "last_event_key": None if self._last_event_key is None else self._last_event_key.tolist(),
            "separator_matrix_sha": str(self._separator_sha),
            "episodes": [
                {
                    "p1": np.asarray(e["p1"]).tolist(),
                    "handle": str(e["handle"]),
                    "adv": float(e["adv"]),
                    "age": int(e["age"]),
                    "version": int(e["version"]),
                    "valid": bool(e["valid"]),
                    "key": None if e.get("key") is None else np.asarray(e["key"]).tolist(),
                    "key_rho": None if e.get("key_rho") is None else np.asarray(e["key_rho"]).tolist(),
                }
                for e in self._episodes
            ],
            "episode_clock": int(self._episode_clock),
            "episode_n_inserts": int(self._episode_n_inserts),
            "episode_n_replaced": int(self._episode_n_replaced),
            "n_rest_replay": int(self._n_rest_replay),
            "n_rest_strengthen": int(self._n_rest_strengthen),
            "rehearsal_pass_debt": int(self._rehearsal_pass_debt),
            "rehearsal_update_debt": int(self._rehearsal_update_debt),
            "act_rehearse_arm": str(self._act_rehearse_arm),
            "act_proj_arm": str(getattr(self, "_act_proj_arm", ACT_PROJ_OFF) or ACT_PROJ_OFF),
            "act_socp_arm": str(getattr(self, "_act_socp_arm", ACT_SOCP_OFF) or ACT_SOCP_OFF),
            "memproj_arm": str(getattr(self, "_memproj_arm", MEMPROJ_OFF) or MEMPROJ_OFF),
            "memproj_frozen": bool(getattr(self, "_memproj_frozen", False)),
            "action_feedback_enabled": bool(getattr(self, "_action_feedback_enabled", False)),
            "opaque_store_enabled": bool(getattr(self, "_opaque_store_enabled", False)),
            "opaque_kv_seq": int(getattr(self, "_opaque_kv_seq", 0)),
            "W_k": tsave(self.W_k),
            "W_q": tsave(self.W_q),
            "W_v": tsave(self.W_v),
            "birth_W_k": tsave(self._birth_W_k),
            "birth_W_q": tsave(self._birth_W_q),
            "birth_W_v": tsave(self._birth_W_v),
            "opaque": self.opaque.snapshot(),
            "act_proj_corrections": {
                str(k): np.asarray(v, dtype=np.float64).tolist()
                for k, v in (getattr(self, "_act_proj_corrections", None) or {}).items()
            },
            "sources": {k: v.tolist() for k, v in self.sources.items()},
            "v_start": self.v_start.tolist(),
            "v_end": self.v_end.tolist(),
            "S": self.memory.snapshot(),
            "rng": {
                "birth": _np_state(self.rng_birth),
                "registry": _np_state(self.rng_registry),
                "source": _np_state(self.rng_source),
                "action": _np_state(self.rng_action),
                "permute": _np_state(self.rng_permute),
                "motor": _np_state(self.rng_motor),
            },
            "last_act_body_adv": float(self._last_act_body_adv),
            "adv_baseline": float(self._adv_baseline),
            "hold_after_conflict": bool(self._hold_after_conflict),
            "symbol_obs_counts": {k: int(v) for k, v in self._symbol_obs_counts.items()},
            "symbol_fam": {k: float(v) for k, v in self._symbol_fam.items()},
            "echoic": list(self._echoic),
            "vocal_next": self._vocal_next,
            "last_motor_class": self._last_motor_class,
        }

    def load_checkpoint(self, snap: dict[str, Any]) -> None:
        def tload(data: list) -> Tensor:
            return self._to_t(np.asarray(data, dtype=np.float64))

        for name in self._plastic_names:
            setattr(self, name, tload(snap["W"][name]))
            self.W_slow[name] = tload(snap["W_slow"][name])
        self.W_body = tload(snap["W_body"])
        if "b_op" in snap:
            self.b_op = tload(snap["b_op"])
        else:
            self.b_op = self._birth_b_op.detach().clone()
        self.M = tload(snap["M"])
        self.rho = tload(snap["rho"])
        self.retrieval_buffer = tload(snap["retrieval_buffer"])
        self.emit_buffer = list(snap.get("emit_buffer") or [])
        self.last_body = np.asarray(snap["last_body"], dtype=np.float64)
        self.last_s = np.asarray(snap["last_s"], dtype=np.float64)
        self.last_s_hat = np.asarray(snap["last_s_hat"], dtype=np.float64)
        self.prev_interaction = snap.get("prev_interaction")
        self.age = int(snap.get("age") or 0)
        self._t = int(snap.get("t") or 0)
        self.dev_epoch = int(snap.get("dev_epoch") or 0)
        self.v_start = np.asarray(snap["v_start"], dtype=np.float64)
        self.v_end = np.asarray(snap["v_end"], dtype=np.float64)
        self.vocab = {
            k: np.asarray(v, dtype=np.float64) for k, v in (snap.get("vocab") or {}).items()
        }
        mv = snap.get("motor_vocab")
        if mv:
            self.motor_vocab = {
                k: np.asarray(v, dtype=np.float64) for k, v in mv.items()
            }
        else:
            self.motor_vocab = {}
        mr = snap.get("motor_registry")
        if mr:
            self._motor_registry = {
                k: np.asarray(v, dtype=np.float64) for k, v in mr.items()
            }
        else:
            # migrate: registry from motor_vocab snapshot
            self._motor_registry = {k: v.copy() for k, v in self.motor_vocab.items()}
        z = np.zeros(self.genome.n, dtype=np.float64)
        pf = snap.get("proto_fast")
        ps = snap.get("proto_slow")
        if pf:
            self._proto_fast = {k: np.asarray(v, dtype=np.float64) for k, v in pf.items()}
        else:
            # v29/v30: missing prototypes initialize to zero; never infer from motor_vocab
            keys = set(self.motor_vocab) | set(self._motor_registry)
            self._proto_fast = {k: z.copy() for k in keys}
        if ps:
            self._proto_slow = {k: np.asarray(v, dtype=np.float64) for k, v in ps.items()}
        else:
            self._proto_slow = {k: z.copy() for k in self._proto_fast}
        for k in set(self._proto_fast) - set(self._proto_slow):
            self._proto_slow[k] = z.copy()
        lp1 = snap.get("last_p1")
        self._last_p1 = None if lp1 is None else np.asarray(lp1, dtype=np.float64)
        lkr = snap.get("last_key_rho")
        self._last_key_rho = None if lkr is None else np.asarray(lkr, dtype=np.float64)
        lek = snap.get("last_event_key")
        self._last_event_key = None if lek is None else np.asarray(lek, dtype=np.float64)
        self._separator = SEPARATOR_MATRIX.copy()
        self._separator_sha = str(snap.get("separator_matrix_sha") or SEPARATOR_MATRIX_SHA)
        self._episodes = []
        for raw in snap.get("episodes") or []:
            self._episodes.append(
                {
                    "p1": np.asarray(raw["p1"], dtype=np.float64),
                    "handle": str(raw["handle"]),
                    "adv": float(raw["adv"]),
                    "age": int(raw.get("age") or 0),
                    "version": int(raw.get("version") or 1),
                    "valid": bool(raw.get("valid", True)),
                    "key": None if raw.get("key") is None else np.asarray(raw["key"], dtype=np.float64),
                    "key_rho": None if raw.get("key_rho") is None else np.asarray(raw["key_rho"], dtype=np.float64),
                }
            )
        self._episode_clock = int(snap.get("episode_clock") or 0)
        self._episode_n_inserts = int(snap.get("episode_n_inserts") or 0)
        self._episode_n_replaced = int(snap.get("episode_n_replaced") or 0)
        self._n_rest_replay = int(snap.get("n_rest_replay") or 0)
        self._n_rest_strengthen = int(snap.get("n_rest_strengthen") or 0)
        self._rehearsal_pass_debt = int(snap.get("rehearsal_pass_debt") or 0)
        self._rehearsal_update_debt = int(snap.get("rehearsal_update_debt") or 0)
        arm = str(snap.get("act_rehearse_arm") or ACT_REHEARSE_V37)
        self._act_rehearse_arm = arm if arm in ACT_REHEARSE_ARMS else ACT_REHEARSE_V37
        self._rehearse_targeting = ACT_REHEARSE_TARGETING
        proj = str(snap.get("act_proj_arm") or ACT_PROJ_OFF)
        self._act_proj_arm = proj if proj in ACT_PROJ_ARMS else ACT_PROJ_OFF
        socp_arm = str(snap.get("act_socp_arm") or ACT_SOCP_OFF)
        self._act_socp_arm = socp_arm if socp_arm in ACT_SOCP_ARMS else ACT_SOCP_OFF
        mp = str(snap.get("memproj_arm") or MEMPROJ_OFF)
        self._memproj_arm = mp if mp in MEMPROJ_ARMS else MEMPROJ_OFF
        self._memproj_frozen = bool(snap.get("memproj_frozen", False))
        self._action_feedback_enabled = bool(snap.get("action_feedback_enabled", False))
        self._opaque_store_enabled = bool(snap.get("opaque_store_enabled", False))
        self._opaque_kv_seq = int(snap.get("opaque_kv_seq") or 0)
        if "W_k" in snap:
            self.W_k = tload(snap["W_k"])
            self.W_q = tload(snap["W_q"])
            self.W_v = tload(snap["W_v"])
        if "birth_W_k" in snap:
            self._birth_W_k = tload(snap["birth_W_k"])
            self._birth_W_q = tload(snap["birth_W_q"])
            self._birth_W_v = tload(snap["birth_W_v"])
        self.opaque = OpaqueMemory()
        if snap.get("opaque"):
            self.opaque.restore(list(snap.get("opaque") or []))
        corr = snap.get("act_proj_corrections") or {}
        loaded: dict[str, np.ndarray] = {}
        for k, v in corr.items():
            arr = np.asarray(v, dtype=np.float64)
            loaded[str(k)] = arr
        self._act_proj_corrections = loaded
        gsnap = snap.get("genome") or {}
        if "act_score_mode" in gsnap:
            self.genome.act_score_mode = str(gsnap["act_score_mode"])
        if "actuator_proto_h_max" in gsnap:
            self.genome.actuator_proto_h_max = int(gsnap["actuator_proto_h_max"])
        if "episodic_act_recall" in gsnap:
            self.genome.episodic_act_recall = bool(gsnap["episodic_act_recall"])
        if "act_recall_mode" in gsnap:
            mode = str(gsnap["act_recall_mode"])
            if mode in ACT_RECALL_MODES or mode == ACT_RECALL_EARLY_RAW_HALF:
                self.genome.act_recall_mode = mode
            else:
                self.genome.act_recall_mode = ACT_RECALL_OFF
        self.sources = {
            k: np.asarray(v, dtype=np.float64) for k, v in (snap.get("sources") or {}).items()
        }
        pend = snap.get("pending")
        if pend is None:
            self._pending = None
        else:
            rho_elig = np.asarray(pend["rho_elig"], dtype=np.float64)
            self._pending = {
                "op": pend["op"],
                "token": pend.get("token"),
                "rho_elig": rho_elig,
                "rho_op": np.asarray(pend.get("rho_op", rho_elig), dtype=np.float64),
                "rho_motor": np.asarray(pend.get("rho_motor", rho_elig), dtype=np.float64),
                "rho_p1": None
                if pend.get("rho_p1") is None
                else np.asarray(pend["rho_p1"], dtype=np.float64),
                "event_key": None
                if pend.get("event_key") is None
                else np.asarray(pend["event_key"], dtype=np.float64),
                "key_rho": None
                if pend.get("key_rho") is None
                else np.asarray(pend["key_rho"], dtype=np.float64),
                "s_hat": np.asarray(pend["s_hat"], dtype=np.float64),
                "body": np.asarray(pend["body"], dtype=np.float64),
                "cost": float(pend["cost"]),
                "motor_vec": None
                if pend.get("motor_vec") is None
                else np.asarray(pend["motor_vec"], dtype=np.float64),
                "authored": bool(pend.get("authored", True)),
                "clamped": bool(pend.get("clamped", False)),
                "t": int(pend.get("t") or 0),
                "interaction_token": pend.get("interaction_token"),
            }
        pp = snap.get("pred_pending")
        if pp is None:
            self._pred_pending = None
        else:
            self._pred_pending = {
                "s_hat": np.asarray(pp["s_hat"], dtype=np.float64),
                "rho_elig": np.asarray(pp["rho_elig"], dtype=np.float64),
            }
        self.memory.restore(snap.get("S") or [])
        g = self.genome
        if "seed_motor" in (snap.get("genome") or {}):
            g.seed_motor = int(snap["genome"]["seed_motor"])
        rs = snap.get("rng") or {}
        self.rng_birth = _restore_rng(g.seed_birth, rs.get("birth"))
        self.rng_registry = _restore_rng(g.seed_registry, rs.get("registry"))
        self.rng_source = _restore_rng(g.seed_source, rs.get("source"))
        self.rng_action = _restore_rng(g.seed_action, rs.get("action"))
        self.rng_permute = _restore_rng(g.seed_permute, rs.get("permute"))
        self.rng_motor = _restore_rng(g.seed_motor, rs.get("motor"))
        self._last_act_body_adv = float(snap.get("last_act_body_adv") or 0.0)
        self._adv_baseline = float(snap.get("adv_baseline") or 0.0)
        self._hold_after_conflict = bool(snap.get("hold_after_conflict") or False)
        self._symbol_obs_counts = {str(k): int(v) for k, v in (snap.get("symbol_obs_counts") or {}).items()}
        self._symbol_fam = {str(k): float(v) for k, v in (snap.get("symbol_fam") or {}).items()}
        self._echoic = [str(x) for x in (snap.get("echoic") or [])][-int(self._lp("echoic_max", float(ECHOIC_MAX))):]
        vn = snap.get("vocal_next")
        self._vocal_next = str(vn) if vn in ("HOLD", "EMIT", "ACT") else None
        lm = snap.get("last_motor_class")
        self._last_motor_class = str(lm) if lm in ("EMIT", "ACT") else None

    def weight_hash(self) -> str:
        h = hashlib.sha256()
        for name in self._plastic_names:
            h.update(self._from_t(getattr(self, name)).tobytes())
        return h.hexdigest()
