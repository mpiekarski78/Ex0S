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
        """
        if not isinstance(handle_ids, (list, tuple)):
            raise TypeError("bind_actuators requires a list of opaque handle id strings")
        bound: list[str] = []
        self.motor_vocab = {}
        for raw in handle_ids:
            if isinstance(raw, dict):
                raise TypeError(
                    "bind_actuators forbids {id, vector} objects — cortex samples vectors"
                )
            hid = str(raw)
            if not hid or hid != hid.strip():
                raise ValueError(f"invalid actuator handle: {raw!r}")
            if hid in self._motor_registry:
                vec = self._motor_registry[hid].copy()
            else:
                # Handle-keyed: bind order must not permute the vector identity.
                material = f"{int(self.genome.seed_motor):d}\0{hid}".encode()
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                raw = rng.normal(0.0, 1.0, size=self.genome.d_sym).astype(np.float64)
                nrm = float(np.linalg.norm(raw)) + 1e-12
                vec = (raw / nrm).astype(np.float64)
                self._motor_registry[hid] = vec.copy()
            self.motor_vocab[hid] = vec
            # deliberately do NOT insert into self.vocab (not neural sensory input)
            bound.append(hid)
        return {"bound": bound, "n": len(bound)}

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
        self.rho = torch.tanh(pre)
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
                tok = self._best_token(
                    self.W_act_query @ self.rho,
                    lexicon=self.motor_vocab,
                    require_thresh=False,
                    rng=self.rng_motor,
                )
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

    def _apply_credit(self, s_t: np.ndarray, body_t: np.ndarray) -> dict[str, float]:
        if self._pending is None:
            return {"adv": 0.0, "pred_err": 0.0}
        p = self._pending
        eps = s_t - p["s_hat"]
        body_prev = p["body"]
        body_adv = float(
            np.linalg.norm(body_prev - self._body_setpoint)
            - np.linalg.norm(body_t - self._body_setpoint)
        )
        adv = body_adv - p["cost"]
        rho_elig = self._to_t(p["rho_elig"])
        # directed prediction
        eta_p = float(self.genome.eta_pred) * self._age_scale("eta_pred_scale", 1.0)
        eta_a = float(self.genome.eta_act) * self._age_scale("eta_act_scale", 1.0)
        self.W_pred = self.W_pred + eta_p * torch.outer(
            self._to_t(eps), rho_elig
        )
        # three-factor on op / motor query used
        e_op = torch.zeros(len(OPS), dtype=self.dtype, device=self.device)
        e_op[OPS.index(p["op"])] = 1.0
        # v7: no-consequence ACT cost must not extinguish ACT (skip W_op when body_adv≈0)
        skip_act_cost = p["op"] == "ACT" and abs(body_adv) < 1e-9
        if not skip_act_cost:
            self.W_op = self.W_op + eta_a * adv * torch.outer(e_op, rho_elig)
        # v13: opposite-sign vs slow baseline → HOLD; do not snap the baseline.
        if p["op"] == "ACT" and abs(body_adv) > CONFLICT_ADV_EPS:
            ema = float(self._adv_baseline)
            if abs(ema) > CONFLICT_ADV_EPS and (ema * body_adv) < 0.0:
                self._hold_after_conflict = True
                e_conf = torch.zeros(len(OPS), dtype=self.dtype, device=self.device)
                e_conf[OPS.index("HOLD")] = 1.0
                e_conf[OPS.index("ACT")] = -1.0
                self.W_op = self.W_op + eta_a * abs(body_adv) * torch.outer(e_conf, rho_elig)
            else:
                alpha = self._lp("adv_baseline_alpha", ADV_BASELINE_ALPHA)
                self._adv_baseline = (1.0 - alpha) * ema + alpha * float(body_adv)
            self._last_act_body_adv = float(body_adv)
        # v4: b_op frozen (non-plastic)
        # v6: motor-query credit only when body consequences differ; use selected snapshot
        if p["token"] is not None and p["op"] in ("EMIT", "ACT"):
            skip_motor = skip_act_cost
            if not skip_motor:
                if p["op"] == "ACT" and p.get("motor_vec") is not None:
                    tok_v = self._to_t(np.asarray(p["motor_vec"], dtype=np.float64))
                elif p["op"] == "ACT" and p["token"] in self.motor_vocab:
                    tok_v = self._to_t(self.motor_vocab[p["token"]])
                else:
                    tok_v = self._to_t(self._vocab_vec(p["token"]))
                mat_name = "W_emit_query" if p["op"] == "EMIT" else "W_act_query"
                W = getattr(self, mat_name)
                setattr(
                    self,
                    mat_name,
                    W + eta_a * adv * torch.outer(tok_v, rho_elig),
                )
        self._clip_and_consolidate()
        self._pending = None
        self._last_pred_err = float(np.linalg.norm(eps))
        return {"adv": adv, "pred_err": float(np.linalg.norm(eps))}

    def _clip_and_consolidate(self) -> None:
        c = self.genome.clip
        beta = float(self.genome.beta) * self._age_scale("beta_scale", 1.0)
        for name in self._plastic_names:
            W = getattr(self, name)
            W = torch.clamp(W, -c, c)
            if name == "W_rec":
                W = W * self.M
            slow = self.W_slow[name]
            slow = (1.0 - beta) * slow + beta * W
            W = slow + 0.5 * (W - slow)
            if name == "W_rec":
                W = W * self.M
            setattr(self, name, W)
            self.W_slow[name] = slow.detach().clone()

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
        metrics = self._apply_credit(s_t, body)

        self.last_trajectory = []
        self.sensory_trajectory = []
        # sensory microticks
        start_inj = self.v_start + self._source_vec(src)
        self._sensory_tick(start_inj, body, same_ix, record_sensory=True)
        for u in ordered:
            self._sensory_tick(self._vocab_vec(u), body, same_ix, record_sensory=True)
        self._sensory_tick(self.v_end, body, same_ix, record_sensory=True)
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
        # store eligibility for next observe
        self._pending = {
            "op": action["op"],
            "token": action["token"],
            "rho_elig": action["rho_elig"],
            "s_hat": action["s_hat"],
            "body": body.copy(),
            "cost": action["cost"],
            "motor_vec": None
            if action.get("motor_vec") is None
            else np.asarray(action["motor_vec"], dtype=np.float64).copy(),
        }
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
        self._pending_retrieve = None
        self.last_trajectory = []

    def rest_epoch(self, n_ticks: int, *, body: np.ndarray | None = None) -> dict[str, Any]:
        """Host rest opportunity. Replay selection is cortical RETRIEVE. Host does not pick S rows."""
        n_ticks = int(max(0, n_ticks))
        body_arr = np.asarray(body if body is not None else self.last_body, dtype=np.float64)
        fatigued = body_arr.copy()
        if fatigued.shape[0] >= 4:
            fatigued[3] = min(1.0, float(fatigued[3]) + 0.3)
        self._resting = True
        ops: dict[str, int] = {}
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
        finally:
            self._resting = False
        self.reset_rho()
        self._maybe_grow_prune()
        self.dev_epoch += 1
        return {"ok": True, "n": n_ticks, "op_counts": ops, "dev_epoch": int(self.dev_epoch)}

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

    def checkpoint(self) -> dict[str, Any]:
        def tsave(x: Tensor) -> list:
            return self._from_t(x).tolist()

        return {
            "genome": self.genome.to_dict(),
            "device": str(self.device),
            "age": self.age,
            "t": self._t,
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
                    if k not in ("rho_elig", "s_hat", "body", "motor_vec")
                },
                "rho_elig": np.asarray(self._pending["rho_elig"]).tolist(),
                "s_hat": np.asarray(self._pending["s_hat"]).tolist(),
                "body": np.asarray(self._pending["body"]).tolist(),
                "motor_vec": None
                if self._pending.get("motor_vec") is None
                else np.asarray(self._pending["motor_vec"]).tolist(),
            },
            "vocab": {k: v.tolist() for k, v in self.vocab.items()},
            "motor_vocab": {k: v.tolist() for k, v in self.motor_vocab.items()},
            "motor_registry": {k: v.tolist() for k, v in self._motor_registry.items()},
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
        self.sources = {
            k: np.asarray(v, dtype=np.float64) for k, v in (snap.get("sources") or {}).items()
        }
        pend = snap.get("pending")
        if pend is None:
            self._pending = None
        else:
            self._pending = {
                "op": pend["op"],
                "token": pend.get("token"),
                "rho_elig": np.asarray(pend["rho_elig"], dtype=np.float64),
                "s_hat": np.asarray(pend["s_hat"], dtype=np.float64),
                "body": np.asarray(pend["body"], dtype=np.float64),
                "cost": float(pend["cost"]),
                "motor_vec": None
                if pend.get("motor_vec") is None
                else np.asarray(pend["motor_vec"], dtype=np.float64),
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
