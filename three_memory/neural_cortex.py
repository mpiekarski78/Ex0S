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
# v2 amendment: dedicated ACT motor lexicon (not grown from sensory symbols)
# v3 amendment: ACT targets restricted to press/harm only
MOTOR_ACT_TOKENS = ("press", "harm")
OP_COST = {
    "RETRIEVE": 1.0,
    "WRITE": 1.0,
    "EMIT": 1.0,
    "ACT": 0.05,  # v3 amendment (was 0.1 in v2; 1.0 in v1)
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
    dtype: str = "float64"

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

        self.rng_birth = np.random.default_rng(g.seed_birth)
        self.rng_registry = np.random.default_rng(g.seed_registry)
        self.rng_source = np.random.default_rng(g.seed_source)
        self.rng_action = np.random.default_rng(g.seed_action)
        self.rng_permute = np.random.default_rng(g.seed_permute)

        self.vocab: dict[str, np.ndarray] = {}
        self.sources: dict[str, np.ndarray] = {}
        # v2: birth-time motor lexicon M_act (diagnosis motor_lexicon_absent)
        self.motor_vocab: dict[str, np.ndarray] = {}
        for tok in MOTOR_ACT_TOKENS:
            vec = self.rng_registry.normal(0.0, 1.0, size=g.d_sym).astype(np.float64)
            self.motor_vocab[tok] = vec
            self.vocab[tok] = vec

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
        self.W_act_query = self._randn(g.d_sym, g.n, g.n)
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
        self.emit_buffer: list[str] = []
        self.last_body = BODY_SETPOINT.copy()
        self.last_s = np.zeros(g.d_sym, dtype=np.float64)
        self.last_s_hat = np.zeros(g.d_sym, dtype=np.float64)
        self.prev_interaction: str | None = None
        self._pending: dict[str, Any] | None = None
        self._pending_writes: list[CortexRecord] = []
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
    ) -> str | None:
        pool = lexicon if lexicon is not None else self.vocab
        if not pool:
            return None
        q = self._from_t(query)
        qn = np.linalg.norm(q) + 1e-12
        best_tok = None
        best = -1.0
        for tok, v in pool.items():
            cos = float(np.dot(q, v) / (qn * (np.linalg.norm(v) + 1e-12)))
            if cos > best:
                best = cos
                best_tok = tok
        if require_thresh and best < self.genome.cos_thresh:
            return None
        return best_tok

    def _do_retrieve(self) -> None:
        q = self.W_att @ self.rho
        qn = self._from_t(q)
        qnorm = np.linalg.norm(qn) + 1e-12
        scored: list[tuple[float, str, np.ndarray]] = []
        for rec in self.memory.records():
            v = np.asarray(rec.content, dtype=np.float64)
            if v.shape[0] != self.genome.d_sym:
                continue
            cos = float(np.dot(qn, v) / (qnorm * (np.linalg.norm(v) + 1e-12)))
            scored.append((cos, rec.fact_id, v))
        scored.sort(key=lambda t: (-t[0], t[1]))
        buf = np.zeros((self.genome.k_s, self.genome.d_sym), dtype=np.float64)
        for i, (_c, _fid, v) in enumerate(scored[: self.genome.k_s]):
            buf[i] = v
        # populate for NEXT tick — store pending apply
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

        for _k in range(g.t_max):
            self._commit_pending_retrieve()
            # internal tick sensory: zeros inject except buffer/body/same_ix
            zero = np.zeros(g.d_sym, dtype=np.float64)
            self._sensory_tick(zero, body, same_ix, record_sensory=False)
            logits = (self.W_op @ self.rho) + self.b_op
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
                tok = self._best_token(self.W_emit_query @ self.rho)
                if tok is None:
                    chosen_op = "HOLD"
                    self.emit_buffer = []
                    break
                self.emit_buffer.append(tok)
                chosen_token = tok
                continue
            if op == "ACT":
                # v2: argmax over M_act only; never force HOLD on cosine miss
                tok = self._best_token(
                    self.W_act_query @ self.rho,
                    lexicon=self.motor_vocab,
                    require_thresh=False,
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
                )
                staged_writes.append(rec)
                continue

        # commit writes after tick
        for rec in staged_writes:
            self.memory.write(rec)

        s_hat = self._from_t(self.W_pred @ self.rho)
        out = {
            "op": chosen_op,
            "token": chosen_token,
            "emit_sequence": list(self.emit_buffer),
            "rho_elig": rho_elig,
            "s_hat": s_hat,
            "body": body.copy(),
            "cost": float(OP_COST[chosen_op]),
            "retrieved": applied_retrieve,
        }
        self.last_action = out
        self.last_s_hat = s_hat
        return out

    def _apply_credit(self, s_t: np.ndarray, body_t: np.ndarray) -> dict[str, float]:
        if self._pending is None:
            return {"adv": 0.0, "pred_err": 0.0}
        p = self._pending
        eps = s_t - p["s_hat"]
        body_prev = p["body"]
        adv = float(
            np.linalg.norm(body_prev - BODY_SETPOINT)
            - np.linalg.norm(body_t - BODY_SETPOINT)
            - p["cost"]
        )
        rho_elig = self._to_t(p["rho_elig"])
        # directed prediction
        self.W_pred = self.W_pred + self.genome.eta_pred * torch.outer(
            self._to_t(eps), rho_elig
        )
        # three-factor on op / motor query used
        e_op = torch.zeros(len(OPS), dtype=self.dtype, device=self.device)
        e_op[OPS.index(p["op"])] = 1.0
        self.W_op = self.W_op + self.genome.eta_act * adv * torch.outer(e_op, rho_elig)
        # v4: b_op frozen (non-plastic)
        if p["token"] is not None and p["op"] in ("EMIT", "ACT"):
            tok_v = self._to_t(self._vocab_vec(p["token"]))
            mat_name = "W_emit_query" if p["op"] == "EMIT" else "W_act_query"
            W = getattr(self, mat_name)
            setattr(
                self,
                mat_name,
                W + self.genome.eta_act * adv * torch.outer(tok_v, rho_elig),
            )
        self._clip_and_consolidate()
        self._pending = None
        return {"adv": adv, "pred_err": float(np.linalg.norm(eps))}

    def _clip_and_consolidate(self) -> None:
        c = self.genome.clip
        beta = self.genome.beta
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

        action = self._motor_loop(body, same_ix)
        # store eligibility for next observe
        self._pending = {
            "op": action["op"],
            "token": action["token"],
            "rho_elig": action["rho_elig"],
            "s_hat": action["s_hat"],
            "body": body.copy(),
            "cost": action["cost"],
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
                **{k: v for k, v in self._pending.items() if k not in ("rho_elig", "s_hat", "body")},
                "rho_elig": np.asarray(self._pending["rho_elig"]).tolist(),
                "s_hat": np.asarray(self._pending["s_hat"]).tolist(),
                "body": np.asarray(self._pending["body"]).tolist(),
            },
            "vocab": {k: v.tolist() for k, v in self.vocab.items()},
            "motor_vocab": {k: v.tolist() for k, v in self.motor_vocab.items()},
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
            },
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
            # migrate: recover M_act keys from vocab if present
            self.motor_vocab = {
                k: self.vocab[k].copy()
                for k in MOTOR_ACT_TOKENS
                if k in self.vocab
            }
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
            }
        self.memory.restore(snap.get("S") or [])
        g = self.genome
        rs = snap.get("rng") or {}
        self.rng_birth = _restore_rng(g.seed_birth, rs.get("birth"))
        self.rng_registry = _restore_rng(g.seed_registry, rs.get("registry"))
        self.rng_source = _restore_rng(g.seed_source, rs.get("source"))
        self.rng_action = _restore_rng(g.seed_action, rs.get("action"))
        self.rng_permute = _restore_rng(g.seed_permute, rs.get("permute"))

    def weight_hash(self) -> str:
        h = hashlib.sha256()
        for name in self._plastic_names:
            h.update(self._from_t(getattr(self, name)).tobytes())
        return h.hexdigest()
