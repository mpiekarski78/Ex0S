"""MEMLANG-1 value adapters. Runtime bind only. Do not edit neural_cortex.py."""

from __future__ import annotations

from typing import Any

import numpy as np

PROTO_EPS = 1e-12


def unit(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm <= PROTO_EPS:
        return np.zeros_like(v)
    return (v / nrm).astype(np.float64)


class ValueAdapter:
    family = "identity"
    name = "identity"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        self.n = int(n)
        self.cfg = dict(cfg or {})
        self.last_motor: np.ndarray | None = None

    def observe_motor(self, motor: np.ndarray | None, adv: float) -> None:
        _ = adv
        if motor is None:
            return
        self.last_motor = unit(motor)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        _ = (rho_post, adv)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        return unit(rho_post)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name}


class SlowTargetAdapter(ValueAdapter):
    family = "slow_target"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "slow_target")
        eta = float(self.cfg.get("eta") or 0.05)
        self.eta = eta
        self.T = np.eye(self.n, dtype=np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        mix = self.eta * (1.0 if abs(float(adv)) > PROTO_EPS else 0.25)
        self.T = (1.0 - mix) * self.T + mix * np.outer(x, x)
        c = 2.0
        np.clip(self.T, -c, c, out=self.T)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        return unit(self.T @ unit(rho_post))

    def geometry(self) -> dict[str, Any]:
        s = np.linalg.svd(self.T, compute_uv=False)
        return {"family": self.family, "name": self.name, "eta": self.eta, "rank_T": int((s > 1e-8).sum())}


class HebbianDeltaAdapter(ValueAdapter):
    family = "hebbian_delta"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "hebbian_delta")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.W = np.eye(self.n, dtype=np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.last_motor if self.last_motor is not None else x
        gated = float(adv) if abs(float(adv)) > PROTO_EPS else 0.0
        self.W = self.W + self.eta * gated * np.outer(tgt, x)
        np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        return unit(self.W @ unit(rho_post))

    def geometry(self) -> dict[str, Any]:
        s = np.linalg.svd(self.W, compute_uv=False)
        return {"family": self.family, "name": self.name, "eta": self.eta, "rank_W": int((s > 1e-8).sum())}


class LowRankAdapter(ValueAdapter):
    family = "lowrank_adapter"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "lowrank_adapter")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.rank = int(self.cfg.get("rank") or 4)
        rng = np.random.default_rng(int(self.cfg.get("seed") or 0) + 17)
        r = max(1, min(self.rank, self.n))
        self.U = (0.1 * rng.normal(size=(self.n, r))).astype(np.float64)
        self.V = (0.1 * rng.normal(size=(r, self.n))).astype(np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.last_motor if self.last_motor is not None else x
        gated = float(adv) if abs(float(adv)) > PROTO_EPS else 0.0
        err = tgt - (self.U @ (self.V @ x))
        self.U = self.U + self.eta * gated * np.outer(err, self.V @ x)
        self.V = self.V + self.eta * gated * np.outer(self.U.T @ err, x)
        np.clip(self.U, -2.0, 2.0, out=self.U)
        np.clip(self.V, -2.0, 2.0, out=self.V)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        x = unit(rho_post)
        return unit(x + self.U @ (self.V @ x))

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "rank": self.rank}


class RecurrentConsistencyAdapter(ValueAdapter):
    family = "recurrent_consistency"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "recurrent_consistency")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.leak = float(self.cfg.get("leak") or 0.3)
        self.W = np.eye(self.n, dtype=np.float64)
        self.h = np.zeros(self.n, dtype=np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        self.h = (1.0 - self.leak) * self.h + self.leak * x
        h = unit(self.h)
        gated = float(adv) if abs(float(adv)) > PROTO_EPS else 0.0
        self.W = self.W + self.eta * gated * np.outer(x, h)
        np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        x = unit(rho_post)
        h = unit((1.0 - self.leak) * self.h + self.leak * x)
        return unit(self.W @ h)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "leak": self.leak}


FAMILIES = {
    "slow_target": SlowTargetAdapter,
    "hebbian_delta": HebbianDeltaAdapter,
    "lowrank_adapter": LowRankAdapter,
    "recurrent_consistency": RecurrentConsistencyAdapter,
}


def make_adapter(family: str, n: int, cfg: dict[str, Any] | None = None) -> ValueAdapter:
    if family == "identity" or not family:
        return ValueAdapter(n, cfg)
    cls = FAMILIES.get(str(family))
    if cls is None:
        raise RuntimeError(f"unknown family {family}")
    return cls(n, cfg)
