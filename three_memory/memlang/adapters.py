"""MEMLANG-1 value adapters. Runtime bind only. No runner motor pad."""

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


def _gated(adv: float) -> float:
    a = float(adv)
    return a if abs(a) > PROTO_EPS else 0.0


class ValueAdapter:
    family = "identity"
    name = "identity"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        self.n = int(n)
        self.cfg = dict(cfg or {})
        self.last_motor: np.ndarray | None = None
        self.last_target: np.ndarray | None = None

    def observe_motor(self, motor: np.ndarray | None, adv: float, *, efference: np.ndarray | None = None) -> None:
        _ = adv
        if motor is not None:
            self.last_motor = np.asarray(motor, dtype=np.float64).reshape(-1).copy()
        if efference is None:
            return
        e = np.asarray(efference, dtype=np.float64).reshape(-1)
        if e.size != self.n:
            return
        self.last_target = unit(e)

    def teaching_target(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is not None and int(self.last_target.size) == self.n:
            return np.asarray(self.last_target, dtype=np.float64)
        return unit(rho_post)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        _ = (rho_post, adv)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        return unit(rho_post)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name}

    def blank_if_no_efference(self, vec: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return np.zeros(self.n, dtype=np.float64)
        return unit(vec)


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
        np.clip(self.T, -2.0, 2.0, out=self.T)

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
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
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
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
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
        gated = _gated(adv)
        self.W = self.W + self.eta * gated * np.outer(x, h)
        np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        x = unit(rho_post)
        h = unit((1.0 - self.leak) * self.h + self.leak * x)
        return unit(self.W @ h)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "leak": self.leak}


class _ProtoBank:
    def __init__(self, max_k: int, spawn: float, eta: float, sep: float) -> None:
        self.max_k = int(max_k)
        self.spawn = float(spawn)
        self.eta = float(eta)
        self.sep = float(sep)
        self.protos: list[np.ndarray] = []
        self.counts: list[int] = []

    def assign(self, tgt: np.ndarray) -> int:
        t = unit(tgt)
        if not self.protos:
            self.protos.append(t.copy())
            self.counts.append(1)
            return 0
        sims = np.array([float(np.dot(p, t)) for p in self.protos], dtype=np.float64)
        j = int(np.argmax(sims))
        if float(sims[j]) < self.spawn and len(self.protos) < self.max_k:
            self.protos.append(t.copy())
            self.counts.append(1)
            return len(self.protos) - 1
        return j

    def update(self, tgt: np.ndarray, gated: float) -> int:
        if gated <= PROTO_EPS:
            if not self.protos:
                return -1
            sims = np.array([float(np.dot(p, unit(tgt))) for p in self.protos], dtype=np.float64)
            return int(np.argmax(sims))
        j = self.assign(tgt)
        t = unit(tgt)
        mix = min(1.0, self.eta * gated)
        self.protos[j] = unit((1.0 - mix) * self.protos[j] + mix * t)
        self.counts[j] += 1
        for i, p in enumerate(self.protos):
            if i == j:
                continue
            sim = float(np.dot(p, self.protos[j]))
            if sim > 0.0:
                self.protos[i] = unit(p - self.sep * sim * self.protos[j])
        return j

    def nearest(self, tgt: np.ndarray) -> np.ndarray | None:
        if not self.protos:
            return None
        t = unit(tgt)
        sims = np.array([float(np.dot(p, t)) for p in self.protos], dtype=np.float64)
        return self.protos[int(np.argmax(sims))]


class FeedbackInvarianceAdapter(ValueAdapter):
    """Same observed dual clusters together; different duals repel. Not an action-index codebook."""

    family = "feedback_invariance"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "feedback_invariance")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.sep = float(self.cfg.get("sep") or 0.05)
        self.mix = float(self.cfg.get("mix") or 0.85)
        self.bank = _ProtoBank(
            max_k=int(self.cfg.get("max_k") or 8),
            spawn=float(self.cfg.get("spawn") or 0.35),
            eta=self.eta,
            sep=self.sep,
        )

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        self.bank.update(self.teaching_target(rho_post), _gated(adv))

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        tgt = self.teaching_target(rho_post)
        proto = self.bank.nearest(tgt)
        if proto is None:
            return self.blank_if_no_efference(tgt)
        m = min(1.0, max(0.0, self.mix))
        return self.blank_if_no_efference(m * proto + (1.0 - m) * tgt)

    def geometry(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "eta": self.eta,
            "sep": self.sep,
            "mix": self.mix,
            "n_protos": len(self.bank.protos),
            "counts": list(self.bank.counts),
        }


class DualTimescaleAdapter(ValueAdapter):
    """Fast residual map from rho; slow proto bank of organism duals (BYOL-like target)."""

    family = "dual_timescale"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "dual_timescale")
        self.eta_fast = float(self.cfg.get("eta_fast") or 0.1)
        self.eta_slow = float(self.cfg.get("eta_slow") or 0.01)
        self.mix = float(self.cfg.get("mix") or 0.7)
        self.W = np.eye(self.n, dtype=np.float64)
        self.bank = _ProtoBank(max_k=8, spawn=0.35, eta=self.eta_slow, sep=0.05)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
        pred = self.W @ x
        self.W = self.W + self.eta_fast * gated * np.outer(tgt - pred, x)
        np.clip(self.W, -2.0, 2.0, out=self.W)
        self.bank.update(tgt, gated)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        proto = self.bank.nearest(tgt)
        fast = unit(self.W @ x)
        if proto is None:
            return self.blank_if_no_efference(self.mix * tgt + (1.0 - self.mix) * fast)
        return self.blank_if_no_efference(self.mix * proto + (1.0 - self.mix) * fast)

    def geometry(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "eta_fast": self.eta_fast,
            "eta_slow": self.eta_slow,
            "n_protos": len(self.bank.protos),
        }


class PredictionErrorAdapter(ValueAdapter):
    family = "prediction_error"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "prediction_error")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.store = str(self.cfg.get("store") or "pred")
        self.W = np.eye(self.n, dtype=np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        pred = unit(self.W @ x)
        err = tgt - pred
        gated = _gated(adv)
        self.W = self.W + self.eta * gated * np.outer(err, x)
        np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        x = unit(rho_post)
        pred = unit(self.W @ x)
        tgt = self.teaching_target(rho_post)
        if self.store == "target":
            return self.blank_if_no_efference(tgt)
        if self.store == "mix":
            return self.blank_if_no_efference(0.5 * pred + 0.5 * tgt)
        return self.blank_if_no_efference(pred)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "store": self.store}


class EvolvedPlasticityAdapter(ValueAdapter):
    """Hebbian / anti-Hebbian / Oja / BCM coefficients. Machinery only; never S."""

    family = "evolved_plasticity"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "evolved_plasticity")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.hebb = float(self.cfg.get("hebb") or 1.0)
        self.anti = float(self.cfg.get("anti") or 0.0)
        self.oja = float(self.cfg.get("oja") or 0.0)
        self.bcm = float(self.cfg.get("bcm") or 0.0)
        self.norm = float(self.cfg.get("norm") or 1.0)
        self.theta = 1.0
        self.W = np.eye(self.n, dtype=np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        y = self.W @ x
        gated = _gated(adv)
        hebb = self.hebb * np.outer(tgt, x)
        anti = self.anti * np.outer(x, x)
        oja = self.oja * np.outer(y, y) @ self.W
        self.theta = 0.95 * self.theta + 0.05 * float(np.dot(y, y))
        bcm = self.bcm * float(np.dot(y, y) - self.theta) * np.outer(y, x)
        self.W = self.W + self.eta * gated * (hebb - anti - oja + bcm)
        if self.norm > 0:
            s = float(np.linalg.norm(self.W))
            cap = self.norm * float(self.n)
            if s > cap:
                self.W *= cap / s
        np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        tgt = self.teaching_target(rho_post)
        mapped = unit(self.W @ unit(rho_post))
        mix = float(self.cfg.get("mix") or 0.5)
        return self.blank_if_no_efference(mix * tgt + (1.0 - mix) * mapped)

    def geometry(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "eta": self.eta,
            "hebb": self.hebb,
            "anti": self.anti,
            "oja": self.oja,
            "bcm": self.bcm,
        }


class LatentManifoldAdapter(ValueAdapter):
    family = "latent_manifold"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "latent_manifold")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.k = int(self.cfg.get("k") or 4)
        rng = np.random.default_rng(int(self.cfg.get("seed") or 0) + 91)
        k = max(1, min(self.k, self.n))
        self.E = (0.1 * rng.normal(size=(k, self.n))).astype(np.float64)
        self.D = (0.1 * rng.normal(size=(self.n, k))).astype(np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        z = self.E @ x
        recon = self.D @ z
        gated = _gated(adv)
        err = tgt - recon
        self.D = self.D + self.eta * gated * np.outer(err, z)
        self.E = self.E + self.eta * gated * np.outer(self.D.T @ err, x)
        np.clip(self.D, -2.0, 2.0, out=self.D)
        np.clip(self.E, -2.0, 2.0, out=self.E)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        tgt = self.teaching_target(rho_post)
        x = unit(rho_post)
        recon = self.D @ (self.E @ x)
        mix = float(self.cfg.get("mix") or 0.5)
        return self.blank_if_no_efference(mix * tgt + (1.0 - mix) * unit(recon))

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "k": self.k}


class SlowFeatureAdapter(ValueAdapter):
    """Wiskott–Sejnowski slowness on organism duals, with between-cluster variance."""

    family = "slow_feature"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "slow_feature")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.sep = float(self.cfg.get("sep") or 0.1)
        self.W = np.eye(self.n, dtype=np.float64)
        self.prev_tgt: np.ndarray | None = None
        self.prev_y: np.ndarray | None = None
        self.bank = _ProtoBank(max_k=8, spawn=0.35, eta=self.eta, sep=self.sep)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
        y = unit(self.W @ tgt)
        self.bank.update(tgt, gated)
        if gated > PROTO_EPS and self.prev_tgt is not None and self.prev_y is not None:
            sim = float(np.dot(unit(self.prev_tgt), tgt))
            if sim > 0.5:
                dy = y - self.prev_y
                self.W = self.W - self.eta * gated * np.outer(dy, tgt)
            else:
                self.W = self.W + self.sep * gated * np.outer(y, tgt)
            np.clip(self.W, -2.0, 2.0, out=self.W)
        self.prev_tgt = tgt.copy()
        self.prev_y = y.copy()

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        tgt = self.teaching_target(rho_post)
        y = unit(self.W @ tgt)
        proto = self.bank.nearest(tgt)
        if proto is None:
            return self.blank_if_no_efference(y)
        return self.blank_if_no_efference(0.5 * y + 0.5 * proto)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "n_protos": len(self.bank.protos)}


class EfferenceCopyAdapter(ValueAdapter):
    """Store the organism dual (or a slow EMA of it). Baseline invariance hypothesis."""

    family = "efference_copy"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "efference_copy")
        self.eta = float(self.cfg.get("eta") or 1.0)
        self.bank = _ProtoBank(max_k=int(self.cfg.get("max_k") or 8), spawn=float(self.cfg.get("spawn") or 0.35), eta=self.eta, sep=float(self.cfg.get("sep") or 0.05))
        self.pure = bool(self.cfg.get("pure", True))

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        self.bank.update(self.teaching_target(rho_post), _gated(adv))

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        tgt = self.teaching_target(rho_post)
        if self.pure or not self.bank.protos:
            return self.blank_if_no_efference(tgt)
        proto = self.bank.nearest(tgt)
        return self.blank_if_no_efference(tgt if proto is None else proto)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "pure": self.pure, "n_protos": len(self.bank.protos)}


FAMILIES = {
    "slow_target": SlowTargetAdapter,
    "hebbian_delta": HebbianDeltaAdapter,
    "lowrank_adapter": LowRankAdapter,
    "recurrent_consistency": RecurrentConsistencyAdapter,
    "feedback_invariance": FeedbackInvarianceAdapter,
    "dual_timescale": DualTimescaleAdapter,
    "prediction_error": PredictionErrorAdapter,
    "evolved_plasticity": EvolvedPlasticityAdapter,
    "latent_manifold": LatentManifoldAdapter,
    "slow_feature": SlowFeatureAdapter,
    "efference_copy": EfferenceCopyAdapter,
}


def make_adapter(family: str, n: int, cfg: dict[str, Any] | None = None) -> ValueAdapter:
    if family == "identity" or not family:
        return ValueAdapter(n, cfg)
    cls = FAMILIES.get(str(family))
    if cls is None:
        raise RuntimeError(f"unknown family {family}")
    return cls(n, cfg)
