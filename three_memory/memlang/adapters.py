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

    def scramble_rho(self, rho_post: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_scramble"):
            rng = np.random.default_rng(int(self.cfg.get("seed") or 0) + 90210)
            q, _ = np.linalg.qr(rng.normal(size=(self.n, self.n)))
            self._scramble = q
        return unit(self._scramble @ unit(rho_post))

    def compose_value(self, invariant: np.ndarray, rho_post: np.ndarray, mix: float) -> np.ndarray:
        """Stable organism direction plus unique rho so split-pin hashes differ."""
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        m = min(0.92, max(0.0, float(mix)))
        return unit(m * unit(invariant) + (1.0 - m) * unit(rho_post))


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
        inv = tgt if proto is None else proto
        return self.compose_value(inv, rho_post, self.mix)

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
        inv = unit((proto if proto is not None else tgt) + 0.15 * fast)
        return self.compose_value(inv, rho_post, self.mix)

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
            inv = tgt
        elif self.store == "mix":
            inv = unit(0.5 * pred + 0.5 * tgt)
        else:
            inv = pred
        return self.compose_value(inv, rho_post, float(self.cfg.get("mix") or 0.75))

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
        return self.compose_value(unit(mix * tgt + (1.0 - mix) * mapped), rho_post, 0.75)

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
        return self.compose_value(unit(mix * tgt + (1.0 - mix) * unit(recon)), rho_post, 0.75)

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
        inv = y if proto is None else unit(0.5 * y + 0.5 * proto)
        return self.compose_value(inv, rho_post, 0.75)

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
        proto = self.bank.nearest(tgt)
        inv = tgt if (self.pure or proto is None) else proto
        mix = 0.7 if self.pure else 0.85
        return self.compose_value(inv, rho_post, mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "pure": self.pure, "n_protos": len(self.bank.protos)}


class RhoClusterAdapter(ValueAdapter):
    """Cluster by organism dual; store slow means of rho in identity coordinates."""

    family = "rho_cluster"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "rho_cluster")
        self.eta = float(self.cfg.get("eta") or 0.1)
        self.mix = float(self.cfg.get("mix") or 0.6)
        self.duals = _ProtoBank(
            max_k=int(self.cfg.get("max_k") or 8),
            spawn=float(self.cfg.get("spawn") or 0.35),
            eta=self.eta,
            sep=float(self.cfg.get("sep") or 0.05),
        )
        self.rhos: list[np.ndarray] = []

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
        j = self.duals.update(tgt, gated)
        while len(self.rhos) < len(self.duals.protos):
            self.rhos.append(x.copy())
        if j >= 0 and gated > PROTO_EPS:
            m = min(1.0, self.eta * gated)
            self.rhos[j] = unit((1.0 - m) * self.rhos[j] + m * x)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        x = unit(rho_post)
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        tgt = self.teaching_target(rho_post)
        if not self.duals.protos or not self.rhos:
            return x
        sims = np.array([float(np.dot(p, unit(tgt))) for p in self.duals.protos], dtype=np.float64)
        j = min(int(np.argmax(sims)), len(self.rhos) - 1)
        return self.compose_value(self.rhos[j], rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "mix": self.mix, "n_protos": len(self.rhos)}


class ContrastiveRhoAdapter(ValueAdapter):
    """Pull rho images of the same dual together; push different duals apart."""

    family = "contrastive_rho"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "contrastive_rho")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.sep = float(self.cfg.get("sep") or 0.05)
        self.mix = float(self.cfg.get("mix") or 0.5)
        self.W = np.eye(self.n, dtype=np.float64)
        self.duals = _ProtoBank(max_k=8, spawn=0.35, eta=0.1, sep=self.sep)
        self.rhos: list[np.ndarray] = []

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
        y = self.W @ x
        j = self.duals.update(tgt, gated)
        while len(self.rhos) < len(self.duals.protos):
            self.rhos.append(x.copy())
        if j >= 0 and gated > PROTO_EPS:
            same = self.rhos[j]
            self.W = self.W + self.eta * gated * np.outer(unit(same) - unit(y), x)
            for i, other in enumerate(self.rhos):
                if i == j:
                    continue
                self.W = self.W - self.sep * gated * np.outer(unit(other), x)
            self.rhos[j] = unit((1.0 - self.eta) * self.rhos[j] + self.eta * x)
            np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        return self.compose_value(unit(self.W @ unit(rho_post)), rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "mix": self.mix, "n_protos": len(self.rhos)}


class ClusterSfaAdapter(ValueAdapter):
    """Slow-feature pull on rho within a dual cluster; variance across clusters."""

    family = "cluster_sfa"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "cluster_sfa")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.mix = float(self.cfg.get("mix") or 0.6)
        self.W = np.eye(self.n, dtype=np.float64)
        self.duals = _ProtoBank(max_k=8, spawn=0.35, eta=0.1, sep=0.08)
        self.prev: dict[int, np.ndarray] = {}

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
        j = self.duals.update(tgt, gated)
        y = unit(self.W @ x)
        if j >= 0 and gated > PROTO_EPS:
            prev = self.prev.get(j)
            if prev is not None:
                self.W = self.W - self.eta * gated * np.outer(y - prev, x)
            for k, pv in self.prev.items():
                if k == j:
                    continue
                self.W = self.W + 0.05 * gated * np.outer(y, pv)
            self.prev[j] = y.copy()
            np.clip(self.W, -2.0, 2.0, out=self.W)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        return self.compose_value(unit(self.W @ unit(rho_post)), rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "mix": self.mix}


class KMeansRhoAdapter(ValueAdapter):
    """Online k-means on write-time rho. k is machinery, not an action-index codebook."""

    family = "kmeans_rho"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "kmeans_rho")
        self.eta = float(self.cfg.get("eta") or 0.1)
        self.mix = float(self.cfg.get("mix") or 0.6)
        self.k = int(self.cfg.get("k") or 4)
        self.centers: list[np.ndarray] = []
        self.last_j = -1

    def _assign(self, x: np.ndarray) -> int:
        if not self.centers:
            self.centers.append(x.copy())
            return 0
        sims = np.array([float(np.dot(c, x)) for c in self.centers], dtype=np.float64)
        j = int(np.argmax(sims))
        spawn = float(self.cfg.get("spawn") or 0.85)
        if float(sims[j]) < spawn and len(self.centers) < self.k:
            self.centers.append(x.copy())
            return len(self.centers) - 1
        return j

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        gated = _gated(adv)
        if gated <= PROTO_EPS:
            return
        j = self._assign(x)
        self.last_j = j
        m = min(1.0, self.eta * gated)
        self.centers[j] = unit((1.0 - m) * self.centers[j] + m * x)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        x = unit(rho_post)
        if not self.centers:
            return x
        j = self.last_j if 0 <= self.last_j < len(self.centers) else int(np.argmax([float(np.dot(c, x)) for c in self.centers]))
        return self.compose_value(self.centers[j], rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "mix": self.mix, "n_protos": len(self.centers)}


class SepClusterAdapter(ValueAdapter):
    """Dual clustering with a high spawn threshold so distinct motors do not share a rho mean."""

    family = "sep_cluster"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "sep_cluster")
        self.eta = float(self.cfg.get("eta") or 0.1)
        self.mix = float(self.cfg.get("mix") or 0.6)
        self.duals = _ProtoBank(
            max_k=int(self.cfg.get("max_k") or 4),
            spawn=float(self.cfg.get("spawn") or 0.9),
            eta=self.eta,
            sep=float(self.cfg.get("sep") or 0.1),
        )
        self.rhos: list[np.ndarray] = []
        self.last_j = -1

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        tgt = self.teaching_target(rho_post)
        gated = _gated(adv)
        j = self.duals.update(tgt, gated)
        self.last_j = j
        while len(self.rhos) < len(self.duals.protos):
            self.rhos.append(x.copy())
        if j >= 0 and gated > PROTO_EPS:
            m = min(1.0, self.eta * gated)
            self.rhos[j] = unit((1.0 - m) * self.rhos[j] + m * x)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        if not self.rhos:
            return unit(rho_post)
        j = self.last_j if 0 <= self.last_j < len(self.rhos) else 0
        return self.compose_value(self.rhos[j], rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "eta": self.eta,
            "mix": self.mix,
            "n_protos": len(self.rhos),
            "n_duals": len(self.duals.protos),
        }


class MotorClusterAdapter(ValueAdapter):
    """Cluster by observed motor vectors, which remain separated when duals collapse."""

    family = "motor_cluster"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "motor_cluster")
        self.eta = float(self.cfg.get("eta") or 0.2)
        self.mix = float(self.cfg.get("mix") or 0.5)
        self.spawn = float(self.cfg.get("spawn") or 0.5)
        self.max_k = int(self.cfg.get("max_k") or 8)
        self.motors: list[np.ndarray] = []
        self.rhos: list[np.ndarray] = []
        self.last_j = -1

    def _assign_motor(self, motor: np.ndarray) -> int:
        m = unit(motor)
        if not self.motors:
            self.motors.append(m.copy())
            return 0
        sims = np.array([float(np.dot(p, m)) for p in self.motors], dtype=np.float64)
        j = int(np.argmax(sims))
        if float(sims[j]) < self.spawn and len(self.motors) < self.max_k:
            self.motors.append(m.copy())
            return len(self.motors) - 1
        return j

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        gated = _gated(adv)
        if gated <= PROTO_EPS:
            return
        if self.last_motor is None or int(self.last_motor.size) == 0:
            return
        j = self._assign_motor(self.last_motor)
        self.last_j = j
        while len(self.rhos) < len(self.motors):
            self.rhos.append(x.copy())
        mix = min(1.0, self.eta * gated)
        self.rhos[j] = unit((1.0 - mix) * self.rhos[j] + mix * x)
        self.motors[j] = unit((1.0 - 0.2) * self.motors[j] + 0.2 * unit(self.last_motor))

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None or self.last_motor is None:
            return self.scramble_rho(rho_post)
        if not self.rhos:
            return unit(rho_post)
        j = self.last_j if 0 <= self.last_j < len(self.rhos) else 0
        return self.compose_value(self.rhos[j], rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "eta": self.eta, "mix": self.mix, "n_protos": len(self.rhos)}


class WhiteningAdapter(ValueAdapter):
    """Online ZCA whitening of rho so a linear decoder sees stationary second-order stats."""

    family = "whitening"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "whitening")
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.eps = float(self.cfg.get("eps") or 1e-3)
        self.cov = np.eye(self.n, dtype=np.float64)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        gated = _gated(adv)
        if gated <= PROTO_EPS:
            return
        self.cov = (1.0 - self.eta * gated) * self.cov + (self.eta * gated) * np.outer(x, x)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        x = unit(rho_post)
        w, v = np.linalg.eigh(0.5 * (self.cov + self.cov.T) + self.eps * np.eye(self.n))
        w = np.clip(w, self.eps, None)
        zca = (v * (w ** -0.5)) @ v.T
        return unit(zca @ x)

    def geometry(self) -> dict[str, Any]:
        s = np.linalg.svd(self.cov, compute_uv=False)
        return {"family": self.family, "name": self.name, "eta": self.eta, "rank": int((s > 1e-8).sum())}


class StickySepAdapter(SepClusterAdapter):
    """Sticky dual-index rho means, no repulsion, spawn high enough for four motors."""

    family = "sticky_sep"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        cfg.setdefault("sep", 0.0)
        cfg.setdefault("spawn", 0.99)
        cfg.setdefault("max_k", 4)
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "sticky_sep")
        self.family = "sticky_sep"



class TanhRhoAdapter(ValueAdapter):
    family = "tanh_rho"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "tanh_rho")
        self.scale = float(self.cfg.get("scale") or 1.0)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        return unit(np.tanh(self.scale * unit(rho_post)))

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "scale": self.scale}


class DelayMixAdapter(ValueAdapter):
    family = "delay_mix"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "delay_mix")
        self.mix = float(self.cfg.get("mix") or 0.3)
        self.prev: np.ndarray | None = None

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        if _gated(adv) > PROTO_EPS:
            self.prev = x.copy() if self.prev is None else unit((1.0 - self.mix) * self.prev + self.mix * x)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        x = unit(rho_post)
        if self.prev is None:
            return x
        return self.compose_value(self.prev, rho_post, self.mix)

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "mix": self.mix}


class WhitenNudgeAdapter(WhiteningAdapter):
    """ZCA of rho plus a small motor-cluster mean."""

    family = "whiten_nudge"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "whiten_nudge")
        self.nudge = float(self.cfg.get("nudge") or 0.2)
        self.motor = MotorClusterAdapter(n, {**self.cfg, "mix": self.nudge, "eta": 0.25, "spawn": 0.5})

    def observe_motor(self, motor: np.ndarray | None, adv: float, *, efference: np.ndarray | None = None) -> None:
        super().observe_motor(motor, adv, efference=efference)
        self.motor.observe_motor(motor, adv, efference=efference)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        super().update(rho_post, adv)
        self.motor.update(rho_post, adv)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        z = super().value(rho_post)
        if self.last_target is None:
            return z
        nudged = self.motor.value(rho_post)
        return unit((1.0 - self.nudge) * z + self.nudge * nudged)

    def geometry(self) -> dict[str, Any]:
        g = super().geometry()
        g.update({"nudge": self.nudge, "n_motor_protos": (self.motor.geometry() or {}).get("n_protos")})
        return g


class MutantAdapter(ValueAdapter):
    """Random legal combination: tanh, optional ZCA, optional delay, optional tiny motor mix."""

    family = "mutant"

    def __init__(self, n: int, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(n, cfg)
        self.name = str(self.cfg.get("name") or "mutant")
        self.scale = float(self.cfg.get("scale") or 1.0)
        self.zca = bool(self.cfg.get("zca", False))
        self.delay = float(self.cfg.get("delay") or 0.0)
        self.nudge = float(self.cfg.get("nudge") or 0.0)
        self.eta = float(self.cfg.get("eta") or 0.05)
        self.eps = 1e-3
        self.cov = np.eye(self.n, dtype=np.float64)
        self.prev: np.ndarray | None = None
        self.motor = MotorClusterAdapter(n, {"mix": max(0.05, self.nudge), "eta": 0.25, "spawn": 0.5, "seed": int(self.cfg.get("seed") or 0)})

    def observe_motor(self, motor: np.ndarray | None, adv: float, *, efference: np.ndarray | None = None) -> None:
        super().observe_motor(motor, adv, efference=efference)
        if self.nudge > 0:
            self.motor.observe_motor(motor, adv, efference=efference)

    def update(self, rho_post: np.ndarray, adv: float) -> None:
        x = unit(rho_post)
        gated = _gated(adv)
        if gated > PROTO_EPS:
            if self.zca:
                self.cov = (1.0 - self.eta * gated) * self.cov + (self.eta * gated) * np.outer(x, x)
            if self.delay > 0:
                self.prev = x.copy() if self.prev is None else unit((1.0 - self.delay) * self.prev + self.delay * x)
            if self.nudge > 0:
                self.motor.update(rho_post, adv)

    def value(self, rho_post: np.ndarray) -> np.ndarray:
        if self.last_target is None:
            return self.scramble_rho(rho_post)
        x = unit(np.tanh(self.scale * unit(rho_post)))
        if self.zca:
            w, v = np.linalg.eigh(0.5 * (self.cov + self.cov.T) + self.eps * np.eye(self.n))
            w = np.clip(w, self.eps, None)
            x = unit(((v * (w ** -0.5)) @ v.T) @ x)
        if self.delay > 0 and self.prev is not None:
            x = unit((1.0 - self.delay) * x + self.delay * self.prev)
        if self.nudge > 0:
            x = unit((1.0 - self.nudge) * x + self.nudge * self.motor.value(rho_post))
        return x

    def geometry(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "scale": self.scale, "zca": self.zca, "delay": self.delay, "nudge": self.nudge}


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
    "rho_cluster": RhoClusterAdapter,
    "contrastive_rho": ContrastiveRhoAdapter,
    "cluster_sfa": ClusterSfaAdapter,
    "kmeans_rho": KMeansRhoAdapter,
    "sep_cluster": SepClusterAdapter,
    "sticky_sep": StickySepAdapter,
    "motor_cluster": MotorClusterAdapter,
    "whitening": WhiteningAdapter,
    "tanh_rho": TanhRhoAdapter,
    "delay_mix": DelayMixAdapter,
    "whiten_nudge": WhitenNudgeAdapter,
    "mutant": MutantAdapter,
}


def make_adapter(family: str, n: int, cfg: dict[str, Any] | None = None) -> ValueAdapter:
    if family == "identity" or not family:
        return ValueAdapter(n, cfg)
    cls = FAMILIES.get(str(family))
    if cls is None:
        raise RuntimeError(f"unknown family {family}")
    return cls(n, cfg)
