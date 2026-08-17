"""TM.0.24.LINEAGE — genome codec, refuse audit, antithetic ES, F_search.

No Python-source mutation. No capability scoring here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from three_memory.neural_cortex import GenomeConfig, NeuralCortex, OPS

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PATH = REPO_ROOT / "docs" / "lineage_genome_layout.json"

TENSOR_SHAPES = {
    "W_rec": None,  # filled from layout
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_layout() -> dict[str, Any]:
    return json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))


def layout_sha() -> str:
    return sha_file(LAYOUT_PATH)


def _slice_map(arm: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(s["name"]): s for s in arm["slices"]}


def defaults_theta(arm: str, layout: dict[str, Any] | None = None) -> np.ndarray:
    layout = layout or load_layout()
    spec = layout["arms"][arm]
    theta = np.zeros(int(spec["dim"]), dtype=np.float64)
    for sl in spec["slices"]:
        if sl["kind"] == "scalar":
            theta[int(sl["offset"])] = float(sl["default"])
        # tensors stay 0 until packed from a live cortex
    return theta


def decode_scalars(theta: np.ndarray, arm: str, layout: dict[str, Any] | None = None) -> dict[str, float]:
    layout = layout or load_layout()
    out: dict[str, float] = {}
    for sl in layout["arms"][arm]["slices"]:
        if sl["kind"] != "scalar":
            continue
        v = float(theta[int(sl["offset"])])
        v = float(np.clip(v, float(sl["lo"]), float(sl["hi"])))
        out[str(sl["name"])] = v
    return out


def refuse_audit(theta: np.ndarray, arm: str, layout: dict[str, Any] | None = None) -> dict[str, Any]:
    layout = layout or load_layout()
    spec = layout["arms"][arm]
    issues: list[str] = []
    if theta.shape != (int(spec["dim"]),):
        issues.append(f"dim {theta.shape} != {spec['dim']}")
    if not np.isfinite(theta).all():
        issues.append("non_finite")
    raw = theta.tobytes()
    # UTF-8 token strings would be long ASCII runs; refuse if theta aliases as text
    try:
        as_txt = raw.decode("utf-8", errors="ignore")
    except Exception:
        as_txt = ""
    for needle in ("the ", "answer", "stage", "hello", "phrase"):
        if needle in as_txt.lower():
            issues.append(f"utf8_needle:{needle}")
    scalars = decode_scalars(theta, arm, layout) if not issues else {}
    for sl in spec["slices"]:
        if sl["kind"] != "scalar":
            continue
        name = str(sl["name"])
        if name in scalars:
            if scalars[name] < float(sl["lo"]) - 1e-12 or scalars[name] > float(sl["hi"]) + 1e-12:
                issues.append(f"bound:{name}")
    return {"ok": not issues, "issues": issues, "arm": arm, "dim": int(spec["dim"])}


def scalars_to_lineage_params(scalars: dict[str, float]) -> dict[str, Any]:
    p: dict[str, Any] = {}
    costs = {}
    sp = [0.0, 0.0, 0.0, 0.0]
    for k, v in scalars.items():
        if k.startswith("dyn.op_cost."):
            costs[k.split(".")[-1]] = float(v)
            p[f"op_cost.{k.split('.')[-1]}"] = float(v)
        elif k.startswith("dyn.body_setpoint."):
            sp[int(k.rsplit(".", 1)[-1])] = float(v)
        elif k.startswith("dyn."):
            p[k[4:]] = float(v)
        else:
            p[k] = float(v)
    p["body_setpoint"] = sp
    if costs:
        p["op_cost"] = costs
    return p


def sample_birth_from_arm_d(
    theta: np.ndarray,
    *,
    life_seed: int,
    s_dir: Path | None,
    device: str = "cpu",
    layout: dict[str, Any] | None = None,
) -> NeuralCortex:
    layout = layout or load_layout()
    audit = refuse_audit(theta, "D", layout)
    if not audit["ok"]:
        raise ValueError(f"refuse_audit: {audit['issues']}")
    scalars = decode_scalars(theta, "D", layout)
    rng = np.random.default_rng(int(life_seed))
    g = GenomeConfig(
        seed_birth=int(life_seed),
        seed_registry=int(life_seed) ^ 0xA11,
        seed_source=int(life_seed) ^ 0xB22,
        seed_action=int(life_seed) ^ 0xC33,
        seed_permute=int(life_seed) ^ 0xD44,
        seed_motor=int(life_seed) ^ 0xE55,
        p_connect=float(scalars.get("connect.p_connect", 0.10)),
        eta_pred=float(scalars.get("dyn.eta_pred", 0.05)),
        eta_act=float(scalars.get("dyn.eta_act", 0.15)),
        beta=float(scalars.get("dyn.beta", 0.01)),
        clip=float(scalars.get("dyn.clip", 2.0)),
        tau=float(scalars.get("dyn.tau", 1.0)),
        t_max=int(round(scalars.get("dyn.t_max", 8.0))),
        cos_thresh=float(scalars.get("dyn.cos_thresh", 0.15)),
        lineage_params=scalars_to_lineage_params(scalars),
    )
    ag = NeuralCortex(s_dir, genome=g, device=device)
    n, d_x, d_sym = g.n, g.d_x, g.d_sym

    def draw(name: str, rows: int, cols: int) -> torch.Tensor:
        mu = float(scalars.get(f"init.{name}.mu", 0.0))
        ls = float(scalars.get(f"init.{name}.log_std", -2.0))
        std = float(math.exp(ls))
        arr = rng.normal(mu, std, size=(rows, cols)).astype(np.float64)
        return torch.tensor(arr, dtype=ag.dtype, device=ag.device)

    m = (rng.random((n, n)) < g.p_connect).astype(np.float64)
    np.fill_diagonal(m, 1.0)
    ag.M = torch.tensor(m, dtype=ag.dtype, device=ag.device)
    ag.W_rec = draw("W_rec", n, n) * ag.M
    ag.W_in = draw("W_in", n, d_x)
    ag.W_pred = draw("W_pred", d_sym, n)
    ag.W_op = draw("W_op", len(OPS), n)
    ag.W_emit_query = draw("W_emit_query", d_sym, n)
    ag.W_act_query = draw("W_act_query", d_sym, n)
    ag.W_write = draw("W_write", d_sym, n)
    ag.W_att = draw("W_att", d_sym, n)
    ag.W_body = draw("W_body", n, g.d_body)
    ag.b = draw("b", n, 1).reshape(n)
    ag.b_op = torch.zeros(len(OPS), dtype=ag.dtype, device=ag.device)
    for i, op in enumerate(OPS):
        ag.b_op[i] = float(scalars.get(f"init.b_op.{op}", 0.85 if op == "ACT" else 0.0))
    ag.v_start = rng.normal(
        float(scalars.get("init.v_start.mu", 0.0)),
        math.exp(float(scalars.get("init.v_start.log_std", 0.0))),
        size=d_sym,
    ).astype(np.float64)
    ag.v_end = rng.normal(
        float(scalars.get("init.v_end.mu", 0.0)),
        math.exp(float(scalars.get("init.v_end.log_std", 0.0))),
        size=d_sym,
    ).astype(np.float64)
    ag._plastic_names = list(ag._plastic_names)
    ag.W_slow = {name: getattr(ag, name).detach().clone() for name in ag._plastic_names}
    ag._birth_W = {name: getattr(ag, name).detach().clone() for name in ag._plastic_names}
    ag._birth_W["W_body"] = ag.W_body.detach().clone()
    ag._birth_M = ag.M.detach().clone()
    ag._birth_v_start = ag.v_start.copy()
    ag._birth_v_end = ag.v_end.copy()
    ag._birth_b_op = ag.b_op.detach().clone()
    return ag


def pack_arm_c_from_cortex(ag: NeuralCortex, layout: dict[str, Any] | None = None) -> np.ndarray:
    layout = layout or load_layout()
    theta = defaults_theta("C", layout)
    smap = _slice_map(layout["arms"]["C"])
    tensors = {
        "tensor.W_rec": ag.W_rec,
        "tensor.M": ag.M,
        "tensor.W_in": ag.W_in,
        "tensor.W_pred": ag.W_pred,
        "tensor.W_op": ag.W_op,
        "tensor.W_emit_query": ag.W_emit_query,
        "tensor.W_act_query": ag.W_act_query,
        "tensor.W_write": ag.W_write,
        "tensor.W_att": ag.W_att,
        "tensor.W_body": ag.W_body,
        "tensor.b": ag.b,
        "tensor.b_op": ag.b_op,
        "tensor.v_start": torch.tensor(ag.v_start, dtype=ag.dtype, device=ag.device),
        "tensor.v_end": torch.tensor(ag.v_end, dtype=ag.dtype, device=ag.device),
    }
    for name, ten in tensors.items():
        sl = smap[name]
        flat = ten.detach().cpu().numpy().astype(np.float64).reshape(-1)
        off = int(sl["offset"])
        theta[off : off + int(sl["size"])] = flat
    return theta


def apply_arm_c_theta(
    theta: np.ndarray,
    *,
    life_seed: int,
    s_dir: Path | None,
    device: str = "cpu",
    layout: dict[str, Any] | None = None,
) -> NeuralCortex:
    layout = layout or load_layout()
    audit = refuse_audit(theta, "C", layout)
    if not audit["ok"]:
        raise ValueError(f"refuse_audit: {audit['issues']}")
    scalars = decode_scalars(theta, "C", layout)
    g = GenomeConfig(
        seed_birth=int(life_seed),
        seed_registry=int(life_seed) ^ 0xA11,
        seed_source=int(life_seed) ^ 0xB22,
        seed_action=int(life_seed) ^ 0xC33,
        seed_permute=int(life_seed) ^ 0xD44,
        seed_motor=int(life_seed) ^ 0xE55,
        p_connect=float(scalars.get("connect.p_connect", 0.10)),
        eta_pred=float(scalars.get("dyn.eta_pred", 0.05)),
        eta_act=float(scalars.get("dyn.eta_act", 0.15)),
        beta=float(scalars.get("dyn.beta", 0.01)),
        clip=float(scalars.get("dyn.clip", 2.0)),
        tau=float(scalars.get("dyn.tau", 1.0)),
        t_max=int(round(scalars.get("dyn.t_max", 8.0))),
        cos_thresh=float(scalars.get("dyn.cos_thresh", 0.15)),
        lineage_params=scalars_to_lineage_params(scalars),
    )
    ag = NeuralCortex(s_dir, genome=g, device=device)
    smap = _slice_map(layout["arms"]["C"])

    def take(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        sl = smap[name]
        off = int(sl["offset"])
        arr = theta[off : off + int(sl["size"])].reshape(shape)
        return torch.tensor(arr.copy(), dtype=ag.dtype, device=ag.device)

    n, d_x, d_sym = g.n, g.d_x, g.d_sym
    ag.M = take("tensor.M", (n, n))
    ag.W_rec = take("tensor.W_rec", (n, n)) * ag.M
    ag.W_in = take("tensor.W_in", (n, d_x))
    ag.W_pred = take("tensor.W_pred", (d_sym, n))
    ag.W_op = take("tensor.W_op", (len(OPS), n))
    ag.W_emit_query = take("tensor.W_emit_query", (d_sym, n))
    ag.W_act_query = take("tensor.W_act_query", (d_sym, n))
    ag.W_write = take("tensor.W_write", (d_sym, n))
    ag.W_att = take("tensor.W_att", (d_sym, n))
    ag.W_body = take("tensor.W_body", (n, g.d_body))
    ag.b = take("tensor.b", (n,))
    ag.b_op = take("tensor.b_op", (len(OPS),))
    ag.v_start = take("tensor.v_start", (d_sym,)).detach().cpu().numpy().astype(np.float64)
    ag.v_end = take("tensor.v_end", (d_sym,)).detach().cpu().numpy().astype(np.float64)
    ag.W_slow = {name: getattr(ag, name).detach().clone() for name in ag._plastic_names}
    ag._birth_W = {name: getattr(ag, name).detach().clone() for name in ag._plastic_names}
    ag._birth_W["W_body"] = ag.W_body.detach().clone()
    ag._birth_M = ag.M.detach().clone()
    ag._birth_v_start = ag.v_start.copy()
    ag._birth_v_end = ag.v_end.copy()
    ag._birth_b_op = ag.b_op.detach().clone()
    return ag


def freeze_plasticity(ag: NeuralCortex) -> None:
    ag.genome.eta_pred = 0.0
    ag.genome.eta_act = 0.0
    ag.genome.beta = 0.0


def antithetic_children(
    theta: np.ndarray, eps: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    return theta + sigma * eps, theta - sigma * eps


def rank_centered(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(-0.5, 0.5, num=values.size, dtype=np.float64)
    return ranks


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int = 0
    lr: float = 0.02
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


def adam_step(theta: np.ndarray, grad: np.ndarray, st: AdamState) -> np.ndarray:
    st.t += 1
    st.m = st.b1 * st.m + (1.0 - st.b1) * grad
    st.v = st.b2 * st.v + (1.0 - st.b2) * (grad * grad)
    mhat = st.m / (1.0 - st.b1 ** st.t)
    vhat = st.v / (1.0 - st.b2 ** st.t)
    return theta + st.lr * mhat / (np.sqrt(vhat) + st.eps)


def f_search(adults: list[float], robustness: float, efficiency: float) -> tuple[float, float, float]:
    """Lexicographic tuple. Always defined, even if G_k fails."""
    if not adults:
        return (0.0, 0.0, 0.0)
    q = float(np.quantile(np.asarray(adults, dtype=np.float64), 0.25))
    return (q, -float(robustness), -float(efficiency))


def g_k(adult: float, birth: float, plas_off: float, tau: float, d_b: float, d_p: float) -> bool:
    return bool(adult >= tau and (adult - birth) >= d_b and (adult - plas_off) >= d_p)


def cluster_bootstrap_lower(
    cells: list[tuple[int, int, float]],
    *,
    n_boot: int = 9999,
    seed: int = 20260817,
    alpha: float = 0.05,
) -> float:
    """Percentile cluster bootstrap lower bound of the mean. Worlds, then births. Ticks never iid."""
    by_world: dict[int, list[float]] = {}
    for world_id, _birth_id, value in cells:
        by_world.setdefault(int(world_id), []).append(float(value))
    worlds = list(by_world.keys())
    if not worlds:
        return 0.0
    rng = np.random.default_rng(int(seed))
    means: list[float] = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(np.asarray(worlds, dtype=np.int64), size=len(worlds), replace=True)
        vals: list[float] = []
        for world_id in sampled:
            pool = by_world[int(world_id)]
            idx = rng.choice(len(pool), size=len(pool), replace=True)
            vals.extend(pool[int(i)] for i in idx)
        means.append(float(np.mean(vals)))
    return float(np.quantile(np.asarray(means, dtype=np.float64), float(alpha)))
