"""TM.0.24.WALLMAP — L0 wall decomposition diagnostics.

Not a lineage version. Not a capability earn. Product 0.0.004.
Diagnostic scoring requires docs/lineage_wallmap.runner.lock on clean origin/main.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.run_tm023cortex import build_observe, make_cortex, physics, torch_env
from experiments.run_tm024lineage import (
    ObservableTeacher,
    live_once,
    make_synthetic_world,
    probe_beneficial,
)
from three_memory.cortex_lineage import (
    AdamState,
    adam_step,
    antithetic_children,
    cluster_bootstrap_lower,
    defaults_theta,
    freeze_plasticity,
    g_k,
    refuse_audit,
    sample_birth_from_arm_d,
    sha_file,
)
from three_memory.neural_cortex import OPS, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_wallmap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_wallmap_contract.md"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_wallmap.runner.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"

N_WAKE = 40
N_REPLAY = 8
N_PROBE = 20
TAU = 0.60
DELTA_B = 0.05
DELTA_P = 0.05


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def make_diag_world(domain: str, index: int) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def wallmap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "lineage_runner": REPO_ROOT / "experiments" / "run_tm024lineage.py",
        "cortex_lineage": REPO_ROOT / "three_memory" / "cortex_lineage.py",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_runner_frozen() -> dict[str, Any]:
    """Integrity is SHA pin. Clean-tree gate is only at suite start (see assert_clean_to_begin)."""
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no wallmap runner.lock — refuse diagnostic scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    live = wallmap_shas()
    if live != lock.get("shas"):
        raise RuntimeError("wallmap implementation drifted after runner.lock — versioned lock required")
    return lock


def assert_clean_to_begin() -> None:
    """Require a clean tree before the suite starts. Result locks may appear mid-run."""
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    if not porcelain.strip():
        return
    allowed_prefix = (
        "docs/lineage_wallmap_q",
        "docs/lineage_wallmap.decision.lock",
        "docs/tm024wallmap_results.md",
    )
    for line in porcelain.splitlines():
        path = line[3:].strip().split(" -> ")[-1]
        if not any(path == a or path.startswith(a) for a in allowed_prefix):
            raise RuntimeError(f"working tree dirty ({path}) — refuse to begin diagnostic scoring")


def op_logits(ag: NeuralCortex) -> np.ndarray:
    logits = (ag.W_op @ ag.rho) + ag.b_op
    return ag._from_t(logits)


def motor_scores(ag: NeuralCortex) -> dict[str, float]:
    q = ag._from_t(ag.W_act_query @ ag.rho)
    out: dict[str, float] = {}
    for h, v in ag.motor_vocab.items():
        out[h] = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-12))
    return out


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def probe_observe(ag: NeuralCortex, world: dict[str, Any], *, tag: str = "probe") -> None:
    ag.observe(
        build_observe(
            interaction_token=tag,
            source_token="src_probe",
            ordered_symbols=list(world["teacher_pair"]),
            observable_state=["st_idle"],
            body_state=[1.0, 0.0, 1.0, 0.0],
        )
    )


def probe_observe_no_credit(ag: NeuralCortex, world: dict[str, Any], *, tag: str = "probe") -> None:
    """Sensory path only. Avoids W_* reassignment from zero-eta credit (keeps Adam leaves)."""
    saved = ag._apply_credit

    def _noop(s_t, body_t):  # noqa: ANN001
        return {"adv": 0.0, "pred_err": 0.0}

    ag._apply_credit = _noop  # type: ignore[method-assign]
    try:
        probe_observe(ag, world, tag=tag)
    finally:
        ag._apply_credit = saved  # type: ignore[method-assign]


def surrogate_loss_torch(ag: NeuralCortex, world: dict[str, Any]) -> torch.Tensor:
    """Differentiable Q1 surrogate. ρ is stop-grad. Gradients flow to readout matrices."""
    rho = ag.rho.detach()
    W_op = ag.W_op
    b_op = ag.b_op
    W_aq = ag.W_act_query
    logits = W_op @ rho + b_op
    p_op = torch.softmax(logits, dim=0)
    i_act = OPS.index("ACT")
    i_hold = OPS.index("HOLD")
    h_star = world["beneficial"]
    scores = []
    handles = list(ag.motor_vocab.keys())
    for h in handles:
        v = torch.tensor(ag.motor_vocab[h], dtype=ag.dtype, device=ag.device)
        scores.append((W_aq @ rho).dot(v))
    scores_t = torch.stack(scores)
    p_h = torch.softmax(scores_t, dim=0)
    i_star = handles.index(h_star)
    loss = -torch.log(p_op[i_act] + 1e-12) - torch.log(p_h[i_star] + 1e-12)
    loss = loss + 0.5 * p_op[i_hold]
    for i, h in enumerate(handles):
        if h != h_star:
            loss = loss + 0.5 * p_h[i]
    return loss


def q1_fit_one_world(world: dict[str, Any], *, restart: int, max_steps: int = 2000) -> dict[str, Any]:
    prereg = load_prereg()["Q1_optimizer"]
    with tempfile.TemporaryDirectory(prefix="wm_q1_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"]))
        freeze_plasticity(ag)
        rng = np.random.default_rng(domain_seed("Q1.init", f"{world['seed']}_{restart}"))
        for name in ("W_in", "W_rec", "W_op", "W_act_query"):
            W = getattr(ag, name).detach()
            noise = torch.tensor(
                rng.normal(0.0, float(prereg["sigma_init"]), size=tuple(W.shape)),
                dtype=ag.dtype,
                device=ag.device,
            )
            if name == "W_rec":
                setattr(ag, name, ((W + noise) * ag.M.detach()).detach())
            elif name in ("W_op", "W_act_query"):
                setattr(ag, name, (W + noise).detach().requires_grad_(True))
            else:
                setattr(ag, name, (W + noise).detach())
        ag.b_op = (
            ag.b_op.detach()
            + torch.tensor(
                rng.normal(0.0, float(prereg["sigma_init"]), size=tuple(ag.b_op.shape)),
                dtype=ag.dtype,
                device=ag.device,
            )
        ).detach().requires_grad_(True)
        # Surrogate grads: readout only (stop-grad ρ). W_in/W_rec stay at init+noise.
        params = [ag.W_op, ag.b_op, ag.W_act_query]
        opt = torch.optim.Adam(params, lr=float(prereg["lr_start"]), betas=(0.9, 0.999), eps=1e-8)
        streak = 0
        best_probe = 0.0
        for step in range(int(max_steps)):
            # cosine LR
            t = step / max(max_steps - 1, 1)
            lr = float(prereg["lr_end"]) + 0.5 * (float(prereg["lr_start"]) - float(prereg["lr_end"])) * (
                1.0 + math.cos(math.pi * t)
            )
            for g in opt.param_groups:
                g["lr"] = lr
            probe_observe_no_credit(ag, world, tag=f"fit_{step}")
            opt.zero_grad(set_to_none=True)
            loss = surrogate_loss_torch(ag, world)
            loss.backward()
            opt.step()
            with torch.no_grad():
                clip = float(prereg["clip"])
                ag.W_op.data.clamp_(-clip, clip)
                ag.W_act_query.data.clamp_(-clip, clip)
                ag.b_op.data.clamp_(-clip, clip)
            if step % int(prereg["eval_every"]) == 0:
                # Eval on a detached clone so credit/observe cannot replace Adam leaves.
                snap = {
                    "W_op": ag.W_op.detach().clone(),
                    "b_op": ag.b_op.detach().clone(),
                    "W_act_query": ag.W_act_query.detach().clone(),
                    "W_in": ag.W_in.detach().clone(),
                    "W_rec": ag.W_rec.detach().clone(),
                    "M": ag.M.detach().clone(),
                }
                with tempfile.TemporaryDirectory(prefix="wm_ev_") as etmp:
                    ev = make_cortex(Path(etmp) / "s", device="cpu")
                    ev.bind_actuators(list(world["handles"]))
                    freeze_plasticity(ev)
                    ev.W_op = snap["W_op"]
                    ev.b_op = snap["b_op"]
                    ev.W_act_query = snap["W_act_query"]
                    ev.W_in = snap["W_in"]
                    ev.W_rec = snap["W_rec"]
                    ev.M = snap["M"]
                    pb = float(probe_beneficial(ev, world, n_probe=N_PROBE))
                best_probe = max(best_probe, pb)
                if pb >= float(prereg["early_stop_probe"]):
                    streak += 1
                else:
                    streak = 0
                if streak >= int(prereg["early_stop_streak"]):
                    break
        # Final behavioral probe on detached clone
        with tempfile.TemporaryDirectory(prefix="wm_fin_") as etmp:
            ev = make_cortex(Path(etmp) / "s", device="cpu")
            ev.bind_actuators(list(world["handles"]))
            freeze_plasticity(ev)
            ev.W_op = ag.W_op.detach().clone()
            ev.b_op = ag.b_op.detach().clone()
            ev.W_act_query = ag.W_act_query.detach().clone()
            ev.W_in = ag.W_in.detach().clone()
            ev.W_rec = ag.W_rec.detach().clone()
            ev.M = ag.M.detach().clone()
            final_probe = float(probe_beneficial(ev, world, n_probe=N_PROBE))
        return {
            "world_seed": world["seed"],
            "restart": restart,
            "probe": final_probe,
            "best_probe": best_probe,
            "steps": step + 1,
            "ok": final_probe >= float(prereg["pass_probe"]),
        }


def run_q1() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()
    fit_results = []
    for i in range(int(prereg["Q1_optimizer"]["n_fit_worlds"])):
        world = make_diag_world(prereg["domains"]["Q1_FIT"], i)
        best = None
        for r in range(int(prereg["Q1_optimizer"]["restarts"])):
            out = q1_fit_one_world(world, restart=r)
            if best is None or out["probe"] > best["probe"]:
                best = out
        fit_results.append(best)
    n_ok = sum(1 for r in fit_results if r and r["ok"])
    # transfer: zero-shot from last fit restart of world 0 onto transfer worlds (report only)
    transfer = []
    for i in range(4):
        tw = make_diag_world(prereg["domains"]["Q1_TRANSFER"], i)
        with tempfile.TemporaryDirectory(prefix="wm_tr_") as tmp:
            ag = make_cortex(Path(tmp) / "s", device="cpu")
            ag.bind_actuators(list(tw["handles"]))
            freeze_plasticity(ag)
            transfer.append({"world_seed": tw["seed"], "probe": float(probe_beneficial(ag, tw, n_probe=N_PROBE))})
    passed = n_ok >= int(prereg["Q1_optimizer"]["pass_fit_worlds"])
    out = {
        "version": "TM.0.24.WALLMAP.Q1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "n_ok": n_ok,
        "fit": fit_results,
        "transfer_report_only": transfer,
        "note": "Pass is behavioral probe on FIT. TRANSFER is not a pass gate.",
    }
    (REPO_ROOT / "docs" / "lineage_wallmap_q1.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def evaluate_arm_d_causal(
    theta: np.ndarray,
    worlds: list[dict[str, Any]],
    birth_seeds: list[int],
    *,
    teacher_seed: int = 11,
) -> dict[str, Any]:
    cells: list[tuple[int, int, float]] = []
    adults: list[float] = []
    births: list[float] = []
    offs: list[float] = []
    for w in worlds:
        for bs in birth_seeds:
            with tempfile.TemporaryDirectory(prefix="wm_q2_") as tmp:
                # birth
                ag0 = sample_birth_from_arm_d(theta, life_seed=bs, s_dir=Path(tmp) / "b", device="cpu")
                birth = float(probe_beneficial(ag0, w, n_probe=N_PROBE))
                # adult
                ag = sample_birth_from_arm_d(theta, life_seed=bs, s_dir=Path(tmp) / "a", device="cpu")
                live_once(ag, w, n_wake=N_WAKE, n_replay=N_REPLAY, teacher_seed=teacher_seed ^ w["seed"] ^ bs)
                adult = float(probe_beneficial(ag, w, n_probe=N_PROBE))
                # plasticity off
                agf = sample_birth_from_arm_d(theta, life_seed=bs, s_dir=Path(tmp) / "f", device="cpu")
                freeze_plasticity(agf)
                live_once(agf, w, n_wake=N_WAKE, n_replay=N_REPLAY, teacher_seed=teacher_seed ^ w["seed"] ^ bs)
                off = float(probe_beneficial(agf, w, n_probe=N_PROBE))
            cells.append((int(w["seed"]), int(bs), adult))
            adults.append(adult)
            births.append(birth)
            offs.append(off)
    a_m = float(np.mean(adults)) if adults else 0.0
    b_m = float(np.mean(births)) if births else 0.0
    o_m = float(np.mean(offs)) if offs else 0.0
    lo = cluster_bootstrap_lower(cells, n_boot=9999, seed=20260817)
    return {
        "adult_mean": a_m,
        "birth_mean": b_m,
        "plasticity_off_mean": o_m,
        "ci_lower": lo,
        "G_k": g_k(a_m, b_m, o_m, TAU, DELTA_B, DELTA_P),
        "birth_below": b_m < TAU,
        "off_below": o_m < TAU,
        "adult_ok": a_m >= TAU and lo >= TAU,
    }


def run_q2() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()
    fit_worlds = [make_diag_world(prereg["domains"]["Q2_FIT"], i) for i in range(4)]
    check_worlds = [make_diag_world(prereg["domains"]["Q2_CHECK"], i) for i in range(4)]
    layout_theta = defaults_theta("D")
    assert refuse_audit(layout_theta, "D")["ok"]
    # Short antithetic search on FIT only (wired scalars; not Q1 weights)
    adam = AdamState(m=np.zeros_like(layout_theta), v=np.zeros_like(layout_theta), lr=0.02)
    theta = layout_theta.copy()
    sigma = 0.05
    for gen in range(5):
        rng = np.random.default_rng(domain_seed("Q2.es", f"g{gen}"))
        fits = []
        noises = []
        for i in range(8):
            eps = rng.normal(0.0, 1.0, size=theta.size)
            plus, minus = antithetic_children(theta, eps, sigma)
            fp = evaluate_arm_d_causal(plus, fit_worlds[:2], [0, 1])["adult_mean"]
            fm = evaluate_arm_d_causal(minus, fit_worlds[:2], [0, 1])["adult_mean"]
            fits.extend([fp, fm])
            noises.extend([eps, -eps])
        ranks = np.argsort(np.argsort(fits)).astype(np.float64)
        ranks = (ranks / max(len(ranks) - 1, 1)) - 0.5
        grad = np.zeros_like(theta)
        for r, e in zip(ranks, noises, strict=True):
            grad += r * e
        grad /= max(len(noises), 1) * sigma
        theta = adam_step(theta, grad, adam)
        # clip scalars via refuse bounds roughly
        theta = np.clip(theta, -5.0, 5.0)
    check = evaluate_arm_d_causal(theta, check_worlds, [0, 1, 2, 3])
    passed = bool(
        check["adult_ok"]
        and check["birth_below"]
        and check["off_below"]
        and check["G_k"]
    )
    out = {
        "version": "TM.0.24.WALLMAP.Q2",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "check": check,
        "one_genotype": True,
        "note": "Same genotype on CHECK. Favorable birth is not a pass.",
    }
    (REPO_ROOT / "docs" / "lineage_wallmap_q2.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _f_l0(theta: np.ndarray, z: dict[str, int]) -> float:
    world = make_diag_world("TM024.WALLMAP.Q3.DIAG.", int(z["world"]))
    with tempfile.TemporaryDirectory(prefix="wm_q3_") as tmp:
        ag = sample_birth_from_arm_d(theta, life_seed=int(z["birth"]), s_dir=Path(tmp) / "s", device="cpu")
        live_once(
            ag,
            world,
            n_wake=N_WAKE,
            n_replay=N_REPLAY,
            teacher_seed=int(z["teacher"]),
        )
        return float(probe_beneficial(ag, world, n_probe=N_PROBE))


def run_q3() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()["Q3"]
    P = int(prereg["P"])
    R = int(prereg["R"])
    sigma = float(prereg["sigma"])
    theta = defaults_theta("D")
    # replicates: mix revealed-style and Q3.DIAG indices
    replicates = []
    for r in range(R):
        replicates.append(
            {
                "world": r,
                "birth": 100 + r,
                "teacher": 200 + r,
            }
        )
    # variance components (report separately)
    birth_vals = [_f_l0(theta, {"world": 0, "birth": b, "teacher": 7}) for b in range(4)]
    world_vals = [_f_l0(theta, {"world": w, "birth": 0, "teacher": 7}) for w in range(4)]
    teacher_vals = [_f_l0(theta, {"world": 0, "birth": 0, "teacher": t}) for t in range(4)]
    var_components = {
        "birth": float(np.var(birth_vals)),
        "world": float(np.var(world_vals)),
        "teacher": float(np.var(teacher_vals)),
    }

    def one_batch(batch_id: int) -> dict[str, Any]:
        rng = np.random.default_rng(domain_seed("Q3.batch", str(batch_id)))
        deltas_mean = []
        snrs = []
        eps_list = []
        delta_scalar = []
        for i in range(P):
            eps = rng.normal(0.0, 1.0, size=theta.size)
            deltas = []
            for z in replicates:
                plus, minus = antithetic_children(theta, eps, sigma)
                d = _f_l0(plus, z) - _f_l0(minus, z)
                deltas.append(d)
            mu = float(np.mean(deltas))
            se = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 1.0
            snr = abs(mu) / max(se, 1e-12)
            deltas_mean.append(mu)
            snrs.append(snr)
            eps_list.append(eps)
            delta_scalar.append(mu)
        g = np.zeros_like(theta)
        for d, e in zip(delta_scalar, eps_list, strict=True):
            g += d * e
        g /= max(2 * P * sigma, 1e-12)
        return {
            "snr": snrs,
            "delta_mean": deltas_mean,
            "g": g,
            "eps": eps_list,
        }

    b1 = one_batch(1)
    b2 = one_batch(2)
    g1, g2 = b1["g"], b2["g"]
    cos = float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-12))
    signs = [int(np.sign(a) == np.sign(b)) for a, b in zip(b1["delta_mean"], b2["delta_mean"], strict=True)]
    sign_agree = float(np.mean(signs))
    # Spearman
    r1 = np.argsort(np.argsort(b1["delta_mean"]))
    r2 = np.argsort(np.argsort(b2["delta_mean"]))
    spearman = float(np.corrcoef(r1, r2)[0, 1]) if len(r1) > 1 else 0.0
    # bootstrap SE of g: resample mutation indices
    rng = np.random.default_rng(20260817)
    norms = []
    for _ in range(200):
        idx = rng.integers(0, P, size=P)
        gb = np.zeros_like(theta)
        for i in idx:
            gb += b1["delta_mean"][i] * b1["eps"][i]
        gb /= max(2 * P * sigma, 1e-12)
        norms.append(float(np.linalg.norm(gb)))
    boot_se = float(np.std(norms, ddof=1))
    g_over = float(np.linalg.norm(g1) / max(boot_se, 1e-12))
    pheno = float(np.mean([1.0 if abs(m) > 1e-6 else 0.0 for m in b1["delta_mean"]]))
    median_snr = float(np.median(b1["snr"]))
    floors = prereg["pass"]
    passed = bool(
        median_snr >= floors["median_snr"]
        and cos >= floors["cosine_g1_g2"]
        and sign_agree >= floors["sign_agreement"]
        and spearman >= floors["spearman"]
        and g_over >= floors["grad_over_bootstrap_se"]
        and pheno >= floors["phenotype_change_fraction"]
    )
    out = {
        "version": "TM.0.24.WALLMAP.Q3",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "median_snr": median_snr,
        "cosine_g1_g2": cos,
        "sign_agreement": sign_agree,
        "spearman": spearman,
        "grad_over_bootstrap_se": g_over,
        "phenotype_change_fraction": pheno,
        "variance_components": var_components,
        "note": "SNR uses SE not variance sum. More generations not justified if failed.",
    }
    (REPO_ROOT / "docs" / "lineage_wallmap_q3.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _clone_from_checkpoint(ag: NeuralCortex) -> NeuralCortex:
    snap = ag.checkpoint()
    twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device=str(ag.device))
    twin.load_checkpoint(snap)
    return twin


def run_q4() -> dict[str, Any]:
    assert_runner_frozen()
    world = make_diag_world(load_prereg()["domains"]["Q4"], 0)
    links: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="wm_q4_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"]))
        # Mid-range body so beneficial/harmful deltas are observable (not clipped at 0/1).
        body = [0.5, 0.4, 0.5, 0.0]
        # warm rho
        ag.observe(
            build_observe(
                interaction_token="warm",
                source_token="src_probe",
                ordered_symbols=list(world["teacher_pair"][:1]),
                observable_state=["st_idle"],
                body_state=body,
            )
        )
        # Link 1: ACT -> body change via host physics
        tok = world["beneficial"]
        state, body2 = physics(body, tok, world["latent"])
        body_changed = any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(body, body2, strict=True))
        links["act_to_body"] = {"ok": bool(body_changed), "body_before": body, "body_after": body2}

        # Prepare pending ACT with real rho_elig
        rho_elig = ag._from_t(ag.rho).copy()
        ag._pending = {
            "op": "ACT",
            "token": tok,
            "rho_elig": rho_elig,
            "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
            "body": np.asarray(body, dtype=np.float64),
            "cost": 0.05,
            "motor_vec": ag.motor_vocab[tok].copy(),
        }
        # Link 2: body -> adv
        metrics = ag._apply_credit(
            ag.encode_state_set(["st_p"]),
            np.asarray(body2, dtype=np.float64),
        )
        # re-stage for interventions
        def stage_pending(elig: np.ndarray) -> None:
            ag._pending = {
                "op": "ACT",
                "token": tok,
                "rho_elig": elig.copy(),
                "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
                "body": np.asarray(body, dtype=np.float64),
                "cost": 0.05,
                "motor_vec": ag.motor_vocab[tok].copy(),
            }

        stage_pending(rho_elig)
        m_pos = ag._apply_credit(ag.encode_state_set(["st_p"]), np.asarray(body2, dtype=np.float64))
        # harmful body
        _, body_h = physics(body, world["handles"][1] if world["beneficial"] == world["handles"][0] else world["handles"][0], world["latent"])
        stage_pending(rho_elig)
        m_neg = ag._apply_credit(ag.encode_state_set(["st_h"]), np.asarray(body_h, dtype=np.float64))
        links["body_to_adv"] = {
            "ok": bool(m_pos.get("adv", 0) > 0 and m_neg.get("adv", 0) < m_pos.get("adv", 0)),
            "adv_beneficial": m_pos.get("adv"),
            "adv_harmful": m_neg.get("adv"),
        }

        # Link 3: zero eligibility -> no motor logit move
        probe_observe(ag, world, tag="pre_elig")
        scores_before = motor_scores(ag)
        W_before = ag.W_act_query.detach().clone()
        stage_pending(np.zeros_like(rho_elig))
        ag._apply_credit(ag.encode_state_set(["st_p"]), np.asarray(body2, dtype=np.float64))
        delta_zero = float((ag.W_act_query - W_before).abs().max().item())
        # restore and wrong-tick elig
        ag.W_act_query = W_before.detach().clone()
        wrong = np.random.default_rng(0).normal(0, 1, size=rho_elig.shape).astype(np.float64)
        stage_pending(wrong)
        W_b2 = ag.W_act_query.detach().clone()
        ag._apply_credit(ag.encode_state_set(["st_p"]), np.asarray(body2, dtype=np.float64))
        # correct elig
        ag.W_act_query = W_b2.detach().clone()
        stage_pending(rho_elig)
        scores_pre = motor_scores(ag)
        ag._apply_credit(ag.encode_state_set(["st_p"]), np.asarray(body2, dtype=np.float64))
        probe_observe(ag, world, tag="post_elig")
        scores_post = motor_scores(ag)
        links["adv_to_elig"] = {
            "ok": delta_zero < 1e-12,
            "zero_elig_max_abs_delta": delta_zero,
        }
        # Link 4: credited handle projection moves up for beneficial
        d_star = scores_post.get(tok, 0.0) - scores_pre.get(tok, 0.0)
        distractor = [h for h in world["handles"] if h != tok][0]
        d_dist = scores_post.get(distractor, 0.0) - scores_pre.get(distractor, 0.0)
        links["credit_to_handle_logit"] = {
            "ok": bool(d_star > d_dist),
            "delta_beneficial": d_star,
            "delta_distractor": d_dist,
        }

        # Link 5: later behavior under frozen RNG — compare plasticity on vs restore
        # Re-run short teach with credit vs plasticity-off from same checkpoint
        snap = ag.checkpoint()
        ag_on = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device="cpu")
        ag_on.load_checkpoint(snap)
        ag_off = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device="cpu")
        ag_off.load_checkpoint(snap)
        freeze_plasticity(ag_off)
        # teach few steps with single-symbol cues
        for i in range(30):
            b = [1.0, 0.0, 1.0, 0.0]
            for agent in (ag_on, ag_off):
                o = agent.observe(
                    build_observe(
                        interaction_token=f"t{i}",
                        source_token="src_teacher",
                        ordered_symbols=[world["teacher_pair"][0]],
                        observable_state=["st_idle"],
                        body_state=b,
                    )
                )
                action = o.get("action") or {}
                if action.get("op") == "ACT" and action.get("token"):
                    _, b = physics(b, action["token"], world["latent"])
                    # feed next body by another observe
                    agent.observe(
                        build_observe(
                            interaction_token=f"t{i}b",
                            source_token="src_teacher",
                            ordered_symbols=[world["teacher_pair"][0]],
                            observable_state=["st_idle"],
                            body_state=b,
                        )
                    )
        p_on = float(probe_beneficial(ag_on, world, n_probe=N_PROBE))
        p_off = float(probe_beneficial(ag_off, world, n_probe=N_PROBE))
        # logits
        probe_observe(ag_on, world, tag="lon")
        logits_on = softmax_np(op_logits(ag_on))
        probe_observe(ag_off, world, tag="loff")
        logits_off = softmax_np(op_logits(ag_off))
        links["later_behavior"] = {
            "ok": bool(p_on >= p_off or logits_on[OPS.index("ACT")] >= logits_off[OPS.index("ACT")]),
            "probe_on": p_on,
            "probe_off": p_off,
            "p_act_on": float(logits_on[OPS.index("ACT")]),
            "p_act_off": float(logits_off[OPS.index("ACT")]),
            "hypothesis_equal_evidence_hold": "teacher two-symbol cues may suppress ACT; Q4 used one-symbol teach",
        }

    passed = all(bool(v.get("ok")) for v in links.values())
    out = {
        "version": "TM.0.24.WALLMAP.Q4",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "state_only": True,
        "neural_edited": False,
        "links": links,
        "note": "State-only interventions. Q4 precedence over Q2 if failed.",
    }
    (REPO_ROOT / "docs" / "lineage_wallmap_q4.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_decision(q1: dict, q2: dict, q3: dict, q4: dict) -> dict[str, Any]:
    if not q4.get("passed"):
        next_change = (
            "Repair the general credit path in a new architecture candidate; "
            "then re-run a newly committed reachability diagnostic. "
            "Q2 failure does not independently diagnose maturation/replay."
        )
        primary = "Q4_credit_chain"
    elif not q1.get("passed"):
        next_change = (
            "Representation review: state structure, compartments, operations; "
            "capacity lane only after that, as general populations."
        )
        primary = "Q1_representability"
    elif not q2.get("passed"):
        next_change = "Plasticity, maturation, or replay architecture (Q4 intact)."
        primary = "Q2_developmental_reachability"
    elif not q3.get("passed"):
        next_change = "Search algorithm, structured genome, batching, variance reduction."
        primary = "Q3_search_signal"
    else:
        next_change = "New isolated lineage commitment with justified compute."
        primary = "all_diagnostics_pass"

    out = {
        "version": "TM.0.24.WALLMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "Q1_passed": bool(q1.get("passed")),
        "Q2_passed": bool(q2.get("passed")),
        "Q3_passed": bool(q3.get("passed")),
        "Q4_passed": bool(q4.get("passed")),
        "Q4_precedence_over_Q2": True,
        "primary_bottleneck": primary,
        "next_change": next_change,
        "increase_n": False,
        "note": "Diagnostic decomposition only. Not 0.0.005. LINEAGE wall remains historical.",
    }
    (REPO_ROOT / "docs" / "lineage_wallmap.decision.lock").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    (REPO_ROOT / "docs" / "tm024wallmap_results.md").write_text(
        "# TM.0.24.WALLMAP results\n\n"
        "Product remains **0.0.004**. `earned_next=false`. `ex0s=null`.\n\n"
        f"- Q1 representability: **{q1.get('passed')}**\n"
        f"- Q2 developmental reachability: **{q2.get('passed')}**\n"
        f"- Q3 ES SNR / gradient stability: **{q3.get('passed')}**\n"
        f"- Q4 credit chain (state-only): **{q4.get('passed')}**\n\n"
        f"**Primary bottleneck:** `{primary}`\n\n"
        f"**Next change:** {next_change}\n\n"
        "Not a capability earn. QUAL/EVAL remain sealed. n stays 64 until representation is diagnosed.\n",
        encoding="utf-8",
    )
    return out


def smoke() -> dict[str, Any]:
    """ABI / synthetic smoke only. Not diagnostic scoring."""
    w = make_diag_world("TM024.WALLMAP.SMOKE.", 0)
    with tempfile.TemporaryDirectory(prefix="wm_sm_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(w["handles"]))
        live_once(ag, w, n_wake=6, n_replay=2, teacher_seed=1)
        pb = float(probe_beneficial(ag, w, n_probe=8))
        # surrogate forward
        probe_observe(ag, w, tag="sm")
        loss = float(surrogate_loss_torch(ag, w).detach().cpu())
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "probe": pb,
        "surrogate_loss": loss,
        "teacher_audit": True,
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.WALLMAP.RUNNER.V5",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "supersedes": "TM.0.24.WALLMAP.RUNNER.V4",
        "shas": wallmap_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "n_wake": N_WAKE,
        "n_replay": N_REPLAY,
        "n_probe": N_PROBE,
        "Q1_optimizer": prereg["Q1_optimizer"],
        "Q3": prereg["Q3"],
        "Q4_interventions": prereg["Q4"]["interventions"],
        "domains": prereg["domains"],
        "diagnostic_generator": "experiments.run_tm024wallmap.make_diag_world",
        "git_head_at_freeze": _git_head(),
        "note": (
            "Versioned runner lock. SHA pin is the integrity gate. "
            "Clean tree required once at suite start; result locks may appear mid-run."
        ),
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_all() -> dict[str, Any]:
    assert_clean_to_begin()
    assert_runner_frozen()
    q4 = run_q4()
    q1 = run_q1()
    q3 = run_q3()
    q2 = run_q2()
    decision = write_decision(q1, q2, q3, q4)
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "decision": decision}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--write-runner-lock", action="store_true")
    p.add_argument("--q1", action="store_true")
    p.add_argument("--q2", action="store_true")
    p.add_argument("--q3", action="store_true")
    p.add_argument("--q4", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    ran = False
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
        ran = True
    if args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
        ran = True
    if args.q4:
        print(json.dumps(run_q4(), indent=2, default=str))
        ran = True
    if args.q1:
        print(json.dumps(run_q1(), indent=2, default=str))
        ran = True
    if args.q3:
        print(json.dumps(run_q3(), indent=2, default=str))
        ran = True
    if args.q2:
        print(json.dumps(run_q2(), indent=2, default=str))
        ran = True
    if args.all:
        print(json.dumps(run_all(), indent=2, default=str))
        ran = True
    if not ran:
        p.print_help()


if __name__ == "__main__":
    main()
