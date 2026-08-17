"""TM.0.24.STATEMAP — developmental state-transfer and maturation cells.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Scoring requires docs/lineage_statemap.runner.lock on clean origin/main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.run_tm023cortex import make_cortex, physics, torch_env
from experiments.run_tm024actorcredit import (
    MID_BODY,
    clone_frozen,
    harmful_handle,
    motor_scores,
    observe_cue,
    op_logits,
    p_handle,
    prep_eval,
    softmax_np,
)
from experiments.run_tm024lineage import make_synthetic_world
from three_memory.cortex_lineage import defaults_theta, load_layout, sample_birth_from_arm_d, sha_file
from three_memory.neural_cortex import OPS, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_statemap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_statemap_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_statemap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_statemap.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_statemap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024statemap_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v29.lock"
REACH_LOCK = REPO_ROOT / "docs" / "lineage_actorcredit_reach.lock"

CELLS_DOMAIN = "TM024.STATEMAP.CELLS."
TWIN_DOMAIN = "TM024.STATEMAP.TWIN."
AGE_STAGES = (
    "birth",
    "high_plasticity",
    "experience_replay",
    "pruning_stabilization",
    "mature_plasticity",
    "novelty_reopen",
)
AGE_USED = {
    "eta_pred_scale",
    "eta_act_scale",
    "beta_scale",
    "conflict_hold_scale",
    "refractory",
    "growth_scale",
    "prune_scale",
}
S2_DELAYS = (1, 2, 4, 8)
S3_DISTRACTORS = (1, 2, 4, 8)


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def make_cell_world(index: int, domain: str = CELLS_DOMAIN) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def statemap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v29": CANDIDATE,
        "actorcredit_reach": REACH_LOCK,
        "cortex_lineage": REPO_ROOT / "three_memory" / "cortex_lineage.py",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no statemap runner.lock — refuse cell scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if statemap_shas() != lock.get("shas"):
        raise RuntimeError("statemap implementation drifted after runner.lock")
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if sha_file(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from v29 candidate")
    if cand.get("genome", {}).get("n") != 64:
        raise RuntimeError("n must stay 64")
    return lock


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    return ag


def fnum(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / den)


def project_onto(probe: np.ndarray, teach: np.ndarray) -> np.ndarray:
    teach = np.asarray(teach, dtype=np.float64).reshape(-1)
    probe = np.asarray(probe, dtype=np.float64).reshape(-1)
    denom = float(np.dot(teach, teach) + 1e-12)
    return teach * float(np.dot(probe, teach) / denom)


def read_policy(ag: NeuralCortex, world: dict[str, Any], *, rho: np.ndarray | None = None) -> dict[str, Any]:
    saved = None
    if rho is not None:
        saved = ag._from_t(ag.rho)
        ag.rho = ag._to_t(np.asarray(rho, dtype=np.float64))
    ben = world["beneficial"]
    harm = harmful_handle(world)
    scores = motor_scores(ag)
    logits = op_logits(ag)
    sm = softmax_np(logits)
    out = {
        "p_act": float(sm[OPS.index("ACT")]),
        "p_hold": float(sm[OPS.index("HOLD")]),
        "act_logit": float(logits[OPS.index("ACT")]),
        "hold_logit": float(logits[OPS.index("HOLD")]),
        "p_handle": p_handle(ag, ben),
        "motor_ben": float(scores.get(ben, 0.0)),
        "motor_harm": float(scores.get(harm, 0.0)),
    }
    if saved is not None:
        ag.rho = ag._to_t(saved)
    return out


def moved(pre: dict[str, Any], post: dict[str, Any]) -> bool:
    return bool(
        post["motor_ben"] > pre["motor_ben"] + 1e-9
        or post["act_logit"] > pre["act_logit"] + 1e-9
    )


def ranking_ok(pol: dict[str, Any]) -> bool:
    return bool(pol["motor_ben"] > pol["motor_harm"] + 1e-9)


def teach_one(
    ag: NeuralCortex,
    world: dict[str, Any],
    tok: str,
    *,
    tag: str,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    body = list(MID_BODY)
    observe_cue(ag, world, tag=f"{tag}_sel", body=body, symbols=symbols)
    rho_teach = np.asarray(ag.last_action["rho_elig"], dtype=np.float64).copy()
    pre = read_policy(ag, world, rho=rho_teach)
    w_op0 = ag.W_op.detach().clone()
    w_q0 = ag.W_act_query.detach().clone()
    ag.clamp_action("ACT", tok)
    _, body2 = physics(body, tok, world["latent"])
    out = observe_cue(ag, world, tag=f"{tag}_obs", body=body2, symbols=symbols)
    prep_eval(ag)
    post = read_policy(ag, world, rho=rho_teach)
    return {
        "rho_teach": rho_teach,
        "pre": pre,
        "post_ident": post,
        "adv": float((out.get("metrics") or {}).get("adv") or 0.0),
        "d_w_op": float((ag.W_op - w_op0).abs().max().item()),
        "d_w_q": float((ag.W_act_query - w_q0).abs().max().item()),
        "moved": moved(pre, post),
        "ranking": ranking_ok(post),
    }


def live_probe(ag: NeuralCortex, world: dict[str, Any], *, tag: str, symbols: list[str] | None = None) -> dict[str, Any]:
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=symbols)
    rho = probe._from_t(probe.rho).copy()
    pol = read_policy(probe, world)
    pol["rho"] = rho
    return pol


def empty_tick(ag: NeuralCortex, *, tag: str) -> None:
    ag.observe(
        {
            "interaction_token": tag,
            "source_token": "src_sm_neutral",
            "ordered_symbols": [],
            "observable_state": ["st_idle"],
            "body_state": list(MID_BODY),
        }
    )
    prep_eval(ag)


def distractor_syms(world: dict[str, Any]) -> list[str]:
    pair = set(world["teacher_pair"])
    return [s for s in world["symbols"] if s not in pair]


def policy_brief(p: dict[str, Any]) -> dict[str, float]:
    return {
        "p_act": fnum(p["p_act"]),
        "p_hold": fnum(p["p_hold"]),
        "act_logit": fnum(p["act_logit"]),
        "motor_ben": fnum(p["motor_ben"]),
        "motor_harm": fnum(p["motor_harm"]),
        "p_handle": fnum(p["p_handle"]),
    }


def run_s0(*, domain: str = CELLS_DOMAIN, index: int = 0) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    with tempfile.TemporaryDirectory(prefix="sm_s0_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s0")
        passed = bool(t["moved"] and (t["d_w_op"] > 1e-12 or t["d_w_q"] > 1e-12))
    return {
        "id": "S0",
        "passed": passed,
        "pre": policy_brief(t["pre"]),
        "post_ident": policy_brief(t["post_ident"]),
        "d_w_op": t["d_w_op"],
        "d_w_q": t["d_w_q"],
        "adv": t["adv"],
        "domain": domain,
    }


def run_s1() -> dict[str, Any]:
    world = make_cell_world(1)
    with tempfile.TemporaryDirectory(prefix="sm_s1_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s1")
        live = live_probe(ag, world, tag="s1_event")
        live_moved = moved(t["pre"], live)
        ident_ok = bool(t["moved"])
        passed = bool(ident_ok and live_moved)
    return {
        "id": "S1",
        "passed": passed,
        "ident_moved": ident_ok,
        "event_moved": live_moved,
        "cosine_teach_vs_event": cosine(t["rho_teach"], live["rho"]),
        "ident": policy_brief(t["post_ident"]),
        "event": policy_brief(live),
    }


def run_s2() -> dict[str, Any]:
    world = make_cell_world(2)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sm_s2_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s2")
        for d in S2_DELAYS:
            twin = clone_frozen(ag)
            # unfreeze delay ticks? delays are sensory; frozen clone still updates rho.
            for i in range(int(d)):
                empty_tick(twin, tag=f"s2_d{d}_{i}")
            live = live_probe(twin, world, tag=f"s2_p{d}")
            rows.append(
                {
                    "delay": int(d),
                    "moved": moved(t["pre"], live),
                    "cosine": cosine(t["rho_teach"], live["rho"]),
                    "policy": policy_brief(live),
                }
            )
        passed = bool(t["moved"] and all(r["moved"] for r in rows))
    return {"id": "S2", "passed": passed, "ident_moved": bool(t["moved"]), "delays": rows}


def run_s3() -> dict[str, Any]:
    world = make_cell_world(3)
    dist = distractor_syms(world)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sm_s3_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s3")
        for n in S3_DISTRACTORS:
            twin = clone_frozen(ag)
            for i in range(int(n)):
                sym = dist[i % len(dist)]
                observe_cue(twin, world, tag=f"s3_x{n}_{i}", body=list(MID_BODY), symbols=[sym])
                prep_eval(twin)
            live = live_probe(twin, world, tag=f"s3_p{n}")
            rows.append(
                {
                    "n": int(n),
                    "moved": moved(t["pre"], live),
                    "cosine": cosine(t["rho_teach"], live["rho"]),
                    "policy": policy_brief(live),
                }
            )
        passed = bool(t["moved"] and all(r["moved"] for r in rows))
    return {"id": "S3", "passed": passed, "ident_moved": bool(t["moved"]), "distractors": rows}


def run_s4() -> dict[str, Any]:
    world = make_cell_world(4)
    with tempfile.TemporaryDirectory(prefix="sm_s4_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s4")
        ag.reset_rho()
        prep_eval(ag)
        live = live_probe(ag, world, tag="s4_reset")
        passed = bool(t["moved"] and moved(t["pre"], live))
    return {
        "id": "S4",
        "passed": passed,
        "ident_moved": bool(t["moved"]),
        "reset_moved": moved(t["pre"], live),
        "cosine_teach_vs_reset": cosine(t["rho_teach"], live["rho"]),
        "ident": policy_brief(t["post_ident"]),
        "after_reset": policy_brief(live),
    }


def run_s5() -> dict[str, Any]:
    world = make_cell_world(5)
    hist = distractor_syms(world)[0]
    with tempfile.TemporaryDirectory(prefix="sm_s5_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s5")
        twin = clone_frozen(ag)
        observe_cue(twin, world, tag="s5_hist", body=list(MID_BODY), symbols=[hist])
        prep_eval(twin)
        live = live_probe(twin, world, tag="s5_cue")
        passed = bool(t["moved"] and moved(t["pre"], live))
    return {
        "id": "S5",
        "passed": passed,
        "ident_moved": bool(t["moved"]),
        "history_moved": moved(t["pre"], live),
        "cosine_teach_vs_history": cosine(t["rho_teach"], live["rho"]),
        "ident": policy_brief(t["post_ident"]),
        "after_history": policy_brief(live),
    }


def run_s6() -> dict[str, Any]:
    world = make_cell_world(6)
    cue_a = list(world["teacher_pair"])[0]
    cue_b = distractor_syms(world)[0]
    ben = world["beneficial"]
    harm = harmful_handle(world)
    with tempfile.TemporaryDirectory(prefix="sm_s6_") as tmp:
        ag = _fresh(tmp, "s", world)
        ta = teach_one(ag, world, ben, tag="s6a", symbols=[cue_a])
        tb = teach_one(ag, world, harm, tag="s6b", symbols=[cue_b])
        pa = live_probe(ag, world, tag="s6pa", symbols=[cue_a])
        pb = live_probe(ag, world, tag="s6pb", symbols=[cue_b])
        sep_ident = bool(
            ta["post_ident"]["motor_ben"] > ta["post_ident"]["motor_harm"]
            and tb["post_ident"]["motor_harm"] > tb["post_ident"]["motor_ben"]
        )
        sep_live = bool(pa["motor_ben"] > pa["motor_harm"] and pb["motor_harm"] > pb["motor_ben"])
        passed = bool(ta["moved"] and tb["moved"] and sep_ident and sep_live)
    return {
        "id": "S6",
        "passed": passed,
        "sep_ident": sep_ident,
        "sep_live": sep_live,
        "cue_a_ident": policy_brief(ta["post_ident"]),
        "cue_b_ident": policy_brief(tb["post_ident"]),
        "cue_a_live": policy_brief(pa),
        "cue_b_live": policy_brief(pb),
        "cosine_a": cosine(ta["rho_teach"], pa["rho"]),
        "cosine_b": cosine(tb["rho_teach"], pb["rho"]),
    }


def _s7_body(world: dict[str, Any], domain: str, index: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sm_s7_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s7")
        probe = clone_frozen(ag)
        observe_cue(probe, world, tag="s7_live", body=list(MID_BODY))
        rho_probe = probe._from_t(probe.rho).copy()
        live = read_policy(probe, world)
        sub = read_policy(probe, world, rho=t["rho_teach"])
        proj = project_onto(rho_probe, t["rho_teach"])
        projected = read_policy(probe, world, rho=proj)
        ident = t["post_ident"]
        live_ok = moved(t["pre"], live)
        sub_ok = moved(t["pre"], sub)
        proj_ok = moved(t["pre"], projected)
        if sub_ok and not live_ok:
            kind = "representation_alignment"
            passed = False
        elif not t["moved"]:
            kind = "readout_or_plasticity"
            passed = False
        elif t["moved"] and live_ok:
            kind = "aligned"
            passed = True
        else:
            kind = "partial"
            passed = False
        return {
            "id": "S7",
            "passed": passed,
            "kind": kind,
            "ident_moved": bool(t["moved"]),
            "live_moved": live_ok,
            "substitute_moved": sub_ok,
            "projected_moved": proj_ok,
            "cosine_teach_vs_probe": cosine(t["rho_teach"], rho_probe),
            "act_logit_teach": fnum(ident["act_logit"]),
            "act_logit_probe": fnum(live["act_logit"]),
            "handle_score_teach": fnum(ident["motor_ben"]),
            "handle_score_probe": fnum(live["motor_ben"]),
            "ident": policy_brief(ident),
            "live": policy_brief(live),
            "substitute": policy_brief(sub),
            "projected": policy_brief(projected),
            "domain": domain,
            "index": index,
        }


def run_s7(*, domain: str = CELLS_DOMAIN, index: int = 7) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    return _s7_body(world, domain, index)


def run_s8() -> dict[str, Any]:
    world = make_cell_world(8)
    with tempfile.TemporaryDirectory(prefix="sm_s8_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s8")
        live = live_probe(ag, world, tag="s8p")
        rank = ranking_ok(t["post_ident"])
        act_up = bool(live["p_act"] > t["pre"]["p_act"] + 1e-9)
        hold_comp = bool(rank and not act_up)
        passed = bool(t["moved"] and rank and act_up)
    return {
        "id": "S8",
        "passed": passed,
        "ranking": rank,
        "act_increased": act_up,
        "hold_competition": hold_comp,
        "ident": policy_brief(t["post_ident"]),
        "live": policy_brief(live),
        "pre": policy_brief(t["pre"]),
    }


def _swap_slow(ag: NeuralCortex) -> dict[str, torch.Tensor]:
    snap = {n: getattr(ag, n).detach().clone() for n in ag._plastic_names}
    for n in ag._plastic_names:
        setattr(ag, n, ag.W_slow[n].detach().clone())
    return snap


def _restore_w(ag: NeuralCortex, snap: dict[str, torch.Tensor]) -> None:
    for n, t in snap.items():
        setattr(ag, n, t)


def run_s9() -> dict[str, Any]:
    world = make_cell_world(9)
    with tempfile.TemporaryDirectory(prefix="sm_s9_") as tmp:
        ag = _fresh(tmp, "s", world)
        t = teach_one(ag, world, world["beneficial"], tag="s9")
        fast = t["post_ident"]
        snap = _swap_slow(ag)
        slow = read_policy(ag, world, rho=t["rho_teach"])
        _restore_w(ag, snap)
        ag.rest_epoch(8, body=np.asarray(MID_BODY, dtype=np.float64))
        prep_eval(ag)
        rest = live_probe(ag, world, tag="s9_rest")
        fast_ok = moved(t["pre"], fast)
        rest_ok = moved(t["pre"], rest)
        slow_ok = moved(t["pre"], slow)
        passed = bool(fast_ok and rest_ok)
    return {
        "id": "S9",
        "passed": passed,
        "fast_moved": fast_ok,
        "slow_moved": slow_ok,
        "rest_moved": rest_ok,
        "fast": policy_brief(fast),
        "slow": policy_brief(slow),
        "post_rest": policy_brief(rest),
        "cosine_teach_vs_rest": cosine(t["rho_teach"], rest["rho"]),
    }


def _age_row(ag: NeuralCortex, epoch: int) -> dict[str, float]:
    ag.dev_epoch = int(epoch)
    return {
        "epoch": float(epoch),
        "eta_act_scale": float(ag._age_scale("eta_act_scale", 1.0)),
        "eta_pred_scale": float(ag._age_scale("eta_pred_scale", 1.0)),
        "beta_scale": float(ag._age_scale("beta_scale", 1.0)),
        "explore_T": float(ag._age_scale("explore_T", 1.0)),
        "conflict_hold_scale": float(ag._age_scale("conflict_hold_scale", 1.0)),
        "growth_scale": float(ag._age_scale("growth_scale", 0.0)),
        "prune_scale": float(ag._age_scale("prune_scale", 0.0)),
        "wm_persist": float(ag._age_scale("wm_persist", 1.0)),
        "refractory": float(ag._age_scale("refractory", 1.0)),
        "effective_eta_act": float(ag.genome.eta_act) * float(ag._age_scale("eta_act_scale", 1.0)),
    }


def run_s10() -> dict[str, Any]:
    world = make_cell_world(10)
    with tempfile.TemporaryDirectory(prefix="sm_s10_") as tmp:
        live = _fresh(tmp, "live", world)
        live_rows = [_age_row(live, e) for e in range(len(AGE_STAGES))]
        theta = defaults_theta("D")
        arm = sample_birth_from_arm_d(theta, life_seed=10, s_dir=Path(tmp) / "armd", device="cpu")
        arm.bind_actuators(list(world["handles"]))
        arm_rows = [_age_row(arm, e) for e in range(len(AGE_STAGES))]
        keys = ("effective_eta_act", "eta_pred_scale", "beta_scale")
        live_uniq = {tuple(r[k] for k in keys) for r in live_rows}
        arm_uniq = {tuple(r[k] for k in keys) for r in arm_rows}
        passed = bool(len(live_uniq) > 1 or len(arm_uniq) > 1)
    return {
        "id": "S10",
        "passed": passed,
        "live_stage_distinct": len(live_uniq) > 1,
        "arm_d_default_stage_distinct": len(arm_uniq) > 1,
        "live_rows": live_rows,
        "arm_d_default_rows": arm_rows,
        "live_lineage_params": bool(live.genome.lineage_params),
    }


def _tensor_stat(ag: NeuralCortex, name: str, kind: str) -> float:
    t = getattr(ag, name)
    arr = ag._from_t(t) if torch.is_tensor(t) else np.asarray(t, dtype=np.float64)
    return float(arr.mean() if kind == "mu" else arr.std())


def runtime_channel(ag: NeuralCortex, name: str) -> tuple[str, float | None]:
    """Return (channel, value). value is None when the gene has no runtime reader."""
    if name.startswith("init.b_op."):
        op = name.split(".")[-1]
        return f"b_op.{op}", float(ag.b_op[OPS.index(op)].item())
    if name.startswith("init.") and name.endswith(".mu"):
        ten = name.split(".")[1]
        if ten == "b":
            return "b.mean", _tensor_stat(ag, "b", "mu")
        if ten in ("v_start", "v_end"):
            return f"{ten}.mean", float(np.mean(getattr(ag, ten)))
        if hasattr(ag, ten):
            return f"{ten}.mean", _tensor_stat(ag, ten, "mu")
        return name, None
    if name.startswith("init.") and name.endswith(".log_std"):
        ten = name.split(".")[1]
        if ten == "b":
            return "b.std", _tensor_stat(ag, "b", "std")
        if ten in ("v_start", "v_end"):
            return f"{ten}.std", float(np.std(getattr(ag, ten)))
        if hasattr(ag, ten):
            return f"{ten}.std", _tensor_stat(ag, ten, "std")
        return name, None
    genome_map = {
        "connect.p_connect": ("genome.p_connect", lambda: float(ag.genome.p_connect)),
        "dyn.eta_pred": ("genome.eta_pred", lambda: float(ag.genome.eta_pred)),
        "dyn.eta_act": ("genome.eta_act", lambda: float(ag.genome.eta_act)),
        "dyn.beta": ("genome.beta", lambda: float(ag.genome.beta)),
        "dyn.clip": ("genome.clip", lambda: float(ag.genome.clip)),
        "dyn.tau": ("genome.tau", lambda: float(ag.genome.tau)),
        "dyn.t_max": ("genome.t_max", lambda: float(ag.genome.t_max)),
        "dyn.cos_thresh": ("genome.cos_thresh", lambda: float(ag.genome.cos_thresh)),
    }
    if name in genome_map:
        ch, fn = genome_map[name]
        return ch, fn()
    lp_used = {
        "connect.growth_rate": "connect.growth_rate",
        "connect.prune_rate": "connect.prune_rate",
        "connect.prune_threshold": "connect.prune_threshold",
        "dyn.familiarity_decay": "familiarity_decay",
        "dyn.familiarity_abs": "familiarity_abs",
        "dyn.echoic_max": "echoic_max",
        "dyn.echoic_bias": "echoic_bias",
        "dyn.vocal_refractory": "vocal_refractory",
        "dyn.conflict_hold_bias": "conflict_hold_bias",
        "dyn.adv_baseline_alpha": "adv_baseline_alpha",
        "dyn.equal_evidence_min_symbols": "equal_evidence_min_symbols",
        "replay.mix.recency": "replay.mix.recency",
        "replay.mix.similarity": "replay.mix.similarity",
        "replay.mix.surprise": "replay.mix.surprise",
        "replay.mix.random": "replay.mix.random",
    }
    if name in lp_used:
        key = lp_used[name]
        return f"lp.{key}", float(ag._lp(key, 0.0))
    if name.startswith("dyn.op_cost."):
        op = name.split(".")[-1]
        return f"op_cost.{op}", float(ag._op_cost.get(op, 0.0))
    if name.startswith("dyn.body_setpoint."):
        i = int(name.rsplit(".", 1)[-1])
        return f"body_setpoint.{i}", float(ag._body_setpoint[i])
    if name.startswith("age."):
        _age, stage, key = name.split(".", 2)
        ag.dev_epoch = AGE_STAGES.index(stage)
        if key not in AGE_USED:
            return f"age.{stage}.{key}", None
        default = 0.0 if key in ("growth_scale", "prune_scale") else 1.0
        return f"age.{stage}.{key}", float(ag._age_scale(key, default))
    return name, None


def run_s11() -> dict[str, Any]:
    layout = load_layout()
    slices = list(layout["arms"]["D"]["slices"])
    world = make_cell_world(11)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sm_s11_") as tmp:
        base_theta = defaults_theta("D", layout)
        base = sample_birth_from_arm_d(base_theta, life_seed=11, s_dir=Path(tmp) / "base", device="cpu")
        base.bind_actuators(list(world["handles"]))
        for sl in slices:
            name = str(sl["name"])
            off = int(sl["offset"])
            lo, hi, default = float(sl["lo"]), float(sl["hi"]), float(sl["default"])
            alt = hi if abs(hi - default) > 1e-12 else lo
            theta = base_theta.copy()
            theta[off] = alt
            ch0, v0 = runtime_channel(base, name)
            pert = sample_birth_from_arm_d(theta, life_seed=11, s_dir=Path(tmp) / f"g{off}", device="cpu")
            pert.bind_actuators(list(world["handles"]))
            ch1, v1 = runtime_channel(pert, name)
            wired = v0 is not None and v1 is not None
            effect = bool(wired and abs(float(v1) - float(v0)) > 1e-12)
            rows.append(
                {
                    "name": name,
                    "channel": ch1,
                    "wired": wired,
                    "effect": effect,
                    "default": default,
                    "perturbed": alt,
                    "v_default": None if v0 is None else float(v0),
                    "v_perturbed": None if v1 is None else float(v1),
                }
            )
    n = len(rows)
    n_effect = sum(1 for r in rows if r["effect"])
    n_dead = sum(1 for r in rows if not r["wired"] or not r["effect"])
    dead = [r["name"] for r in rows if not r["wired"] or not r["effect"]]
    passed = bool(n_dead == 0)
    return {
        "id": "S11",
        "passed": passed,
        "n_genes": n,
        "n_effect": n_effect,
        "n_dead": n_dead,
        "dead": dead,
        "rows": rows,
    }


def run_s12(s0: dict[str, Any], s7: dict[str, Any]) -> dict[str, Any]:
    twin0 = run_s0(domain=TWIN_DOMAIN, index=0)
    twin7 = run_s7(domain=TWIN_DOMAIN, index=0)
    same_s0 = bool(twin0["passed"] == s0["passed"])
    same_s7 = bool(twin7["kind"] == s7["kind"] and twin7["passed"] == s7["passed"])
    passed = bool(same_s0 and same_s7)
    return {
        "id": "S12",
        "passed": passed,
        "same_s0": same_s0,
        "same_s7": same_s7,
        "primary_s0": s0["passed"],
        "twin_s0": twin0["passed"],
        "primary_s7_kind": s7["kind"],
        "twin_s7_kind": twin7["kind"],
        "twin_s0": {k: twin0[k] for k in ("passed", "pre", "post_ident", "d_w_op", "d_w_q") if k in twin0},
        "twin_s7": {
            k: twin7[k]
            for k in (
                "passed",
                "kind",
                "cosine_teach_vs_probe",
                "live_moved",
                "substitute_moved",
                "projected_moved",
            )
            if k in twin7
        },
        "domain": TWIN_DOMAIN,
    }


def decide(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by = {c["id"]: c for c in cells}
    if not by["S0"]["passed"]:
        code = "plasticity_magnitude_or_readout_mismatch"
        note = "Immediate identical-state failed."
    elif not by["S1"]["passed"]:
        code = "event_state_persistence_boundary"
        note = "Immediate passed; event boundary failed."
    elif not by["S2"]["passed"]:
        code = "delay_state_persistence"
        note = "Boundary-scale transfer failed over 1/2/4/8 neutral ticks."
    elif not by["S3"]["passed"]:
        code = "interference_state_stabilization"
        note = "Boundary passed; distractors failed."
    elif not by["S4"]["passed"]:
        code = "rho_not_reconstructed_from_cortex_s"
        note = "Acquired behavior did not survive ρ reset."
    elif not by["S5"]["passed"]:
        code = "history_conditioned_representation"
        note = "Same cue failed after a different preceding history."
    elif not by["S6"]["passed"]:
        code = "cue_collision_insufficient_separation"
        note = "Two cues did not keep opposite actuator consequences."
    elif by["S7"]["kind"] == "representation_alignment":
        code = "representation_alignment"
        note = "Saved teaching state succeeded at probe; live probe state failed."
    elif not by["S7"]["passed"]:
        code = "teacher_probe_state_mismatch"
        note = "Teacher→probe alignment cell failed without a clean substitute/live split."
    elif by["S8"].get("hold_competition"):
        code = "operation_hold_competition"
        note = "Motor ranking succeeded but ACT remained suppressed."
    elif not by["S8"]["passed"]:
        code = "operation_hold_competition"
        note = "S8 failed: operation state did not carry ACT with the motor preference."
    elif by["S9"]["fast_moved"] and not by["S9"]["rest_moved"]:
        code = "consolidation"
        note = "Fast policy worked; post-REST failed."
    elif not by["S9"]["passed"]:
        code = "consolidation"
        note = "S9 consolidation comparison failed."
    elif not by["S10"]["passed"] and by["S11"]["n_dead"] > 0:
        code = "dead_developmental_genome_surface"
        note = "Age schedules are flat and some Arm D genes have no runtime effect."
    elif not by["S10"]["passed"]:
        code = "flat_maturation_schedule"
        note = "Age genes are wired or default-flat; stages do not change learning parameters."
    elif not by["S11"]["passed"]:
        code = "dead_developmental_genome_surface"
        note = "Perturbing declared Arm D genes did not all produce a runtime effect."
    elif not by["S12"]["passed"]:
        code = "rename_fragile_state_transfer"
        note = "State-transfer conclusions did not survive independent renaming."
    else:
        code = "curriculum_or_unresolved_stochasticity"
        note = "Deterministic state cells passed. Complete-life CHECK failure remains curriculum or stochasticity."
    return {
        "code": code,
        "note": note,
        "two_timescale_hypothesis": code
        in {
            "event_state_persistence_boundary",
            "delay_state_persistence",
            "interference_state_stabilization",
            "history_conditioned_representation",
            "representation_alignment",
            "teacher_probe_state_mismatch",
        },
        "amendment_authorized": False,
        "increase_n": False,
        "another_lineage_run": False,
    }


CELL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_cell_world(0)
    with tempfile.TemporaryDirectory(prefix="sm_smk_") as tmp:
        ag = _fresh(tmp, "s", w)
        t = teach_one(ag, w, w["beneficial"], tag="smk")
        live = live_probe(ag, w, tag="smk_p")
        cos = cosine(t["rho_teach"], live["rho"])
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "clamp_moved_or_recorded": True,
        "d_w_op": t["d_w_op"],
        "cosine_teach_vs_probe": cos,
        "cells": prereg["cells"],
        "domain": CELLS_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if not _git_clean():
        raise RuntimeError("write runner.lock only on a clean tree")
    if not CANDIDATE.exists():
        raise RuntimeError("candidate.v29 required")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.STATEMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "shas": statemap_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "isolation_sha": sha_file(ISOLATION),
        "candidate_v29_sha": sha_file(CANDIDATE),
        "actorcredit_reach_sha": sha_file(REACH_LOCK),
        "n": 64,
        "domain": CELLS_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "cells": prereg["cells"],
        "git_head": _git_head(),
        "note": "Frozen S-cell runner. Score only after this lock is on origin/main. No neural edit. v29 live.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_all() -> dict[str, Any]:
    assert_runner_frozen()
    s0 = run_s0()
    s1 = run_s1()
    s2 = run_s2()
    s3 = run_s3()
    s4 = run_s4()
    s5 = run_s5()
    s6 = run_s6()
    s7 = run_s7()
    s8 = run_s8()
    s9 = run_s9()
    s10 = run_s10()
    s11 = run_s11()
    s12 = run_s12(s0, s7)
    cells = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
    decision = decide(cells)
    out = {
        "version": "TM.0.24.STATEMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "neural_edit": False,
        "n": 64,
        "n_pass": sum(1 for c in cells if c["passed"]),
        "n_cells": len(cells),
        "cells": cells,
        "decision": decision,
        "another_lineage_run": False,
        "q3": False,
        "amendment_authorized": False,
        "increase_n": False,
        "git_head": _git_head(),
        "env": torch_env(),
        "note": "S0–S12 on unused TM024.STATEMAP.CELLS. / TWIN. Diagnosis only. Not 0.0.005.",
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    lines = ["# TM.0.24.STATEMAP results\n"]
    for c in cells:
        lines.append(f"- `{c['id']}`: **{'PASS' if c['passed'] else 'FAIL'}**")
    lines.append(f"\nPrimary diagnosis: `{decision['code']}` — {decision['note']}")
    lines.append("\nn stays 64. Product 0.0.004. Amendment not authorized this package.\n")
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cell", choices=CELL_ORDER)
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
    elif args.score:
        print(json.dumps(run_all(), indent=2, default=str))
    elif args.cell:
        if args.cell == "S12":
            print(json.dumps(run_s12(run_s0(), run_s7()), indent=2, default=str))
        else:
            fn = {
                "S0": run_s0,
                "S1": run_s1,
                "S2": run_s2,
                "S3": run_s3,
                "S4": run_s4,
                "S5": run_s5,
                "S6": run_s6,
                "S7": run_s7,
                "S8": run_s8,
                "S9": run_s9,
                "S10": run_s10,
                "S11": run_s11,
            }[args.cell]
            print(json.dumps(fn(), indent=2, default=str))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
