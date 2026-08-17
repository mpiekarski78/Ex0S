"""TM.0.24.WRITEGEOM — actuator-local plastic-write geometry.

v31 amendment candidate. Not a product earn. Product 0.0.004.
DEV on unused TM024.WRITEGEOM.DEV. worlds.
Scoring requires docs/lineage_writegeom.runner.lock on clean origin/main
after W1 is frozen. Neural edit only after this freeze is on origin/main.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import make_cortex, physics, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, motor_scores, observe_cue
from experiments.run_tm024lineage import make_synthetic_world, opaque_spelling
from experiments.run_tm024motorpersist import POS_DELTA, TEACH_ORDERS
from experiments.run_tm024statemap import teach_one
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_writegeom.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_writegeom_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_writegeom.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_writegeom.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_writegeom.decision.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_writegeom.dev.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024writegeom_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_LIVE = REPO_ROOT / "docs" / "cortex.candidate.lock"
V30_NEURAL = "cc22cf381839049246776d2c223683078f8c13abf00cbd8e99ab2554206538b5"

DEV_DOMAIN = "TM024.WRITEGEOM.DEV."
SCORE_DOMAIN = "TM024.WRITEGEOM.SCORE."
TWIN_DOMAIN = "TM024.WRITEGEOM.TWIN."
REGRESS_DOMAIN = "TM024.WRITEGEOM.REGRESS."


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def thr() -> dict[str, Any]:
    return load_prereg()["margin"]


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def writegeom_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "v31_prereg": REPO_ROOT / "docs" / "cortex_v31.prereg.lock",
        "v31_isolation": REPO_ROOT / "docs" / "cortex_v31.isolation.lock",
        "v31_amendment": REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.lock",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def neural_has_proto() -> bool:
    src = NEURAL.read_text(encoding="utf-8")
    return sha_file(NEURAL) != V30_NEURAL and "actuator_scores" in src and "ACT_SCORE_PROTO" in src


def make_cell_world(index: int, domain: str) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def capacity_world(
    index: int,
    domain: str,
    *,
    n_cues: int,
    n_handles: int,
) -> dict[str, Any]:
    """Fixed-handle overlay: extra cues/handles, identical positive deltas, balanced map."""
    prereg = load_prereg()
    if n_handles > int(prereg["H_max"]):
        raise RuntimeError(f"n_handles {n_handles} exceeds H_max {prereg['H_max']}")
    w = make_cell_world(index, domain)
    rng = np.random.default_rng(domain_seed(domain, f"cap_{index}_{n_cues}_{n_handles}"))
    handles = list(w["handles"])
    symbols = list(w["symbols"])
    while len(handles) < n_handles:
        handles.append(opaque_spelling(rng, "h"))
    while len(symbols) < n_cues:
        symbols.append(opaque_spelling(rng, "s"))
    handles = handles[:n_handles]
    cues = symbols[:n_cues]
    effects = {}
    mapping = []
    for i, cue in enumerate(cues):
        h = handles[i % n_handles]
        effects[h] = {"state": [f"st_p{i % n_handles}"], "delta": list(POS_DELTA)}
        mapping.append({"cue": cue, "handle": h})
    w["handles"] = handles
    w["symbols"] = symbols
    w["cues"] = cues
    w["capacity"] = {"n_cues": n_cues, "n_handles": n_handles}
    w["cue_handle"] = mapping
    w["latent"] = {"act_effects": effects}
    w["beneficial"] = handles[0]
    return w


def enable_w1(ag: NeuralCortex) -> None:
    if not hasattr(ag.genome, "act_score_mode"):
        raise RuntimeError("act_score_mode missing — implement W1 after this freeze is on origin/main")
    ag.genome.act_score_mode = "proto"


def _fresh(tmp: str, tag: str, world: dict[str, Any], *, proto: bool = False) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    if proto:
        enable_w1(ag)
    return ag


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ranking_margin(scores: dict[str, float], winner: str) -> float:
    if winner not in scores:
        return 0.0
    win = float(scores[winner])
    others = [float(v) for k, v in scores.items() if k != winner]
    if not others:
        return win
    return float(win - max(others))


def perturb_stable(
    ag: Any,
    rho: np.ndarray,
    winner: str,
    *,
    domain: str,
    key: str,
) -> dict[str, Any]:
    m = thr()
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = np.asarray(rho, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r0)) + 1e-12
    r_hat = r0 / nrm
    n_ok = 0
    for _i in range(n):
        noise = rng.normal(0.0, sigma, size=r_hat.shape)
        rp = r_hat + noise
        pn = float(np.linalg.norm(rp)) + 1e-12
        unit = rp / pn
        if hasattr(ag, "actuator_scores"):
            scores = ag.actuator_scores(unit)
        elif hasattr(ag, "scores"):
            scores = ag.scores(unit)
        else:
            scores = {}
        if not scores:
            continue
        ranked = max(scores, key=lambda h: scores[h])
        if ranked == winner:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


NEG_DELTA = [-0.35, 0.45, -0.15, 0.0]


class SequentialRLS:
    """Runner-only sequential RLS. Not installed in NeuralCortex."""

    def __init__(self, n: int, handles: list[str], lam: float = 0.01):
        self.handles = list(handles)
        self.h_index = {h: i for i, h in enumerate(self.handles)}
        self.W = np.zeros((len(self.handles), int(n)), dtype=np.float64)
        self.P = np.eye(int(n), dtype=np.float64) / float(lam)
        self.lam = float(lam)

    def update(self, rho: np.ndarray, handle: str, adv: float) -> None:
        if adv == 0.0 or handle not in self.h_index:
            return
        x = np.asarray(rho, dtype=np.float64).reshape(-1)
        xn = float(np.linalg.norm(x))
        if not np.isfinite(xn) or xn <= 1e-12:
            return
        x = x / xn
        y = np.zeros(len(self.handles), dtype=np.float64)
        y[self.h_index[handle]] = 1.0 if adv > 0.0 else -1.0
        px = self.P @ x
        denom = 1.0 + float(x @ px)
        k = px / denom
        err = y - self.W @ x
        self.W = self.W + np.outer(err, k)
        self.P = self.P - np.outer(k, px)

    def scores(self, rho: np.ndarray) -> dict[str, float]:
        x = np.asarray(rho, dtype=np.float64).reshape(-1)
        xn = float(np.linalg.norm(x)) + 1e-12
        x = x / xn
        out: dict[str, float] = {}
        for h, i in self.h_index.items():
            row = self.W[i]
            rn = float(np.linalg.norm(row))
            out[h] = 0.0 if rn <= 1e-12 else float(np.dot(row, x) / rn)
        return out

    def diagnostics(self) -> dict[str, float]:
        s = np.linalg.svd(self.W, compute_uv=False)
        rel = float(s.max()) * 1e-9 if s.size else 0.0
        rank = int(np.sum(s > max(rel, 1e-12)))
        cond = float(np.linalg.cond(self.P)) if self.P.size else float("inf")
        return {
            "effective_rank": float(rank),
            "cond_P": cond,
            "weight_norm": float(np.linalg.norm(self.W)),
        }


def set_handle_delta(world: dict[str, Any], handle: str, delta: list[float]) -> dict[str, Any]:
    w = copy.deepcopy(world)
    effects = dict(w["latent"]["act_effects"])
    prev = dict(effects.get(handle) or {"state": ["st_p"], "delta": list(POS_DELTA)})
    prev["delta"] = list(delta)
    effects[handle] = prev
    w["latent"] = {"act_effects": effects}
    return w


def mapping_pairs(world: dict[str, Any], *, flip: bool) -> list[tuple[str, str]]:
    pairs = [(str(m["cue"]), str(m["handle"])) for m in world["cue_handle"]]
    if flip:
        handles = [h for _c, h in pairs]
        rev = list(reversed(handles))
        pairs = [(c, rev[i % len(rev)]) for i, (c, _h) in enumerate(pairs)]
    return pairs


def live_rank(ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str) -> dict[str, Any]:
    from experiments.run_tm024wallmap import op_logits

    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    scores = probe.actuator_scores(probe.rho)
    winner = max(scores, key=lambda h: scores[h]) if scores else None
    margin = ranking_margin(scores, winner) if winner else 0.0
    logits = op_logits(probe)
    from three_memory.neural_cortex import OPS

    act_logit = float(logits[OPS.index("ACT")])
    hold_logit = float(logits[OPS.index("HOLD")])
    action = probe.last_action or {}
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "winner": winner,
        "margin": float(margin),
        "act_logit": act_logit,
        "hold_logit": hold_logit,
        "prefer_act": bool(act_logit > hold_logit),
        "live_op": action.get("op"),
        "live_token": action.get("token"),
        "rho": probe._from_t(probe.rho).copy(),
    }


def w2_rank(rls: SequentialRLS, rho: np.ndarray) -> dict[str, Any]:
    scores = rls.scores(rho)
    winner = max(scores, key=lambda h: scores[h]) if scores else None
    margin = ranking_margin(scores, winner) if winner else 0.0
    return {"scores": {k: float(v) for k, v in scores.items()}, "winner": winner, "margin": float(margin)}


def pass_margin(margin: float, stable: bool) -> bool:
    m = thr()
    return bool(margin >= float(m["cosine_margin_min"]) and stable)


def teach_stream(
    ag: NeuralCortex,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    tag: str,
    rls: SequentialRLS | None,
) -> tuple[list[dict[str, Any]], float]:
    taught = []
    wq0 = ag.W_act_query.detach().clone()
    for i, (cue, handle) in enumerate(pairs):
        t = teach_one(ag, world, handle, tag=f"{tag}_{i}", symbols=[cue])
        rho = np.asarray(t["rho_teach"], dtype=np.float64).copy()
        if rls is not None:
            rls.update(rho, handle, float(t["adv"]))
        taught.append({"cue": cue, "handle": handle, "adv": float(t["adv"]), "moved": bool(t["moved"])})
    d_wq = float((ag.W_act_query - wq0).abs().max().item())
    return taught, d_wq


def eval_capacity_cell(
    *,
    arm: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    proto = arm == "W1"
    rls = SequentialRLS(64, list(world["handles"]), lam=float(load_prereg()["w2"]["lambda"])) if arm == "W2" else None
    with tempfile.TemporaryDirectory(prefix="wg_cap_") as tmp:
        ag = _fresh(tmp, "s", world, proto=proto)
        taught, d_wq = teach_stream(ag, world, seq, tag=tag, rls=rls)
        probes = []
        all_ok = True
        ranking_ok = True
        for i, (cue, handle) in enumerate(pairs):
            live = live_rank(ag, world, cue, tag=f"{tag}_p{i}")
            if arm == "W2":
                ranked = w2_rank(rls, live["rho"])
                winner, margin = ranked["winner"], ranked["margin"]
                scores_for_perturb = rls
            else:
                winner, margin = live["winner"], live["margin"]
                scores_for_perturb = ag
            stab = perturb_stable(scores_for_perturb, live["rho"], winner or "", domain=world["domain"], key=f"{tag}_{cue}")
            ok = bool(winner == handle and pass_margin(margin, bool(stab["stable"])))
            all_ok = all_ok and ok
            ranking_ok = ranking_ok and bool(winner == handle)
            probes.append(
                {
                    "cue": cue,
                    "want": handle,
                    "winner": winner,
                    "margin": margin,
                    "perturb_stable": bool(stab["stable"]),
                    "ok": ok,
                    "live_op": live["live_op"],
                    "prefer_act": live["prefer_act"],
                }
            )
    w2d = rls.diagnostics() if rls is not None else {}
    return {
        "arm": arm,
        "order": order,
        "taught": taught,
        "probes": probes,
        "passed": all_ok,
        "ranking_ok": ranking_ok,
        "d_w_act_query": d_wq,
        "w2": w2d,
        "n_cues": world["capacity"]["n_cues"],
        "n_handles": world["capacity"]["n_handles"],
    }


def eval_ecological(arm: str, world: dict[str, Any], *, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1 = world["handles"][0]
    h2 = world["handles"][1]
    proto = arm == "W1"
    with tempfile.TemporaryDirectory(prefix="wg_eco_") as tmp:
        ag = _fresh(tmp, "s", world, proto=proto)
        t1 = teach_one(ag, world, h1, tag=f"{tag}_p", symbols=[cue])
        wneg = set_handle_delta(world, h1, NEG_DELTA)
        t2 = teach_one(ag, wneg, h1, tag=f"{tag}_n", symbols=[cue])
        t3 = teach_one(ag, world, h2, tag=f"{tag}_r", symbols=[cue])
        live = live_rank(ag, world, cue, tag=f"{tag}_q")
        stab = perturb_stable(ag, live["rho"], live["winner"] or "", domain=world["domain"], key=f"{tag}_eco")
        ok = bool(
            t1["adv"] > 0.0
            and t2["adv"] < 0.0
            and t3["adv"] > 0.0
            and live["winner"] == h2
            and pass_margin(live["margin"], bool(stab["stable"]))
        )
    return {
        "arm": arm,
        "id": "ecological_reversal",
        "passed": ok if arm == "W1" else bool(live["winner"] == h2),
        "required": arm == "W1",
        "adv": [float(t1["adv"]), float(t2["adv"]), float(t3["adv"])],
        "winner": live["winner"],
        "want": h2,
        "margin": live["margin"],
        "perturb_stable": bool(stab["stable"]),
        "ranking": live["scores"],
        "prefer_act": live["prefer_act"],
        "live_op": live["live_op"],
        "live_token": live["live_token"],
    }


def eval_positive_only(arm: str, world: dict[str, Any], *, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1 = world["handles"][0]
    h2 = world["handles"][1]
    proto = arm == "W1"
    with tempfile.TemporaryDirectory(prefix="wg_po_") as tmp:
        ag = _fresh(tmp, "s", world, proto=proto)
        teach_one(ag, world, h1, tag=f"{tag}_1", symbols=[cue])
        teach_one(ag, world, h2, tag=f"{tag}_2", symbols=[cue])
        live = live_rank(ag, world, cue, tag=f"{tag}_q")
    return {
        "arm": arm,
        "id": "positive_only_reassignment",
        "required": False,
        "diagnostic": True,
        "winner": live["winner"],
        "want": h2,
        "margin": live["margin"],
        "scores": live["scores"],
        "both_match": bool(live["scores"].get(h1, 0) > 0 and live["scores"].get(h2, 0) > 0),
        "passed_if_required": bool(live["winner"] == h2),
    }


def eval_integrity(world: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="wg_int_") as tmp:
        ag = _fresh(tmp, "birth", world, proto=True)
        birth_zero = all(float(np.linalg.norm(v)) == 0.0 for v in ag._proto_fast.values())
        innate = live_rank(ag, world, world["cue_handle"][0]["cue"], tag="innate")
        innate_ok = bool(innate["margin"] < 1e-9)
        cue, handle = world["cue_handle"][0]["cue"], world["cue_handle"][0]["handle"]
        wq0 = ag.W_act_query.detach().clone()
        teach_one(ag, world, handle, tag="int_t", symbols=[cue])
        d_wq = float((ag.W_act_query - wq0).abs().max().item())
        snap = ag.checkpoint()
        twin = make_cortex(Path(tmp) / "load", device="cpu")
        twin.load_checkpoint(snap)
        enable_w1(twin)
        live = live_rank(twin, world, cue, tag="int_load")
        # permutation of bind order
        hlist = list(world["handles"])
        ag.bind_actuators(list(reversed(hlist)))
        live2 = live_rank(ag, world, cue, tag="int_perm")
        perm_ok = live2["winner"] == live["winner"]
        # missing proto migration
        bare = dict(snap)
        bare.pop("proto_fast", None)
        bare.pop("proto_slow", None)
        mig = make_cortex(Path(tmp) / "mig", device="cpu")
        mig.load_checkpoint(bare)
        zeros = all(float(np.linalg.norm(v)) == 0.0 for v in mig._proto_fast.values())
    return {
        "id": "integrity",
        "passed": bool(birth_zero and innate_ok and d_wq == 0.0 and live["winner"] == handle and perm_ok and zeros),
        "birth_zero": birth_zero,
        "innate_ok": innate_ok,
        "w_act_query_delta": d_wq,
        "checkpoint_winner": live["winner"],
        "perm_ok": perm_ok,
        "old_ckpt_zeros": zeros,
    }


def run_dev() -> dict[str, Any]:
    refuse_dev()
    prereg = load_prereg()
    cells: list[dict[str, Any]] = []
    for spec in prereg["capacity"]:
        n_cues, n_handles = int(spec["n_cues"]), int(spec["n_handles"])
        required = bool(spec["required"])
        worlds_n = 1 if not required else 2
        orders = list(TEACH_ORDERS) if required else ["A_then_B"]
        flips = (False, True) if required else (False,)
        for idx in range(worlds_n):
            world = capacity_world(idx, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            for flip in flips:
                pairs = mapping_pairs(world, flip=flip)
                for order in orders:
                    for arm in ("W0", "W1", "W2"):
                        tag = f"c{n_cues}h{n_handles}_{idx}_{int(flip)}_{order}_{arm}"
                        cell = eval_capacity_cell(arm=arm, world=world, pairs=pairs, order=order, tag=tag)
                        cell["required"] = required
                        cell["flip"] = bool(flip)
                        cell["purpose"] = spec["purpose"]
                        cells.append(cell)
    world2 = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    for arm in ("W0", "W1", "W2"):
        cells.append(eval_ecological(arm, world2, tag=f"eco_{arm}"))
        cells.append(eval_positive_only(arm, world2, tag=f"po_{arm}"))
    integ = eval_integrity(world2)
    cells.append(integ)

    def _arm_cap(arm: str, required_only: bool = True) -> list[dict[str, Any]]:
        out = []
        for c in cells:
            if c.get("arm") != arm:
                continue
            if "n_cues" not in c:
                continue
            if required_only and not c.get("required", True):
                continue
            out.append(c)
        return out

    w1_req = _arm_cap("W1", True)
    w0_req = _arm_cap("W0", True)
    w1_2cue = [c for c in w1_req if c.get("n_cues") == 2 and c.get("n_handles") == 2]
    w1_2cue_pass = bool(w1_2cue) and all(c["passed"] for c in w1_2cue)
    w1_2cue_rank = bool(w1_2cue) and all(c.get("ranking_ok") for c in w1_2cue)
    w1_req_pass = bool(w1_req) and all(c["passed"] for c in w1_req)
    w0_fail_opposing = bool(w0_req) and not any(c["passed"] for c in w0_req if c.get("n_cues") == 2)
    eco = [c for c in cells if c.get("id") == "ecological_reversal"]
    eco_w1 = next(c for c in eco if c["arm"] == "W1")
    pos = [c for c in cells if c.get("id") == "positive_only_reassignment"]
    w1_v31 = bool(w1_2cue_pass and eco_w1["passed"] and integ["passed"] and w1_req_pass)
    out = {
        "version": "TM.0.24.WRITEGEOM.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "domain": DEV_DOMAIN,
        "score_domain_opened": False,
        "w1_2cue_opposing": w1_2cue_pass,
        "w1_2cue_ranking": w1_2cue_rank,
        "w1_required_capacity": w1_req_pass,
        "w0_2cue_fails": w0_fail_opposing,
        "ecological_w1": eco_w1["passed"],
        "integrity": integ["passed"],
        "w1_v31_eligible": w1_v31,
        "positive_only": [{k: c[k] for k in ("arm", "winner", "want", "margin", "passed_if_required", "both_match") if k in c} for c in pos],
        "n_cells": len(cells),
        "cells": cells,
        "note": "DEV only. SCORE unopened. 8-cue/8-handle calibration is not sufficient evidence.",
        "env": torch_env(),
        "git_head": _git_head(),
    }
    RESULT_MD.write_text(
        "# TM.0.24.WRITEGEOM DEV\n\n"
        f"W1 2-cue ranking (winner=handle): **{w1_2cue_rank}**. "
        f"W1 2-cue opposing with frozen margin/perturb: **{w1_2cue_pass}**. "
        f"Required capacity: **{w1_req_pass}**. "
        f"Ecological reversal: **{eco_w1['passed']}**. Integrity: **{integ['passed']}**. "
        f"W0 2-cue fails (regression): **{w0_fail_opposing}**. "
        f"v31 eligible: **{w1_v31}**.\n\n"
        "W1 can rank opposing 2-cue maps, but cosine margins stay crumbs (~3e-4) below the frozen 0.01 floor, "
        "a single negative teach cannot flip a unit prototype, and ecological reversal fails. "
        "Positive-only reassignment is diagnostic, not a required pass. SCORE unopened. "
        "No `cortex.candidate.v31.lock`. Next: compact connection-local eligibility. "
        "Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def _compact_cell(c: dict[str, Any]) -> dict[str, Any]:
    keep = dict(c)
    if "taught" in keep:
        keep["taught_adv"] = [float(t.get("adv", 0.0)) for t in keep.pop("taught") or []]
    if "probes" in keep:
        keep["probes"] = [
            {k: p[k] for k in ("cue", "want", "winner", "margin", "perturb_stable", "ok", "live_op", "prefer_act") if k in p}
            for p in keep["probes"]
        ]
    return keep


def write_dev_lock(dev: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dev or run_dev()
    compact = dict(out)
    compact["cells"] = [_compact_cell(c) for c in out.get("cells") or []]
    if DEV_LOCK.exists():
        raise RuntimeError("writegeom.dev.lock exists")
    DEV_LOCK.write_text(json.dumps(compact, indent=2, default=str) + "\n", encoding="utf-8")
    return compact


def write_decision(dev: dict[str, Any] | None = None) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("writegeom.decision.lock exists")
    if dev is None:
        if DEV_LOCK.exists():
            dev = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        else:
            dev = write_dev_lock()
    if dev.get("w1_v31_eligible"):
        code = "actuator_local_prototype_succeeds"
        next_step = "freeze_v31_then_A0A11_C4C6_SCORE_once_lineage_still_closed"
        passed = True
    elif dev.get("w1_2cue_ranking") and not dev.get("w1_2cue_opposing"):
        code = "w1_ranking_crumb_margin_ecological_fail"
        next_step = "compact_connection_local_eligibility_inside_n64"
        passed = False
    else:
        code = "w1_fails_2cue_opposing"
        next_step = "compact_connection_local_eligibility_inside_n64"
        passed = False
    out = {
        "version": "TM.0.24.WRITEGEOM.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "H_max": 8,
        "state_budget": 1024,
        "scored_worlds": False,
        "w1_v31_eligible": bool(dev.get("w1_v31_eligible")),
        "w1_2cue_ranking": bool(dev.get("w1_2cue_ranking")),
        "w1_2cue_opposing": bool(dev.get("w1_2cue_opposing")),
        "ecological_w1": bool(dev.get("ecological_w1")),
        "integrity": bool(dev.get("integrity")),
        "candidate_v31": False,
        "lineage_reopened": False,
        "q3": False,
        "decision": {"code": code, "next": next_step, "passed": passed},
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": torch_env(),
        "git_head": _git_head(),
        "note": "W1 ranking can oppose two cues but fails the frozen 0.01 margin/perturbation gate and ecological reversal. SCORE unopened. No v31 candidate. Lineage stays closed. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_cell_world(0, DEV_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="wg_smk_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(w["handles"]))
        t = teach_one(ag, w, w["beneficial"], tag="smk")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "H_max": prereg["H_max"],
        "state_budget": prereg["state_budget"],
        "cosine_margin_min": prereg["margin"]["cosine_margin_min"],
        "d_w_op": t["d_w_op"],
        "neural_has_proto": neural_has_proto(),
        "env": torch_env(),
    }


def refuse_dev() -> None:
    if not neural_has_proto():
        raise RuntimeError("WRITEGEOM DEV requires W1 neural law after this freeze is on origin/main")


def refuse_score() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no writegeom runner.lock — refuse cell scoring")
    if SCORE_DOMAIN != load_prereg()["domains"]["SCORE"]:
        raise RuntimeError("SCORE domain drifted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--write-dev-lock", action="store_true")
    ap.add_argument("--write-decision", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.verify_prereg:
        p = load_prereg()
        assert p["n"] == 64
        assert p["H_max"] == 8
        assert p["state_budget"] == 2 * 8 * 64
        assert p["margin"]["cosine_margin_min"] == 0.01
        assert p["arms"]["W2"]["lambda"] == 0.01
        assert p["reversal"]["ecological"]["required_w1_pass"] is True
        assert p["reversal"]["positive_only_reassignment"]["required_w1_pass"] is False
        print(json.dumps({"ok": True, "product": p["product"], "H_max": p["H_max"]}, indent=2))
    elif args.dev:
        out = run_dev()
        brief = {k: v for k, v in out.items() if k != "cells"}
        print(json.dumps(brief, indent=2, default=str))
        print(json.dumps(
            [
                {
                    "arm": c.get("arm"),
                    "id": c.get("id"),
                    "n_cues": c.get("n_cues"),
                    "n_handles": c.get("n_handles"),
                    "passed": c.get("passed"),
                    "purpose": c.get("purpose"),
                }
                for c in out["cells"]
            ],
            indent=2,
        ))
    elif args.write_dev_lock:
        print(json.dumps({k: v for k, v in write_dev_lock().items() if k != "cells"}, indent=2, default=str))
    elif args.write_decision:
        print(json.dumps(write_decision(), indent=2, default=str))
    elif args.score:
        refuse_score()
        raise RuntimeError("SCORE opens only after runner.lock and candidate hash on origin/main")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
