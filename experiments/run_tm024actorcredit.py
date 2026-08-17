"""TM.0.24.ACTORCREDIT — A0–A11 action-owned delayed credit cells.

Not a lineage version. Not a capability earn. Product 0.0.004.
Scoring requires docs/lineage_actorcredit.runner.lock on clean origin/main.
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
import torch

from experiments.cortex_mact_boundary import control_c4_v6
from experiments.run_tm023cortex import build_observe, make_cortex, physics, torch_env
from experiments.run_tm024lineage import make_synthetic_world
from experiments.run_tm024wallmap import motor_scores, op_logits, softmax_np
from three_memory.cortex_lineage import freeze_plasticity, sha_file
from three_memory.neural_cortex import ELIG_EPS, OPS, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_actorcredit.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_actorcredit_contract.md"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_actorcredit.runner.lock"
RESULT = REPO_ROOT / "docs" / "lineage_actorcredit.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024actorcredit_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v29.lock"
V28_NEURAL = "0a4014ce91bf08b69693924ee645bdc912ae4c6e0a9b6529bda6a6fe8a281892"

MID_BODY = [0.5, 0.4, 0.5, 0.0]
ACTOR = ("W_op", "W_act_query")
CELLS_DOMAIN = "TM024.ACTORCREDIT.CELLS."


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def make_cell_world(index: int) -> dict[str, Any]:
    seed = domain_seed(CELLS_DOMAIN, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = CELLS_DOMAIN
    w["diag_index"] = int(index)
    return w


def actorcredit_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "candidate_v29": CANDIDATE,
        "cortex_lineage": REPO_ROOT / "three_memory" / "cortex_lineage.py",
        "v29_pipeline": REPO_ROOT / "experiments" / "cortex_v29_pipeline.py",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no actorcredit runner.lock — refuse cell scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if actorcredit_shas() != lock.get("shas"):
        raise RuntimeError("actorcredit implementation drifted after runner.lock")
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if sha_file(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from v29 candidate")
    if cand.get("genome", {}).get("n") != 64:
        raise RuntimeError("n must stay 64")
    return lock


def harmful_handle(world: dict[str, Any]) -> str:
    for h in world["handles"][:2]:
        if h != world["beneficial"]:
            return str(h)
    return str(world["handles"][1])


def observe_cue(
    ag: NeuralCortex,
    world: dict[str, Any],
    *,
    tag: str,
    body: list[float],
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    pair = list(world["teacher_pair"])
    return ag.observe(
        build_observe(
            interaction_token=tag,
            source_token="src_ac",
            ordered_symbols=list(symbols if symbols is not None else [pair[0]]),
            observable_state=["st_idle"],
            body_state=list(body),
        )
    )


def p_act(ag: NeuralCortex) -> float:
    return float(softmax_np(op_logits(ag))[OPS.index("ACT")])


def p_handle(ag: NeuralCortex, tok: str) -> float:
    scores = motor_scores(ag)
    keys = list(scores)
    arr = np.asarray([scores[k] for k in keys], dtype=np.float64)
    sm = softmax_np(arr)
    return float(sm[keys.index(tok)]) if tok in keys else 0.0


def actor_snap(ag: NeuralCortex) -> dict[str, torch.Tensor]:
    names = list(ag._plastic_names)
    return {
        "W": {n: getattr(ag, n).detach().clone() for n in names},
        "slow": {n: ag.W_slow[n].detach().clone() for n in names},
    }


def max_named(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], names: tuple[str, ...]) -> float:
    return max(float((a[n] - b[n]).abs().max().item()) for n in names)


def clamp_cycle(ag: NeuralCortex, world: dict[str, Any], tok: str, *, tag: str) -> dict[str, Any]:
    body = list(MID_BODY)
    observe_cue(ag, world, tag=f"{tag}_sel", body=body)
    clamped = ag.clamp_action("ACT", tok)
    _, body2 = physics(body, tok, world["latent"])
    out = observe_cue(ag, world, tag=f"{tag}_obs", body=body2)
    return {"clamp": clamped, "observe": out, "adv": float((out.get("metrics") or {}).get("adv") or 0.0)}


def beneficial_exposure(ag: NeuralCortex, world: dict[str, Any], *, n: int, prefix: str) -> None:
    ben = world["beneficial"]
    for i in range(int(n)):
        clamp_cycle(ag, world, ben, tag=f"{prefix}{i}")


def balanced_exposure(ag: NeuralCortex, world: dict[str, Any], *, n: int, prefix: str) -> None:
    ben = world["beneficial"]
    harm = harmful_handle(world)
    for i in range(int(n)):
        clamp_cycle(ag, world, ben, tag=f"{prefix}b{i}")
        clamp_cycle(ag, world, harm, tag=f"{prefix}h{i}")


def prep_eval(ag: NeuralCortex) -> None:
    """Do not credit a leftover actor trace or refractory residue while measuring policy."""
    ag.drop_actor_pending()
    ag._pred_pending = None
    ag._vocal_next = None
    ag._last_motor_class = None
    ag._hold_after_conflict = False


def clone_frozen(ag: NeuralCortex) -> NeuralCortex:
    snap = ag.checkpoint()
    twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device=str(ag.device))
    twin.load_checkpoint(snap)
    freeze_plasticity(twin)
    prep_eval(twin)
    return twin


def probe_policy(ag: NeuralCortex, world: dict[str, Any], *, n_probe: int, prefix: str) -> dict[str, Any]:
    ben = world["beneficial"]
    harm = harmful_handle(world)
    counts = {ben: 0, harm: 0, "HOLD": 0, "other": 0, "ACT": 0}
    body = list(MID_BODY)
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=f"{prefix}_rho", body=body)
    p0 = p_act(probe)
    ph = p_handle(probe, ben)
    scores = motor_scores(probe)
    for i in range(int(n_probe)):
        out = observe_cue(probe, world, tag=f"{prefix}{i}", body=body)
        act = out.get("action") or {}
        op = act.get("op")
        tok = act.get("token")
        if op == "HOLD":
            counts["HOLD"] += 1
        elif op == "ACT":
            counts["ACT"] += 1
            if tok == ben:
                counts[ben] += 1
            elif tok == harm:
                counts[harm] += 1
            else:
                counts["other"] += 1
        else:
            counts["other"] += 1
    return {
        "counts": counts,
        "p_act": p0,
        "p_ben_given_act": ph,
        "motor_ben": float(scores.get(ben, 0.0)),
        "motor_harm": float(scores.get(harm, 0.0)),
        "sampled_ben": int(counts[ben]),
        "sampled_act": int(counts["ACT"]),
        "sampled_hold": int(counts["HOLD"]),
    }


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    return ag


def run_a0() -> dict[str, Any]:
    world = make_cell_world(0)
    with tempfile.TemporaryDirectory(prefix="ac_a0_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="warm", body=list(MID_BODY))
        before = actor_snap(ag)
        rho0 = np.zeros(ag.genome.n, dtype=np.float64)
        ag._pending = {
            "op": "ACT",
            "token": world["beneficial"],
            "rho_elig": rho0,
            "rho_op": rho0.copy(),
            "rho_motor": rho0.copy(),
            "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
            "body": np.asarray(MID_BODY, dtype=np.float64),
            "cost": 0.05,
            "motor_vec": ag.motor_vocab[world["beneficial"]].copy(),
            "authored": True,
            "clamped": False,
            "t": int(ag._t),
            "interaction_token": "a0",
        }
        _, body2 = physics(MID_BODY, world["beneficial"], world["latent"])
        ag._apply_credit(np.zeros(ag.genome.d_sym, dtype=np.float64), np.asarray(body2, dtype=np.float64))
        after = actor_snap(ag)
        d_w = max_named(before["W"], after["W"], ACTOR)
        d_s = max_named(before["slow"], after["slow"], ACTOR)
        passed = bool(d_w < 1e-12 and d_s < 1e-12)
    return {"id": "A0", "passed": passed, "max_fast": d_w, "max_slow": d_s, "elig_eps": ELIG_EPS}


def run_a1() -> dict[str, Any]:
    world = make_cell_world(1)
    with tempfile.TemporaryDirectory(prefix="ac_a1_") as tmp:
        ag_c = _fresh(tmp, "c", world)
        observe_cue(ag_c, world, tag="c0", body=list(MID_BODY))
        before_c = {n: getattr(ag_c, n).detach().clone() for n in ACTOR}
        clamp_cycle(ag_c, world, world["beneficial"], tag="c")
        d_c = max(float((getattr(ag_c, n) - before_c[n]).abs().max().item()) for n in ACTOR)

        ag_p = _fresh(tmp, "p", world)
        observe_cue(ag_p, world, tag="p0", body=list(MID_BODY))
        ag_p.drop_actor_pending()
        before_p = {n: getattr(ag_p, n).detach().clone() for n in ACTOR}
        _, body2 = physics(MID_BODY, world["beneficial"], world["latent"])
        observe_cue(ag_p, world, tag="p1", body=body2)
        d_p = max(float((getattr(ag_p, n) - before_p[n]).abs().max().item()) for n in ACTOR)
        passed = bool(d_c > 1e-12 and d_p < 1e-12)
    return {
        "id": "A1",
        "passed": passed,
        "clamped_delta": d_c,
        "passive_delta": d_p,
    }


def run_a2() -> dict[str, Any]:
    world = make_cell_world(2)
    with tempfile.TemporaryDirectory(prefix="ac_a2_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="sel", body=list(MID_BODY))
        saved = np.zeros(ag.genome.n, dtype=np.float64)
        saved[0] = 1.0
        current = np.zeros(ag.genome.n, dtype=np.float64)
        current[-1] = 1.0
        ag._pending = {
            "op": "ACT",
            "token": world["beneficial"],
            "rho_elig": saved.copy(),
            "rho_op": saved.copy(),
            "rho_motor": saved.copy(),
            "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
            "body": np.asarray(MID_BODY, dtype=np.float64),
            "cost": 0.05,
            "motor_vec": ag.motor_vocab[world["beneficial"]].copy(),
            "authored": True,
            "clamped": False,
            "t": int(ag._t),
            "interaction_token": "a2",
        }
        ag.rho = ag._to_t(current)
        w_before = ag.W_act_query.detach().clone()
        w_op_before = ag.W_op.detach().clone()
        _, body2 = physics(MID_BODY, world["beneficial"], world["latent"])
        ag._apply_credit(np.zeros(ag.genome.d_sym, dtype=np.float64), np.asarray(body2, dtype=np.float64))
        d_q = ag.W_act_query - w_before
        tok_v = ag._to_t(ag.motor_vocab[world["beneficial"]])
        score_saved = float((d_q * torch.outer(tok_v, ag._to_t(saved))).sum().item())
        score_current = float((d_q * torch.outer(tok_v, ag._to_t(current))).sum().item())
        d_op = ag.W_op - w_op_before
        e_act = torch.zeros(len(OPS), dtype=ag.dtype, device=ag.device)
        e_act[OPS.index("ACT")] = 1.0
        op_saved = float((d_op * torch.outer(e_act, ag._to_t(saved))).sum().item())
        op_current = float((d_op * torch.outer(e_act, ag._to_t(current))).sum().item())
        passed = bool(
            score_saved > score_current + 1e-12
            and op_saved > op_current + 1e-12
            and score_saved > 0.0
            and op_saved > 0.0
        )
    return {
        "id": "A2",
        "passed": passed,
        "motor_saved": score_saved,
        "motor_current": score_current,
        "op_saved": op_saved,
        "op_current": op_current,
    }


def run_a3() -> dict[str, Any]:
    world = make_cell_world(3)
    with tempfile.TemporaryDirectory(prefix="ac_a3_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="sel", body=list(MID_BODY))
        clamp_cycle(ag, world, world["beneficial"], tag="once")
        ag.drop_actor_pending()
        after_one = {n: getattr(ag, n).detach().clone() for n in ACTOR}
        pending_gone = ag._pending is None
        _, body2 = physics(MID_BODY, world["beneficial"], world["latent"])
        observe_cue(ag, world, tag="repeat", body=body2)
        d_rep = max(float((getattr(ag, n) - after_one[n]).abs().max().item()) for n in ACTOR)
        passed = bool(pending_gone and d_rep < 1e-12)
    return {"id": "A3", "passed": passed, "repeat_delta": d_rep, "consumed": pending_gone}


def run_a4() -> dict[str, Any]:
    world = make_cell_world(4)
    ben = world["beneficial"]
    harm = harmful_handle(world)
    with tempfile.TemporaryDirectory(prefix="ac_a4_") as tmp:
        ag_b = _fresh(tmp, "b", world)
        observe_cue(ag_b, world, tag="b0", body=list(MID_BODY))
        p_b0 = p_act(ag_b)
        ph_b0 = p_handle(ag_b, ben)
        beneficial_exposure(ag_b, world, n=8, prefix="b")
        ag_b.drop_actor_pending()
        observe_cue(ag_b, world, tag="b1", body=list(MID_BODY))
        p_b1 = p_act(ag_b)
        ph_b1 = p_handle(ag_b, ben)

        ag_h = _fresh(tmp, "h", world)
        observe_cue(ag_h, world, tag="h0", body=list(MID_BODY))
        p_h0 = p_act(ag_h)
        pharm_h0 = p_handle(ag_h, harm)
        for i in range(8):
            clamp_cycle(ag_h, world, harm, tag=f"h{i}")
        ag_h.drop_actor_pending()
        observe_cue(ag_h, world, tag="h1", body=list(MID_BODY))
        p_h1 = p_act(ag_h)
        pharm_h1 = p_handle(ag_h, harm)
        passed = bool(p_b1 > p_b0 and ph_b1 > ph_b0 and p_h1 < p_h0 and pharm_h1 < pharm_h0)
    return {
        "id": "A4",
        "passed": passed,
        "ben": {"p_act": [p_b0, p_b1], "p_handle": [ph_b0, ph_b1]},
        "harm": {"p_act": [p_h0, p_h1], "p_handle_harm": [pharm_h0, pharm_h1]},
    }


def run_a5() -> dict[str, Any]:
    world = make_cell_world(5)
    ben = world["beneficial"]
    other = world["handles"][2]
    with tempfile.TemporaryDirectory(prefix="ac_a5_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="w", body=list(MID_BODY))
        s0 = motor_scores(ag)
        beneficial_exposure(ag, world, n=8, prefix="x")
        ag.drop_actor_pending()
        observe_cue(ag, world, tag="w2", body=list(MID_BODY))
        s1 = motor_scores(ag)
        d_ben = float(s1.get(ben, 0.0) - s0.get(ben, 0.0))
        d_other = float(s1.get(other, 0.0) - s0.get(other, 0.0))
        passed = bool(d_ben > 1e-9 and d_ben > d_other + 1e-9)
    return {"id": "A5", "passed": passed, "delta_chosen": d_ben, "delta_unrelated": d_other}


def run_a6() -> dict[str, Any]:
    world = make_cell_world(6)
    with tempfile.TemporaryDirectory(prefix="ac_a6_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="w", body=list(MID_BODY))
        p0 = p_act(ag)
        beneficial_exposure(ag, world, n=16, prefix="e")
        prep_eval(ag)
        observe_cue(ag, world, tag="w2", body=list(MID_BODY))
        p1 = p_act(ag)
        passed = bool(p1 > p0)
    return {"id": "A6", "passed": passed, "p_act_before": p0, "p_act_after": p1}


def run_a7() -> dict[str, Any]:
    world = make_cell_world(7)
    ben = world["beneficial"]
    harm = harmful_handle(world)
    with tempfile.TemporaryDirectory(prefix="ac_a7_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="w", body=list(MID_BODY))
        p0 = p_handle(ag, ben)
        beneficial_exposure(ag, world, n=16, prefix="e")
        prep_eval(ag)
        observe_cue(ag, world, tag="w2", body=list(MID_BODY))
        p1 = p_handle(ag, ben)
        scores = motor_scores(ag)
        passed = bool(p1 > p0 and scores.get(ben, 0.0) > scores.get(harm, 0.0))
    return {
        "id": "A7",
        "passed": passed,
        "p_handle_before": p0,
        "p_handle_after": p1,
        "motor_ben": float(scores.get(ben, 0.0)),
        "motor_harm": float(scores.get(harm, 0.0)),
    }


def run_a8() -> dict[str, Any]:
    world = make_cell_world(8)
    with tempfile.TemporaryDirectory(prefix="ac_a8_") as tmp:
        ag = _fresh(tmp, "s", world)
        observe_cue(ag, world, tag="w", body=list(MID_BODY))
        pre = probe_policy(ag, world, n_probe=20, prefix="pre")
        beneficial_exposure(ag, world, n=24, prefix="e")
        post = probe_policy(ag, world, n_probe=20, prefix="post")
        passed = bool(
            post["p_act"] > pre["p_act"]
            and post["p_ben_given_act"] > pre["p_ben_given_act"]
            and post["sampled_ben"] > pre["sampled_ben"]
            and post["sampled_act"] > 0
        )
    return {"id": "A8", "passed": passed, "pre": pre, "post": post}


def run_a9() -> dict[str, Any]:
    world = make_cell_world(9)
    sib = make_cell_world(10)
    with tempfile.TemporaryDirectory(prefix="ac_a9_") as tmp:
        ag = _fresh(tmp, "a", world)
        beneficial_exposure(ag, world, n=16, prefix="a")
        prep_eval(ag)
        observe_cue(ag, world, tag="aw", body=list(MID_BODY))
        rename_ok = motor_scores(ag).get(world["beneficial"], -1.0) > motor_scores(ag).get(
            harmful_handle(world), 0.0
        )

        ag_b = _fresh(tmp, "b", world)
        ag_b.bind_actuators(list(reversed(world["handles"])))
        beneficial_exposure(ag_b, world, n=16, prefix="bo")
        prep_eval(ag_b)
        observe_cue(ag_b, world, tag="bw", body=list(MID_BODY))
        bind_ok = motor_scores(ag_b).get(world["beneficial"], -1.0) > motor_scores(ag_b).get(
            harmful_handle(world), 0.0
        )

        ag_s = _fresh(tmp, "s", sib)
        beneficial_exposure(ag_s, sib, n=16, prefix="s")
        prep_eval(ag_s)
        observe_cue(ag_s, sib, tag="sw", body=list(MID_BODY))
        sib_ok = motor_scores(ag_s).get(sib["beneficial"], -1.0) > motor_scores(ag_s).get(
            harmful_handle(sib), 0.0
        )

        swap = control_c4_v6()
        swap_ok = bool(swap.get("ok"))
        passed = bool(rename_ok and bind_ok and sib_ok and swap_ok)
    return {
        "id": "A9",
        "passed": passed,
        "rename_ok": rename_ok,
        "bind_order_ok": bind_ok,
        "sibling_ok": sib_ok,
        "swap_ok": swap_ok,
        "swap": {k: v for k, v in swap.items() if k != "rows"},
    }


def run_a10() -> dict[str, Any]:
    world = make_cell_world(11)
    with tempfile.TemporaryDirectory(prefix="ac_a10_") as tmp:
        ag = _fresh(tmp, "s", world)
        beneficial_exposure(ag, world, n=24, prefix="e")
        before = probe_policy(ag, world, n_probe=16, prefix="pre")
        ag.rest_epoch(8, body=np.asarray(MID_BODY, dtype=np.float64))
        after = probe_policy(ag, world, n_probe=16, prefix="post")
        passed = bool(
            after["p_act"] >= 0.5 * before["p_act"]
            and after["motor_ben"] > after["motor_harm"]
            and after["sampled_ben"] >= max(1, before["sampled_ben"] // 2)
        )
    return {"id": "A10", "passed": passed, "before": before, "after": after}


def run_a11() -> dict[str, Any]:
    world = make_cell_world(12)
    with tempfile.TemporaryDirectory(prefix="ac_a11_") as tmp:
        ag = _fresh(tmp, "s", world)
        for i, sym in enumerate(world["symbols"][:3]):
            observe_cue(ag, world, tag=f"fam{i}", body=list(MID_BODY), symbols=[sym])
        beneficial_exposure(ag, world, n=8, prefix="e")
        probe = clone_frozen(ag)
        holds_two = 0
        n_probe = 20
        for i in range(n_probe):
            prep_eval(probe)
            two = observe_cue(
                probe,
                world,
                tag=f"two{i}",
                body=list(MID_BODY),
                symbols=list(world["teacher_pair"]),
            )
            if (two.get("action") or {}).get("op") == "HOLD":
                holds_two += 1
        probe2 = clone_frozen(ag)
        holds_novel = 0
        for i in range(n_probe):
            prep_eval(probe2)
            novel = observe_cue(
                probe2,
                world,
                tag=f"nov{i}",
                body=list(MID_BODY),
                symbols=["zz_unseen_symbol"],
            )
            if (novel.get("action") or {}).get("op") == "HOLD":
                holds_novel += 1
        passed = bool(holds_two >= 14 and holds_novel >= 14)
    return {
        "id": "A11",
        "passed": passed,
        "two_symbol_hold": holds_two,
        "novel_hold": holds_novel,
        "n_probe": n_probe,
    }


CELL_FNS = {
    "A0": run_a0,
    "A1": run_a1,
    "A2": run_a2,
    "A3": run_a3,
    "A4": run_a4,
    "A5": run_a5,
    "A6": run_a6,
    "A7": run_a7,
    "A8": run_a8,
    "A9": run_a9,
    "A10": run_a10,
    "A11": run_a11,
}


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_cell_world(0)
    with tempfile.TemporaryDirectory(prefix="ac_sm_") as tmp:
        ag = _fresh(tmp, "s", w)
        observe_cue(ag, w, tag="sm", body=list(MID_BODY))
        cl = ag.clamp_action("ACT", w["beneficial"])
        ag.drop_actor_pending()
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "clamp_abi": bool(cl.get("ok")),
        "cells": prereg["cells"],
        "domain": CELLS_DOMAIN,
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if not _git_clean():
        raise RuntimeError("write runner.lock only on a clean tree")
    if not CANDIDATE.exists():
        raise RuntimeError("candidate.v29 required")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.ACTORCREDIT.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "shas": actorcredit_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "candidate_v29_sha": sha_file(CANDIDATE),
        "n": 64,
        "domain": CELLS_DOMAIN,
        "cells": prereg["cells"],
        "git_head": _git_head(),
        "note": "Frozen A-cell runner. Score only after this lock is on origin/main. v28 historical.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_all() -> dict[str, Any]:
    assert_runner_frozen()
    cells = [CELL_FNS[k]() for k in load_prereg()["cells"]]
    all_pass = all(c["passed"] for c in cells)
    out = {
        "version": "TM.0.24.ACTORCREDIT.RESULT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "all_cells_pass": all_pass,
        "n_pass": sum(1 for c in cells if c["passed"]),
        "n_cells": len(cells),
        "cells": cells,
        "reachability_authorized": bool(all_pass),
        "another_lineage_run": False,
        "git_head": _git_head(),
        "env": torch_env(),
        "note": "A0–A11 on unused TM024.ACTORCREDIT.CELLS. Reachability only if all pass. Not 0.0.005.",
    }
    RESULT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.ACTORCREDIT results\n\n"
        + "\n".join(f"- `{c['id']}`: **{'PASS' if c['passed'] else 'FAIL'}**" for c in cells)
        + f"\n\nall_cells_pass={all_pass}. n stays 64. Product 0.0.004.\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
    elif args.score:
        print(json.dumps(run_all(), indent=2, default=str))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
