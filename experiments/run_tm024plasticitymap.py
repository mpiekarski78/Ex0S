"""TM.0.24.PLASTICITYMAP — developmental motor-learning decomposition.

Not a lineage version. Not a capability earn. Product 0.0.004.
Scoring requires docs/lineage_plasticitymap.runner.lock on clean origin/main.
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

from experiments.run_tm023cortex import build_observe, make_cortex, physics, torch_env
from experiments.run_tm024lineage import live_once, make_synthetic_world, probe_beneficial
from experiments.run_tm024wallmap import motor_scores, op_logits, softmax_np
from three_memory.cortex_lineage import freeze_plasticity, sha_file
from three_memory.neural_cortex import ELIG_EPS, OP_COST, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_plasticitymap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_plasticitymap_contract.md"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_plasticitymap.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_plasticitymap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024plasticitymap_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v28.lock"

MID_BODY = [0.5, 0.4, 0.5, 0.0]
UNUSED = ("W_rec", "W_in", "W_write", "W_att", "W_emit_query")


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def make_pmap_world(domain: str, index: int) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def pmap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "candidate_v28": CANDIDATE,
        "cortex_lineage": REPO_ROOT / "three_memory" / "cortex_lineage.py",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no plasticitymap runner.lock — refuse diagnostic scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if pmap_shas() != lock.get("shas"):
        raise RuntimeError("plasticitymap implementation drifted after runner.lock")
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if sha_file(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from v28 candidate")
    if cand.get("genome", {}).get("n") != 64:
        raise RuntimeError("n must stay 64")
    return lock


def assert_clean_to_begin() -> None:
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    if not porcelain.strip():
        return
    allowed_prefix = ("docs/lineage_plasticitymap", "docs/tm024plasticitymap_results.md")
    for line in porcelain.splitlines():
        path = line[3:].strip()
        if any(path.startswith(p) for p in allowed_prefix):
            continue
        raise RuntimeError(f"dirty tree blocks plasticitymap scoring: {path}")


def harmful_handle(world: dict[str, Any]) -> str:
    for h in world["handles"][:2]:
        if h != world["beneficial"]:
            return str(h)
    return str(world["handles"][1])


def clone_from_checkpoint(ag: NeuralCortex) -> NeuralCortex:
    snap = ag.checkpoint()
    twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device=str(ag.device))
    twin.load_checkpoint(snap)
    return twin


def plastic_snap(ag: NeuralCortex) -> dict[str, Any]:
    return {
        "W": {n: getattr(ag, n).detach().clone() for n in ag._plastic_names},
        "slow": {n: ag.W_slow[n].detach().clone() for n in ag._plastic_names},
    }


def max_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
    return max(float((a[key][n] - b[key][n]).abs().max().item()) for n in a[key])


def observe_cue(ag: NeuralCortex, world: dict[str, Any], *, tag: str, body: list[float], one_symbol: bool = True) -> dict[str, Any]:
    pair = list(world["teacher_pair"])
    syms = [pair[0]] if one_symbol else pair
    return ag.observe(
        build_observe(
            interaction_token=tag,
            source_token="src_pmap",
            ordered_symbols=syms,
            observable_state=["st_idle"],
            body_state=list(body),
        )
    )


def stage_forced_act(ag: NeuralCortex, tok: str, body: list[float]) -> None:
    pending = ag._pending or {}
    rho = np.asarray(pending.get("rho_elig", ag._from_t(ag.rho)), dtype=np.float64)
    ag._pending = {
        "op": "ACT",
        "token": tok,
        "rho_elig": rho.copy(),
        "s_hat": np.asarray(pending.get("s_hat", np.zeros(ag.genome.d_sym)), dtype=np.float64),
        "body": np.asarray(body, dtype=np.float64),
        "cost": float(OP_COST["ACT"]),
        "motor_vec": ag.motor_vocab[tok].copy(),
    }


def force_balanced_exposure(ag: NeuralCortex, world: dict[str, Any], *, n_cycles: int) -> dict[str, int]:
    """Host-scheduled equal ACT opportunities. Ordinary v28 credit. Body reset each opportunity."""
    ben = world["beneficial"]
    harm = harmful_handle(world)
    counts = {ben: 0, harm: 0}
    for i in range(int(n_cycles)):
        for tok in (ben, harm):
            body = list(MID_BODY)
            observe_cue(ag, world, tag=f"f{i}_{tok}", body=body, one_symbol=True)
            stage_forced_act(ag, tok, body)
            _, body2 = physics(body, tok, world["latent"])
            observe_cue(ag, world, tag=f"c{i}_{tok}", body=body2, one_symbol=True)
            counts[tok] += 1
    return counts


def probe_handle_counts(ag: NeuralCortex, world: dict[str, Any], *, n_probe: int) -> dict[str, Any]:
    ben = world["beneficial"]
    harm = harmful_handle(world)
    counts = {ben: 0, harm: 0, "other": 0}
    body = [1.0, 0.0, 1.0, 0.0]
    for i in range(int(n_probe)):
        out = observe_cue(ag, world, tag=f"pr{i}", body=body, one_symbol=True)
        tok = (out.get("action") or {}).get("token")
        if tok == ben:
            counts[ben] += 1
        elif tok == harm:
            counts[harm] += 1
        else:
            counts["other"] += 1
    scores = motor_scores(ag)
    return {
        "counts": counts,
        "probe_beneficial": float(counts[ben] / max(n_probe, 1)),
        "motor_ben": float(scores.get(ben, 0.0)),
        "motor_harm": float(scores.get(harm, 0.0)),
        "ben_beats_harm_score": bool(scores.get(ben, 0.0) > scores.get(harm, 0.0)),
        "ben_beats_harm_count": bool(counts[ben] > counts[harm]),
    }


def _warm(ag: NeuralCortex, world: dict[str, Any]) -> np.ndarray:
    observe_cue(ag, world, tag="warm", body=list(MID_BODY), one_symbol=True)
    return ag._from_t(ag.rho).copy()


def run_d0() -> dict[str, Any]:
    assert_runner_frozen()
    world = make_pmap_world(load_prereg()["domains"]["CREDIT"], 0)
    ben = world["beneficial"]
    harm = harmful_handle(world)
    links: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="pmap_d0_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"]))
        body = list(MID_BODY)
        rho_prev = _warm(ag, world)
        rho_curr = _warm(ag, world)

        _, body_b = physics(body, ben, world["latent"])
        _, body_h = physics(body, harm, world["latent"])
        body_changed = any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(body, body_b, strict=True))

        def apply_with(elig: np.ndarray, tok: str, body_next: list[float]) -> dict[str, float]:
            ag._pending = {
                "op": "ACT",
                "token": tok,
                "rho_elig": np.asarray(elig, dtype=np.float64).copy(),
                "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
                "body": np.asarray(body, dtype=np.float64),
                "cost": float(OP_COST["ACT"]),
                "motor_vec": ag.motor_vocab[tok].copy(),
            }
            return ag._apply_credit(ag.encode_state_set(["st_p"]), np.asarray(body_next, dtype=np.float64))

        m_pos = apply_with(rho_curr, ben, body_b)
        m_neg = apply_with(rho_curr, harm, body_h)
        links["body_to_opposite_adv"] = {
            "ok": bool(m_pos.get("adv", 0) > 0 and m_neg.get("adv", 0) < 0 and body_changed),
            "adv_beneficial": m_pos.get("adv"),
            "adv_harmful": m_neg.get("adv"),
        }

        before = plastic_snap(ag)
        apply_with(np.zeros_like(rho_curr), ben, body_b)
        after_z = plastic_snap(ag)
        d_w = max_delta(before, after_z, "W")
        d_s = max_delta(before, after_z, "slow")
        links["zero_elig_no_plastic_motion"] = {
            "ok": bool(d_w < 1e-12 and d_s < 1e-12),
            "max_abs_delta": d_w,
            "max_slow_delta": d_s,
            "elig_eps": ELIG_EPS,
        }

        def logit_gain(elig: np.ndarray) -> float:
            twin = clone_from_checkpoint(ag)
            pre = motor_scores(twin).get(ben, 0.0)
            twin._pending = {
                "op": "ACT",
                "token": ben,
                "rho_elig": np.asarray(elig, dtype=np.float64).copy(),
                "s_hat": np.zeros(twin.genome.d_sym, dtype=np.float64),
                "body": np.asarray(body, dtype=np.float64),
                "cost": float(OP_COST["ACT"]),
                "motor_vec": twin.motor_vocab[ben].copy(),
            }
            twin._apply_credit(twin.encode_state_set(["st_p"]), np.asarray(body_b, dtype=np.float64))
            observe_cue(twin, world, tag="lg", body=list(MID_BODY), one_symbol=True)
            post = motor_scores(twin).get(ben, 0.0)
            return float(post - pre)

        g_curr = logit_gain(rho_curr)
        g_prev = logit_gain(rho_prev)
        wrong = np.random.default_rng(0).normal(0.0, 1.0, size=rho_curr.shape).astype(np.float64)
        g_wrong = logit_gain(wrong)
        links["correct_prior_elig_used"] = {
            "ok": bool(g_curr > g_prev),
            "gain_current": g_curr,
            "gain_previous_tick": g_prev,
        }
        links["wrong_tick_elig_fails"] = {
            "ok": bool(g_curr > g_wrong),
            "gain_current": g_curr,
            "gain_wrong_tick": g_wrong,
        }

        pre_scores = motor_scores(ag)
        unused_before = plastic_snap(ag)
        apply_with(rho_curr, ben, body_b)
        observe_cue(ag, world, tag="postc", body=list(MID_BODY), one_symbol=True)
        post_scores = motor_scores(ag)
        d_star = float(post_scores.get(ben, 0.0) - pre_scores.get(ben, 0.0))
        d_dist = float(post_scores.get(harm, 0.0) - pre_scores.get(harm, 0.0))
        links["credit_to_handle_logit"] = {
            "ok": bool(d_star > d_dist),
            "delta_beneficial": d_star,
            "delta_distractor": d_dist,
        }

        unused_d = {
            n: float((getattr(ag, n) - unused_before["W"][n]).abs().max().item()) for n in UNUSED
        }
        unused_slow = {
            n: float((ag.W_slow[n] - unused_before["slow"][n]).abs().max().item()) for n in UNUSED
        }
        credited_w = float((ag.W_act_query - unused_before["W"]["W_act_query"]).abs().max().item())
        credited_slow = float((ag.W_slow["W_act_query"] - unused_before["slow"]["W_act_query"]).abs().max().item())
        links["consolidation_boundary"] = {
            "ok": bool(
                credited_w > 1e-12
                and credited_slow > 1e-12
                and max(unused_d.values()) < 1e-12
                and max(unused_slow.values()) < 1e-12
            ),
            "credited_W_act_query": credited_w,
            "credited_slow": credited_slow,
            "unused_max": max(unused_d.values()),
            "unused_slow_max": max(unused_slow.values()),
        }

        snap = ag.checkpoint()
        ag_on = clone_from_checkpoint(ag)
        ag_off = clone_from_checkpoint(ag)
        freeze_plasticity(ag_off)
        force_balanced_exposure(ag_on, world, n_cycles=8)
        force_balanced_exposure(ag_off, world, n_cycles=8)
        on_p = probe_handle_counts(ag_on, world, n_probe=20)
        off_p = probe_handle_counts(ag_off, world, n_probe=20)
        links["later_probability_and_sampled_behavior"] = {
            "ok": bool(
                on_p["probe_beneficial"] > off_p["probe_beneficial"]
                and on_p["motor_ben"] > off_p["motor_ben"]
            ),
            "probe_on": on_p["probe_beneficial"],
            "probe_off": off_p["probe_beneficial"],
            "motor_ben_on": on_p["motor_ben"],
            "motor_ben_off": off_p["motor_ben"],
            "start_checkpoint": bool(snap),
        }

    passed = all(bool(v.get("ok")) for v in links.values())
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D0",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "n": 64,
        "links": links,
        "note": "Complete v28 credit chain on fresh worlds. Zero-elig alone is not a pass.",
    }
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d0.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_d1() -> dict[str, Any]:
    assert_runner_frozen()
    world = make_pmap_world(load_prereg()["domains"]["READOUT"], 0)
    with tempfile.TemporaryDirectory(prefix="pmap_d1_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"][:2]))
        _warm(ag, world)
        rho = ag.rho
        denom = float((rho * rho).sum().clamp(min=1e-12).item())
        winners = []
        for h in world["handles"][:2]:
            vec = ag._to_t(ag.motor_vocab[h])
            ag.W_act_query = torch_outer(vec, rho) / denom
            scores = motor_scores(ag)
            ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
            winners.append({"handle": h, "top": ranked[0], "scores": scores, "ok": ranked[0] == h})
    passed = all(w["ok"] for w in winners) and len(winners) == 2
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "n": 64,
        "winners": winners,
        "note": "Direct set of W_act_query. Not a learning claim.",
    }
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d1.lock").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def torch_outer(vec, rho):  # noqa: ANN001
    import torch

    return torch.outer(vec, rho)


def _d2_pass(trained: dict[str, Any], frozen: dict[str, Any]) -> bool:
    return bool(
        trained["ben_beats_harm_score"]
        and trained["ben_beats_harm_count"]
        and trained["probe_beneficial"] > frozen["probe_beneficial"]
        and trained["motor_ben"] > frozen["motor_ben"]
    )


def run_d2() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()
    world = make_pmap_world(prereg["domains"]["FORCE"], 0)
    n_cycles = int(prereg["D2"]["n_cycles"])
    n_probe = int(prereg["D2"]["n_probe"])
    with tempfile.TemporaryDirectory(prefix="pmap_d2_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"]))
        _warm(ag, world)
        frozen = clone_from_checkpoint(ag)
        freeze_plasticity(frozen)
        force_balanced_exposure(frozen, world, n_cycles=n_cycles)
        force_balanced_exposure(ag, world, n_cycles=n_cycles)
        tr = probe_handle_counts(ag, world, n_probe=n_probe)
        fr = probe_handle_counts(frozen, world, n_probe=n_probe)
        birth = make_cortex(Path(tmp) / "b", device="cpu")
        birth.bind_actuators(list(world["handles"]))
        _warm(birth, world)
        br = probe_handle_counts(birth, world, n_probe=n_probe)
    passed = _d2_pass(tr, fr)
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D2",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "n": 64,
        "n_cycles": n_cycles,
        "trained": tr,
        "frozen": fr,
        "birth": br,
        "note": "Forced equal ACT opportunities. Not teaching the answer. Not a capability earn.",
    }
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d2.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_d3() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()
    world = make_pmap_world(prereg["domains"]["AUTO"], 0)
    with tempfile.TemporaryDirectory(prefix="pmap_d3_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"]))
        frozen = clone_from_checkpoint(ag)
        freeze_plasticity(frozen)
        live_once(ag, world, n_wake=int(prereg["D3"]["n_wake"]), n_replay=int(prereg["D3"]["n_replay"]), teacher_seed=11)
        live_once(frozen, world, n_wake=int(prereg["D3"]["n_wake"]), n_replay=int(prereg["D3"]["n_replay"]), teacher_seed=11)
        tr = probe_handle_counts(ag, world, n_probe=int(prereg["D3"]["n_probe"]))
        fr = probe_handle_counts(frozen, world, n_probe=int(prereg["D3"]["n_probe"]))
        pb = float(probe_beneficial(ag, world, n_probe=int(prereg["D3"]["n_probe"])))
    passed = _d2_pass(tr, fr)
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D3",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "n": 64,
        "trained": tr,
        "frozen": fr,
        "probe_beneficial_l0_unit": pb,
        "note": "Ordinary live_once. No forced ACT.",
    }
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d3.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_d4() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()
    world = make_pmap_world(prereg["domains"]["REST"], 0)
    n_cycles = int(prereg["D2"]["n_cycles"])
    n_probe = int(prereg["D2"]["n_probe"])
    with tempfile.TemporaryDirectory(prefix="pmap_d4_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(world["handles"]))
        force_balanced_exposure(ag, world, n_cycles=n_cycles)
        pre = probe_handle_counts(ag, world, n_probe=n_probe)
        ag.rest_epoch(int(prereg["D4"]["n_rest"]), body=np.asarray(MID_BODY, dtype=np.float64))
        post = probe_handle_counts(ag, world, n_probe=n_probe)
    kept = bool(post["ben_beats_harm_score"] and post["motor_ben"] >= pre["motor_ben"] - 1e-6)
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D4",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": bool(pre["ben_beats_harm_score"] and kept),
        "immediate_preference": bool(pre["ben_beats_harm_score"]),
        "n": 64,
        "pre_rest": pre,
        "post_rest": post,
        "note": "REST after forced exposure. Scored even if D2 on FORCE world differs.",
    }
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d4.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_d5() -> dict[str, Any]:
    assert_runner_frozen()
    prereg = load_prereg()
    n_cycles = int(prereg["D2"]["n_cycles"])
    n_probe = int(prereg["D2"]["n_probe"])
    w0 = make_pmap_world(prereg["domains"]["SIBLING"], 0)
    w1 = make_pmap_world(prereg["domains"]["SIBLING"], 1)
    assert w0["handles"] != w1["handles"]
    with tempfile.TemporaryDirectory(prefix="pmap_d5_") as tmp:
        a = make_cortex(Path(tmp) / "a", device="cpu")
        a.bind_actuators(list(w0["handles"]))
        fa = clone_from_checkpoint(a)
        freeze_plasticity(fa)
        force_balanced_exposure(a, w0, n_cycles=n_cycles)
        force_balanced_exposure(fa, w0, n_cycles=n_cycles)
        d_a = probe_handle_counts(a, w0, n_probe=n_probe)
        f_a = probe_handle_counts(fa, w0, n_probe=n_probe)
        # zero-shot transfer onto renamed sibling (report only)
        a.bind_actuators(list(w1["handles"]))
        transfer = probe_handle_counts(a, w1, n_probe=n_probe)
        b = make_cortex(Path(tmp) / "b", device="cpu")
        b.bind_actuators(list(w1["handles"]))
        fb = clone_from_checkpoint(b)
        freeze_plasticity(fb)
        force_balanced_exposure(b, w1, n_cycles=n_cycles)
        force_balanced_exposure(fb, w1, n_cycles=n_cycles)
        d_b = probe_handle_counts(b, w1, n_probe=n_probe)
        f_b = probe_handle_counts(fb, w1, n_probe=n_probe)
    pass_a = _d2_pass(d_a, f_a)
    pass_b = _d2_pass(d_b, f_b)
    passed = bool(pass_a and pass_b)
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D5",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "deterministic_world_pass": pass_a,
        "renamed_sibling_pass": pass_b,
        "zero_shot_transfer": transfer,
        "zero_shot_transfer_is_not_pass_gate": True,
        "n": 64,
        "world0": d_a,
        "world1": d_b,
        "note": "Independent D2 procedure on renamed sibling. Zero-shot transfer is reported only.",
    }
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d5.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_d6(*, d1: dict[str, Any], d2: dict[str, Any], d3: dict[str, Any]) -> dict[str, Any]:
    released = bool(d1.get("passed") and d2.get("passed") and d3.get("passed"))
    q3 = json.loads((REPO_ROOT / "docs" / "lineage_wallmap_q3.lock").read_text(encoding="utf-8"))
    out = {
        "version": "TM.0.24.PLASTICITYMAP.D6",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": False if released else None,
        "scored": False,
        "skipped_because_earlier_fail": not released,
        "historical_q3_passed": bool(q3.get("passed")),
        "historical_q3": "docs/lineage_wallmap_q3.lock",
        "n": 64,
        "note": (
            "D6 released only if D1–D3 pass. Historical WALLMAP Q3 remains the ES record."
            if not released
            else "Developmental diagnostics passed; historical Q3 still red. Outer-search wall."
        ),
    }
    if released:
        out["passed"] = False
        out["scored"] = False
        out["historical_override"] = True
    (REPO_ROOT / "docs" / "lineage_plasticitymap_d6.lock").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_decision(d0: dict, d1: dict, d2: dict, d3: dict, d4: dict, d5: dict, d6: dict) -> dict[str, Any]:
    if not d0.get("passed"):
        primary = "D0_credit_chain_incomplete"
        nxt = "Do not treat Q4 as fully repaired. Complete remaining credit-chain links before reading D2 as plasticity vs exploration."
    elif not d1.get("passed"):
        primary = "D1_readout_expressivity"
        nxt = "Readout cannot rank handles when directly set."
    elif not d2.get("passed"):
        primary = "D2_forced_exposure_plasticity"
        nxt = "Amend the general three-factor actor pathway: eligibility ownership, update direction, fast/slow separation, or motor-vector credit."
    elif not d3.get("passed"):
        primary = "D3_autonomous_exploration"
        nxt = "Forced exposure works; autonomous learning fails. Address exploration and action-selection dynamics."
    elif d2.get("passed") and not d4.get("passed"):
        primary = "D4_rest_retention"
        nxt = "Immediate preference succeeds; post-REST fails. Repair consolidation/retention."
    elif d2.get("passed") and not d5.get("passed"):
        primary = "D5_renamed_siblings"
        nxt = "Deterministic world passes; renamed sibling fails. Developmental generalization wall."
    elif d1.get("passed") and d2.get("passed") and d3.get("passed"):
        primary = "D6_outer_search"
        nxt = "Developmental diagnostics pass; historical ES SNR remains unstable. Redesign outer search. Do not increase n from this package."
    else:
        primary = "undetermined"
        nxt = "See per-diagnostic locks."
    out = {
        "version": "TM.0.24.PLASTICITYMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "D0_passed": bool(d0.get("passed")),
        "D1_passed": bool(d1.get("passed")),
        "D2_passed": bool(d2.get("passed")),
        "D3_passed": bool(d3.get("passed")),
        "D4_passed": bool(d4.get("passed")),
        "D5_passed": bool(d5.get("passed")),
        "D6_passed": d6.get("passed"),
        "primary_bottleneck": primary,
        "next_change": nxt,
        "increase_n": False,
        "another_lineage_run": False,
        "note": "Diagnostic decomposition only. Not 0.0.005. LINEAGE/WALLMAP/REACH historical. QUAL/EVAL sealed.",
    }
    DECISION.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.PLASTICITYMAP results\n\n"
        "Product remains **0.0.004**. `earned_next=false`. `ex0s=null`.\n\n"
        f"| D | Question | Passed |\n| --- | --- | --- |\n"
        f"| D0 | Complete v28 credit chain | **{d0.get('passed')}** |\n"
        f"| D1 | Readout expressivity | **{d1.get('passed')}** |\n"
        f"| D2 | Forced balanced ACT exposure | **{d2.get('passed')}** |\n"
        f"| D3 | Autonomous exploration | **{d3.get('passed')}** |\n"
        f"| D4 | Post-REST retention | **{d4.get('passed')}** |\n"
        f"| D5 | Renamed siblings | **{d5.get('passed')}** |\n"
        f"| D6 | Outer search | **{d6.get('passed')}** |\n\n"
        f"**Primary bottleneck:** `{primary}`\n\n"
        f"**Next change:** {nxt}\n\n"
        "n stays 64. LINEAGE/WALLMAP/REACH historical. QUAL/EVAL sealed. Not a capability earn.\n",
        encoding="utf-8",
    )
    return out


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_pmap_world(prereg["domains"]["FORCE"], 0)
    with tempfile.TemporaryDirectory(prefix="pmap_sm_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(w["handles"]))
        _warm(ag, w)
        force_balanced_exposure(ag, w, n_cycles=1)
        pb = probe_handle_counts(ag, w, n_probe=4)
        vec = ag._to_t(ag.motor_vocab[w["beneficial"]])
        rho = ag.rho
        denom = float((rho * rho).sum().clamp(min=1e-12).item())
        ag.W_act_query = torch_outer(vec, rho) / denom
        scores = motor_scores(ag)
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "forced_cycle_ok": True,
        "readout_set_ok": bool(scores.get(w["beneficial"], -1) >= max(scores.values())),
        "probe": pb["probe_beneficial"],
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if not _git_clean():
        raise RuntimeError("write runner.lock only on a clean tree")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.PLASTICITYMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "shas": pmap_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "candidate_v28_sha": sha_file(CANDIDATE),
        "n": 64,
        "domains": prereg["domains"],
        "D2_n_cycles": prereg["D2"]["n_cycles"],
        "git_head_at_freeze": _git_head(),
        "note": "SHA pin is the integrity gate. Clean tree required once at suite start; result locks may appear mid-run.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_all() -> dict[str, Any]:
    assert_clean_to_begin()
    assert_runner_frozen()
    d0 = run_d0()
    d1 = run_d1()
    d2 = run_d2()
    d3 = run_d3()
    d4 = run_d4()
    d5 = run_d5()
    d6 = run_d6(d1=d1, d2=d2, d3=d3)
    decision = write_decision(d0, d1, d2, d3, d4, d5, d6)
    return {"D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "decision": decision}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--write-runner-lock", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    ran = False
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
        ran = True
    if args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
        ran = True
    if args.all:
        print(json.dumps(run_all(), indent=2, default=str))
        ran = True
    if not ran:
        p.print_help()


if __name__ == "__main__":
    main()
