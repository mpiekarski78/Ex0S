"""TM.0.27.GATEDREHEARSAL — v34 prediction-error-gated competitive rehearsal battery.

Not a product earn. Product 0.0.004. SCORE reserved.
DEV on unused TM027.GATEDREHEARSAL.DEV. / TWIN.
Runner frozen at Phase 1 push; byte-identical through DEV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import build_observe, make_cortex, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024convergencemap import unique_winner
from experiments.run_tm024statemap import prep_eval, read_policy
from experiments.run_tm024writegeom import (
    NEG_DELTA,
    capacity_world,
    mapping_pairs,
    ranking_margin,
    set_handle_delta,
)
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import (
    ACT_MARGIN_FLOOR,
    ACT_SCORE_QUERY,
    EPISODE_MATCH_L2,
    EPISODE_REPLAY_EPOCHS,
    EPISODE_SLOTS,
    NeuralCortex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
LAB = "TM.0.27.GATEDREHEARSAL"
PREREG = REPO_ROOT / "docs" / "lineage_gatedrehearsal.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_gatedrehearsal_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_gatedrehearsal.isolation.lock"
V34_PREREG = REPO_ROOT / "docs" / "cortex_v34.prereg.lock"
V34_ISO = REPO_ROOT / "docs" / "cortex_v34.isolation.lock"
V34_AMEND = REPO_ROOT / "docs" / "cortex_v34_architecture_amendment.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_gatedrehearsal.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_gatedrehearsal.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm027gatedrehearsal_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"

DEV_DOMAIN = "TM027.GATEDREHEARSAL.DEV."
TWIN_DOMAIN = "TM027.GATEDREHEARSAL.TWIN."
SCORE_DOMAIN = "TM027.GATEDREHEARSAL.SCORE."
EXPECTED_N_CELLS = 54
MANIFEST_SHA = "1140e68472d1cfc147d003bb158d68e3dba5b38a0b66348c8a7ee02a988c2e6d"


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def gated_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "v34_prereg": V34_PREREG,
        "v34_isolation": V34_ISO,
        "v34_amendment": V34_AMEND,
        "competitive_addendum": REPO_ROOT / "docs" / "lineage_competitive.decision.addendum.lock",
        "candidate_v30": REPO_ROOT / "docs" / "cortex.candidate.v30.lock",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def refuse_score() -> None:
    raise RuntimeError("TM.0.27.GATEDREHEARSAL SCORE is reserved and must not be opened")


def refuse_dev_lock() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("TM.0.27.GATEDREHEARSAL DEV lock exists; same frozen DEV execution refused again")
    if not V34_PREREG.exists():
        raise RuntimeError("GATEDREHEARSAL DEV lock must wait for cortex_v34.prereg.lock")


def refuse_runner_mutation(frozen_runner_sha: str) -> None:
    live = sha_file(THIS)
    if live != frozen_runner_sha:
        raise RuntimeError("GATEDREHEARSAL runner SHA drifted after Phase 1 freeze; DEV refused")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def assert_finite_record(rec: dict[str, Any], *, ctx: str) -> None:
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (float, np.floating)):
            if not _finite(float(x)):
                raise RuntimeError(f"non-finite float in {ctx}: {x!r}")

    walk(rec)


def manifest_sha(ids: list[str] | None = None) -> str:
    cell_ids = ids if ids is not None else expected_cell_ids()
    p = load_prereg()
    domains = dict(p.get("domains") or {})
    payload = {"lab": LAB, "domains": sorted(domains.items()), "cell_ids": sorted(cell_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    root = Path(tmp) / tag
    ag = make_cortex(root, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    if str(ag.genome.act_score_mode) != ACT_SCORE_QUERY:
        raise RuntimeError("v34 default scoring must remain query")
    return ag


def p1_probe(ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str) -> dict[str, Any]:
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    p1 = probe._last_p1
    if p1 is None:
        p1 = probe._from_t(probe.rho)
    scores = probe.actuator_scores(p1)
    winner = unique_winner(scores)
    gap = ranking_margin(scores, winner) if winner else 0.0
    gamma = float(probe._act_geometric_margin(p1, winner)) if winner else 0.0
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "winner": winner,
        "pairwise_score_gap": float(gap),
        "normalized_geometric_margin": float(gamma),
        "p1": np.asarray(p1, dtype=np.float64).copy(),
        "n_episodes": len(probe._episodes),
    }


def perturb_p1(ag: NeuralCortex, p1: np.ndarray, want: str, *, domain: str, key: str) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = np.asarray(p1, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r0)) + 1e-12
    r_hat = r0 / nrm
    n_ok = 0
    for _i in range(n):
        unit = r_hat + rng.normal(0.0, sigma, size=r_hat.shape)
        pn = float(np.linalg.norm(unit)) + 1e-12
        unit = unit / pn
        scores = ag.actuator_scores(unit)
        if unique_winner(scores) == want:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def teach_one(
    ag: NeuralCortex,
    world: dict[str, Any],
    tok: str,
    *,
    tag: str,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    from experiments.run_tm024statemap import moved, physics, ranking_ok

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
    metrics = out.get("metrics") or {}
    return {
        "rho_teach": rho_teach,
        "pre": pre,
        "post_ident": post,
        "adv": float(metrics.get("adv") or 0.0),
        "d_w_op": float((ag.W_op - w_op0).abs().max().item()),
        "d_w_q": float((ag.W_act_query - w_q0).abs().max().item()),
        "moved": moved(pre, post),
        "ranking": ranking_ok(post),
        "rehearsal_burst": metrics.get("rehearsal_burst"),
    }


def _aggregate_awake_bursts(bursts: list[dict[str, Any] | None]) -> dict[str, Any]:
    awake_bursts: list[dict[str, Any]] = []
    total_awake_updates = 0
    awake_budget_exhausted = False
    final_first: int | None = None
    for bi, raw in enumerate(bursts):
        if not raw:
            continue
        passes = list(raw.get("passes") or [])
        total_awake_updates += int(raw.get("total_updates") or 0)
        if bool(raw.get("budget_exhausted")):
            awake_budget_exhausted = True
        fc = raw.get("first_converged_pass")
        awake_bursts.append(
            {
                "burst_index": int(bi),
                "passes": passes,
                "first_converged_pass": fc,
                "budget_exhausted": bool(raw.get("budget_exhausted")),
                "total_updates": int(raw.get("total_updates") or 0),
            }
        )
        if fc is not None:
            final_first = int(fc)
    return {
        "awake_bursts": awake_bursts,
        "final_awake_burst_first_converged_pass": final_first,
        "awake_budget_exhausted": awake_budget_exhausted,
        "total_awake_updates": int(total_awake_updates),
    }


def _aggregate_rest_rehearsal(rest_out: dict[str, Any] | None) -> dict[str, Any]:
    if not rest_out:
        return {
            "rest_epochs": [],
            "rest_first_converged_epoch": None,
            "rest_budget_exhausted": False,
            "total_rest_updates": 0,
            "violations_pre_mix": None,
            "violations_post_mix": None,
        }
    rh = rest_out.get("rehearsal") or {}
    return {
        "rest_epochs": list(rh.get("epochs") or []),
        "rest_first_converged_epoch": rh.get("first_converged_epoch"),
        "rest_budget_exhausted": bool(rh.get("budget_exhausted")),
        "total_rest_updates": int(rh.get("total_updates") or rest_out.get("n_replay") or 0),
        "violations_pre_mix": rh.get("violations_pre_mix"),
        "violations_post_mix": rh.get("violations_post_mix"),
    }


def classify_failure(
    *,
    stored_pre_mix: dict[str, Any],
    stored_post_mix: dict[str, Any],
    live_probes_pass: bool,
) -> str:
    pre_ok = bool(stored_pre_mix.get("all_margin_ok"))
    post_ok = bool(stored_post_mix.get("all_margin_ok"))
    if not pre_ok and not live_probes_pass:
        return "store_and_live_fail"
    if not pre_ok:
        return "store_nonconvergence"
    if pre_ok and not post_ok:
        return "consolidation_margin_loss"
    if post_ok and not live_probes_pass:
        return "reinstatement_wall"
    return "none"


def teach_pairs(ag: NeuralCortex, world: dict[str, Any], pairs: list[tuple[str, str]], *, tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
    taught = []
    bursts: list[dict[str, Any] | None] = []
    for i, (cue, handle) in enumerate(pairs):
        t = teach_one(ag, world, handle, tag=f"{tag}_{i}", symbols=[cue])
        bursts.append(t.get("rehearsal_burst"))
        taught.append({"cue": cue, "handle": handle, "adv": float(t["adv"]), "n_episodes": len(ag._episodes)})
    return taught, bursts


def probe_map(ag: NeuralCortex, world: dict[str, Any], pairs: list[tuple[str, str]], *, tag: str, domain: str) -> dict[str, Any]:
    probes = []
    ranking_ok = True
    n_probe_correct = 0
    gammas = []
    gaps = []
    pert_ok = True
    for i, (cue, handle) in enumerate(pairs):
        live = p1_probe(ag, world, cue, tag=f"{tag}_p{i}")
        rank = bool(live["winner"] == handle)
        ranking_ok = ranking_ok and rank
        n_probe_correct += int(rank)
        g = float(ag._act_geometric_margin(live["p1"], handle))
        if not _finite(g):
            raise RuntimeError(f"non-finite margin probe {cue}")
        gammas.append(g)
        gaps.append(float(live["pairwise_score_gap"]))
        stab = perturb_p1(ag, live["p1"], handle, domain=domain, key=f"{tag}_{cue}")
        pert_ok = pert_ok and bool(stab["stable"])
        probes.append(
            {
                "cue": cue,
                "want": handle,
                "winner": live["winner"],
                "ranking_ok": rank,
                "normalized_geometric_margin": g,
                "pairwise_score_gap": float(live["pairwise_score_gap"]),
                "perturbation_ok": bool(stab["stable"]),
            }
        )
    min_g = min(gammas) if gammas else 0.0
    geometric_ok = bool(min_g >= float(load_prereg()["margin"]["geometric_min"]))
    n_cues = int(world["capacity"]["n_cues"])
    return {
        "probes": probes,
        "ranking_ok": ranking_ok,
        "n_probe_correct": int(n_probe_correct),
        "n_cues": n_cues,
        "probe_count_matches_cues": bool(n_probe_correct == n_cues),
        "perturbation_ok": pert_ok,
        "geometric_ok": geometric_ok,
        "min_normalized_geometric_margin": float(min_g),
        "min_pairwise_score_gap": float(min(gaps) if gaps else 0.0),
        "n_episodes": len(ag._episodes),
    }


def expected_cell_ids() -> list[str]:
    p = load_prereg()
    ids: list[str] = []
    orders = list(p["orders"])
    n_worlds = int(p["n_worlds"])
    for spec in p["capacity"]:
        n = int(spec["n_cues"])
        for order in orders:
            for wi in range(n_worlds):
                ids.append(f"acquire|c{n}|{order}|w{wi}")
                ids.append(f"stable|c{n}|{order}|w{wi}")
    for order in orders:
        for wi in range(n_worlds):
            ids.append(f"twin|c2|{order}|w{wi}")
            ids.append(f"eco|{order}|w{wi}")
            ids.append(f"spec|{order}|w{wi}")
    for gate in p["integrity_gates"]:
        for wi in range(n_worlds):
            ids.append(f"{gate}|w{wi}")
    for spec in p["scaling"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for order in orders:
            for wi in range(n_worlds):
                ids.append(f"scale|acquire|c{n_cues}h{n_handles}|{order}|w{wi}")
                ids.append(f"scale|stable|c{n_cues}h{n_handles}|{order}|w{wi}")
    if manifest_sha(ids) != MANIFEST_SHA:
        raise RuntimeError("cell manifest drifted from frozen MANIFEST_SHA")
    if len(ids) != EXPECTED_N_CELLS:
        raise RuntimeError(f"expected {EXPECTED_N_CELLS} cells, got {len(ids)}")
    return ids


def eval_acquire_stable(
    *,
    kind: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool,
    cell_id: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="gr_cap_") as tmp:
        ag = _fresh(tmp, "s", world)
        taught, burst_raw = teach_pairs(ag, world, seq, tag=tag)
        stored_post_awake = ag.store_rehearsal_checkpoint()
        awake_diag = _aggregate_awake_bursts(burst_raw)
        rest_out: dict[str, Any] | None = None
        stored_post_rest_rehearsal_pre_mix: dict[str, Any] | None = None
        stored_post_slow_mix: dict[str, Any] | None = None
        if rest:
            rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
            rh = rest_out.get("rehearsal") or {}
            stored_post_rest_rehearsal_pre_mix = {
                "n_violations": int(rh.get("violations_pre_mix") or 0),
                "all_margin_ok": bool(int(rh.get("violations_pre_mix") or 0) == 0),
                "n_episodes": int(stored_post_awake.get("n_episodes") or 0),
            }
            stored_post_slow_mix = ag.store_rehearsal_checkpoint()
        rest_diag = _aggregate_rest_rehearsal(rest_out)
        probed = probe_map(ag, world, pairs, tag=tag, domain=str(world["domain"]))
    ranking_ok = bool(probed["ranking_ok"])
    live_probes_pass = bool(ranking_ok and probed["probe_count_matches_cues"]) if kind == "acquire" else bool(ranking_ok)
    if kind == "acquire":
        passed = bool(ranking_ok and probed["probe_count_matches_cues"])
    else:
        passed = bool(ranking_ok and probed["geometric_ok"] and probed["perturbation_ok"])
    pre_mix = stored_post_rest_rehearsal_pre_mix or stored_post_awake
    post_mix = stored_post_slow_mix or stored_post_awake
    failure_class = classify_failure(
        stored_pre_mix=pre_mix,
        stored_post_mix=post_mix,
        live_probes_pass=bool(probed["ranking_ok"] and (probed["probe_count_matches_cues"] if kind == "acquire" else probed["geometric_ok"])),
    )
    out = {
        "kind": kind,
        "order": order,
        "n_cues": int(world["capacity"]["n_cues"]),
        "id": cell_id,
        "taught": taught,
        "rest": rest_out,
        "passed": passed,
        "pass_statistic": "normalized_geometric_margin",
        "stored_post_awake": stored_post_awake,
        "stored_post_rest_rehearsal_pre_mix": stored_post_rest_rehearsal_pre_mix,
        "stored_post_slow_mix": stored_post_slow_mix,
        "live_probes_pass": live_probes_pass,
        "failure_class": failure_class,
        **awake_diag,
        **rest_diag,
        **probed,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_eco(world: dict[str, Any], *, order: str, tag: str, cell_id: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1, h2 = world["handles"][0], world["handles"][1]
    if order == "B_then_A":
        h1, h2 = h2, h1
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="cp_eco_") as tmp:
        ag = _fresh(tmp, "s", world)
        t1 = teach_one(ag, world, h1, tag=f"{tag}_p", symbols=[cue])
        wneg = set_handle_delta(world, h1, NEG_DELTA)
        t2 = teach_one(ag, wneg, h1, tag=f"{tag}_n", symbols=[cue])
        t3 = teach_one(ag, world, h2, tag=f"{tag}_r", symbols=[cue])
        n_replaced = int(ag._episode_n_replaced)
        rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        live = p1_probe(ag, world, cue, tag=f"{tag}_q")
        g = float(ag._act_geometric_margin(live["p1"], h2))
        stab = perturb_p1(ag, live["p1"], h2, domain=str(world["domain"]), key=f"{tag}_eco")
        ranking_ok = bool(live["winner"] == h2)
        geometric_ok = bool(g >= float(p["margin"]["geometric_min"]))
        passed = bool(
            t1["adv"] > 0.0
            and t2["adv"] < 0.0
            and t3["adv"] > 0.0
            and ranking_ok
            and geometric_ok
            and stab["stable"]
        )
    out = {
        "kind": "eco",
        "order": order,
        "n_cues": 2,
        "id": cell_id,
        "passed": passed,
        "ranking_ok": ranking_ok,
        "geometric_ok": geometric_ok,
        "perturbation_ok": bool(stab["stable"]),
        "normalized_geometric_margin": g,
        "min_normalized_geometric_margin": g,
        "pairwise_score_gap": float(live["pairwise_score_gap"]),
        "winner": live["winner"],
        "want": h2,
        "adv": [float(t1["adv"]), float(t2["adv"]), float(t3["adv"])],
        "n_replaced": n_replaced,
        "rest": rest_out,
        "n_episodes": live["n_episodes"],
        "pass_statistic": "normalized_geometric_margin",
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_spec(world: dict[str, Any], *, order: str, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    if len(pairs) < 2:
        raise RuntimeError("spec needs two cues")
    (c_a, h_a), (c_b, h_b) = pairs[0], pairs[1]
    if order == "B_then_A":
        (c_a, h_a), (c_b, h_b) = (c_b, h_b), (c_a, h_a)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="cp_spec_") as tmp:
        ag = _fresh(tmp, "s", world)
        teach_one(ag, world, h_a, tag=f"{tag}_a", symbols=[c_a])
        teach_one(ag, world, h_b, tag=f"{tag}_b", symbols=[c_b])
        wneg = set_handle_delta(world, h_a, NEG_DELTA)
        teach_one(ag, wneg, h_a, tag=f"{tag}_n", symbols=[c_a])
        teach_one(ag, world, h_b, tag=f"{tag}_ar", symbols=[c_a])
        rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        pa = p1_probe(ag, world, c_a, tag=f"{tag}_qa")
        pb = p1_probe(ag, world, c_b, tag=f"{tag}_qb")
        ga = float(ag._act_geometric_margin(pa["p1"], h_b))
        gb = float(ag._act_geometric_margin(pb["p1"], h_b))
        sa = perturb_p1(ag, pa["p1"], h_b, domain=str(world["domain"]), key=f"{tag}_a")
        sb = perturb_p1(ag, pb["p1"], h_b, domain=str(world["domain"]), key=f"{tag}_b")
        ranking_ok = bool(pa["winner"] == h_b and pb["winner"] == h_b)
        gmin = min(ga, gb)
        geometric_ok = bool(gmin >= float(p["margin"]["geometric_min"]))
        pert_ok = bool(sa["stable"] and sb["stable"])
        passed = bool(ranking_ok and geometric_ok and pert_ok)
    out = {
        "kind": "spec",
        "order": order,
        "n_cues": 2,
        "id": cell_id,
        "passed": passed,
        "ranking_ok": ranking_ok,
        "geometric_ok": geometric_ok,
        "perturbation_ok": pert_ok,
        "min_normalized_geometric_margin": float(gmin),
        "normalized_geometric_margin": float(gmin),
        "a_winner": pa["winner"],
        "b_winner": pb["winner"],
        "want_a": h_b,
        "want_b": h_b,
        "rest": rest_out,
        "n_episodes": pa["n_episodes"],
        "pass_statistic": "normalized_geometric_margin",
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_neg(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    wrong = world["handles"][1]
    right = world["handles"][0]
    eta = float(load_prereg().get("eta_act") or 0.15)
    with tempfile.TemporaryDirectory(prefix="cp_neg_") as tmp:
        ag = _fresh(tmp, "s", world)
        w0 = ag.W_act_query.detach().clone()
        wneg = set_handle_delta(world, wrong, NEG_DELTA)
        t = teach_one(ag, wneg, wrong, tag=f"{tag}_n", symbols=[cue])
        adv = float(t["adv"])
        dw = (ag.W_act_query - w0).detach()
        ep = ag._episodes[-1] if ag._episodes else None
        if ep is None:
            raise RuntimeError("neg teach must write an episode")
        p1u = ag._unit_or_zero(np.asarray(ep["p1"], dtype=np.float64))
        mh = ag._to_t(ag.motor_vocab[wrong])
        expected = float(eta) * adv * torch_outer(mh, ag._to_t(p1u))
        max_delta_err = float((dw - expected).abs().max().item())
        live = p1_probe(ag, world, cue, tag=f"{tag}_q")
        passed = bool(adv < 0.0 and max_delta_err < 1e-9 and live["winner"] != wrong)
    out = {
        "kind": "neg",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "adv": adv,
        "max_delta_err": max_delta_err,
        "winner": live["winner"],
        "wrong": wrong,
        "right": right,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def torch_outer(a, b):
    import torch

    return torch.outer(a, b)


def eval_perm(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    cue, handle = pairs[0]
    with tempfile.TemporaryDirectory(prefix="cp_perm_") as tmp:
        ag = _fresh(tmp, "s", world)
        teach_one(ag, world, handle, tag=f"{tag}_t", symbols=[cue])
        before = p1_probe(ag, world, cue, tag=f"{tag}_b")
        ag.bind_actuators(list(reversed(list(world["handles"]))))
        after = p1_probe(ag, world, cue, tag=f"{tag}_a")
        passed = bool(before["winner"] == handle and after["winner"] == handle)
    out = {
        "kind": "perm",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "winner_before": before["winner"],
        "winner_after": after["winner"],
        "want": handle,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_ckpt(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    half = max(1, len(pairs) // 2)
    first, second = pairs[:half], pairs[half:]
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="cp_ckpt_") as tmp:
        ag_ctrl = _fresh(tmp, "ctrl", world)
        teach_pairs(ag_ctrl, world, pairs, tag=f"{tag}_full")[0]
        ctrl = probe_map(ag_ctrl, world, pairs, tag=f"{tag}_ctrl", domain=str(world["domain"]))
        ag = _fresh(tmp, "split", world)
        teach_pairs(ag, world, first, tag=f"{tag}_a")[0]
        snap = ag.checkpoint()
        ag2 = _fresh(tmp, "load", world)
        ag2.load_checkpoint(snap)
        teach_pairs(ag2, world, second, tag=f"{tag}_b")[0]
        if len(first) < len(pairs):
            ag2.rest_epoch(int(p["n_rest_ticks"]))
        live = probe_map(ag2, world, pairs, tag=f"{tag}_live", domain=str(world["domain"]))
        passed = bool(live["ranking_ok"] == ctrl["ranking_ok"])
    out = {
        "kind": "ckpt",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "ctrl_ranking_ok": ctrl["ranking_ok"],
        "live_ranking_ok": live["ranking_ok"],
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_noteach(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory(prefix="cp_nt_") as tmp:
        ag = _fresh(tmp, "s", world)
        winners = []
        for i, (cue, _h) in enumerate(pairs):
            live = p1_probe(ag, world, cue, tag=f"{tag}_{i}")
            winners.append(live["winner"])
        counts: dict[str, int] = {}
        for w in winners:
            if w is not None:
                counts[str(w)] = counts.get(str(w), 0) + 1
        dominant = max(counts.values()) if counts else 0
        passed = bool(dominant <= 1)
    out = {
        "kind": "noteach",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "winners": winners,
        "dominant_count": dominant,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_hold(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cp_hold_") as tmp:
        ag = _fresh(tmp, "s", world)
        hold_seen = False
        max_delta = 0.0
        for attempt in range(128):
            ag._pending = None
            ag._pred_pending = None
            w_before = ag.W_act_query.detach().clone()
            out = ag.observe(
                build_observe(
                    interaction_token=f"{tag}_{attempt}",
                    source_token="src_hold",
                    ordered_symbols=[world["symbols"][0]],
                    observable_state=["st_idle"],
                    body_state=list(MID_BODY),
                )
            )
            action = out.get("action") or {}
            if str(action.get("op")) == "HOLD":
                hold_seen = True
                max_delta = float((ag.W_act_query - w_before).abs().max().item())
                break
        passed = bool(hold_seen and max_delta <= 0.0)
    out = {
        "kind": "hold",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "hold_seen": hold_seen,
        "max_w_delta": float(max_delta),
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    def rows(kind: str, *, prefix: str | None = None, scale: bool = False) -> list[dict[str, Any]]:
        out = []
        for c in cells:
            if c["kind"] != kind:
                continue
            cid = str(c.get("id") or "")
            if scale and not cid.startswith("scale|"):
                continue
            if not scale and cid.startswith("scale|"):
                continue
            if prefix and not cid.startswith(prefix):
                continue
            out.append(c)
        return out

    def all_pass(rs: list[dict[str, Any]]) -> bool:
        return bool(rs) and all(bool(c["passed"]) for c in rs)

    flags = {
        "core_acquire_2": all_pass([c for c in cells if c["kind"] == "acquire" and int(c.get("n_cues") or 0) == 2 and not str(c["id"]).startswith("scale|")]),
        "core_acquire_4": all_pass([c for c in cells if c["kind"] == "acquire" and int(c.get("n_cues") or 0) == 4]),
        "core_acquire_8": all_pass([c for c in cells if c["kind"] == "acquire" and int(c.get("n_cues") or 0) == 8 and not str(c["id"]).startswith("scale|")]),
        "core_stable": all_pass([c for c in cells if c["kind"] == "stable" and not str(c["id"]).startswith("scale|")]),
        "twin": all_pass([c for c in cells if c["kind"] == "twin"]),
        "eco": all_pass([c for c in cells if c["kind"] == "eco"]),
        "spec": all_pass([c for c in cells if c["kind"] == "spec"]),
        "integrity": all_pass([c for c in cells if c["kind"] in ("neg", "perm", "ckpt", "noteach", "hold")]),
        "scale_acquire": all_pass([c for c in cells if c["kind"] == "acquire" and str(c["id"]).startswith("scale|")]),
        "scale_stable": all_pass([c for c in cells if c["kind"] == "stable" and str(c["id"]).startswith("scale|")]),
    }
    if not flags["core_acquire_2"] or not flags["core_acquire_4"] or not flags["core_acquire_8"]:
        return "gated_rehearsal_core_acquire_fail", "gated_rehearsal_core_acquire_fail", flags
    if not flags["core_stable"] or not flags["twin"]:
        return "gated_rehearsal_core_stability_fail", "gated_rehearsal_core_stability_fail", flags
    if not flags["eco"]:
        return "gated_rehearsal_reversal_fail", "gated_rehearsal_reversal_fail", flags
    if not flags["spec"]:
        return "gated_rehearsal_specificity_fail", "gated_rehearsal_specificity_fail", flags
    if not flags["integrity"]:
        return "gated_rehearsal_integrity_fail", "gated_rehearsal_integrity_fail", flags
    if not flags["scale_acquire"]:
        return "gated_rehearsal_multiactuator_acquire_fail", "gated_rehearsal_multiactuator_acquire_fail", flags
    if not flags["scale_stable"]:
        return "gated_rehearsal_multiactuator_stability_fail", "gated_rehearsal_multiactuator_stability_fail", flags
    return "gated_rehearsal_battery_pass", "reopen_lineage_readiness", flags


def eval_dev_battery() -> dict[str, Any]:
    p = load_prereg()
    frozen_runner_sha = str(p.get("frozen_runner_sha") or "")
    if frozen_runner_sha:
        refuse_runner_mutation(frozen_runner_sha)
    cells: list[dict[str, Any]] = []
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                acq_id = f"acquire|c{n_cues}|{order}|w{wi}"
                acq = eval_acquire_stable(
                    kind="acquire",
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"acq_c{n_cues}_{order}_w{wi}",
                    rest=False,
                    cell_id=acq_id,
                )
                acq["world"] = wi
                acq["domain"] = DEV_DOMAIN
                cells.append(acq)
                st_id = f"stable|c{n_cues}|{order}|w{wi}"
                st = eval_acquire_stable(
                    kind="stable",
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"st_c{n_cues}_{order}_w{wi}",
                    rest=True,
                    cell_id=st_id,
                )
                st["world"] = wi
                st["domain"] = DEV_DOMAIN
                cells.append(st)
    for wi in range(int(p["n_worlds"])):
        twin_w = capacity_world(wi, TWIN_DOMAIN, n_cues=2, n_handles=2)
        pairs = mapping_pairs(twin_w, flip=False)
        for order in list(p["orders"]):
            tw_id = f"twin|c2|{order}|w{wi}"
            tw = eval_acquire_stable(
                kind="acquire",
                world=twin_w,
                pairs=pairs,
                order=order,
                tag=f"twin_{order}_w{wi}",
                rest=False,
                cell_id=tw_id,
            )
            tw["kind"] = "twin"
            tw["world"] = wi
            tw["domain"] = TWIN_DOMAIN
            cells.append(tw)
            eco_w = capacity_world(wi, DEV_DOMAIN, n_cues=2, n_handles=2)
            eco_id = f"eco|{order}|w{wi}"
            eco = eval_eco(eco_w, order=order, tag=f"eco_{order}_w{wi}", cell_id=eco_id)
            eco["world"] = wi
            eco["domain"] = DEV_DOMAIN
            cells.append(eco)
            spec_id = f"spec|{order}|w{wi}"
            spec = eval_spec(eco_w, order=order, tag=f"spec_{order}_w{wi}", cell_id=spec_id)
            spec["world"] = wi
            spec["domain"] = DEV_DOMAIN
            cells.append(spec)
    for wi in range(int(p["n_worlds"])):
        w = capacity_world(wi, DEV_DOMAIN, n_cues=2, n_handles=2)
        for gate, fn in (
            ("neg", eval_neg),
            ("perm", eval_perm),
            ("ckpt", eval_ckpt),
            ("noteach", eval_noteach),
            ("hold", eval_hold),
        ):
            cid = f"{gate}|w{wi}"
            cell = fn(w, wi=wi, tag=f"{gate}_w{wi}", cell_id=cid)
            cell["domain"] = DEV_DOMAIN
            cells.append(cell)
    for spec in p["scaling"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                acq_id = f"scale|acquire|c{n_cues}h{n_handles}|{order}|w{wi}"
                acq = eval_acquire_stable(
                    kind="acquire",
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"scale_acq_{order}_w{wi}",
                    rest=False,
                    cell_id=acq_id,
                )
                acq["world"] = wi
                acq["domain"] = DEV_DOMAIN
                cells.append(acq)
                st_id = f"scale|stable|c{n_cues}h{n_handles}|{order}|w{wi}"
                st = eval_acquire_stable(
                    kind="stable",
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"scale_st_{order}_w{wi}",
                    rest=True,
                    cell_id=st_id,
                )
                st["world"] = wi
                st["domain"] = DEV_DOMAIN
                cells.append(st)
    ids = [str(c["id"]) for c in cells]
    expect = expected_cell_ids()
    if sorted(ids) != sorted(expect):
        raise RuntimeError(f"cell id mismatch {len(ids)} vs {len(expect)}")
    if SCORE_DOMAIN in json.dumps(cells):
        raise RuntimeError("SCORE domain leaked into DEV")
    code, then, flags = _decision(cells, p)
    env = torch_env()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    return {
        "version": "TM.0.27.GATEDREHEARSAL.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "n_cells": len(cells),
        "manifest_sha": MANIFEST_SHA,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain_opened": False,
        "neural_edit": True,
        "gated_rehearsal": True,
        "act_score_mode": ACT_SCORE_QUERY,
        "episode_slots": EPISODE_SLOTS,
        "match_l2": EPISODE_MATCH_L2,
        "replay_epochs": EPISODE_REPLAY_EPOCHS,
        "pass_statistic": "normalized_geometric_margin",
        "geometric_min": ACT_MARGIN_FLOOR,
        "decision_code": code,
        "decision_then": then,
        "phase_flags": flags,
        "cells": cells,
        "shas": gated_shas(),
        "env": env,
        "git_head": git_head,
        "frozen_runner_sha": sha_file(THIS),
        "note": "v34 gated competitive rehearsal organism battery. SCORE unopened. Product remains 0.0.004.",
    }


def run_dev() -> dict[str, Any]:
    refuse_dev_lock()
    return eval_dev_battery()


def write_dev_lock(out: dict[str, Any]) -> None:
    refuse_dev_lock()
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_decision(out: dict[str, Any]) -> None:
    dec = {
        "version": "TM.0.27.GATEDREHEARSAL.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "neural_edit": True,
        "gated_rehearsal": True,
        "pass_statistic": "normalized_geometric_margin",
        "manifest_sha": MANIFEST_SHA,
        "decision": {
            "code": out["decision_code"],
            "then": out["decision_then"],
            "phase_flags": out["phase_flags"],
        },
        "dev_lock_sha": hashlib.sha256(DEV_LOCK.read_bytes()).hexdigest() if DEV_LOCK.exists() else None,
        "git_head": out.get("git_head"),
        "frozen_runner_sha": out.get("frozen_runner_sha"),
        "lineage_reopened": False,
        "candidate_v34_lock_written": False,
        "note": "v34 gated competitive rehearsal battery. SCORE unopened. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(dec, indent=2) + "\n", encoding="utf-8")


def write_results(out: dict[str, Any]) -> None:
    if RESULT_MD.exists():
        return
    flags = out["phase_flags"]
    lines = [
        "# TM.0.27.GATEDREHEARSAL DEV",
        "",
        f"Decision: **{out['decision_code']}**.",
        "",
        "v34 prediction-error-gated competitive rehearsal at P1. Unused `TM027.GATEDREHEARSAL.DEV.` / `TWIN.`. "
        "SCORE unopened. Product **0.0.004**. `earned_next=false`. Lineage stays closed.",
        "",
        f"Phase flags: `{flags}`.",
        "",
        "Same frozen DEV execution refused.",
        "",
    ]
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def smoke() -> dict[str, Any]:
    p = load_prereg()
    assert p["n"] == 64
    assert p["expected_n_cells"] == EXPECTED_N_CELLS
    assert p["manifest_sha"] == MANIFEST_SHA
    assert len(expected_cell_ids()) == EXPECTED_N_CELLS
    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory(prefix="gr_smk_") as tmp:
        ag = _fresh(tmp, "s", world)
        assert len(ag._episodes) == 0
        assert float(ag.W_act_query.abs().max().item()) == 0.0
        cue, handle = world["cue_handle"][0]["cue"], world["cue_handle"][0]["handle"]
        t = teach_one(ag, world, handle, tag="smk", symbols=[cue])
        assert abs(float(t["adv"])) > 0.0
        assert len(ag._episodes) >= 1
        assert ag._last_p1 is not None
        assert t.get("rehearsal_burst") is not None
        rest = ag.rest_epoch(int(p["n_rest_ticks"]))
        live = p1_probe(ag, world, cue, tag="smk_p")
    return {
        "smoke_ok": True,
        "expected_id_count": EXPECTED_N_CELLS,
        "manifest_sha": MANIFEST_SHA,
        "n_episodes": int(live["n_episodes"]),
        "winner": live["winner"],
        "want": handle,
        "ranking_ok": bool(live["winner"] == handle),
        "adv": float(t["adv"]),
        "n_replay": int(rest.get("n_replay") or 0),
        "n_strengthen": int(rest.get("n_strengthen") or 0),
        "neural_edit": True,
        "gated_rehearsal": True,
        "v34_candidate_exists": (REPO_ROOT / "docs" / "cortex.candidate.v34.lock").exists(),
        "tau": float(ag.genome.tau),
        "n": int(ag.genome.n),
        "act_score_mode": str(ag.genome.act_score_mode),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run-dev", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=_json_default))
        return
    if args.dev or args.run_dev:
        out = run_dev()
        write_dev_lock(out)
        write_decision(out)
        write_results(out)
        print(json.dumps({"decision": out["decision_code"], "n_cells": out["n_cells"], "flags": out["phase_flags"]}, indent=2))
        return
    refuse_score()


if __name__ == "__main__":
    main()
