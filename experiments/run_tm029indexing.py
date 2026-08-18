"""TM.0.29.INDEXING — v36 hippocampal indexing battery.

Standalone runner. Does not patch TM028 or TM027 module globals.
Product 0.0.004.
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

from experiments.run_tm023cortex import build_observe, make_cortex, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024convergencemap import unique_winner
from experiments.run_tm024statemap import prep_eval
from experiments.run_tm024writegeom import (
    NEG_DELTA,
    capacity_world,
    mapping_pairs,
    ranking_margin,
    set_handle_delta,
)
from experiments.run_tm027gatedrehearsal import (
    _aggregate_awake_bursts,
    _aggregate_rest_rehearsal,
    assert_finite_record,
    classify_failure,
    domain_seed,
    teach_one,
    teach_pairs,
    torch_outer,
)
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW,
    ACT_RECALL_MODES,
    ACT_RECALL_OFF,
    ACT_RECALL_RAW_P1,
    ACT_RECALL_SEP,
    ACT_RECALL_SEP_NO_FAM,
    ACT_SCORE_QUERY,
    KEY_MATCH_MIN_OVERLAP,
    PROTO_EPS,
    SEPARATOR_MATRIX_SHA,
    NeuralCortex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
LAB = "TM.0.29.INDEXING"
PREREG = REPO_ROOT / "docs" / "lineage_indexing.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_indexing_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_indexing.isolation.lock"
V36_PREREG = REPO_ROOT / "docs" / "cortex_v36.prereg.lock"
V36_ISO = REPO_ROOT / "docs" / "cortex_v36.isolation.lock"
V36_AMEND = REPO_ROOT / "docs" / "cortex_v36_architecture_amendment.md"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_indexing.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_indexing.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm029indexing_results.md"

DEV_DOMAIN = "TM029.INDEXING.DEV."
TWIN_DOMAIN = "TM029.INDEXING.TWIN."
SCORE_DOMAIN = "TM029.INDEXING.SCORE."
EXPECTED_N_CELLS = 82
MANIFEST_SHA = "4ac2fd49c9a27e40ad13c9ed52b9d862900b2ff07e8ea7d0d94df8ca98797bca"
TREATMENT_MODE = ACT_RECALL_SEP
RECALL_MODES = list(ACT_RECALL_MODES)


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _fresh(tmp: str, tag: str, world: dict[str, Any], *, mode: str = TREATMENT_MODE) -> NeuralCortex:
    root = Path(tmp) / tag
    ag = make_cortex(root, device="cpu")
    ag.genome.act_recall_mode = str(mode)
    ag.genome.episodic_act_recall = str(mode) == ACT_RECALL_RAW_P1
    ag.bind_actuators(list(world["handles"]))
    if str(ag.genome.act_score_mode) != ACT_SCORE_QUERY:
        raise RuntimeError("v36 default scoring must remain query")
    return ag


def clone_recall_mode(ag: NeuralCortex, *, mode: str) -> NeuralCortex:
    if mode not in ACT_RECALL_MODES:
        raise ValueError(mode)
    snap = ag.checkpoint()
    twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device=str(ag.device))
    twin.load_checkpoint(snap)
    twin.genome.act_recall_mode = str(mode)
    twin.genome.episodic_act_recall = str(mode) == ACT_RECALL_RAW_P1
    prep_eval(twin)
    return twin


def probe_geometric_margin(ag: NeuralCortex, score_addr: np.ndarray, scores: dict[str, float], handle: str) -> float:
    others = [h for h in scores if h != handle]
    if not others:
        return 0.0
    rival = max(others, key=lambda h: float(scores[h]))
    w_ch = ag._act_effective_row(handle)
    w_ot = ag._act_effective_row(rival)
    x = ag._unit_or_zero(score_addr)
    d = w_ch - w_ot
    dn = float(np.linalg.norm(d))
    if dn <= PROTO_EPS:
        return 0.0
    return float(np.dot(d, x) / dn)


def motor_probe(ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str) -> dict[str, Any]:
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    p1 = probe._last_p1
    if p1 is None:
        p1 = probe._from_t(probe.rho)
    scores, score_addr, recall_meta = probe.actuator_decision_scores(p1)
    winner = unique_winner(scores)
    gap = ranking_margin(scores, winner) if winner else 0.0
    gamma = probe_geometric_margin(probe, score_addr, scores, winner) if winner else 0.0
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "winner": winner,
        "pairwise_score_gap": float(gap),
        "normalized_geometric_margin": float(gamma),
        "p1": np.asarray(p1, dtype=np.float64).copy(),
        "key_rho": None if probe._last_key_rho is None else np.asarray(probe._last_key_rho, dtype=np.float64).copy(),
        "event_key": None if probe._last_event_key is None else np.asarray(probe._last_event_key, dtype=np.float64).copy(),
        "scoring_address": np.asarray(score_addr, dtype=np.float64).copy(),
        "recall_meta": dict(recall_meta),
        "n_episodes": len(probe._episodes),
    }


def _perturb_uses_key_source(mode: str) -> bool:
    return mode in (ACT_RECALL_EARLY_RAW, ACT_RECALL_SEP, ACT_RECALL_SEP_NO_FAM)


def perturb_motor(
    ag: NeuralCortex,
    live: dict[str, Any],
    want: str,
    *,
    domain: str,
    key: str,
) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    mode = ag._resolve_act_recall_mode()
    use_key = _perturb_uses_key_source(mode) and live.get("key_rho") is not None
    src = np.asarray(live["key_rho"] if use_key else live["p1"], dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(src)) + 1e-12
    r_hat = src / nrm
    p1 = np.asarray(live["p1"], dtype=np.float64)
    if use_key:
        base_scores, _addr, base_meta = ag.actuator_decision_scores(p1, key_rho=r_hat)
    else:
        base_scores, _addr, base_meta = ag.actuator_decision_scores(r_hat)
    base_slot = base_meta.get("slot")
    n_ok = 0
    n_id = 0
    trials: list[dict[str, Any]] = []
    for i in range(n):
        unit = r_hat + rng.normal(0.0, sigma, size=r_hat.shape)
        pn = float(np.linalg.norm(unit)) + 1e-12
        unit = unit / pn
        if use_key:
            scores, _score_addr, meta = ag.actuator_decision_scores(p1, key_rho=unit)
        else:
            scores, _score_addr, meta = ag.actuator_decision_scores(unit)
        ok = unique_winner(scores) == want
        ident = bool(base_slot is not None and meta.get("slot") == base_slot)
        n_ok += int(ok)
        n_id += int(ident)
        trials.append(
            {
                "trial": int(i),
                "winner": unique_winner(scores),
                "ranking_ok": bool(ok),
                "slot": meta.get("slot"),
                "identity_survived": ident,
                "overlap": meta.get("overlap"),
                "familiar": meta.get("familiar"),
                "path": meta.get("path"),
            }
        )
    return {
        "n_ok": n_ok,
        "n": n,
        "stable": n_ok >= need,
        "n_identity": int(n_id),
        "identity_stable": n_id >= need,
        "base_slot": base_slot,
        "perturbed_key_source": bool(use_key),
        "trials": trials,
    }


def probe_map_motor(
    ag: NeuralCortex,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    tag: str,
    domain: str,
) -> dict[str, Any]:
    probes = []
    ranking_ok = True
    n_probe_correct = 0
    gammas = []
    gaps = []
    pert_ok = True
    geo_min = float(load_prereg()["margin"]["geometric_min"])
    for i, (cue, handle) in enumerate(pairs):
        live = motor_probe(ag, world, cue, tag=f"{tag}_p{i}")
        rank = bool(live["winner"] == handle)
        ranking_ok = ranking_ok and rank
        n_probe_correct += int(rank)
        g = probe_geometric_margin(ag, live["scoring_address"], live["scores"], handle)
        if not math.isfinite(g):
            raise RuntimeError(f"non-finite margin probe {cue}")
        gammas.append(g)
        gaps.append(float(live["pairwise_score_gap"]))
        stab = perturb_motor(ag, live, handle, domain=domain, key=f"{tag}_{cue}")
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
                "recall_meta": live["recall_meta"],
                "perturbation": {
                    "n_ok": stab["n_ok"],
                    "n": stab["n"],
                    "stable": stab["stable"],
                    "n_identity": stab["n_identity"],
                    "identity_stable": stab["identity_stable"],
                    "base_slot": stab["base_slot"],
                    "perturbed_key_source": stab["perturbed_key_source"],
                    "trials": stab["trials"],
                },
            }
        )
    min_g = min(gammas) if gammas else 0.0
    return {
        "probes": probes,
        "ranking_ok": ranking_ok,
        "n_probe_correct": int(n_probe_correct),
        "n_cues": int(world["capacity"]["n_cues"]),
        "probe_count_matches_cues": bool(n_probe_correct == len(pairs)),
        "perturbation_ok": pert_ok,
        "geometric_ok": bool(min_g >= geo_min),
        "min_normalized_geometric_margin": float(min_g),
        "min_pairwise_score_gap": float(min(gaps) if gaps else 0.0),
        "n_episodes": len(ag._episodes),
    }


def novel_reject_ok(meta: dict[str, Any]) -> bool:
    return str(meta.get("path")) == "cortical_fallback" and meta.get("familiar") is False


def _synthetic_symbol(domain: str, key: str, *, suffix: str) -> str:
    h = hashlib.sha256(f"{domain}:{key}:{suffix}".encode()).hexdigest()[:12]
    return f"s_{suffix}_{h}"


def _extra_sym(world: dict[str, Any], pairs: list[tuple[str, str]], *, key: str, suffix: str) -> str:
    taught = {c for c, _h in pairs}
    for s in world.get("symbols") or []:
        if s not in taught:
            return str(s)
    return _synthetic_symbol(str(world["domain"]), key, suffix=suffix)


def eval_acquire_stable(
    *,
    kind: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool,
    cell_id: str,
    mode: str = TREATMENT_MODE,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="ix_cap_") as tmp:
        ag = _fresh(tmp, "s", world, mode=mode)
        taught, burst_raw = teach_pairs(ag, world, seq, tag=tag)
        stored_post_awake = ag.store_rehearsal_checkpoint()
        awake_diag = _aggregate_awake_bursts(burst_raw)
        rest_out = None
        stored_post_rest_rehearsal_pre_mix = None
        stored_post_slow_mix = None
        if rest:
            rest_out = ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
            rh = rest_out.get("rehearsal") or {}
            stored_post_rest_rehearsal_pre_mix = {
                "n_violations": int(rh.get("violations_pre_mix") or 0),
                "all_margin_ok": bool(int(rh.get("violations_pre_mix") or 0) == 0),
                "n_episodes": int(stored_post_awake.get("n_episodes") or 0),
            }
            stored_post_slow_mix = ag.store_rehearsal_checkpoint()
        rest_diag = _aggregate_rest_rehearsal(rest_out)
        probed = probe_map_motor(ag, world, pairs, tag=tag, domain=str(world["domain"]))
    if kind == "acquire":
        passed = bool(probed["ranking_ok"] and probed["probe_count_matches_cues"])
        live_ranking_ok = passed
        live_geometric_ok = passed
        live_perturbation_ok = True
    else:
        passed = bool(probed["ranking_ok"] and probed["geometric_ok"] and probed["perturbation_ok"])
        live_ranking_ok = bool(probed["ranking_ok"])
        live_geometric_ok = bool(probed["geometric_ok"])
        live_perturbation_ok = bool(probed["perturbation_ok"])
    pre_mix = stored_post_rest_rehearsal_pre_mix or stored_post_awake
    post_mix = stored_post_slow_mix or stored_post_awake
    failure_class = classify_failure(
        stored_pre_mix=pre_mix,
        stored_post_mix=post_mix,
        live_ranking_ok=live_ranking_ok,
        live_geometric_ok=live_geometric_ok,
        live_perturbation_ok=live_perturbation_ok,
    )
    out = {
        "kind": kind,
        "order": order,
        "n_cues": int(world["capacity"]["n_cues"]),
        "id": cell_id,
        "taught": taught,
        "rest": rest_out,
        "passed": passed,
        "behavioral_pass": passed,
        "act_recall_mode": mode,
        "pass_statistic": "normalized_geometric_margin",
        "stored_post_awake": stored_post_awake,
        "stored_post_rest_rehearsal_pre_mix": stored_post_rest_rehearsal_pre_mix,
        "stored_post_slow_mix": stored_post_slow_mix,
        "failure_class": failure_class,
        **awake_diag,
        **rest_diag,
        **probed,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_hist_stable(
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    order: str,
    tag: str,
    cell_id: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    hist_sym = _extra_sym(world, pairs, key=tag, suffix="hist")
    with tempfile.TemporaryDirectory(prefix="ix_hist_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        snap = ag.checkpoint()
        twin = clone_recall_mode(ag, mode=TREATMENT_MODE)
        twin.load_checkpoint(snap)
        observe_cue(twin, world, tag=f"{tag}_hist", body=list(MID_BODY), symbols=[hist_sym])
        prep_eval(twin)
        probed = probe_map_motor(twin, world, pairs, tag=f"{tag}_probe", domain=str(world["domain"]))
    passed = bool(probed["ranking_ok"] and probed["geometric_ok"] and probed["perturbation_ok"])
    out = {
        "kind": "hist",
        "id": cell_id,
        "order": order,
        "passed": passed,
        "behavioral_pass": passed,
        "history_symbol": hist_sym,
        "act_recall_mode": TREATMENT_MODE,
        **probed,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_novel_stable(
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    order: str,
    tag: str,
    cell_id: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    novel = _extra_sym(world, pairs, key=tag, suffix="novel")
    with tempfile.TemporaryDirectory(prefix="ix_novel_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        live = motor_probe(ag, world, novel, tag=f"{tag}_n")
    rejected = novel_reject_ok(live["recall_meta"])
    overgeneralization = not rejected
    out = {
        "kind": "novel",
        "id": cell_id,
        "order": order,
        "passed": rejected,
        "behavioral_pass": rejected,
        "novel_symbol": novel,
        "winner": live["winner"],
        "overgeneralization": overgeneralization,
        "recall_meta": live["recall_meta"],
        "act_recall_mode": TREATMENT_MODE,
        "normalized_geometric_margin": float(live["normalized_geometric_margin"]),
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_ablation_matched(
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    order: str,
    tag: str,
    cell_id: str,
    mode: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    novel = _extra_sym(world, pairs, key=f"{tag}_{mode}", suffix="novel")
    with tempfile.TemporaryDirectory(prefix="ix_abl_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        twin = clone_recall_mode(ag, mode=mode)
        probed = probe_map_motor(twin, world, pairs, tag=f"{tag}_{mode}", domain=str(world["domain"]))
        live_novel = motor_probe(twin, world, novel, tag=f"{tag}_{mode}_n")
    stable_pass = bool(probed["ranking_ok"] and probed["geometric_ok"] and probed["perturbation_ok"])
    novel_ok = novel_reject_ok(live_novel["recall_meta"])
    fc = classify_failure(
        stored_pre_mix={"all_margin_ok": True},
        stored_post_mix={"all_margin_ok": True},
        live_ranking_ok=bool(probed["ranking_ok"]),
        live_geometric_ok=bool(probed["geometric_ok"]),
        live_perturbation_ok=bool(probed["perturbation_ok"]),
    )
    out = {
        "kind": "ablation",
        "id": cell_id,
        "order": order,
        "passed": False,
        "behavioral_pass": None,
        "act_recall_mode": mode,
        "stable_gate_passed": stable_pass,
        "stable_gate_failed": not stable_pass,
        "novel_rejected": novel_ok,
        "novel_overgeneralization": not novel_ok,
        "novel_symbol": novel,
        "novel_winner": live_novel["winner"],
        "novel_recall_meta": live_novel["recall_meta"],
        "failure_class": fc,
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
    with tempfile.TemporaryDirectory(prefix="ix_eco_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        t1 = teach_one(ag, world, h1, tag=f"{tag}_p", symbols=[cue])
        wneg = set_handle_delta(world, h1, NEG_DELTA)
        t2 = teach_one(ag, wneg, h1, tag=f"{tag}_n", symbols=[cue])
        t3 = teach_one(ag, world, h2, tag=f"{tag}_r", symbols=[cue])
        n_replaced = int(ag._episode_n_replaced)
        rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        live = motor_probe(ag, world, cue, tag=f"{tag}_q")
        g = probe_geometric_margin(ag, live["scoring_address"], live["scores"], h2)
        stab = perturb_motor(ag, live, h2, domain=str(world["domain"]), key=f"{tag}_eco")
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
        "behavioral_pass": passed,
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
        "act_recall_mode": TREATMENT_MODE,
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
    with tempfile.TemporaryDirectory(prefix="ix_spec_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        teach_one(ag, world, h_a, tag=f"{tag}_a", symbols=[c_a])
        teach_one(ag, world, h_b, tag=f"{tag}_b", symbols=[c_b])
        wneg = set_handle_delta(world, h_a, NEG_DELTA)
        teach_one(ag, wneg, h_a, tag=f"{tag}_n", symbols=[c_a])
        teach_one(ag, world, h_b, tag=f"{tag}_ar", symbols=[c_a])
        rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        pa = motor_probe(ag, world, c_a, tag=f"{tag}_qa")
        pb = motor_probe(ag, world, c_b, tag=f"{tag}_qb")
        ga = probe_geometric_margin(ag, pa["scoring_address"], pa["scores"], h_b)
        gb = probe_geometric_margin(ag, pb["scoring_address"], pb["scores"], h_b)
        sa = perturb_motor(ag, pa, h_b, domain=str(world["domain"]), key=f"{tag}_a")
        sb = perturb_motor(ag, pb, h_b, domain=str(world["domain"]), key=f"{tag}_b")
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
        "behavioral_pass": passed,
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
        "act_recall_mode": TREATMENT_MODE,
        "pass_statistic": "normalized_geometric_margin",
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_neg(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    wrong = world["handles"][1]
    right = world["handles"][0]
    eta = float(load_prereg().get("eta_act") or 0.15)
    with tempfile.TemporaryDirectory(prefix="ix_neg_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
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
        live = motor_probe(ag, world, cue, tag=f"{tag}_q")
        passed = bool(adv < 0.0 and max_delta_err < 1e-9 and live["winner"] != wrong)
    out = {
        "kind": "neg",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "behavioral_pass": passed,
        "adv": adv,
        "max_delta_err": max_delta_err,
        "winner": live["winner"],
        "wrong": wrong,
        "right": right,
        "act_recall_mode": TREATMENT_MODE,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_perm(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    cue, handle = pairs[0]
    with tempfile.TemporaryDirectory(prefix="ix_perm_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        teach_one(ag, world, handle, tag=f"{tag}_t", symbols=[cue])
        before = motor_probe(ag, world, cue, tag=f"{tag}_b")
        ag.bind_actuators(list(reversed(list(world["handles"]))))
        after = motor_probe(ag, world, cue, tag=f"{tag}_a")
        passed = bool(before["winner"] == handle and after["winner"] == handle)
    out = {
        "kind": "perm",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "behavioral_pass": passed,
        "winner_before": before["winner"],
        "winner_after": after["winner"],
        "want": handle,
        "act_recall_mode": TREATMENT_MODE,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_ckpt(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    half = max(1, len(pairs) // 2)
    first, second = pairs[:half], pairs[half:]
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="ix_ckpt_") as tmp:
        ag_ctrl = _fresh(tmp, "ctrl", world, mode=TREATMENT_MODE)
        teach_pairs(ag_ctrl, world, pairs, tag=f"{tag}_full")
        ctrl = probe_map_motor(ag_ctrl, world, pairs, tag=f"{tag}_ctrl", domain=str(world["domain"]))
        ag = _fresh(tmp, "split", world, mode=TREATMENT_MODE)
        teach_pairs(ag, world, first, tag=f"{tag}_a")
        snap = ag.checkpoint()
        ag2 = _fresh(tmp, "load", world, mode=TREATMENT_MODE)
        ag2.load_checkpoint(snap)
        teach_pairs(ag2, world, second, tag=f"{tag}_b")
        if len(first) < len(pairs):
            ag2.rest_epoch(int(p["n_rest_ticks"]))
        live = probe_map_motor(ag2, world, pairs, tag=f"{tag}_live", domain=str(world["domain"]))
        passed = bool(live["ranking_ok"] == ctrl["ranking_ok"])
    out = {
        "kind": "ckpt",
        "id": cell_id,
        "world": wi,
        "passed": passed,
        "behavioral_pass": passed,
        "ctrl_ranking_ok": ctrl["ranking_ok"],
        "live_ranking_ok": live["ranking_ok"],
        "act_recall_mode": TREATMENT_MODE,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_noteach(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory(prefix="ix_nt_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        winners = []
        for i, (cue, _h) in enumerate(pairs):
            live = motor_probe(ag, world, cue, tag=f"{tag}_{i}")
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
        "behavioral_pass": passed,
        "winners": winners,
        "dominant_count": dominant,
        "act_recall_mode": TREATMENT_MODE,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def eval_hold(world: dict[str, Any], *, wi: int, tag: str, cell_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ix_hold_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        hold_seen = False
        max_delta = 0.0
        for attempt in range(128):
            ag._pending = None
            ag._pred_pending = None
            w_before = ag.W_act_query.detach().clone()
            out_obs = ag.observe(
                build_observe(
                    interaction_token=f"{tag}_{attempt}",
                    source_token="src_hold",
                    ordered_symbols=[world["symbols"][0]],
                    observable_state=["st_idle"],
                    body_state=list(MID_BODY),
                )
            )
            action = out_obs.get("action") or {}
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
        "behavioral_pass": passed,
        "hold_seen": hold_seen,
        "max_w_delta": float(max_delta),
        "act_recall_mode": TREATMENT_MODE,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


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
    for order in orders:
        for wi in range(n_worlds):
            ids.append(f"hist|stable|c8|{order}|w{wi}")
            ids.append(f"novel|stable|c8|{order}|w{wi}")
            for mode in RECALL_MODES:
                ids.append(f"ablation|{mode}|stable|c8|{order}|w{wi}")
    payload = {"lab": LAB, "domains": sorted(p["domains"].items()), "cell_ids": sorted(ids)}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != MANIFEST_SHA:
        raise RuntimeError("TM029 manifest drifted")
    if len(ids) != EXPECTED_N_CELLS:
        raise RuntimeError(f"expected {EXPECTED_N_CELLS} cells, got {len(ids)}")
    return ids


def _tpass(rs: list[dict[str, Any]]) -> bool:
    return bool(rs) and all(bool(c.get("behavioral_pass") if c.get("behavioral_pass") is not None else c.get("passed")) for c in rs)


def _decision(cells: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    treatment = [c for c in cells if c.get("kind") != "ablation"]
    ablation = [c for c in cells if c.get("kind") == "ablation"]
    novel = [c for c in cells if c.get("kind") == "novel"]

    def abl(mode: str) -> list[dict[str, Any]]:
        return [c for c in ablation if str(c.get("act_recall_mode")) == mode]

    flags = {
        "treatment_acquire_2": _tpass(
            [c for c in treatment if c.get("kind") == "acquire" and int(c.get("n_cues") or 0) == 2 and not str(c["id"]).startswith("scale|")]
        ),
        "treatment_acquire_4": _tpass([c for c in treatment if c.get("kind") == "acquire" and int(c.get("n_cues") or 0) == 4]),
        "treatment_acquire_8": _tpass(
            [c for c in treatment if c.get("kind") == "acquire" and int(c.get("n_cues") or 0) == 8 and not str(c["id"]).startswith("scale|")]
        ),
        "treatment_stable": _tpass(
            [c for c in treatment if c.get("kind") in ("stable", "hist") or str(c.get("id", "")).startswith("stable|")]
        ),
        "treatment_twin": _tpass([c for c in treatment if c.get("kind") == "twin"]),
        "treatment_eco": _tpass([c for c in treatment if c.get("kind") == "eco"]),
        "treatment_spec": _tpass([c for c in treatment if c.get("kind") == "spec"]),
        "treatment_integrity": _tpass(
            [c for c in treatment if c.get("kind") in ("neg", "perm", "ckpt", "noteach", "hold")]
        ),
        "treatment_scale_acquire": _tpass(
            [c for c in treatment if c.get("kind") == "acquire" and str(c["id"]).startswith("scale|")]
        ),
        "treatment_scale_stable": _tpass(
            [c for c in treatment if c.get("kind") == "stable" and str(c["id"]).startswith("scale|")]
        ),
        "novel_pass": _tpass(novel),
        "off_stable_pass": bool(abl(ACT_RECALL_OFF)) and all(bool(c.get("stable_gate_passed")) for c in abl(ACT_RECALL_OFF)),
        "raw_p1_stable_pass": bool(abl(ACT_RECALL_RAW_P1)) and all(bool(c.get("stable_gate_passed")) for c in abl(ACT_RECALL_RAW_P1)),
        "early_raw_stable_pass": bool(abl(ACT_RECALL_EARLY_RAW))
        and all(bool(c.get("stable_gate_passed")) for c in abl(ACT_RECALL_EARLY_RAW)),
        "no_fam_novel_reject": bool(abl(ACT_RECALL_SEP_NO_FAM))
        and all(bool(c.get("novel_rejected")) for c in abl(ACT_RECALL_SEP_NO_FAM)),
        "off_unnecessary": False,
        "raw_p1_reproduced_v35": False,
        "sparse_pattern_separation_supported": False,
        "familiarity_gate_causal": False,
    }
    flags["off_unnecessary"] = bool(flags["off_stable_pass"])
    flags["raw_p1_reproduced_v35"] = not bool(flags["raw_p1_stable_pass"])
    if not flags["treatment_acquire_2"] or not flags["treatment_acquire_4"] or not flags["treatment_acquire_8"]:
        return "indexing_core_acquire_fail", "indexing_core_acquire_fail", flags
    if not flags["treatment_stable"] or not flags["treatment_twin"]:
        return "indexing_core_stability_fail", "indexing_core_stability_fail", flags
    if not flags["treatment_eco"]:
        return "indexing_reversal_fail", "indexing_reversal_fail", flags
    if not flags["treatment_spec"]:
        return "indexing_specificity_fail", "indexing_specificity_fail", flags
    if not flags["treatment_integrity"]:
        return "indexing_integrity_fail", "indexing_integrity_fail", flags
    if not flags["novel_pass"]:
        return "indexing_episodic_overgeneralization", "indexing_episodic_overgeneralization", flags
    if not flags["treatment_scale_acquire"]:
        return "indexing_multiactuator_acquire_fail", "indexing_multiactuator_acquire_fail", flags
    if not flags["treatment_scale_stable"]:
        return "indexing_multiactuator_stability_fail", "indexing_multiactuator_stability_fail", flags
    sparse_ok = bool(not flags["early_raw_stable_pass"])
    fam_ok = bool(flags["novel_pass"] and not flags["no_fam_novel_reject"])
    flags["sparse_pattern_separation_supported"] = sparse_ok
    flags["familiarity_gate_causal"] = fam_ok
    if sparse_ok and fam_ok:
        return "indexing_battery_pass", "indexing_battery_pass", flags
    if not sparse_ok:
        return "indexing_operational_pass__separation_not_causal", "indexing_operational_pass__separation_not_causal", flags
    return "sparse_pattern_separation_supported", "sparse_pattern_separation_supported", flags


def eval_dev_battery() -> dict[str, Any]:
    p = load_prereg()
    frozen = str(p.get("frozen_runner_sha") or "")
    if frozen and sha_file(THIS) != frozen:
        raise RuntimeError("TM029 runner SHA drifted after freeze push")
    if p.get("separator_matrix_sha") != SEPARATOR_MATRIX_SHA:
        raise RuntimeError("separator matrix SHA drifted")
    if int(p.get("key_match_min_overlap") or 0) != KEY_MATCH_MIN_OVERLAP:
        raise RuntimeError("familiarity overlap drifted")
    cells: list[dict[str, Any]] = []
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                for kind, rest in (("acquire", False), ("stable", True)):
                    cid = f"{kind}|c{n_cues}|{order}|w{wi}"
                    cell = eval_acquire_stable(
                        kind=kind,
                        world=world,
                        pairs=pairs,
                        order=order,
                        tag=f"{kind}_c{n_cues}_{order}_w{wi}",
                        rest=rest,
                        cell_id=cid,
                        mode=TREATMENT_MODE,
                    )
                    cell["world"] = wi
                    cell["domain"] = DEV_DOMAIN
                    cells.append(cell)
    for wi in range(int(p["n_worlds"])):
        twin_w = capacity_world(wi, TWIN_DOMAIN, n_cues=2, n_handles=2)
        pairs = mapping_pairs(twin_w, flip=False)
        for order in list(p["orders"]):
            tw = eval_acquire_stable(
                kind="acquire",
                world=twin_w,
                pairs=pairs,
                order=order,
                tag=f"twin_{order}_w{wi}",
                rest=False,
                cell_id=f"twin|c2|{order}|w{wi}",
                mode=TREATMENT_MODE,
            )
            tw["kind"] = "twin"
            tw["world"] = wi
            cells.append(tw)
            eco_w = capacity_world(wi, DEV_DOMAIN, n_cues=2, n_handles=2)
            eco = eval_eco(eco_w, order=order, tag=f"eco_{order}_w{wi}", cell_id=f"eco|{order}|w{wi}")
            cells.append(eco)
            spec = eval_spec(eco_w, order=order, tag=f"spec_{order}_w{wi}", cell_id=f"spec|{order}|w{wi}")
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
            cells.append(fn(w, wi=wi, tag=f"{gate}_w{wi}", cell_id=f"{gate}|w{wi}"))
    for spec in p["scaling"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                for kind, rest in (("acquire", False), ("stable", True)):
                    cid = f"scale|{kind}|c{n_cues}h{n_handles}|{order}|w{wi}"
                    cells.append(
                        eval_acquire_stable(
                            kind=kind,
                            world=world,
                            pairs=pairs,
                            order=order,
                            tag=f"scale_{kind}_{order}_w{wi}",
                            rest=rest,
                            cell_id=cid,
                            mode=TREATMENT_MODE,
                        )
                    )
    for wi in range(int(p["n_worlds"])):
        world = capacity_world(wi, DEV_DOMAIN, n_cues=8, n_handles=2)
        pairs = mapping_pairs(world, flip=False)
        for order in list(p["orders"]):
            hid = f"hist|stable|c8|{order}|w{wi}"
            cells.append(eval_hist_stable(world, pairs, order=order, tag=f"hist_{order}_w{wi}", cell_id=hid))
            nid = f"novel|stable|c8|{order}|w{wi}"
            cells.append(eval_novel_stable(world, pairs, order=order, tag=f"novel_{order}_w{wi}", cell_id=nid))
            for mode in RECALL_MODES:
                aid = f"ablation|{mode}|stable|c8|{order}|w{wi}"
                cells.append(
                    eval_ablation_matched(
                        world, pairs, order=order, tag=f"abl_{order}_w{wi}", cell_id=aid, mode=mode
                    )
                )
    ids = [str(c["id"]) for c in cells]
    expect = expected_cell_ids()
    if sorted(ids) != sorted(expect):
        raise RuntimeError(f"cell id mismatch {len(ids)} vs {len(expect)}")
    code, then, flags = _decision(cells)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    return {
        "version": "TM.0.29.INDEXING.DEV",
        "product": "0.0.004",
        "n_cells": len(cells),
        "manifest_sha": MANIFEST_SHA,
        "domain": DEV_DOMAIN,
        "decision_code": code,
        "decision_then": then,
        "phase_flags": flags,
        "cells": cells,
        "git_head": git_head,
        "frozen_runner_sha": sha_file(THIS),
        "act_recall_mode_treatment": TREATMENT_MODE,
        "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
        "note": "v36 hippocampal indexing battery. Product remains 0.0.004.",
    }


def refuse_dev_lock() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("TM029 DEV lock exists; refused")


def run_dev() -> dict[str, Any]:
    refuse_dev_lock()
    return eval_dev_battery()


def write_dev_lock(out: dict[str, Any]) -> None:
    refuse_dev_lock()
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def write_decision(out: dict[str, Any]) -> None:
    dec = {
        "version": "TM.0.29.INDEXING.DECISION",
        "product": "0.0.004",
        "manifest_sha": MANIFEST_SHA,
        "decision": {"code": out["decision_code"], "then": out["decision_then"], "phase_flags": out["phase_flags"]},
        "dev_lock_sha": hashlib.sha256(DEV_LOCK.read_bytes()).hexdigest() if DEV_LOCK.exists() else None,
        "git_head": out.get("git_head"),
        "frozen_runner_sha": out.get("frozen_runner_sha"),
        "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
        "note": "v36 indexing battery decision. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(dec, indent=2) + "\n", encoding="utf-8")


def freeze_runner_sha() -> None:
    p = load_prereg()
    if p.get("frozen_runner_sha"):
        raise RuntimeError("already frozen")
    p["frozen_runner_sha"] = sha_file(THIS)
    PREREG.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")


def smoke() -> dict[str, Any]:
    assert load_prereg()["manifest_sha"] == MANIFEST_SHA
    assert len(expected_cell_ids()) == EXPECTED_N_CELLS
    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory(prefix="ix_smk_") as tmp:
        ag = _fresh(tmp, "s", world, mode=TREATMENT_MODE)
        cue, handle = world["cue_handle"][0]["cue"], world["cue_handle"][0]["handle"]
        teach_one(ag, world, handle, tag="smk", symbols=[cue])
        live = motor_probe(ag, world, cue, tag="smk_p")
        off = clone_recall_mode(ag, mode=ACT_RECALL_OFF)
        raw = clone_recall_mode(ag, mode=ACT_RECALL_RAW_P1)
    return {
        "smoke_ok": True,
        "manifest_sha": MANIFEST_SHA,
        "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
        "recall_path": live["recall_meta"].get("path"),
        "winner": live["winner"],
        "act_recall_mode": TREATMENT_MODE,
        "clone_off_mode": off.genome.act_recall_mode,
        "clone_raw_mode": raw.genome.act_recall_mode,
        "n_key": 0 if live["event_key"] is None else int(np.sum(live["event_key"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run-dev", action="store_true")
    ap.add_argument("--freeze-runner-sha", action="store_true")
    args = ap.parse_args()
    if args.freeze_runner_sha:
        freeze_runner_sha()
        print(json.dumps({"frozen_runner_sha": sha_file(THIS)}, indent=2))
        return
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=_json_default))
        return
    if args.dev or args.run_dev:
        out = run_dev()
        write_dev_lock(out)
        write_decision(out)
        print(json.dumps({"decision": out["decision_code"], "flags": out["phase_flags"]}, indent=2))
        return
    raise RuntimeError("TM029 SCORE reserved")


if __name__ == "__main__":
    main()
