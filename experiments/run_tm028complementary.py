"""TM.0.28.COMPLEMENTARY — v35 episodic–cortical ACT recall battery.

Matched ON/OFF ablation. Fresh TM028.COMPLEMENTARY.* worlds.
Runner frozen before neural edit. Product 0.0.004.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from experiments import run_tm027gatedrehearsal as gr
from experiments.run_tm023cortex import make_cortex, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024convergencemap import unique_winner
from experiments.run_tm024statemap import prep_eval
from experiments.run_tm024writegeom import capacity_world, mapping_pairs, ranking_margin
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import ACT_SCORE_QUERY, NeuralCortex, PROTO_EPS

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
LAB = "TM.0.28.COMPLEMENTARY"
PREREG = REPO_ROOT / "docs" / "lineage_complementary.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_complementary_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_complementary.isolation.lock"
V35_PREREG = REPO_ROOT / "docs" / "cortex_v35.prereg.lock"
V35_ISO = REPO_ROOT / "docs" / "cortex_v35.isolation.lock"
V35_AMEND = REPO_ROOT / "docs" / "cortex_v35_architecture_amendment.md"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_complementary.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_complementary.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm028complementary_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"

DEV_DOMAIN = "TM028.COMPLEMENTARY.DEV."
TWIN_DOMAIN = "TM028.COMPLEMENTARY.TWIN."
SCORE_DOMAIN = "TM028.COMPLEMENTARY.SCORE."
EXPECTED_N_CELLS = 66
MANIFEST_SHA = "4c59793a32573143a57b22a10a16728f1be3323b6c2d9b4d11b9b558e42f894c"

_ORIGINAL_ACT_GEOMETRIC_MARGIN = NeuralCortex._act_geometric_margin
_GR_PATCH_ATTRS = (
    "LAB",
    "PREREG",
    "DEV_DOMAIN",
    "TWIN_DOMAIN",
    "SCORE_DOMAIN",
    "MANIFEST_SHA",
    "EXPECTED_N_CELLS",
    "load_prereg",
    "_fresh",
    "p1_probe",
    "perturb_p1",
)


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _apply_gr_patch() -> dict[str, Any]:
    saved = {attr: getattr(gr, attr) for attr in _GR_PATCH_ATTRS}
    gr.LAB = LAB
    gr.PREREG = PREREG
    gr.DEV_DOMAIN = DEV_DOMAIN
    gr.TWIN_DOMAIN = TWIN_DOMAIN
    gr.SCORE_DOMAIN = SCORE_DOMAIN
    gr.MANIFEST_SHA = MANIFEST_SHA
    gr.EXPECTED_N_CELLS = EXPECTED_N_CELLS
    gr.load_prereg = load_prereg
    gr._fresh = lambda tmp, tag, world: _fresh(tmp, tag, world, recall=True)
    gr.p1_probe = motor_probe
    gr.perturb_p1 = perturb_motor

    def _geo_recall(self: NeuralCortex, p1: Any, handle: str) -> float:
        scores, addr, _ = self.actuator_decision_scores(p1)
        return probe_geometric_margin(self, addr, scores, handle)

    NeuralCortex._act_geometric_margin = _geo_recall  # type: ignore[method-assign]
    return saved


def _restore_gr_patch(saved: dict[str, Any]) -> None:
    for attr, val in saved.items():
        setattr(gr, attr, val)
    NeuralCortex._act_geometric_margin = _ORIGINAL_ACT_GEOMETRIC_MARGIN  # type: ignore[method-assign]


@contextmanager
def _gr_patch():
    saved = _apply_gr_patch()
    try:
        yield
    finally:
        _restore_gr_patch(saved)


def _fresh(tmp: str, tag: str, world: dict[str, Any], *, recall: bool = True) -> NeuralCortex:
    root = Path(tmp) / tag
    ag = make_cortex(root, device="cpu")
    ag.genome.episodic_act_recall = bool(recall)
    ag.bind_actuators(list(world["handles"]))
    if str(ag.genome.act_score_mode) != ACT_SCORE_QUERY:
        raise RuntimeError("v35 default scoring must remain query")
    return ag


def clone_recall_variant(ag: NeuralCortex, *, recall: bool) -> NeuralCortex:
    snap = ag.checkpoint()
    twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device=str(ag.device))
    twin.load_checkpoint(snap)
    twin.genome.episodic_act_recall = bool(recall)
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
        "scoring_address": np.asarray(score_addr, dtype=np.float64).copy(),
        "recall_meta": dict(recall_meta),
        "n_episodes": len(probe._episodes),
    }


def perturb_motor(ag: NeuralCortex, p1: np.ndarray, want: str, *, domain: str, key: str) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(gr.domain_seed(domain, key))
    r0 = np.asarray(p1, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r0)) + 1e-12
    r_hat = r0 / nrm
    base_scores, _base_addr, base_meta = ag.actuator_decision_scores(r_hat)
    base_slot = base_meta.get("slot")
    n_ok = 0
    trials: list[dict[str, Any]] = []
    for i in range(n):
        unit = r_hat + rng.normal(0.0, sigma, size=r_hat.shape)
        pn = float(np.linalg.norm(unit)) + 1e-12
        unit = unit / pn
        scores, score_addr, meta = ag.actuator_decision_scores(unit)
        ok = unique_winner(scores) == want
        n_ok += int(ok)
        trials.append(
            {
                "trial": int(i),
                "winner": unique_winner(scores),
                "ranking_ok": bool(ok),
                "slot": meta.get("slot"),
                "identity_survived": bool(base_slot is not None and meta.get("slot") == base_slot),
                "nearest_dist": meta.get("nearest_dist"),
                "path": meta.get("path"),
            }
        )
    return {
        "n_ok": n_ok,
        "n": n,
        "stable": n_ok >= need,
        "base_slot": base_slot,
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
        stab = perturb_motor(ag, live["p1"], handle, domain=domain, key=f"{tag}_{cue}")
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
                "perturbation": stab,
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


def eval_acquire_stable_comp(
    *,
    kind: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool,
    cell_id: str,
    recall: bool = True,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="cp_cap_") as tmp:
        ag = _fresh(tmp, "s", world, recall=recall)
        taught, burst_raw = gr.teach_pairs(ag, world, seq, tag=tag)
        stored_post_awake = ag.store_rehearsal_checkpoint()
        awake_diag = gr._aggregate_awake_bursts(burst_raw)
        rest_out = None
        stored_post_rest_rehearsal_pre_mix = None
        stored_post_slow_mix = None
        if rest:
            rest_out = ag.rest_epoch(int(p.get("n_rest_ticks") or gr.load_prereg().get("n_rest_ticks") or 4))
            rh = rest_out.get("rehearsal") or {}
            stored_post_rest_rehearsal_pre_mix = {
                "n_violations": int(rh.get("violations_pre_mix") or 0),
                "all_margin_ok": bool(int(rh.get("violations_pre_mix") or 0) == 0),
                "n_episodes": int(stored_post_awake.get("n_episodes") or 0),
            }
            stored_post_slow_mix = ag.store_rehearsal_checkpoint()
        rest_diag = gr._aggregate_rest_rehearsal(rest_out)
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
    failure_class = gr.classify_failure(
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
        "behavioral_pass": passed if recall else None,
        "expected_ablation_failure": (not passed) if not recall and kind == "stable" else None,
        "episodic_act_recall": bool(recall),
        "pass_statistic": "normalized_geometric_margin",
        "stored_post_awake": stored_post_awake,
        "stored_post_rest_rehearsal_pre_mix": stored_post_rest_rehearsal_pre_mix,
        "stored_post_slow_mix": stored_post_slow_mix,
        "failure_class": failure_class,
        **awake_diag,
        **rest_diag,
        **probed,
    }
    gr.assert_finite_record(out, ctx=cell_id)
    return out


def _synthetic_symbol(domain: str, key: str, *, suffix: str) -> str:
    h = hashlib.sha256(f"{domain}:{key}:{suffix}".encode()).hexdigest()[:12]
    return f"s_{suffix}_{h}"


def _distractor_sym(world: dict[str, Any], pairs: list[tuple[str, str]], *, key: str) -> str:
    taught = {c for c, _h in pairs}
    for s in world.get("symbols") or []:
        if s not in taught:
            return str(s)
    return _synthetic_symbol(str(world["domain"]), key, suffix="hist")


def _novel_sym(world: dict[str, Any], pairs: list[tuple[str, str]], *, key: str) -> str:
    taught = {c for c, _h in pairs}
    for s in world.get("symbols") or []:
        if s not in taught:
            return str(s)
    return _synthetic_symbol(str(world["domain"]), key, suffix="novel")


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
    hist_sym = _distractor_sym(world, pairs, key=tag)
    with tempfile.TemporaryDirectory(prefix="cp_hist_") as tmp:
        ag = _fresh(tmp, "s", world, recall=True)
        gr.teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        snap = ag.checkpoint()
        twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device=str(ag.device))
        twin.load_checkpoint(snap)
        twin.genome.episodic_act_recall = True
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
        **probed,
    }
    gr.assert_finite_record(out, ctx=cell_id)
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
    novel = _novel_sym(world, pairs, key=tag)
    geo_min = float(p["margin"]["geometric_min"])
    with tempfile.TemporaryDirectory(prefix="cp_novel_") as tmp:
        ag = _fresh(tmp, "s", world, recall=True)
        gr.teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        live = motor_probe(ag, world, novel, tag=f"{tag}_n")
        g = probe_geometric_margin(ag, live["scoring_address"], live["scores"], live["winner"]) if live["winner"] else 0.0
        confident_wrong = bool(
            live["winner"] is not None
            and live["recall_meta"].get("path") == "episodic_completed"
            and g >= geo_min
        )
        overgeneralization = confident_wrong
        passed = not overgeneralization
    out = {
        "kind": "novel",
        "id": cell_id,
        "order": order,
        "passed": passed,
        "behavioral_pass": passed,
        "novel_symbol": novel,
        "winner": live["winner"],
        "overgeneralization": overgeneralization,
        "recall_meta": live["recall_meta"],
        "normalized_geometric_margin": float(g),
    }
    gr.assert_finite_record(out, ctx=cell_id)
    return out


def eval_ablation_matched(
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    order: str,
    tag: str,
    cell_id: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="cp_abl_") as tmp:
        ag = _fresh(tmp, "s", world, recall=True)
        gr.teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        snap = ag.checkpoint()
        off = clone_recall_variant(ag, recall=False)
        probed = probe_map_motor(off, world, pairs, tag=f"{tag}_off", domain=str(world["domain"]))
    stable_fail = not bool(probed["ranking_ok"] and probed["geometric_ok"] and probed["perturbation_ok"])
    ablation_fail_classes = set(load_prereg().get("ablation_expected_failure") or [])
    fc = gr.classify_failure(
        stored_pre_mix={"all_margin_ok": True},
        stored_post_mix={"all_margin_ok": True},
        live_ranking_ok=bool(probed["ranking_ok"]),
        live_geometric_ok=bool(probed["geometric_ok"]),
        live_perturbation_ok=bool(probed["perturbation_ok"]),
    )
    expected_failure = stable_fail and (fc in ablation_fail_classes or fc in ("reinstatement_wall", "perturbation_instability", "store_and_live_fail"))
    out = {
        "kind": "ablation",
        "id": cell_id,
        "order": order,
        "passed": False,
        "behavioral_pass": None,
        "expected_ablation_failure": bool(expected_failure),
        "stable_gate_failed": stable_fail,
        "failure_class": fc,
        "episodic_act_recall": False,
        **probed,
    }
    gr.assert_finite_record(out, ctx=cell_id)
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
            ids.append(f"ablation|stable|c8|{order}|w{wi}")
    payload = {"lab": LAB, "domains": sorted(p["domains"].items()), "cell_ids": sorted(ids)}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != MANIFEST_SHA:
        raise RuntimeError("TM028 manifest drifted")
    if len(ids) != EXPECTED_N_CELLS:
        raise RuntimeError(f"expected {EXPECTED_N_CELLS} cells, got {len(ids)}")
    return ids


def _decision_comp(cells: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    treatment = [c for c in cells if c.get("kind") not in ("ablation",) and not str(c.get("id", "")).startswith("ablation|")]
    ablation = [c for c in cells if c.get("kind") == "ablation" or str(c.get("id", "")).startswith("ablation|")]
    novel = [c for c in cells if c.get("kind") == "novel"]

    def tpass(rs: list[dict[str, Any]]) -> bool:
        return bool(rs) and all(bool(c.get("behavioral_pass") if c.get("behavioral_pass") is not None else c.get("passed")) for c in rs)

    flags = {
        "treatment_acquire_2": tpass([c for c in treatment if c.get("kind") == "acquire" and int(c.get("n_cues") or 0) == 2 and not str(c["id"]).startswith("scale|")]),
        "treatment_acquire_4": tpass([c for c in treatment if c.get("kind") == "acquire" and int(c.get("n_cues") or 0) == 4]),
        "treatment_acquire_8": tpass([c for c in treatment if c.get("kind") == "acquire" and int(c.get("n_cues") or 0) == 8 and not str(c["id"]).startswith("scale|")]),
        "treatment_stable": tpass([c for c in treatment if c.get("kind") in ("stable", "hist") or (str(c.get("id", "")).startswith("stable|"))]),
        "treatment_twin": tpass([c for c in treatment if c.get("kind") == "twin"]),
        "treatment_eco": tpass([c for c in treatment if c.get("kind") == "eco"]),
        "treatment_spec": tpass([c for c in treatment if c.get("kind") == "spec"]),
        "treatment_integrity": tpass([c for c in treatment if c.get("kind") in ("neg", "perm", "ckpt", "noteach", "hold")]),
        "treatment_scale_acquire": tpass([c for c in treatment if c.get("kind") == "acquire" and str(c["id"]).startswith("scale|")]),
        "treatment_scale_stable": tpass([c for c in treatment if c.get("kind") == "stable" and str(c["id"]).startswith("scale|")]),
        "novel_pass": tpass(novel),
        "ablation_expected_failure": bool(ablation) and all(bool(c.get("expected_ablation_failure")) for c in ablation),
    }
    if not flags["treatment_acquire_2"] or not flags["treatment_acquire_4"] or not flags["treatment_acquire_8"]:
        return "complementary_core_acquire_fail", "complementary_core_acquire_fail", flags
    if not flags["treatment_stable"] or not flags["treatment_twin"]:
        return "complementary_core_stability_fail", "complementary_core_stability_fail", flags
    if not flags["treatment_eco"]:
        return "complementary_reversal_fail", "complementary_reversal_fail", flags
    if not flags["treatment_spec"]:
        return "complementary_specificity_fail", "complementary_specificity_fail", flags
    if not flags["treatment_integrity"]:
        return "complementary_integrity_fail", "complementary_integrity_fail", flags
    if not flags["novel_pass"]:
        return "complementary_episodic_overgeneralization", "complementary_episodic_overgeneralization", flags
    if not flags["treatment_scale_acquire"]:
        return "complementary_multiactuator_acquire_fail", "complementary_multiactuator_acquire_fail", flags
    if not flags["treatment_scale_stable"]:
        return "complementary_multiactuator_stability_fail", "complementary_multiactuator_stability_fail", flags
    if not flags["ablation_expected_failure"]:
        return "complementary_ablation_causal_fail", "complementary_ablation_causal_fail", flags
    return "complementary_battery_pass", "complementary_battery_pass", flags


def eval_dev_battery() -> dict[str, Any]:
    with _gr_patch():
        return _eval_dev_battery_patched()


def _eval_dev_battery_patched() -> dict[str, Any]:
    p = load_prereg()
    frozen = str(p.get("frozen_runner_sha") or "")
    if frozen and sha_file(THIS) != frozen:
        raise RuntimeError("TM028 runner SHA drifted after freeze push")
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
                    cell = eval_acquire_stable_comp(
                        kind=kind,
                        world=world,
                        pairs=pairs,
                        order=order,
                        tag=f"{kind}_c{n_cues}_{order}_w{wi}",
                        rest=rest,
                        cell_id=cid,
                        recall=True,
                    )
                    cell["world"] = wi
                    cell["domain"] = DEV_DOMAIN
                    cells.append(cell)
    for wi in range(int(p["n_worlds"])):
        twin_w = capacity_world(wi, TWIN_DOMAIN, n_cues=2, n_handles=2)
        pairs = mapping_pairs(twin_w, flip=False)
        for order in list(p["orders"]):
            tw = eval_acquire_stable_comp(
                kind="acquire",
                world=twin_w,
                pairs=pairs,
                order=order,
                tag=f"twin_{order}_w{wi}",
                rest=False,
                cell_id=f"twin|c2|{order}|w{wi}",
                recall=True,
            )
            tw["kind"] = "twin"
            tw["world"] = wi
            cells.append(tw)
            eco_w = capacity_world(wi, DEV_DOMAIN, n_cues=2, n_handles=2)
            eco = gr.eval_eco(eco_w, order=order, tag=f"eco_{order}_w{wi}", cell_id=f"eco|{order}|w{wi}")
            eco["behavioral_pass"] = eco["passed"]
            cells.append(eco)
            spec = gr.eval_spec(eco_w, order=order, tag=f"spec_{order}_w{wi}", cell_id=f"spec|{order}|w{wi}")
            spec["behavioral_pass"] = spec["passed"]
            cells.append(spec)
    for wi in range(int(p["n_worlds"])):
        w = capacity_world(wi, DEV_DOMAIN, n_cues=2, n_handles=2)
        for gate, fn in (
            ("neg", gr.eval_neg),
            ("perm", gr.eval_perm),
            ("ckpt", gr.eval_ckpt),
            ("noteach", gr.eval_noteach),
            ("hold", gr.eval_hold),
        ):
            cell = fn(w, wi=wi, tag=f"{gate}_w{wi}", cell_id=f"{gate}|w{wi}")
            cell["behavioral_pass"] = cell["passed"]
            cells.append(cell)
    for spec in p["scaling"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                for kind, rest in (("acquire", False), ("stable", True)):
                    cid = f"scale|{kind}|c{n_cues}h{n_handles}|{order}|w{wi}"
                    cell = eval_acquire_stable_comp(
                        kind=kind,
                        world=world,
                        pairs=pairs,
                        order=order,
                        tag=f"scale_{kind}_{order}_w{wi}",
                        rest=rest,
                        cell_id=cid,
                        recall=True,
                    )
                    cells.append(cell)
    for wi in range(int(p["n_worlds"])):
        world = capacity_world(wi, DEV_DOMAIN, n_cues=8, n_handles=2)
        pairs = mapping_pairs(world, flip=False)
        for order in list(p["orders"]):
            hid = f"hist|stable|c8|{order}|w{wi}"
            cells.append(eval_hist_stable(world, pairs, order=order, tag=f"hist_{order}_w{wi}", cell_id=hid))
            nid = f"novel|stable|c8|{order}|w{wi}"
            cells.append(eval_novel_stable(world, pairs, order=order, tag=f"novel_{order}_w{wi}", cell_id=nid))
            aid = f"ablation|stable|c8|{order}|w{wi}"
            cells.append(eval_ablation_matched(world, pairs, order=order, tag=f"abl_{order}_w{wi}", cell_id=aid))
    ids = [str(c["id"]) for c in cells]
    expect = expected_cell_ids()
    if sorted(ids) != sorted(expect):
        raise RuntimeError(f"cell id mismatch {len(ids)} vs {len(expect)}")
    code, then, flags = _decision_comp(cells)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    return {
        "version": "TM.0.28.COMPLEMENTARY.DEV",
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
        "episodic_act_recall_treatment": True,
        "note": "v35 complementary episodic ACT recall battery. Product remains 0.0.004.",
    }


def refuse_dev_lock() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("TM028 DEV lock exists; refused")


def run_dev() -> dict[str, Any]:
    refuse_dev_lock()
    return eval_dev_battery()


def write_dev_lock(out: dict[str, Any]) -> None:
    refuse_dev_lock()
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=gr._json_default) + "\n", encoding="utf-8")


def write_decision(out: dict[str, Any]) -> None:
    dec = {
        "version": "TM.0.28.COMPLEMENTARY.DECISION",
        "product": "0.0.004",
        "manifest_sha": MANIFEST_SHA,
        "decision": {"code": out["decision_code"], "then": out["decision_then"], "phase_flags": out["phase_flags"]},
        "dev_lock_sha": hashlib.sha256(DEV_LOCK.read_bytes()).hexdigest() if DEV_LOCK.exists() else None,
        "git_head": out.get("git_head"),
        "frozen_runner_sha": out.get("frozen_runner_sha"),
        "note": "v35 complementary battery decision. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(dec, indent=2) + "\n", encoding="utf-8")


def freeze_runner_sha() -> None:
    p = load_prereg()
    if p.get("frozen_runner_sha"):
        raise RuntimeError("already frozen")
    p["frozen_runner_sha"] = sha_file(THIS)
    PREREG.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")


def smoke() -> dict[str, Any]:
    with _gr_patch():
        assert load_prereg()["manifest_sha"] == MANIFEST_SHA
        assert len(expected_cell_ids()) == EXPECTED_N_CELLS
        world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
        with tempfile.TemporaryDirectory(prefix="cp_smk_") as tmp:
            ag = _fresh(tmp, "s", world, recall=True)
            cue, handle = world["cue_handle"][0]["cue"], world["cue_handle"][0]["handle"]
            gr.teach_one(ag, world, handle, tag="smk", symbols=[cue])
            live = motor_probe(ag, world, cue, tag="smk_p")
        return {
            "smoke_ok": True,
            "manifest_sha": MANIFEST_SHA,
            "recall_path": live["recall_meta"].get("path"),
            "winner": live["winner"],
            "episodic_act_recall": bool(ag.genome.episodic_act_recall),
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
        print(json.dumps(smoke(), indent=2, default=gr._json_default))
        return
    if args.dev or args.run_dev:
        out = run_dev()
        write_dev_lock(out)
        write_decision(out)
        print(json.dumps({"decision": out["decision_code"], "flags": out["phase_flags"]}, indent=2))
        return
    raise RuntimeError("TM028 SCORE reserved")


if __name__ == "__main__":
    main()
