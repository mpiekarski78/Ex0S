"""TM.0.30.KEYGEOM — frozen key-geometry diagnostic wall.

Standalone runner. Does not patch TM028, TM029, or gr module globals.
Product 0.0.004. No v37 architecture or threshold change in this pass.
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

from experiments.run_tm023cortex import make_cortex
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024statemap import prep_eval
from experiments.run_tm024writegeom import capacity_world, mapping_pairs
from experiments.run_tm027gatedrehearsal import assert_finite_record, domain_seed, teach_pairs
from three_memory.cortex_lineage import freeze_plasticity, sha_file
from three_memory.neural_cortex import (
    ACT_RECALL_SEP,
    KEY_MATCH_MIN_OVERLAP,
    PROTO_EPS,
    SEPARATOR_MATRIX_SHA,
    NeuralCortex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
LAB = "TM.0.30.KEYGEOM"
PREREG = REPO_ROOT / "docs" / "lineage_keygeom.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_keygeom_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_keygeom.isolation.lock"
V36_PREREG = REPO_ROOT / "docs" / "cortex_v36.prereg.lock"
TM029_DECISION = REPO_ROOT / "docs" / "lineage_indexing.decision.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_keygeom.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_keygeom.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm030keygeom_results.md"

DEV_DOMAIN = "TM030.KEYGEOM.DEV."
TWIN_DOMAIN = "TM030.KEYGEOM.TWIN."
EXPECTED_N_CELLS = 8
MANIFEST_SHA = "dd8a22c5c8f9196919fc58968426bc044bb8ddf55fe187a96ecf125f32d9dffc"
TRAIN_MODE = ACT_RECALL_SEP


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    root = Path(tmp) / tag
    ag = make_cortex(root, device="cpu")
    ag.genome.act_recall_mode = TRAIN_MODE
    ag.genome.episodic_act_recall = False
    ag.bind_actuators(list(world["handles"]))
    return ag


def _parent_fingerprint(ag: NeuralCortex) -> dict[str, Any]:
    return {
        "episode_clock": int(ag._episode_clock),
        "episode_n_inserts": int(ag._episode_n_inserts),
        "episode_n_replaced": int(ag._episode_n_replaced),
        "n_episodes": len(ag._episodes),
        "w_act_query_sum": float(ag.W_act_query.detach().sum().item()),
    }


def _assert_parent_unchanged(before: dict[str, Any], after: dict[str, Any], *, ctx: str) -> None:
    if before != after:
        raise RuntimeError(f"parent checkpoint mutated during probe {ctx}: {before} -> {after}")


def _synthetic_symbol(domain: str, key: str, *, suffix: str) -> str:
    h = hashlib.sha256(f"{domain}:{key}:{suffix}".encode()).hexdigest()[:12]
    return f"s_{suffix}_{h}"


def pinned_hist_symbol(domain: str, cell_id: str) -> str:
    p = load_prereg()
    suffix = str(p["pinned_probe_suffixes"]["hist"])
    return _synthetic_symbol(domain, cell_id, suffix=suffix)


def pinned_novel_symbol(domain: str, cell_id: str, cue: str) -> str:
    p = load_prereg()
    suffix = str(p["pinned_probe_suffixes"]["novel"])
    return _synthetic_symbol(domain, f"{cell_id}:{cue}", suffix=suffix)


def pinned_perturb_vectors(domain: str, cell_id: str, cue: str, base_hat: np.ndarray) -> list[np.ndarray]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    r_hat = np.asarray(base_hat, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(domain_seed(domain, f"{cell_id}:{cue}:pert"))
    out: list[np.ndarray] = []
    for _i in range(n):
        unit = r_hat + rng.normal(0.0, sigma, size=r_hat.shape)
        pn = float(np.linalg.norm(unit)) + 1e-12
        out.append((unit / pn).astype(np.float64))
    return out


def perturb_vector_hashes(vectors: list[np.ndarray]) -> list[str]:
    return [hashlib.sha256(v.tobytes()).hexdigest() for v in vectors]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= PROTO_EPS or ny <= PROTO_EPS:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def _retrieval_record(meta: dict[str, Any], *, path: str) -> dict[str, Any]:
    best = meta.get("nearest_dist")
    second = meta.get("second_nearest_dist")
    margin = None
    if best is not None and second is not None:
        if path == "raw":
            margin = float(second) - float(best)
        else:
            margin = float(best) - float(second)
    return {
        "slot": meta.get("slot"),
        "best": None if best is None else float(best),
        "second": None if second is None else float(second),
        "margin": margin,
        "tie": bool(meta.get("ambiguous")),
        "path_label": meta.get("path"),
        "reason": meta.get("reason"),
        "familiar": meta.get("familiar"),
        "overlap": meta.get("overlap"),
    }


def _probe_state_from_checkpoint(
    checkpoint: dict[str, Any],
    genome: Any,
    world: dict[str, Any],
    *,
    cue: str,
    tag: str,
    hist_symbol: str | None = None,
    perturbed_key_rho: np.ndarray | None = None,
) -> dict[str, Any]:
    ag = NeuralCortex(None, genome=copy.deepcopy(genome), device="cpu")
    ag.load_checkpoint(checkpoint)
    freeze_plasticity(ag)
    prep_eval(ag)
    if hist_symbol is not None:
        observe_cue(ag, world, tag=f"{tag}_hist", body=list(MID_BODY), symbols=[hist_symbol])
    observe_cue(ag, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    p1 = ag._last_p1
    if p1 is None:
        p1 = ag._from_t(ag.rho)
    key_rho = perturbed_key_rho if perturbed_key_rho is not None else ag._last_key_rho
    key_rho_arr = None if key_rho is None else np.asarray(key_rho, dtype=np.float64).reshape(-1)
    event_key = None if key_rho_arr is None else ag._separate_event_key(key_rho_arr)

    raw_stored, raw_meta = (
        ag._nearest_episode_by_key_rho(key_rho_arr) if key_rho_arr is not None else (None, {"ambiguous": False, "slot": None})
    )
    sparse_stored, sparse_meta = (
        ag._nearest_episode_by_sparse_key(event_key, require_familiarity=True)
        if event_key is not None
        else (None, {"ambiguous": False, "slot": None})
    )
    raw = _retrieval_record(raw_meta, path="raw")
    sparse = _retrieval_record(sparse_meta, path="sparse")
    return {
        "cue": cue,
        "p1_norm": float(np.linalg.norm(np.asarray(p1, dtype=np.float64))),
        "key_rho_norm": None if key_rho_arr is None else float(np.linalg.norm(key_rho_arr)),
        "raw": raw,
        "sparse": sparse,
        "live_key_rho": None if key_rho_arr is None else key_rho_arr.copy(),
        "live_event_key": None if event_key is None else np.asarray(event_key, dtype=np.float64).copy(),
    }


def _stored_episode_table(checkpoint: dict[str, Any], genome: Any) -> list[dict[str, Any]]:
    ag = NeuralCortex(None, genome=copy.deepcopy(genome), device="cpu")
    ag.load_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    for i, ep in enumerate(ag._episodes):
        if not ep.get("valid"):
            continue
        key = ep.get("key")
        key_rho = ep.get("key_rho")
        rows.append(
            {
                "physical_slot": int(i),
                "handle": str(ep.get("handle")),
                "age": int(ep.get("age") or 0),
                "version": int(ep.get("version") or 1),
                "key_rho_norm": None if key_rho is None else float(np.linalg.norm(np.asarray(key_rho))),
                "key_popcount": None if key is None else int(np.sum(np.asarray(key))),
            }
        )
    return rows


def _between_cue_geometry(checkpoint: dict[str, Any], genome: Any, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    ag = NeuralCortex(None, genome=copy.deepcopy(genome), device="cpu")
    ag.load_checkpoint(checkpoint)
    valid = [ep for ep in ag._episodes if ep.get("valid")]
    n = len(valid)
    l2: list[float] = []
    cos: list[float] = []
    sparse_off: list[int] = []
    cues = [c for c, _h in pairs]
    for i in range(n):
        for j in range(i + 1, n):
            ki = valid[i].get("key_rho")
            kj = valid[j].get("key_rho")
            if ki is not None and kj is not None:
                a = np.asarray(ki, dtype=np.float64)
                b = np.asarray(kj, dtype=np.float64)
                l2.append(float(np.linalg.norm(a - b)))
                cos.append(_cosine(a, b))
            sk = valid[i].get("key")
            tk = valid[j].get("key")
            if sk is not None and tk is not None:
                sparse_off.append(int(ag._key_overlap(np.asarray(sk), np.asarray(tk))))
    return {
        "n_stored": n,
        "taught_cues": cues,
        "key_rho_l2_off_diagonal": l2,
        "key_rho_cosine_off_diagonal": cos,
        "sparse_overlap_off_diagonal": sparse_off,
        "max_sparse_off_diagonal": max(sparse_off) if sparse_off else None,
        "mean_sparse_off_diagonal": float(np.mean(sparse_off)) if sparse_off else None,
    }


def train_checkpoint(
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    order: str,
    tag: str,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="kg_train_") as tmp:
        ag = _fresh(tmp, "train", world)
        before = _parent_fingerprint(ag)
        teach_pairs(ag, world, seq, tag=tag)
        ag.rest_epoch(int(p.get("n_rest_ticks") or 4))
        after_train = _parent_fingerprint(ag)
        snap = ag.checkpoint()
        genome = copy.deepcopy(ag.genome)
    return snap, genome, {"train_fingerprint": after_train, "pre_train_fingerprint": before}


def eval_geom_cell(
    *,
    cell_id: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
) -> dict[str, Any]:
    domain = str(world["domain"])
    hist_sym = pinned_hist_symbol(domain, cell_id)
    snap, genome, train_meta = train_checkpoint(world, pairs, order=order, tag=tag)

    # Rebuild parent for side-effect checks (fresh load from snap)
    with tempfile.TemporaryDirectory(prefix="kg_parent_") as tmp:
        parent = _fresh(tmp, "parent", world)
        parent.load_checkpoint(snap)
        parent_before = _parent_fingerprint(parent)

        expected: dict[str, dict[str, Any]] = {}
        baseline_probes: list[dict[str, Any]] = []
        for i, (cue, handle) in enumerate(pairs):
            probe = _probe_state_from_checkpoint(snap, genome, world, cue=cue, tag=f"{tag}_base_{i}")
            _assert_parent_unchanged(parent_before, _parent_fingerprint(parent), ctx=f"baseline:{cue}")
            expected[cue] = {
                "logical_id": int(i),
                "handle": handle,
                "expected_baseline_raw_slot": probe["raw"]["slot"],
                "expected_baseline_sparse_slot": probe["sparse"]["slot"],
            }
            baseline_probes.append(probe)

        hist_probes: list[dict[str, Any]] = []
        for i, (cue, handle) in enumerate(pairs):
            base_k = baseline_probes[i].get("live_key_rho")
            probe = _probe_state_from_checkpoint(
                snap, genome, world, cue=cue, tag=f"{tag}_hist_{i}", hist_symbol=hist_sym
            )
            _assert_parent_unchanged(parent_before, _parent_fingerprint(parent), ctx=f"hist:{cue}")
            l2_delta = None
            cos_delta = None
            if base_k is not None and probe.get("live_key_rho") is not None:
                l2_delta = float(np.linalg.norm(np.asarray(probe["live_key_rho"]) - np.asarray(base_k)))
                cos_delta = _cosine(probe["live_key_rho"], base_k)
            exp = expected[cue]
            raw_drift = (
                probe["raw"]["slot"] != exp["expected_baseline_raw_slot"]
                if not probe["raw"]["tie"] and exp["expected_baseline_raw_slot"] is not None
                else None
            )
            sparse_drift = (
                probe["sparse"]["slot"] != exp["expected_baseline_sparse_slot"]
                if not probe["sparse"]["tie"] and exp["expected_baseline_sparse_slot"] is not None
                else None
            )
            hist_probes.append(
                {
                    **probe,
                    "history_symbol": hist_sym,
                    "within_cue_key_rho_l2_delta": l2_delta,
                    "within_cue_key_rho_cosine": cos_delta,
                    "raw_identity_drift": raw_drift,
                    "sparse_identity_drift": sparse_drift,
                    "logical_id": int(i),
                    "expected_baseline_raw_slot": exp["expected_baseline_raw_slot"],
                }
            )

        perturb_probes: list[dict[str, Any]] = []
        for i, (cue, handle) in enumerate(pairs):
            base = baseline_probes[i]
            base_k = base.get("live_key_rho")
            if base_k is None:
                continue
            unit_hat = base_k / (float(np.linalg.norm(base_k)) + 1e-12)
            vectors = pinned_perturb_vectors(domain, cell_id, cue, unit_hat)
            vec_hashes = perturb_vector_hashes(vectors)
            exp = expected[cue]
            trials: list[dict[str, Any]] = []
            n_raw_drift = 0
            n_sparse_drift = 0
            n_raw_ok = 0
            n_sparse_ok = 0
            for ti, vec in enumerate(vectors):
                probe = _probe_state_from_checkpoint(
                    snap, genome, world, cue=cue, tag=f"{tag}_pert_{i}_{ti}", perturbed_key_rho=vec
                )
                _assert_parent_unchanged(parent_before, _parent_fingerprint(parent), ctx=f"pert:{cue}:{ti}")
                raw_drift = (
                    probe["raw"]["slot"] != exp["expected_baseline_raw_slot"]
                    if not probe["raw"]["tie"] and exp["expected_baseline_raw_slot"] is not None
                    else None
                )
                sparse_drift = (
                    probe["sparse"]["slot"] != exp["expected_baseline_sparse_slot"]
                    if not probe["sparse"]["tie"] and exp["expected_baseline_sparse_slot"] is not None
                    else None
                )
                if raw_drift is True:
                    n_raw_drift += 1
                if raw_drift is False:
                    n_raw_ok += 1
                if sparse_drift is True:
                    n_sparse_drift += 1
                if sparse_drift is False:
                    n_sparse_ok += 1
                trials.append(
                    {
                        "trial": int(ti),
                        "vector_sha256": vec_hashes[ti],
                        "raw": probe["raw"],
                        "sparse": probe["sparse"],
                        "raw_identity_drift": raw_drift,
                        "sparse_identity_drift": sparse_drift,
                    }
                )
            perturb_probes.append(
                {
                    "cue": cue,
                    "logical_id": int(i),
                    "expected_baseline_raw_slot": exp["expected_baseline_raw_slot"],
                    "perturb_n": len(trials),
                    "vector_sha256s": vec_hashes,
                    "raw_identity_drift_rate": float(n_raw_drift / (n_raw_ok + n_raw_drift))
                    if (n_raw_ok + n_raw_drift)
                    else 0.0,
                    "sparse_identity_drift_rate": float(n_sparse_drift / (n_sparse_ok + n_sparse_drift))
                    if (n_sparse_ok + n_sparse_drift)
                    else 0.0,
                    "n_raw_identity_drift": int(n_raw_drift),
                    "n_sparse_identity_drift": int(n_sparse_drift),
                    "trials": trials,
                }
            )

        novel_probes: list[dict[str, Any]] = []
        for i, (cue, _handle) in enumerate(pairs):
            novel = pinned_novel_symbol(domain, cell_id, cue)
            probe = _probe_state_from_checkpoint(snap, genome, world, cue=novel, tag=f"{tag}_novel_{i}")
            _assert_parent_unchanged(parent_before, _parent_fingerprint(parent), ctx=f"novel:{novel}")
            ov = probe["sparse"].get("overlap")
            tie = bool(probe["sparse"]["tie"])
            false_fam = bool(
                ov is not None
                and int(ov) >= KEY_MATCH_MIN_OVERLAP
                and not tie
                and probe["sparse"]["path_label"] == "episodic_completed"
            )
            novel_probes.append(
                {
                    "reference_cue": cue,
                    "novel_symbol": novel,
                    "raw": probe["raw"],
                    "sparse": probe["sparse"],
                    "false_familiarity": false_fam,
                }
            )

        between = _between_cue_geometry(snap, genome, pairs)
        episode_table = _stored_episode_table(snap, genome)

    out = {
        "kind": "geom",
        "id": cell_id,
        "order": order,
        "n_cues": int(world["capacity"]["n_cues"]),
        "n_handles": int(world["capacity"]["n_handles"]),
        "domain": domain,
        "history_symbol": hist_sym,
        "train_recall_mode": TRAIN_MODE,
        "expected_slots": expected,
        "episode_table": episode_table,
        "between_cue": between,
        "baseline_probes": baseline_probes,
        "hist_probes": hist_probes,
        "perturb_probes": perturb_probes,
        "novel_probes": novel_probes,
        "side_effect_free_verified": True,
        **train_meta,
    }
    assert_finite_record(out, ctx=cell_id)
    return out


def expected_cell_ids() -> list[str]:
    p = load_prereg()
    ids: list[str] = []
    orders = list(p["orders"])
    n_worlds = int(p["n_worlds"])
    for spec in p["checkpoints"]:
        prefix = str(spec["id_prefix"])
        for order in orders:
            for wi in range(n_worlds):
                ids.append(f"geom|{prefix}|{order}|w{wi}")
    payload = {
        "lab": LAB,
        "domains": sorted(p["domains"].items()),
        "cell_ids": sorted(ids),
    }
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != MANIFEST_SHA:
        raise RuntimeError("TM030 manifest drifted")
    if len(ids) != EXPECTED_N_CELLS:
        raise RuntimeError(f"expected {EXPECTED_N_CELLS} cells, got {len(ids)}")
    return ids


def _distribution_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    l2_hist: list[float] = []
    cos_hist: list[float] = []
    raw_hist_drifts = 0
    raw_hist_total = 0
    sparse_hist_drifts = 0
    sparse_hist_total = 0
    raw_pert_drifts = 0
    raw_pert_total = 0
    sparse_pert_drifts = 0
    sparse_pert_total = 0
    raw_ties = 0
    sparse_ties = 0
    raw_probes = 0
    sparse_probes = 0
    sparse_off: list[int] = []
    novel_best: list[float] = []
    novel_second: list[float] = []
    novel_false_fam = 0
    novel_n = 0
    sep_collision_baseline = 0
    sep_collision_n = 0
    separator_added_hist = 0
    separator_added_pert = 0

    for cell in cells:
        for hp in cell.get("hist_probes") or []:
            if hp.get("within_cue_key_rho_l2_delta") is not None:
                l2_hist.append(float(hp["within_cue_key_rho_l2_delta"]))
            if hp.get("within_cue_key_rho_cosine") is not None:
                cos_hist.append(float(hp["within_cue_key_rho_cosine"]))
            if hp.get("raw_identity_drift") is not None:
                raw_hist_total += 1
                raw_hist_drifts += int(hp["raw_identity_drift"])
            if hp.get("sparse_identity_drift") is not None:
                sparse_hist_total += 1
                sparse_hist_drifts += int(hp["sparse_identity_drift"])
            raw = hp.get("raw") or {}
            sparse = hp.get("sparse") or {}
            if raw.get("tie"):
                raw_ties += 1
            if sparse.get("tie"):
                sparse_ties += 1
            raw_probes += 1
            sparse_probes += 1
            exp_raw = hp.get("expected_baseline_raw_slot")
            if (
                exp_raw is not None
                and not raw.get("tie")
                and not sparse.get("tie")
                and raw.get("slot") == exp_raw
                and sparse.get("slot") != raw.get("slot")
            ):
                separator_added_hist += 1

        for bp in cell.get("baseline_probes") or []:
            raw = bp.get("raw") or {}
            sparse = bp.get("sparse") or {}
            if raw.get("tie"):
                raw_ties += 1
            if sparse.get("tie"):
                sparse_ties += 1
            raw_probes += 1
            sparse_probes += 1
            if not raw.get("tie") and not sparse.get("tie") and raw.get("slot") is not None:
                sep_collision_n += 1
                if sparse.get("slot") != raw.get("slot"):
                    sep_collision_baseline += 1

        for pp in cell.get("perturb_probes") or []:
            for t in pp.get("trials") or []:
                if t.get("raw_identity_drift") is not None:
                    raw_pert_total += 1
                    raw_pert_drifts += int(t["raw_identity_drift"])
                if t.get("sparse_identity_drift") is not None:
                    sparse_pert_total += 1
                    sparse_pert_drifts += int(t["sparse_identity_drift"])
                raw = t.get("raw") or {}
                sparse = t.get("sparse") or {}
                if raw.get("tie"):
                    raw_ties += 1
                if sparse.get("tie"):
                    sparse_ties += 1
                raw_probes += 1
                sparse_probes += 1
                exp_raw = pp.get("expected_baseline_raw_slot")
                if (
                    exp_raw is not None
                    and t.get("raw_identity_drift") is False
                    and t.get("sparse_identity_drift") is True
                ):
                    separator_added_pert += 1

        bc = cell.get("between_cue") or {}
        sparse_off.extend(list(bc.get("sparse_overlap_off_diagonal") or []))

        for np_ in cell.get("novel_probes") or []:
            novel_n += 1
            sp = np_.get("sparse") or {}
            if sp.get("best") is not None:
                novel_best.append(float(sp["best"]))
            if sp.get("second") is not None:
                novel_second.append(float(sp["second"]))
            if np_.get("false_familiarity"):
                novel_false_fam += 1

    def _rate(num: int, den: int) -> float | None:
        return None if den == 0 else float(num / den)

    return {
        "within_cue_hist_key_rho_l2": l2_hist,
        "within_cue_hist_key_rho_cosine": cos_hist,
        "raw_hist_identity_drift_rate": _rate(raw_hist_drifts, raw_hist_total),
        "sparse_hist_identity_drift_rate": _rate(sparse_hist_drifts, sparse_hist_total),
        "raw_pert_identity_drift_rate": _rate(raw_pert_drifts, raw_pert_total),
        "sparse_pert_identity_drift_rate": _rate(sparse_pert_drifts, sparse_pert_total),
        "taught_sparse_off_diagonal_overlaps": sparse_off,
        "novel_best_overlap": novel_best,
        "novel_second_overlap": novel_second,
        "novel_false_familiarity_rate": _rate(novel_false_fam, novel_n),
        "raw_tie_rate": _rate(raw_ties, raw_probes),
        "sparse_tie_rate": _rate(sparse_ties, sparse_probes),
        "baseline_raw_sparse_slot_mismatch_count": sep_collision_baseline,
        "baseline_raw_sparse_slot_mismatch_denom": sep_collision_n,
        "separator_added_on_hist_when_raw_stable": separator_added_hist,
        "separator_added_on_pert_when_raw_stable": separator_added_pert,
    }


def _decision(cells: list[dict[str, Any]], dist: dict[str, Any]) -> dict[str, Any]:
    p = load_prereg()
    defs = p["outcome_definitions"]

    hist_raw_drift = any(
        hp.get("raw_identity_drift") is True for c in cells for hp in c.get("hist_probes") or []
    )
    pert_raw_drift = any(
        t.get("raw_identity_drift") is True
        for c in cells
        for pp in c.get("perturb_probes") or []
        for t in pp.get("trials") or []
    )
    sep_collision = bool(
        (dist.get("baseline_raw_sparse_slot_mismatch_count") or 0) > 0
        or (dist.get("separator_added_on_hist_when_raw_stable") or 0) > 0
        or (dist.get("separator_added_on_pert_when_raw_stable") or 0) > 0
    )
    novel_ff = any(np_.get("false_familiarity") for c in cells for np_ in c.get("novel_probes") or [])
    complete = len(cells) == EXPECTED_N_CELLS and all(c.get("side_effect_free_verified") for c in cells)

    vector = {
        "key_rho_history_drift": "observed" if hist_raw_drift else "not_observed",
        "key_rho_perturbation_drift": "observed" if pert_raw_drift else "not_observed",
        "separator_added_collisions": "observed" if sep_collision else "not_observed",
        "novel_overlap_false_familiarity": "observed" if novel_ff else "not_observed",
        "geometry_wall_complete": bool(complete),
    }
    return {
        "outcome_vector": vector,
        "outcome_definitions": defs,
        "observations": {
            "hist_raw_drift_any": hist_raw_drift,
            "pert_raw_drift_any": pert_raw_drift,
            "separator_collision_any": sep_collision,
            "novel_false_familiarity_any": novel_ff,
            "n_cells": len(cells),
        },
    }


def eval_dev_battery() -> dict[str, Any]:
    p = load_prereg()
    frozen = str(p.get("frozen_runner_sha") or "")
    if frozen and sha_file(THIS) != frozen:
        raise RuntimeError("TM030 runner SHA drifted after freeze push")
    if p.get("separator_matrix_sha") != SEPARATOR_MATRIX_SHA:
        raise RuntimeError("separator matrix SHA drifted")
    if int(p.get("key_match_min_overlap") or 0) != KEY_MATCH_MIN_OVERLAP:
        raise RuntimeError("familiarity overlap drifted")

    cells: list[dict[str, Any]] = []
    for spec in p["checkpoints"]:
        prefix = str(spec["id_prefix"])
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                cid = f"geom|{prefix}|{order}|w{wi}"
                cell = eval_geom_cell(
                    cell_id=cid,
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"kg_{prefix}_{order}_w{wi}".replace("|", "_"),
                )
                cell["world"] = wi
                cells.append(cell)

    ids = [str(c["id"]) for c in cells]
    expect = expected_cell_ids()
    if sorted(ids) != sorted(expect):
        raise RuntimeError(f"cell id mismatch {len(ids)} vs {len(expect)}")

    dist = _distribution_summary(cells)
    dec = _decision(cells, dist)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    return {
        "version": "TM.0.30.KEYGEOM.DEV",
        "product": "0.0.004",
        "n_cells": len(cells),
        "manifest_sha": MANIFEST_SHA,
        "domain": DEV_DOMAIN,
        "outcome_vector": dec["outcome_vector"],
        "outcome_definitions": dec["outcome_definitions"],
        "observations": dec["observations"],
        "distributions": dist,
        "cells": cells,
        "git_head": git_head,
        "frozen_runner_sha": sha_file(THIS),
        "train_recall_mode": TRAIN_MODE,
        "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
        "key_match_min_overlap": KEY_MATCH_MIN_OVERLAP,
    }


def refuse_dev_lock() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError(f"refuse overwrite {DEV_LOCK}")


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
        "version": "TM.0.30.KEYGEOM.DECISION",
        "product": "0.0.004",
        "manifest_sha": MANIFEST_SHA,
        "outcome_vector": out["outcome_vector"],
        "outcome_definitions": out["outcome_definitions"],
        "observations": out["observations"],
        "distributions_summary": {
            k: v
            for k, v in out["distributions"].items()
            if k
            not in (
                "within_cue_hist_key_rho_l2",
                "within_cue_hist_key_rho_cosine",
                "taught_sparse_off_diagonal_overlaps",
                "novel_best_overlap",
                "novel_second_overlap",
            )
        },
        "dev_lock_sha": hashlib.sha256(DEV_LOCK.read_bytes()).hexdigest() if DEV_LOCK.exists() else None,
        "git_head": out.get("git_head"),
        "frozen_runner_sha": out.get("frozen_runner_sha"),
        "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
        "note": "TM030 key-geometry diagnostic. Descriptive outcome vector only. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(dec, indent=2) + "\n", encoding="utf-8")


def write_results_md(out: dict[str, Any]) -> None:
    ov = out["outcome_vector"]
    dist = out["distributions"]
    lines = [
        "# TM.0.30.KEYGEOM results",
        "",
        "Frozen key-geometry diagnostic wall on v36 (`separated_key` training). Product **0.0.004**.",
        "",
        "## Outcome vector (descriptive)",
        "",
        f"- `key_rho_history_drift`: **{ov['key_rho_history_drift']}**",
        f"- `key_rho_perturbation_drift`: **{ov['key_rho_perturbation_drift']}**",
        f"- `separator_added_collisions`: **{ov['separator_added_collisions']}**",
        f"- `novel_overlap_false_familiarity`: **{ov['novel_overlap_false_familiarity']}**",
        f"- `geometry_wall_complete`: **{ov['geometry_wall_complete']}**",
        "",
        "## Distribution highlights",
        "",
        f"- Raw hist identity drift rate: {dist.get('raw_hist_identity_drift_rate')}",
        f"- Raw pert identity drift rate: {dist.get('raw_pert_identity_drift_rate')}",
        f"- Sparse hist identity drift rate: {dist.get('sparse_hist_identity_drift_rate')}",
        f"- Sparse pert identity drift rate: {dist.get('sparse_pert_identity_drift_rate')}",
        f"- Novel false-familiarity rate: {dist.get('novel_false_familiarity_rate')}",
        f"- Raw tie rate: {dist.get('raw_tie_rate')}",
        f"- Sparse tie rate: {dist.get('sparse_tie_rate')}",
        f"- Baseline raw≠sparse slot mismatches: {dist.get('baseline_raw_sparse_slot_mismatch_count')}/{dist.get('baseline_raw_sparse_slot_mismatch_denom')}",
        "",
        "## Wording",
        "",
        "v36 killed the **event-end P1 retrieval** story (`raw_p1` reinstatement wall on TM029). It did **not** kill raw retrieval generally — `early_raw` passed ordinary taught-stable probes.",
        "",
        "No v37 architecture, candidate lock, or threshold change in this pass.",
        "",
    ]
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory(prefix="kg_smk_") as tmp:
        ag = _fresh(tmp, "s", world)
        teach_pairs(ag, world, pairs, tag="smk")
        snap = ag.checkpoint()
        genome = copy.deepcopy(ag.genome)
        cue = pairs[0][0]
        probe = _probe_state_from_checkpoint(snap, genome, world, cue=cue, tag="smk_p")
        parent = _fresh(tmp, "p", world)
        parent.load_checkpoint(snap)
        before = _parent_fingerprint(parent)
        _probe_state_from_checkpoint(snap, genome, world, cue=cue, tag="smk_p2")
        after = _parent_fingerprint(parent)
    return {
        "smoke_ok": True,
        "manifest_sha": MANIFEST_SHA,
        "separator_matrix_sha": SEPARATOR_MATRIX_SHA,
        "raw_slot": probe["raw"]["slot"],
        "sparse_slot": probe["sparse"]["slot"],
        "parent_unchanged": before == after,
        "train_recall_mode": TRAIN_MODE,
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
        write_results_md(out)
        print(json.dumps({"outcome_vector": out["outcome_vector"], "n_cells": out["n_cells"]}, indent=2))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
