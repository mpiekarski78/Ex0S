"""TM030 key-geometry diagnostic tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_keygeom.prereg.lock"
V35_ISO = REPO / "docs" / "cortex_v35.isolation.lock"
V36_PREREG = REPO / "docs" / "cortex_v36.prereg.lock"
TM029_DEC = REPO / "docs" / "lineage_indexing.decision.lock"
DEV = REPO / "docs" / "lineage_keygeom.dev.lock"
DEC = REPO / "docs" / "lineage_keygeom.decision.lock"
CLOSURE = REPO / "docs" / "lineage_keygeom.closure.lock"
MANIFEST = "dd8a22c5c8f9196919fc58968426bc044bb8ddf55fe187a96ecf125f32d9dffc"
SEPARATOR_SHA = "afaef71091d7350d84843646a80b4ea82e332edeeb4a64fb4ebfded0da3cb1ac"
HISTORICAL_V35_ISO_SHA = "8d1b72fc45aac48f72f38d9ed753e37de81c75df2a0a1b23ee6d880f8b42f8d8"
HISTORICAL_TM029_DEC_SHA = "f3fc981d9516e5ecade86ed39fbf95f027ca7dcd8aa4cccd68601a5ec78083b0"
FROZEN_RUNNER_SHA = "52b019c8a5c982635f8b6fbc49446ff0528521583b6813bd6065f9c30314ea6d"
HISTORICAL_DEV_SHA = "610f79246af390c1cf02b3ae862da80f72c6ed9e0ab43a94ef02433ea7086a9b"
HISTORICAL_DEC_SHA = "88309df4d15bb9fccc3f85e169be3b97b0dd4b2eb53be9b2a1f6d730c11e231f"
HISTORICAL_PREREG_SHA = "601b7271f1ac756ef55c448ecd19a8d7047eacd66ca57a5097f812206a9b5cde"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_manifest():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 8
    assert p["key_match_min_overlap"] == 5
    assert p["sep_k"] == 8
    assert p["separator_matrix_sha"] == SEPARATOR_SHA
    assert p["train_recall_mode"] == "separated_key"
    assert p["neural_edit_authorized"] is False
    assert "geometry_wall_complete" in p["decision_vector"]
    assert p["tm029_decision_sha"] == HISTORICAL_TM029_DEC_SHA
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA


def test_v35_isolation_unedited():
    assert _sha(V35_ISO) == HISTORICAL_V35_ISO_SHA


def test_tm029_decision_pinned():
    dec = json.loads(TM029_DEC.read_text())
    assert _sha(TM029_DEC) == HISTORICAL_TM029_DEC_SHA


def test_dev_decision_locks():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(PREREG) == HISTORICAL_PREREG_SHA
    dec = json.loads(DEC.read_text())
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    assert dec["outcome_vector"]["geometry_wall_complete"] is True
    assert dec["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert "W" not in dec
    assert "raw_familiarity" not in json.dumps(dec)
    assert dec.get("threshold_installed") is None


def _wnb_from_dev(dev: dict) -> dict:
    w_hist = []
    w_pert = []
    n_vals = []
    b_vals = []
    sep = {"c8": 0, "c8h4": 0}
    tot = {"c8": 0, "c8h4": 0}
    for cell in dev["cells"]:
        fam = "c8h4" if "scale" in cell["id"] else "c8"
        for hp in cell["hist_probes"]:
            w_hist.append(float(hp["raw"]["best"]))
        for pp in cell["perturb_probes"]:
            for t in pp["trials"]:
                tot[fam] += 1
                w_pert.append(float(t["raw"]["best"]))
                if t.get("raw_identity_drift") is False and t.get("sparse_identity_drift") is True:
                    sep[fam] += 1
        for np_ in cell["novel_probes"]:
            n_vals.append(float(np_["raw"]["best"]))
        b_vals.extend(float(x) for x in cell["between_cue"]["key_rho_l2_off_diagonal"])
    W = max(max(w_hist), max(w_pert))
    N = min(n_vals)
    B = min(b_vals)
    return {
        "W": W,
        "N": N,
        "B": B,
        "W_hist_max": max(w_hist),
        "W_pert_max": max(w_pert),
        "sep": sep,
        "tot": tot,
        "n_unique": len(set(n_vals)),
        "n_le_W": sum(1 for x in n_vals if x <= W),
    }


def test_closure_wnb_from_existing_dev_no_threshold():
    cl = json.loads(CLOSURE.read_text())
    assert cl["rewrite_historical_dev"] is False
    assert cl["rewrite_historical_decision"] is False
    assert cl["threshold_installed_in_decision_lock"] is False
    assert cl["v37_architecture_in_this_pass"] is False
    assert cl["dev_rerun"] is False
    assert cl["historical_dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert cl["historical_decision_sha"] == HISTORICAL_DEC_SHA
    assert _sha(DEV) == cl["historical_dev_lock_sha"]
    assert _sha(DEC) == cl["historical_decision_sha"]
    assert cl["outcome_vector_unchanged"] == json.loads(DEC.read_text())["outcome_vector"]

    got = _wnb_from_dev(json.loads(DEV.read_text()))
    cmp_ = cl["post_dev_comparison"]
    assert got["W"] == cmp_["W"]
    assert got["N"] == cmp_["N"]
    assert got["B"] == cmp_["B"]
    assert cmp_["W_lt_N"] is True
    assert cmp_["W_overlaps_N"] is False
    assert cmp_["W_approaches_B"] is False
    assert got["n_le_W"] == 0
    assert got["n_unique"] == 1
    den = cl["separator_added_collisions_denominator"]
    assert den["n_separator_added_when_raw_stable"] == 18
    assert den["n_perturbation_trials"] == 1280
    assert den["split"]["c8"]["n_separator_added_when_raw_stable"] == got["sep"]["c8"] == 10
    assert den["split"]["c8h4"]["n_separator_added_when_raw_stable"] == got["sep"]["c8h4"] == 8
    assert den["split"]["c8"]["n_perturbation_trials"] == got["tot"]["c8"] == 640
    assert den["split"]["c8h4"]["n_perturbation_trials"] == got["tot"]["c8h4"] == 640
    assert cmp_["route"] == "W_lt_N__raw_space_contains_familiarity_gap"
    assert "learned_separator_as_first_jump" in cl["refuse"]
    assert cl["failed_v36_control"] == "fixed_hadamard_k_wta_sparse_separator"
    assert cl["leading_v37_hypothesis"][0] == "early_key_rho"
    assert cl["leading_v37_hypothesis"][2] == "preregistered_raw_distance_familiarity_rejection"


def test_expected_cell_ids():
    from experiments.run_tm030keygeom import expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 8
    assert ids[0].startswith("geom|")


def test_smoke_and_side_effect_free():
    from experiments.run_tm030keygeom import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["parent_unchanged"]


def test_dual_retrieval_fields():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024writegeom import capacity_world, mapping_pairs
    from experiments.run_tm027gatedrehearsal import teach_pairs
    from experiments.run_tm030keygeom import (
        DEV_DOMAIN,
        _probe_state_from_checkpoint,
        pinned_perturb_vectors,
        perturb_vector_hashes,
    )
    import copy

    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory() as tmp:
        from experiments.run_tm030keygeom import _fresh

        ag = _fresh(tmp, "t", world)
        teach_pairs(ag, world, pairs, tag="t")
        snap = ag.checkpoint()
        genome = copy.deepcopy(ag.genome)
        cue = pairs[0][0]
        probe = _probe_state_from_checkpoint(snap, genome, world, cue=cue, tag="p")
        assert "raw" in probe and "sparse" in probe
        assert "slot" in probe["raw"]
        assert "margin" in probe["raw"]
        assert "tie" in probe["sparse"]
        vecs = pinned_perturb_vectors(str(world["domain"]), "geom|c8|A_then_B|w0", cue, probe["live_key_rho"])
        assert len(vecs) == 20
        hashes = perturb_vector_hashes(vecs)
        assert len(hashes) == 20
        assert len(set(hashes)) == 20


def test_pinned_symbols_stable():
    from experiments.run_tm030keygeom import pinned_hist_symbol, pinned_novel_symbol

    d = "TM030.KEYGEOM.DEV."
    h1 = pinned_hist_symbol(d, "geom|c8|A_then_B|w0")
    h2 = pinned_hist_symbol(d, "geom|c8|A_then_B|w0")
    assert h1 == h2
    n1 = pinned_novel_symbol(d, "geom|c8|A_then_B|w0", "cue_a")
    n2 = pinned_novel_symbol(d, "geom|c8|A_then_B|w0", "cue_b")
    assert n1 != n2


def test_outcome_vector_keys():
    from experiments.run_tm030keygeom import _decision, _distribution_summary

    cells = [
        {
            "side_effect_free_verified": True,
            "hist_probes": [{"raw_identity_drift": False, "sparse_identity_drift": False, "raw": {}, "sparse": {}}],
            "perturb_probes": [{"trials": [{"raw_identity_drift": False, "sparse_identity_drift": False, "raw": {}, "sparse": {}}]}],
            "baseline_probes": [{"raw": {"slot": 0, "tie": False}, "sparse": {"slot": 0, "tie": False}}],
            "novel_probes": [{"false_familiarity": False, "sparse": {}}],
            "between_cue": {"sparse_overlap_off_diagonal": [2]},
        }
    ]
    dist = _distribution_summary(cells)
    dec = _decision(cells, dist)
    ov = dec["outcome_vector"]
    assert ov["geometry_wall_complete"] is False  # only 1 cell
    assert set(ov.keys()) == {
        "key_rho_history_drift",
        "key_rho_perturbation_drift",
        "separator_added_collisions",
        "novel_overlap_false_familiarity",
        "geometry_wall_complete",
    }


def test_no_tm029_monkey_patch():
    import experiments.run_tm029indexing as ix
    import experiments.run_tm030keygeom as kg

    assert ix.DEV_DOMAIN.startswith("TM029")
    assert kg.DEV_DOMAIN.startswith("TM030")
    assert ix.TREATMENT_MODE == "separated_key"
    assert kg.TRAIN_MODE == "separated_key"
