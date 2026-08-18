"""TM031 v37 half-spacing tests. Neural-gate tests skip until the mode exists."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_halfspace.prereg.lock"
V37_PREREG = REPO / "docs" / "cortex_v37.prereg.lock"
V37_ISO = REPO / "docs" / "cortex_v37.isolation.lock"
V35_ISO = REPO / "docs" / "cortex_v35.isolation.lock"
TM030_DEC = REPO / "docs" / "lineage_keygeom.decision.lock"
TM030_DEV = REPO / "docs" / "lineage_keygeom.dev.lock"
TM029_DEC = REPO / "docs" / "lineage_indexing.decision.lock"
RUNNER = REPO / "experiments" / "run_tm031halfspace.py"
DEV = REPO / "docs" / "lineage_halfspace.dev.lock"
DEC = REPO / "docs" / "lineage_halfspace.decision.lock"
ADDENDUM = REPO / "docs" / "lineage_halfspace.decision.addendum.lock"
MANIFEST = "2abd592a9c29c352038c92349424bd2e524b6b223457fb727bbb0232cb8afc93"
HISTORICAL_V35_ISO_SHA = "8d1b72fc45aac48f72f38d9ed753e37de81c75df2a0a1b23ee6d880f8b42f8d8"
HISTORICAL_TM030_DEC_SHA = "88309df4d15bb9fccc3f85e169be3b97b0dd4b2eb53be9b2a1f6d730c11e231f"
HISTORICAL_TM030_DEV_SHA = "610f79246af390c1cf02b3ae862da80f72c6ed9e0ab43a94ef02433ea7086a9b"
HISTORICAL_TM029_DEC_SHA = "f3fc981d9516e5ecade86ed39fbf95f027ca7dcd8aa4cccd68601a5ec78083b0"
FROZEN_RUNNER_SHA = "480f7400ada06143acaa3242e75aed315e941f40ebb253a7d0bf67caaa16f564"
HISTORICAL_DEV_SHA = "56dc67affd8fe5d2bb2263dc617c4d28922bfbac03c40b8bf8f6b7cccad67d9a"
HISTORICAL_DEC_SHA = "00d5e289068a68967af2c53669a0f4c2d16abc6617c851014261a1358dea07c6"
HISTORICAL_ADDENDUM_SHA = "5be880f3edf1549e75a6929d549d4a8d8d8d493a789dbed197443f25b76abdd1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mode_ready() -> bool:
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF

    return ACT_RECALL_EARLY_RAW_HALF == "early_raw_half_spacing"


def test_prereg_manifest_and_seeds():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 148
    assert p["treatment_mode"] == "early_raw_half_spacing"
    assert p["n_registry_seeds"] == 4
    assert p["registry_seeds_are_replicates_not_a_grid"] is True
    assert p["genome_varies_only"] == "seed_registry"
    assert 22222 not in p["registry_seeds"]
    assert p["oov_skips"] == [0, 1, 2, 3]
    assert p["novelty_success"] == {"path": "cortical_fallback", "familiar": False, "hold_not_required": True}
    assert p["not_a_novelty_theorem"] is True
    assert p["separated_key_control_label"] == "matched_retrieval_path_control__not_v36_train_reproduction"
    assert p["failed_v36_control"] == "historical_TM029"
    assert p["neural_edit_before_runner_freeze"] is False
    if FROZEN_RUNNER_SHA:
        assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
        assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_historical_locks_unedited():
    assert _sha(V35_ISO) == HISTORICAL_V35_ISO_SHA
    assert _sha(TM030_DEC) == HISTORICAL_TM030_DEC_SHA
    assert _sha(TM030_DEV) == HISTORICAL_TM030_DEV_SHA
    assert _sha(TM029_DEC) == HISTORICAL_TM029_DEC_SHA
    v37 = json.loads(V37_PREREG.read_text())
    assert v37["v35_isolation_sha"] == HISTORICAL_V35_ISO_SHA
    assert v37["tm030_decision_sha"] == HISTORICAL_TM030_DEC_SHA
    assert v37["hadamard_on_treatment_path"] is False
    assert v37["boundary_equality"] == "accept_d1_le_R_ieee_no_epsilon"
    iso = json.loads(V37_ISO.read_text())
    assert "install_tm030_W_N_cutoff" in iso["refuse"]


def test_dev_decision_locks_no_tm030_cutoff():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    dec = json.loads(DEC.read_text())
    assert dec["decision"]["code"] == "halfspace_core_acquire_fail"
    assert dec["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dec["not_a_tm030_cutoff"] is True
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    blob = json.dumps(dec)
    assert "0.294" not in blob
    assert "0.603" not in blob
    assert "0.344" not in blob


def test_expected_cell_ids():
    from experiments.run_tm031halfspace import expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 148
    assert "novel|stable|c8|A_then_B|reg3" in ids
    assert "ablation|early_raw_half_spacing|stable|c8|B_then_A|reg0" in ids


def test_genome_varies_only_seed_registry():
    from experiments.run_tm031halfspace import genome_for_registry, registry_seeds
    from three_memory.neural_cortex import GenomeConfig

    base = GenomeConfig().to_dict()
    seeds = registry_seeds()
    dicts = []
    for s in seeds:
        g = genome_for_registry(s)
        d = g.to_dict()
        dicts.append(d)
        for k, v in base.items():
            if k == "seed_registry":
                assert d[k] == s
            else:
                assert d[k] == v
    assert len({d["seed_registry"] for d in dicts}) == 4


def test_dummy_allocation_only_registry():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm031halfspace import allocate_dummy_oovs, genome_for_registry, plastic_fingerprint, registry_seeds

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), genome=genome_for_registry(registry_seeds()[0]), device="cpu")
        before = plastic_fingerprint(ag)
        rho0 = ag.rho.detach().clone()
        allocate_dummy_oovs(ag, 3)
        assert plastic_fingerprint(ag) == before
        assert float((ag.rho - rho0).abs().max().item()) == 0.0
        assert len(ag.vocab) == 3


def test_four_spellings_same_skip_are_not_independent():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024actorcredit import MID_BODY, observe_cue
    from experiments.run_tm024writegeom import capacity_world
    from experiments.run_tm031halfspace import DEV_DOMAIN, arr_sha, clone_from_snap, genome_for_registry, registry_seeds

    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), genome=genome_for_registry(registry_seeds()[0]), device="cpu")
        ag.bind_actuators(list(world["handles"]))
        snap = ag.checkpoint()
        hashes = []
        for name in ("s_novel_aaaa", "s_novel_bbbb", "s_novel_cccc", "s_novel_dddd"):
            twin = clone_from_snap(snap, copy.deepcopy(ag.genome), mode="off")
            observe_cue(twin, world, tag=name, body=list(MID_BODY), symbols=[name])
            hashes.append(arr_sha(twin._last_key_rho))
        assert len(set(hashes)) == 1


def test_skips_change_oov_key_rho():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024actorcredit import MID_BODY, observe_cue
    from experiments.run_tm024writegeom import capacity_world
    from experiments.run_tm031halfspace import (
        DEV_DOMAIN,
        allocate_dummy_oovs,
        arr_sha,
        clone_from_snap,
        genome_for_registry,
        registry_seeds,
    )

    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), genome=genome_for_registry(registry_seeds()[0]), device="cpu")
        ag.bind_actuators(list(world["handles"]))
        trained = ag.checkpoint()
        hashes = []
        for skip in (0, 1, 2, 3):
            dummy = clone_from_snap(trained, copy.deepcopy(ag.genome), mode="off")
            allocate_dummy_oovs(dummy, skip)
            frozen = dummy.checkpoint()
            probe = clone_from_snap(frozen, copy.deepcopy(ag.genome), mode="off")
            observe_cue(probe, world, tag=f"n{skip}", body=list(MID_BODY), symbols=["s_tm031_novel_probe"])
            hashes.append(arr_sha(probe._last_key_rho))
        assert len(set(hashes)) == 4


def test_smoke():
    from experiments.run_tm031halfspace import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["dummy_plastic_unchanged"]


def test_no_tm030_cutoff_in_prereg():
    p = json.dumps(json.loads(PREREG.read_text()))
    assert "0.294" not in p
    assert "0.603" not in p
    assert "0.344" not in p


@pytest.mark.skipif(not _mode_ready(), reason="v37 neural gate not installed yet")
def test_gate_edge_cases():
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = "early_raw_half_spacing"
        ag.bind_actuators(["h_a", "h_b"])
        k0 = np.zeros(64)
        k0[0] = 1.0
        k1 = np.zeros(64)
        k1[1] = 1.0
        p1a = k0.copy()
        p1b = k1.copy()
        ag._episodes = [
            {"p1": p1a, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True, "key_rho": k0.copy()},
            {"p1": p1b, "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True, "key_rho": k1.copy()},
        ]
        b = float(np.linalg.norm(k0 - k1))
        r = 0.5 * b
        q_in = k0.copy()
        stored, meta = ag._nearest_episode_by_key_rho(q_in, require_familiarity=True)
        assert stored is not None
        assert meta["path"] == "episodic_completed"
        assert meta["familiar"] is True
        assert meta["R"] == r
        assert meta["min_pair_slots"] == [0, 1]
        # unique nearest at d1 == R (perpendicular, already unit)
        sin_t = float(np.sqrt(1.0 - 0.75 * 0.75))
        q_eq = np.zeros(64)
        q_eq[0] = 0.75
        q_eq[2] = sin_t
        stored_eq, meta_eq = ag._nearest_episode_by_key_rho(q_eq, require_familiarity=True)
        assert meta_eq["nearest_dist"] == r
        assert stored_eq is not None
        assert meta_eq["path"] == "episodic_completed"
        q_out = np.zeros(64)
        q_out[2] = 1.0
        stored_out, meta_out = ag._nearest_episode_by_key_rho(q_out, require_familiarity=True)
        assert stored_out is None
        assert meta_out["path"] == "cortical_fallback"
        assert meta_out["familiar"] is False
        ag._episodes = [ag._episodes[0]]
        stored1, meta1 = ag._nearest_episode_by_key_rho(k0, require_familiarity=True)
        assert stored1 is None
        assert meta1["reason"] == "n_keyed_lt_2"
        ag._episodes = [
            {"p1": p1a, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True, "key_rho": k0.copy()},
            {"p1": p1b, "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True, "key_rho": k0.copy()},
        ]
        stored0, meta0 = ag._nearest_episode_by_key_rho(k0, require_familiarity=True)
        assert stored0 is None
        assert meta0["reason"] == "R_eq_0"
        ag._episodes = [
            {"p1": p1a, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True, "key_rho": k0.copy()},
            {"p1": p1b, "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True, "key_rho": k1.copy()},
        ]
        q_tie = 0.5 * (k0 + k1)
        stored_t, meta_t = ag._nearest_episode_by_key_rho(q_tie, require_familiarity=True)
        d0 = float(np.linalg.norm(k0 - q_tie))
        d1v = float(np.linalg.norm(k1 - q_tie))
        assert d0 == d1v
        assert stored_t is None
        assert meta_t["ambiguous"] is True
        ag._episodes = [
            {"p1": p1a, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True},
            {"p1": p1b, "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True},
        ]
        stored_m, meta_m = ag._nearest_episode_by_key_rho(k0, require_familiarity=True)
        assert stored_m is None
        assert "missing" in str(meta_m.get("reason"))


@pytest.mark.skipif(not _mode_ready(), reason="v37 neural gate not installed yet")
def test_treatment_no_separator():
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import SEPARATOR_MATRIX

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = "early_raw_half_spacing"
        ag.bind_actuators(["h_a", "h_b"])
        k0 = np.zeros(64)
        k0[0] = 1.0
        k1 = np.zeros(64)
        k1[1] = 1.0
        ag._episodes = [
            {"p1": k0.copy(), "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True, "key_rho": k0.copy(), "key": None},
            {"p1": k1.copy(), "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True, "key_rho": k1.copy(), "key": None},
        ]
        ag._last_key_rho = k0.copy()
        called = {"sep": False}
        orig = ag._separate_event_key

        def wrapped(x):
            called["sep"] = True
            return orig(x)

        ag._separate_event_key = wrapped  # type: ignore[method-assign]
        scores, _addr, meta = ag.actuator_decision_scores(k0)
        assert called["sep"] is False
        assert meta["act_recall_mode"] == "early_raw_half_spacing"


@pytest.mark.skipif(not _mode_ready(), reason="v37 neural gate not installed yet")
def test_checkpoint_roundtrip_preserves_mode():
    from experiments.run_tm023cortex import make_cortex

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = "early_raw_half_spacing"
        snap = ag.checkpoint()
        ag2 = make_cortex(Path(tmp) / "b", device="cpu")
        ag2.load_checkpoint(snap)
        assert ag2.genome.act_recall_mode == "early_raw_half_spacing"


@pytest.mark.skipif(not _mode_ready(), reason="v37 neural gate not installed yet")
def test_on_off_hash_match_protocol():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024actorcredit import MID_BODY, observe_cue
    from experiments.run_tm024writegeom import capacity_world
    from experiments.run_tm027gatedrehearsal import teach_one
    from experiments.run_tm031halfspace import (
        DEV_DOMAIN,
        TREATMENT_MODE,
        allocate_dummy_oovs,
        arr_sha,
        clone_from_snap,
        genome_for_registry,
        registry_seeds,
    )
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW

    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    cue, handle = world["cue_handle"][0]["cue"], world["cue_handle"][0]["handle"]
    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), genome=genome_for_registry(registry_seeds()[0]), device="cpu")
        ag.genome.act_recall_mode = TREATMENT_MODE
        ag.bind_actuators(list(world["handles"]))
        teach_one(ag, world, handle, tag="t", symbols=[cue])
        trained = ag.checkpoint()
        dummy = clone_from_snap(trained, copy.deepcopy(ag.genome), mode=TREATMENT_MODE)
        allocate_dummy_oovs(dummy, 1)
        frozen = dummy.checkpoint()
        on = clone_from_snap(frozen, copy.deepcopy(ag.genome), mode=TREATMENT_MODE)
        off = clone_from_snap(frozen, copy.deepcopy(ag.genome), mode=ACT_RECALL_EARLY_RAW)
        observe_cue(on, world, tag="on", body=list(MID_BODY), symbols=["s_tm031_causal_novel"])
        observe_cue(off, world, tag="off", body=list(MID_BODY), symbols=["s_tm031_causal_novel"])
        assert arr_sha(on._last_key_rho) == arr_sha(off._last_key_rho)


def test_write_radius_unchanged():
    from three_memory.neural_cortex import EPISODE_MATCH_L2

    assert EPISODE_MATCH_L2 == 0.05
    assert json.loads(PREREG.read_text())["write_match_l2"] == 0.05
    assert json.loads(V37_PREREG.read_text())["write_match_l2"] == 0.05


def test_half_mode_omitted_from_act_recall_modes_tuple():
    from experiments.run_tm029indexing import RECALL_MODES
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES

    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert ACT_RECALL_EARLY_RAW_HALF not in RECALL_MODES
    assert list(RECALL_MODES) == list(ACT_RECALL_MODES)
    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = ACT_RECALL_EARLY_RAW_HALF
        assert ag._resolve_act_recall_mode() == ACT_RECALL_EARLY_RAW_HALF
        snap = ag.checkpoint()
        ag2 = make_cortex(Path(tmp) / "b", device="cpu")
        ag2.load_checkpoint(snap)
        assert ag2.genome.act_recall_mode == ACT_RECALL_EARLY_RAW_HALF


def test_decision_addendum_does_not_rewrite_historical():
    add = json.loads(ADDENDUM.read_text())
    assert _sha(ADDENDUM) == HISTORICAL_ADDENDUM_SHA
    assert add["rewrite_historical_decision"] is False
    assert add["historical_decision_sha"] == HISTORICAL_DEC_SHA
    assert add["historical_dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert add["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert add["first_match_unchanged"] == "halfspace_core_acquire_fail"
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_dev_honest_core_vs_scale_and_acquire_identity():
    cells = json.loads(DEV.read_text())["cells"]
    core_stable_hist = [
        c
        for c in cells
        if c.get("kind") in ("stable", "hist") and not str(c["id"]).startswith("scale|")
    ]
    assert core_stable_hist
    assert all(bool(c.get("passed")) for c in core_stable_hist)
    scale_stable = [c for c in cells if str(c["id"]).startswith("scale|stable")]
    assert sum(1 for c in scale_stable if not c.get("passed")) == 1
    acquire_c8 = [c for c in cells if str(c["id"]).startswith("acquire|c8|")]
    fails = [c for c in acquire_c8 if not c.get("passed")]
    assert len(acquire_c8) == 8
    assert len(fails) == 2
    assert {c["id"] for c in fails} == {"acquire|c8|A_then_B|reg1", "acquire|c8|B_then_A|reg1"}
    wrong = 0
    for c in fails:
        for p in c["probes"]:
            if p.get("ranking_ok"):
                continue
            wrong += 1
            rm = p["recall_meta"]
            assert rm["path"] == "episodic_completed"
            assert rm["familiar"] is True
            assert float(rm["nearest_dist"]) < float(rm["R"])
            assert p["want"] != p["winner"]
    assert wrong == 4
    geometries = {
        c["stored_key_rho_sha256"]
        for c in cells
        if str(c["id"]).startswith("stable|c8|A_then_B|")
    }
    assert len(geometries) == 4
    novels = [c for c in cells if c.get("kind") == "novel"]
    assert len(novels) == 8
    assert all(c.get("passed") and c.get("gate_path_toggle") and c.get("hash_matched") for c in novels)
    for c in novels:
        for skip in c["skips"]:
            assert skip["on"]["path"] == "cortical_fallback"
            assert skip["on"]["familiar"] is False
            assert skip["off"]["path"] == "episodic_completed"
