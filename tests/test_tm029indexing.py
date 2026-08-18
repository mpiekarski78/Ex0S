"""TM029 v36 hippocampal indexing tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_indexing.prereg.lock"
V35_ISO = REPO / "docs" / "cortex_v35.isolation.lock"
V36_PREREG = REPO / "docs" / "cortex_v36.prereg.lock"
DEV = REPO / "docs" / "lineage_indexing.dev.lock"
DEC = REPO / "docs" / "lineage_indexing.decision.lock"
MANIFEST = "4ac2fd49c9a27e40ad13c9ed52b9d862900b2ff07e8ea7d0d94df8ca98797bca"
SEPARATOR_SHA = "afaef71091d7350d84843646a80b4ea82e332edeeb4a64fb4ebfded0da3cb1ac"
HISTORICAL_V35_ISO_SHA = "8d1b72fc45aac48f72f38d9ed753e37de81c75df2a0a1b23ee6d880f8b42f8d8"
FROZEN_RUNNER_SHA = "6752fa9a54c8a578e29b5300d4ade138e83fa1c5901d434ff3e560e38d61aacd"
HISTORICAL_DEV_SHA = "c04fafb25814e40057ce7467c76fce0759308a3b291a21aac48cb76cf8f15b49"
HISTORICAL_DEC_SHA = "f3fc981d9516e5ecade86ed39fbf95f027ca7dcd8aa4cccd68601a5ec78083b0"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_manifest():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 82
    assert p["key_match_min_overlap"] == 5
    assert p["sep_k"] == 8
    assert p["sep_sparsity"] == 0.125
    assert p["separator_matrix_sha"] == SEPARATOR_SHA
    assert p["treatment_mode"] == "separated_key"
    assert p["recall_modes"] == [
        "off",
        "raw_p1",
        "early_raw",
        "separated_key_no_familiarity",
        "separated_key",
    ]
    assert p["controls_are_observations"] is True
    assert p["novelty_success"]["hold_not_required"] is True
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA


def test_v35_isolation_unedited():
    assert _sha(V35_ISO) == HISTORICAL_V35_ISO_SHA
    v36 = json.loads(V36_PREREG.read_text())
    assert v36["v35_isolation_sha"] == HISTORICAL_V35_ISO_SHA
    assert v36["key_match_min_overlap"] == 5
    assert v36["write_match_statistic"] == "p1_l2"


def test_separator_matrix_sha():
    from three_memory.neural_cortex import SEPARATOR_MATRIX, SEPARATOR_MATRIX_SHA

    assert SEPARATOR_MATRIX_SHA == SEPARATOR_SHA
    assert SEPARATOR_MATRIX.shape == (64, 64)
    assert hashlib.sha256(SEPARATOR_MATRIX.tobytes()).hexdigest() == SEPARATOR_SHA


def test_k_wta_lower_index_on_equal():
    from three_memory.neural_cortex import k_wta_binary

    a = np.array([1.0, 1.0, 0.0, 0.5])
    key = k_wta_binary(a, k=2)
    assert key.tolist() == [1.0, 1.0, 0.0, 0.0]
    b = np.zeros(8)
    b[3] = 2.0
    b[1] = 2.0
    key2 = k_wta_binary(b, k=1)
    assert int(np.argmax(key2)) == 1


def test_integer_overlap_tie_fallback():
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import ACT_RECALL_SEP, KEY_MATCH_MIN_OVERLAP

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = ACT_RECALL_SEP
        ag.bind_actuators(["h_a", "h_b"])
        k = np.zeros(64)
        k[:8] = 1.0
        rho = np.zeros(64)
        rho[0] = 1.0
        p1a = rho.copy()
        p1b = rho.copy()
        p1b[1] = 0.1
        p1b = p1b / (np.linalg.norm(p1b) + 1e-12)
        ag._episodes = [
            {"p1": p1a, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True, "key": k.copy(), "key_rho": rho.copy()},
            {"p1": p1b, "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True, "key": k.copy(), "key_rho": rho.copy()},
        ]
        stored, meta = ag._nearest_episode_by_sparse_key(k, require_familiarity=True)
        assert stored is None
        assert meta["path"] == "cortical_fallback"
        assert meta["ambiguous"] is True
        assert meta["overlap"] >= KEY_MATCH_MIN_OVERLAP


def test_missing_legacy_key_fallback():
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import ACT_RECALL_SEP

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = ACT_RECALL_SEP
        ag.bind_actuators(["h_a", "h_b"])
        p1 = np.zeros(64)
        p1[0] = 1.0
        ag._episodes = [
            {"p1": p1, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True}
        ]
        k = np.zeros(64)
        k[:8] = 1.0
        stored, meta = ag._nearest_episode_by_sparse_key(k, require_familiarity=True)
        assert stored is None
        assert meta["reason"] == "missing_legacy_keys"


def test_pending_key_not_live_key():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024writegeom import capacity_world
    from experiments.run_tm024actorcredit import MID_BODY, observe_cue
    from three_memory.neural_cortex import ACT_RECALL_SEP

    world = capacity_world(0, "TM029.INDEXING.TEST.", n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = ACT_RECALL_SEP
        ag.bind_actuators(list(world["handles"]))
        observe_cue(ag, world, tag="sel", body=list(MID_BODY), symbols=[world["cue_handle"][0]["cue"]])
        ag.clamp_action("ACT", world["handles"][0])
        pending_key = None if ag._pending is None else np.asarray(ag._pending["event_key"]).copy()
        assert pending_key is not None
        ag._last_event_key = np.zeros_like(pending_key)
        ag._last_event_key[-8:] = 1.0
        from experiments.run_tm024statemap import physics

        _, body2 = physics(list(MID_BODY), world["handles"][0], world["latent"])
        observe_cue(ag, world, tag="obs", body=list(body2), symbols=[world["cue_handle"][0]["cue"]])
        assert ag._episodes
        stored = np.asarray(ag._episodes[0]["key"])
        assert np.allclose(stored, pending_key)
        assert not np.allclose(stored, ag._last_event_key)


def test_reset_clears_keys():
    from experiments.run_tm023cortex import make_cortex

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag._last_event_key = np.ones(64)
        ag._last_key_rho = np.ones(64)
        ag.reset_rho()
        assert ag._last_event_key is None
        assert ag._last_key_rho is None


def test_default_recall_off():
    from experiments.run_tm023cortex import make_cortex

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        assert ag.genome.act_recall_mode == "off"
        assert ag.genome.episodic_act_recall is False
        _, _, meta = ag.actuator_decision_scores(np.zeros(ag.genome.n))
        assert meta["path"] == "cortical"


def test_raw_p1_legacy_flag():
    from experiments.run_tm023cortex import make_cortex

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.episodic_act_recall = True
        assert ag._resolve_act_recall_mode() == "raw_p1"


def test_matched_clone_weights():
    from experiments.run_tm024writegeom import capacity_world
    from experiments.run_tm029indexing import RECALL_MODES, _fresh, clone_recall_mode

    world = capacity_world(0, "TM029.INDEXING.CLONE.", n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory() as tmp:
        ag = _fresh(tmp, "s", world, mode="separated_key")
        w0 = ag.W_act_query.detach().clone()
        for mode in RECALL_MODES:
            twin = clone_recall_mode(ag, mode=mode)
            assert twin.genome.act_recall_mode == mode
            assert float((twin.W_act_query.detach() - w0).abs().max()) == 0.0
            assert len(twin._episodes) == len(ag._episodes)


def test_novel_reject_semantics():
    from experiments.run_tm029indexing import novel_reject_ok

    assert novel_reject_ok({"path": "cortical_fallback", "familiar": False})
    assert not novel_reject_ok({"path": "episodic_completed", "familiar": True})
    assert not novel_reject_ok({"path": "cortical", "familiar": False})
    assert not novel_reject_ok({"path": "cortical_fallback", "familiar": True})


def test_tm029_smoke():
    from experiments.run_tm029indexing import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n_key"] == 8


def test_runner_not_patched_and_sha_matches():
    from three_memory.cortex_lineage import sha_file

    src = (REPO / "experiments" / "run_tm029indexing.py").read_text()
    assert "_patch_gr" not in src
    assert "run_tm028complementary" not in src
    assert sha_file(REPO / "experiments" / "run_tm029indexing.py") == FROZEN_RUNNER_SHA


def test_write_match_is_p1_l2_not_key():
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import EPISODE_MATCH_L2

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        key = np.zeros(64)
        key[:8] = 1.0
        p1a = np.zeros(64)
        p1a[0] = 1.0
        p1b = np.zeros(64)
        p1b[20] = 1.0
        assert float(np.linalg.norm(p1a - p1b)) > EPISODE_MATCH_L2
        ag._episode_write(p1a, "h_a", 1.0, event_key=key, key_rho=p1a)
        ag._episode_write(p1b, "h_b", 1.0, event_key=key, key_rho=p1b)
        assert len(ag._episodes) == 2


def test_raw_p1_nearest_is_v35_p1_l2():
    from experiments.run_tm023cortex import make_cortex
    from three_memory.neural_cortex import ACT_RECALL_RAW_P1

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        ag.genome.act_recall_mode = ACT_RECALL_RAW_P1
        ag.bind_actuators(["h_a", "h_b"])
        p1a = np.zeros(64)
        p1a[0] = 1.0
        p1b = np.zeros(64)
        p1b[10] = 1.0
        k = np.zeros(64)
        k[:8] = 1.0
        ag._episodes = [
            {"p1": p1a, "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True, "key": k.copy(), "key_rho": p1a.copy()},
            {"p1": p1b, "handle": "h_b", "adv": 1.0, "age": 2, "version": 1, "valid": True, "key": k.copy(), "key_rho": p1b.copy()},
        ]
        live = p1a * 0.9 + np.ones(64) * 0.01
        live = live / (np.linalg.norm(live) + 1e-12)
        stored, meta = ag._nearest_episode_for_recall(live)
        assert meta["slot"] == 0
        assert np.allclose(stored, p1a)
        _, _, dmeta = ag.actuator_decision_scores(live)
        assert dmeta["path"] == "episodic_completed"
        assert dmeta["slot"] == 0
        assert dmeta["act_recall_mode"] == ACT_RECALL_RAW_P1


def test_dev_first_match_holds_without_scale_mix():
    """Frozen treatment_stable also counts scale cells; hist still fails on its own."""
    dev = json.loads(DEV.read_text())
    core_stable = [
        c
        for c in dev["cells"]
        if c.get("kind") == "stable" and not str(c["id"]).startswith("scale|")
    ]
    hist = [c for c in dev["cells"] if c.get("kind") == "hist"]
    assert core_stable and all(bool(c.get("passed")) for c in core_stable)
    assert hist and not any(bool(c.get("passed")) for c in hist)


def test_historical_boundary_immutable():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["n_cells"] == 82
    assert dev["decision_code"] == "indexing_core_stability_fail"
    assert dec["decision"]["code"] == "indexing_core_stability_fail"
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert dec["frozen_runner_sha"] == FROZEN_RUNNER_SHA


def test_dev_refused_twice():
    from experiments.run_tm029indexing import refuse_dev_lock

    with pytest.raises(RuntimeError, match="DEV lock exists"):
        refuse_dev_lock()
