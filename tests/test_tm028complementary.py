"""TM028 complementary recall tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_complementary.prereg.lock"
DEV = REPO / "docs" / "lineage_complementary.dev.lock"
DEC = REPO / "docs" / "lineage_complementary.decision.lock"
V35_ISO = REPO / "docs" / "cortex_v35.isolation.lock"
V35_CLOSURE = REPO / "docs" / "cortex_v35.closure.lock"
LINEAGE_CLOSURE = REPO / "docs" / "lineage_complementary.v35.closure.lock"
MANIFEST = "4c59793a32573143a57b22a10a16728f1be3323b6c2d9b4d11b9b558e42f894c"
HISTORICAL_DEV_SHA = "7aeb60f284aedd492a252db62be22a28882761244b1df94badbebfaaf5a823d0"
HISTORICAL_DEC_SHA = "0f879cd2e4f5913f1850d7f9d63e92388fb21f3267e3351f5a74bbb1b6feffaa"
HISTORICAL_V35_ISO_SHA = "8d1b72fc45aac48f72f38d9ed753e37de81c75df2a0a1b23ee6d880f8b42f8d8"
FROZEN_RUNNER_SHA = "e36339cf7f7fcf59e520e4ed9fae65a380e6cc53b56ab0adbfeceb2b7b2569cd"


def test_prereg_manifest():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 66
    assert p["recall_rule"] == "unique_nearest_no_radius"
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA


def test_historical_boundary_immutable():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    assert _sha(V35_ISO) == HISTORICAL_V35_ISO_SHA
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dec["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA


def test_v35_closure_does_not_mutate_isolation():
    assert V35_CLOSURE.exists()
    assert LINEAGE_CLOSURE.exists()
    c = json.loads(V35_CLOSURE.read_text())
    lc = json.loads(LINEAGE_CLOSURE.read_text())
    assert c["rewrite_historical_dev"] is False
    assert lc["rewrite_historical_dev"] is False
    assert c["closed_at_git"].startswith("42ce89f")
    assert c["historical_isolation_sha"] == HISTORICAL_V35_ISO_SHA
    assert lc["historical_isolation_sha"] == HISTORICAL_V35_ISO_SHA
    assert _sha(V35_ISO) == c["historical_isolation_sha"]


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_dev_lock_exists():
    assert DEV.exists()
    dev = json.loads(DEV.read_text())
    assert dev["n_cells"] == 66
    assert dev["decision_code"] == "complementary_core_acquire_fail"


def test_actuator_decision_scores_no_handle_shortcut():
    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024writegeom import capacity_world
    from three_memory.neural_cortex import NeuralCortex

    world = capacity_world(0, "TM028.COMPLEMENTARY.TEST.", n_cues=2, n_handles=2)
    ag = make_cortex(REPO / "tmp_test_cortex", device="cpu")
    ag.genome.episodic_act_recall = True
    ag.bind_actuators(list(world["handles"]))
    p1 = np.random.default_rng(0).normal(0, 1, size=ag.genome.n)
    p1 = p1 / (np.linalg.norm(p1) + 1e-12)
    ag._episodes.append(
        {"p1": p1.copy(), "handle": world["handles"][0], "adv": 1.0, "age": 1, "version": 1, "valid": True}
    )
    live = np.random.default_rng(1).normal(0, 1, size=ag.genome.n)
    scores, addr, meta = ag.actuator_decision_scores(live)
    assert meta["path"] in ("episodic_completed", "cortical_fallback", "cortical")
    assert isinstance(scores, dict)
    assert addr.shape == (ag.genome.n,)


def test_matched_clone_weights():
    import copy
    import tempfile

    from experiments.run_tm028complementary import _fresh, clone_recall_variant
    from experiments.run_tm024writegeom import capacity_world

    world = capacity_world(0, "TM028.COMPLEMENTARY.CLONE.", n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory() as tmp:
        ag = _fresh(tmp, "s", world, recall=True)
        snap = ag.checkpoint()
        off = clone_recall_variant(ag, recall=False)
        on = clone_recall_variant(ag, recall=True)
        assert off.genome.episodic_act_recall is False
        assert on.genome.episodic_act_recall is True
        assert len(off._episodes) == len(on._episodes)
        w0 = off.W_act_query.detach().clone()
        w1 = on.W_act_query.detach().clone()
        assert float((w0 - w1).abs().max()) == 0.0


def test_tm028_smoke():
    from experiments.run_tm028complementary import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["episodic_act_recall"] is True


def test_gr_patch_does_not_leak():
    from experiments.run_tm028complementary import _ORIGINAL_ACT_GEOMETRIC_MARGIN, smoke
    from three_memory.neural_cortex import NeuralCortex

    smoke()
    assert NeuralCortex._act_geometric_margin is _ORIGINAL_ACT_GEOMETRIC_MARGIN


def test_episodic_act_recall_default_off():
    import tempfile

    from experiments.run_tm023cortex import make_cortex

    with tempfile.TemporaryDirectory() as tmp:
        ag = make_cortex(Path(tmp), device="cpu")
        assert ag.genome.episodic_act_recall is False
        _, _, meta = ag.actuator_decision_scores(np.zeros(ag.genome.n))
        assert meta["path"] == "cortical"


def test_dev_refused_twice():
    from experiments.run_tm028complementary import refuse_dev_lock

    with pytest.raises(RuntimeError, match="DEV lock exists"):
        refuse_dev_lock()
