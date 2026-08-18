"""Canonical memory vs motor telemetry. Product 0.0.004.

Does not edit TM044 runner, DEV, or decision. Does not add ACT recall modes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from experiments.run_tm023cortex import make_cortex
from experiments.run_tm024actorcredit import MID_BODY, observe_cue
from experiments.run_tm024writegeom import capacity_world, mapping_pairs
from experiments.run_tm027gatedrehearsal import teach_one
from experiments.run_tm031halfspace import genome_for_registry
from three_memory.neural_cortex import (
    ACT_RECALL_OFF,
    ACT_RECALL_RAW_P1,
    MEMORY_PATH_EMPTY,
    MEMORY_PATH_EPISODIC,
    MEMORY_PATH_REJECTED,
    MEMPROJ_LEARNED,
    MEMPROJ_OFF,
    MOTOR_PATH_CORTICAL,
    NeuralCortex,
    SCORE_SRC_LIVE,
    SCORE_SRC_REINSTATED,
)
from three_memory.opaque_memory import OpaqueRow


def _fresh(tmp: str, world: dict, *, seed: int = 404500045) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / "tel", genome=genome_for_registry(seed), device="cpu")
    ag.genome.episodic_act_recall = False
    ag.bind_actuators(list(world["handles"]))
    ag.set_act_socp_arm("off")
    return ag


def test_recall_off_reports_empty_live_rho():
    ag = NeuralCortex()
    ag.genome.act_recall_mode = ACT_RECALL_OFF
    live = np.zeros(ag.genome.n, dtype=np.float64)
    live[0] = 1.0
    _scores, _addr, meta = ag.actuator_decision_scores(live)
    assert meta["memory_path"] == MEMORY_PATH_EMPTY
    assert meta["motor_path"] == MOTOR_PATH_CORTICAL
    assert meta["scoring_address_source"] == SCORE_SRC_LIVE


def test_event_memory_scores_splits_fields_and_before_after():
    ag = NeuralCortex()
    ag.set_memproj_arm(MEMPROJ_LEARNED)
    ag.genome.act_recall_mode = ACT_RECALL_OFF
    ag._last_p1 = np.zeros(ag.genome.n, dtype=np.float64)
    ag._last_p1[0] = 1.0
    scores, _addr, meta = ag.event_memory_scores()
    assert meta["memory_path"] == MEMORY_PATH_EMPTY
    assert meta["motor_path"] == MOTOR_PATH_CORTICAL
    assert meta["scoring_address_source"] == SCORE_SRC_LIVE
    assert meta["path"] == MEMORY_PATH_EMPTY
    assert meta["path"] != "cortical"
    assert meta["retrieved"] is False
    assert set(meta["scores_before_reinstatement"]) == set(scores)
    assert set(meta["scores_after_reinstatement"]) == set(scores)


def test_opaque_empty_and_rejected_memory_path():
    ag = NeuralCortex()
    ag.set_memproj_arm(MEMPROJ_LEARNED)
    ag.genome.act_recall_mode = ACT_RECALL_OFF
    n = ag.genome.n
    qlive = np.zeros(n, dtype=np.float64)
    qlive[0] = 1.0
    ag._last_p1 = qlive
    _s, _a, empty = ag.event_memory_scores()
    assert empty["memory_path"] == MEMORY_PATH_EMPTY
    assert empty["scoring_address_source"] == SCORE_SRC_LIVE
    assert empty["reject_reason"] == "empty_store"
    k = np.zeros(n, dtype=np.float64)
    k[0] = 1.0
    v1 = np.zeros(n, dtype=np.float64)
    v1[1] = 1.0
    v2 = np.zeros(n, dtype=np.float64)
    v2[2] = 1.0
    ag.opaque.replace_rows(
        [
            OpaqueRow(key=k.copy(), value=v1, when=1, provenance_id="a"),
            OpaqueRow(key=k.copy(), value=v2, when=2, provenance_id="b"),
        ]
    )
    _s2, _a2, tied = ag.event_memory_scores()
    assert tied["memory_path"] == MEMORY_PATH_REJECTED
    assert tied["scoring_address_source"] == SCORE_SRC_LIVE
    assert tied["reject_reason"] == "exact_distance_tie"
    assert tied["retrieved"] is False


def test_opaque_hit_reinstates_without_calling_it_cortical():
    ag = NeuralCortex()
    ag.set_memproj_arm(MEMPROJ_LEARNED)
    ag.genome.act_recall_mode = ACT_RECALL_OFF
    n = ag.genome.n
    live = np.zeros(n, dtype=np.float64)
    live[0] = 1.0
    ag._last_p1 = live
    k = ag._from_t(ag.W_q) @ live
    v = np.zeros(n, dtype=np.float64)
    v[3] = 1.0
    ag.opaque.write(k, v, provenance_id="hit")
    scores, addr, meta = ag.event_memory_scores()
    assert meta["retrieved"] is True
    assert meta["scoring_address_source"] == SCORE_SRC_REINSTATED
    assert meta["motor_path"] == MOTOR_PATH_CORTICAL
    assert meta["memory_path"] == MEMORY_PATH_EMPTY
    assert meta["path"] != "cortical"
    np.testing.assert_allclose(addr, v / np.linalg.norm(v), atol=1e-9)
    assert scores == meta["scores_after_reinstatement"]
    assert "scores_before_reinstatement" in meta


def test_oracle_episode_completion_is_not_labeled_cortical():
    world = capacity_world(0, "TM045.TEL.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory(prefix="tel_") as tmp:
        ag = _fresh(tmp, world)
        ag.genome.act_recall_mode = ACT_RECALL_RAW_P1
        ag.set_memproj_arm(MEMPROJ_OFF)
        for i, (cue, handle) in enumerate(pairs):
            teach_one(ag, world, handle, tag=f"t{i}", symbols=[cue])
        observe_cue(ag, world, tag="p0", body=list(MID_BODY), symbols=[pairs[0][0]])
        _scores, _addr, meta = ag.event_memory_scores()
    assert meta["memory_path"] == MEMORY_PATH_EPISODIC
    assert meta["motor_path"] == MOTOR_PATH_CORTICAL
    assert meta["scoring_address_source"] == SCORE_SRC_REINSTATED
    assert meta["path"] == MEMORY_PATH_EPISODIC
    assert meta["path"] != "cortical"
    assert meta["slot"] is not None
    assert "scores_before_reinstatement" in meta
    assert "scores_after_reinstatement" in meta
