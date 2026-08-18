"""v40 candidate review pins. No neural or solver edits. Product 0.0.004."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    ACT_SOCP_FALLBACK,
    ACT_SOCP_OFF,
    EPISODE_MATCH_L2,
    EPISODE_SLOTS,
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
CAND = REPO / "docs" / "cortex.candidate.v40.lock"
REVIEW = REPO / "docs" / "lineage_v40.candidate_review.lock"
LIVE = REPO / "docs" / "cortex.candidate.lock"
V30 = REPO / "docs" / "cortex.candidate.v30.lock"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
TM042_DEC = REPO / "docs" / "lineage_postinstall.decision.lock"
REVIEW_SHA = "b931aa6c6332048b68e97b8444466855ab6c91d3d7f38632caef257efb7818cf"
CAND_SHA = "dc8c13d1607034781864f1dcfd969ad146bf267fd78fb0ba588a88fe2a0e0319"
NEURAL_SHA = "2eb45d8769402330f5ee39a04afffe110a435a0e64a40b12bc2d874b36f5ed59"
SOLVER_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
LIVE_SHA = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_review_accepts_narrow_claim():
    rev = json.loads(REVIEW.read_text())
    cand = json.loads(CAND.read_text())
    assert _sha(REVIEW) == REVIEW_SHA
    assert _sha(CAND) == CAND_SHA
    assert rev["answer"] == "yes_as_computational_slow_consolidation_organ"
    assert rev["verdict"] == "accept"
    assert rev["product"] == "0.0.004"
    assert rev["earned_next"] is False
    assert rev["eligible_for_000005"] is False
    assert rev["further_dev_run"] is False
    assert rev["candidate_discussion_open_on_frozen_tm042_ladder"] is False
    assert all(rev["checklist"].values())
    assert cand["version"] == "TM.0.23.CORTEX.CANDIDATE.V40"
    assert cand["product"] == "0.0.004"
    assert cand["earned_next"] is False
    assert cand["candidate_arm"] == "fallback_joint"
    assert cand["default_arm"] == "off"
    assert cand["instance_flag_not_genomeconfig"] is True
    assert cand["claim"].startswith("Frozen v37 local learning")
    assert "biologically local learning" in cand["not_claimed"]
    assert cand["tested_bound"] == {
        "episode_slots": 8,
        "n_cues": 8,
        "n_handles": 4,
        "write_match_l2": 0.05,
        "n": 64,
    }
    assert cand["neural_cortex_sha"] == NEURAL_SHA
    assert cand["joint_socp_sha"] == SOLVER_SHA
    assert cand["review_lock_sha"] == REVIEW_SHA
    assert EPISODE_SLOTS == 8
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_live_pointer_and_tm042_ladder_unchanged():
    assert _sha(LIVE) == LIVE_SHA
    assert _sha(V30) == LIVE_SHA
    live = json.loads(LIVE.read_text())
    assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V30"
    tm042 = json.loads(TM042_DEC.read_text())
    assert tm042["decision"]["code"] == "postinstall_mech_install_fail"
    assert tm042["decision"]["phase_flags"]["candidate_discussion_open"] is False
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(SOLVER) == SOLVER_SHA


def test_init_and_checkpoint_without_harness_restore():
    ag = NeuralCortex()
    assert ag._act_socp_arm == ACT_SOCP_OFF
    assert "act_socp_arm" not in GenomeConfig().to_dict()
    ag.set_act_socp_arm(ACT_SOCP_FALLBACK)
    snap = ag.checkpoint()
    twin = NeuralCortex()
    twin.load_checkpoint(snap)
    assert twin._act_socp_arm == ACT_SOCP_FALLBACK
    missing = dict(snap)
    missing.pop("act_socp_arm")
    off = NeuralCortex()
    off.load_checkpoint(missing)
    assert off._act_socp_arm == ACT_SOCP_OFF
