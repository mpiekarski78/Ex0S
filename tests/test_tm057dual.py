"""TM057 dual-tape drift freeze tests.

Diagnostic only. No neural edit. Do not install W_star. Do not retune 0.05.
Leave TM046–TM056 runner/DEV/decision/addendum untouched.
v_t is the write argument. Resident value is what S retains.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    EPISODE_SLOTS,
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_dual.prereg.lock"
ISO = REPO / "docs" / "lineage_dual.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_dual_contract.md"
RUNNER = REPO / "experiments" / "run_tm057dual.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
LAW = REPO / "docs" / "lineage_write_time_law.lock"
TM056_RUNNER = REPO / "experiments" / "run_tm056epprov.py"
TM056_DEV = REPO / "docs" / "lineage_epprov.dev.lock"
TM056_DEC = REPO / "docs" / "lineage_epprov.decision.lock"
TM056_ADD = REPO / "docs" / "lineage_epprov.decision.addendum.lock"
MANIFEST = "5e370740191417e33d1fa75279367a43f78c8f74941d670f8b21973e2aee28a1"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM056_RUNNER_SHA = "8e89a96393ed2543247d9de3f7a8a019073e62e948d8f94d09c66b09644be656"
TM056_DEV_SHA = "793429c7eada8b768433c467266548168c003bbd3f9819ceb634c06944355ad4"
TM056_DEC_SHA = "179b6f15986b4d99eeed81790771c371c6ce656ddc3d89376dd3a672a60c00be"
TM056_ADD_SHA = "3ba67414bfbd8c054c2b7478479d3f7faa7fdf0ec7f7c8a52651ad4971282f17"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
RUNNER_SHA = "1f1ee4b8d4d2da7893622d8692a91b3912ed7130f9a868dffe02fc19d5cd8f61"
LADDER = [
    "setup_precondition_fail",
    "storage_integrity_failure",
    "cue_addressing_collapsed",
    "prefix_infeasible",
    "representation_drift",
    "storage_selection_capacity_wall",
    "grounding_consolidation_plausible",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm056_wall_untouched():
    assert _sha(TM056_RUNNER) == TM056_RUNNER_SHA
    assert _sha(TM056_DEV) == TM056_DEV_SHA
    assert _sha(TM056_DEC) == TM056_DEC_SHA
    assert _sha(TM056_ADD) == TM056_ADD_SHA
    assert _sha(LAW) == LAW_SHA
    dec = json.loads(TM056_DEC.read_text())
    add = json.loads(TM056_ADD.read_text())
    law = json.loads(LAW.read_text())
    assert dec["decision"]["code"] == "replaced_under_write_law"
    assert add["interpretation"] == "tm052_selected_surviving_records"
    assert add["rest_innocent"] is True
    assert add["canonical_law_reconsidered"] is False
    assert law["canonical_state_generator"] == "write_time_last_p1"
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_dual_tapes():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.57.DUAL"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["episode_match_l2_retune_authorized"] is False
    assert p["canonical_law_reconsider_authorized"] is False
    assert p["v_t_is_episode_write_argument"] is True
    assert p["resident_value_is_what_S_retains"] is True
    assert p["unique_cues_per_event"] is True
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 16
    assert p["expected_n_cells"] == 18
    assert p["n_dev_repeats"] == 4
    assert p["n_online_repeats"] == 8
    assert p["n_grid"] == [4, 8, 16, 24]
    assert p["arms"] == ["attempts", "residents"]
    assert p["floor"] == 0.05
    assert p["discard_every_W_star"] is True
    assert p["seed_registry"] == 404900057
    assert 404900056 in p["forbidden_seeds"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "retune_EPISODE_MATCH_L2" in p["refuse"]
    assert "experiments/run_tm056epprov.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_no_neural_edit():
    assert _sha(NEURAL) == NEURAL_SHA


def test_ids_and_decision_ladder():
    from experiments.run_tm057dual import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[0] == "decoder|w0"
    assert ids[-1] == "residents|n24|w1"
    assert len(ids) == 18
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["canonical_law_reconsidered"] is False
        if step == "grounding_consolidation_plausible":
            assert flags["generic_consolidation_plausible"] is True
            assert flags["p1_replacement_law_compatible_with_opaque_kv"] is True
        if step in ("storage_integrity_failure", "cue_addressing_collapsed"):
            assert flags["p1_replacement_law_compatible_with_opaque_kv"] is False


def test_runner_receipts_and_smoke():
    from experiments.run_tm057dual import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "attempted_v_hash" in src
    assert "refresh_crosses_cue" in src
    assert "refresh_crosses_action" in src
    assert "latest_episode(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "_run_joint_socp_consolidation" not in names
    assert "latest_episode" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 18
    assert out["ladder"]["setup"] == "setup_precondition_fail"
    assert out["ladder"]["storage_integrity_failure"] == "storage_integrity_failure"
    assert out["ladder"]["cue_addressing_collapsed"] == "cue_addressing_collapsed"
    assert out["ladder"]["storage_selection_capacity_wall"] == "storage_selection_capacity_wall"
    assert out["ladder"]["grounding_consolidation_plausible"] == "grounding_consolidation_plausible"
    assert out["floor"] == 0.05
    assert out["slots"] == 8
    assert out["candidate_exists"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    assert not hasattr(NeuralCortex, "set_feedback_ticks")
