"""TM056 episode-provenance freeze tests.

Diagnostic only. No neural edit. Do not install W_star. Do not retune 0.05.
Leave TM046–TM055 runner/DEV/decision/addendum untouched.
Canonical generator is the _episode_write argument. Product 0.0.004.
Never write cortex.candidate.v41.lock.
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
PREREG = REPO / "docs" / "lineage_epprov.prereg.lock"
ISO = REPO / "docs" / "lineage_epprov.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_epprov_contract.md"
RUNNER = REPO / "experiments" / "run_tm056epprov.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
LAW = REPO / "docs" / "lineage_write_time_law.lock"
TM055_RUNNER = REPO / "experiments" / "run_tm055drift.py"
TM055_DEV = REPO / "docs" / "lineage_drift.dev.lock"
TM055_DEC = REPO / "docs" / "lineage_drift.decision.lock"
TM055_ADD = REPO / "docs" / "lineage_drift.decision.addendum.lock"
MANIFEST = "7371096ac620854187739ec37c2a9f71dca1a592daf7a6762a6731e492d93484"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM055_RUNNER_SHA = "23a3002029560e6a83d5ae5646e5631101fee6b89981e8cd814688e87f9a392b"
TM055_DEV_SHA = "177395281c25171628132c38eeda1056321d5a1cf9887600e73514f5558e7f8c"
TM055_DEC_SHA = "9040428ebadb8cc0e692eca3825dc74d4f52539ba4207ec19f9748eacca69caf"
TM055_ADD_SHA = "45c4279997d1bf1c9c67e769484bdb60da31f2af807455044a12458c210ab20b"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
RUNNER_SHA = "8e89a96393ed2543247d9de3f7a8a019073e62e948d8f94d09c66b09644be656"
LADDER = [
    "setup_precondition_fail",
    "reconstruction_setup_mismatch",
    "pinned_live_last_p1",
    "p1_mutated_during_rest",
    "replaced_under_write_law",
    "later_record_selected",
    "tm052_measured_wrong_record",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm055_wall_untouched():
    assert _sha(TM055_RUNNER) == TM055_RUNNER_SHA
    assert _sha(TM055_DEV) == TM055_DEV_SHA
    assert _sha(TM055_DEC) == TM055_DEC_SHA
    assert _sha(TM055_ADD) == TM055_ADD_SHA
    assert _sha(LAW) == LAW_SHA
    dec = json.loads(TM055_DEC.read_text())
    add = json.loads(TM055_ADD.read_text())
    law = json.loads(LAW.read_text())
    assert dec["decision"]["code"] == "setup_precondition_fail"
    assert add["interpretation"] == "valid_setup_stop"
    assert add["canonical_law_reconsidered"] is False
    assert law["canonical_state_generator"] == "write_time_last_p1"
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_write_identity():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.56.EPPROV"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["socp_scoring_authorized"] is False
    assert p["canonical_law_reconsider_authorized"] is False
    assert p["v_t_is_episode_write_argument"] is True
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 2
    assert p["expected_n_cells"] == 4
    assert p["n_dev_repeats"] == 4
    assert p["episode_slots"] == 8
    assert p["floor"] == 0.05
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["seed_registry"] == 404900056
    assert 404900055 in p["forbidden_seeds"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "identify target by live _last_p1" in p["refuse"]
    assert "experiments/run_tm055drift.py" in iso["historical_immutable"]
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
    from experiments.run_tm056epprov import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids == ["decoder|w0", "decoder|w1", "provenance|w0", "provenance|w1"]
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["socp_scored"] is False
        assert flags["canonical_law_reconsidered"] is False
        assert flags["target_is_write_provenance"] is True


def test_runner_refuses_selectors_and_smoke():
    from experiments.run_tm056epprov import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "_episode_write" in src
    assert "write_id" in src
    assert "latest_episode(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "solve_min_change_socp" not in names
    assert "_run_joint_socp_consolidation" not in names
    assert "latest_episode" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 4
    assert out["ladder"]["setup"] == "setup_precondition_fail"
    assert out["ladder"]["later_record_selected"] == "later_record_selected"
    assert out["floor"] == 0.05
    assert out["slots"] == 8
    assert out["candidate_exists"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    assert not hasattr(NeuralCortex, "set_feedback_ticks")


DEV_SHA = "793429c7eada8b768433c467266548168c003bbd3f9819ceb634c06944355ad4"
DEC_SHA = "179b6f15986b4d99eeed81790771c371c6ce656ddc3d89376dd3a672a60c00be"
DEV_GIT = "7d62fe0d7c632b25340706904f5d99004ce15b55"


def test_dev_lock_replaced_under_write_law():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm056epprov import expected_cell_ids

    devp = REPO / "docs" / "lineage_epprov.dev.lock"
    decp = REPO / "docs" / "lineage_epprov.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM055_RUNNER) == TM055_RUNNER_SHA
    assert _sha(TM055_DEV) == TM055_DEV_SHA
    assert _sha(TM055_ADD) == TM055_ADD_SHA
    assert _sha(LAW) == LAW_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "replaced_under_write_law"
    assert dev["install_W_star"] is False
    assert dev["socp_scored"] is False
    assert dev["canonical_law_reconsidered"] is False
    assert dec["architectural_conclusion"] == "none"
    assert dec["decision"]["code"] == "replaced_under_write_law"
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["w0_match"] is True
    assert cells["decoder|w1"]["w0_match"] is True
    assert cells["decoder|w0"]["n_rest_writes"] == 0
    assert cells["provenance|w0"]["scan_matches_pins"] is True
    assert cells["provenance|w1"]["scan_matches_pins"] is True
    assert cells["provenance|w0"]["target_is_write_provenance"] is True
    assert cells["provenance|w0"]["target_is_handle_lookup"] is False
    assert cells["provenance|w0"]["target_is_live_last_p1"] is False
    assert cells["provenance|w0"]["event_cause"][-1] == "in_place_refresh"
    assert cells["provenance|w0"]["event_arg_p1"][-1] != cells["provenance|w0"]["event_stored_p1"][-1]
    rest = next(iter(cells["provenance|w0"]["after_rest"].values()))
    assert rest["present"] is True
    assert rest["p1_unchanged"] is True


ADD_SHA = "3ba67414bfbd8c054c2b7478479d3f7faa7fdf0ec7f7c8a52651ad4971282f17"


def test_addendum_surviving_records_without_rewrite():
    addp = REPO / "docs" / "lineage_epprov.decision.addendum.lock"
    assert _sha(addp) == ADD_SHA
    assert _sha(REPO / "docs" / "lineage_epprov.dev.lock") == DEV_SHA
    assert _sha(REPO / "docs" / "lineage_epprov.decision.lock") == DEC_SHA
    assert _sha(RUNNER) == RUNNER_SHA
    add = json.loads(addp.read_text())
    assert add["rewrite_historical_decision"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "replaced_under_write_law"
    assert add["interpretation"] == "tm052_selected_surviving_records"
    assert add["rest_innocent"] is True
    assert add["live_state_telemetry_innocent"] is True
    assert add["canonical_law_reconsidered"] is False
    assert add["episode_match_l2_not_changed"] is True
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()


