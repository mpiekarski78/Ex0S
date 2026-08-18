"""TM049 action-feedback freeze tests.

Freeze before the neural edit. No K/Q/V. No new decoder.
Leave TM046/TM047/TM048 runner/DEV/decision untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
Never add action_feedback to ACT_RECALL_MODES or GenomeConfig.
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
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_actfeed.prereg.lock"
ISO = REPO / "docs" / "lineage_actfeed.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_actfeed_contract.md"
RUNNER = REPO / "experiments" / "run_tm049actfeed.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM047_RUNNER = REPO / "experiments" / "run_tm047reinstate.py"
TM047_DEC = REPO / "docs" / "lineage_reinstate.decision.lock"
TM048_RUNNER = REPO / "experiments" / "run_tm048creditinfo.py"
TM048_DEV = REPO / "docs" / "lineage_creditinfo.dev.lock"
TM048_DEC = REPO / "docs" / "lineage_creditinfo.decision.lock"
TM048_ADD = REPO / "docs" / "lineage_creditinfo.decision.addendum.lock"
MANIFEST = "08bcc7153ac5548bed92061c6e1eef45bf90701d5e8ccf782dba15c4dc1f6608"
NEURAL_SHA_AT_FREEZE = "b0785af069c79c62bd3972a0a3f03f53f9bfbb7221accfb76061b6ee52bb0f1c"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM047_RUNNER_SHA = "c5d5a0be88e8704039c8c2e0d8e3fb86de1fc85ec69863129c5f11c26eccc6c4"
TM047_DEC_SHA = "025419faab0e67fd1342ae8670d752b582ad34180d00467a368941a82cf24ef9"
TM048_RUNNER_SHA = "57b8d4c6908908e25bd6fedcd561bf60c5a1a3b8a7e5e11e085be5596350a7c9"
TM048_DEV_SHA = "8a182169c7e3d62de7f6bab6578c3990cc456d01f08ca5ed29283aa32c6b3044"
TM048_DEC_SHA = "574b4599a9cbb7f2a0727fdac04198a17945796b9fba865ff31aedd7cb23bc13"
TM048_ADD_SHA = "49dfaf9b045597cb4dd1aaebcd22f2bfc74f0f7cb23a780d9b2feac69c5b5ced"
RUNNER_SHA = "3def01d5502b28a5ffafeab58b07ee481d5748e5c765b1cbbf52d1c1ed6f275d"
LADDER = [
    "setup_precondition_fail",
    "feedback_not_action_separable",
    "feedback_rho_fail",
    "value_projection_fail",
    "reinstatement_fail",
    "canonical_fail",
    "scalar_control_changed",
    "memory_not_necessary",
    "action_feedback_pass",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm048_wall_untouched():
    assert _sha(TM048_RUNNER) == TM048_RUNNER_SHA
    assert _sha(TM048_DEV) == TM048_DEV_SHA
    assert _sha(TM048_DEC) == TM048_DEC_SHA
    assert _sha(TM048_ADD) == TM048_ADD_SHA
    assert _sha(TM047_RUNNER) == TM047_RUNNER_SHA
    assert _sha(TM047_DEC) == TM047_DEC_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    dec = json.loads(TM048_DEC.read_text())
    add = json.loads(TM048_ADD.read_text())
    assert dec["decision"]["code"] == "credit_action_information_absent"
    assert add["frozen_first_match_unchanged"] is True
    assert add["interpretation"] == "rho_after_credit_identical_across_actions__no_downstream_recovery"
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_transition_and_lifecycle():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.49.ACTFEED"
    assert p["product"] == "0.0.004"
    assert p["kqv_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["neural_edit_authorized"] is True
    assert p["action_feedback_edit_authorized"] is True
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 6
    assert p["expected_n_cells"] == 8
    assert p["setup_excluded_from_behavioral_first_match"] is True
    assert p["flag_default"] is False
    assert p["flag_is_act_recall_mode"] is False
    assert p["flag_is_genome_field"] is False
    assert p["teacher_clamp_public_api"] == "clamp_action"
    tr = p["transition"]
    assert tr["named_op"] == "_sensory_tick"
    assert tr["n_ticks"] == 1
    assert tr["activation"] == "tanh"
    assert tr["record_sensory"] is True
    assert tr["motor_persist_mix"] is False
    assert tr["injected"] == "pending.motor_vec"
    assert tr["projection"] == "_x_tick_then_W_in"
    assert tr["new_matrix"] is False
    assert tr["feedback_hyperparameter"] is False
    assert tr["value_p1"] == "_unit_or_zero"
    life = p["lifecycle"]
    assert life["cue_state_consumed_once"] is True
    assert life["motor_vec_consumed_once"] is True
    assert life["credit_without_pending_fails_closed"] is True
    assert life["no_stale_feedback_reuse"] is True
    assert life["runner_writes_private_state"] is False
    assert life["nonpositive_adv_cannot_write_episode"] is True
    assert life["action_tick_and_write_atomic_in_observe"] is True
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "new_fitted_matrix" in p["refuse"]
    assert "feedback_specific_hyperparameter" in iso["refuse"]
    assert "three_memory/neural_cortex.py" not in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert "action_feedback" not in ACT_RECALL_MODES
    assert "action_feedback_enabled" not in GenomeConfig().to_dict()
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_no_premature_neural_edit():
    assert _sha(NEURAL) == NEURAL_SHA_AT_FREEZE
    assert not hasattr(NeuralCortex, "set_action_feedback_enabled")


def test_ids_and_decision_ladder():
    from experiments.run_tm049actfeed import BEHAVIORAL_LADDER, TRANSITION, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[:2] == ["decoder|w0", "decoder|w1"]
    assert ids[2:] == [
        "scalar_only|w0",
        "scalar_only|w1",
        "action_feedback|w0",
        "action_feedback|w1",
        "feedback_no_memory|w0",
        "feedback_no_memory|w1",
    ]
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    assert TRANSITION["n_ticks"] == 1
    assert TRANSITION["named_op"] == "_sensory_tick"
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["setup_excluded_from_behavioral_first_match"] is True
    assert fl["n_scored_cells"] == 6
    code2, _, _ = _decision(synthetic_grid(code="feedback_not_action_separable"))
    assert code2 == "feedback_not_action_separable"
    code3, _, _ = _decision(synthetic_grid(code="feedback_rho_fail"))
    assert code3 == "feedback_rho_fail"
    code4, _, _ = _decision(synthetic_grid(code="value_projection_fail"))
    assert code4 == "value_projection_fail"
    code5, _, fl5 = _decision(synthetic_grid(code="action_feedback_pass"))
    assert code5 == "action_feedback_pass"
    assert fl5["episodic_loop_complete"] is True
    assert fl5["earned_kqv"] is False
    code6, _, _ = _decision(synthetic_grid(code="scalar_control_changed"))
    assert code6 == "scalar_control_changed"
    code7, _, _ = _decision(synthetic_grid(code="memory_not_necessary"))
    assert code7 == "memory_not_necessary"


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm049actfeed import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "set_action_feedback_enabled" in src
    assert "_sensory_tick" in src
    assert "clamp_action" in src
    assert "cortex.candidate.v41.lock" in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "retrieve_by_query" not in names
    assert "actuator_scores" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 8
    assert out["n_setup_cells"] == 2
    assert out["n_scored_cells"] == 6
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_separable"] == "feedback_not_action_separable"
    assert out["ladder_rho"] == "feedback_rho_fail"
    assert out["ladder_value"] == "value_projection_fail"
    assert out["ladder_scalar"] == "scalar_control_changed"
    assert out["ladder_pass"] == "action_feedback_pass"
    assert out["ladder_nomem"] == "memory_not_necessary"
    assert out["setup_excluded"] is True
    assert out["episodic_loop_complete"] is True
    assert out["earned_kqv"] is False
    assert out["candidate_exists"] is False
    assert out["action_feedback_in_recall_modes"] is False
    assert out["action_feedback_in_genome"] is False
    assert out["api_present"] is False
    assert out["transition"]["n_ticks"] == 1
    assert "memproj_arm" not in GenomeConfig().to_dict()
