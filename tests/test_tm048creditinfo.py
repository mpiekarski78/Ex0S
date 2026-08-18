"""TM048 credit-information freeze tests.

No K/Q/V. No new decoder. No action-feedback edit.
Leave TM046/TM047 runner/DEV/decision untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2, GenomeConfig

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_creditinfo.prereg.lock"
ISO = REPO / "docs" / "lineage_creditinfo.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_creditinfo_contract.md"
RUNNER = REPO / "experiments" / "run_tm048creditinfo.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM046_DEV = REPO / "docs" / "lineage_oneshot.dev.lock"
TM046_DEC = REPO / "docs" / "lineage_oneshot.decision.lock"
TM046_ADD = REPO / "docs" / "lineage_oneshot.decision.addendum.lock"
TM047_RUNNER = REPO / "experiments" / "run_tm047reinstate.py"
TM047_DEV = REPO / "docs" / "lineage_reinstate.dev.lock"
TM047_DEC = REPO / "docs" / "lineage_reinstate.decision.lock"
MANIFEST = "ebcc409e7e829ba4ba5883edd7c2181db1a42d8f833804e5d83d9f19024c39a5"
NEURAL_SHA = "b0785af069c79c62bd3972a0a3f03f53f9bfbb7221accfb76061b6ee52bb0f1c"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM046_DEV_SHA = "68088c1728e9c3367c5cd30bd88b7adc8df30502afa7db3d5a1546b13fa6110d"
TM046_DEC_SHA = "da0e4e82cbaca107543029af16cd0bfe5cfc6027b457c2798b5c34134ec24323"
TM046_ADD_SHA = "8afcc27a9919baaf4052323b46b639129961848c4a7a30a6c9bfa920d0b6f337"
TM047_RUNNER_SHA = "c5d5a0be88e8704039c8c2e0d8e3fb86de1fc85ec69863129c5f11c26eccc6c4"
TM047_DEV_SHA = "5c3c7a8172f90f1265fedb1cb5e9659840f5a528932fbf34045cfc0573e4aa53"
TM047_DEC_SHA = "025419faab0e67fd1342ae8670d752b582ad34180d00467a368941a82cf24ef9"
RUNNER_SHA = "57b8d4c6908908e25bd6fedcd561bf60c5a1a3b8a7e5e11e085be5596350a7c9"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm046_and_tm047_untouched():
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    assert _sha(TM046_DEV) == TM046_DEV_SHA
    assert _sha(TM046_DEC) == TM046_DEC_SHA
    assert _sha(TM046_ADD) == TM046_ADD_SHA
    assert _sha(TM047_RUNNER) == TM047_RUNNER_SHA
    assert _sha(TM047_DEV) == TM047_DEV_SHA
    assert _sha(TM047_DEC) == TM047_DEC_SHA
    dec046 = json.loads(TM046_DEC.read_text())
    add = json.loads(TM046_ADD.read_text())
    dec047 = json.loads(TM047_DEC.read_text())
    assert dec046["decision"]["code"] == "generic_reinstatement_fail"
    assert add["frozen_first_match_unchanged"] is True
    assert dec047["decision"]["code"] == "credit_rho_fail"
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_and_no_neural_edit():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.48.CREDITINFO"
    assert p["product"] == "0.0.004"
    assert p["kqv_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["neural_edit_authorized"] is False
    assert p["action_feedback_edit_authorized"] is False
    assert p["v41_candidate_authorized"] is False
    assert iso["implementation_authorized"] is False
    assert iso["action_feedback_edit_authorized"] is False
    assert p["n"] == 64
    assert p["n_credit_clones"] == 4
    assert p["expected_n_cells"] == 4
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["reconstruction_domain"] == "TM046.ONESHOT.DEV."
    assert p["tm047_decision_code"] == "credit_rho_fail"
    assert p["tm046_decision_code"] == "generic_reinstatement_fail"
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert [d["code"] for d in p["decision_ladder"]] == [
        "setup_precondition_fail",
        "credit_action_information_absent",
        "credit_trace_present_not_decodable",
        "value_projection_loss",
        "credit_information_pass",
    ]
    assert "implement_action_feedback" in p["refuse"]
    assert "copy_action_into_S" in p["refuse"]
    assert "experiments/run_tm047reinstate.py" in iso["historical_immutable"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(NEURAL) == NEURAL_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_ids_and_decision_ladder():
    from experiments.run_tm048creditinfo import _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids == ["decoder|w0", "decoder|w1", "credit|w0", "credit|w1"]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    code2, _, fl2 = _decision(synthetic_grid(code="credit_action_information_absent"))
    assert code2 == "credit_action_information_absent"
    assert fl2["earned_action_feedback"] is True
    assert fl2["earned_kqv"] is False
    code3, _, fl3 = _decision(synthetic_grid(code="credit_trace_present_not_decodable"))
    assert code3 == "credit_trace_present_not_decodable"
    assert fl3["earned_action_feedback"] is False
    code4, _, _ = _decision(synthetic_grid(code="value_projection_loss"))
    assert code4 == "value_projection_loss"
    code5, _, fl5 = _decision(synthetic_grid(code="credit_information_pass"))
    assert code5 == "credit_information_pass"
    assert fl5["earned_action_feedback"] is False


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm048creditinfo import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "freeze_plasticity" in src
    assert "credit_one" in src
    assert "cortex.candidate.v41.lock" in src
    assert "copied_action_into_s" in src
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
    assert out["n_cells"] == 4
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_absent"] == "credit_action_information_absent"
    assert out["ladder_trace"] == "credit_trace_present_not_decodable"
    assert out["ladder_proj"] == "value_projection_loss"
    assert out["ladder_pass"] == "credit_information_pass"
    assert out["earned_action_feedback"] is True
    assert out["pass_earned_feedback"] is False
    assert out["candidate_exists"] is False
    assert out["action_feedback_edit_authorized"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
