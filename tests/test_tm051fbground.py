"""TM051 feedback-grounding freeze tests.

Freeze before the neural edit. No new decoder. No K/Q/V.
Leave TM046–TM050 runner/DEV/decision untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
credit_token is absent until the authorized clamp_action edit.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
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
PREREG = REPO / "docs" / "lineage_fbground.prereg.lock"
ISO = REPO / "docs" / "lineage_fbground.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_fbground_contract.md"
RUNNER = REPO / "experiments" / "run_tm051fbground.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM047_RUNNER = REPO / "experiments" / "run_tm047reinstate.py"
TM048_RUNNER = REPO / "experiments" / "run_tm048creditinfo.py"
TM049_RUNNER = REPO / "experiments" / "run_tm049actfeed.py"
TM049_DEV = REPO / "docs" / "lineage_actfeed.dev.lock"
TM049_DEC = REPO / "docs" / "lineage_actfeed.decision.lock"
TM050_RUNNER = REPO / "experiments" / "run_tm050feedgeom.py"
TM050_DEV = REPO / "docs" / "lineage_feedgeom.dev.lock"
TM050_DEC = REPO / "docs" / "lineage_feedgeom.decision.lock"
TM050_ADD = REPO / "docs" / "lineage_feedgeom.decision.addendum.lock"
MANIFEST = "05a18a09dc46ae09665ef4ed196d5abef846d7ca492cf94464edbdf87ccb78b6"
NEURAL_SHA_AT_FREEZE = "a33f04479716d21624f9f8d0167ceaf4a658fd57a9070b058933f71fa1ae155c"
NEURAL_SHA_AFTER_EDIT = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM047_RUNNER_SHA = "c5d5a0be88e8704039c8c2e0d8e3fb86de1fc85ec69863129c5f11c26eccc6c4"
TM048_RUNNER_SHA = "57b8d4c6908908e25bd6fedcd561bf60c5a1a3b8a7e5e11e085be5596350a7c9"
TM049_RUNNER_SHA = "3def01d5502b28a5ffafeab58b07ee481d5748e5c765b1cbbf52d1c1ed6f275d"
TM049_DEV_SHA = "5d956e80abef0e41beda251acdd5ae23e1c5eff1c0862fe5f8ae652455d532e0"
TM049_DEC_SHA = "86fca1a366c290ce072efd522b8071fb5771de01e9fbee90e2adda15fda9760d"
TM050_RUNNER_SHA = "504925a78645f32576e90e5b734a99dc31171471ae1a5db599a7be53b7452ba1"
TM050_DEV_SHA = "ede465c08e9f5af4da7b17343e30ce111d1d81230c015d3686bb76d4b14ef228"
TM050_DEC_SHA = "3acd39eee4f3d10e295178067fd3abce86daa7cb11f09752234039c1e8a927bc"
TM050_ADD_SHA = "e01f7d69a5c5e5e433db8b31841a3b61b24c43c8c053a0cfcd2fe1f10bc4a162"
RUNNER_SHA = "f73c8671db2f3bac6f6b4e22eb08687d933559e476d5c2adb2ff4c879c230706"
LADDER = [
    "setup_precondition_fail",
    "reference_control_changed",
    "heldout_feedback_decode_fail",
    "shared_decoder_interference",
    "shuffled_not_causal",
    "value_projection_fail",
    "reinstatement_fail",
    "canonical_fail",
    "memory_not_necessary",
    "feedback_grounding_pass",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm050_wall_untouched():
    assert _sha(TM050_RUNNER) == TM050_RUNNER_SHA
    assert _sha(TM050_DEV) == TM050_DEV_SHA
    assert _sha(TM050_DEC) == TM050_DEC_SHA
    assert _sha(TM050_ADD) == TM050_ADD_SHA
    assert _sha(TM049_RUNNER) == TM049_RUNNER_SHA
    assert _sha(TM049_DEV) == TM049_DEV_SHA
    assert _sha(TM049_DEC) == TM049_DEC_SHA
    assert _sha(TM048_RUNNER) == TM048_RUNNER_SHA
    assert _sha(TM047_RUNNER) == TM047_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    dec = json.loads(TM050_DEC.read_text())
    add = json.loads(TM050_ADD.read_text())
    assert dec["decision"]["code"] == "states_separate_never_decode"
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "states_separate_never_decode"
    assert add["interpretation"] == "action_identity_preserved_existing_decoder_ungrounded"
    assert add["audit"]["stronger_signal_not_earned"] is True
    assert add["audit"]["new_decoder_not_earned"] is True
    assert add["audit"]["next_wall"] == "TM.0.51.FBGROUND"
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_grounding_wall():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.51.FBGROUND"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is True
    assert p["kqv_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["action_feedback_edit_authorized"] is False
    assert p["tick_count_fit_authorized"] is False
    assert p["v41_candidate_authorized"] is False
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 8
    assert p["expected_n_cells"] == 10
    assert p["setup_excluded_from_behavioral_first_match"] is True
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["reconstruction_domain"] == "TM046.ONESHOT.DEV."
    assert p["seed_registry"] == 404900051
    assert 404600046 in p["forbidden_seeds"]
    assert 404900049 in p["forbidden_seeds"]
    assert 404900050 in p["forbidden_seeds"]
    assert p["seed_registry"] != p["reconstruction_seed_registry"]
    assert p["protocols"] == ["reference_only", "feedback_grounded", "shuffled_grounding"]
    assert p["arms"] == ["reference_only", "feedback_grounded", "shuffled_grounding", "feedback_no_memory"]
    assert p["permutation"] == "rotate_plus_one"
    assert p["wrapped_state"] == "tm049_observe_last_p1"
    assert p["not_tm050_isolated_insertion"] is True
    assert p["credit_token_public_api"] is True
    assert p["kqv_not_tuned"] is True
    assert p["decoder_not_retrained"] is True
    assert p["tm050_decision_code"] == "states_separate_never_decode"
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "train_new_decoder" in p["refuse"]
    assert "scale_motor_vec" in p["refuse"]
    assert "runner_edits_after_freeze_push" in p["refuse"]
    assert "three_memory/neural_cortex.py" not in iso["historical_immutable"]
    assert "experiments/run_tm049actfeed.py" in iso["historical_immutable"]
    assert "experiments/run_tm050feedgeom.py" in iso["historical_immutable"]
    assert "three_memory/joint_socp.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert p["neural_cortex_sha_at_freeze"] == NEURAL_SHA_AT_FREEZE
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert "action_feedback" not in ACT_RECALL_MODES
    assert "action_feedback_enabled" not in GenomeConfig().to_dict()
    assert "credit_token" not in GenomeConfig().to_dict()
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_authorized_neural_edit_matches_freeze():
    import numpy as np

    assert _sha(NEURAL) == NEURAL_SHA_AFTER_EDIT
    assert _sha(NEURAL) != NEURAL_SHA_AT_FREEZE
    sig = inspect.signature(NeuralCortex.clamp_action)
    assert "credit_token" in sig.parameters
    assert sig.parameters["credit_token"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "credit_token" not in GenomeConfig().to_dict()
    assert "credit_token" not in ACT_RECALL_MODES
    src = NEURAL.read_text()
    assert "W_feedback" not in src
    assert "feedback_scale" not in src
    mid = [0.5, 0.4, 0.5, 0.0]
    ag = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
    ag.bind_actuators(["h_a", "h_b"])
    ag.observe(
        {
            "interaction_token": "ct_sel",
            "source_token": "src_t",
            "ordered_symbols": ["cue"],
            "observable_state": ["st_idle"],
            "body_state": list(mid),
        }
    )
    default = ag.clamp_action("ACT", "h_a")
    assert default["ok"] is True
    assert default["token"] == "h_a"
    assert np.allclose(ag._pending["motor_vec"], ag.motor_vocab["h_a"])
    assert ag._pending["token"] == "h_a"
    shuffled = ag.clamp_action("ACT", "h_a", credit_token="h_b")
    assert shuffled["ok"] is True
    assert shuffled["token"] == "h_a"
    assert np.allclose(ag._pending["motor_vec"], ag.motor_vocab["h_a"])
    assert not np.allclose(ag._pending["motor_vec"], ag.motor_vocab["h_b"])
    assert ag._pending["token"] == "h_b"


def test_ids_and_decision_ladder():
    from experiments.run_tm051fbground import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[:2] == ["decoder|w0", "decoder|w1"]
    assert ids[2:] == [
        "reference_only|w0",
        "reference_only|w1",
        "feedback_grounded|w0",
        "feedback_grounded|w1",
        "shuffled_grounding|w0",
        "shuffled_grounding|w1",
        "feedback_no_memory|w0",
        "feedback_no_memory|w1",
    ]
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["setup_excluded_from_behavioral_first_match"] is True
    assert fl["n_scored_cells"] == 8
    assert fl["new_decoder"] is False
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["earned_kqv"] is False
        assert flags["new_decoder"] is False
        if step == "feedback_grounding_pass":
            assert flags["episodic_loop_complete"] is True
        else:
            assert flags["episodic_loop_complete"] is False


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm051fbground import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "credit_token" in src
    assert "set_action_feedback_enabled" in src
    assert "ground_one" in src
    assert "permute_handle" in src
    assert "s_hold_" in src
    assert "s_dev_" in src
    assert "credit_tagged" in src
    assert "cortex.candidate.v41.lock" in src
    assert "teach_one(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "retrieve_by_query" not in names
    assert "actuator_scores" not in names
    assert "teach_one" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 10
    assert out["n_setup_cells"] == 2
    assert out["n_scored_cells"] == 8
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_reference"] == "reference_control_changed"
    assert out["ladder_heldout"] == "heldout_feedback_decode_fail"
    assert out["ladder_shared"] == "shared_decoder_interference"
    assert out["ladder_shuffled"] == "shuffled_not_causal"
    assert out["ladder_value"] == "value_projection_fail"
    assert out["ladder_reinstatement"] == "reinstatement_fail"
    assert out["ladder_canonical"] == "canonical_fail"
    assert out["ladder_nomem"] == "memory_not_necessary"
    assert out["ladder_pass"] == "feedback_grounding_pass"
    assert out["episodic_loop_complete"] is True
    assert out["earned_kqv"] is False
    assert out["new_decoder"] is False
    assert out["setup_excluded"] is True
    assert out["candidate_exists"] is False
    assert out["credit_token_present"] is True
    assert out["action_feedback_in_genome"] is False
    assert out["action_feedback_in_recall_modes"] is False
    assert out["permutation"] == "rotate_plus_one"
    assert out["wrapped_state"] == "tm049_observe_last_p1"
    assert out["floor"] == 0.05
    assert "memproj_arm" not in GenomeConfig().to_dict()


DEV_SHA = "87148c7e5fc181d8558e3a80caa23ac282676123b87e1e7ca09f4d196825b571"
DEC_SHA = "404c5401a4ffb66708f8c541593fc7a5dd153ce2cfaa60b30b80b84a817c5443"
DEV_GIT = "10ec197da1efa85c2f4884a5076786fc07742f4b"


def test_dev_lock_heldout_feedback_decode_fail_and_no_v41():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm051fbground import expected_cell_ids

    devp = REPO / "docs" / "lineage_fbground.dev.lock"
    decp = REPO / "docs" / "lineage_fbground.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM050_RUNNER) == TM050_RUNNER_SHA
    assert _sha(TM050_DEV) == TM050_DEV_SHA
    assert _sha(TM050_DEC) == TM050_DEC_SHA
    assert _sha(TM050_ADD) == TM050_ADD_SHA
    assert _sha(TM049_RUNNER) == TM049_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert _sha(NEURAL) == NEURAL_SHA_AFTER_EDIT
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "heldout_feedback_decode_fail"
    assert dev["n_cells"] == 10
    assert dev["n_setup_cells"] == 2
    assert dev["n_scored_cells"] == 8
    assert dev["candidate_v41_lock"] is False
    assert dev["kqv_edited"] is False
    assert dev["new_decoder"] is False
    assert dec["earned_next"] is False
    assert dec["eligible_for_000005"] is False
    assert dec["new_decoder"] is False
    assert dec["decision"]["code"] == "heldout_feedback_decode_fail"
    assert dec["decision"]["phase_flags"]["episodic_loop_complete"] is False
    assert dec["decision"]["phase_flags"]["earned_kqv"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["passed"] is True
    assert cells["decoder|w1"]["passed"] is True
    assert cells["decoder|w0"]["n_ok"] == 4
    assert cells["reference_only|w0"]["cell_code"] == "reference_ok"
    assert cells["reference_only|w1"]["cell_code"] == "reference_ok"
    assert cells["reference_only|w0"]["wrap"]["n_ok_true"] == 1
    assert cells["reference_only|w1"]["wrap"]["n_ok_true"] == 1
    assert cells["feedback_grounded|w0"]["cell_code"] == "heldout_feedback_decode_fail"
    assert cells["feedback_grounded|w1"]["cell_code"] == "heldout_feedback_decode_fail"
    assert cells["feedback_grounded|w0"]["wrap"]["n_ok_true"] == 1
    assert cells["feedback_grounded|w1"]["wrap"]["n_ok_true"] == 1
    assert cells["feedback_grounded|w0"]["wrap"]["n_unique_p1"] == 4
    assert cells["feedback_grounded|w1"]["wrap"]["n_unique_p1"] == 4
    assert cells["shuffled_grounding|w0"]["cell_code"] == "shuffled_ok"
    assert cells["shuffled_grounding|w1"]["cell_code"] == "shuffled_ok"
    assert cells["shuffled_grounding|w0"]["wrap"]["n_ok_true"] == 1
    assert cells["feedback_no_memory|w0"]["cell_code"] == "heldout_feedback_decode_fail"
    assert cells["feedback_no_memory|w1"]["cell_code"] == "heldout_feedback_decode_fail"
