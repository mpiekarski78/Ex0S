"""TM050 feedback-geometry freeze tests.

Diagnostic only. No neural edit. No fitted tick count. No decoder retrain.
Leave TM046/TM047/TM048/TM049 runner/DEV/decision untouched.
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
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_feedgeom.prereg.lock"
ISO = REPO / "docs" / "lineage_feedgeom.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_feedgeom_contract.md"
RUNNER = REPO / "experiments" / "run_tm050feedgeom.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM047_RUNNER = REPO / "experiments" / "run_tm047reinstate.py"
TM048_RUNNER = REPO / "experiments" / "run_tm048creditinfo.py"
TM049_RUNNER = REPO / "experiments" / "run_tm049actfeed.py"
TM049_DEV = REPO / "docs" / "lineage_actfeed.dev.lock"
TM049_DEC = REPO / "docs" / "lineage_actfeed.decision.lock"
TM049_ADD = REPO / "docs" / "lineage_actfeed.decision.addendum.lock"
MANIFEST = "70005dc9c8c3316af885062a23b95b9811a97f85baf548c9921a22da7fcdf7f3"
NEURAL_SHA = "a33f04479716d21624f9f8d0167ceaf4a658fd57a9070b058933f71fa1ae155c"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM047_RUNNER_SHA = "c5d5a0be88e8704039c8c2e0d8e3fb86de1fc85ec69863129c5f11c26eccc6c4"
TM048_RUNNER_SHA = "57b8d4c6908908e25bd6fedcd561bf60c5a1a3b8a7e5e11e085be5596350a7c9"
TM049_RUNNER_SHA = "3def01d5502b28a5ffafeab58b07ee481d5748e5c765b1cbbf52d1c1ed6f275d"
TM049_DEV_SHA = "5d956e80abef0e41beda251acdd5ae23e1c5eff1c0862fe5f8ae652455d532e0"
TM049_DEC_SHA = "86fca1a366c290ce072efd522b8071fb5771de01e9fbee90e2adda15fda9760d"
TM049_ADD_SHA = "98ed0efd761a7fdd3def78e74599a6c7d1178796419a0fb511ed4062a800bbc6"
RUNNER_SHA = "504925a78645f32576e90e5b734a99dc31171471ae1a5db599a7be53b7452ba1"
LADDER = [
    "setup_precondition_fail",
    "collapse_before_recurrence",
    "input_separates_tanh_compresses",
    "later_ticks_decode",
    "neutral_passes_cue_fails",
    "states_separate_never_decode",
    "geometry_unresolved",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm049_wall_untouched():
    assert _sha(TM049_RUNNER) == TM049_RUNNER_SHA
    assert _sha(TM049_DEV) == TM049_DEV_SHA
    assert _sha(TM049_DEC) == TM049_DEC_SHA
    assert _sha(TM049_ADD) == TM049_ADD_SHA
    assert _sha(TM048_RUNNER) == TM048_RUNNER_SHA
    assert _sha(TM047_RUNNER) == TM047_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    dec = json.loads(TM049_DEC.read_text())
    add = json.loads(TM049_ADD.read_text())
    assert dec["decision"]["code"] == "feedback_not_action_separable"
    assert add["frozen_first_match_unchanged"] is True
    assert add["interpretation"] == "action_information_present_but_weak_and_behaviorally_unreadable"
    assert add["audit"]["geometrically_weak"]["floor_is_diagnostic_boundary_not_proof_of_zero_information"] is True
    assert add["tick_count_fitted"] is False
    assert add["motor_vec_scaled"] is False
    assert add["episode_match_l2_retuned"] is False
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_diagnostic_geometry():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.50.FEEDGEOM"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["kqv_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["tick_count_fit_authorized"] is False
    assert p["action_feedback_edit_authorized"] is False
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 4
    assert p["expected_n_cells"] == 6
    assert p["setup_excluded_from_behavioral_first_match"] is True
    assert p["ticks"] == [0, 1, 2, 4, 8, 16]
    assert p["ticks_are_measurement_grid_not_fitted_hyperparameter"] is True
    assert p["separability_floor"] == 0.05
    assert p["separability_floor_is_EPISODE_MATCH_L2"] is True
    assert p["named_op"] == "_sensory_tick"
    assert p["flag_must_stay_off"] is True
    assert p["side_effect_free"] is True
    assert p["tm049_decision_code"] == "feedback_not_action_separable"
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "fit_tick_count" in p["refuse"]
    assert "scale_motor_vec" in p["refuse"]
    assert "retune_EPISODE_MATCH_L2" in iso["refuse"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert "experiments/run_tm049actfeed.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert "action_feedback" not in ACT_RECALL_MODES
    assert "action_feedback_enabled" not in GenomeConfig().to_dict()
    assert not hasattr(NeuralCortex, "set_feedback_ticks")
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_no_fitted_feedback_matrix():
    src = NEURAL.read_text()
    assert "W_feedback" not in src
    assert "feedback_scale" not in src
    assert not hasattr(NeuralCortex, "set_feedback_ticks")


def test_ids_and_decision_ladder():
    from experiments.run_tm050feedgeom import BEHAVIORAL_LADDER, TRANSITION, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[:2] == ["decoder|w0", "decoder|w1"]
    assert ids[2:] == ["cue|w0", "cue|w1", "neutral|w0", "neutral|w1"]
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    assert TRANSITION["fit_tick_count"] is False
    assert TRANSITION["n_ticks_organism"] == 1
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["setup_excluded_from_behavioral_first_match"] is True
    assert fl["n_scored_cells"] == 4
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        if step != "geometry_unresolved":
            assert flags[step] is True
        assert flags["tick_count_fitted"] is False
        assert flags["earned_kqv"] is False


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm050feedgeom import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "_sensory_tick" in src
    assert "_x_tick" in src
    assert "reset_rho" in src
    assert "cortex.candidate.v41.lock" in src
    assert "fit_tick_count" in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "retrieve_by_query" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 6
    assert out["n_setup_cells"] == 2
    assert out["n_scored_cells"] == 4
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_collapse"] == "collapse_before_recurrence"
    assert out["ladder_tanh"] == "input_separates_tanh_compresses"
    assert out["ladder_later"] == "later_ticks_decode"
    assert out["ladder_neutral"] == "neutral_passes_cue_fails"
    assert out["ladder_misalign"] == "states_separate_never_decode"
    assert out["ladder_unresolved"] == "geometry_unresolved"
    assert out["candidate_exists"] is False
    assert out["neural_edit_authorized"] is False
    assert out["floor"] == 0.05
    assert out["transition"]["ticks_grid"] == [0, 1, 2, 4, 8, 16]
    assert "memproj_arm" not in GenomeConfig().to_dict()


DEV_SHA = "ede465c08e9f5af4da7b17343e30ce111d1d81230c015d3686bb76d4b14ef228"
DEC_SHA = "3acd39eee4f3d10e295178067fd3abce86daa7cb11f09752234039c1e8a927bc"
DEV_GIT = "72a9a4ae9d3bc7a343d87dc59064f0ab8a87012e"


def test_dev_lock_states_separate_never_decode_and_no_v41():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm050feedgeom import expected_cell_ids
    from three_memory.neural_cortex import EPISODE_MATCH_L2

    devp = REPO / "docs" / "lineage_feedgeom.dev.lock"
    decp = REPO / "docs" / "lineage_feedgeom.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM049_RUNNER) == TM049_RUNNER_SHA
    assert _sha(TM049_DEV) == TM049_DEV_SHA
    assert _sha(TM049_DEC) == TM049_DEC_SHA
    assert _sha(TM049_ADD) == TM049_ADD_SHA
    assert _sha(TM048_RUNNER) == TM048_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "states_separate_never_decode"
    assert dev["n_cells"] == 6
    assert dev["n_setup_cells"] == 2
    assert dev["n_scored_cells"] == 4
    assert dev["candidate_v41_lock"] is False
    assert dev["kqv_edited"] is False
    assert dev["tick_count_fitted"] is False
    assert dev["neural_edited"] is False
    assert dec["earned_next"] is False
    assert dec["eligible_for_000005"] is False
    assert dec["tick_count_fitted"] is False
    assert dec["decision"]["code"] == "states_separate_never_decode"
    assert dec["decision"]["phase_flags"]["collapse_before_recurrence"] is False
    assert dec["decision"]["phase_flags"]["input_separates_tanh_compresses"] is False
    assert dec["decision"]["phase_flags"]["later_ticks_decode"] is False
    assert dec["decision"]["phase_flags"]["neutral_passes_cue_fails"] is False
    assert dec["decision"]["phase_flags"]["states_separate_never_decode"] is True
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["passed"] is True
    assert cells["decoder|w1"]["passed"] is True
    t1 = cells["cue|w0"]["ticks"]["1"]
    assert t1["stages"]["motor_vec"]["separable"] is True
    assert t1["stages"]["x_tick"]["separable"] is True
    assert t1["stages"]["w_in_x"]["separable"] is True
    assert t1["stages"]["rho_t"]["separable"] is True
    assert float(t1["stages"]["rho_t"]["max_l2"]) > float(EPISODE_MATCH_L2)
    assert t1["n_ok_decode"] == 1
    assert cells["cue|w0"]["ticks"]["16"]["n_ok_decode"] == 1
    assert cells["neutral|w0"]["cell_code"] == "neutral_no_decode"
    assert cells["neutral|w1"]["cell_code"] == "neutral_no_decode"
    assert cells["cue|w0"]["ticks"]["1"]["handles"][0]["tanh_saturation"]["frac_abs_tanh_gt_sat"] == 0.0
