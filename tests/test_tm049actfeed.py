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
    ELIG_EPS,
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
NEURAL_SHA_AFTER_EDIT = "a33f04479716d21624f9f8d0167ceaf4a658fd57a9070b058933f71fa1ae155c"
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
    assert p["neural_cortex_sha_at_freeze"] == NEURAL_SHA_AT_FREEZE


def test_authorized_neural_edit_matches_freeze():
    import numpy as np

    assert _sha(NEURAL) != NEURAL_SHA_AT_FREEZE
    assert hasattr(NeuralCortex, "set_action_feedback_enabled")
    ag = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
    assert ag._action_feedback_enabled is False
    assert "action_feedback_enabled" not in GenomeConfig().to_dict()
    assert "action_feedback" not in ACT_RECALL_MODES
    ag.set_action_feedback_enabled(True)
    snap = ag.checkpoint()
    assert snap["action_feedback_enabled"] is True
    ag2 = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
    ag2.load_checkpoint(snap)
    assert ag2._action_feedback_enabled is True
    src = NEURAL.read_text()
    assert "new_fitted_matrix" not in src
    assert "W_feedback" not in src
    assert "feedback_scale" not in src


def _obs(ag: NeuralCortex, tag: str, body: list[float], symbols: list[str] | None = None) -> dict:
    return ag.observe(
        {
            "interaction_token": tag,
            "source_token": "src_t",
            "ordered_symbols": list(symbols or ["cue"]),
            "observable_state": ["st_idle"],
            "body_state": list(body),
        }
    )


def test_action_feedback_lifecycle_and_transition():
    import numpy as np

    mid = [0.5, 0.4, 0.5, 0.0]
    better = [1.0, 0.0, 1.0, 0.0]
    worse = [0.0, 1.0, 0.0, 1.0]
    birth = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
    birth.bind_actuators(["h_a", "h_b"])
    _obs(birth, "warm", mid)
    frozen = birth.checkpoint()

    def credit(flag: bool, handle: str, body2: list[float], tag: str):
        ag = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
        ag.load_checkpoint(frozen)
        ag.set_action_feedback_enabled(flag)
        n0 = len(ag._episodes)
        _obs(ag, f"{tag}_sel", mid)
        key_cue = None if ag._last_key_rho is None else np.asarray(ag._last_key_rho).copy()
        p1_cue = None if ag._last_p1 is None else np.asarray(ag._last_p1).copy()
        clamped = ag.clamp_action("ACT", handle)
        assert clamped["ok"] is True
        mv = np.asarray(ag._pending["motor_vec"], dtype=np.float64).copy()
        _obs(ag, f"{tag}_obs", body2)
        p1 = None if ag._last_p1 is None else np.asarray(ag._last_p1).copy()
        return ag, n0, key_cue, p1_cue, mv, p1

    off_a = credit(False, "h_a", better, "offa")
    off_b = credit(False, "h_b", better, "offb")
    assert np.allclose(off_a[5], off_b[5])

    on_a = credit(True, "h_a", better, "ona")
    on_b = credit(True, "h_b", better, "onb")
    assert not np.allclose(on_a[5], on_b[5])
    ag_a = on_a[0]
    assert len(ag_a._episodes) == on_a[1] + 1
    stored = ag_a._episodes[-1]
    assert stored["handle"] == "h_a"
    assert float(stored["adv"]) > ELIG_EPS
    assert np.allclose(stored["p1"], on_a[5])
    assert not np.allclose(stored["p1"], on_a[3])
    assert np.allclose(stored["key_rho"], on_a[2])
    if ag_a._pending is not None and ag_a._pending.get("motor_vec") is not None:
        assert not (
            str(ag_a._pending.get("token")) == "h_a"
            and np.allclose(ag_a._pending["motor_vec"], on_a[4])
            and ag_a._pending.get("key_rho") is not None
            and np.allclose(ag_a._pending["key_rho"], on_a[2])
        )

    closed = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
    closed.load_checkpoint(frozen)
    closed.set_action_feedback_enabled(True)
    n0 = len(closed._episodes)
    _obs(closed, "nopend", better)
    assert len(closed._episodes) == n0

    neg = credit(True, "h_a", worse, "neg")
    assert neg[5] is not None
    assert len(neg[0]._episodes) == neg[1]

    stale = NeuralCortex(None, genome=GenomeConfig(), device="cpu")
    stale.load_checkpoint(frozen)
    stale.set_action_feedback_enabled(True)
    _obs(stale, "stale_sel", mid)
    stale.clamp_action("ACT", "h_a")
    stale._pending["motor_vec"] = np.zeros(stale.genome.d_sym)
    n0 = len(stale._episodes)
    _obs(stale, "stale_obs", better)
    assert len(stale._episodes) == n0


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
    assert out["api_present"] is True
    assert out["transition"]["n_ticks"] == 1
    assert "memproj_arm" not in GenomeConfig().to_dict()


DEV_SHA = "5d956e80abef0e41beda251acdd5ae23e1c5eff1c0862fe5f8ae652455d532e0"
DEC_SHA = "86fca1a366c290ce072efd522b8071fb5771de01e9fbee90e2adda15fda9760d"
DEV_GIT = "30c69d8df99c2246dba5a597cae2356d5c8126e1"


def test_dev_lock_feedback_not_action_separable_and_no_v41():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm049actfeed import expected_cell_ids
    from three_memory.neural_cortex import EPISODE_MATCH_L2

    devp = REPO / "docs" / "lineage_actfeed.dev.lock"
    decp = REPO / "docs" / "lineage_actfeed.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM048_RUNNER) == TM048_RUNNER_SHA
    assert _sha(TM048_DEV) == TM048_DEV_SHA
    assert _sha(TM048_DEC) == TM048_DEC_SHA
    assert _sha(TM048_ADD) == TM048_ADD_SHA
    assert _sha(TM047_RUNNER) == TM047_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "feedback_not_action_separable"
    assert dev["n_cells"] == 8
    assert dev["n_setup_cells"] == 2
    assert dev["n_scored_cells"] == 6
    assert dev["candidate_v41_lock"] is False
    assert dev["kqv_edited"] is False
    assert dec["earned_next"] is False
    assert dec["eligible_for_000005"] is False
    assert dec["earned_learned_addressing"] is False
    assert dec["decision"]["code"] == "feedback_not_action_separable"
    assert dec["decision"]["phase_flags"]["setup_excluded_from_behavioral_first_match"] is True
    assert dec["decision"]["phase_flags"]["earned_kqv"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["passed"] is True
    assert cells["decoder|w1"]["passed"] is True
    assert cells["decoder|w0"]["n_ok"] == 4
    assert cells["scalar_only|w0"]["cell_code"] == "scalar_ok"
    assert cells["scalar_only|w1"]["cell_code"] == "scalar_ok"
    for wi in (0, 1):
        sc = cells[f"scalar_only|w{wi}"]["identity"]
        assert sc["n_unique_rho_feedback"] == 1
        assert float(sc["max_pairwise_l2"]) == 0.0
        assert sc["n_ok_credit"] == 1
        assert sc["n_ok_ceiling"] == 4
        fb = cells[f"action_feedback|w{wi}"]
        assert fb["cell_code"] == "feedback_not_action_separable"
        ident = fb["identity"]
        assert ident["n_unique_rho_feedback"] == 4
        assert float(ident["max_pairwise_l2"]) <= float(EPISODE_MATCH_L2)
        assert ident["distinguishable"] is False
        assert ident["n_ok_credit"] == 1
        assert ident["n_ok_ceiling"] == 4
        assert ident["copied_handle_into_s"] is False
        assert all(bool(cl["key_from_cue"]) for cl in ident["clones"])
        nm = cells[f"feedback_no_memory|w{wi}"]
        assert nm["cell_code"] == "feedback_not_action_separable"


ADD_SHA = "98ed0efd761a7fdd3def78e74599a6c7d1178796419a0fb511ed4062a800bbc6"


def test_audit_addendum_does_not_rewrite_first_match():
    addp = REPO / "docs" / "lineage_actfeed.decision.addendum.lock"
    add = json.loads(addp.read_text())
    assert _sha(addp) == ADD_SHA
    assert add["rewrite_historical_decision"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "feedback_not_action_separable"
    assert add["interpretation"] == "action_information_present_but_weak_and_behaviorally_unreadable"
    assert add["audit"]["action_information_present"]["w0_n_unique_rho_feedback"] == 4
    assert add["audit"]["geometrically_weak"]["floor"] == 0.05
    assert add["audit"]["behaviorally_unreadable"]["n_ok_credit"] == [1, 1]
    assert add["tick_count_fitted"] is False
    assert add["audit"]["next_wall"] == "TM.0.50.FEEDGEOM"
    assert not CANDIDATE_V41.exists()
