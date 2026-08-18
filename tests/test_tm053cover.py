"""TM053 grounding-coverage freeze tests.

Diagnostic only. No neural edit. Do not install W_star. Do not fit N.
Leave TM046–TM052 runner/DEV/decision/addendum untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import (
    ACT_MARGIN_FLOOR,
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_cover.prereg.lock"
ISO = REPO / "docs" / "lineage_cover.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_cover_contract.md"
RUNNER = REPO / "experiments" / "run_tm053cover.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM049_RUNNER = REPO / "experiments" / "run_tm049actfeed.py"
TM051_RUNNER = REPO / "experiments" / "run_tm051fbground.py"
TM052_RUNNER = REPO / "experiments" / "run_tm052sharefeas.py"
TM052_DEV = REPO / "docs" / "lineage_sharefeas.dev.lock"
TM052_DEC = REPO / "docs" / "lineage_sharefeas.decision.lock"
TM052_PREREG = REPO / "docs" / "lineage_sharefeas.prereg.lock"
TM052_ADD = REPO / "docs" / "lineage_sharefeas.decision.addendum.lock"
MANIFEST = "8fac9a60057217b97327f731b8e959a88a428ba19ccd11fcf12238ec0aecdb29"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
NEURAL_NOW_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM049_RUNNER_SHA = "3def01d5502b28a5ffafeab58b07ee481d5748e5c765b1cbbf52d1c1ed6f275d"
TM051_RUNNER_SHA = "f73c8671db2f3bac6f6b4e22eb08687d933559e476d5c2adb2ff4c879c230706"
TM052_RUNNER_SHA = "36c119262be5a7b2e186b22d3a5e37ffc4e27c4706249156562905f7d025abeb"
TM052_DEV_SHA = "e80ec58901ab456202a6715c74807c1a9b93a34baa546703494b9d90eb55b64a"
TM052_DEC_SHA = "b27ba8f614f41b13b5bdba1eea4468345e8183489318c49960b4c45ef096de5d"
TM052_PREREG_SHA = "56be32c0bfd711498555aac950e16a4ef70fed51dea1c43c1d759976fe9bb812"
TM052_ADD_SHA = "4d388b024e5d9836c8296cf384cfb974557366576c98ed06491b546d4aa6cf43"
RUNNER_SHA = "62e2fe15a3e0565d9041c36cfb16b7fae24d98c8211dec9358a0437770dd2bb4"
ADDENDUM = REPO / "docs" / "lineage_cover.decision.addendum.lock"
ADD_SHA = "ee78a1be25ed1db0b5217120be2cfab959c7cded398f7074014035ee3fe7916c"
LADDER = [
    "setup_precondition_fail",
    "coverage_infeasible",
    "reference_interference",
    "no_transfer",
    "seed_dependent",
    "coverage_generalizes",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm052_wall_untouched():
    assert _sha(TM052_RUNNER) == TM052_RUNNER_SHA
    assert _sha(TM052_DEV) == TM052_DEV_SHA
    assert _sha(TM052_DEC) == TM052_DEC_SHA
    assert _sha(TM052_PREREG) == TM052_PREREG_SHA
    assert _sha(TM052_ADD) == TM052_ADD_SHA
    assert _sha(TM051_RUNNER) == TM051_RUNNER_SHA
    assert _sha(TM049_RUNNER) == TM049_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    dec = json.loads(TM052_DEC.read_text())
    add = json.loads(TM052_ADD.read_text())
    dev = json.loads(TM052_DEV.read_text())
    assert dec["decision"]["code"] == "shared_W_star_satisfies"
    assert add["frozen_first_match_unchanged"] is True
    assert add["install_socp_not_licensed"] is True
    assert add["interpretation"] == "sampled_capacity_without_generalizable_grounding"
    assert add["audit"]["train_only_W_star_hold_w0"] == "1/4"
    assert add["audit"]["train_only_W_star_hold_w1"] == "1/4"
    assert add["audit"]["full_oracle_contains_future_test_information"] is True
    assert dev["install_W_star"] is False
    cells = {c["id"]: c for c in dev["cells"]}
    assert cells["wrapped_train|w0"]["hold_on_Wstar_train"]["n_ok"] == 1
    assert cells["full_oracle|w0"]["applied"] is False
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_coverage_curve():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.53.COVER"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["solver_edit_authorized"] is False
    assert p["install_oracle_authorized"] is False
    assert p["extend_socp_authorized"] is False
    assert p["n_dev_repeats_increase_authorized"] is False
    assert p["n_as_organism_constant_authorized"] is False
    assert p["n_dev_repeats"] == 4
    assert p["n_grid"] == [1, 2, 4, 8, 16, 32]
    assert p["n_train_pool_per_action"] == 32
    assert p["n_hold_contexts_per_action"] == 4
    assert p["n_setup_cells"] == 6
    assert p["n_scored_cells"] == 36
    assert p["expected_n_cells"] == 42
    assert p["held_out_never_in_constraints"] is True
    assert p["install_W_star"] is False
    assert p["n_is_not_organism_constant"] is True
    assert p["w0_source"] == "reference_only_W_act_query"
    assert p["curve_seeds"] == [404910053, 404920053, 404930053]
    assert p["seed_registry"] == 404900053
    assert 404900052 in p["forbidden_seeds"]
    assert 404600046 in p["forbidden_seeds"]
    assert p["tm052_decision_code"] == "shared_W_star_satisfies"
    assert p["tm052_interpretation"] == "sampled_capacity_without_generalizable_grounding"
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "fit_N_as_organism_constant" in p["refuse"]
    assert "install_W_star" in p["refuse"]
    assert "experiments/run_tm052sharefeas.py" in iso["historical_immutable"]
    assert "three_memory/joint_socp.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(NEURAL) == NEURAL_NOW_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_MARGIN_FLOOR == 0.01
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


def test_no_neural_or_solver_edit():
    assert _sha(NEURAL) == NEURAL_NOW_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    src = NEURAL.read_text()
    assert "W_feedback" not in src


def test_ids_and_decision_ladder():
    from experiments.run_tm053cover import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[:6] == [
        "decoder|s0|w0",
        "decoder|s0|w1",
        "decoder|s1|w0",
        "decoder|s1|w1",
        "decoder|s2|w0",
        "decoder|s2|w1",
    ]
    assert ids[6] == "n1|s0|w0"
    assert ids[-1] == "n32|s2|w1"
    assert len(ids) == 42
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["setup_excluded_from_behavioral_first_match"] is True
    assert fl["n_scored_cells"] == 36
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["earned_second_decoder"] is False
        assert flags["install_W_star"] is False
        assert flags["n_fitted_as_organism_constant"] is False
        if step == "coverage_generalizes":
            assert flags["generic_consolidation_plausible"] is True
        if step == "no_transfer":
            assert flags["investigate_representation_invariance"] is True


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm053cover import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "solve_min_change_socp" in src
    assert "held_out_in_constraints" in src
    assert "n_is_organism_constant" in src
    assert "cortex.candidate.v41.lock" in src
    assert "s_cov_" in src
    assert "s_hold_" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "retrieve_by_query" not in names
    assert "_run_joint_socp_consolidation" not in names
    assigns = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assigns.append(t.attr)
    assert "W_act_query" not in assigns
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 42
    assert out["n_setup_cells"] == 6
    assert out["n_scored_cells"] == 36
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_infeasible"] == "coverage_infeasible"
    assert out["ladder_ref"] == "reference_interference"
    assert out["ladder_no_transfer"] == "no_transfer"
    assert out["ladder_seed"] == "seed_dependent"
    assert out["ladder_pass"] == "coverage_generalizes"
    assert out["earned_second_decoder"] is False
    assert out["install_W_star"] is False
    assert out["n_fitted_as_organism_constant"] is False
    assert out["generic_consolidation_plausible"] is True
    assert out["investigate_representation_invariance"] is True
    assert out["candidate_exists"] is False
    assert out["n_grid"] == [1, 2, 4, 8, 16, 32]
    assert "memproj_arm" not in GenomeConfig().to_dict()


DEV_SHA = "de2b615eb2b386b10d4f9aac5346d7b0e44301e34455806b4b27f339daa7e374"
DEC_SHA = "d54dd1be3989d12fc22dc86f31a1b5cf8aa0675bbf200ef2d91a3a32a811565c"
DEV_GIT = "721fc8a7cbd5be8b96012bfbe602b4cbaa63d511"


def test_dev_lock_coverage_generalizes_but_contexts_collapsed():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm053cover import expected_cell_ids

    devp = REPO / "docs" / "lineage_cover.dev.lock"
    decp = REPO / "docs" / "lineage_cover.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM052_RUNNER) == TM052_RUNNER_SHA
    assert _sha(TM052_DEV) == TM052_DEV_SHA
    assert _sha(TM052_ADD) == TM052_ADD_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(NEURAL) == NEURAL_NOW_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "coverage_generalizes"
    assert dev["n_cells"] == 42
    assert dev["install_W_star"] is False
    assert dev["n_fitted_as_organism_constant"] is False
    assert dev["new_decoder"] is False
    assert dec["earned_next"] is False
    assert dec["install_W_star"] is False
    assert dec["n_fitted_as_organism_constant"] is False
    assert dec["decision"]["code"] == "coverage_generalizes"
    assert dec["decision"]["phase_flags"]["generic_consolidation_plausible"] is True
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    for si in range(3):
        for wi in range(2):
            assert cells[f"decoder|s{si}|w{wi}"]["n_ok"] == 4
            top = cells[f"n32|s{si}|w{wi}"]
            assert top["feasible"] is True
            assert top["applied"] is False
            assert top["W_installed"] is False
            assert top["held_out_in_constraints"] is False
            assert top["n_is_organism_constant"] is False
            assert top["hold"]["n_ok"] == 16
            assert top["reference"]["n_ok"] == 4
            train_h = {h["p1_hash"] for h in top["train"]["handles"]}
            hold_h = {h["p1_hash"] for h in top["hold"]["handles"]}
            assert len(train_h) == 4
            assert train_h == hold_h
            assert top["geometry_train"]["within_l2_mean"] == 0.0
    n1 = cells["n1|s0|w0"]
    assert n1["hold"]["n_ok"] == 16
    assert n1["train"]["n_ok"] == 4


def test_addendum_invalidated_measurement_without_rewrite():
    assert _sha(ADDENDUM) == ADD_SHA
    assert _sha(REPO / "docs" / "lineage_cover.dev.lock") == DEV_SHA
    assert _sha(REPO / "docs" / "lineage_cover.decision.lock") == DEC_SHA
    assert _sha(RUNNER) == RUNNER_SHA
    add = json.loads(ADDENDUM.read_text())
    assert add["rewrite_historical_decision"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "coverage_generalizes"
    assert add["interpretation"] == "invalidated_measurement__duplicate_state_support"
    assert add["architectural_conclusion"] == "none"
    assert add["value_need_not_carry_cue_context"] is True
    assert add["action_invariant_value_is_not_a_defect"] is True
    assert add["episode_match_l2_retuned"] is False
    assert add["floor"] == 0.05
    assert add["audit"]["did_not_reproduce_tm052_states"] is True
    assert add["audit"]["do_not_force_context_into_value"] is True
    assert add["audit"]["next_wall"] == "TM.0.54.PROV"
    assert EPISODE_MATCH_L2 == 0.05
    assert add["historical_dev_lock_sha"] == DEV_SHA
    assert add["historical_decision_sha"] == DEC_SHA
    assert add["frozen_runner_sha"] == RUNNER_SHA

