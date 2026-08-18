"""TM052 shared-decoder feasibility freeze tests.

Diagnostic only. No neural edit. Do not install W_star.
Leave TM046–TM051 runner/DEV/decision untouched.
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
PREREG = REPO / "docs" / "lineage_sharefeas.prereg.lock"
ISO = REPO / "docs" / "lineage_sharefeas.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_sharefeas_contract.md"
RUNNER = REPO / "experiments" / "run_tm052sharefeas.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM049_RUNNER = REPO / "experiments" / "run_tm049actfeed.py"
TM050_RUNNER = REPO / "experiments" / "run_tm050feedgeom.py"
TM051_RUNNER = REPO / "experiments" / "run_tm051fbground.py"
TM051_DEV = REPO / "docs" / "lineage_fbground.dev.lock"
TM051_DEC = REPO / "docs" / "lineage_fbground.decision.lock"
TM051_PREREG = REPO / "docs" / "lineage_fbground.prereg.lock"
MANIFEST = "7fc10e580d2f488bd60233a2d4e91ff4bf87946113da3fcea8c7c2f496cb05c1"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
NEURAL_NOW_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM049_RUNNER_SHA = "3def01d5502b28a5ffafeab58b07ee481d5748e5c765b1cbbf52d1c1ed6f275d"
TM050_RUNNER_SHA = "504925a78645f32576e90e5b734a99dc31171471ae1a5db599a7be53b7452ba1"
TM051_RUNNER_SHA = "f73c8671db2f3bac6f6b4e22eb08687d933559e476d5c2adb2ff4c879c230706"
TM051_DEV_SHA = "87148c7e5fc181d8558e3a80caa23ac282676123b87e1e7ca09f4d196825b571"
TM051_DEC_SHA = "404c5401a4ffb66708f8c541593fc7a5dd153ce2cfaa60b30b80b84a817c5443"
TM051_PREREG_SHA = "70d62ecb25682c9b97a93583d686796bb202ac3022a221e02de590551975d652"
RUNNER_SHA = "36c119262be5a7b2e186b22d3a5e37ffc4e27c4706249156562905f7d025abeb"
ADDENDUM = REPO / "docs" / "lineage_sharefeas.decision.addendum.lock"
ADD_SHA = "4d388b024e5d9836c8296cf384cfb974557366576c98ed06491b546d4aa6cf43"
LADDER = [
    "setup_precondition_fail",
    "training_infeasible",
    "reference_feedback_conflict",
    "context_entangled",
    "shared_W_star_satisfies",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm051_wall_untouched():
    assert _sha(TM051_RUNNER) == TM051_RUNNER_SHA
    assert _sha(TM051_DEV) == TM051_DEV_SHA
    assert _sha(TM051_DEC) == TM051_DEC_SHA
    assert _sha(TM051_PREREG) == TM051_PREREG_SHA
    assert _sha(TM050_RUNNER) == TM050_RUNNER_SHA
    assert _sha(TM049_RUNNER) == TM049_RUNNER_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    dec = json.loads(TM051_DEC.read_text())
    dev = json.loads(TM051_DEV.read_text())
    assert dec["decision"]["code"] == "heldout_feedback_decode_fail"
    assert dev["decision_code"] == "heldout_feedback_decode_fail"
    cells = {c["id"]: c for c in dev["cells"]}
    assert cells["decoder|w0"]["n_ok"] == 4
    assert cells["decoder|w1"]["n_ok"] == 4
    assert cells["feedback_grounded|w0"]["wrap"]["n_ok_true"] == 1
    assert cells["feedback_grounded|w1"]["wrap"]["n_ok_true"] == 1
    assert "tm051_train_decode" not in cells["feedback_grounded|w0"]
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_side_effect_free_feasibility():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.52.SHAREFEAS"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["kqv_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["solver_edit_authorized"] is False
    assert p["install_oracle_authorized"] is False
    assert p["extend_socp_authorized"] is False
    assert p["n_dev_repeats_increase_authorized"] is False
    assert p["n_dev_repeats"] == 4
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 6
    assert p["expected_n_cells"] == 8
    assert p["setup_excluded_from_behavioral_first_match"] is True
    assert p["side_effect_free"] is True
    assert p["install_W_star"] is False
    assert p["ceilings"] == ["wrapped_train", "train_ref", "full_oracle"]
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["seed_registry"] == 404900052
    assert 404900051 in p["forbidden_seeds"]
    assert p["tm051_decision_code"] == "heldout_feedback_decode_fail"
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "install_W_star" in p["refuse"]
    assert "extend_joint_socp" in p["refuse"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert "three_memory/joint_socp.py" in iso["historical_immutable"]
    assert "experiments/run_tm051fbground.py" in iso["historical_immutable"]
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
    assert "W_star" not in SOLVER.read_text()


def test_ids_and_decision_ladder():
    from experiments.run_tm052sharefeas import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[:2] == ["decoder|w0", "decoder|w1"]
    assert ids[2:] == [
        "wrapped_train|w0",
        "wrapped_train|w1",
        "train_ref|w0",
        "train_ref|w1",
        "full_oracle|w0",
        "full_oracle|w1",
    ]
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["setup_excluded_from_behavioral_first_match"] is True
    assert fl["n_scored_cells"] == 6
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["earned_second_decoder"] is False
        assert flags["install_W_star"] is False
        assert flags["new_decoder"] is False
        if step == "shared_W_star_satisfies":
            assert flags["investigate_generic_consolidation"] is True


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm052sharefeas import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "solve_min_change_socp" in src
    assert "develop_protocol" in src
    assert "applied" in src
    assert "W_installed" in src
    assert "cortex.candidate.v41.lock" in src
    assert "n_dev_repeats" in src
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
    assert out["n_cells"] == 8
    assert out["n_setup_cells"] == 2
    assert out["n_scored_cells"] == 6
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_train"] == "training_infeasible"
    assert out["ladder_conflict"] == "reference_feedback_conflict"
    assert out["ladder_entangled"] == "context_entangled"
    assert out["ladder_pass"] == "shared_W_star_satisfies"
    assert out["earned_second_decoder"] is False
    assert out["install_W_star"] is False
    assert out["investigate_generic_consolidation"] is True
    assert out["candidate_exists"] is False
    assert out["tau"] == 0.01
    assert "memproj_arm" not in GenomeConfig().to_dict()


DEV_SHA = "e80ec58901ab456202a6715c74807c1a9b93a34baa546703494b9d90eb55b64a"
DEC_SHA = "b27ba8f614f41b13b5bdba1eea4468345e8183489318c49960b4c45ef096de5d"
DEV_GIT = "f9f267ae6a54889a3fadae25004476dc2f495b0e"


def test_dev_lock_shared_W_star_satisfies_and_no_install():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm052sharefeas import expected_cell_ids

    devp = REPO / "docs" / "lineage_sharefeas.dev.lock"
    decp = REPO / "docs" / "lineage_sharefeas.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM051_RUNNER) == TM051_RUNNER_SHA
    assert _sha(TM051_DEV) == TM051_DEV_SHA
    assert _sha(TM051_DEC) == TM051_DEC_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(NEURAL) == NEURAL_NOW_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "shared_W_star_satisfies"
    assert dev["n_cells"] == 8
    assert dev["n_setup_cells"] == 2
    assert dev["n_scored_cells"] == 6
    assert dev["install_W_star"] is False
    assert dev["new_decoder"] is False
    assert dev["candidate_v41_lock"] is False
    assert dev["neural_untouched"] is True
    assert dec["earned_next"] is False
    assert dec["eligible_for_000005"] is False
    assert dec["install_W_star"] is False
    assert dec["new_decoder"] is False
    assert dec["decision"]["code"] == "shared_W_star_satisfies"
    assert dec["decision"]["phase_flags"]["investigate_generic_consolidation"] is True
    assert dec["decision"]["phase_flags"]["earned_second_decoder"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["n_ok"] == 4
    assert cells["decoder|w1"]["n_ok"] == 4
    assert cells["wrapped_train|w0"]["cell_code"] == "train_feasible"
    assert cells["wrapped_train|w1"]["cell_code"] == "train_feasible"
    assert cells["train_ref|w0"]["cell_code"] == "train_ref_feasible"
    assert cells["full_oracle|w0"]["cell_code"] == "full_feasible"
    assert cells["full_oracle|w1"]["cell_code"] == "full_feasible"
    assert cells["full_oracle|w0"]["predicate"]["n_ok"] == 12
    assert cells["full_oracle|w1"]["predicate"]["n_ok"] == 12
    assert cells["full_oracle|w0"]["applied"] is False
    assert cells["full_oracle|w0"]["W_installed"] is False
    assert cells["wrapped_train|w0"]["tm051_train_decode"]["n_ok"] == 1
    assert cells["wrapped_train|w1"]["tm051_train_decode"]["n_ok"] == 4
    assert cells["wrapped_train|w0"]["hold_on_Wstar_train"]["n_ok"] == 1
    assert cells["wrapped_train|w1"]["hold_on_Wstar_train"]["n_ok"] == 1
    hold0 = [h["p1_hash"] for h in cells["full_oracle|w0"]["predicate"]["handles"][-4:]]
    assert hold0 == [
        "843d1549a3f6a6190adeabf1883acf1d51ad64813b4803a85f737438122acb80",
        "6095d4147b59706e954c1f83188d24679b1c0b07a69ab6e687a178d1563d0527",
        "4533bafecb2c96c543c40acde6ea5d5f1559fb2a7b3703b5ea6455ff5f72f471",
        "191ddeaa64b0129170154c45d97cf4271e3d679606745822eed947731d63145a",
    ]


def test_addendum_narrow_interpretation_without_rewrite():
    assert ADD_SHA != "PLACEHOLDER"
    assert _sha(ADDENDUM) == ADD_SHA
    assert _sha(REPO / "docs" / "lineage_sharefeas.dev.lock") == DEV_SHA
    assert _sha(REPO / "docs" / "lineage_sharefeas.decision.lock") == DEC_SHA
    assert _sha(RUNNER) == RUNNER_SHA
    add = json.loads(ADDENDUM.read_text())
    assert add["rewrite_historical_decision"] is False
    assert add["rewrite_historical_dev"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "shared_W_star_satisfies"
    assert add["install_socp_not_licensed"] is True
    assert add["install_W_star"] is False
    assert add["extend_socp"] is False
    assert add["new_decoder"] is False
    assert add["interpretation"] == "sampled_capacity_without_generalizable_grounding"
    assert add["audit"]["no_second_decoder_required_by_sampled_capacity"] is True
    assert add["audit"]["local_plasticity_insufficient"] is True
    assert add["audit"]["observed_grounding_set_insufficient_for_generalization"] is True
    assert add["audit"]["full_oracle_contains_future_test_information"] is True
    assert add["audit"]["full_oracle_weights_diagnostic_only"] is True
    assert add["audit"]["train_only_W_star_hold_w0"] == "1/4"
    assert add["audit"]["train_only_W_star_hold_w1"] == "1/4"
    assert add["audit"]["next_wall"] == "TM.0.53.COVER"
    assert add["historical_dev_lock_sha"] == DEV_SHA
    assert add["historical_decision_sha"] == DEC_SHA
    assert add["frozen_runner_sha"] == RUNNER_SHA

