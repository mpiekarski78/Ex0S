"""TM043 mechanistic correction freeze tests. No neural or solver edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_mechcorr.prereg.lock"
ISO = REPO / "docs" / "lineage_mechcorr.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_mechcorr_contract.md"
ADDENDUM = REPO / "docs" / "lineage_postinstall.decision.addendum.lock"
RUNNER = REPO / "experiments" / "run_tm043mechcorr.py"
TM042_RUNNER = REPO / "experiments" / "run_tm042postinstall.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANONICAL = REPO / "experiments" / "canonical_act_probe.py"
TM042_DEC = REPO / "docs" / "lineage_postinstall.decision.lock"
TM042_DEV = REPO / "docs" / "lineage_postinstall.dev.lock"
TM042_PREREG = REPO / "docs" / "lineage_postinstall.prereg.lock"
MANIFEST = "12d2939dbb651701a76f391e2cd94168115d2953f8246cde50c22027563abe71"
FROZEN_NEURAL_SHA = "2eb45d8769402330f5ee39a04afffe110a435a0e64a40b12bc2d874b36f5ed59"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
CANONICAL_SHA = "51e24d272417df5ae689301d9600d49aa86daec6608dfc7ff26f8ad4c2e22aef"
TM042_DEC_SHA = "eec6263f4f85e94569eecded557dde6839123ef95ff58973005ce4994d343be8"
TM042_DEV_SHA = "b70481893ee8d8a43163ced9334ed0caa8e3bbe05a33204bca21808a48325488"
TM042_RUNNER_SHA = "56dc11791ec0d7bbb9316e9f3a7c8006b1e1a069542176cd1f909d81234b43fb"
ADDENDUM_SHA = "46d1b5d51e985bab51568adf23333ac6e9b26dde9993b554f426b94f7738c024"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm042_wall_immutable_and_addendum():
    from three_memory.cortex_lineage import sha_file

    assert _sha(TM042_DEC) == TM042_DEC_SHA
    assert _sha(TM042_DEV) == TM042_DEV_SHA
    assert sha_file(TM042_RUNNER) == TM042_RUNNER_SHA
    assert _sha(ADDENDUM) == ADDENDUM_SHA
    dec = json.loads(TM042_DEC.read_text())
    add = json.loads(ADDENDUM.read_text())
    assert dec["decision"]["code"] == "postinstall_mech_install_fail"
    assert dec["decision"]["phase_flags"]["n_cells"] == 12
    assert dec["decision"]["phase_flags"]["candidate_discussion_open"] is False
    assert add["rewrite_historical_decision"] is False
    assert add["rewrite_historical_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "postinstall_mech_install_fail"
    assert add["interpretation"] == "targeting_mistake__preregistered_tm039_regs_2_3_not_diagnostic_reg1"
    assert add["scientifically_valid_organism_failure"] is False
    assert add["candidate_discussion_open_on_frozen_ladder"] is False
    assert add["targeting_mistake"]["diagnostic_reg"] == 1
    assert add["targeting_mistake"]["diagnostic_seed"] == 1584000025
    assert add["preserved_natural_c8h4"]["n_installed"] == 4
    assert add["scientific_pair_already_satisfied_by_natural_c8h4"]["untouched_fallback_activation"] is True


def test_prereg_pins_and_no_edits():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 2
    assert p["tm039_diagnostic_reg"] == 1
    assert p["tm039_registry_seed"] == 1584000025
    assert p["rerun_natural_cells"] is False
    assert p["v40_candidate_authorized"] is False
    assert p["neural_edit_authorized"] is False
    assert p["setup_precondition_fail_is_not_organism_failure"] is True
    assert p["on_pass"]["auto_candidate_lock"] is False
    assert p["on_pass"]["candidate_discussion_open_on_frozen_tm042_ladder"] is False
    assert p["tm042_decision_sha"] == TM042_DEC_SHA
    assert p["tm042_dev_sha"] == TM042_DEV_SHA
    assert p["tm042_addendum_sha"] == ADDENDUM_SHA
    assert p["frozen_neural_sha"] == FROZEN_NEURAL_SHA
    assert p["joint_socp_sha"] == JOINT_SOCP_SHA
    assert sha_file(CANONICAL) == CANONICAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert "auto_cortex.candidate.v40.lock" in iso["refuse"]
    assert "rerun TM042 natural cells" in iso["refuse"]
    assert CONTRACT.is_file()
    frozen = p["frozen_runner_sha"]
    if frozen != "PLACEHOLDER":
        assert frozen == sha_file(RUNNER)


def test_ids_preconditions_and_decision_ladder():
    from experiments.run_tm043mechcorr import _decision, expected_cell_ids, setup_precondition_reasons

    ids = expected_cell_ids()
    assert ids == ["mechcorr|tm039|reg1|A_then_B", "mechcorr|tm039|reg1|B_then_A"]
    pin = json.loads(PREREG.read_text())["pinned_cells"]["A_then_B"]
    assert setup_precondition_reasons(
        violations_after_v37=3,
        fallback_invoked=True,
        solver_installed=True,
        parent_identity_sha=pin["parent_identity_sha"],
        v37_w_hash=pin["v37_w_hash"],
        installed_w_hash=pin["installed_w_hash"],
        pin=pin,
    ) == []
    reasons = setup_precondition_reasons(
        violations_after_v37=0,
        fallback_invoked=False,
        solver_installed=False,
        parent_identity_sha=pin["parent_identity_sha"],
        v37_w_hash=pin["v37_w_hash"],
        installed_w_hash=pin["installed_w_hash"],
        pin=pin,
    )
    assert reasons == ["violations_after_v37", "fallback_invoked", "solver_installed"]
    setup_fail = {
        "kind": "mechanistic_correction",
        "cell_code": "setup_precondition_fail",
        "setup_precondition_ok": False,
        "installed": False,
        "continuity_ok": False,
        "organism_failure": False,
    }
    later_fail = {
        "kind": "mechanistic_correction",
        "cell_code": "mechcorr_later_fail",
        "setup_precondition_ok": True,
        "installed": True,
        "continuity_ok": False,
        "organism_failure": True,
    }
    ok = {
        "kind": "mechanistic_correction",
        "cell_code": "mechcorr_ok",
        "setup_precondition_ok": True,
        "installed": True,
        "continuity_ok": True,
        "organism_failure": False,
    }
    code, _t, fl = _decision([dict(setup_fail), dict(ok)])
    assert code == "setup_precondition_fail"
    assert fl["candidate_v40_lock"] is False
    assert fl["separate_candidate_review_open"] is False
    assert fl["candidate_discussion_open_on_frozen_tm042_ladder"] is False
    code2, _t2, fl2 = _decision([dict(later_fail), dict(ok)])
    assert code2 == "mechcorr_later_fail"
    assert fl2["separate_candidate_review_open"] is False
    code3, _t3, fl3 = _decision([dict(ok), dict(ok)])
    assert code3 == "mechcorr_continuity_pass"
    assert fl3["separate_candidate_review_open"] is True
    assert fl3["candidate_v40_lock"] is False
    assert fl3["candidate_discussion_open_on_frozen_tm042_ladder"] is False


def test_refuse_raw_no_natural_rerun_and_smoke():
    from experiments.run_tm043mechcorr import refuse_eval_natural, refuse_raw_scores, smoke

    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src
    assert ".actuator_scores(" not in src
    assert "cortex.candidate.v40.lock" in src
    assert refuse_raw_scores(RUNNER) == []
    refuse_eval_natural(RUNNER)
    out = smoke()
    assert out["smoke_ok"]
    assert out["setup_ok_reasons"] == []
    assert "violations_after_v37" in out["setup_fail_reasons"]
    assert out["raw_score_leak"] == []
    assert out["hard_budget"] == 16


def test_dev_lock_reg1_continuity_and_no_candidate():
    from three_memory.cortex_lineage import sha_file

    devp = REPO / "docs" / "lineage_mechcorr.dev.lock"
    decp = REPO / "docs" / "lineage_mechcorr.decision.lock"
    revp = REPO / "docs" / "lineage_mechcorr.candidate_review.lock"
    assert _sha(devp) == "12d0536cd4a25645ebcaa867885b250dab8880e35ab91ee80f24c4003df33b99"
    assert _sha(decp) == "bf1740df882d095b37edb98479695521917202c175e6515997c523008f56200d"
    assert _sha(revp) == "b544b27274ef70cec1b9281ff918d65f314eddb2916bcf524487fef837bfab26"
    assert json.loads(PREREG.read_text())["frozen_neural_sha"] == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(TM042_DEC) == TM042_DEC_SHA
    assert _sha(TM042_DEV) == TM042_DEV_SHA
    assert sha_file(RUNNER) == "739c0457c4c2f12b41f34dbf6c8f246c497512e9d402f5e3e2924f87a390e775"
    p = json.loads(PREREG.read_text())
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    rev = json.loads(revp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "ff6d31246c8022601c32a06c6bf1f6f097bdf6fc"
    assert dev["decision_code"] == "mechcorr_continuity_pass"
    assert dev["natural_rerun"] is False
    assert dev["candidate_v40_lock"] is False
    assert dec["candidate_v40_lock"] is False
    assert dec["candidate_discussion_open_on_frozen_tm042_ladder"] is False
    assert dec["separate_candidate_review_open"] is True
    assert rev["not_a_candidate_lock"] is True
    assert rev["candidate_v40_lock"] is False
    flags = dev["phase_flags"]
    assert int(flags["n_setup_ok"]) == 2
    assert int(flags["n_installed"]) == 2
    assert int(flags["n_continuity_ok"]) == 2
    assert flags["separate_candidate_review_open"] is True
    assert flags["candidate_discussion_open_on_frozen_tm042_ladder"] is False
    cells = {c["id"]: c for c in dev["cells"]}
    assert set(cells) == {"mechcorr|tm039|reg1|A_then_B", "mechcorr|tm039|reg1|B_then_A"}
    for _cid, cell in cells.items():
        pin = p["pinned_cells"][cell["order"]]
        proc = cell["process"]
        assert int(proc["violations_after_v37"]) > 0
        assert proc["fallback_invoked"] is True
        assert proc["solver_installed"] is True
        assert cell["setup_precondition_ok"] is True
        assert cell["organism_failure"] is False
        assert cell["cell_code"] == "mechcorr_ok"
        assert proc["parent_identity_sha"] == pin["parent_identity_sha"]
        assert proc["v37_w_hash"] == pin["v37_w_hash"]
        assert proc["installed_w_hash"] == pin["installed_w_hash"]
        assert proc["socp"]["status"] == "optimal"
        assert proc["socp"]["applied"] is True
        assert cell["retain_ok"] is True
        assert cell["new_mapping_ok"] is True
        assert cell["reversal_ok"] is True
        assert cell["subsequent_fallback_atomic"] is True
        assert cell["novelty_unfamiliar"] is True
        assert cell["familiarity_unchanged"] is True
        assert cell["solver_handles_never_in_act"] is True
        assert cell["handoff"]["post_credit_w_hash"]
