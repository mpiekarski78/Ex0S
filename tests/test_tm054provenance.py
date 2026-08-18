"""TM054 provenance freeze tests.

Diagnostic only. No neural edit. Do not score SOCP. Do not retune 0.05.
Leave TM046–TM053 runner/DEV/decision/addendum untouched.
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
PREREG = REPO / "docs" / "lineage_provenance.prereg.lock"
ISO = REPO / "docs" / "lineage_provenance.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_provenance_contract.md"
RUNNER = REPO / "experiments" / "run_tm054provenance.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM052_RUNNER = REPO / "experiments" / "run_tm052sharefeas.py"
TM052_DEV = REPO / "docs" / "lineage_sharefeas.dev.lock"
TM053_RUNNER = REPO / "experiments" / "run_tm053cover.py"
TM053_DEV = REPO / "docs" / "lineage_cover.dev.lock"
TM053_DEC = REPO / "docs" / "lineage_cover.decision.lock"
TM053_ADD = REPO / "docs" / "lineage_cover.decision.addendum.lock"
MANIFEST = "b6b4f7096cacce136689097327314f49895f964bcb3fe7d2e6afa74d0032a8d9"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM052_RUNNER_SHA = "36c119262be5a7b2e186b22d3a5e37ffc4e27c4706249156562905f7d025abeb"
TM052_DEV_SHA = "e80ec58901ab456202a6715c74807c1a9b93a34baa546703494b9d90eb55b64a"
TM053_RUNNER_SHA = "62e2fe15a3e0565d9041c36cfb16b7fae24d98c8211dec9358a0437770dd2bb4"
TM053_DEV_SHA = "de2b615eb2b386b10d4f9aac5346d7b0e44301e34455806b4b27f339daa7e374"
TM053_DEC_SHA = "d54dd1be3989d12fc22dc86f31a1b5cf8aa0675bbf200ef2d91a3a32a811565c"
TM053_ADD_SHA = "ee78a1be25ed1db0b5217120be2cfab959c7cded398f7074014035ee3fe7916c"
RUNNER_SHA = "cbfe1c37de4532141d2e770ae03b2d50a32f720276fe6f2064c9a7f7b4ee685d"
LADDER = [
    "setup_precondition_fail",
    "state_generator_mismatch",
    "runner_reset_sequence",
    "action_invariant_values",
    "context_varying_values",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm053_wall_untouched():
    assert _sha(TM053_RUNNER) == TM053_RUNNER_SHA
    assert _sha(TM053_DEV) == TM053_DEV_SHA
    assert _sha(TM053_DEC) == TM053_DEC_SHA
    assert _sha(TM053_ADD) == TM053_ADD_SHA
    assert _sha(TM052_RUNNER) == TM052_RUNNER_SHA
    assert _sha(TM052_DEV) == TM052_DEV_SHA
    dec = json.loads(TM053_DEC.read_text())
    add = json.loads(TM053_ADD.read_text())
    assert dec["decision"]["code"] == "coverage_generalizes"
    assert add["frozen_first_match_unchanged"] is True
    assert add["interpretation"] == "invalidated_measurement__duplicate_state_support"
    assert add["architectural_conclusion"] == "none"
    assert add["value_need_not_carry_cue_context"] is True
    assert add["episode_match_l2_retuned"] is False
    assert add["floor"] == 0.05
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_provenance_gate():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.54.PROV"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["socp_scoring_authorized"] is False
    assert p["episode_match_l2_retune_authorized"] is False
    assert p["force_cue_into_value_authorized"] is False
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 4
    assert p["expected_n_cells"] == 6
    assert p["arms"] == ["boundary", "canonical"]
    assert p["n_dev_repeats"] == 4
    assert p["floor"] == 0.05
    assert p["value_need_not_carry_cue_context"] is True
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["seed_registry"] == 404900054
    assert 404900053 in p["forbidden_seeds"]
    assert p["tm053_decision_code"] == "coverage_generalizes"
    assert p["tm053_interpretation"] == "invalidated_measurement__duplicate_state_support"
    assert len(p["pinned_train_hashes"]["w0"]) == 4
    assert len(p["pinned_hold_hashes"]["w0"]) == 4
    assert p["pinned_train_hashes"]["w0"] != p["pinned_hold_hashes"]["w0"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "force_cue_into_value" in p["refuse"]
    assert "retune_EPISODE_MATCH_L2" in p["refuse"]
    assert "experiments/run_tm053cover.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_no_neural_edit():
    assert _sha(NEURAL) == NEURAL_SHA
    assert "W_feedback" not in NEURAL.read_text()


def test_ids_and_decision_ladder():
    from experiments.run_tm054provenance import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids == [
        "decoder|w0",
        "decoder|w1",
        "boundary|w0",
        "boundary|w1",
        "canonical|w0",
        "canonical|w1",
    ]
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["socp_scored"] is False
        if step == "action_invariant_values":
            assert flags["one_exemplar_grounding_earned"] is True
        if step == "context_varying_values":
            assert flags["genuine_coverage_curve_earned"] is True


def test_runner_refuses_socp_and_smoke():
    from experiments.run_tm054provenance import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "collect_world" in src
    assert "pinned_train_hashes" in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "solve_min_change_socp" not in names
    assert "_run_joint_socp_consolidation" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 6
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_mismatch"] == "state_generator_mismatch"
    assert out["ladder_reset"] == "runner_reset_sequence"
    assert out["ladder_invariant"] == "action_invariant_values"
    assert out["ladder_varying"] == "context_varying_values"
    assert out["floor"] == 0.05
    assert out["candidate_exists"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    assert not hasattr(NeuralCortex, "set_feedback_ticks")


DEV_SHA = "0f16bdbd3a068d788e138df0545ed0aa1e395c6875ce4b5ec365405412a05543"
DEC_SHA = "9ac7a14f6776c04a9bbfe6f4f44418e4f28d830b6b5e49514deba6a34f5f393e"
DEV_GIT = "4e7706ca8e5e6dcd739e368ad6a0538030cba2e4"


def test_dev_lock_state_generator_mismatch_no_socp():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm054provenance import expected_cell_ids

    devp = REPO / "docs" / "lineage_provenance.dev.lock"
    decp = REPO / "docs" / "lineage_provenance.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM053_RUNNER) == TM053_RUNNER_SHA
    assert _sha(TM053_DEV) == TM053_DEV_SHA
    assert _sha(TM053_ADD) == TM053_ADD_SHA
    assert _sha(TM052_DEV) == TM052_DEV_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "state_generator_mismatch"
    assert dev["socp_scored"] is False
    assert dev["install_W_star"] is False
    assert dev["episode_match_l2_retuned"] is False
    assert dec["architectural_conclusion"] == "none"
    assert dec["decision"]["code"] == "state_generator_mismatch"
    assert dec["decision"]["phase_flags"]["genuine_coverage_curve_earned"] is False
    assert dec["decision"]["phase_flags"]["one_exemplar_grounding_earned"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["tm052_train_hash_match"] is True
    assert cells["decoder|w0"]["tm052_hold_hash_match"] is True
    assert cells["canonical|w0"]["n1_hold_matches_pin"] is True
    assert cells["canonical|w0"]["n1_train_matches_pin"] is False
    assert cells["canonical|w1"]["n1_hold_matches_pin"] is True
    assert cells["canonical|w1"]["n1_train_matches_pin"] is False
    assert cells["boundary|w0"]["first_mismatch_site"] == "frozen_dev_probe_vs_write_last_p1"
    assert cells["boundary|w0"]["frozen_wrap_action_invariant"] is True
    assert cells["boundary|w0"]["reset_changes_hold"] is False
    assert cells["boundary|w0"]["scored_socp"] is False


ADD_SHA = "bfa9b1937041afa28c4813b6366751f0e12e0edf9941e9f9585dfe23f5d37fb8"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"


def test_addendum_chooses_write_time_last_p1_without_rewrite():
    addp = REPO / "docs" / "lineage_provenance.decision.addendum.lock"
    lawp = REPO / "docs" / "lineage_write_time_law.lock"
    assert _sha(addp) == ADD_SHA
    assert _sha(lawp) == LAW_SHA
    assert _sha(REPO / "docs" / "lineage_provenance.dev.lock") == DEV_SHA
    assert _sha(REPO / "docs" / "lineage_provenance.decision.lock") == DEC_SHA
    assert _sha(RUNNER) == RUNNER_SHA
    add = json.loads(addp.read_text())
    law = json.loads(lawp.read_text())
    dec = json.loads((REPO / "docs" / "lineage_provenance.decision.lock").read_text())
    assert add["rewrite_historical_decision"] is False
    assert add["rewrite_historical_dev"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "state_generator_mismatch"
    assert add["canonical_state_generator"] == "write_time_last_p1"
    assert add["frozen_wrap_is_not_canonical"] is True
    assert add["architectural_conclusion"] == "none"
    assert add["episode_match_l2_retuned"] is False
    assert law["canonical_state_generator"] == "write_time_last_p1"
    assert law["v_t"] if False else law["law"]["v_t"] == "unit(rho_post_feedback_t)"
    assert law["law"]["captured"] == "at_event_time"
    assert dec["decision"]["code"] == "state_generator_mismatch"
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()


