"""TM047 reinstatement-split freeze tests.

No K/Q/V. No new decoder. Leave TM046 runner/DEV/decision untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2, GenomeConfig

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_reinstate.prereg.lock"
ISO = REPO / "docs" / "lineage_reinstate.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_reinstate_contract.md"
RUNNER = REPO / "experiments" / "run_tm047reinstate.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM046_RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
TM046_DEV = REPO / "docs" / "lineage_oneshot.dev.lock"
TM046_DEC = REPO / "docs" / "lineage_oneshot.decision.lock"
TM046_ADD = REPO / "docs" / "lineage_oneshot.decision.addendum.lock"
MANIFEST = "e06ea0bc9ee5aae6dfe03438e5569f20fa890fdc010ef366917ca414e2b91969"
NEURAL_SHA = "b0785af069c79c62bd3972a0a3f03f53f9bfbb7221accfb76061b6ee52bb0f1c"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM046_RUNNER_SHA = "8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0"
TM046_DEV_SHA = "68088c1728e9c3367c5cd30bd88b7adc8df30502afa7db3d5a1546b13fa6110d"
TM046_DEC_SHA = "da0e4e82cbaca107543029af16cd0bfe5cfc6027b457c2798b5c34134ec24323"
TM046_ADD_SHA = "8afcc27a9919baaf4052323b46b639129961848c4a7a30a6c9bfa920d0b6f337"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm046_wall_untouched():
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    assert _sha(TM046_DEV) == TM046_DEV_SHA
    assert _sha(TM046_DEC) == TM046_DEC_SHA
    assert _sha(TM046_ADD) == TM046_ADD_SHA
    dec = json.loads(TM046_DEC.read_text())
    add = json.loads(TM046_ADD.read_text())
    assert dec["decision"]["code"] == "generic_reinstatement_fail"
    assert add["frozen_first_match_unchanged"] is True
    assert add["rerun_dev"] is False
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_and_no_kqv():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.47.REINSTATE"
    assert p["product"] == "0.0.004"
    assert p["kqv_edit_authorized"] is False
    assert p["decoder_edit_authorized"] is False
    assert p["v41_candidate_authorized"] is False
    assert p["n"] == 64
    assert p["n_facts"] == 4
    assert p["expected_n_cells"] == 6
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["reconstruction_domain"] == "TM046.ONESHOT.DEV."
    assert p["tm046_decision_code"] == "generic_reinstatement_fail"
    assert p["tm046_runner_sha"] == TM046_RUNNER_SHA
    assert p["tm046_addendum_sha"] == TM046_ADD_SHA
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert p["boundaries"] == [
        "development_reference",
        "credit_full_rho",
        "stored_value_direct",
        "post_reinstatement",
        "canonical_path",
    ]
    assert [d["code"] for d in p["decision_ladder"]][0:3] == [
        "setup_precondition_fail",
        "development_reference_fail",
        "credit_rho_fail",
    ]
    assert "tune_kqv" in p["refuse"]
    assert "train_new_decoder" in p["refuse"]
    assert "experiments/run_tm046oneshot.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(RUNNER)


def test_ids_and_decision_ladder():
    from experiments.run_tm047reinstate import _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids == [
        "decoder|w0",
        "decoder|w1",
        "split|symbolic_oracle|w0",
        "split|symbolic_oracle|w1",
        "split|no_persistent_memory|w0",
        "split|no_persistent_memory|w1",
    ]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    code2, _, fl2 = _decision(synthetic_grid(first_fail="credit_full_rho"))
    assert code2 == "credit_rho_fail"
    assert fl2["first_failing_boundary"] == "credit_full_rho"
    assert fl2["earned_kqv"] is False
    code3, _, fl3 = _decision(synthetic_grid(first_fail="canonical_path"))
    assert code3 == "reinstatement_interface_earned"
    assert fl3["earned_interface"] is True
    code4, _, _ = _decision(synthetic_grid(first_fail="development_reference"))
    assert code4 == "development_reference_fail"


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm047reinstate import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "freeze_plasticity" in src
    assert "event_memory_scores" in src
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
    assert out["n_cells"] == 6
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_credit"] == "credit_rho_fail"
    assert out["ladder_earned"] == "reinstatement_interface_earned"
    assert out["candidate_exists"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()


DEV_SHA = "5c3c7a8172f90f1265fedb1cb5e9659840f5a528932fbf34045cfc0573e4aa53"
DEC_SHA = "025419faab0e67fd1342ae8670d752b582ad34180d00467a368941a82cf24ef9"
DEV_GIT = "acac6de2a932fce4ac3da6f23beb7e590ab3f3ca"
RUNNER_SHA = "c5d5a0be88e8704039c8c2e0d8e3fb86de1fc85ec69863129c5f11c26eccc6c4"


def test_dev_lock_credit_rho_fail_and_no_v41():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm047reinstate import expected_cell_ids

    devp = REPO / "docs" / "lineage_reinstate.dev.lock"
    decp = REPO / "docs" / "lineage_reinstate.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM046_RUNNER) == TM046_RUNNER_SHA
    assert _sha(TM046_DEV) == TM046_DEV_SHA
    assert _sha(TM046_DEC) == TM046_DEC_SHA
    assert _sha(TM046_ADD) == TM046_ADD_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "credit_rho_fail"
    assert dev["n_cells"] == 6
    assert dev["candidate_v41_lock"] is False
    assert dev["kqv_edited"] is False
    assert dev["decoder_retrained"] is False
    assert dec["earned_next"] is False
    assert dec["eligible_for_000005"] is False
    assert dec["decision"]["code"] == "credit_rho_fail"
    assert dec["decision"]["phase_flags"]["first_failing_boundary"] == "credit_full_rho"
    assert dec["decision"]["phase_flags"]["earned_interface"] is False
    assert dec["decision"]["phase_flags"]["earned_kqv"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert set(cells) == set(expected_cell_ids())
    assert cells["decoder|w0"]["passed"] is True
    assert cells["decoder|w1"]["passed"] is True
    assert cells["decoder|w0"]["n_ok"] == 4
    assert cells["decoder|w1"]["n_ok"] == 4
    for wi in (0, 1):
        oracle = cells[f"split|symbolic_oracle|w{wi}"]
        none = cells[f"split|no_persistent_memory|w{wi}"]
        assert oracle["passed"] is False
        assert none["passed"] is False
        assert oracle["first_failing_boundary"] == "credit_full_rho"
        assert none["first_failing_boundary"] == "credit_full_rho"
        assert oracle["n_ok_by_boundary"] == {
            "development_reference": 4,
            "credit_full_rho": 1,
            "stored_value_direct": 1,
            "post_reinstatement": 1,
            "canonical_path": 1,
        }
        assert oracle["w_act_query_frozen"] is True
        slots = [int(f["boundaries"]["canonical_path"]["retrieved_slot"]) for f in oracle["facts"]]
        assert slots == [0, 1, 2, 3]
        assert all(bool(f["boundaries"]["canonical_path"]["intended_record"]) for f in oracle["facts"])
        addrs = [f["boundaries"]["canonical_path"]["addr_hash"] for f in oracle["facts"]]
        assert len(set(addrs)) == 4
        stored = [f["boundaries"]["canonical_path"]["stored_p1_hash"] for f in oracle["facts"]]
        retrieved = [f["boundaries"]["canonical_path"]["retrieved_p1_hash"] for f in oracle["facts"]]
        assert stored == retrieved
        for f in oracle["facts"]:
            rel = f["boundaries"]["post_reinstatement"]["rel_to_stored"]
            assert rel["same_hash"] is True
            assert rel["l2"] == 0.0
        winners = {f["boundaries"]["canonical_path"]["winner"] for f in oracle["facts"]}
        assert len(winners) == 1
        assert all(not bool(f["boundaries"]["canonical_path"]["intended_record"]) for f in none["facts"])
