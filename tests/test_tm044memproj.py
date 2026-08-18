"""TM044 memory-projection freeze tests.

Authorized neural edit for learned K/Q/V plus opaque S.
No SOCP, recall-mode, R, or half-spacing edits. Product 0.0.004.
Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    GenomeConfig,
    MEMPROJ_ARMS,
    MEMPROJ_LEARNED,
    NeuralCortex,
)
from three_memory.opaque_memory import OpaqueRow, retrieve_by_query

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_memproj.prereg.lock"
ISO = REPO / "docs" / "lineage_memproj.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_memproj_contract.md"
AMEND = REPO / "docs" / "cortex_v41_memproj_architecture_amendment.md"
BOUNDARY = REPO / "docs" / "neural_memory_boundary.lock"
BOUNDARY_MD = REPO / "docs" / "neural_memory_boundary_contract.md"
V40_CLOSE = REPO / "docs" / "cortex.candidate.v40.close.lock"
V40 = REPO / "docs" / "cortex.candidate.v40.lock"
LIVE = REPO / "docs" / "cortex.candidate.lock"
RUNNER = REPO / "experiments" / "run_tm044memproj.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
V40_CLOSE_SHA = "d6f954369d57a3632615a14f1003cb55e985d57995d0ef10a7f89ccd8e0570bc"
BOUNDARY_SHA = "1c62d8b3f9bf0e2217632d5a13f9374512832541fa3f5ee07d8dc39dc8ac14da"
V40_SHA = "dc8c13d1607034781864f1dcfd969ad146bf267fd78fb0ba588a88fe2a0e0319"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
NEURAL_SHA = "20d21f91c275e856ea1ec1faec58e5e1e633c270a79c887afa4b5e41397bb5be"
OPAQUE_SHA = "3f938950f3bb9e7ec96a659538a01e430e2459bd6b1477f417fe43c51a3c85a5"
OPAQUE_NOW_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
LIVE_SHA = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
FROZEN_RUNNER_SHA = "9bbde3eafd7c56ea2a39835405fe78221687a49c01ace0261a41710db7a2cfd0"
MANIFEST = "c0300666815cbe2b7ad761af3acbf813e9d6b31d7efb372397e5e1017b47bb58"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_v40_close_and_boundary_frozen():
    assert _sha(V40_CLOSE) == V40_CLOSE_SHA
    assert _sha(BOUNDARY) == BOUNDARY_SHA
    assert _sha(V40) == V40_SHA
    assert _sha(LIVE) == LIVE_SHA
    close = json.loads(V40_CLOSE.read_text())
    bound = json.loads(BOUNDARY.read_text())
    assert close["candidate_lock_sha"] == V40_SHA
    assert close["next_wall"] == "TM.0.44.MEMPROJ"
    assert close["earned_next"] is False
    assert close["socp_frozen_as"]["atomic_fail_closed"] is True
    assert close["default_arm"] == "off"
    assert "edit joint_socp.py" in close["refuse"]
    assert bound["opaque_s_schema"] == ["key", "value", "when", "provenance_id"]
    assert bound["tm044_does_not_use_kappa"] is True
    assert BOUNDARY_MD.is_file()
    assert AMEND.is_file()


def test_prereg_pins_and_no_socp_recall_edits():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.44.MEMPROJ"
    assert p["product"] == "0.0.004"
    assert p["earned_next"] is False
    assert p["v41_candidate_authorized"] is False
    assert p["neural_edit_authorized"] is True
    assert p["solver_edit_authorized"] is False
    assert p["recall_edit_authorized"] is False
    assert p["n"] == 64
    assert p["n_cues"] == 2
    assert p["n_handles"] == 2
    assert p["n_worlds"] == 2
    assert p["expected_n_cells"] == 17
    assert p["seed_registry"] == 404400044
    assert 22222 in p["forbidden_seeds"]
    assert p["v40_close_sha"] == V40_CLOSE_SHA
    assert p["boundary_sha"] == BOUNDARY_SHA
    assert p["joint_socp_sha"] == JOINT_SOCP_SHA
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert p["opaque_memory_sha"] == OPAQUE_SHA
    assert p["value_source"] == "post_credit_rho"
    assert p["kqv_update_after_write"] is True
    assert p["novelty_familiarity_in_claim"] is False
    assert p["arms"] == [
        "symbolic_oracle",
        "learned_projection",
        "birth_projection",
        "no_persistent_memory",
    ]
    assert [d["code"] for d in p["decision_ladder"]] == [
        "setup_precondition_fail",
        "canonical_path_ownership_fail",
        "donor_basis_mismatch",
        "memory_not_necessary",
        "projection_learning_not_causal",
        "address_not_organism_owned",
        "projection_fail",
        "projection_pass",
    ]
    assert "auto_cortex.candidate.v41.lock" in p["refuse"]
    assert "edit_joint_socp.py" in iso["refuse"]
    assert "auto_cortex.candidate.v41.lock" in iso["refuse"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_NOW_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    frozen = p["frozen_runner_sha"]
    assert frozen == FROZEN_RUNNER_SHA
    assert frozen == sha_file(RUNNER)
    assert p["manifest_sha"] == MANIFEST


def test_ids_and_decision_ladder():
    from experiments.run_tm044memproj import _decision, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 17
    assert ids[0] == "associate|symbolic_oracle|w0"
    assert ids[3] == "associate|no_persistent_memory|w0"
    assert "birth_restore|learned_projection|w0" in ids
    assert "donor|A_to_host|w0" in ids
    oracle = {"id": "associate|symbolic_oracle|w0", "kind": "associate", "passed": False, "cell_code": "associate_fail"}
    learned = {"id": "associate|learned_projection|w0", "kind": "associate", "passed": True, "cell_code": "associate_ok"}
    birth = {"id": "associate|birth_projection|w0", "kind": "associate", "passed": True, "cell_code": "associate_ok"}
    none = {"id": "associate|no_persistent_memory|w0", "kind": "associate", "passed": True, "cell_code": "associate_ok"}
    code, _t, fl = _decision([oracle, learned, birth, none])
    assert code == "setup_precondition_fail"
    assert fl["candidate_v41_lock"] is False
    own = dict(learned, cell_code="canonical_path_ownership_fail", passed=False)
    code2, _, _ = _decision([dict(oracle, passed=True, cell_code="associate_ok"), own, birth, none])
    assert code2 == "canonical_path_ownership_fail"
    none_fail = dict(none, passed=False, cell_code="associate_fail")
    donor_mm = {"id": "donor|A_to_host|w0", "kind": "donor", "passed": False, "cell_code": "donor_basis_mismatch"}
    code3, _, _ = _decision(
        [dict(oracle, passed=True, cell_code="associate_ok"), learned, birth, none_fail, donor_mm]
    )
    assert code3 == "donor_basis_mismatch"
    code4, _, _ = _decision(
        [dict(oracle, passed=True, cell_code="associate_ok"), learned, birth, none]
    )
    assert code4 == "memory_not_necessary"
    learned_r = {"id": "revision|learned_projection|w0", "kind": "revision", "passed": True, "cell_code": "revision_ok"}
    birth_r = {"id": "revision|birth_projection|w0", "kind": "revision", "passed": True, "cell_code": "revision_ok"}
    wipe = {"id": "wipe|learned_projection|w0", "kind": "wipe", "passed": True, "cell_code": "wipe_ok"}
    chk = {"id": "checkpoint|learned_projection|w0", "kind": "checkpoint", "passed": True, "cell_code": "checkpoint_ok"}
    br = {
        "id": "birth_restore|learned_projection|w0",
        "kind": "birth_restore",
        "passed": False,
        "cell_code": "projection_learning_not_causal",
    }
    donor_ok = {"id": "donor|A_to_host|w0", "kind": "donor", "passed": True, "cell_code": "donor_ok"}
    code5, _, fl5 = _decision(
        [
            dict(oracle, passed=True, cell_code="associate_ok"),
            learned,
            birth,
            none_fail,
            learned_r,
            birth_r,
            wipe,
            chk,
            br,
            donor_ok,
        ]
    )
    assert code5 == "projection_learning_not_causal"
    assert fl5["candidate_v41_lock"] is False


def test_runner_refuses_retrieve_and_v41():
    from experiments.run_tm044memproj import refuse_runner_retrieve, smoke

    src = RUNNER.read_text()
    assert "cortex.candidate.v41.lock" in src
    assert "set_act_socp_arm" in src
    assert "early_raw_half_spacing" not in src
    assert refuse_runner_retrieve(RUNNER) == []
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 17
    assert out["retrieve_leak"] == []
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_memory"] == "memory_not_necessary"
    assert out["candidate_exists"] is False
    assert out["memproj_in_genome"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "retrieve_by_query" not in names
    assert "event_memory_scores" in names


def test_opaque_store_hygiene():
    q = np.array([1.0, 0.0], dtype=np.float64)
    a = OpaqueRow(key=np.array([1.0, 0.0]), value=np.array([0.0, 1.0]), when=1, provenance_id="a")
    b = OpaqueRow(key=np.array([0.0, 1.0]), value=np.array([1.0, 0.0]), when=2, provenance_id="b")
    hit = retrieve_by_query(q, [a, b])
    perm = retrieve_by_query(q, [b, a])
    assert hit["hit"] is True
    assert perm["hit"] is True
    np.testing.assert_allclose(hit["value"], perm["value"])
    assert retrieve_by_query(q, [])["reject_reason"] == "empty_store"
    bad = OpaqueRow(key=np.array([1.0]), value=np.array([1.0]), when=3, provenance_id="c")
    assert retrieve_by_query(q, [bad])["reject_reason"] == "dimensional_mismatch"
    nan = OpaqueRow(key=np.array([np.nan, 0.0]), value=np.array([0.0, 1.0]), when=4, provenance_id="d")
    assert retrieve_by_query(q, [nan])["reject_reason"] == "nonfinite_record"
    twin = OpaqueRow(key=np.array([1.0, 0.0]), value=np.array([0.0, -1.0]), when=5, provenance_id="e")
    assert retrieve_by_query(q, [a, twin])["reject_reason"] == "exact_distance_tie"
    infq = np.array([np.inf, 0.0], dtype=np.float64)
    assert retrieve_by_query(infq, [a])["reject_reason"] == "bad_query"


def test_memproj_init_checkpoint_without_harness():
    ag = NeuralCortex()
    assert ag._memproj_arm == "off"
    assert "memproj_arm" not in GenomeConfig().to_dict()
    h0 = ag.memproj_hashes()
    assert h0["W_k"] == h0["W_q"]
    ag.set_memproj_arm(MEMPROJ_LEARNED)
    assert ag._memproj_arm in MEMPROJ_ARMS
    snap = ag.checkpoint()
    twin = NeuralCortex()
    twin.load_checkpoint(snap)
    assert twin._memproj_arm == MEMPROJ_LEARNED
    assert twin.memproj_hashes() == ag.memproj_hashes()
    missing = dict(snap)
    missing.pop("memproj_arm")
    missing.pop("W_k")
    off = NeuralCortex()
    off.load_checkpoint(missing)
    assert off._memproj_arm == "off"
    src = NEURAL.read_text()
    assert "W_k" in src
    assert "event_memory_scores" in src
    assert "early_raw_half_spacing" in src
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(OPAQUE) == OPAQUE_NOW_SHA
    assert not CANDIDATE_V41.exists()


def test_dev_lock_memory_not_necessary_and_no_v41():
    from three_memory.cortex_lineage import sha_file

    devp = REPO / "docs" / "lineage_memproj.dev.lock"
    decp = REPO / "docs" / "lineage_memproj.decision.lock"
    assert _sha(devp) == "e375a4ae9e19f1697dddc8d1055bd34ead6f667c92db575ed3e6512be4a6fc8e"
    assert _sha(decp) == "bf3fa56665dfad02657307879a2491e3d1315ecc84024f52e51b782bf0d12efb"
    assert json.loads(PREREG.read_text())["neural_cortex_sha"] == NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(RUNNER) == FROZEN_RUNNER_SHA
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "580588f22b21b0cd457082bc17d4b6a0ed471ed8"
    assert dev["decision_code"] == "memory_not_necessary"
    assert dev["n_cells"] == 17
    assert dev["candidate_v41_lock"] is False
    assert dev["solver_edited"] is False
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dec["candidate_v41_lock"] is False
    assert dec["earned_next"] is False
    assert dec["decision"]["code"] == "memory_not_necessary"
    assert dec["decision"]["phase_flags"]["separate_candidate_review_open"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    from experiments.run_tm044memproj import expected_cell_ids

    cells = {c["id"]: c for c in dev["cells"]}
    assert set(cells) == set(expected_cell_ids())
    for wi in (0, 1):
        assert cells[f"associate|symbolic_oracle|w{wi}"]["passed"] is True
        assert cells[f"associate|no_persistent_memory|w{wi}"]["passed"] is True
        assert cells[f"associate|learned_projection|w{wi}"]["passed"] is False
        assert cells[f"associate|birth_projection|w{wi}"]["passed"] is False
    assert cells["donor|A_to_host|w0"]["cell_code"] == "address_not_organism_owned"
    assert cells["donor|B_to_host|w0"]["cell_code"] == "address_not_organism_owned"
    assert cells["wipe|learned_projection|w0"]["cell_code"] == "memory_not_necessary"


def test_audit_addendum_does_not_rewrite_first_match():
    addp = REPO / "docs" / "lineage_memproj.decision.addendum.lock"
    add = json.loads(addp.read_text())
    assert _sha(addp) == "3b28a031798024f443df88bbc01e10fa8eca30d36ff701b39e69f049a4bc06d1"
    assert add["rewrite_historical_decision"] is False
    assert add["rewrite_historical_dev"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "memory_not_necessary"
    assert add["candidate_v41_lock"] is False
    assert add["scientifically_valid_organism_result"] is True
    assert add["audit"]["donor_protocol_gap"]["determined_first_match"] is False
    assert add["historical_decision_sha"] == "bf3fa56665dfad02657307879a2491e3d1315ecc84024f52e51b782bf0d12efb"
    assert add["historical_dev_lock_sha"] == "e375a4ae9e19f1697dddc8d1055bd34ead6f667c92db575ed3e6512be4a6fc8e"
    assert add["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert not CANDIDATE_V41.exists()
