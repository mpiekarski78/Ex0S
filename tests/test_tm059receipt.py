"""TM059 receipt-following freeze tests.

No neural edit. Do not rerun or edit TM058.
Do not install W_star. Do not retune 0.05.
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
    EPISODE_SLOTS,
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_receipt.prereg.lock"
ISO = REPO / "docs" / "lineage_receipt.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_receipt_contract.md"
RUNNER = REPO / "experiments" / "run_tm059receipt.py"
TM058_RUNNER = REPO / "experiments" / "run_tm058storeint.py"
TM058_DEV = REPO / "docs" / "lineage_storeint.dev.lock"
TM058_DEC = REPO / "docs" / "lineage_storeint.decision.lock"
TM058_ADD = REPO / "docs" / "lineage_storeint.decision.addendum.lock"
LAW = REPO / "docs" / "lineage_opaque_store_law.lock"
LAW_ADD = REPO / "docs" / "lineage_opaque_store_law.addendum.lock"
COMPAT = REPO / "docs" / "lineage_runner_compat.lock"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
MANIFEST = "8aa8a5ba1a2f28865e4310c34a6c3455d27a299f6b524be43c23db5fa159bfe4"
NEURAL_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
OPAQUE_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
LAW_SHA = "86893cc7614b1e270fb004028dfde82dc5e06054bc6f5d6ca2aaa6ba82c4260d"
LAW_ADD_SHA = "18f349737ab6827fc371b1c80aa67b174c7ffb54a90b71051e3bf24db242ed54"
COMPAT_SHA = "a475d3f2bbea6e35832d46b467468cd989eba2df40c2e1a7372ca018ff451f14"
TM058_RUNNER_SHA = "2f7e497c216fe4a2fcdd5bbff73ed7f2fb7bd3f43c23a00630b4055ac278bb2d"
TM058_DEV_SHA = "5f5ea89e0f6fa5981f26eeddcef8fcf8abd4168e60a05465ca769b57d854cf52"
TM058_DEC_SHA = "3501b928d2de16cb181d8003fc79efd9b6b43489e3b38bba8561c872c41a987c"
TM058_ADD_SHA = "549545f6bf7130cd3790146e5cf1f013de557543440efe93670af8e9003fd93a"
LADDER = [
    "setup_precondition_fail",
    "cross_action_refresh",
    "cross_cue_merge",
    "observer_used_runner_provenance",
    "resident_missing_at_receipt",
    "attempted_hash_mismatch",
    "unreported_capacity_loss",
    "checkpoint_not_byte_identical",
    "opaque_storage_integrity_holds",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm058_remains_invalidated_measurement():
    assert _sha(TM058_RUNNER) == TM058_RUNNER_SHA
    assert _sha(TM058_DEV) == TM058_DEV_SHA
    assert _sha(TM058_DEC) == TM058_DEC_SHA
    assert _sha(TM058_ADD) == TM058_ADD_SHA
    dec = json.loads(TM058_DEC.read_text())
    add = json.loads(TM058_ADD.read_text())
    dev = json.loads(TM058_DEV.read_text())
    assert dev["decision_code"] == "attempted_not_resident"
    assert dec["decision"]["code"] == "attempted_not_resident"
    assert dec["architectural_conclusion"] == "none"
    assert add["historical_decision_code"] == "attempted_not_resident"
    assert add["interpretation"] == "invalidated_measurement__observer_used_runner_provenance"
    assert add["architectural_conclusion"] == "none"
    assert add["rewrite_historical_decision"] is False
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True


def test_prereg_pins_receipt_identity():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    compat = json.loads(COMPAT.read_text())
    assert p["lab"] == "TM.0.59.RECEIPT"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["opaque_memory_edit_authorized"] is False
    assert p["dev_authorized_before_runner_compat"] is False
    assert p["tm058_decision_code"] == "attempted_not_resident"
    assert p["tm058_interpretation"] == "invalidated_measurement__observer_used_runner_provenance"
    assert p["tm058_architectural_conclusion"] == "none"
    assert p["write_method"] == "write_opaque_kv"
    assert p["unused_provenance_keyword"] == "runner_keyword_ignored"
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 6
    assert p["expected_n_cells"] == 8
    assert p["n_dev_repeats"] == 4
    assert p["probes"] == ["write", "capacity", "checkpoint"]
    assert p["floor"] == 0.05
    assert p["seed_registry"] == 404900059
    assert 404900058 in p["forbidden_seeds"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "DEV before runner-compat against the law addendum" in p["refuse"]
    assert "experiments/run_tm058storeint.py" in iso["historical_immutable"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(LAW) == LAW_SHA
    assert _sha(LAW_ADD) == LAW_ADD_SHA
    assert _sha(COMPAT) == COMPAT_SHA
    assert compat["rule"]["pre_dev_law_addendum_requires_runner_compat_test"] is True
    assert (
        compat["rule"]["if_addendum_invalidates_unexecuted_frozen_runner"]
        == "supersede_and_refreeze_runner_before_dev"
    )
    assert compat["tm058_interpretation"] == "invalidated_measurement__observer_used_runner_provenance"
    assert "opaque_store_enabled" not in GenomeConfig().to_dict()
    assert "opaque_store_enabled" not in ACT_RECALL_MODES
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(RUNNER)


def test_ids_and_decision_ladder():
    from experiments.run_tm059receipt import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[0] == "flag|w0"
    assert ids[-1] == "checkpoint|w1"
    assert len(ids) == 8
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["canonical_law_reconsidered"] is False
        if step == "opaque_storage_integrity_holds":
            assert flags["opaque_store_law_compatible"] is True
        else:
            assert flags["opaque_store_law_compatible"] is False


def test_runner_follows_receipts_and_tm058_does_not():
    from experiments.run_tm059receipt import receipt_contract_violations, refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "write_opaque_kv" in src
    assert "locate_by_receipt" in src
    assert 'provenance_id=f"' not in src
    assert "latest_episode(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    assert receipt_contract_violations(RUNNER) == []
    tm058_bad = receipt_contract_violations(TM058_RUNNER)
    assert "constructed_provenance_keyword" in tm058_bad
    assert "write_return_discarded" in tm058_bad
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "solve_ceiling" not in names
    assert "_run_joint_socp_consolidation" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 8
    assert out["receipt_contract"] == []
    assert "constructed_provenance_keyword" in out["tm058_receipt_contract"]
    assert out["ladder"]["setup"] == "setup_precondition_fail"
    assert out["ladder"]["observer_used_runner_provenance"] == "observer_used_runner_provenance"
    assert out["ladder"]["opaque_storage_integrity_holds"] == "opaque_storage_integrity_holds"
    assert out["floor"] == 0.05
    assert out["slots"] == 8
    assert out["candidate_exists"] is False
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert hasattr(NeuralCortex, "write_opaque_kv")
    assert not CANDIDATE_V41.exists()
