"""TM060 receipt-identity online drift freeze tests.

No neural edit. No SOCP installation. Do not install W_star.
Do not retune 0.05. Do not rerun or edit TM055–TM059.
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
PREREG = REPO / "docs" / "lineage_ondrift.prereg.lock"
ISO = REPO / "docs" / "lineage_ondrift.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_ondrift_contract.md"
RUNNER = REPO / "experiments" / "run_tm060ondrift.py"
TM059_RUNNER = REPO / "experiments" / "run_tm059receipt.py"
TM059_DEV = REPO / "docs" / "lineage_receipt.dev.lock"
TM059_DEC = REPO / "docs" / "lineage_receipt.decision.lock"
TM058_RUNNER = REPO / "experiments" / "run_tm058storeint.py"
TM058_ADD = REPO / "docs" / "lineage_storeint.decision.addendum.lock"
TM057_RUNNER = REPO / "experiments" / "run_tm057dual.py"
TM057_DEC = REPO / "docs" / "lineage_dual.decision.lock"
TM057_ADD = REPO / "docs" / "lineage_dual.decision.addendum.lock"
TM055_RUNNER = REPO / "experiments" / "run_tm055drift.py"
LAW = REPO / "docs" / "lineage_write_time_law.lock"
OPAQUE_LAW = REPO / "docs" / "lineage_opaque_store_law.lock"
COMPAT = REPO / "docs" / "lineage_runner_compat.lock"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
MANIFEST = "6d351d1638d93bb6864a24009baa9151f8c3b969ac6d117892ae8ca3dbe360b5"
NEURAL_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
OPAQUE_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM059_RUNNER_SHA = "8aad0201e391deb5c01d7aca7f50d561a20b3af6d12106a77351b15b3e06229f"
TM059_DEV_SHA = "0eac010b6ca5dffc5f475de32cc14c01b24514695fcc2e85746242e4a703e41e"
TM059_DEC_SHA = "d11ed78e963770ac871b864263e529ffbab1251c014b5743763edef8b66062fc"
TM058_RUNNER_SHA = "2f7e497c216fe4a2fcdd5bbff73ed7f2fb7bd3f43c23a00630b4055ac278bb2d"
TM058_ADD_SHA = "549545f6bf7130cd3790146e5cf1f013de557543440efe93670af8e9003fd93a"
TM057_RUNNER_SHA = "1f1ee4b8d4d2da7893622d8692a91b3912ed7130f9a868dffe02fc19d5cd8f61"
TM057_DEC_SHA = "be06acd8116a356fce06239d89522d0cb7b850ebc97c4b66feb9d6d78fd9ac88"
TM057_ADD_SHA = "501594d36c0ca4fb9e4a163d8be0d624c4d30613ddb89f0de89dba1a63a350e7"
TM055_RUNNER_SHA = "23a3002029560e6a83d5ae5646e5631101fee6b89981e8cd814688e87f9a392b"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
OPAQUE_LAW_SHA = "86893cc7614b1e270fb004028dfde82dc5e06054bc6f5d6ca2aaa6ba82c4260d"
COMPAT_SHA = "a475d3f2bbea6e35832d46b467468cd989eba2df40c2e1a7372ca018ff451f14"
LADDER = [
    "setup_precondition_fail",
    "observer_used_runner_provenance",
    "prefix_infeasible",
    "representation_drift",
    "capacity_eviction_limits_consolidation",
    "generic_grounding_consolidation_earned",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prior_walls_untouched():
    assert _sha(TM059_RUNNER) == TM059_RUNNER_SHA
    assert _sha(TM059_DEV) == TM059_DEV_SHA
    assert _sha(TM059_DEC) == TM059_DEC_SHA
    assert _sha(TM058_RUNNER) == TM058_RUNNER_SHA
    assert _sha(TM058_ADD) == TM058_ADD_SHA
    assert _sha(TM057_RUNNER) == TM057_RUNNER_SHA
    assert _sha(TM057_DEC) == TM057_DEC_SHA
    assert _sha(TM057_ADD) == TM057_ADD_SHA
    assert _sha(TM055_RUNNER) == TM055_RUNNER_SHA
    dec059 = json.loads(TM059_DEC.read_text())
    add058 = json.loads(TM058_ADD.read_text())
    add057 = json.loads(TM057_ADD.read_text())
    assert dec059["decision"]["code"] == "opaque_storage_integrity_holds"
    assert dec059["architectural_conclusion"] == "earned_opaque_storage_integrity"
    assert add058["historical_decision_code"] == "attempted_not_resident"
    assert add058["interpretation"] == "invalidated_measurement__observer_used_runner_provenance"
    assert add058["architectural_conclusion"] == "none"
    assert add057["historical_decision_code"] == "storage_integrity_failure"
    assert add057["interpretation"] == "architectural_falsification__value_cannot_define_record_identity"
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_receipt_identity_drift():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.60.ONDRIFT"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["opaque_memory_edit_authorized"] is False
    assert p["solver_edit_authorized"] is False
    assert p["install_oracle_authorized"] is False
    assert p["dev_authorized_before_runner_compat"] is False
    assert p["attempted_vt_is_offline_diagnostic_ceiling"] is True
    assert p["residents_are_receipt_identified"] is True
    assert p["held_out_are_later_write_time_values"] is True
    assert p["w_star_chronological_prefix_only"] is True
    assert p["ordinary_reference_constraints_retained"] is True
    assert p["discard_every_W_star"] is True
    assert p["arms"] == ["attempts", "residents"]
    assert p["n_grid"] == [4, 8, 16, 24]
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 16
    assert p["expected_n_cells"] == 18
    assert p["n_dev_repeats"] == 4
    assert p["seed_registry"] == 404900060
    assert 404900059 in p["forbidden_seeds"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "experiments/run_tm059receipt.py" in iso["historical_immutable"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert "three_memory/joint_socp.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(LAW) == LAW_SHA
    assert _sha(OPAQUE_LAW) == OPAQUE_LAW_SHA
    assert _sha(COMPAT) == COMPAT_SHA
    assert "opaque_store_enabled" not in GenomeConfig().to_dict()
    assert "opaque_store_enabled" not in ACT_RECALL_MODES
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(RUNNER)


def test_ids_and_decision_ladder():
    from experiments.run_tm060ondrift import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[0] == "decoder|w0"
    assert ids[-1] == "residents|n24|w1"
    assert len(ids) == 18
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["canonical_law_reconsidered"] is False
        assert flags["discard_every_W_star"] is True
        if step == "generic_grounding_consolidation_earned":
            assert flags["generic_consolidation_earned"] is True
        else:
            assert flags["generic_consolidation_earned"] is False


def test_runner_follows_receipts_and_discards_w_star():
    from experiments.run_tm059receipt import receipt_contract_violations
    from experiments.run_tm060ondrift import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "write_opaque_kv" in src
    assert "locate_by_receipt" in src
    assert "later_write_time_values" in src
    assert "discarded" in src
    assert 'provenance_id=f"' not in src
    assert "latest_episode(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    assert receipt_contract_violations(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "_run_joint_socp_consolidation" not in names
    assert hasattr(NeuralCortex, "write_opaque_kv")
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 18
    assert out["receipt_contract"] == []
    assert out["ladder"]["setup"] == "setup_precondition_fail"
    assert out["ladder"]["representation_drift"] == "representation_drift"
    assert out["ladder"]["capacity_eviction_limits_consolidation"] == "capacity_eviction_limits_consolidation"
    assert out["ladder"]["generic_grounding_consolidation_earned"] == "generic_grounding_consolidation_earned"
    assert out["floor"] == 0.05
    assert out["slots"] == 8
    assert out["candidate_exists"] is False
