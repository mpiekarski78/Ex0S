"""TM058 opaque-store integrity freeze tests.

Diagnostic only. Do not implement the write path in this freeze.
Do not install W_star. Do not retune 0.05.
Leave TM046–TM057 runner/DEV/decision/addendum untouched.
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
PREREG = REPO / "docs" / "lineage_storeint.prereg.lock"
ISO = REPO / "docs" / "lineage_storeint.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_storeint_contract.md"
RUNNER = REPO / "experiments" / "run_tm058storeint.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
LAW = REPO / "docs" / "lineage_opaque_store_law.lock"
WRITE_TIME = REPO / "docs" / "lineage_write_time_law.lock"
TM057_RUNNER = REPO / "experiments" / "run_tm057dual.py"
TM057_DEV = REPO / "docs" / "lineage_dual.dev.lock"
TM057_DEC = REPO / "docs" / "lineage_dual.decision.lock"
TM057_ADD = REPO / "docs" / "lineage_dual.decision.addendum.lock"
MANIFEST = "0a77945d8d675a3cd8ae5f2bc12b3dff30d9311bf3d946eb2e62410a07a1eac2"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
OPAQUE_SHA = "3f938950f3bb9e7ec96a659538a01e430e2459bd6b1477f417fe43c51a3c85a5"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM057_RUNNER_SHA = "1f1ee4b8d4d2da7893622d8692a91b3912ed7130f9a868dffe02fc19d5cd8f61"
TM057_DEV_SHA = "2f7649e1e7214fe93c8a34fb174d7c4c8e87a1da6cd78d57f6b963b1b7f650e0"
TM057_DEC_SHA = "be06acd8116a356fce06239d89522d0cb7b850ebc97c4b66feb9d6d78fd9ac88"
TM057_ADD_SHA = "501594d36c0ca4fb9e4a163d8be0d624c4d30613ddb89f0de89dba1a63a350e7"
LAW_SHA = "86893cc7614b1e270fb004028dfde82dc5e06054bc6f5d6ca2aaa6ba82c4260d"
WRITE_TIME_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
RUNNER_SHA = "2f7e497c216fe4a2fcdd5bbff73ed7f2fb7bd3f43c23a00630b4055ac278bb2d"
LADDER = [
    "setup_precondition_fail",
    "cross_action_refresh",
    "cross_cue_merge",
    "attempted_not_resident",
    "unreported_capacity_loss",
    "checkpoint_not_byte_identical",
    "storage_integrity_holds",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm057_wall_untouched():
    assert _sha(TM057_RUNNER) == TM057_RUNNER_SHA
    assert _sha(TM057_DEV) == TM057_DEV_SHA
    assert _sha(TM057_DEC) == TM057_DEC_SHA
    assert _sha(TM057_ADD) == TM057_ADD_SHA
    assert _sha(WRITE_TIME) == WRITE_TIME_SHA
    dec = json.loads(TM057_DEC.read_text())
    add = json.loads(TM057_ADD.read_text())
    assert dec["decision"]["code"] == "storage_integrity_failure"
    assert add["interpretation"] == "architectural_falsification__value_cannot_define_record_identity"
    assert add["architectural_conclusion"] == "earned_separate_opaque_store_law"
    assert add["rewrite_historical_decision"] is False
    assert add["episode_match_l2_not_changed"] is True
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_storage_identity():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    law = json.loads(LAW.read_text())
    assert p["lab"] == "TM.0.58.STOREINT"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["opaque_memory_edit_authorized"] is False
    assert p["dev_authorized_before_implementation"] is False
    assert p["value_is_not_record_identity"] is True
    assert p["flag_name"] == "opaque_store_enabled"
    assert p["write_method"] == "write_opaque_kv"
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 10
    assert p["expected_n_cells"] == 12
    assert p["n_dev_repeats"] == 4
    assert p["probes"] == ["near_action", "cue_key", "repeat", "capacity", "checkpoint"]
    assert p["floor"] == 0.05
    assert p["seed_registry"] == 404900058
    assert 404900057 in p["forbidden_seeds"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "DEV before write_opaque_kv exists" in p["refuse"]
    assert "experiments/run_tm057dual.py" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(LAW) == LAW_SHA
    assert law["implementation_in_this_freeze"] is False
    assert law["implementation_earned"] is True
    assert law["value_is_not_record_identity"] is True
    assert law["historical_episode_storage"]["episode_match_l2"] == 0.05
    assert law["flag"]["default"] is False
    assert law["flag"]["genome_field"] is False
    assert law["flag"]["act_recall_mode"] is False
    assert "opaque_store_enabled" not in GenomeConfig().to_dict()
    assert "opaque_store_enabled" not in ACT_RECALL_MODES
    assert not hasattr(NeuralCortex, "set_opaque_store_enabled")
    assert not hasattr(NeuralCortex, "write_opaque_kv")
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == RUNNER_SHA
    assert frozen == sha_file(RUNNER)


def test_no_neural_edit():
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_SHA


def test_ids_and_decision_ladder():
    from experiments.run_tm058storeint import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[0] == "flag|w0"
    assert ids[-1] == "checkpoint|w1"
    assert len(ids) == 12
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["canonical_law_reconsidered"] is False
        if step == "storage_integrity_holds":
            assert flags["opaque_store_law_compatible"] is True
        else:
            assert flags["opaque_store_law_compatible"] is False


def test_runner_smoke_and_refuses_dev_before_implementation():
    from experiments.run_tm058storeint import implementation_present, refuse_runner_leaks, run_dev, smoke

    src = RUNNER.read_text()
    assert "write_opaque_kv" in src
    assert "flag_checkpoint_key" in src
    assert "latest_episode(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "solve_ceiling" not in names
    assert "_run_joint_socp_consolidation" not in names
    assert implementation_present() is False
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 12
    assert out["implementation_present"] is False
    assert out["ladder"]["setup"] == "setup_precondition_fail"
    assert out["ladder"]["cross_action_refresh"] == "cross_action_refresh"
    assert out["ladder"]["storage_integrity_holds"] == "storage_integrity_holds"
    assert out["floor"] == 0.05
    assert out["slots"] == 8
    assert out["candidate_exists"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    try:
        run_dev()
        raise AssertionError("DEV must refuse before implementation")
    except RuntimeError as exc:
        assert "before opaque-store implementation" in str(exc)
