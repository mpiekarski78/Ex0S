"""TM061 co-drift localization freeze tests.

No neural edit. No installed W_star. No K/Q/V redesign.
Do not retune 0.05. Do not rerun or edit TM060.
Product 0.0.004. Never write cortex.candidate.v41.lock.
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
    EPISODE_SLOTS,
    GenomeConfig,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_codrift.prereg.lock"
ISO = REPO / "docs" / "lineage_codrift.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_codrift_contract.md"
RUNNER = REPO / "experiments" / "run_tm061codrift.py"
TM060_RUNNER = REPO / "experiments" / "run_tm060ondrift.py"
TM060_DEV = REPO / "docs" / "lineage_ondrift.dev.lock"
TM060_DEC = REPO / "docs" / "lineage_ondrift.decision.lock"
TM060_PREREG = REPO / "docs" / "lineage_ondrift.prereg.lock"
TM059_RUNNER = REPO / "experiments" / "run_tm059receipt.py"
LAW = REPO / "docs" / "lineage_write_time_law.lock"
OPAQUE_LAW = REPO / "docs" / "lineage_opaque_store_law.lock"
COMPAT = REPO / "docs" / "lineage_runner_compat.lock"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
MANIFEST = "8191323c0a44ff01d08066d3aad8f017acf68f33df334186457936e8d00d1bf2"
NEURAL_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
OPAQUE_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM060_RUNNER_SHA = "c783babf8cb1e63bda402ee7c1f22461718be6ef2007a6177f7b735805b38a42"
TM060_DEV_SHA = "64ec147180c343549b20c97e9e2a00789e0a02dc33d062378eb763fa9cf2d0fb"
TM060_DEC_SHA = "5b9f0ee91c76e04be984a50c8d4b77802e91a570d54ef307d2d30849ad851eb4"
TM060_PREREG_SHA = "7e8cd3268767883a8f4c2c4030961ae66ed6dbea7890c75f09a84be1d2dc6773"
TM059_RUNNER_SHA = "8aad0201e391deb5c01d7aca7f50d561a20b3af6d12106a77351b15b3e06229f"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
OPAQUE_LAW_SHA = "86893cc7614b1e270fb004028dfde82dc5e06054bc6f5d6ca2aaa6ba82c4260d"
COMPAT_SHA = "a475d3f2bbea6e35832d46b467468cd989eba2df40c2e1a7372ca018ff451f14"
LADDER = [
    "setup_precondition_fail",
    "observer_used_runner_provenance",
    "full_oracle_infeasible",
    "prefix_baseline_fail",
    "tm060_cross_not_reproduced",
    "later_values_unreadable",
    "transport_restores_decoding",
    "nonlinear_manifold_reorganization",
    "matching_time_co_drift",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm060_remains_representation_drift_not_architecture():
    assert _sha(TM060_RUNNER) == TM060_RUNNER_SHA
    assert _sha(TM060_DEV) == TM060_DEV_SHA
    assert _sha(TM060_DEC) == TM060_DEC_SHA
    assert _sha(TM060_PREREG) == TM060_PREREG_SHA
    assert _sha(TM059_RUNNER) == TM059_RUNNER_SHA
    dec = json.loads(TM060_DEC.read_text())
    dev = json.loads(TM060_DEV.read_text())
    assert dev["decision_code"] == "representation_drift"
    assert dec["decision"]["code"] == "representation_drift"
    assert dec["architectural_conclusion"] == "none"
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_codrift_wall():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.61.CODRIFT"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["opaque_memory_edit_authorized"] is False
    assert p["kqv_edit_authorized"] is False
    assert p["solver_edit_authorized"] is False
    assert p["install_oracle_authorized"] is False
    assert p["dev_authorized_before_runner_compat"] is False
    assert p["tm060_decision_code"] == "representation_drift"
    assert p["tm060_architectural_conclusion"] == "none"
    assert p["tm060_earned_architecture"] is False
    assert p["tm060_earned_localization_wall"] is True
    assert p["discard_every_W_star"] is True
    assert p["discard_every_transport"] is True
    assert p["side_effect_free"] is True
    assert p["install_W_star"] is False
    assert p["split_n"] == 16
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 12
    assert p["expected_n_cells"] == 14
    assert p["n_dev_repeats"] == 4
    assert p["n_online_repeats"] == 8
    assert p["seed_registry"] == 404900061
    assert 404900060 in p["forbidden_seeds"]
    assert p["probes"] == [
        "prefix_on_prefix",
        "prefix_on_later",
        "later_on_prefix",
        "later_on_later",
        "linear_transport",
        "orthogonal_transport",
    ]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "experiments/run_tm060ondrift.py" in iso["historical_immutable"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert "three_memory/joint_socp.py" in iso["historical_immutable"]
    assert iso["implementation_authorized"] is False
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


def test_ladder_and_transport_maps():
    from experiments.run_tm061codrift import (
        BEHAVIORAL_LADDER,
        _decision,
        expected_cell_ids,
        fit_linear,
        fit_orthogonal,
        map_later_to_prefix,
        map_prefix_to_later,
        synthetic_grid,
    )

    ids = expected_cell_ids()
    assert ids[0] == "decoder|w0"
    assert ids[-1] == "orthogonal_transport|w1"
    assert len(ids) == 14
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["canonical_law_reconsidered"] is False
        assert flags["discard_every_W_star"] is True
        assert flags["tm060_earned_architecture"] is False

    rng = np.random.default_rng(61)
    r = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    if np.linalg.det(r) < 0:
        r[:, -1] *= -1
    prefix = rng.normal(size=(12, 8))
    prefix = prefix / np.linalg.norm(prefix, axis=1, keepdims=True)
    later = prefix @ r
    later = later / np.linalg.norm(later, axis=1, keepdims=True)
    q_lin = fit_linear(prefix, later)
    q_orth = fit_orthogonal(prefix, later)
    x = prefix[0]
    y = later[0]
    got_lin = map_prefix_to_later(x, q_lin, orthogonal=False)
    got_orth = map_prefix_to_later(x, q_orth, orthogonal=True)
    back_lin = map_later_to_prefix(y, q_lin, orthogonal=False)
    back_orth = map_later_to_prefix(y, q_orth, orthogonal=True)
    assert float(np.linalg.norm(got_lin - y / np.linalg.norm(y))) < 1e-6
    assert float(np.linalg.norm(got_orth - y / np.linalg.norm(y))) < 1e-6
    assert float(np.linalg.norm(back_lin - x / np.linalg.norm(x))) < 1e-6
    assert float(np.linalg.norm(back_orth - x / np.linalg.norm(x))) < 1e-6


def test_runner_reuses_receipts_and_does_not_import_tm060_collect_bundle():
    from experiments.run_tm059receipt import receipt_contract_violations
    from experiments.run_tm061codrift import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "from experiments.run_tm060ondrift import" in src
    assert "Receipts" in src
    assert "opaque_live" in src
    assert "collect_bundle," not in src
    assert "run_tm060ondrift import collect_bundle" not in src
    assert "linear_transport" in src
    assert "orthogonal_transport" in src
    assert "discarded" in src
    assert "install_W_star" in src
    assert 'provenance_id=f"' not in src
    assert "latest_episode(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    assert receipt_contract_violations(RUNNER) == []
    assert receipt_contract_violations(TM060_RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "_run_joint_socp_consolidation" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 14
    assert out["receipt_contract"] == []
    assert out["ladder"]["setup"] == "setup_precondition_fail"
    assert out["ladder"]["later_values_unreadable"] == "later_values_unreadable"
    assert out["ladder"]["transport_restores_decoding"] == "transport_restores_decoding"
    assert out["ladder"]["nonlinear_manifold_reorganization"] == "nonlinear_manifold_reorganization"
    assert out["ladder"]["matching_time_co_drift"] == "matching_time_co_drift"
    assert out["ladder"]["tm060_cross_not_reproduced"] == "tm060_cross_not_reproduced"
    assert out["floor"] == 0.05
    assert out["slots"] == 8
    assert out["candidate_exists"] is False


DEV_SHA = "79a799768c4300938c63f75d99ccc2c9bf295865240f2ad9496b8f26ccb011d1"
DEC_SHA = "5ecd0a650666d3d31df281c7e37f87daa3fc72feb55c590f3f35fd9c7607e1af"
DEV_GIT = "502e7010516b5f3dbb422343f8fbe53c0fface8e"


def test_dev_lock_transport_restores_decoding():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm061codrift import expected_cell_ids

    devp = REPO / "docs" / "lineage_codrift.dev.lock"
    decp = REPO / "docs" / "lineage_codrift.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert sha_file(RUNNER) == "c746ddaa7d9e270922b8386eb6c4e90932f6c2bdcf2d2cb1b41d241fa572e99f"
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "transport_restores_decoding"
    assert dev["install_W_star"] is False
    assert dev["neural_untouched"] is True
    assert dec["architectural_conclusion"] == "none"
    assert dec["tm060_earned_architecture"] is False
    assert dec["decision"]["code"] == "transport_restores_decoding"
    assert dec["decision"]["phase_flags"]["transport_restores_decoding"] is True
    assert dec["decision"]["phase_flags"]["later_values_unreadable"] is False
    assert dec["decision"]["phase_flags"]["nonlinear_manifold_reorganization"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["passed"] is True
    assert cells["decoder|w1"]["passed"] is True
    assert cells["decoder|w0"]["full_oracle_feasible"] is True
    assert cells["decoder|w1"]["full_oracle_feasible"] is True
    assert cells["prefix_on_prefix|w0"]["ok"] is True
    assert cells["prefix_on_prefix|w1"]["ok"] is True
    assert cells["prefix_on_later|w0"]["ok"] is False
    assert cells["prefix_on_later|w1"]["ok"] is False
    assert cells["later_on_later|w0"]["ok"] is True
    assert cells["later_on_later|w1"]["ok"] is True
    assert cells["linear_transport|w0"]["restored"] is True
    assert cells["linear_transport|w1"]["restored"] is True
    assert cells["orthogonal_transport|w0"]["restored"] is False
    assert cells["orthogonal_transport|w1"]["restored"] is False
    for cid in expected_cell_ids():
        if cells[cid].get("kind") != "scored":
            continue
        assert cells[cid]["W_installed"] is False
        assert cells[cid]["discarded"] is True
        assert cells[cid]["parent_w_act_query_unchanged"] is True
        assert cells[cid]["future_never_socp_constraints"] is True
