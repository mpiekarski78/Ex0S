"""TM062 transport-generalization freeze tests.

No neural edit. No installed W_star. No K/Q/V redesign.
Do not retune 0.05. Do not rerun or edit TM060 or TM061.
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
PREREG = REPO / "docs" / "lineage_xgen.prereg.lock"
ISO = REPO / "docs" / "lineage_xgen.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_xgen_contract.md"
RUNNER = REPO / "experiments" / "run_tm062xgen.py"
TM061_RUNNER = REPO / "experiments" / "run_tm061codrift.py"
TM061_DEV = REPO / "docs" / "lineage_codrift.dev.lock"
TM061_DEC = REPO / "docs" / "lineage_codrift.decision.lock"
TM061_ADD = REPO / "docs" / "lineage_codrift.decision.addendum.lock"
TM060_RUNNER = REPO / "experiments" / "run_tm060ondrift.py"
LAW = REPO / "docs" / "lineage_write_time_law.lock"
OPAQUE_LAW = REPO / "docs" / "lineage_opaque_store_law.lock"
COMPAT = REPO / "docs" / "lineage_runner_compat.lock"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
MANIFEST = "886c5f440ccd583254f1af438c99f68fc7ed392083f720c75f1f17ba66f9af5d"
NEURAL_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
OPAQUE_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM061_RUNNER_SHA = "c746ddaa7d9e270922b8386eb6c4e90932f6c2bdcf2d2cb1b41d241fa572e99f"
TM061_DEV_SHA = "79a799768c4300938c63f75d99ccc2c9bf295865240f2ad9496b8f26ccb011d1"
TM061_DEC_SHA = "5ecd0a650666d3d31df281c7e37f87daa3fc72feb55c590f3f35fd9c7607e1af"
TM061_ADD_SHA = "31c76369663ebb666d0088ec163f2de8c2e63f05296b17ba29fcfe48fcfb1e9f"
TM060_RUNNER_SHA = "c783babf8cb1e63bda402ee7c1f22461718be6ef2007a6177f7b735805b38a42"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
OPAQUE_LAW_SHA = "86893cc7614b1e270fb004028dfde82dc5e06054bc6f5d6ca2aaa6ba82c4260d"
COMPAT_SHA = "a475d3f2bbea6e35832d46b467468cd989eba2df40c2e1a7372ca018ff451f14"
LADDER = [
    "setup_precondition_fail",
    "observer_used_runner_provenance",
    "full_oracle_infeasible",
    "prefix_baseline_fail",
    "later_values_unreadable",
    "tm061_in_sample_not_reproduced",
    "only_unrestricted_interpolates",
    "action_conditioned_transport",
    "constrained_transport_generalizes",
    "shared_map_transfers",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm061_remains_valid_and_narrow():
    assert _sha(TM061_RUNNER) == TM061_RUNNER_SHA
    assert _sha(TM061_DEV) == TM061_DEV_SHA
    assert _sha(TM061_DEC) == TM061_DEC_SHA
    assert _sha(TM061_ADD) == TM061_ADD_SHA
    assert _sha(TM060_RUNNER) == TM060_RUNNER_SHA
    dec = json.loads(TM061_DEC.read_text())
    add = json.loads(TM061_ADD.read_text())
    assert dec["decision"]["code"] == "transport_restores_decoding"
    assert dec["architectural_conclusion"] == "none"
    assert add["historical_decision_code"] == "transport_restores_decoding"
    assert add["rewrite_historical_decision"] is False
    assert add["rerun_dev"] is False
    assert add["narrow_conclusion"] is True
    assert add["interpretation"] == "in_sample_linear_map_exists_underdetermined_not_a_generic_transport_law"
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_holdout_transport():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.62.XGEN"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["kqv_edit_authorized"] is False
    assert p["install_oracle_authorized"] is False
    assert p["tm061_decision_code"] == "transport_restores_decoding"
    assert p["tm061_earned_architecture"] is False
    assert p["tm061_narrow_conclusion"] is True
    assert p["tm061_underdetermined_interpolation_caveat"] is True
    assert p["fit_cycles"] == 2
    assert p["split_n"] == 16
    assert p["low_rank"] == 4
    assert p["ridge_l2"] == 1.0
    assert p["discard_every_transport"] is True
    assert p["correspondence_is_organism_emitted_action_and_occurrence"] is True
    assert p["runner_action_labels_diagnostic_only"] is True
    assert p["holdout_are_later_contexts_not_used_to_fit"] is True
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 18
    assert p["expected_n_cells"] == 20
    assert p["n_dev_repeats"] == 4
    assert p["seed_registry"] == 404900062
    assert 404900061 in p["forbidden_seeds"]
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "experiments/run_tm061codrift.py" in iso["historical_immutable"]
    assert "docs/lineage_codrift.decision.addendum.lock" in iso["historical_immutable"]
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert iso["implementation_authorized"] is False
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(LAW) == LAW_SHA
    assert _sha(OPAQUE_LAW) == OPAQUE_LAW_SHA
    assert _sha(COMPAT) == COMPAT_SHA
    assert "opaque_store_enabled" not in GenomeConfig().to_dict()
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(RUNNER)


def test_ladder_and_holdout_split():
    from experiments.run_tm062xgen import (
        BEHAVIORAL_LADDER,
        _decision,
        aligned_pairs,
        expected_cell_ids,
        synthetic_grid,
    )

    ids = expected_cell_ids()
    assert ids[0] == "decoder|w0"
    assert ids[-1] == "action_conditioned|w1"
    assert len(ids) == 20
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, _ = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step, (step, c)
        assert flags["install_W_star"] is False
        assert flags["tm061_earned_architecture"] is False
        assert flags["tm061_narrow_conclusion"] is True

    handles = ["a", "b", "c", "d"]
    prefix = []
    later = []
    for occ in range(4):
        for i, h in enumerate(handles):
            v = np.zeros(8)
            v[i] = 1.0 + 0.01 * occ
            prefix.append({"handle": h, "p1": v})
            later.append({"handle": h, "p1": v + 0.2})
    pairs = aligned_pairs(prefix, later, handles, 8)
    assert [r["occ"] for r in pairs[:4]] == [0, 0, 0, 0]
    assert {r["handle"] for r in pairs if r["occ"] < 2} == set(handles)
    hold = [r for r in pairs if r["occ"] >= 2]
    fit = [r for r in pairs if r["occ"] < 2]
    assert len(fit) == 8 and len(hold) == 8
    assert min(r["occ"] for r in hold) == 2


def test_runner_reuses_receipts_and_does_not_import_tm061_collect_bundle():
    from experiments.run_tm059receipt import receipt_contract_violations
    from experiments.run_tm062xgen import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "from experiments.run_tm060ondrift import" in src
    assert "from experiments.run_tm061codrift import" in src
    assert "collect_bundle," not in src
    assert "run_tm061codrift import collect_bundle" not in src
    assert "holdout_later_contexts" in src
    assert "organism_emitted_action_and_occurrence" in src
    assert "rank_A" in src
    assert "effective_dof" in src
    assert 'provenance_id=f"' not in src
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
    assert out["n_cells"] == 20
    assert out["receipt_contract"] == []
    assert out["ladder"]["only_unrestricted_interpolates"] == "only_unrestricted_interpolates"
    assert out["ladder"]["action_conditioned_transport"] == "action_conditioned_transport"
    assert out["ladder"]["constrained_transport_generalizes"] == "constrained_transport_generalizes"
    assert out["ladder"]["shared_map_transfers"] == "shared_map_transfers"
    assert out["floor"] == 0.05
    assert out["candidate_exists"] is False


DEV_SHA = "08eff57988216d813c34cd69cba277c9e0f0bb5e33d18071e987510ac2d2452e"
DEC_SHA = "bb529975eee280fbf0d0ca688b5bace90f1bd41f36499716b1ee965a6f89880b"
DEV_GIT = "5e2fe6c2196f03de549163326d03f293de2c36da"


def test_dev_lock_only_unrestricted_interpolates():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm062xgen import expected_cell_ids

    devp = REPO / "docs" / "lineage_xgen.dev.lock"
    decp = REPO / "docs" / "lineage_xgen.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert sha_file(RUNNER) == "c62188da88a9dee4f3a75f26c438e82cd7d063185b50c6c5f18b58e444ffaf76"
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "only_unrestricted_interpolates"
    assert dev["install_W_star"] is False
    assert dev["neural_untouched"] is True
    assert dev["tm061_narrow_conclusion"] is True
    assert dec["architectural_conclusion"] == "none"
    assert dec["tm061_earned_architecture"] is False
    assert dec["decision"]["code"] == "only_unrestricted_interpolates"
    assert dec["decision"]["phase_flags"]["only_unrestricted_interpolates"] is True
    assert dec["decision"]["phase_flags"]["constrained_transport_generalizes"] is False
    assert dec["decision"]["phase_flags"]["shared_map_transfers"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["prefix_ok"] is True
    assert cells["decoder|w1"]["later_ok"] is True
    assert cells["decoder|w0"]["full_oracle_feasible"] is True
    assert cells["in_sample_linear|w0"]["restored"] is True
    assert cells["in_sample_linear|w1"]["restored"] is True
    assert cells["in_sample_linear|w0"]["scored_on"] == "fit_pairs"
    assert cells["min_norm|w0"]["scored_on"] == "holdout_later_contexts"
    assert cells["min_norm|w0"]["restored"] is False
    assert cells["min_norm|w1"]["restored"] is False
    assert cells["ridge|w0"]["restored"] is False
    assert cells["identity|w0"]["restored"] is False
    assert cells["orthogonal|w0"]["restored"] is False
    assert cells["action_conditioned|w0"]["restored"] is False
    assert cells["in_sample_linear|w0"]["rank_A"] == 8
    assert cells["in_sample_linear|w0"]["n_fit"] == 8
    for cid in expected_cell_ids():
        if cells[cid].get("kind") != "scored":
            continue
        assert cells[cid]["W_installed"] is False
        assert cells[cid]["discarded"] is True
        assert cells[cid]["parent_w_act_query_unchanged"] is True
        assert cells[cid]["future_never_socp_constraints"] is True
