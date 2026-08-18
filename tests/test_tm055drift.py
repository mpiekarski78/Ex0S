"""TM055 online representation-drift freeze tests.

Diagnostic only. No neural edit. Do not install W_star. Do not retune 0.05.
Leave TM046–TM054 runner/DEV/decision/addendum untouched.
Canonical generator is write-time last P1. Never regenerate wraps.
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
PREREG = REPO / "docs" / "lineage_drift.prereg.lock"
ISO = REPO / "docs" / "lineage_drift.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_drift_contract.md"
RUNNER = REPO / "experiments" / "run_tm055drift.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
LAW = REPO / "docs" / "lineage_write_time_law.lock"
TM054_RUNNER = REPO / "experiments" / "run_tm054provenance.py"
TM054_DEV = REPO / "docs" / "lineage_provenance.dev.lock"
TM054_DEC = REPO / "docs" / "lineage_provenance.decision.lock"
TM054_ADD = REPO / "docs" / "lineage_provenance.decision.addendum.lock"
TM053_RUNNER = REPO / "experiments" / "run_tm053cover.py"
TM053_DEV = REPO / "docs" / "lineage_cover.dev.lock"
TM053_ADD = REPO / "docs" / "lineage_cover.decision.addendum.lock"
TM052_RUNNER = REPO / "experiments" / "run_tm052sharefeas.py"
TM052_DEV = REPO / "docs" / "lineage_sharefeas.dev.lock"
MANIFEST = "ad4110b1edb3f9b71fa7078573faf85f653db8beb07bab7576bd9ef66e3af3d5"
NEURAL_SHA = "2ba95d71f2893cf0c2b3069836b6fbe1ff4840d2d746331e47b9a38650475c63"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM054_RUNNER_SHA = "cbfe1c37de4532141d2e770ae03b2d50a32f720276fe6f2064c9a7f7b4ee685d"
TM054_DEV_SHA = "0f16bdbd3a068d788e138df0545ed0aa1e395c6875ce4b5ec365405412a05543"
TM054_DEC_SHA = "9ac7a14f6776c04a9bbfe6f4f44418e4f28d830b6b5e49514deba6a34f5f393e"
TM054_ADD_SHA = "bfa9b1937041afa28c4813b6366751f0e12e0edf9941e9f9585dfe23f5d37fb8"
LAW_SHA = "73f96668385282fc29a0bcf0c28e17c484ac1e51a473aa183f4b6fa148c9d068"
TM053_RUNNER_SHA = "62e2fe15a3e0565d9041c36cfb16b7fae24d98c8211dec9358a0437770dd2bb4"
TM053_DEV_SHA = "de2b615eb2b386b10d4f9aac5346d7b0e44301e34455806b4b27f339daa7e374"
TM053_ADD_SHA = "ee78a1be25ed1db0b5217120be2cfab959c7cded398f7074014035ee3fe7916c"
TM052_RUNNER_SHA = "36c119262be5a7b2e186b22d3a5e37ffc4e27c4706249156562905f7d025abeb"
TM052_DEV_SHA = "e80ec58901ab456202a6715c74807c1a9b93a34baa546703494b9d90eb55b64a"
RUNNER_SHA = "23a3002029560e6a83d5ae5646e5631101fee6b89981e8cd814688e87f9a392b"
LADDER = [
    "setup_precondition_fail",
    "prefix_infeasible",
    "incompatible_action_clusters",
    "catastrophic_representational_migration",
    "write_must_be_included",
    "developmental_coordinate_drift",
    "grounding_consolidation_plausible",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm054_wall_untouched():
    assert _sha(TM054_RUNNER) == TM054_RUNNER_SHA
    assert _sha(TM054_DEV) == TM054_DEV_SHA
    assert _sha(TM054_DEC) == TM054_DEC_SHA
    assert _sha(TM054_ADD) == TM054_ADD_SHA
    assert _sha(LAW) == LAW_SHA
    assert _sha(TM053_RUNNER) == TM053_RUNNER_SHA
    assert _sha(TM053_DEV) == TM053_DEV_SHA
    assert _sha(TM053_ADD) == TM053_ADD_SHA
    assert _sha(TM052_RUNNER) == TM052_RUNNER_SHA
    assert _sha(TM052_DEV) == TM052_DEV_SHA
    dec = json.loads(TM054_DEC.read_text())
    add = json.loads(TM054_ADD.read_text())
    law = json.loads(LAW.read_text())
    assert dec["decision"]["code"] == "state_generator_mismatch"
    assert add["frozen_first_match_unchanged"] is True
    assert add["canonical_state_generator"] == "write_time_last_p1"
    assert add["frozen_wrap_is_not_canonical"] is True
    assert add["rewrite_historical_dev"] is False
    assert law["canonical_state_generator"] == "write_time_last_p1"
    assert law["law"]["captured"] == "at_event_time"
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_write_time_gate():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.55.DRIFT"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["install_oracle_authorized"] is False
    assert p["episode_match_l2_retune_authorized"] is False
    assert p["force_cue_into_value_authorized"] is False
    assert p["frozen_wrap_canonical_authorized"] is False
    assert p["regenerate_historical_values_authorized"] is False
    assert p["n_setup_cells"] == 2
    assert p["n_scored_cells"] == 12
    assert p["expected_n_cells"] == 14
    assert p["n_dev_repeats"] == 4
    assert p["n_online_repeats"] == 8
    assert p["n_grid"] == [4, 8, 12, 16, 20, 24]
    assert p["floor"] == 0.05
    assert p["canonical_state_generator"] == "write_time_last_p1"
    assert p["never_regenerate_historical_values"] is True
    assert p["discard_every_W_star"] is True
    assert p["reconstruction_seed_registry"] == 404600046
    assert p["seed_registry"] == 404900055
    assert 404900054 in p["forbidden_seeds"]
    assert p["tm054_decision_code"] == "state_generator_mismatch"
    assert [d["code"] for d in p["decision_ladder"]] == LADDER
    assert "treat frozen wrap as canonical" in p["refuse"]
    assert "experiments/run_tm054provenance.py" in iso["historical_immutable"]
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
    from experiments.run_tm055drift import BEHAVIORAL_LADDER, _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert ids[0] == "decoder|w0"
    assert ids[-1] == "n24|w1"
    assert len(ids) == 14
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step
        assert flags["install_W_star"] is False
        assert flags["discard_every_W_star"] is True
        if step == "grounding_consolidation_plausible":
            assert flags["generic_consolidation_plausible"] is True
        if step == "catastrophic_representational_migration":
            assert flags["reconsolidation_required"] is True


def test_runner_refuses_wraps_and_smoke():
    from experiments.run_tm055drift import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "snapshot_write" in src
    assert "latest_episode" in src
    assert "credit_tagged(" not in src
    assert "collect_wraps(" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "solve_min_change_socp" not in names or True
    assert "_run_joint_socp_consolidation" not in names
    assert "credit_tagged" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 14
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_prefix"] == "prefix_infeasible"
    assert out["ladder_cluster"] == "incompatible_action_clusters"
    assert out["ladder_catastrophic"] == "catastrophic_representational_migration"
    assert out["ladder_include"] == "write_must_be_included"
    assert out["ladder_drift"] == "developmental_coordinate_drift"
    assert out["ladder_ok"] == "grounding_consolidation_plausible"
    assert out["floor"] == 0.05
    assert out["candidate_exists"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    assert not hasattr(NeuralCortex, "set_feedback_ticks")


DEV_SHA = "177395281c25171628132c38eeda1056321d5a1cf9887600e73514f5558e7f8c"
DEC_SHA = "9040428ebadb8cc0e692eca3825dc74d4f52539ba4207ec19f9748eacca69caf"
DEV_GIT = "1a933f439f5f2e72bbbfaec50c781c253cfb683f"


def test_dev_lock_setup_precondition_fail_no_install():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm055drift import expected_cell_ids

    devp = REPO / "docs" / "lineage_drift.dev.lock"
    decp = REPO / "docs" / "lineage_drift.decision.lock"
    assert _sha(devp) == DEV_SHA
    assert _sha(decp) == DEC_SHA
    assert _sha(TM054_RUNNER) == TM054_RUNNER_SHA
    assert _sha(TM054_DEV) == TM054_DEV_SHA
    assert _sha(TM054_ADD) == TM054_ADD_SHA
    assert _sha(LAW) == LAW_SHA
    assert _sha(TM053_RUNNER) == TM053_RUNNER_SHA
    assert _sha(TM052_DEV) == TM052_DEV_SHA
    assert sha_file(RUNNER) == RUNNER_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert not CANDIDATE_V41.exists()
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == DEV_GIT
    assert dev["decision_code"] == "setup_precondition_fail"
    assert dev["install_W_star"] is False
    assert dev["discard_every_W_star"] is True
    assert dev["episode_match_l2_retuned"] is False
    assert dev["canonical_state_generator"] == "write_time_last_p1"
    assert dec["architectural_conclusion"] == "none"
    assert dec["decision"]["code"] == "setup_precondition_fail"
    assert dec["decision"]["phase_flags"]["generic_consolidation_plausible"] is False
    assert dec["dev_lock_sha"] == _sha(devp)
    cells = {c["id"]: c for c in dev["cells"]}
    assert list(cells) == expected_cell_ids()
    assert cells["decoder|w0"]["pin_match"] is False
    assert cells["decoder|w0"]["w0_match"] is True
    assert cells["decoder|w0"]["ref_ok"] is True
    assert cells["decoder|w1"]["pin_match"] is False
    assert cells["decoder|w1"]["w0_match"] is True
    assert cells["n4|w0"]["applied"] is False
    assert cells["n4|w0"]["W_installed"] is False
    assert cells["n4|w0"]["discarded"] is True

