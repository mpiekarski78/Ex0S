"""TM042 post-install continuity freeze tests. No neural or solver edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_postinstall.prereg.lock"
ISO = REPO / "docs" / "lineage_postinstall.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_postinstall_contract.md"
RUNNER = REPO / "experiments" / "run_tm042postinstall.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANONICAL = REPO / "experiments" / "canonical_act_probe.py"
R2_DEC = REPO / "docs" / "lineage_causalbattery.r2.decision.lock"
R2_DEV = REPO / "docs" / "lineage_causalbattery.r2.dev.lock"
TM040_DEC = REPO / "docs" / "lineage_causalbattery.decision.lock"
MANIFEST = "a1d2160fc3a13346bf3cfc1995a69a7d9d6e6c662a47536db1d30aa0fb8cbc0c"
FROZEN_NEURAL_SHA = "2eb45d8769402330f5ee39a04afffe110a435a0e64a40b12bc2d874b36f5ed59"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
CANONICAL_SHA = "51e24d272417df5ae689301d9600d49aa86daec6608dfc7ff26f8ad4c2e22aef"
R2_DEC_SHA = "bcd40fba96ff96d90958aaf4c03fd4bb8fa2995dccd313600a09e9fc50124f23"
R2_DEV_SHA = "a13838622a76fb3b7f62a73ef3e58001db0a4bf99cb9ede9c575bd7f7c438ab3"
TM040_DEC_SHA = "734204f628362f58e4f3b19237dd82398016544655d2c13800f3408854bd1b99"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_r2_stop_and_no_edits():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    r2 = json.loads(R2_DEC.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 12
    assert p["lineage_stop"] == "canonical_r2_later_learning_not_exercised"
    assert r2["decision"]["code"] == "canonical_r2_later_learning_not_exercised"
    assert p["v40_candidate_authorized"] is False
    assert p["neural_edit_authorized"] is False
    assert p["seed_registry"] not in (22222, 404000039, 404200040)
    assert p["domains"]["DEV"] == "TM042.POSTINSTALL.DEV."
    assert p["mechanistic_reconstruct_domain"] == "TM039.JOINTSOCP.DEV."
    assert "treat_mechanistic_as_untouched_generalization" in iso["refuse"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(CANONICAL) == CANONICAL_SHA
    assert _sha(R2_DEC) == R2_DEC_SHA
    assert _sha(R2_DEV) == R2_DEV_SHA
    assert _sha(TM040_DEC) == TM040_DEC_SHA
    assert p["frozen_runner_sha"] == sha_file(RUNNER)
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_ids_and_decision_ladder():
    from experiments.run_tm042postinstall import _decision, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 12
    assert ids[0] == "natural|c8h2|A_then_B|w0"
    assert ids[-1] == "mech|tm039|reg3|B_then_A"
    nat = {
        "kind": "natural",
        "installed": False,
        "continuity_ok": False,
    }
    mech = {
        "kind": "mechanistic",
        "installed": True,
        "continuity_ok": True,
    }
    code, _t, fl = _decision([dict(nat), dict(nat), dict(mech)])
    assert code == "postinstall_not_exercised"
    assert fl["lineage_stop"] == "canonical_r2_later_learning_not_exercised"
    assert fl["candidate_v40_lock"] is False
    nat_ok = dict(nat, installed=True, continuity_ok=True)
    code2, _t2, fl2 = _decision([nat_ok, dict(mech)])
    assert code2 == "postinstall_continuity_pass"
    assert fl2["candidate_discussion_open"] is True
    assert fl2["candidate_v40_lock"] is False
    code3, _t3, _f3 = _decision([dict(nat, installed=True, continuity_ok=False), dict(mech)])
    assert code3 == "postinstall_natural_later_fail"


def test_refuse_raw_and_smoke():
    from experiments.run_tm042postinstall import refuse_raw_scores, smoke

    assert refuse_raw_scores(RUNNER) == []
    out = smoke()
    assert out["smoke_ok"]
    assert out["raw_score_leak"] == []
    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src
    assert ".actuator_scores(" not in src
