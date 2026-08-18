"""TM041 live-address freeze tests. No neural or solver edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_liveaddr.prereg.lock"
ISO = REPO / "docs" / "lineage_liveaddr.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_liveaddr_contract.md"
RUNNER = REPO / "experiments" / "run_tm041liveaddr.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
TM040_DEC = REPO / "docs" / "lineage_causalbattery.decision.lock"
TM040_DEV = REPO / "docs" / "lineage_causalbattery.dev.lock"
TM040_ADD = REPO / "docs" / "lineage_causalbattery.decision.addendum.lock"
TM040_RUNNER = REPO / "experiments" / "run_tm040causal.py"
MANIFEST = "b3f13740a0839f9750b279bedcb367add69d92c93388cc010a35b175666f970e"
FROZEN_RUNNER_SHA = "63dd0aac8c769d1d19df3330936734d1376accbd48d69bcd1b520861dd13ab9a"
FROZEN_NEURAL_SHA = "2eb45d8769402330f5ee39a04afffe110a435a0e64a40b12bc2d874b36f5ed59"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM040_DEC_SHA = "734204f628362f58e4f3b19237dd82398016544655d2c13800f3408854bd1b99"
TM040_DEV_SHA = "b10865b5f6fea382396db736549488c68dcdc5000932907a3612e29b53354ad7"
TM040_RUNNER_SHA = "0739de3225e36bf88b66001ae9af2c3232a937d0bc6bc3b64eb34bdf7d2f9c6b"
TM040_ADD_SHA = "47de79c310e7ac6232cd69ca50d183430ecb98ba16f261f1120ac29963eba5e2"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_reconstructs_tm040_and_forbids_edits():
    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    add = json.loads(TM040_ADD.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 12
    assert p["reconstruct_domain"] == "TM040.CAUSAL.DEV."
    assert p["v40_candidate_authorized"] is False
    assert p["invoke_SOCP_unconditionally"] is False
    assert p["add_live_constraints"] is False
    assert p["change_fallback_trigger"] is False
    assert p["modify_half_spacing"] is False
    assert p["later_learning"] == "not_exercised"
    assert p["contradict"] == "jointly_feasible_atomic_apply"
    assert add["first_match_unchanged"] == "jointsocp_fallback_acquire_fail"
    assert add["corrections"]["later_learning"] == "not_exercised"
    assert add["corrections"]["contradict"] == "jointly_feasible_atomic_apply"
    assert "edit_neural_cortex.py" in iso["refuse"]
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA
    assert p["frozen_neural_sha"] == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(TM040_DEC) == TM040_DEC_SHA
    assert _sha(TM040_DEV) == TM040_DEV_SHA
    assert _sha(TM040_RUNNER) == TM040_RUNNER_SHA
    assert _sha(TM040_ADD) == TM040_ADD_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_ids_and_classify():
    from experiments.run_tm041liveaddr import _decision, classify_failed_cue, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 12
    assert ids[0] == "acquire|c8|A_then_B|w0|v37"
    assert ids[-1] == "acquire|c8|B_then_A|w1|always_joint"
    row = {
        "teach_index": 2,
        "canonical": {
            "path": "cortical_fallback",
            "familiar": False,
            "slot": None,
            "scoring_address_hash": "a",
        },
        "store": {"violation": False, "p1_hash": "s"},
        "live_tm040": {"p1_hash": "a", "ranking_ok": False},
        "counterfactual_stored": {"ranking_ok": True},
        "counterfactual_live": {"ranking_ok": False},
    }
    assert classify_failed_cue(row) == "cortical_fallback_unfamiliar"
    row["canonical"]["path"] = "episodic_completed"
    row["canonical"]["familiar"] = True
    row["canonical"]["slot"] = 4
    assert classify_failed_cue(row) == "wrong_slot"
    row["canonical"]["slot"] = 2
    assert classify_failed_cue(row) == "store_pass_live_addr_fail"
    row["canonical"]["scoring_address_hash"] = "s"
    row["counterfactual_live"]["ranking_ok"] = True
    assert classify_failed_cue(row) == "canonical_path_inconsistency"
    row["store"]["violation"] = True
    row["counterfactual_stored"]["ranking_ok"] = False
    assert classify_failed_cue(row) == "stored_p1_fails"
    code, _t, fl = _decision(
        [
            {
                "arm": "fallback_joint",
                "stem": "acquire|c8|A_then_B|w0",
                "failed_classes": ["cortical_fallback_unfamiliar"],
                "passed_tm040_live": False,
                "w_hash": "aa",
                "cues": [],
            },
            {
                "arm": "always_joint",
                "stem": "acquire|c8|A_then_B|w0",
                "failed_classes": [],
                "passed_tm040_live": True,
                "w_hash": "bb",
                "cues": [],
            },
            {
                "arm": "v37",
                "stem": "acquire|c8|A_then_B|w0",
                "failed_classes": ["cortical_fallback_unfamiliar"],
                "passed_tm040_live": False,
                "w_hash": "aa",
                "cues": [],
            },
        ]
    )
    assert code == "liveaddr_cortical_fallback_unfamiliar"
    assert fl["later_learning"] == "not_exercised"
    assert fl["contradict"] == "jointly_feasible_atomic_apply"
    assert fl["n_stems_v37_fallback_w_equal"] == 1


def test_smoke():
    from experiments.run_tm041liveaddr import smoke

    out = smoke()
    assert out["smoke_ok"]
    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src


def test_dev_lock_path_inconsistency_and_no_candidate():
    devp = REPO / "docs" / "lineage_liveaddr.dev.lock"
    decp = REPO / "docs" / "lineage_liveaddr.decision.lock"
    assert _sha(devp) == "53b0dfd33f3787fc6b5a4e3b55c2dd3f835a7b4350addc94bd2f63ed00d7d8f6"
    assert _sha(decp) == "6f2c27c2729c1640c3b37a6381744bde0becd1ba45d7cf6ff10862fc1b1eaa2d"
    assert json.loads(PREREG.read_text())["frozen_neural_sha"] == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "129fdd863b1c3948978273a85e4885f204990f91"
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["decision_code"] == "liveaddr_canonical_path_inconsistency"
    assert dev["candidate_v40_lock"] is False
    assert dec["candidate_v40_lock"] is False
    assert dec["later_learning"] == "not_exercised"
    flags = dev["phase_flags"]
    assert flags["unique_failed_classes"] == ["canonical_path_inconsistency"]
    assert flags["always_joint_read"] == "same_retrieval_W_covers_live_address"
    assert flags["always_joint_retrieval_unchanged"] is True
    assert int(flags["n_stems_v37_fallback_w_equal"]) == 4
    assert flags["later_learning"] == "not_exercised"
    assert flags["contradict"] == "jointly_feasible_atomic_apply"
    fb = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|fallback_joint"][0]
    aj = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|always_joint"][0]
    v37 = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|v37"][0]
    assert int(fb["n_store_violations"]) == 0
    assert int(fb["n_probe_correct_tm040"]) == 7
    assert int(aj["n_probe_correct_tm040"]) == 8
    assert v37["w_hash"] == fb["w_hash"]
    assert aj["w_hash"] != fb["w_hash"]
    fail = [r for r in fb["cues"] if r["cue"] == "s_294555646"][0]
    assert fail["canonical"]["path"] == "episodic_completed"
    assert fail["canonical"]["familiar"] is True
    assert int(fail["canonical"]["slot"]) == 2
    assert fail["addr_equals_stored_p1"] is True
    assert fail["addr_equals_live_p1"] is False
    assert fail["canonical"]["ranking_ok"] is True
    assert fail["live_tm040"]["ranking_ok"] is False
    assert fail["counterfactual_stored"]["ranking_ok"] is True
    assert fail["counterfactual_live"]["ranking_ok"] is True
    assert fail["fail_class"] == "canonical_path_inconsistency"
    assert fail["live_tm040"]["winner"] == "h_679764572"
