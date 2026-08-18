"""TM035 credit-split freeze tests. No DEV execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_creditsplit.prereg.lock"
ISO = REPO / "docs" / "lineage_creditsplit.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_creditsplit_contract.md"
RUNNER = REPO / "experiments" / "run_tm035creditsplit.py"
TM034_DEC = REPO / "docs" / "lineage_temporalwall.decision.lock"
TM034_DEV = REPO / "docs" / "lineage_temporalwall.dev.lock"
TM034_RUNNER = REPO / "experiments" / "run_tm034temporalwall.py"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "249430e64d794562b93fa11f91e79916b9ee44f063df3d3717c9c48767e60e09"
HISTORICAL_TM034_DEV_SHA = "084bad8780674a00a75848f9ef0c7443a55aea89f4f989406f0a5607d3666ffe"
HISTORICAL_TM034_DEC_SHA = "31943d48ac6170e1d60551c537418f3f09fefa875ad23941e86cea524e99e625"
HISTORICAL_TM034_RUNNER_SHA = "f097b6e2685d9b4ffe10854ced7ed0418a10f49ded5ef8a6815e873ed137415c"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
FROZEN_RUNNER_SHA = "52a5f32cf690fd3efde3e594fcb59f62970d78bf35d2939933d25d42be4c8ca2"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_and_historical_locks():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 32
    assert p["expected_n_worlds"] == 8
    assert p["hard_budget_passes"] == 16
    assert p["fit_44_row_updates"] is False
    assert p["plateau_relaxed"] is False
    assert p["neural_edit_authorized"] is False
    assert p["v38_repair_authorized"] is False
    assert p["v39_freeze_authorized"] is False
    assert p["arms"] == ["write_only", "oneshot_only", "burst_only", "complete"]
    assert p["exclude_newly_written_slot"] is True
    assert p["complete_is_native_credit_act_p1_episode"] is True
    assert p["probes_on_another_clone"] is True
    assert p["pure_helpers_only"] is True
    assert 22222 not in p["registry_seeds"]
    assert p["tm034_dev_sha"] == HISTORICAL_TM034_DEV_SHA
    assert p["tm034_decision_sha"] == HISTORICAL_TM034_DEC_SHA
    assert p["tm032_dev_sha"] == HISTORICAL_TM032_DEV_SHA
    iso = json.loads(ISO.read_text())
    assert iso["neural_edit_authorized"] is False
    assert iso["v39_freeze_authorized"] is False
    assert "TM032_monkey_patch" in iso["refuse"]
    assert CONTRACT.is_file()
    assert p.get("frozen_runner_sha")
    assert _sha(RUNNER) == p["frozen_runner_sha"]
    if FROZEN_RUNNER_SHA:
        assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
        assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_historical_helpers_unedited():
    assert _sha(TM034_DEV) == HISTORICAL_TM034_DEV_SHA
    assert _sha(TM034_DEC) == HISTORICAL_TM034_DEC_SHA
    assert _sha(TM034_RUNNER) == HISTORICAL_TM034_RUNNER_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    src = RUNNER.read_text()
    assert "attach_pass_probe" not in src
    assert "apply_arm" not in src
    assert "from experiments.run_tm032restsplit import clone_plastic, live_probes, stored_rows" in src


def test_half_mode_untouched():
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2, EPISODE_REPLAY_EPOCHS

    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_ids_and_routes():
    from experiments.run_tm035creditsplit import _decision, classify_world, expected_cell_ids, broke_protected

    ids = expected_cell_ids()
    assert len(ids) == 32
    assert ids[0] == "split|c8|A_then_B|reg0|write_only"
    assert ids[3] == "split|c8|A_then_B|reg0|complete"
    assert ids[-1] == "split|c8|B_then_A|reg3|complete"
    broke, slots = broke_protected(
        protected=[0, 1, 2],
        new_slot=7,
        after_rows=[
            {"slot": 0, "store_violation": False},
            {"slot": 1, "store_violation": True},
            {"slot": 2, "store_violation": False},
            {"slot": 7, "store_violation": True},
        ],
    )
    assert broke is True
    assert slots == [1]
    broke2, slots2 = broke_protected(
        protected=[0, 4, 6],
        new_slot=7,
        after_rows=[
            {"slot": 0, "store_violation": True},
            {"slot": 4, "store_violation": True},
            {"slot": 6, "store_violation": True},
            {"slot": 7, "store_violation": True},
        ],
    )
    assert broke2 is True
    assert slots2 == [0, 4, 6]
    assert classify_world({"write_only": False, "oneshot_only": False, "burst_only": False, "complete": False}) == "no_rebreak"
    assert classify_world({"write_only": True, "oneshot_only": True, "burst_only": False, "complete": True}) == "write_causal"
    assert classify_world({"write_only": False, "oneshot_only": True, "burst_only": False, "complete": True}) == "oneshot_causal"
    assert classify_world({"write_only": False, "oneshot_only": False, "burst_only": True, "complete": True}) == "burst_causal"
    assert classify_world({"write_only": False, "oneshot_only": False, "burst_only": False, "complete": True}) == "interaction_only"
    assert classify_world({"write_only": False, "oneshot_only": True, "burst_only": True, "complete": True}) == "mixed_oneshot_and_burst"
    code, _t, fl = _decision(["no_rebreak"] * 8)
    assert code == "creditsplit_no_diagnostic"
    assert fl["v39_gate_open"] is False
    code2, _t2, fl2 = _decision(["oneshot_causal", "oneshot_causal"] + ["no_rebreak"] * 6)
    assert code2 == "creditsplit_oneshot_causal"
    assert fl2["v39_gate_open"] is True
    code3, _t3, fl3 = _decision(["oneshot_causal", "burst_causal"] + ["no_rebreak"] * 6)
    assert code3 == "creditsplit_mixed_oneshot_and_burst"
    assert fl3["v39_gate_open"] is False
    code4, _t4, _f4 = _decision(["write_causal"] * 2 + ["no_rebreak"] * 6)
    assert code4 == "creditsplit_write_causal"


def test_smoke():
    from experiments.run_tm035creditsplit import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["n_arms"] == 4
    assert out["complete_uses_native"] is True
    assert out["clone_matched"] is True
    assert out["probes_on_another_clone"] is True
    assert out["hard_budget_passes"] == 16
