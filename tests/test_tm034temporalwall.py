"""TM034 temporal-wall freeze tests. No DEV execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_temporalwall.prereg.lock"
ISO = REPO / "docs" / "lineage_temporalwall.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_temporalwall_contract.md"
RUNNER = REPO / "experiments" / "run_tm034temporalwall.py"
CLOSE = REPO / "docs" / "lineage_adaptrehearse.closure.lock"
TM033_DEV = REPO / "docs" / "lineage_adaptrehearse.dev.lock"
TM033_DEC = REPO / "docs" / "lineage_adaptrehearse.decision.lock"
TM033_RUNNER = REPO / "experiments" / "run_tm033adaptrehearse.py"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "5e5e01e0c2a94f34854a5976294f5b3061d819277de9852840d64349e00432d6"
HISTORICAL_TM033_DEV_SHA = "a049f5fc44e921e77b762e6da9fd0c19285b0b6143beab758e26a22ace337375"
HISTORICAL_TM033_DEC_SHA = "cc22cbe82378164d61793b14727639ace35dca7fc95f5f5fc824bbc27ce081e6"
HISTORICAL_TM033_RUNNER_SHA = "91d8074d74b3724bd38e69b1d3860ea8ad0ba9835eb615a513cc9fcf620f5a49"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
CLOSURE_SHA = "a8ea4f6aa14158575cf9ba31a0ee3521e859b33ceaeb693fec282798dfa1abbe"
FROZEN_RUNNER_SHA = "f097b6e2685d9b4ffe10854ced7ed0418a10f49ded5ef8a6815e873ed137415c"
HISTORICAL_DEV_SHA = "084bad8780674a00a75848f9ef0c7443a55aea89f4f989406f0a5607d3666ffe"
HISTORICAL_DEC_SHA = "31943d48ac6170e1d60551c537418f3f09fefa875ad23941e86cea524e99e625"
DEV = REPO / "docs" / "lineage_temporalwall.dev.lock"
DEC = REPO / "docs" / "lineage_temporalwall.decision.lock"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_v38_closed_unchanged():
    c = json.loads(CLOSE.read_text())
    assert c["v38_closed"] is True
    assert c["cortex_candidate_v38"] is False
    assert c["plateau_relaxed"] is False
    assert c["hard_budget_passes_unchanged"] == 16
    assert c["fit_44_row_updates"] is False
    assert c["leftover_debt_after_rest"] == 0
    assert c["first_match"] == "adaptrehearse_core_acquire_fail"
    assert c["compute_conservation_invalidated_experiment"] is False
    assert "plateau_relaxation" in c["refuse"]
    assert _sha(CLOSE) == CLOSURE_SHA
    assert _sha(TM033_DEV) == HISTORICAL_TM033_DEV_SHA
    assert _sha(TM033_DEC) == HISTORICAL_TM033_DEC_SHA
    assert _sha(TM033_RUNNER) == HISTORICAL_TM033_RUNNER_SHA


def test_prereg_and_historical_locks():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 40
    assert p["hard_budget_passes"] == 16
    assert p["fit_44_row_updates"] is False
    assert p["plateau_relaxed"] is False
    assert p["neural_edit_authorized"] is False
    assert p["v38_repair_authorized"] is False
    assert p["trace_arms"] == ["v37_awake_cap", "adaptive_violation", "fixed_extra_replay"]
    assert p["rescue_arms"] == ["none", "tm032_awake_only"]
    assert p["tm033_dev_sha"] == HISTORICAL_TM033_DEV_SHA
    assert p["tm033_decision_sha"] == HISTORICAL_TM033_DEC_SHA
    assert p["tm032_dev_sha"] == HISTORICAL_TM032_DEV_SHA
    assert p["tm033_closure_sha"] == CLOSURE_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    iso = json.loads(ISO.read_text())
    assert iso["neural_edit_authorized"] is False
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_half_mode_untouched():
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2, EPISODE_REPLAY_EPOCHS

    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_ids_and_routes():
    from experiments.run_tm034temporalwall import _decision, expected_cell_ids, route_world, summarize_trace

    ids = expected_cell_ids()
    assert len(ids) == 40
    assert ids[0] == "acquire_trace|c8|A_then_B|reg0|v37_awake_cap"
    assert ids[4] == "rescue|c8|A_then_B|reg0|tm032_awake_only"
    fail = {"n_store_violations": 3, "live_ranking_ok": False, "n_probe_correct": 6}
    ok = {"n_store_violations": 0, "live_ranking_ok": True, "n_probe_correct": 8}
    assert route_world(ok, ok) == "v37_already_converged"
    assert route_world(fail, ok) == "tm032_rescue_holds"
    assert route_world(fail, fail) == "tm032_rescue_fails"
    flags = summarize_trace(
        [
            {"credit_index": 0, "n_violations": 1, "violating_slots": [0]},
            {"credit_index": 1, "n_violations": 0, "violating_slots": []},
            {"credit_index": 2, "n_violations": 2, "violating_slots": [0, 2]},
        ],
        [
            {
                "n_updates": 1,
                "broke_slots": [2],
                "fixed_slots": [],
                "margin_improved_still_violating": [],
                "count_dropped": False,
            }
        ],
    )
    assert flags["rebreak_after_zero"] is True
    assert flags["cross_row_break"] is True
    assert flags["first_incorrect_credit_by_slot"]["2"] == 2
    code, _t, fl = _decision(["v37_already_converged"] * 8, [])
    assert code == "temporalwall_no_v37_acquire_fail"
    code2, _t2, _f2 = _decision(["tm032_rescue_fails"] * 2 + ["v37_already_converged"] * 6, [])
    assert code2 == "temporalwall_tm032_rescue_fails"
    traces = [
        {
            "id": "acquire_trace|c8|A_then_B|reg1|adaptive_violation",
            "kind": "acquire_trace",
            "trace_flags": {"rebreak_after_zero": True, "cross_row_break": False, "margin_without_count": False},
        }
    ]
    code3, _t3, fl3 = _decision(["tm032_rescue_holds"] * 2 + ["v37_already_converged"] * 6, traces)
    assert code3 == "temporalwall_temporal_interference"
    assert fl3["n_rebreak"] == 1


def test_smoke():
    from experiments.run_tm034temporalwall import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["n_credits"] == 2
    assert out["hard_budget_passes"] == 16


def test_dev_lock_first_match_and_rescue():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "b781696da7d4998fc17676cefdb05a725540e03b"
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["decision_code"] == "temporalwall_temporal_interference"
    assert dev["fit_44_row_updates"] is False
    assert dev["hard_budget_passes"] == 16
    assert set(dev["phase_flags"]["routes"]) == {"v37_already_converged", "tm032_rescue_holds"}
    assert "tm032_rescue_fails" not in dev["phase_flags"]["routes"]
    dec_ok = [c for c in dev["cells"] if c["id"] == "rescue|c8|A_then_B|reg1|tm032_awake_only"][0]
    assert dec_ok["fixed"] is True
    assert int(dec_ok["process"]["n_updates"]) == 44
    v37 = [c for c in dev["cells"] if c["id"] == "acquire_trace|c8|A_then_B|reg1|v37_awake_cap"][0]
    assert v37["trace_flags"]["rebreak_after_zero"] is True
    assert v37["trace_flags"]["zero_after_credits"] == [0, 1, 2, 3, 4, 5, 6]
    assert v37["trace_flags"]["first_incorrect_credit_by_slot"] == {"0": 7, "4": 7, "6": 7}
    assert dec["decision"]["code"] == "temporalwall_temporal_interference"
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert dec["v38_repair"] is False

