"""TM033 adaptive-rehearsal freeze tests. Neural controller not required except skip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_adaptrehearse.prereg.lock"
ISO = REPO / "docs" / "lineage_adaptrehearse.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_adaptrehearse_contract.md"
RUNNER = REPO / "experiments" / "run_tm033adaptrehearse.py"
V38_PREREG = REPO / "docs" / "cortex_v38.prereg.lock"
V38_ISO = REPO / "docs" / "cortex_v38.isolation.lock"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "5fa2dac5200a6944b0a31530d57983d7dd61cbe84d69ace45ac361a93079a3ad"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
V38_ISO_SHA = "73543f2f67f6356e2218aa162cdf00db6beeea033a3cb177299efdf9237af866"
V38_PREREG_SHA = "de924a8df50f0b20c902bf3cd3f689e882a162df0d28fe33e509ac6cb074c510"
FROZEN_RUNNER_SHA = "91d8074d74b3724bd38e69b1d3860ea8ad0ba9835eb615a513cc9fcf620f5a49"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_budget_plateau_debt_and_limitation():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 24
    assert p["hard_budget_passes"] == 16
    assert p["fit_44_row_updates"] is False
    assert p["work_metric"] == "n_awake_updates"
    assert p["plateau"]["tolerance"] == 0
    assert p["plateau"]["unchanged_passes_to_stop"] == 1
    assert p["debt"]["rest_debit_uses"] == "rehearsal_pass_debt"
    assert p["debt"]["reset"] == "after_rest_replay"
    assert p["fixed_extra_replay"]["targeting"] == "violation_rows"
    assert p["fixed_extra_replay"]["plateau_stop"] is False
    assert p["fixed_extra_replay"]["rest_debit"] is False
    assert p["tm032_n_informative_splits"] == 2
    assert p["tm032_informative_replicate"] == "reg1"
    assert p["neural_edit_before_runner_freeze"] is False
    assert p["arms"] == ["v37_awake_cap", "adaptive_violation", "fixed_extra_replay"]
    assert "fit_tm032_44_row_updates_as_constant" in p["refuse"]
    assert "adaptrehearse_debt_integrity_fail" in {d["code"] for d in p["decision_ladder"]}
    assert "adaptrehearse_adaptive_control" in {d["code"] for d in p["decision_ladder"]}
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_tm032_locks_unedited():
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    p = json.loads(PREREG.read_text())
    assert p["tm032_decision_sha"] == HISTORICAL_TM032_DEC_SHA
    assert p["tm032_dev_sha"] == HISTORICAL_TM032_DEV_SHA
    v38p = json.loads(V38_PREREG.read_text())
    assert v38p["fit_44_row_updates"] is False
    assert v38p["hard_budget_passes"] == 16
    assert v38p["plateau"]["unchanged_passes_to_stop"] == 1
    assert _sha(V38_ISO) == V38_ISO_SHA
    assert _sha(V38_PREREG) == V38_PREREG_SHA


def test_half_mode_and_write_radius_untouched():
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2, EPISODE_REPLAY_EPOCHS

    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not hasattr(__import__("three_memory.neural_cortex", fromlist=["NeuralCortex"]).NeuralCortex, "set_act_rehearse_arm")


def test_expected_ids_and_routes():
    from experiments.run_tm033adaptrehearse import _decision, expected_cell_ids, route_triple

    ids = expected_cell_ids()
    assert len(ids) == 24
    fail = {
        "n_store_violations": 3,
        "live_ranking_ok": False,
        "n_probe_correct": 6,
        "n_awake_updates": 0,
        "targeting": "violation_rows",
    }
    ok_cheap = {
        "n_store_violations": 0,
        "live_ranking_ok": True,
        "n_probe_correct": 8,
        "n_awake_updates": 10,
        "targeting": "violation_rows",
    }
    ok_dear = {
        "n_store_violations": 0,
        "live_ranking_ok": True,
        "n_probe_correct": 8,
        "n_awake_updates": 44,
        "targeting": "violation_rows",
    }
    bundle_ad = dict(ok_cheap, targeting="all_rows")
    assert route_triple(ok_cheap, ok_cheap, ok_dear) == "v37_already_converged"
    assert route_triple(fail, ok_cheap, fail) == "adaptive_control"
    assert route_triple(fail, bundle_ad, fail) == "controller_bundle"
    assert route_triple(fail, ok_cheap, ok_dear) == "efficient_scheduling"
    assert route_triple(fail, ok_dear, ok_cheap) == "both_pass_compute_tied"
    assert route_triple(fail, fail, ok_dear) == "extra_compute_not_scheduler"
    assert route_triple(fail, fail, fail) == "both_fail"
    leftover = {
        "id": "acquire|c8|A_then_B|reg1|adaptive_violation",
        "arm": "adaptive_violation",
        "rehearsal_pass_debt_after_rest": 2,
        "rehearsal_update_debt_after_rest": 0,
        "awake_budget_exhausted": False,
    }
    code, _then, flags = _decision(["v37_already_converged"] * 8, [leftover])
    assert code == "adaptrehearse_debt_integrity_fail"
    assert flags["n_debt_remaining_after_rest"] == 1
    code2, _t2, flags2 = _decision(["v37_already_converged"] * 8, [])
    assert code2 == "adaptrehearse_no_v37_acquire_fail"
    assert flags2["n_diagnostic"] == 0
    code3, _t3, flags3 = _decision(["adaptive_control", "both_fail"], [])
    assert code3 == "adaptrehearse_mixed_routes"


def test_smoke_v37_arm():
    from experiments.run_tm033adaptrehearse import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["arm"] == "v37_awake_cap"
    assert out["hard_budget_passes"] == 16
    assert out["neural_ready"] is False
