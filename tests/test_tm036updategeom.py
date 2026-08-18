"""TM036 update-geometry freeze tests. No DEV execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_updategeom.prereg.lock"
ISO = REPO / "docs" / "lineage_updategeom.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_updategeom_contract.md"
RUNNER = REPO / "experiments" / "run_tm036updategeom.py"
TM035_DEC = REPO / "docs" / "lineage_creditsplit.decision.lock"
TM035_DEV = REPO / "docs" / "lineage_creditsplit.dev.lock"
TM035_RUNNER = REPO / "experiments" / "run_tm035creditsplit.py"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "2c24982d66b4cd0d51e312c61adc049f3e6064d1bf3bc9063109e6ed8fcfc7c6"
HISTORICAL_TM035_DEV_SHA = "1b052c569d1276d78395ae5236c03994b4cf2ee84219080fb35cba54f0d2cda6"
HISTORICAL_TM035_DEC_SHA = "5d7353ab9831229b59dd31ef626727dfb309a0f2fb9a0ddda1a656fbdd9be342"
HISTORICAL_TM035_RUNNER_SHA = "52a5f32cf690fd3efde3e594fcb59f62970d78bf35d2939933d25d42be4c8ca2"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
FROZEN_RUNNER_SHA = "fdcd11d376d22965670f8bb4dcd411720d2d759c953933dd278b64a4e4c6c927"
HISTORICAL_DEV_SHA = "885c6b7b1d6fad934996d849dd84fd6e4c46fc2644c37257a6f18eb6f8784183"
HISTORICAL_DEC_SHA = "0a029bf7cd0cffec37707ad5ce65e7c5c161701d279e6aac26fb9d5684b8601a"
DEV = REPO / "docs" / "lineage_updategeom.dev.lock"
DEC = REPO / "docs" / "lineage_updategeom.decision.lock"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_and_historical_locks():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 40
    assert p["expected_n_worlds"] == 8
    assert p["hard_budget_passes"] == 16
    assert p["fit_44_row_updates"] is False
    assert p["neural_edit_authorized"] is False
    assert p["v39_freeze_authorized"] is False
    assert p["oracle_is_diagnostic_only"] is True
    assert p["arms"] == ["interference", "native", "jacobi", "protect", "oracle"]
    assert 22222 not in p["registry_seeds"]
    assert p["tm035_dev_sha"] == HISTORICAL_TM035_DEV_SHA
    assert p["tm035_decision_sha"] == HISTORICAL_TM035_DEC_SHA
    iso = json.loads(ISO.read_text())
    assert iso["neural_edit_authorized"] is False
    assert iso["v39_freeze_authorized"] is False
    assert "one_shot_only_v39" in iso["refuse"]
    assert CONTRACT.is_file()
    assert p.get("frozen_runner_sha")
    assert _sha(RUNNER) == p["frozen_runner_sha"]
    if FROZEN_RUNNER_SHA:
        assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
        assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_historical_helpers_unedited():
    assert _sha(TM035_DEV) == HISTORICAL_TM035_DEV_SHA
    assert _sha(TM035_DEC) == HISTORICAL_TM035_DEC_SHA
    assert _sha(TM035_RUNNER) == HISTORICAL_TM035_RUNNER_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    src = RUNNER.read_text()
    assert "apply_arm" not in src
    assert "from experiments.run_tm035creditsplit import" in src
    assert "from experiments.run_tm032restsplit import clone_plastic, eta_act, live_probes, stored_rows" in src


def test_half_mode_untouched():
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2, EPISODE_REPLAY_EPOCHS

    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_ids_and_routes():
    from experiments.run_tm036updategeom import _decision, arm_fixes, classify_world, expected_cell_ids, homogeneous_hard_margin

    ids = expected_cell_ids()
    assert len(ids) == 40
    assert ids[0] == "geometry|c8|A_then_B|reg0|interference"
    assert ids[4] == "geometry|c8|A_then_B|reg0|oracle"
    assert ids[-1] == "geometry|c8|B_then_A|reg3|oracle"
    ok = {"n_store_violations": 0, "live_ranking_ok": True, "n_probe_correct": 8}
    fail = {"n_store_violations": 3, "live_ranking_ok": False, "n_probe_correct": 6}
    assert arm_fixes(ok) is True
    assert arm_fixes(fail) is False
    assert classify_world({"native": True, "jacobi": True, "protect": True, "oracle": True}) == "native_already_converged"
    assert classify_world({"native": False, "jacobi": True, "protect": True, "oracle": True}) == "jacobi_sufficient"
    assert classify_world({"native": False, "jacobi": False, "protect": True, "oracle": True}) == "protect_sufficient"
    assert classify_world({"native": False, "jacobi": False, "protect": False, "oracle": True}) == "oracle_only"
    assert classify_world({"native": False, "jacobi": False, "protect": False, "oracle": False}) == "capacity_wall"
    code, _t, fl = _decision(["native_already_converged"] * 8)
    assert code == "updategeom_no_native_fail"
    assert fl["v39_freeze"] is False
    code2, _t2, fl2 = _decision(["jacobi_sufficient", "jacobi_sufficient"] + ["native_already_converged"] * 6)
    assert code2 == "updategeom_jacobi_sufficient"
    code3, _t3, _f3 = _decision(["protect_sufficient"] * 2 + ["native_already_converged"] * 6)
    assert code3 == "updategeom_protect_sufficient"
    code4, _t4, _f4 = _decision(["oracle_only"] * 2 + ["native_already_converged"] * 6)
    assert code4 == "updategeom_oracle_only"
    code5, _t5, _f5 = _decision(["capacity_wall"] * 2 + ["native_already_converged"] * 6)
    assert code5 == "updategeom_capacity_wall"
    code6, _t6, _f6 = _decision(["jacobi_sufficient", "protect_sufficient"] + ["native_already_converged"] * 6)
    assert code6 == "updategeom_mixed_routes"
    rng = np.random.default_rng(0)
    xa = rng.normal(size=(4, 8))
    xa = xa / np.linalg.norm(xa, axis=1, keepdims=True)
    xb = -xa + 0.01 * rng.normal(size=xa.shape)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    X = np.concatenate([xa, xb], axis=0)
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    fit = homogeneous_hard_margin(X, y)
    assert fit["status"] == "optimal"
    assert float(fit["gamma_star"]) > 0.01


def test_smoke():
    from experiments.run_tm036updategeom import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["n_arms"] == 5
    assert out["arms"] == ["interference", "native", "jacobi", "protect", "oracle"]
    assert out["clone_matched"] is True
    assert out["probes_on_another_clone"] is True
    assert out["hard_budget_passes"] == 16
    assert len(out["interference_slots"]) >= 1


def test_dev_lock_first_match_and_no_v39():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "f27b618894c3a9ceba1dd9eafa06eeddb13e82eb"
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["decision_code"] == "updategeom_oracle_only"
    assert dev["fit_44_row_updates"] is False
    assert dev["hard_budget_passes"] == 16
    assert dev["v39_freeze"] is False
    assert dev["phase_flags"]["n_diagnostic"] == 2
    assert set(dev["phase_flags"]["diagnostic_routes"]) == {"oracle_only"}
    native = [c for c in dev["cells"] if c["id"] == "geometry|c8|A_then_B|reg1|native"][0]
    jacobi = [c for c in dev["cells"] if c["id"] == "geometry|c8|A_then_B|reg1|jacobi"][0]
    protect = [c for c in dev["cells"] if c["id"] == "geometry|c8|A_then_B|reg1|protect"][0]
    oracle = [c for c in dev["cells"] if c["id"] == "geometry|c8|A_then_B|reg1|oracle"][0]
    inter = [c for c in dev["cells"] if c["id"] == "geometry|c8|A_then_B|reg1|interference"][0]
    assert native["fixed"] is False
    assert native["broke_slots"] == [0, 4, 6]
    assert int(native["process"]["n_awake_updates"]) == 122
    assert jacobi["fixed"] is False
    assert jacobi["broke_slots"] == [1, 3, 5]
    assert protect["fixed"] is False
    assert protect["previously_correct_broke"] is False
    assert int(protect["n_store_violations"]) == 1
    assert int(protect["n_probe_correct"]) == 7
    assert oracle["fixed"] is True
    assert oracle["process"]["status"] == "feasible"
    assert float(oracle["process"]["gamma_star"]) > 0.01
    assert int(inter["process"]["n_negative_offdiag"]) == 32
    assert dec["decision"]["code"] == "updategeom_oracle_only"
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert dec["v39_freeze"] is False
