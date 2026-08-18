"""TM038 cone-split freeze tests. No DEV execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from three_memory.neural_cortex import (
    ACT_MARGIN_FLOOR,
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    EPISODE_REPLAY_EPOCHS,
    PROTO_EPS,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_conesplit.prereg.lock"
ISO = REPO / "docs" / "lineage_conesplit.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_conesplit_contract.md"
RUNNER = REPO / "experiments" / "run_tm038conesplit.py"
TM037_DEC = REPO / "docs" / "lineage_jointproj.decision.lock"
TM037_DEV = REPO / "docs" / "lineage_jointproj.dev.lock"
TM037_RUNNER = REPO / "experiments" / "run_tm037jointproj.py"
TM036_DEC = REPO / "docs" / "lineage_updategeom.decision.lock"
TM036_DEV = REPO / "docs" / "lineage_updategeom.dev.lock"
TM036_RUNNER = REPO / "experiments" / "run_tm036updategeom.py"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "0278191e6e296ab9d2434e9cd079748aba75ff30e15d78b53585d9f7481d4e87"
HISTORICAL_TM037_DEV_SHA = "a21ff312947f3b503342fbe3df963ed079d7c71d446cb1425087b96317fcbe48"
HISTORICAL_TM037_DEC_SHA = "55e29766d53f37ffc72e825547df9db3642e3e6839d4eb85cff32e522d447d4f"
HISTORICAL_TM037_RUNNER_SHA = "897144551ee418d8d7579b1e78b5871463c6cff3d664f716c4c9f142b69a3380"
HISTORICAL_TM036_DEV_SHA = "885c6b7b1d6fad934996d849dd84fd6e4c46fc2644c37257a6f18eb6f8784183"
HISTORICAL_TM036_DEC_SHA = "0a029bf7cd0cffec37707ad5ce65e7c5c161701d279e6aac26fb9d5684b8601a"
HISTORICAL_TM036_RUNNER_SHA = "fdcd11d376d22965670f8bb4dcd411720d2d759c953933dd278b64a4e4c6c927"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
FROZEN_RUNNER_SHA = "b06088386512c8d199a218e6de08c1909cf65895add979f8d71b3d788480f02b"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_geometry_budget_and_degenerate_law():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 32
    assert p["hard_budget_passes"] == 16
    assert p["diagnostic_cycle_cap"] == 256
    assert p["diagnostic_cycle_cap_is_not_fitted_from_tm037"] is True
    assert p["fitted_degenerate_epsilon"] is False
    assert p["degenerate_zero_normal"] == "skip_if_u_norm_le_PROTO_EPS"
    assert p["open_set_projection_at_origin"] is False
    assert p["nudge_zero_normal_along_x"] is False
    assert p["soc_adds_separate_ranking_halfspaces"] is False
    assert p["neural_edit_authorized"] is False
    assert p["v40_freeze_authorized"] is False
    assert p["oracle_is_diagnostic_only"] is True
    assert p["call_v39_set_act_proj_arm_for_soc"] is False
    assert p["arms"] == ["lin_dykstra_conv", "soc_16", "soc_conv", "oracle"]
    assert 22222 not in p["registry_seeds"]
    assert p["tm037_dev_sha"] == HISTORICAL_TM037_DEV_SHA
    iso = json.loads(ISO.read_text())
    assert iso["neural_edit_authorized"] is False
    assert "fitted_degenerate_epsilon" in iso["refuse"]
    assert "v40_architecture" in iso["refuse"]
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_historical_helpers_unedited():
    assert _sha(TM037_DEV) == HISTORICAL_TM037_DEV_SHA
    assert _sha(TM037_DEC) == HISTORICAL_TM037_DEC_SHA
    assert _sha(TM037_RUNNER) == HISTORICAL_TM037_RUNNER_SHA
    assert _sha(TM036_DEV) == HISTORICAL_TM036_DEV_SHA
    assert _sha(TM036_DEC) == HISTORICAL_TM036_DEC_SHA
    assert _sha(TM036_RUNNER) == HISTORICAL_TM036_RUNNER_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src
    assert "closest_feasible_W" not in (REPO / "three_memory" / "neural_cortex.py").read_text()


def test_half_mode_and_organism_eps():
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_MARGIN_FLOOR == 0.01
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert PROTO_EPS == 1e-12


def test_soc_projector_and_degenerate_skip():
    from experiments.run_tm038conesplit import project_soc_W, project_u_onto_closed_soc

    tau = float(ACT_MARGIN_FLOOR)
    rng = np.random.default_rng(3)
    x = rng.normal(size=8)
    x = x / np.linalg.norm(x)
    inside = 2.0 * x
    u2, st = project_u_onto_closed_soc(inside, x, tau)
    assert st == "already"
    assert np.allclose(u2, inside)
    gamma_in = float(np.dot(inside, x) / np.linalg.norm(inside))
    assert gamma_in >= tau
    perp = rng.normal(size=8)
    perp = perp - float(np.dot(perp, x)) * x
    perp = perp / np.linalg.norm(perp)
    u3, st3 = project_u_onto_closed_soc(perp, x, tau)
    assert st3 == "boundary"
    g = float(np.dot(u3, x) / np.linalg.norm(u3))
    assert abs(g - tau) < 1e-10
    assert float(np.dot(u3, x)) > 0.0
    polar = -10.0 * x
    u0, st0 = project_u_onto_closed_soc(polar, x, tau)
    assert st0 == "origin"
    assert float(np.linalg.norm(u0)) == 0.0
    z = np.zeros(8)
    u_z, st_z = project_u_onto_closed_soc(z, x, tau)
    assert st_z == "origin"
    W = rng.normal(size=(6, 8))
    d = rng.normal(size=6)
    d = d / np.linalg.norm(d)
    W0 = W.copy()
    Wp, hit, status = project_soc_W(W0, d, x, tau)
    if float(np.linalg.norm(W0.T @ d)) <= PROTO_EPS:
        assert hit is False
        assert status == "degenerate_skip"
        assert np.allclose(Wp, W0)
    Wd = W.copy()
    # force a zero normal
    u = Wd.T @ d
    Wd = Wd - np.outer(d, u) / float(np.dot(d, d))
    assert float(np.linalg.norm(Wd.T @ d)) <= 1e-10
    W2, hit2, st2 = project_soc_W(Wd, d, x, tau)
    assert st2 == "degenerate_skip"
    assert hit2 is False
    assert np.allclose(W2, Wd)


def test_ids_and_routes():
    from experiments.run_tm038conesplit import _decision, classify_world, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 32
    assert ids[0] == "acquire|c8|A_then_B|reg0|lin_dykstra_conv"
    assert ids[3] == "acquire|c8|A_then_B|reg0|oracle"
    ok = {"soc_16": True, "lin_dykstra_conv": True, "soc_conv": True, "oracle": True}
    fail = {"soc_16": False, "lin_dykstra_conv": False, "soc_conv": False, "oracle": False}
    assert classify_world(lin16=True, fixes=ok) == "lin16_already_converged"
    assert classify_world(lin16=False, fixes=ok) == "linearization_causal"
    assert classify_world(lin16=False, fixes={**fail, "lin_dykstra_conv": True, "oracle": True}) == "budget_causal"
    assert classify_world(lin16=False, fixes={**fail, "soc_conv": True, "oracle": True}) == "geometry_and_budget"
    assert classify_world(lin16=False, fixes={**fail, "oracle": True}) == "joint_socp_only"
    assert classify_world(lin16=False, fixes=fail) == "capacity_wall"
    code, _t, fl = _decision(["lin16_already_converged"] * 8)
    assert code == "conesplit_no_lin16_fail"
    assert fl["v40_freeze"] is False
    code2, _t2, _f2 = _decision(["linearization_causal"] * 2 + ["lin16_already_converged"] * 6)
    assert code2 == "conesplit_linearization_causal"
    code3, _t3, _f3 = _decision(["budget_causal"] * 2 + ["lin16_already_converged"] * 6)
    assert code3 == "conesplit_budget_causal"
    code4, _t4, _f4 = _decision(["geometry_and_budget"] * 2 + ["lin16_already_converged"] * 6)
    assert code4 == "conesplit_geometry_and_budget"
    code5, _t5, _f5 = _decision(["joint_socp_only"] * 2 + ["lin16_already_converged"] * 6)
    assert code5 == "conesplit_joint_socp_only"
    code6, _t6, _f6 = _decision(["capacity_wall"] * 2 + ["lin16_already_converged"] * 6)
    assert code6 == "conesplit_capacity_wall"
    code7, _t7, _f7 = _decision(["linearization_causal", "joint_socp_only"] + ["lin16_already_converged"] * 6)
    assert code7 == "conesplit_mixed_routes"


def test_smoke():
    from experiments.run_tm038conesplit import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["n_arms"] == 4
    assert out["arms"] == ["lin_dykstra_conv", "soc_16", "soc_conv", "oracle"]
    assert out["clone_matched"] is True
    assert out["hard_budget_passes"] == 16
    assert out["diagnostic_cycle_cap"] == 256
    assert out["soc_separate_ranking"] is False
    assert out["soc_fitted_eps"] is False
