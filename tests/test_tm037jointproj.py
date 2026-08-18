"""TM037 joint-projection freeze tests. Neural controller required after freeze push."""

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
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_jointproj.prereg.lock"
ISO = REPO / "docs" / "lineage_jointproj.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_jointproj_contract.md"
RUNNER = REPO / "experiments" / "run_tm037jointproj.py"
V39_PREREG = REPO / "docs" / "cortex_v39.prereg.lock"
V39_ISO = REPO / "docs" / "cortex_v39.isolation.lock"
V39_AMD = REPO / "docs" / "cortex_v39_architecture_amendment.md"
TM036_DEC = REPO / "docs" / "lineage_updategeom.decision.lock"
TM036_DEV = REPO / "docs" / "lineage_updategeom.dev.lock"
TM036_RUNNER = REPO / "experiments" / "run_tm036updategeom.py"
TM035_DEC = REPO / "docs" / "lineage_creditsplit.decision.lock"
TM035_DEV = REPO / "docs" / "lineage_creditsplit.dev.lock"
TM035_RUNNER = REPO / "experiments" / "run_tm035creditsplit.py"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "f1baa82f81d6e3cf326d25b059c0d07d828863213c7497db716677a398c42dd4"
HISTORICAL_TM036_DEV_SHA = "885c6b7b1d6fad934996d849dd84fd6e4c46fc2644c37257a6f18eb6f8784183"
HISTORICAL_TM036_DEC_SHA = "0a029bf7cd0cffec37707ad5ce65e7c5c161701d279e6aac26fb9d5684b8601a"
HISTORICAL_TM036_RUNNER_SHA = "fdcd11d376d22965670f8bb4dcd411720d2d759c953933dd278b64a4e4c6c927"
HISTORICAL_TM035_DEV_SHA = "1b052c569d1276d78395ae5236c03994b4cf2ee84219080fb35cba54f0d2cda6"
HISTORICAL_TM035_DEC_SHA = "5d7353ab9831229b59dd31ef626727dfb309a0f2fb9a0ddda1a656fbdd9be342"
HISTORICAL_TM035_RUNNER_SHA = "52a5f32cf690fd3efde3e594fcb59f62970d78bf35d2939933d25d42be4c8ca2"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
V39_ISO_SHA = "558a20172b1316433e2ddaf0c6eb7cdc89f97cec7326c6d97babb088f3ec6020"
V39_PREREG_SHA = "54d67b69c94ce73f766d53333abf13e8fc8513b8147a46a37bea1701c7c491c8"
V39_AMD_SHA = "da712d13eb02d0268da8615b676dcf0e4b851107561afec602a69c9e4ed543c5"
FROZEN_RUNNER_SHA = "897144551ee418d8d7579b1e78b5871463c6cff3d664f716c4c9f142b69a3380"
HISTORICAL_DEV_SHA = "a21ff312947f3b503342fbe3df963ed079d7c71d446cb1425087b96317fcbe48"
HISTORICAL_DEC_SHA = "55e29766d53f37ffc72e825547df9db3642e3e6839d4eb85cff32e522d447d4f"
DEV = REPO / "docs" / "lineage_jointproj.dev.lock"
DEC = REPO / "docs" / "lineage_jointproj.decision.lock"
GENOME_TO_DICT_KEYS = {
    "n",
    "d_sym",
    "k_s",
    "d_body",
    "d_x",
    "p_connect",
    "t_max",
    "tau",
    "cos_thresh",
    "eta_pred",
    "eta_act",
    "beta",
    "clip",
    "seed_birth",
    "seed_registry",
    "seed_source",
    "seed_action",
    "seed_permute",
    "seed_motor",
    "dtype",
    "motor_persist_p",
    "act_score_mode",
    "actuator_proto_h_max",
    "episodic_act_recall",
    "act_recall_mode",
    "separator_matrix_sha",
    "body_setpoint",
    "ops",
    "op_cost",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_convexity_budget_and_oracle_ceiling():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 32
    assert p["expected_n_worlds"] == 8
    assert p["hard_budget_passes"] == 16
    assert p["fit_44_row_updates"] is False
    assert p["fitted_learning_rate"] is False
    assert p["neural_edit_before_runner_freeze"] is False
    assert p["oracle_is_diagnostic_only"] is True
    assert p["oracle_never_installed_in_neural_cortex"] is True
    assert p["episode_handles_in_act_routing"] is False
    assert p["convexity"]["ranking"] == "halfspace_in_W_act_query"
    assert p["convexity"]["geometric_gamma"] == "second_order_cone_scale_invariant"
    assert p["convexity"]["pa_projects"] == "supporting_halfspace_b_frozen_at_cycle_start"
    assert p["arms"] == ["v37", "pa_cyclic", "dykstra", "oracle"]
    assert 22222 not in p["registry_seeds"]
    assert p["tm036_dev_sha"] == HISTORICAL_TM036_DEV_SHA
    assert p["tm036_decision_sha"] == HISTORICAL_TM036_DEC_SHA
    iso = json.loads(ISO.read_text())
    assert iso["neural_edit_before_runner_freeze"] is False
    assert "claim_geometric_gamma_is_a_halfspace" in iso["refuse"]
    assert "install_oracle_W_star_in_neural_cortex" in iso["refuse"]
    assert "auto_cortex.candidate.v39.lock" in iso["refuse"]
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_historical_helpers_unedited():
    assert _sha(TM036_DEV) == HISTORICAL_TM036_DEV_SHA
    assert _sha(TM036_DEC) == HISTORICAL_TM036_DEC_SHA
    assert _sha(TM036_RUNNER) == HISTORICAL_TM036_RUNNER_SHA
    assert _sha(TM035_DEV) == HISTORICAL_TM035_DEV_SHA
    assert _sha(TM035_DEC) == HISTORICAL_TM035_DEC_SHA
    assert _sha(TM035_RUNNER) == HISTORICAL_TM035_RUNNER_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    p = json.loads(PREREG.read_text())
    assert p["tm036_runner_sha"] == HISTORICAL_TM036_RUNNER_SHA
    assert _sha(V39_ISO) == V39_ISO_SHA
    assert _sha(V39_PREREG) == V39_PREREG_SHA
    assert _sha(V39_AMD) == V39_AMD_SHA
    assert p["v39_prereg_sha"] == V39_PREREG_SHA
    src = RUNNER.read_text()
    assert "from experiments.run_tm036updategeom import apply_native, arm_fixes, closest_feasible_W, geometry_parent" in src
    assert "closest_feasible_W" not in (REPO / "three_memory" / "neural_cortex.py").read_text()


def test_half_mode_write_radius_and_genome_keys():
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_MARGIN_FLOOR == 0.01
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    g = GenomeConfig().to_dict()
    assert set(g) == GENOME_TO_DICT_KEYS
    assert "act_proj_arm" not in g
    assert "act_rehearse_arm" not in g


def test_ranking_is_halfspace_gamma_is_not():
    from experiments.run_tm037jointproj import geometric_gamma, pa_project_W, ranking_inner

    rng = np.random.default_rng(7)
    W = rng.normal(size=(8, 6))
    d = rng.normal(size=8)
    d = d / np.linalg.norm(d)
    x = rng.normal(size=6)
    x = x / np.linalg.norm(x)
    inner = ranking_inner(W, d, x)
    inner2 = ranking_inner(2.0 * W, d, x)
    assert abs(inner2 - 2.0 * inner) < 1e-12
    g0 = geometric_gamma(W, d, x)
    g2 = geometric_gamma(2.0 * W, d, x)
    assert abs(g2 - g0) < 1e-12
    tau = float(ACT_MARGIN_FLOOR)
    b = tau * float(np.linalg.norm(W.T @ d))
    # On the true cone, λW stays feasible. The frozen-b halfspace is affine, not that cone.
    Wp = 0.5 * W
    gamma_half = geometric_gamma(Wp, d, x)
    assert abs(gamma_half - g0) < 1e-12
    inner_half = ranking_inner(Wp, d, x)
    if abs(inner - b) < 1e-9:
        assert inner_half < b - 1e-12
    W_star, hit = pa_project_W(np.zeros_like(W), d, x, b)
    if b > 0:
        assert hit is True
        assert abs(ranking_inner(W_star, d, x) - b) < 1e-10


def test_ids_and_routes():
    from experiments.run_tm037jointproj import _decision, classify_world, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 32
    assert ids[0] == "acquire|c8|A_then_B|reg0|v37"
    assert ids[3] == "acquire|c8|A_then_B|reg0|oracle"
    assert ids[-1] == "acquire|c8|B_then_A|reg3|oracle"
    assert classify_world({"v37": True, "pa_cyclic": True, "dykstra": True, "oracle": True}) == "v37_already_converged"
    assert classify_world({"v37": False, "pa_cyclic": True, "dykstra": True, "oracle": True}) == "pa_sufficient"
    assert classify_world({"v37": False, "pa_cyclic": False, "dykstra": True, "oracle": True}) == "dykstra_required"
    assert classify_world({"v37": False, "pa_cyclic": False, "dykstra": False, "oracle": True}) == "oracle_only"
    assert classify_world({"v37": False, "pa_cyclic": False, "dykstra": False, "oracle": False}) == "capacity_wall"
    code, _t, fl = _decision(["v37_already_converged"] * 8)
    assert code == "jointproj_no_v37_fail"
    assert fl["candidate_v39_lock"] is False
    code2, _t2, _f2 = _decision(["pa_sufficient"] * 2 + ["v37_already_converged"] * 6)
    assert code2 == "jointproj_pa_sufficient"
    code3, _t3, _f3 = _decision(["dykstra_required"] * 2 + ["v37_already_converged"] * 6)
    assert code3 == "jointproj_dykstra_required"
    code4, _t4, _f4 = _decision(["oracle_only"] * 2 + ["v37_already_converged"] * 6)
    assert code4 == "jointproj_oracle_only"
    code5, _t5, _f5 = _decision(["capacity_wall"] * 2 + ["v37_already_converged"] * 6)
    assert code5 == "jointproj_capacity_wall"
    code6, _t6, _f6 = _decision(["pa_sufficient", "dykstra_required"] + ["v37_already_converged"] * 6)
    assert code6 == "jointproj_mixed_routes"


def test_smoke():
    from experiments.run_tm037jointproj import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["n_arms"] == 4
    assert out["arms"] == ["v37", "pa_cyclic", "dykstra", "oracle"]
    assert out["clone_matched"] is True
    assert out["probes_on_another_clone"] is True
    assert out["hard_budget_passes"] == 16
    assert out["pa_fitted_lr"] is False
    assert hasattr(NeuralCortex, "set_act_rehearse_arm")
    assert hasattr(NeuralCortex, "set_act_proj_arm")
    assert out["neural_ready"] is True


def test_v39_default_off_rejects_oracle_and_checkpoints():
    from three_memory.neural_cortex import ACT_PROJ_ARMS, ACT_PROJ_OFF, ACT_PROJ_PA, ACT_PROJ_DYKSTRA

    ag = NeuralCortex()
    assert ag._act_proj_arm == ACT_PROJ_OFF
    assert ag._act_proj_arm_active() is False
    assert "oracle" not in ACT_PROJ_ARMS
    try:
        ag.set_act_proj_arm("oracle")
        raise AssertionError("oracle must not be a proj arm")
    except ValueError:
        pass
    ag.set_act_proj_arm(ACT_PROJ_DYKSTRA)
    ag._act_proj_corrections = {"0|h|r": np.ones((2, 3), dtype=np.float64)}
    snap = ag.checkpoint()
    assert snap["act_proj_arm"] == ACT_PROJ_DYKSTRA
    assert "0|h|r" in snap["act_proj_corrections"]
    twin = NeuralCortex()
    twin.load_checkpoint(snap)
    assert twin._act_proj_arm == ACT_PROJ_DYKSTRA
    assert np.allclose(twin._act_proj_corrections["0|h|r"], 1.0)
    missing = dict(snap)
    missing.pop("act_proj_arm")
    missing.pop("act_proj_corrections")
    twin2 = NeuralCortex()
    twin2.load_checkpoint(missing)
    assert twin2._act_proj_arm == ACT_PROJ_OFF
    assert twin2._act_proj_corrections == {}
    src = Path(REPO / "three_memory" / "neural_cortex.py").read_text()
    assert "closest_feasible_W" not in src
    assert ACT_PROJ_PA in src
    credit = NeuralCortex._credit_act_p1_episode
    import inspect

    body = inspect.getsource(credit)
    assert body.index("_act_proj_arm_active") < body.index("_apply_act_query_update")
    assert body.index("_apply_act_query_update") < body.index("_run_awake_rehearsal_burst")


def test_dev_lock_first_match_and_no_candidate():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "6f4e052a768db8dae2c271b94f02097927e2ca06"
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["decision_code"] == "jointproj_oracle_only"
    assert dev["fit_44_row_updates"] is False
    assert dev["fitted_learning_rate"] is False
    assert dev["hard_budget_passes"] == 16
    assert dev["candidate_v39_lock"] is False
    assert dev["oracle_installed_in_organism"] is False
    assert dev["phase_flags"]["n_diagnostic"] == 2
    assert set(dev["phase_flags"]["diagnostic_routes"]) == {"oracle_only"}
    native = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|v37"][0]
    pa = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|pa_cyclic"][0]
    dyk = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|dykstra"][0]
    oracle = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|oracle"][0]
    assert native["fixed"] is False
    assert native["broke_slots"] == [0, 4, 6]
    assert int(native["process"]["n_awake_updates"]) == 122
    assert pa["fixed"] is False
    assert pa["broke_slots"] == [6]
    assert int(pa["n_probe_correct"]) == 7
    assert pa["process"]["fitted_learning_rate"] is False
    assert pa["process"]["corrections"] is False
    assert dyk["fixed"] is False
    assert dyk["broke_slots"] == [6]
    assert dyk["process"]["corrections"] is True
    assert oracle["fixed"] is True
    assert oracle["process"]["status"] == "feasible"
    assert float(oracle["process"]["gamma_star"]) > 0.01
    assert dec["decision"]["code"] == "jointproj_oracle_only"
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert dec["candidate_v39_lock"] is False
    assert not (REPO / "docs" / "cortex.candidate.v39.lock").exists()

