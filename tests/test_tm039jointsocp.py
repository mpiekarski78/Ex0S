"""TM039 joint-SOCP freeze tests. Neural wiring required after freeze push."""

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
    PROTO_EPS,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_jointsocp.prereg.lock"
ISO = REPO / "docs" / "lineage_jointsocp.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_jointsocp_contract.md"
RUNNER = REPO / "experiments" / "run_tm039jointsocp.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
V40_PREREG = REPO / "docs" / "cortex_v40.prereg.lock"
V40_ISO = REPO / "docs" / "cortex_v40.isolation.lock"
V40_AMD = REPO / "docs" / "cortex_v40_architecture_amendment.md"
TM038_DEC = REPO / "docs" / "lineage_conesplit.decision.lock"
TM038_DEV = REPO / "docs" / "lineage_conesplit.dev.lock"
TM038_RUNNER = REPO / "experiments" / "run_tm038conesplit.py"
TM036_DEC = REPO / "docs" / "lineage_updategeom.decision.lock"
TM036_DEV = REPO / "docs" / "lineage_updategeom.dev.lock"
TM036_RUNNER = REPO / "experiments" / "run_tm036updategeom.py"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "86ea21ead0395860ca7284ec27546b7d438847eab7354d6c827679fd5a2daf5c"
HISTORICAL_TM038_DEV_SHA = "871be97be4fa888aec802a877847ea4e9831a09ce738cb9b82ddc8d3b02a972d"
HISTORICAL_TM038_DEC_SHA = "d06177fb93809ca90a9012dfdbd5d7af39ba0dc782db617786366612777e3426"
HISTORICAL_TM038_RUNNER_SHA = "b06088386512c8d199a218e6de08c1909cf65895add979f8d71b3d788480f02b"
HISTORICAL_TM036_DEV_SHA = "885c6b7b1d6fad934996d849dd84fd6e4c46fc2644c37257a6f18eb6f8784183"
HISTORICAL_TM036_DEC_SHA = "0a029bf7cd0cffec37707ad5ce65e7c5c161701d279e6aac26fb9d5684b8601a"
HISTORICAL_TM036_RUNNER_SHA = "fdcd11d376d22965670f8bb4dcd411720d2d759c953933dd278b64a4e4c6c927"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
V40_ISO_SHA = "6003436fd6a0a6e5cd9de10863f363b767d4ec4c13f7addefffa16e9d16c74d7"
V40_PREREG_SHA = "c60a5a55980c633453df47eac18126df81fdc9153c6a6a0dd7186f563ab6e3ac"
V40_AMD_SHA = "e4310c701006d7feda5b63d896a864f4f1b10e80c104d51d2af3448d2ea01fbf"
SOLVER_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
FROZEN_RUNNER_SHA = "e40304ee4ece2c834390094c9853d122bb89cb8fb18923aa348e052d146e288d"
HISTORICAL_DEV_SHA = "c2f5dae06f591c6e09a651122a4f893d2e4ccac45d42c5bf7965ba8b9891b21d"
HISTORICAL_DEC_SHA = "0275f20ebc9d0ff528b4edbf2767d21c69689b94f6559a25ce00eca9f7d1618d"
DEV = REPO / "docs" / "lineage_jointsocp.dev.lock"
DEC = REPO / "docs" / "lineage_jointsocp.decision.lock"
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


def test_prereg_solver_pins_and_no_candidate():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 24
    assert p["hard_budget_passes"] == 16
    assert p["not_an_exact_projector"] is True
    assert p["solver"] == "CLARABEL"
    assert p["cvxpy_version"] == "1.7.3"
    assert p["clarabel_version"] == "0.11.1"
    assert p["max_threads"] == 1
    assert p["partial_install"] is False
    assert p["nudge"] is False
    assert p["v40_candidate_authorized"] is False
    assert p["neural_edit_before_runner_freeze"] is False
    assert p["arms"] == ["v37", "fallback_joint", "always_joint"]
    assert 22222 not in p["registry_seeds"]
    iso = json.loads(ISO.read_text())
    assert "disguise_solver_as_local_plasticity" in iso["refuse"]
    assert "auto_cortex.candidate.v40.lock" in iso["refuse"]
    assert CONTRACT.is_file()
    v40 = json.loads(V40_PREREG.read_text())
    assert "later_learning_on_untouched_worlds" in v40["candidate_lock_requires"]
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA
    assert _sha(SOLVER) == SOLVER_SHA
    assert _sha(V40_PREREG) == V40_PREREG_SHA
    assert _sha(V40_ISO) == V40_ISO_SHA
    assert _sha(V40_AMD) == V40_AMD_SHA


def test_historical_helpers_unedited():
    assert _sha(TM038_DEV) == HISTORICAL_TM038_DEV_SHA
    assert _sha(TM038_DEC) == HISTORICAL_TM038_DEC_SHA
    assert _sha(TM038_RUNNER) == HISTORICAL_TM038_RUNNER_SHA
    assert _sha(TM036_DEV) == HISTORICAL_TM036_DEV_SHA
    assert _sha(TM036_DEC) == HISTORICAL_TM036_DEC_SHA
    assert _sha(TM036_RUNNER) == HISTORICAL_TM036_RUNNER_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src
    assert "closest_feasible_W" not in (REPO / "three_memory" / "neural_cortex.py").read_text()


def test_half_mode_genome_keys_and_pinned_solver():
    from three_memory.joint_socp import CLARABEL_VERSION, CVXPY_VERSION, assert_pinned_solver

    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_MARGIN_FLOOR == 0.01
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    g = GenomeConfig().to_dict()
    assert set(g) == GENOME_TO_DICT_KEYS
    assert "act_socp_arm" not in g
    pins = assert_pinned_solver()
    assert pins["cvxpy"] == CVXPY_VERSION
    assert pins["clarabel"] == CLARABEL_VERSION


def test_socp_rejects_zero_normal_and_lands_on_cone():
    from three_memory.joint_socp import solve_min_change_socp

    rng = np.random.default_rng(4)
    W0 = rng.normal(size=(6, 8))
    d = rng.normal(size=6)
    d = d / np.linalg.norm(d)
    x = rng.normal(size=8)
    x = x / np.linalg.norm(x)
    tau = float(ACT_MARGIN_FLOOR)
    out = solve_min_change_socp(W0, [{"d": d, "x": x}], tau, float(PROTO_EPS))
    assert out["status"] == "optimal"
    assert out["applied"] is False
    W = out["W"]
    u = W.T @ d
    g = float(np.dot(u, x) / np.linalg.norm(u))
    assert g >= tau - 1e-8
    assert float(np.linalg.norm(u)) > float(PROTO_EPS)
    # already feasible: numerical residual, not an exact projector
    out2 = solve_min_change_socp(W, [{"d": d, "x": x}], tau, float(PROTO_EPS))
    assert out2["status"] == "optimal"
    assert float(out2["frobenius_delta"]) < 1e-3
    u2 = out2["W"].T @ d
    g2 = float(np.dot(u2, x) / np.linalg.norm(u2))
    assert g2 >= tau - 1e-8
    # reject degenerate motor difference
    bad = solve_min_change_socp(W0, [{"d": np.zeros(6), "x": x}], tau, float(PROTO_EPS))
    assert bad["status"] == "reject"
    assert bad["reject_reason"] == "degenerate_motor_difference"
    assert bad["W"] is None


def test_ids_and_routes():
    from experiments.run_tm039jointsocp import _decision, classify_world, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 24
    assert ids[0] == "acquire|c8|A_then_B|reg0|v37"
    assert ids[-1] == "acquire|c8|B_then_A|reg3|always_joint"
    assert classify_world({"v37": True, "fallback_joint": True, "always_joint": True}) == "v37_already_converged"
    assert classify_world({"v37": False, "fallback_joint": True, "always_joint": True}) == "fallback_sufficient"
    assert classify_world({"v37": False, "fallback_joint": False, "always_joint": True}) == "always_required"
    assert classify_world({"v37": False, "fallback_joint": False, "always_joint": False}) == "solver_fail"
    code, _t, fl = _decision(["v37_already_converged"] * 8)
    assert code == "jointsocp_no_v37_fail"
    assert fl["candidate_v40_lock"] is False
    code2, _t2, _f2 = _decision(["fallback_sufficient"] * 2 + ["v37_already_converged"] * 6)
    assert code2 == "jointsocp_fallback_sufficient"
    code3, _t3, _f3 = _decision(["always_required"] * 2 + ["v37_already_converged"] * 6)
    assert code3 == "jointsocp_fallback_blocks"
    code4, _t4, _f4 = _decision(["solver_fail"] * 2 + ["v37_already_converged"] * 6)
    assert code4 == "jointsocp_solver_fail"
    code5, _t5, _f5 = _decision(["fallback_sufficient", "solver_fail"] + ["v37_already_converged"] * 6)
    assert code5 == "jointsocp_mixed_routes"


def test_smoke():
    from experiments.run_tm039jointsocp import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["n_arms"] == 3
    assert out["arms"] == ["v37", "fallback_joint", "always_joint"]
    assert out["clone_matched"] is True
    assert out["hard_budget_passes"] == 16
    assert out["not_an_exact_projector"] is True
    assert hasattr(NeuralCortex, "set_act_socp_arm")
    assert out["neural_ready"] is True


def test_v40_default_off_checkpoints_and_credit_order():
    from three_memory.neural_cortex import (
        ACT_SOCP_ARMS,
        ACT_SOCP_FALLBACK,
        ACT_SOCP_OFF,
    )

    ag = NeuralCortex()
    assert ag._act_socp_arm == ACT_SOCP_OFF
    assert ag._act_socp_always() is False
    assert ag._act_socp_fallback() is False
    assert "exact_projector" not in ACT_SOCP_ARMS
    try:
        ag.set_act_socp_arm("exact_projector")
        raise AssertionError("exact projector must not be a socp arm")
    except ValueError:
        pass
    ag.set_act_socp_arm(ACT_SOCP_FALLBACK)
    snap = ag.checkpoint()
    assert snap["act_socp_arm"] == ACT_SOCP_FALLBACK
    twin = NeuralCortex()
    twin.load_checkpoint(snap)
    assert twin._act_socp_arm == ACT_SOCP_FALLBACK
    missing = dict(snap)
    missing.pop("act_socp_arm")
    twin2 = NeuralCortex()
    twin2.load_checkpoint(missing)
    assert twin2._act_socp_arm == ACT_SOCP_OFF
    g = GenomeConfig().to_dict()
    assert "act_socp_arm" not in g
    credit = NeuralCortex._credit_act_p1_episode
    import inspect

    body = inspect.getsource(credit)
    assert body.index("_act_socp_always") < body.index("_act_proj_arm_active")
    assert body.index("_act_proj_arm_active") < body.index("_apply_act_query_update")
    assert body.index("_apply_act_query_update") < body.index("_run_awake_rehearsal_burst")
    assert body.index("_run_awake_rehearsal_burst") < body.index("_act_socp_fallback")
    src = (REPO / "three_memory" / "neural_cortex.py").read_text()
    assert "closest_feasible_W" not in src
    assert "ACT_SOCP_ALWAYS" in src


def test_socp_reject_does_not_partial_install():
    from three_memory.joint_socp import weight_hash

    ag = NeuralCortex()
    ag.set_act_socp_arm("fallback_joint")
    w0 = ag._from_t(ag.W_act_query).copy()
    h0 = weight_hash(w0)
    rec = ag._run_joint_socp_consolidation()
    assert rec["applied"] is False
    assert rec["status"] in ("no_constraints", "reject")
    assert weight_hash(ag._from_t(ag.W_act_query)) == h0
    assert np.allclose(ag._from_t(ag.W_act_query), w0)


def test_dev_lock_fallback_sufficient_and_no_candidate():
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    assert not (REPO / "docs" / "cortex.candidate.v40.lock").exists()
    assert not (REPO / "docs" / "cortex.candidate.v39.lock").exists()
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "9da95277ac702a48ea0947531d89a1cfc0410c2c"
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["decision_code"] == "jointsocp_fallback_sufficient"
    assert dev["candidate_v40_lock"] is False
    assert dev["not_an_exact_projector"] is True
    assert dev["oracle_installed_in_organism"] is False
    assert dev["phase_flags"]["n_diagnostic"] == 2
    assert dev["phase_flags"]["n_v37_already"] == 6
    assert set(dev["phase_flags"]["diagnostic_routes"]) == {"fallback_sufficient"}
    assert int(dev["phase_flags"]["n_both_fixed_hash_mismatch"]) == 8
    assert dec["decision"]["code"] == "jointsocp_fallback_sufficient"
    assert dec["candidate_v40_lock"] is False
    v37 = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|v37"][0]
    fb = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|fallback_joint"][0]
    aj = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|reg1|always_joint"][0]
    assert v37["fixed"] is False
    assert int(v37["n_store_violations"]) == 3
    assert int(v37["n_probe_correct"]) == 6
    assert fb["fixed"] is True
    assert aj["fixed"] is True
    assert int(fb["n_store_violations"]) == 0
    assert int(aj["n_store_violations"]) == 0
    assert bool(fb["process"]["socp_invoked"]) is True
    assert str(fb["process"]["socp"]["status"]) == "optimal"
    assert bool(fb["process"]["socp"]["applied"]) is True
    assert str(aj["process"]["socp"]["status"]) == "optimal"
    assert fb["w_hash"] != aj["w_hash"]
