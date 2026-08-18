"""TM040 untouched causal-battery freeze tests. No neural or solver edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from three_memory.joint_socp import solve_min_change_socp, weight_hash
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
PREREG = REPO / "docs" / "lineage_causalbattery.prereg.lock"
ISO = REPO / "docs" / "lineage_causalbattery.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_causalbattery_contract.md"
RUNNER = REPO / "experiments" / "run_tm040causal.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
TM039_DEC = REPO / "docs" / "lineage_jointsocp.decision.lock"
TM039_DEV = REPO / "docs" / "lineage_jointsocp.dev.lock"
TM039_RUNNER = REPO / "experiments" / "run_tm039jointsocp.py"
MANIFEST = "7f422a2d81d2258c764334db8a802c06c5f28b13b58fa819b50a160438493f33"
FROZEN_RUNNER_SHA = "0739de3225e36bf88b66001ae9af2c3232a937d0bc6bc3b64eb34bdf7d2f9c6b"
FROZEN_NEURAL_SHA = "2eb45d8769402330f5ee39a04afffe110a435a0e64a40b12bc2d874b36f5ed59"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM039_DEC_SHA = "0275f20ebc9d0ff528b4edbf2767d21c69689b94f6559a25ce00eca9f7d1618d"
TM039_DEV_SHA = "c2f5dae06f591c6e09a651122a4f893d2e4ccac45d42c5bf7965ba8b9891b21d"
TM039_RUNNER_SHA = "e40304ee4ece2c834390094c9853d122bb89cb8fb18923aa348e052d146e288d"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_no_candidate_and_no_edits():
    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 102
    assert p["v40_candidate_authorized"] is False
    assert p["neural_edit_authorized"] is False
    assert p["solver_edit_authorized"] is False
    assert p["hashes_are_diagnostic"] is True
    assert p["later_learning_from_reset_checkpoint"] is False
    assert p["scored_arm"] == "fallback_joint"
    assert p["observational_arm"] == "always_joint"
    assert p["seed_registry"] != 22222
    assert "jointsocp_generalization_not_exercised" in [d["code"] for d in p["decision_ladder"]]
    assert "auto_cortex.candidate.v40.lock" in iso["refuse"]
    assert "edit_neural_cortex.py" in iso["refuse"]
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA
    assert _sha(NEURAL) == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(TM039_DEC) == TM039_DEC_SHA
    assert _sha(TM039_DEV) == TM039_DEV_SHA
    assert _sha(TM039_RUNNER) == TM039_RUNNER_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert "act_socp_arm" not in GenomeConfig().to_dict()


def test_ids_and_decision_ladder():
    from experiments.run_tm040causal import _decision, expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 102
    assert ids[0] == "acquire|c8|A_then_B|w0|v37"
    assert ids[-1] == "contradict|w1|always_joint"
    fb = {
        "kind": "acquire",
        "arm": "fallback_joint",
        "passed": True,
        "stem": "acquire|c8|A_then_B|w0",
        "socp_activated": False,
        "socp_installed": False,
        "socp_installed_before_later_probes": False,
        "total_frobenius_applied": 0.0,
        "n_socp_calls": 0,
    }

    def cells(activated=False, later=False, v37_fail=False):
        out = []
        for kind in ("acquire", "stable", "hist", "novel", "eco", "spec", "scale", "later", "contradict"):
            for arm, passed in (("v37", not v37_fail), ("fallback_joint", True), ("always_joint", True)):
                row = dict(fb)
                row["kind"] = kind
                row["arm"] = arm
                row["passed"] = passed
                row["stem"] = f"{kind}|stem"
                if arm == "fallback_joint":
                    row["socp_activated"] = activated
                    row["socp_installed"] = activated
                    row["socp_installed_before_later_probes"] = bool(later and kind == "later")
                out.append(row)
        return out

    code, _t, fl = _decision(cells(activated=False))
    assert code == "jointsocp_generalization_not_exercised"
    assert fl["candidate_v40_lock"] is False
    code2, _t2, fl2 = _decision(cells(activated=True, later=False))
    assert code2 == "jointsocp_later_learning_not_exercised"
    code3, _t3, fl3 = _decision(cells(activated=True, later=True))
    assert code3 == "jointsocp_fallback_battery_pass"
    assert fl3["candidate_v40_lock"] is False
    fail = cells(activated=True, later=True)
    for c in fail:
        if c["arm"] == "fallback_joint" and c["kind"] == "acquire":
            c["passed"] = False
    code4, _t4, _f4 = _decision(fail)
    assert code4 == "jointsocp_fallback_acquire_fail"
    rescue = cells(activated=True, later=True, v37_fail=True)
    _c, _t, flr = _decision(rescue)
    assert flr["n_causal_rescues"] >= 1


def test_solver_reject_laws_leave_no_partial_W():
    rng = np.random.default_rng(9)
    W0 = rng.normal(size=(6, 8))
    h0 = weight_hash(W0)
    d = rng.normal(size=6)
    d = d / np.linalg.norm(d)
    x = rng.normal(size=8)
    x = x / np.linalg.norm(x)
    tau = float(ACT_MARGIN_FLOOR)
    # infeasible: same x ranked both ways
    inf = solve_min_change_socp(W0, [{"d": d, "x": x}, {"d": -d, "x": x}], tau, float(PROTO_EPS))
    assert inf["status"] == "reject"
    assert inf["W"] is None
    assert inf.get("applied") is False
    # NaN
    bad = W0.copy()
    bad[0, 0] = np.nan
    nan = solve_min_change_socp(bad, [{"d": d, "x": x}], tau, float(PROTO_EPS))
    assert nan["status"] == "reject"
    assert nan["W"] is None
    # unavailable solver
    with patch("cvxpy.Problem.solve", side_effect=RuntimeError("solver missing")):
        unav = solve_min_change_socp(W0, [{"d": d, "x": x}], tau, float(PROTO_EPS))
    assert unav["status"] == "reject"
    assert str(unav.get("reject_reason") or "").startswith("solver_exception")
    assert unav["W"] is None
    # non-optimal status
    with patch("cvxpy.Problem.solve", lambda self, **k: setattr(self, "_status", "infeasible") or None):
        # fall back: force status after solve by wrapping
        pass
    import cvxpy as cp

    real_solve = cp.Problem.solve

    def fake_solve(self, *a, **k):
        real_solve(self, *a, **k)
        self._status = "user_limit"

    with patch.object(cp.Problem, "solve", fake_solve):
        nopt = solve_min_change_socp(W0, [{"d": d, "x": x}], tau, float(PROTO_EPS))
    assert nopt["status"] == "reject"
    assert nopt["W"] is None
    # organism predicate reject restores bytes
    ag = NeuralCortex()
    w_bytes = np.ascontiguousarray(ag._from_t(ag.W_act_query)).tobytes()
    rec = ag._run_joint_socp_consolidation()
    assert rec["applied"] is False
    assert np.ascontiguousarray(ag._from_t(ag.W_act_query)).tobytes() == w_bytes
    assert weight_hash(W0) == h0


def test_excessive_residual_and_inf_reject():
    rng = np.random.default_rng(3)
    W0 = rng.normal(size=(4, 5))
    d = np.array([1.0, 0.0, 0.0, 0.0])
    x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    infw = W0.copy()
    infw[0, 0] = np.inf
    out = solve_min_change_socp(infw, [{"d": d, "x": x}], float(ACT_MARGIN_FLOOR), float(PROTO_EPS))
    assert out["status"] == "reject"
    assert out["W"] is None
    # mock optimal W that still has negative slack
    import cvxpy as cp

    def fake(self, *a, **k):
        self._status = "optimal"
        # leave value as zeros which may violate; solver path uses W.value
        if self.variables():
            self.variables()[0].value = -np.ones_like(W0)

    with patch.object(cp.Problem, "solve", fake):
        res = solve_min_change_socp(W0, [{"d": d, "x": x}], float(ACT_MARGIN_FLOOR), float(PROTO_EPS))
    assert res["status"] == "reject"
    assert res["W"] is None
    assert res.get("reject_reason") in ("soc_residual_negative", "zero_normal_or_tie", "nonfinite_W") or str(
        res.get("reject_reason") or ""
    ).startswith("solver")


def test_smoke_and_neural_untouched():
    from experiments.run_tm040causal import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["neural_sha_ok"]
    assert out["solver_sha_ok"]
    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src
    assert "closest_feasible_W" not in src


def test_dev_lock_acquire_fail_and_no_candidate():
    devp = REPO / "docs" / "lineage_causalbattery.dev.lock"
    decp = REPO / "docs" / "lineage_causalbattery.decision.lock"
    assert _sha(devp) == "b10865b5f6fea382396db736549488c68dcdc5000932907a3612e29b53354ad7"
    assert _sha(decp) == "734204f628362f58e4f3b19237dd82398016544655d2c13800f3408854bd1b99"
    assert _sha(NEURAL) == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "3a9b64f67c61bade2745a7a18d13e13e667e309f"
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["decision_code"] == "jointsocp_fallback_acquire_fail"
    assert dev["candidate_v40_lock"] is False
    assert dec["candidate_v40_lock"] is False
    flags = dev["phase_flags"]
    assert flags["fallback_acquire"] is False
    assert flags["fallback_eco"] is True
    assert flags["fallback_spec"] is True
    assert flags["fallback_contradict"] is True
    assert int(flags["n_untouched_socp_activated"]) == 4
    assert int(flags["n_causal_rescues"]) == 2
    assert set(flags["causal_rescue_stems"]) == {"stable|c8|A_then_B|w0", "hist|c8|A_then_B|w0"}
    acq_fb = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|fallback_joint"][0]
    acq_aj = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|always_joint"][0]
    acq_v = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|v37"][0]
    assert acq_fb["passed"] is False
    assert int(acq_fb["n_store_violations"]) == 0
    assert int(acq_fb["n_probe_correct"]) == 7
    assert bool(acq_fb["socp_activated"]) is False
    assert acq_v["passed"] is False
    assert acq_aj["passed"] is True
    assert int(acq_aj["n_probe_correct"]) == 8
    ct = [c for c in dev["cells"] if c["id"] == "contradict|w0|fallback_joint"][0]
    assert ct["passed"] is True
