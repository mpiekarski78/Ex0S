"""TM.0.27.GATEDREHEARSAL / v34 gated competitive rehearsal — live organism tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

V34_PREREG = "7f11fbb0231149ba5b1b6eb53ae035a47fac38e049ca3c1f6ae76049d31ca123"
V34_ISO = "ac7bb5370c10910a55fae6be7fd6f5d262f4bd5625f2daaa03d3f239bcd2ca49"
V34_AMEND_MD = "e2d9a59d03fa6528deff2bf53095710b3b8f386de6f86104ceb4398b954e8c96"
V34_AMEND = "2d6a2d57d2f99dc23f6fa683ade0a9baa7190a41a4c1fcef4f164de7eb5ad356"
GR_PREREG = "9b5ae678d93f472e4e554b56b481fbeda65166a2db273e07f1407c964c4a3963"
GR_ISO = "59bc57ab8aec39e10040ff10f187464bf66e1dacfc3460e67903647a478a8e91"
GR_CONTRACT = "5a0b2b16c02d0aff0f495efd389fe8d8dfb45646ea9d2dcf5b7f4b24b757885e"
RUNNER_SHA = "2d3c9677ba23fb0a16899bdb12d6ce24797c50e7d5e6fb58a374fbe045e8a99b"
MANIFEST_SHA = "1140e68472d1cfc147d003bb158d68e3dba5b38a0b66348c8a7ee02a988c2e6d"
DEV_LOCK_SHA = "2d784be0d5b80f416aeb88114b43905fc72d3544ea6255d9f9c4339948ad603a"
DECISION_SHA = "9f447bc7f1ecdd909daa3b4dab55776498003c35c79ef816f650c9ff7604a8a0"
EXPECTED_N_CELLS = 54
COMP_ADDENDUM = "17a523e4487fcfec88184c26dfa6ec818e73c5762f22bac518c637df006c6728"
TM026_RUNNER = "cf1844642fc346a85c4e74cea464f726f17148c3e2d3c676fa40d8fe8a5500f4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_bound():
    from experiments.run_tm023cortex import make_cortex

    ag = make_cortex(None, device="cpu")
    ag.bind_actuators(["h_a", "h_b", "h_c", "h_d"])
    return ag


def test_freeze_files() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_gatedrehearsal.prereg.lock") == GR_PREREG
    assert sha(REPO_ROOT / "docs" / "lineage_gatedrehearsal.isolation.lock") == GR_ISO
    assert sha(REPO_ROOT / "docs" / "cortex_v34.prereg.lock") == V34_PREREG
    assert sha(REPO_ROOT / "docs" / "cortex_v34.isolation.lock") == V34_ISO
    assert sha(REPO_ROOT / "docs" / "cortex_v34_architecture_amendment.md") == V34_AMEND_MD
    assert sha(REPO_ROOT / "docs" / "cortex_v34_architecture_amendment.lock") == V34_AMEND
    assert sha(REPO_ROOT / "docs" / "lineage_gatedrehearsal_contract.md") == GR_CONTRACT
    assert sha(REPO_ROOT / "experiments" / "run_tm027gatedrehearsal.py") == RUNNER_SHA
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_gatedrehearsal.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    assert prereg["manifest_sha"] == MANIFEST_SHA
    assert prereg["frozen_runner_sha"] == RUNNER_SHA
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v34.lock").exists()


def test_cell_ids() -> None:
    from experiments.run_tm027gatedrehearsal import expected_cell_ids, manifest_sha

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert manifest_sha(ids) == MANIFEST_SHA


def test_decision_ladder() -> None:
    from experiments.run_tm027gatedrehearsal import _decision, load_prereg

    p = load_prereg()
    base_ids = json.loads((REPO_ROOT / "docs" / "lineage_gatedrehearsal.prereg.lock").read_text(encoding="utf-8"))[
        "expected_cell_ids"
    ]

    def cell(kind: str, ok: bool, *, cid: str, n: int = 2) -> dict[str, object]:
        return {"kind": kind, "passed": ok, "id": cid, "n_cues": n}

    cells: list[dict[str, object]] = []
    for i in base_ids:
        if i.startswith("scale|acquire|"):
            cells.append(cell("acquire", True, cid=i, n=8))
        elif i.startswith("scale|stable|"):
            cells.append(cell("stable", True, cid=i, n=8))
        elif i.startswith("acquire|"):
            n = int(i.split("|c")[1].split("|")[0])
            cells.append(cell("acquire", True, cid=i, n=n))
        elif i.startswith("stable|"):
            n = int(i.split("|c")[1].split("|")[0])
            cells.append(cell("stable", True, cid=i, n=n))
        elif i.startswith("twin|"):
            cells.append(cell("twin", True, cid=i))
        elif i.startswith("eco|"):
            cells.append(cell("eco", True, cid=i))
        elif i.startswith("spec|"):
            cells.append(cell("spec", True, cid=i))
        elif "|w" in i:
            cells.append(cell(i.split("|")[0], True, cid=i))
    code, then, _flags = _decision(cells, p)
    assert code == "gated_rehearsal_battery_pass"
    assert then == "reopen_lineage_readiness"
    cells[0]["passed"] = False
    assert _decision(cells, p)[0] == "gated_rehearsal_core_acquire_fail"


def test_dev_opened() -> None:
    dev_p = REPO_ROOT / "docs" / "lineage_gatedrehearsal.dev.lock"
    dec_p = REPO_ROOT / "docs" / "lineage_gatedrehearsal.decision.lock"
    assert dev_p.exists()
    assert dec_p.exists()
    assert sha(dev_p) == DEV_LOCK_SHA
    assert sha(dec_p) == DECISION_SHA
    dev = json.loads(dev_p.read_text(encoding="utf-8"))
    dec = json.loads(dec_p.read_text(encoding="utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert dev["manifest_sha"] == MANIFEST_SHA
    assert dev["decision_code"] == "gated_rehearsal_core_stability_fail"
    assert dev["phase_flags"]["core_acquire_8"] is True
    assert dev["phase_flags"]["core_stable"] is False
    assert dec["decision"]["code"] == "gated_rehearsal_core_stability_fail"
    assert dec["candidate_v34_lock_written"] is False


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm027gatedrehearsal import DEV_LOCK, refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    try:
        refuse_dev_lock()
    except RuntimeError as e:
        assert "again" in str(e)
    else:
        raise AssertionError("same frozen DEV execution must be refused")


def test_tm026_historical_immutable() -> None:
    assert sha(REPO_ROOT / "experiments" / "run_tm026competitive.py") == TM026_RUNNER
    assert sha(REPO_ROOT / "docs" / "lineage_competitive.decision.addendum.lock") == COMP_ADDENDUM


def test_positive_competitive_delta() -> None:
    import numpy as np
    import torch

    ag = _fresh_bound()
    p1 = np.zeros(64, dtype=np.float64)
    p1[0] = 1.0
    rho = ag._unit_or_zero(p1)
    scores = {"h_a": 0.2, "h_b": 0.5, "h_c": 0.1, "h_d": 0.0}
    orig = ag.actuator_scores
    ag.actuator_scores = lambda _p: scores  # type: ignore[method-assign]
    w0 = ag.W_act_query.detach().clone()
    ag._apply_act_query_update(rho, "h_a", 1.0, 0.15, mix_slow=False)
    dw = (ag.W_act_query - w0).detach()
    m_h = ag._to_t(ag.motor_vocab["h_a"])
    m_r = ag._to_t(ag._rival_mean_vector(rho, "h_a"))
    expected = 0.15 * torch.outer(m_h - m_r, ag._to_t(rho))
    assert float((dw - expected).abs().max().item()) < 1e-9
    ag.actuator_scores = orig


def test_violation_predicate_single_actuator() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a"])
    p1 = np.zeros(64, dtype=np.float64)
    p1[0] = 1.0
    orig_scores = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.5}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.005  # type: ignore[method-assign]
    assert ag._episode_rehearsal_violation(p1, "h_a", 1.0) is True
    ag._act_geometric_margin = lambda _p, _h: 0.02  # type: ignore[method-assign]
    assert ag._episode_rehearsal_violation(p1, "h_a", 1.0) is False
    ag.actuator_scores = orig_scores
    ag._act_geometric_margin = orig_margin


def test_gated_pass_skips_converged_rows() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.asarray(ag._unit_or_zero(np.array([1.0] + [0.0] * 63)), dtype=np.float64)
    ag._episodes = [{"p1": p1.copy(), "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True}]
    orig_scores = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.9, "h_b": 0.1}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.02  # type: ignore[method-assign]
    w0 = ag.W_act_query.detach().clone()
    ps = ag._gated_rehearsal_pass(0.15, pass_index=1)
    assert ps["n_updates"] == 0
    assert ps["violations_after"] == 0
    assert float((ag.W_act_query - w0).abs().max().item()) == 0.0
    ag.actuator_scores = orig_scores
    ag._act_geometric_margin = orig_margin


def test_burst_skips_safe_current_episode() -> None:
    import numpy as np

    from three_memory.neural_cortex import BODY_SETPOINT

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.zeros(64, dtype=np.float64)
    p1[0] = 1.0
    orig_scores = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.9, "h_b": 0.1}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.02  # type: ignore[method-assign]
    ag._pending = {
        "op": "ACT",
        "token": "h_a",
        "rho_elig": p1.copy(),
        "rho_op": p1.copy(),
        "rho_motor": p1.copy(),
        "rho_p1": p1.copy(),
        "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
        "body": np.zeros(4, dtype=np.float64),
        "cost": 0.0,
        "motor_vec": ag.motor_vocab["h_a"].copy(),
        "authored": True,
        "clamped": True,
        "t": 0,
        "interaction_token": "v34_skip",
    }
    w0 = ag.W_act_query.detach().clone()
    out = ag._apply_credit(np.zeros(ag.genome.d_sym), BODY_SETPOINT)
    burst = out.get("rehearsal_burst") or {}
    assert burst.get("total_updates", 0) == 0
    assert float((ag.W_act_query - w0).abs().max().item()) == 0.0
    ag.actuator_scores = orig_scores
    ag._act_geometric_margin = orig_margin


def test_rest_post_slow_mix_recount() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.asarray(ag._unit_or_zero(np.array([1.0] + [0.0] * 63)), dtype=np.float64)
    ag._episodes = [{"p1": p1.copy(), "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True}]
    orig_scores = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.2, "h_b": 0.1}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.005  # type: ignore[method-assign]
    rep = ag._replay_episodes()
    assert "violations_pre_mix" in rep
    assert "violations_post_mix" in rep
    assert isinstance(rep["violations_pre_mix"], int)
    assert isinstance(rep["violations_post_mix"], int)
    ag.actuator_scores = orig_scores
    ag._act_geometric_margin = orig_margin


def test_awake_no_update_when_ranking_correct() -> None:
    import numpy as np

    from three_memory.neural_cortex import BODY_SETPOINT

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.zeros(64, dtype=np.float64)
    p1[0] = 1.0
    orig = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.9, "h_b": 0.1}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.02  # type: ignore[method-assign]
    w0 = ag.W_act_query.detach().clone()
    ag._pending = {
        "op": "ACT",
        "token": "h_a",
        "rho_elig": p1.copy(),
        "rho_op": p1.copy(),
        "rho_motor": p1.copy(),
        "rho_p1": p1.copy(),
        "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
        "body": np.zeros(4, dtype=np.float64),
        "cost": 0.0,
        "motor_vec": ag.motor_vocab["h_a"].copy(),
        "authored": True,
        "clamped": True,
        "t": 0,
        "interaction_token": "v34_awake",
    }
    ag._apply_credit(np.zeros(ag.genome.d_sym), BODY_SETPOINT)
    assert float((ag.W_act_query - w0).abs().max().item()) == 0.0
    ag.actuator_scores = orig
    ag._act_geometric_margin = orig_margin


def test_smoke() -> None:
    from experiments.run_tm027gatedrehearsal import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["manifest_sha"] == MANIFEST_SHA
    assert out["gated_rehearsal"] is True
    assert out["v34_candidate_exists"] is False


def test_violation_predicate_negative() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.zeros(64, dtype=np.float64)
    p1[0] = 1.0
    orig = ag.actuator_scores
    ag.actuator_scores = lambda _p: {"h_a": 0.1, "h_b": 0.8}  # type: ignore[method-assign]
    assert ag._episode_rehearsal_violation(p1, "h_b", -1.0) is True
    ag.actuator_scores = lambda _p: {"h_a": 0.8, "h_b": 0.1}  # type: ignore[method-assign]
    assert ag._episode_rehearsal_violation(p1, "h_b", -1.0) is False
    ag.actuator_scores = orig


def test_rest_gated_skips_converged_store() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.asarray(ag._unit_or_zero(np.array([1.0] + [0.0] * 63)), dtype=np.float64)
    ag._episodes = [{"p1": p1.copy(), "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True}]
    orig_scores = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.9, "h_b": 0.1}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.02  # type: ignore[method-assign]
    w0 = ag.W_act_query.detach().clone()
    rep = ag._replay_episodes()
    assert int(rep.get("total_updates") or 0) == 0
    assert float((ag.W_act_query - w0).abs().max().item()) == 0.0
    ag.actuator_scores = orig_scores
    ag._act_geometric_margin = orig_margin


def test_awake_burst_early_exit() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.asarray(ag._unit_or_zero(np.array([1.0] + [0.0] * 63)), dtype=np.float64)
    ag._episodes = [{"p1": p1.copy(), "handle": "h_a", "adv": 1.0, "age": 1, "version": 1, "valid": True}]
    orig_scores = ag.actuator_scores
    orig_margin = ag._act_geometric_margin
    ag.actuator_scores = lambda _p: {"h_a": 0.9, "h_b": 0.1}  # type: ignore[method-assign]
    ag._act_geometric_margin = lambda _p, _h: 0.02  # type: ignore[method-assign]
    burst = ag._run_awake_rehearsal_burst()
    assert burst["first_converged_pass"] == 1
    assert len(burst["passes"]) == 1
    assert burst["total_updates"] == 0
    assert burst["budget_exhausted"] is False
    ag.actuator_scores = orig_scores
    ag._act_geometric_margin = orig_margin


def test_classify_failure() -> None:
    from experiments.run_tm027gatedrehearsal import classify_failure

    assert classify_failure(stored_pre_mix={"all_margin_ok": False}, stored_post_mix={"all_margin_ok": False}, live_probes_pass=False) == "store_and_live_fail"
    assert classify_failure(stored_pre_mix={"all_margin_ok": False}, stored_post_mix={"all_margin_ok": False}, live_probes_pass=True) == "store_nonconvergence"
    assert classify_failure(stored_pre_mix={"all_margin_ok": True}, stored_post_mix={"all_margin_ok": False}, live_probes_pass=True) == "consolidation_margin_loss"
    assert classify_failure(stored_pre_mix={"all_margin_ok": True}, stored_post_mix={"all_margin_ok": True}, live_probes_pass=False) == "reinstatement_wall"
    assert classify_failure(stored_pre_mix={"all_margin_ok": True}, stored_post_mix={"all_margin_ok": True}, live_probes_pass=True) == "none"


if __name__ == "__main__":
    test_freeze_files()
    for name in sorted(n for n in globals() if n.startswith("test_") and n != "test_freeze_files"):
        globals()[name]()
        print("ok", name)
    print("ok")
