"""TM.0.26.COMPETITIVE / v33 competitive plasticity provenance."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

V33_PREREG = "badbc56923737fffc5712c89af9d594a77a310d5c44f44ef689a15b8157c1523"
V33_ISO = "ec92232595ef196b1dbceb40343be30fb296f2a38b4c537cad17ce83acc863f7"
V33_AMEND_MD = "acb9efa2e5861abdc4597c3e419f985ae6b483d2bd8ee71cdffed888e6c1b5bf"
V33_AMEND = "f68a44fa5778f7fd85eb2b4b61d6e3cde68c90f7ff804300ca40559252e47ae5"
COMP_PREREG = "9f3f206fa0532348cfa5ec4cd396636758f1605744b414662a6dd77e65820703"
COMP_ISO = "a14eb7ccc45ab509c633a99049e33b81063437a021f7980409314b71dad6224f"
COMP_CONTRACT = "4c79df93aa9d23790d11a287000aea52a47c6eac451b9333e74361302d826e83"
RUNNER_SHA = "cf1844642fc346a85c4e74cea464f726f17148c3e2d3c676fa40d8fe8a5500f4"
MANIFEST_SHA = "06dcafcd1d3c108ca2ebcef4bf32ec5ff01e279d750b3cec8a1196eba2e4eedb"
EXPECTED_N_CELLS = 54
COMPAT_SHA = "bf74c92e8eaaf115946c74b2e1b030ba61757141c39fb01e757ea6ad2fe9983b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_files() -> None:
    assert sha(REPO_ROOT / "docs" / "cortex_v33.prereg.lock") == V33_PREREG
    assert sha(REPO_ROOT / "docs" / "cortex_v33.isolation.lock") == V33_ISO
    assert sha(REPO_ROOT / "docs" / "cortex_v33_architecture_amendment.md") == V33_AMEND_MD
    assert sha(REPO_ROOT / "docs" / "cortex_v33_architecture_amendment.lock") == V33_AMEND
    assert sha(REPO_ROOT / "docs" / "lineage_competitive.prereg.lock") == COMP_PREREG
    assert sha(REPO_ROOT / "docs" / "lineage_competitive.isolation.lock") == COMP_ISO
    assert sha(REPO_ROOT / "docs" / "lineage_competitive_contract.md") == COMP_CONTRACT
    assert sha(REPO_ROOT / "docs" / "lineage_twoscale.compat.lock") == COMPAT_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm026competitive.py") == RUNNER_SHA
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_competitive.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    assert prereg["manifest_sha"] == MANIFEST_SHA
    assert prereg["frozen_runner_sha"] == RUNNER_SHA
    assert prereg["geometry_only"] is True
    v33 = json.loads((REPO_ROOT / "docs" / "cortex_v33.prereg.lock").read_text(encoding="utf-8"))
    assert v33["authorized_law"] == "competitive_plasticity_at_p1_geometry_only"
    assert v33["low_margin_awake_refused"] is True
    assert v33["preserve_missing_p1_fallback"] is True
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v33.lock").exists()


def test_dev_opened() -> None:
    dev_p = REPO_ROOT / "docs" / "lineage_competitive.dev.lock"
    dec_p = REPO_ROOT / "docs" / "lineage_competitive.decision.lock"
    assert dev_p.exists()
    assert dec_p.exists()
    assert sha(dev_p) == "70e4787676b74b04c3eeadb250c20079bd1f03b3f2bce043a95f904d5496daad"
    assert sha(dec_p) == "71bcc14e471e3142c033d6b2ee817f6a14fd35f03736fabb446202f01b1a8e3e"
    dev = json.loads(dev_p.read_text(encoding="utf-8"))
    dec = json.loads(dec_p.read_text(encoding="utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert dev["manifest_sha"] == MANIFEST_SHA
    assert dev["decision_code"] == "competitive_core_acquire_fail"
    assert dev["phase_flags"]["core_acquire_4"] is True
    assert dev["phase_flags"]["core_acquire_8"] is False
    assert dev["phase_flags"]["integrity"] is True
    assert dec["decision"]["code"] == "competitive_core_acquire_fail"
    assert dec["lineage_reopened"] is False
    assert dec["frozen_runner_sha"] == RUNNER_SHA


def test_cell_ids() -> None:
    from experiments.run_tm026competitive import expected_cell_ids, manifest_sha

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert manifest_sha(ids) == MANIFEST_SHA
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_competitive.prereg.lock").read_text(encoding="utf-8"))
    assert sorted(ids) == sorted(prereg["expected_cell_ids"])
    assert "scale|acquire|c8h4|A_then_B|w0" in ids
    assert "hold|w1" in ids
    assert "neg|w0" in ids


def test_decision_ladder() -> None:
    from experiments.run_tm026competitive import _decision, load_prereg

    p = load_prereg()
    base_ids = json.loads((REPO_ROOT / "docs" / "lineage_competitive.prereg.lock").read_text(encoding="utf-8"))[
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
    assert code == "competitive_plasticity_battery_pass"
    assert then == "reopen_lineage_readiness"
    cells[0]["passed"] = False
    assert _decision(cells, p)[0] == "competitive_core_acquire_fail"


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm026competitive import DEV_LOCK, refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    if DEV_LOCK.exists():
        try:
            refuse_dev_lock()
        except RuntimeError as e:
            assert "again" in str(e)
        else:
            raise AssertionError("same frozen DEV execution must be refused")
        return
    refuse_dev_lock()


def _fresh_bound():
    from experiments.run_tm023cortex import make_cortex

    ag = make_cortex(None, device="cpu")
    ag.bind_actuators(["h_a", "h_b", "h_c", "h_d"])
    return ag


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
    pure = 0.15 * torch.outer(m_h, ag._to_t(rho))
    assert float((dw - expected).abs().max().item()) < 1e-9
    assert float((dw - pure).abs().max().item()) > 1e-9
    ag.actuator_scores = orig


def test_negative_delta_mh_only() -> None:
    import numpy as np
    import torch

    ag = _fresh_bound()
    p1 = np.zeros(64, dtype=np.float64)
    p1[1] = 1.0
    rho = ag._unit_or_zero(p1)
    w0 = ag.W_act_query.detach().clone()
    ag._apply_act_query_update(rho, "h_b", -0.2, 0.15, mix_slow=False)
    dw = (ag.W_act_query - w0).detach()
    m_h = ag._to_t(ag.motor_vocab["h_b"])
    expected = -0.2 * 0.15 * torch.outer(m_h, ag._to_t(rho))
    assert float((dw - expected).abs().max().item()) < 1e-9


def test_two_handle_rival_is_single() -> None:
    import numpy as np

    ag = _fresh_bound()
    ag.bind_actuators(["h_a", "h_b"])
    p1 = np.zeros(64, dtype=np.float64)
    p1[2] = 1.0
    rho = ag._unit_or_zero(p1)
    scores = {"h_a": 0.3, "h_b": 0.6}
    orig = ag.actuator_scores
    ag.actuator_scores = lambda _p: scores  # type: ignore[method-assign]
    m_r = ag._rival_mean_vector(rho, "h_a")
    single = np.asarray(ag.motor_vocab["h_b"], dtype=np.float64).reshape(-1)
    assert np.allclose(m_r, single)
    ag.actuator_scores = orig


def test_tie_mean_rival_exchangeable() -> None:
    import numpy as np

    ag = _fresh_bound()
    p1 = np.zeros(64, dtype=np.float64)
    p1[3] = 1.0
    rho = ag._unit_or_zero(p1)
    scores = {"h_a": 0.4, "h_b": 0.5, "h_c": 0.5, "h_d": 0.1}
    orig = ag.actuator_scores
    ag.actuator_scores = lambda _p: scores  # type: ignore[method-assign]
    m_r = ag._rival_mean_vector(rho, "h_a")
    b = np.asarray(ag.motor_vocab["h_b"], dtype=np.float64)
    c = np.asarray(ag.motor_vocab["h_c"], dtype=np.float64)
    assert np.allclose(m_r, 0.5 * (b + c))
    ag.actuator_scores = orig


def test_smoke() -> None:
    from experiments.run_tm026competitive import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert out["manifest_sha"] == MANIFEST_SHA
    assert out["geometry_only"] is True
    assert out["v33_candidate_exists"] is False
