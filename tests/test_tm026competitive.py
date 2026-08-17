"""TM.0.26.COMPETITIVE / v33 competitive plasticity — historical pins only.

Live v33 behavioral expectations must not rerun through post-v34 neural code.
See tests/test_tm027gatedrehearsal.py for live organism tests after v34.
"""

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
NEURAL_V33_SHA = "ead50d187845010b5cd81a80cf32adb6ab3665ef34e6d3452aab4a4e084b1fc5"
MANIFEST_SHA = "06dcafcd1d3c108ca2ebcef4bf32ec5ff01e279d750b3cec8a1196eba2e4eedb"
EXPECTED_N_CELLS = 54
COMPAT_SHA = "bf74c92e8eaaf115946c74b2e1b030ba61757141c39fb01e757ea6ad2fe9983b"
DEV_LOCK_SHA = "70e4787676b74b04c3eeadb250c20079bd1f03b3f2bce043a95f904d5496daad"
DECISION_SHA = "71bcc14e471e3142c033d6b2ee817f6a14fd35f03736fabb446202f01b1a8e3e"
ADDENDUM_SHA = "17a523e4487fcfec88184c26dfa6ec818e73c5762f22bac518c637df006c6728"


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
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v33.lock").exists()


def test_dev_opened() -> None:
    dev_p = REPO_ROOT / "docs" / "lineage_competitive.dev.lock"
    dec_p = REPO_ROOT / "docs" / "lineage_competitive.decision.lock"
    assert dev_p.exists()
    assert dec_p.exists()
    assert sha(dev_p) == DEV_LOCK_SHA
    assert sha(dec_p) == DECISION_SHA
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


def test_addendum() -> None:
    p = REPO_ROOT / "docs" / "lineage_competitive.decision.addendum.lock"
    assert p.exists()
    add = json.loads(p.read_text(encoding="utf-8"))
    assert add["historical_decision_sha"] == DECISION_SHA
    assert add["historical_dev_lock_sha"] == DEV_LOCK_SHA
    assert add["rewrite_historical_decision"] is False
    assert add["rewrite_historical_dev"] is False
    assert add["neural_edit_this_addendum"] is False
    assert add["dev_rerun_required"] is False
    assert add["honest_decision"] == "competitive_core_acquire_fail"
    assert add["eight_cue_acquire_probes_correct"] == "7/8"
    assert add["uniform_sum_mechanism"] == "analytic_interpretation_not_measured_causality"
    assert add["authorize_next"] == "v34_prediction_error_gated_competitive_rehearsal"
    assert add["candidate_v33_lock_written"] is False
    assert sha(p) == ADDENDUM_SHA


def test_competitive_historical_boundary_immutable() -> None:
    """Historical TM026 7/8 stays pinned; live v34 owns new behavior via TM027."""
    dev = json.loads((REPO_ROOT / "docs" / "lineage_competitive.dev.lock").read_text(encoding="utf-8"))
    dec = json.loads((REPO_ROOT / "docs" / "lineage_competitive.decision.lock").read_text(encoding="utf-8"))
    assert dev["decision_code"] == "competitive_core_acquire_fail"
    assert dec["decision"]["code"] == "competitive_core_acquire_fail"
    rec = next(c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0")
    n_ok = sum(1 for p in rec["probes"] if p["ranking_ok"])
    assert n_ok == 7
    assert rec["passed"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_competitive.dev.lock") == DEV_LOCK_SHA
    assert sha(REPO_ROOT / "docs" / "lineage_competitive.decision.lock") == DECISION_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm026competitive.py") == RUNNER_SHA
    assert (REPO_ROOT / "docs" / "lineage_gatedrehearsal.prereg.lock").exists()


if __name__ == "__main__":
    for name in sorted(n for n in globals() if n.startswith("test_")):
        globals()[name]()
        print("ok", name)
    print("ok")
