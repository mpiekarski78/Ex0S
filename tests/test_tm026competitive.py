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
COMP_PREREG = "7f2f5cb2f8b8262e77dda0f956e918b6bf345e029e15dcfcdd3e6b327993654a"
COMP_ISO = "a14eb7ccc45ab509c633a99049e33b81063437a021f7980409314b71dad6224f"
COMP_CONTRACT = "4c79df93aa9d23790d11a287000aea52a47c6eac451b9333e74361302d826e83"
RUNNER_SHA = "0e69cc59fc15f47563b4eb4d30e4f95ec0a84f50e335c4ec552eaf58c8326de2"
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
    assert not (REPO_ROOT / "docs" / "lineage_competitive.dev.lock").exists()


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
