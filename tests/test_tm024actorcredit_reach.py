"""TM.0.24.ACTORCREDIT.REACH provenance and smoke. CPU only. Scoring requires runner.lock."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_actorcredit_reach_contract.md",
        "docs/lineage_actorcredit_reach.prereg.lock",
        "docs/cortex.candidate.v29.lock",
        "docs/lineage_actorcredit.lock",
        "experiments/run_tm024actorcredit_reach.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_actorcredit_reach.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["n"] == 64
    assert prereg["tau"] == 0.6
    assert prereg["fit_domain"] == "TM024.ACTORCREDIT.REACH.FIT."
    assert prereg["check_domain"] == "TM024.ACTORCREDIT.REACH.CHECK."
    assert "TM024.REACH.DIAG.FIT." in prereg["distinct_from"]
    cells = json.loads((REPO_ROOT / "docs" / "lineage_actorcredit.lock").read_text(encoding="utf-8"))
    assert cells["all_cells_pass"] is True


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_actorcredit_reach_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "n stays **64**" in text
    assert "TM024.ACTORCREDIT.REACH.CHECK." in text
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_actorcredit_reach.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024actorcredit_reach import ac_reach_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["n"] == 64
    assert lock["shas"] == ac_reach_shas()
    assert lock["fit_domain"] == "TM024.ACTORCREDIT.REACH.FIT."


def test_smoke() -> None:
    from experiments.run_tm024actorcredit_reach import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["credit_precondition_ok"] is True
    assert out["n"] == 64


def test_result_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_actorcredit_reach.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["n"] == 64
    assert d["historical_reach_not_rescored"] is True
    assert d["another_lineage_run"] is False
    if not d.get("passed"):
        assert d.get("q3_authorized") is False


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_smoke()
    test_result_if_present()
    print("test_tm024actorcredit_reach: ok")


if __name__ == "__main__":
    main()
