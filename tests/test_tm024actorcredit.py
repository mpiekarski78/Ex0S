"""TM.0.24.ACTORCREDIT provenance and smoke. CPU only. Scoring requires runner.lock."""

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
        "docs/lineage_actorcredit_contract.md",
        "docs/lineage_actorcredit.prereg.lock",
        "docs/cortex_v29.prereg.lock",
        "docs/lineage_plasticitymap.decision.lock",
        "experiments/run_tm024actorcredit.py",
        "experiments/cortex_v29_pipeline.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_actorcredit.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["cells"][-1] == "A11"
    assert prereg["reachability_only_if_cells_pass"] is True
    assert prereg["domains"]["CELLS"] == "TM024.ACTORCREDIT.CELLS."
    assert "TM024.WALLMAP" not in prereg["domains"]["CELLS"]
    assert "TM024.REACH" not in prereg["domains"]["CELLS"]
    assert "TM024.PLASTICITYMAP" not in prereg["domains"]["CELLS"]


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_actorcredit_contract.md").read_text(encoding="utf-8")
    assert "Action-owned delayed credit" in text
    assert "0.0.004" in text
    assert "n stays **64**" in text
    assert "clamped" in text.lower()
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_actorcredit.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024actorcredit import actorcredit_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["n"] == 64
    assert lock["shas"] == actorcredit_shas()
    assert lock["domain"] == "TM024.ACTORCREDIT.CELLS."


def test_smoke() -> None:
    from experiments.run_tm024actorcredit import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["capability_claim"] is False
    assert out["product"] == "0.0.004"
    assert out["n"] == 64
    assert out["clamp_abi"] is True


def test_result_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_actorcredit.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["eligible_for_000005"] is False
    assert d["n"] == 64
    assert d["another_lineage_run"] is False
    if d.get("all_cells_pass"):
        assert d["n_pass"] == d["n_cells"] == 12


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_smoke()
    test_result_if_present()
    print("test_tm024actorcredit: ok")


if __name__ == "__main__":
    main()
