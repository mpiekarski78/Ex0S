"""TM.0.24.WALLMAP Phase A provenance. CPU only. No diagnostic answers yet."""

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
        "docs/lineage_wallmap_contract.md",
        "docs/lineage_wallmap.prereg.lock",
        "docs/lineage_engine.candidate.lock",
        "docs/lineage_wall.lock",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    assert not (REPO_ROOT / "docs" / "lineage_wallmap.runner.lock").exists()
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_wallmap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["Q3"]["denominator"] == "standard_error_not_variance_sum"
    assert prereg["Q4"]["neural_edits_forbidden"] is True
    assert prereg["decision_precedence"]["Q4_breaks_over_Q2"] is True
    assert prereg["Q1_optimizer"]["surrogate_is_not_pass_gate"] is True
    assert prereg["Q2"]["one_genotype"] is True


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_wallmap_contract.md").read_text(encoding="utf-8")
    assert "SE_r" in text or "standard error" in text.lower()
    assert "DIAG.FIT" in text or "Q1.DIAG.FIT" in text
    assert "DIAG.TRANSFER" in text or "Q1.DIAG.TRANSFER" in text
    assert "state-only" in text.lower() or "state only" in text.lower()
    assert "0.0.004" in text
    assert "earned_next=false" in text or "`earned_next=false`" in text
    assert "Q4 breaks" in text or "Q4 breaks" in text.replace("**", "")
    assert "favorable birth" in text.lower() or "favorable birth" in text
    assert "variance" in text.lower()
    # lineage candidate untouched
    cand = json.loads((REPO_ROOT / "docs" / "lineage_engine.candidate.lock").read_text(encoding="utf-8"))
    assert cand["version"] == "TM.0.24.LINEAGE.ENGINE.CANDIDATE"
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )


def test_prereg_snr_and_optimizer() -> None:
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_wallmap.prereg.lock").read_text(encoding="utf-8"))
    assert "SE_r" in prereg["Q3"]["snr"] or "SE_r(Delta_ir)" in prereg["Q3"]["snr"]
    assert prereg["Q1_optimizer"]["type"] == "Adam"
    assert prereg["Q1_optimizer"]["max_steps"] == 2000
    assert prereg["Q1_optimizer"]["pass_probe"] == 0.6
    assert "W_act_query" in prereg["Q1_optimizer"]["matrices"]
    assert prereg["Q3"]["P"] == 32
    assert prereg["Q3"]["R"] == 8
    assert prereg["tau"] == 0.6
    assert prereg["delta_B"] == 0.05


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_prereg_snr_and_optimizer()
    print("test_tm024wallmap: ok")


if __name__ == "__main__":
    main()
