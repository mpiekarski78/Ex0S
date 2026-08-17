"""TM.0.24.WALLMAP provenance and smoke. CPU only. Diagnostics require runner.lock."""

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
        "experiments/run_tm024wallmap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
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
    assert "standard error" in text.lower() or "SE_r" in text
    assert "Q1.DIAG.FIT" in text or "DIAG.FIT" in text
    assert "TRANSFER" in text
    assert "state-only" in text.lower() or "State-only" in text or "state only" in text.lower()
    assert "0.0.004" in text
    cand = json.loads((REPO_ROOT / "docs" / "lineage_engine.candidate.lock").read_text(encoding="utf-8"))
    assert cand["version"] == "TM.0.24.LINEAGE.ENGINE.CANDIDATE"
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )


def test_prereg_snr_and_optimizer() -> None:
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_wallmap.prereg.lock").read_text(encoding="utf-8"))
    assert "SE_r" in prereg["Q3"]["snr"]
    assert prereg["Q1_optimizer"]["type"] == "Adam"
    assert prereg["Q1_optimizer"]["max_steps"] == 2000
    assert prereg["Q1_optimizer"]["pass_probe"] == 0.6
    assert "W_act_query" in prereg["Q1_optimizer"]["matrices"]
    assert prereg["Q3"]["P"] == 32
    assert prereg["Q3"]["R"] == 8
    assert prereg["tau"] == 0.6
    assert prereg["delta_B"] == 0.05


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_wallmap.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024wallmap import wallmap_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    live = wallmap_shas()
    frozen = lock["shas"]
    if live.get("neural_cortex") != frozen.get("neural_cortex"):
        assert (REPO_ROOT / "docs" / "cortex_v28_architecture_amendment.lock").is_file()
        assert frozen["neural_cortex"] == "2b563a9c5de3ec8b411121bd5518c09f49f422f44108138ec34a1d5708c98d2e"
        for key, val in frozen.items():
            if key == "neural_cortex":
                continue
            if key == "runner" and (REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.lock").exists():
                continue
            assert live.get(key) == val, key
    else:
        assert frozen == live
    assert lock["Q3"]["denominator"] == "standard_error_not_variance_sum" or "SE" in str(lock["Q3"])


def test_smoke() -> None:
    from experiments.run_tm024wallmap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["capability_claim"] is False
    assert out["product"] == "0.0.004"


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_wallmap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["eligible_for_000005"] is False
    assert d["Q4_precedence_over_Q2"] is True
    assert d["increase_n"] is False


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_prereg_snr_and_optimizer()
    test_runner_lock_if_present()
    test_smoke()
    test_decision_if_present()
    print("test_tm024wallmap: ok")


if __name__ == "__main__":
    main()
