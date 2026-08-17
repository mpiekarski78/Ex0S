"""TM.0.24.REACH provenance and smoke. CPU only. Scoring requires runner.lock."""

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
        "docs/lineage_reach_contract.md",
        "docs/lineage_reach.prereg.lock",
        "docs/cortex.candidate.v28.lock",
        "docs/lineage_wallmap.decision.lock",
        "docs/lineage_wallmap_q2.lock",
        "experiments/run_tm024reach.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_reach.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["tau"] == 0.6
    assert prereg["delta_B"] == 0.05
    assert prereg["one_genotype"] is True
    assert prereg["fit_domain"] == "TM024.REACH.DIAG.FIT."
    assert prereg["check_domain"] == "TM024.REACH.DIAG.CHECK."
    assert prereg["fit_domain"] not in prereg["distinct_from_wallmap_q2"]
    assert "TM024.WALLMAP.Q2.DIAG.FIT." in prereg["distinct_from_wallmap_q2"]


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_reach_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "TM024.REACH.DIAG.CHECK." in text
    assert "n stays" in text.lower() or "n stays **64**" in text
    assert "WALLMAP" in text
    cand = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v28.lock").read_text(encoding="utf-8"))
    assert cand["version"] == "TM.0.23.CORTEX.CANDIDATE.V28"
    assert cand["genome"]["n"] == 64
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_reach.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024reach import reach_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["n"] == 64
    live = reach_shas()
    frozen = lock["shas"]
    if live.get("neural_cortex") != frozen.get("neural_cortex"):
        assert (REPO_ROOT / "docs" / "cortex_v29_architecture_amendment.lock").is_file()
        assert frozen["neural_cortex"] == "0a4014ce91bf08b69693924ee645bdc912ae4c6e0a9b6529bda6a6fe8a281892"
        for key, val in frozen.items():
            if key == "neural_cortex":
                continue
            assert live.get(key) == val, key
    else:
        assert lock["shas"] == live
    assert lock["fit_domain"] == "TM024.REACH.DIAG.FIT."
    assert lock["check_domain"] == "TM024.REACH.DIAG.CHECK."


def test_smoke() -> None:
    from experiments.run_tm024reach import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["capability_claim"] is False
    assert out["product"] == "0.0.004"
    assert out["n"] == 64
    assert out["credit_precondition_ok"] is True


def test_result_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_reach.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["eligible_for_000005"] is False
    assert d["n"] == 64
    assert d["check_domain"] == "TM024.REACH.DIAG.CHECK."
    assert d["wallmap_q2_historical"] is True


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_smoke()
    test_result_if_present()
    print("test_tm024reach: ok")


if __name__ == "__main__":
    main()
