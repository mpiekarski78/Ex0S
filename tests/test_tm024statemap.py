"""TM.0.24.STATEMAP provenance and smoke. CPU only. Scoring requires runner.lock."""

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
        "docs/lineage_statemap_contract.md",
        "docs/lineage_statemap.prereg.lock",
        "docs/lineage_statemap.isolation.lock",
        "docs/cortex.candidate.v29.lock",
        "docs/lineage_actorcredit_reach.lock",
        "experiments/run_tm024statemap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_statemap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["neural_edit"] is False
    assert prereg["cells"][0] == "S0"
    assert prereg["cells"][-1] == "S12"
    assert prereg["domains"]["CELLS"] == "TM024.STATEMAP.CELLS."
    assert prereg["domains"]["TWIN"] == "TM024.STATEMAP.TWIN."
    assert "TM024.ACTORCREDIT" not in prereg["domains"]["CELLS"]
    assert "TM024.WALLMAP" not in prereg["domains"]["CELLS"]
    assert "TM024.REACH" not in prereg["domains"]["CELLS"]
    assert "TM024.PLASTICITYMAP" not in prereg["domains"]["CELLS"]
    iso = json.loads((REPO_ROOT / "docs" / "lineage_statemap.isolation.lock").read_text(encoding="utf-8"))
    assert iso["n"] == 64
    assert iso["hypothesis_status"] == "not_yet_diagnosis"
    assert iso["actorcredit_reach_sha"] == sha(REPO_ROOT / "docs" / "lineage_actorcredit_reach.lock")


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_statemap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "n stays **64**" in text
    assert "S7" in text
    assert "two-timescale" in text
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )
    cand = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v29.lock").read_text(encoding="utf-8"))
    assert cand["neural_cortex_sha"] == "d75b8da7f251378c9638cf9a0c4a859f12b0215d9f6f7b1623e704d831f86d03"
    v30p = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
    if v30p.exists():
        live = json.loads(v30p.read_text(encoding="utf-8"))
        neural = sha(REPO_ROOT / "three_memory" / "neural_cortex.py")
        src = (REPO_ROOT / "three_memory" / "neural_cortex.py").read_text(encoding="utf-8")
        if (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists():
            v31 = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v31.lock").read_text(encoding="utf-8"))
            assert v31["neural_cortex_sha"] == neural
        elif "ACT_SCORE_PROTO" in src:
            assert live["neural_cortex_sha"] != neural
        else:
            assert live["neural_cortex_sha"] == neural
    assert cand["genome"]["n"] == 64


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_statemap.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024statemap import statemap_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["n"] == 64
    assert lock["shas"]["neural_cortex"] == "d75b8da7f251378c9638cf9a0c4a859f12b0215d9f6f7b1623e704d831f86d03"
    if sha(REPO_ROOT / "three_memory" / "neural_cortex.py") == lock["shas"]["neural_cortex"]:
        assert lock["shas"] == statemap_shas()
    assert lock["domain"] == "TM024.STATEMAP.CELLS."
    assert lock["twin_domain"] == "TM024.STATEMAP.TWIN."


def test_smoke() -> None:
    from experiments.run_tm024statemap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["capability_claim"] is False
    assert out["product"] == "0.0.004"
    assert out["n"] == 64
    assert out["domain"] == "TM024.STATEMAP.CELLS."


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_statemap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["eligible_for_000005"] is False
    assert d["n"] == 64
    assert d["neural_edit"] is False
    assert d["another_lineage_run"] is False
    assert d["amendment_authorized"] is False
    assert d["n_cells"] == 13
    assert [c["id"] for c in d["cells"]] == [
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S11",
        "S12",
    ]


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_smoke()
    test_decision_if_present()
    print("test_tm024statemap: ok")


if __name__ == "__main__":
    main()
