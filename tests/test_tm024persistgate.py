"""TM.0.24.PERSISTGATE provenance. No scoring. Neural remains v29."""

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
        "docs/lineage_persistgate_contract.md",
        "docs/lineage_persistgate.prereg.lock",
        "docs/lineage_persistgate.isolation.lock",
        "docs/cortex.candidate.v29.lock",
        "docs/lineage_collisionmap.decision.lock",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_persistgate.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["neural_edit"] is False
    assert prereg["implementation_authorized"] is False
    assert prereg["investigation_justified"] is True
    assert prereg["credit_historical_state_only"] == "insufficient"
    assert prereg["first_candidate_status"] == "not_authorized"
    assert prereg["thresholds"]["cos_distinct_max"] == 0.99
    assert prereg["thresholds"]["l2_distinct_min"] == 0.05
    iso = json.loads((REPO_ROOT / "docs" / "lineage_persistgate.isolation.lock").read_text(encoding="utf-8"))
    assert iso["collisionmap_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_collisionmap.decision.lock")
    assert iso["implementation_authorized"] is False


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_persistgate_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "n stays **64**" in text
    assert "does **not** authorize a particular implementation" in text
    assert "zero-input motor tick" in text
    assert "live motor readout" in text.lower()
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


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    print("test_tm024persistgate: ok")


if __name__ == "__main__":
    main()
