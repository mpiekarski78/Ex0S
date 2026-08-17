"""TM.0.24.DISCRIMMAP provenance. Runner-only. No neural edit. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
ELIG_DEC = "b10343a6e27ade4d189e922ce1dd32c0c4b0dd8618b82d48db7edff0e0de4e86"
ELIG_ADD = "afdfc406c0747c16ca6f5403d9363a034f7bfbf39278ca5e508e980937c6c967"
ELIG_DEV = "33f79b6b83fb5b7e33b452019e010b5a05a8b0cd762b8ed6f355346b6a4a7578"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_discrimmap_contract.md",
        "docs/lineage_discrimmap.prereg.lock",
        "docs/lineage_discrimmap.isolation.lock",
        "docs/lineage_eligmap.decision.lock",
        "docs/lineage_eligmap.decision.addendum.lock",
        "docs/lineage_eligmap.dev.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024discrimmap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_discrimmap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["implementation_authorized"] is False
    assert prereg["declared_budget_remains_closed"] == 1536
    assert prereg["eligibility_budget_installed"] is False
    assert prereg["margin"]["kind"] == "normalized_geometric"
    assert prereg["margin"]["geometric_margin_min"] == 0.01
    assert prereg["margin"]["reject_raw_linear_margin"] is True
    assert prereg["margin"]["frozen_before_dev"] is True
    assert prereg["arms"]["D4"]["v_eligible"] is False
    assert prereg["arms"]["D2"]["lambda"] == 0.01
    assert prereg["arms"]["D2"]["sklearn"] is False
    assert prereg["n1n2_secondary"] is True
    assert prereg["domains"]["DEV"] == "TM024.DISCRIMMAP.DEV."
    assert prereg["score_reserved_unopened"] is True
    assert prereg["addresses"] == ["E0", "E1", "Edelta", "Elam_0.9"]
    iso = json.loads((REPO_ROOT / "docs" / "lineage_discrimmap.isolation.lock").read_text(encoding="utf-8"))
    assert iso["eligmap_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_eligmap.decision.lock")
    assert iso["eligmap_addendum_sha"] == sha(REPO_ROOT / "docs" / "lineage_eligmap.decision.addendum.lock")
    assert iso["eligmap_dev_sha"] == sha(REPO_ROOT / "docs" / "lineage_eligmap.dev.lock")
    assert iso["implementation_authorized"] is False
    assert iso["neural_edit"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_eligmap.decision.lock") == ELIG_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_eligmap.decision.addendum.lock") == ELIG_ADD
    assert sha(REPO_ROOT / "docs" / "lineage_eligmap.dev.lock") == ELIG_DEV
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v32.lock").exists()


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_discrimmap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "runner-only" in text.lower()
    assert "geometric margin" in text.lower()
    assert "D1" in text and "D3" in text and "D4" in text
    assert "not a neural amendment" in text.lower()
    assert "1536" in text or "1,536" in text
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT


def test_geometric_margin_scale_invariant() -> None:
    import numpy as np
    from experiments.run_tm024discrimmap import geometric_margin

    x = np.arange(64, dtype=np.float64) + 1.0
    x = x / np.linalg.norm(x)
    w = x.copy()
    g1 = geometric_margin(w, 0.0, x, 1.0)
    g10 = geometric_margin(10.0 * w, 0.0, x, 1.0)
    assert abs(g1 - g10) <= 1e-12
    assert abs(g1 - 1.0) <= 1e-9


def test_smoke() -> None:
    from experiments.run_tm024discrimmap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["geometric_margin_scale_invariant"] is True
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["eligibility_budget_installed"] is False
    assert out["product"] == "0.0.004"
    assert out["earned_next"] is False


def test_score_and_dev_lock_refused() -> None:
    from experiments.run_tm024discrimmap import refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    try:
        refuse_dev_lock()
    except RuntimeError as e:
        assert "runner.lock" in str(e)
    else:
        raise AssertionError("DEV lock must wait for runner.lock")


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_discrimmap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["neural_edit"] is False
    assert d["candidate_v31"] is False
    assert "TM024.DISCRIMMAP.SCORE." not in json.dumps(d)


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_geometric_margin_scale_invariant()
    test_smoke()
    test_score_and_dev_lock_refused()
    test_decision_if_present()
    print("test_tm024discrimmap: ok")


if __name__ == "__main__":
    main()
