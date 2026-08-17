"""TM.0.24.COLLISIONMAP provenance and smoke. CPU only. Scoring requires runner.lock."""

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
        "docs/lineage_collisionmap_contract.md",
        "docs/lineage_collisionmap.prereg.lock",
        "docs/lineage_collisionmap.isolation.lock",
        "docs/cortex.candidate.v29.lock",
        "docs/lineage_statemap.decision.lock",
        "experiments/run_tm024collisionmap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_collisionmap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["neural_edit"] is False
    assert prereg["cells"] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert prereg["domains"]["CELLS"] == "TM024.COLLISIONMAP.CELLS."
    assert prereg["domains"]["TWIN"] == "TM024.COLLISIONMAP.TWIN."
    assert "TM024.STATEMAP" not in prereg["domains"]["CELLS"]
    assert "TM024.ACTORCREDIT" not in prereg["domains"]["CELLS"]
    assert prereg["frozen_readout_fit"]["method"] == "closed_form_ridge"
    assert prereg["frozen_readout_fit"]["iterative"] is False
    assert prereg["frozen_readout_fit"]["lambda"] == 0.01
    assert prereg["frozen_readout_fit"]["state"] == "rho_elig"
    assert prereg["thresholds"]["cos_distinct_max"] == 0.99
    iso = json.loads((REPO_ROOT / "docs" / "lineage_collisionmap.isolation.lock").read_text(encoding="utf-8"))
    assert iso["statemap_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_statemap.decision.lock")
    assert iso["n"] == 64


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_collisionmap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "n stays **64**" in text
    assert "attractor" in text.lower()
    assert "closed-form ridge" in text.lower()
    assert "two-timescale" in text
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )
    cand = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v29.lock").read_text(encoding="utf-8"))
    assert cand["neural_cortex_sha"] == sha(REPO_ROOT / "three_memory" / "neural_cortex.py")
    assert cand["genome"]["n"] == 64


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_collisionmap.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024collisionmap import collisionmap_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["n"] == 64
    assert lock["shas"] == collisionmap_shas()
    assert lock["domain"] == "TM024.COLLISIONMAP.CELLS."
    assert lock["frozen_readout_fit"]["lambda"] == 0.01


def test_smoke() -> None:
    from experiments.run_tm024collisionmap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["capability_claim"] is False
    assert out["product"] == "0.0.004"
    assert out["n"] == 64
    assert out["domain"] == "TM024.COLLISIONMAP.CELLS."


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_collisionmap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["n"] == 64
    assert d["neural_edit"] is False
    assert d["another_lineage_run"] is False
    assert d["amendment_authorized"] is False
    assert d["two_timescale_authorized"] is False
    assert d["n_cells"] == 6
    assert [c["id"] for c in d["cells"]] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert d["decision"]["code"] in {
        "attractor_collapse",
        "sequential_plastic_write_interference",
        "representation_rank_failure",
        "plastic_update_geometry_failure",
        "unresolved_cross_cue_collision",
    }


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_smoke()
    test_decision_if_present()
    print("test_tm024collisionmap: ok")


if __name__ == "__main__":
    main()
