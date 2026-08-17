"""TM.0.24.PLASTICITYMAP provenance and smoke. CPU only. Scoring requires runner.lock."""

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
        "docs/lineage_plasticitymap_contract.md",
        "docs/lineage_plasticitymap.prereg.lock",
        "docs/cortex.candidate.v28.lock",
        "docs/lineage_reach.lock",
        "docs/lineage_wallmap.decision.lock",
        "experiments/run_tm024plasticitymap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_plasticitymap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["D0"]["later_behavior_requires_both"] is True
    assert prereg["D2"]["not_teaching_the_answer"] is True
    assert prereg["D2"]["equal_opportunities"] is True
    assert prereg["D6"]["score_only_if_D1_D2_D3_pass"] is True
    assert "TM024.WALLMAP" not in prereg["domains"]["FORCE"]
    assert "TM024.REACH" not in prereg["domains"]["FORCE"]


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_plasticitymap_contract.md").read_text(encoding="utf-8")
    assert "Forced balanced ACT exposure" in text
    assert "This is not teaching the answer" in text
    assert "0.0.004" in text
    assert "n stays **64**" in text
    cand = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v28.lock").read_text(encoding="utf-8"))
    assert cand["version"] == "TM.0.23.CORTEX.CANDIDATE.V28"
    assert cand["genome"]["n"] == 64
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == (
        "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
    )


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_plasticitymap.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024plasticitymap import pmap_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["n"] == 64
    live = pmap_shas()
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


def test_smoke() -> None:
    from experiments.run_tm024plasticitymap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["capability_claim"] is False
    assert out["product"] == "0.0.004"
    assert out["n"] == 64
    assert out["readout_set_ok"] is True


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_plasticitymap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["increase_n"] is False
    assert d["another_lineage_run"] is False


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_smoke()
    test_decision_if_present()
    print("test_tm024plasticitymap: ok")


if __name__ == "__main__":
    main()
