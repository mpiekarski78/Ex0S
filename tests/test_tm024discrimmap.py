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
RUNNER_LOCK_SHA = "ae0dd8752341b2b727453010bcef6b380425b03a59c36c7c788bb40c7cff8c88"
RUNNER_SHA = "9167437c33224cf35ce065a58c56afdb2e14dc5f6ca0677e8f39588a0c37f7c3"
DEV_LOCK_SHA = None  # pinned after DEV lock lands
DEV_MANIFEST_SHA = None
EXPECTED_N_RANK = 240
EXPECTED_N_TWIN = 40
EXPECTED_N_CELLS = 280
ADDRESSES = ["E0", "E1", "Edelta", "Elam_0.9"]
ARMS = ("D0", "D1", "D2", "D3", "D4")
CUE_COUNTS = (2, 4, 8)
ORDERS = ("A_then_B", "B_then_A")
WORLDS = (0, 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_discrimmap_contract.md",
        "docs/lineage_discrimmap.prereg.lock",
        "docs/lineage_discrimmap.isolation.lock",
        "docs/lineage_discrimmap.runner.lock",
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
    runner = json.loads((REPO_ROOT / "docs" / "lineage_discrimmap.runner.lock").read_text(encoding="utf-8"))
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.runner.lock") == RUNNER_LOCK_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm024discrimmap.py") == RUNNER_SHA
    assert runner["shas"]["runner"] == RUNNER_SHA
    assert runner["n"] == 64
    assert runner["expected_n_cells"] == EXPECTED_N_CELLS
    assert runner["expected_n_rank"] == EXPECTED_N_RANK
    assert runner["expected_n_twin"] == EXPECTED_N_TWIN
    assert runner["geometric_margin_min"] == 0.01
    assert runner["reject_raw_linear_margin"] is True
    assert runner["score_reserved_unopened"] is True
    assert runner["neural_edit"] is False
    assert runner["implementation_authorized"] is False
    assert runner["domain"] == "TM024.DISCRIMMAP.DEV."
    assert runner["twin_domain"] == "TM024.DISCRIMMAP.TWIN."
    assert runner["score_domain"] == "TM024.DISCRIMMAP.SCORE."
    assert runner["arms"] == list(ARMS)
    assert runner["addresses"] == ADDRESSES


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
    from experiments.run_tm024discrimmap import assert_runner_frozen, refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    refuse_dev_lock()
    lock = assert_runner_frozen()
    assert lock["expected_n_cells"] == EXPECTED_N_CELLS
    assert lock["geometric_margin_min"] == 0.01


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_discrimmap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["neural_edit"] is False
    assert d["implementation_authorized"] is False
    assert d["candidate_v31"] is False
    assert d["candidate_v32"] is False
    assert d["lineage_reopened"] is False
    assert d["eligibility_budget_installed"] is False
    assert d["declared_budget_remains_closed"] == 1536
    assert d["n"] == 64
    assert d["scored_worlds"] is False
    assert "TM024.DISCRIMMAP.SCORE." not in json.dumps(d)
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v32.lock").exists()


def discrimmap_manifest(dev: dict) -> str:
    rows = []
    for c in dev["cells"]:
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "address": c["address"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "order": c["order"],
                "world": c["world"],
                "passed": c["passed"],
                "ranking_ok": c["ranking_ok"],
                "train_ranking_ok": c["train_ranking_ok"],
                "train_geometric_margin": round(float(c["train_geometric_margin"]), 12),
                "probe_geometric_margin": round(float(c["probe_geometric_margin"]), 12),
                "perturb_stable": c["perturb_stable"],
            }
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_discrimmap.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    raw = p.read_bytes()
    if DEV_LOCK_SHA is not None:
        assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert dev["n_rank"] == EXPECTED_N_RANK
    assert dev["n_twin"] == EXPECTED_N_TWIN
    assert len(dev["cells"]) == EXPECTED_N_CELLS
    ids = [c["id"] for c in dev["cells"]]
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert dev["product"] == "0.0.004"
    assert dev["earned_next"] is False
    assert dev["neural_edit"] is False
    assert dev["implementation_authorized"] is False
    assert dev["n"] == 64
    assert dev["score_domain_opened"] is False
    assert dev["domain"] == "TM024.DISCRIMMAP.DEV."
    assert dev["twin_domain"] == "TM024.DISCRIMMAP.TWIN."
    blob = json.dumps(dev)
    assert "TM024.DISCRIMMAP.SCORE." not in blob
    assert Counter(c["arm"] for c in dev["cells"]) == {a: 56 for a in ARMS}
    assert Counter(c["address"] for c in dev["cells"]) == {a: 70 for a in ADDRESSES}
    assert Counter(c["kind"] for c in dev["cells"]) == {"rank": EXPECTED_N_RANK, "twin": EXPECTED_N_TWIN}
    assert Counter(c["n_cues"] for c in dev["cells"] if c["kind"] == "rank") == {2: 80, 4: 80, 8: 80}
    assert Counter(c["order"] for c in dev["cells"]) == {o: 140 for o in ORDERS}
    assert Counter(c["world"] for c in dev["cells"] if c["kind"] == "rank") == {w: 120 for w in WORLDS}
    assert all(c["n_handles"] == 2 for c in dev["cells"])
    assert all(c["n_train"] == c["n_cues"] and c["n_probe"] == c["n_cues"] for c in dev["cells"])
    gmin = 0.01
    for c in dev["cells"]:
        if c["arm"] == "D0":
            expected = bool(c["ranking_ok"] and float(c["probe_geometric_margin"]) >= gmin)
        else:
            expected = bool(
                c["train_ranking_ok"]
                and c["ranking_ok"]
                and float(c["train_geometric_margin"]) >= gmin
                and float(c["probe_geometric_margin"]) >= gmin
                and c["perturb_stable"]
            )
        assert bool(c["passed"]) is expected, c["id"]
        assert c["domain"] in ("TM024.DISCRIMMAP.DEV.", "TM024.DISCRIMMAP.TWIN.")
        if c["kind"] == "twin":
            assert c["n_cues"] == 2
            assert c["world"] == 0
            assert c["domain"] == "TM024.DISCRIMMAP.TWIN."
        else:
            assert c["domain"] == "TM024.DISCRIMMAP.DEV."
    if DEV_MANIFEST_SHA is not None:
        assert discrimmap_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_geometric_margin_scale_invariant()
    test_smoke()
    test_score_and_dev_lock_refused()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024discrimmap: ok")


if __name__ == "__main__":
    main()
