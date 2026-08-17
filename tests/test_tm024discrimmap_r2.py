"""TM.0.24.DISCRIMMAP.R2 provenance. Runner-only. No neural edit. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
HIST_DEC = "183b9a5f582f2c645adfe2df1424af92072255a59ad0cf8ec6358325f42083b0"
HIST_DEV = "4c0c8677e7e95f18b9adaa788ec941f5cf17cb62bc486bca4099a1f2b5dcea13"
HIST_RUN = "ae0dd8752341b2b727453010bcef6b380425b03a59c36c7c788bb40c7cff8c88"
HIST_RUNNER_PY = "9167437c33224cf35ce065a58c56afdb2e14dc5f6ca0677e8f39588a0c37f7c3"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "8605f6c9508667b8e02e2f87bef7bdd4fed6f2f2d30c73b89f61a8e289670abf"
RUNNER_SHA = "06f5f2c6edc0dffef570e75295708ea2816ea737cd0af9dab157cd94f4c26b41"
DEV_LOCK_SHA = "99f9097a3f30df7b69d46d5c20f858e35f540faa387c8e9bcb9062db07ab3831"
DEV_MANIFEST_SHA = "9f6e7e38a56456bb25acd2dc3a9ec936c6ea033897beefd0d88f440d5a2b51f5"
DECISION_SHA = "a888a8af5495e19139f8691725241c2b31322f192218983d02c131ebf6c84675"
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
        "docs/lineage_discrimmap.r2.contract.md",
        "docs/lineage_discrimmap.r2.prereg.lock",
        "docs/lineage_discrimmap.r2.isolation.lock",
        "docs/lineage_discrimmap.r2.runner.lock",
        "docs/lineage_discrimmap.r2.dev.lock",
        "docs/lineage_discrimmap.r2.decision.lock",
        "docs/lineage_discrimmap.decision.addendum.lock",
        "docs/lineage_discrimmap.decision.lock",
        "docs/lineage_discrimmap.dev.lock",
        "docs/lineage_discrimmap.runner.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024discrimmap_r2.py",
        "experiments/run_tm024discrimmap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_discrimmap.r2.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["implementation_authorized"] is False
    assert prereg["domains"]["DEV"] == "TM024.DISCRIMMAP.R2.DEV."
    assert prereg["domains"]["TWIN"] == "TM024.DISCRIMMAP.R2.TWIN."
    assert prereg["score_reserved_unopened"] is True
    assert prereg["rescore_historical_worlds"] is False
    assert prereg["arms"]["D1"]["soft_margin"] is False
    assert prereg["arms"]["D1"]["soft_margin_C"] is None
    assert prereg["arms"]["D1"]["no_automatic_soft_margin"] is True
    assert prereg["arms"]["D1"]["method"] == "sv_subset_kkt_enumeration"
    assert prereg["arms"]["D2"]["y_encoding"] == [-1, 1]
    assert prereg["arms"]["D2"]["intercept"] is True
    assert prereg["arms"]["D2"]["intercept_in_norm"] is False
    assert prereg["arms"]["D2"]["row_unit_l2"] is True
    assert prereg["arms"]["D2"]["mean_center"] is False
    assert prereg["arms"]["D3"]["eta"] == 0.15
    assert prereg["arms"]["D3"]["epochs"] == 1
    assert prereg["arms"]["D3"]["shuffle"] is False
    assert prereg["arms"]["D3"]["init"] == "zeros"
    assert prereg["arms"]["D3"]["c_max"] == 1.0
    assert prereg["arms"]["D3"]["error_only"] is False
    assert prereg["arms"]["D3"]["pool_orders"] is False
    assert prereg["arms"]["D4"]["rbf_gamma"] == 0.5
    assert prereg["arms"]["D4"]["lambda"] == 0.01
    assert prereg["arms"]["D4"]["v_eligible"] is False
    assert prereg["margin"]["w_norm_excludes_intercept"] is True
    assert prereg["margin"]["geometric_margin_min"] == 0.01
    assert prereg["margin"]["require_correct_classification"] is True
    assert prereg["margin"]["training_only_pass_is_interpolation"] is True
    assert prereg["decision_ladder"][-1]["then"] == "robust_linear_boundary_absent"
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.decision.lock") == HIST_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.dev.lock") == HIST_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.runner.lock") == HIST_RUN
    assert sha(REPO_ROOT / "experiments" / "run_tm024discrimmap.py") == HIST_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    runner_p = REPO_ROOT / "docs" / "lineage_discrimmap.r2.runner.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py") == RUNNER_SHA
            assert runner["shas"]["runner"] == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["d1"]["no_automatic_soft_margin"] is True
        assert runner["d3"]["epochs"] == 1
        assert runner["d4"]["rbf_gamma"] == 0.5
        assert runner["w_norm_excludes_intercept"] is True
        assert runner["domain"] == "TM024.DISCRIMMAP.R2.DEV."


def test_hard_margin_and_geometry() -> None:
    import numpy as np
    from experiments.run_tm024discrimmap_r2 import geometric_margin, hard_margin_linear, load_prereg

    spec = load_prereg()["arms"]["D1"]
    x = np.arange(64, dtype=np.float64) + 1.0
    x = x / np.linalg.norm(x)
    g1 = geometric_margin(x, 0.0, x, 1.0)
    g10 = geometric_margin(10.0 * x, 0.0, x, 1.0)
    gb = geometric_margin(x, 100.0, x, 1.0)
    assert abs(g1 - g10) <= 1e-12
    assert abs(g1 - 1.0) <= 1e-9
    assert abs(gb - 101.0) <= 1e-9
    sep = hard_margin_linear(np.stack([x, -x]), np.asarray([1.0, -1.0]), spec)
    assert sep["status"] == "optimal"
    inf = hard_margin_linear(np.stack([x, x]), np.asarray([1.0, -1.0]), spec)
    assert inf["status"] == "infeasible"
    assert spec["soft_margin"] is False
    assert spec["soft_margin_C"] is None


def test_smoke() -> None:
    from experiments.run_tm024discrimmap_r2 import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["geometric_margin_scale_invariant"] is True
    assert out["geometric_margin_excludes_bias"] is True
    assert out["hard_margin_separable_status"] == "optimal"
    assert out["hard_margin_infeasible_status"] == "infeasible"
    assert out["no_soft_margin"] is True
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["product"] == "0.0.004"


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024discrimmap_r2 import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    if not RUNNER_LOCK.exists():
        try:
            refuse_dev_lock()
        except RuntimeError as e:
            assert "runner.lock" in str(e)
        else:
            raise AssertionError("DEV lock must wait for runner.lock")
        return
    if DEV_LOCK.exists():
        try:
            refuse_dev_lock()
        except RuntimeError as e:
            assert "again" in str(e)
        else:
            raise AssertionError("same frozen DEV execution must be refused")
        return
    refuse_dev_lock()


def test_historical_immutable() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.decision.lock") == HIST_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.dev.lock") == HIST_DEV
    add = json.loads((REPO_ROOT / "docs" / "lineage_discrimmap.decision.addendum.lock").read_text(encoding="utf-8"))
    assert add["rewrite_historical_decision"] is False
    assert add["rescore_historical_worlds"] is False
    assert add["next"] == "TM.0.24.DISCRIMMAP.R2"
    assert add["historical_decision_sha"] == HIST_DEC


def r2_manifest(dev: dict) -> str:
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
                "solver_status": c["solver_status"],
                "ranking_ok": c["ranking_ok"],
                "train_ranking_ok": c["train_ranking_ok"],
                "train_clean": c["train_clean"],
                "train_geometric_margin": round(float(c["train_geometric_margin"]), 12),
                "probe_geometric_margin": round(float(c["probe_geometric_margin"]), 12),
                "perturb_stable": c["perturb_stable"],
            }
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_discrimmap.r2.decision.lock"
    if not p.exists():
        return
    if DECISION_SHA is not None:
        assert sha(p) == DECISION_SHA
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["neural_edit"] is False
    assert d["implementation_authorized"] is False
    assert d["candidate_v31"] is False
    assert d["historical_discrimmap_preserved"] is True
    assert d["decision"]["code"] == "robust_linear_boundary_absent"
    assert d["decision"]["d1_robust"] is False
    assert d["decision"]["d3_robust"] is False
    assert d["decision"]["d4_robust"] is False
    assert d["decision"]["d1_train_clean"] is False
    assert d["dev_lock_sha"] == DEV_LOCK_SHA
    assert "TM024.DISCRIMMAP.R2.SCORE." not in json.dumps(d)
    assert "TM024.DISCRIMMAP.SCORE." not in json.dumps(d)


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_discrimmap.r2.dev.lock"
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
    ids = [c["id"] for c in dev["cells"]]
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.DISCRIMMAP.R2.DEV."
    assert dev["twin_domain"] == "TM024.DISCRIMMAP.R2.TWIN."
    assert dev["score_domain_opened"] is False
    blob = json.dumps(dev)
    assert "TM024.DISCRIMMAP.R2.SCORE." not in blob
    assert "TM024.DISCRIMMAP.SCORE." not in blob
    assert Counter(c["arm"] for c in dev["cells"]) == {a: 56 for a in ARMS}
    assert Counter(c["address"] for c in dev["cells"]) == {a: 70 for a in ADDRESSES}
    assert Counter(c["kind"] for c in dev["cells"]) == {"rank": EXPECTED_N_RANK, "twin": EXPECTED_N_TWIN}
    assert Counter(c["n_cues"] for c in dev["cells"] if c["kind"] == "rank") == {2: 80, 4: 80, 8: 80}
    assert Counter(c["order"] for c in dev["cells"]) == {o: 140 for o in ORDERS}
    gmin = 0.01
    for c in dev["cells"]:
        assert c["n_train"] == c["n_cues"] and c["n_probe"] == c["n_cues"]
        assert c["solver_status"] in ("optimal", "infeasible", "not_applicable")
        if c["arm"] == "D0":
            expected = bool(c["ranking_ok"] and float(c["probe_geometric_margin"]) >= gmin)
        else:
            expected = bool(
                c["solver_status"] == "optimal"
                and c["train_ranking_ok"]
                and c["ranking_ok"]
                and float(c["train_geometric_margin"]) >= gmin
                and float(c["probe_geometric_margin"]) >= gmin
                and c["perturb_stable"]
            )
        assert bool(c["passed"]) is expected, c["id"]
        if c["arm"] == "D1" and c["solver_status"] == "infeasible":
            assert c["passed"] is False
            assert c["train_clean"] is False
    if DEV_MANIFEST_SHA is not None:
        assert r2_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_phase_a_files()
    test_hard_margin_and_geometry()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_historical_immutable()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024discrimmap_r2: ok")


if __name__ == "__main__":
    main()
