"""TM.0.24.AFFINEMAP provenance. LIFECYCLEMARGINMAP freeze preserved. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
LMM_DEC = "851d4a9312a7a8164600f53b857f65d3f50fc22fba136e52f00d3266422ddff0"
LMM_DEV = "57015fef334b533a77173bb06323e3f28e8d9bc5ad41e3453bfe126ee4a34bf8"
LMM_ADD = "d4dd4ca797d4c6c0aff6725fa79723abd870491a59f4cae41f73ca03fd75f794"
LMM_RUNNER_LOCK = "d0e5eee16752a7ae89bdfc16f3e0294ce14cfd4726aa8e71f7a8eba1c7c848dd"
LMM_RUNNER_PY = "5f7dc1a79e49c42edc45ccd7d12d4c4a8d2989a067071becc433b46c6234ddce"
LMM_PREREG = "4a753d811a14b428321f88767ce5d018a42dc047eccaa959ab83ed9f4c1ee8e2"
R2_DEC = "484c38d90582b650633e76a9a92481022a5d3c97308c72e8d51d30d6c9b266dd"
R2_DEV = "9321e57bb4f3bd1f4fe108c8fcb7751eca4fdb9da3d23401da5e5e2abd09eaed"
R2_ADD = "92321043267e95092863e1d6e0ac08256d36cdf313c11078f352471fb25c7228"
R2_RUNNER_LOCK = "c05c9254e9e1b1d6b6039d7cee43b487f83a53b2d1c0b46b60feb039dd6a1077"
R2_RUNNER_PY = "30f3c4ee67fb4e6524088cc545232682ea7c189758c06d11db0a80428af825a2"
R2_PREREG = "d1063d354cbe4162355c377d8e7a42cf6508035c9f7c6273291096477c8a2924"
V1_RUNNER_LOCK_SHA = "28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593"
V1_RUNNER_PY_SHA = "edec8809938f3f1ab77948feb3661bea4fc3e6bb1abf573a81089ae628dfc974"
CVG_DEC = "6de34e295b54ef51c2684b0ce1cf7295064300db75e9e7ee258c7bb95665072d"
CVG_ADD = "04a5e91cdc23839c9e3c954dc8c921f902c092de14acd906ada7caa678a6b083"
CVG_RUNNER_PY = "232cffa23619de1fcdbde7b8c82fc3de8e1c2fbe84a014a40bc27f3723cbbcf6"
D1_RUNNER_PY = "06f5f2c6edc0dffef570e75295708ea2816ea737cd0af9dab157cd94f4c26b41"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
NEURAL = "90dae9a652fdefa0e7942b775053e2b765991eb26ee3ee4cdcfdb994b901d9e0"
MEMORY = "fc3942efaffb8b18e891c545510aa4949b52c86c773c707036bbc6d162fe35d7"
RUNNER_LOCK_SHA = "5cd319ecc1872dadcc1e193f05a1794dd850c2d37d8500fa11e2bf96b3577669"
RUNNER_SHA = "be7a360ef8b635085b7cd22490812c311ad2335c463054a7471e443dec664eea"
MANIFEST_SHA = "e8acd76acf61c62d487487d94a3ea8acdf8a032a61982a1d7d17629d3c600e59"
EXPECTED_N_CELLS = 104
ARMS = ("A0", "A1", "A2", "A3")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_freeze_preserved() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.lock") == LMM_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock") == LMM_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.addendum.lock") == LMM_ADD
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.runner.lock") == LMM_RUNNER_LOCK
    assert sha(REPO_ROOT / "experiments" / "run_tm024lifecyclemarginmap.py") == LMM_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.prereg.lock") == LMM_PREREG
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.lock") == R2_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.dev.lock") == R2_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.addendum.lock") == R2_ADD
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.runner.lock") == R2_RUNNER_LOCK
    assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap_r2.py") == R2_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock") == V1_RUNNER_LOCK_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py") == V1_RUNNER_PY_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py") == D1_RUNNER_PY
    assert sha(REPO_ROOT / "experiments" / "run_tm024convergencemap.py") == CVG_RUNNER_PY
    assert sha(REPO_ROOT / "three_memory" / "neural_cortex.py") == NEURAL
    assert sha(REPO_ROOT / "three_memory" / "cortex_memory.py") == MEMORY


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_affinemap.contract.md",
        "docs/lineage_affinemap.prereg.lock",
        "docs/lineage_affinemap.isolation.lock",
        "docs/lineage_lifecyclemarginmap.decision.addendum.lock",
        "docs/lineage_lifecyclemarginmap.decision.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024affinemap.py",
        "experiments/run_tm024lifecyclemarginmap.py",
        "experiments/run_tm024discrimmap_r2.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["act_score_mode"] == "query"
    assert prereg["no_new_eta_grid"] is True
    assert prereg["no_new_match_radius"] is True
    assert prereg["no_new_replay_budget"] is True
    assert prereg["no_new_margin_threshold"] is True
    assert prereg["no_lifecycle_changes"] is True
    assert prereg["skip_eco_spec"] is True
    assert prereg["do_not_investigate_consolidation"] is True
    assert prereg["a3_implementation_authorized"] is False
    assert prereg["a3_not_an_instinct"] is True
    assert prereg["domains"]["DEV"] == "TM024.AFFINEMAP.DEV."
    assert prereg["domains"]["EXTRACT"] == "TM024.LIFECYCLEMARGINMAP.DEV."
    assert prereg["arms"]["A0"]["replay_learner"] == "d1_hard_margin_affine"
    assert prereg["arms"]["A1"]["replay_learner"] == "d1_hard_margin_homogeneous"
    assert prereg["arms"]["A2"]["replay_learner"] == "c3_passive_aggressive"
    assert prereg["arms"]["A3"]["replay_learner"] == "c3_passive_aggressive_local_bias"
    assert prereg["learner"]["d1_hard_margin_affine"]["intercept"] is True
    assert prereg["learner"]["d1_hard_margin_homogeneous"]["intercept"] is False
    assert prereg["learner"]["c3_passive_aggressive"]["intercept"] is False
    assert prereg["first_pa_fail"] == "m1_four_cue_acquire_ranking"
    assert prereg["immediate_wall"] == "online_acquisition_geometry"
    assert prereg["historical_lmm_prereg_sha"] == LMM_PREREG
    assert prereg["decision_ladder"][0]["then"] == "affine_intercept_required"
    assert prereg["decision_ladder"][1]["then"] == "online_optimization_failure"
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    iso = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.isolation.lock").read_text(encoding="utf-8"))
    assert iso["rescore_historical_worlds"] is False
    assert iso["do_not_investigate_consolidation"] is True
    assert iso["a3_implementation_authorized"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock") == CVG_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock") == CVG_ADD
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    runner_p = REPO_ROOT / "docs" / "lineage_affinemap.runner.lock"
    man_p = REPO_ROOT / "docs" / "lineage_affinemap.manifest.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024affinemap.py") == RUNNER_SHA
            assert runner["shas"]["runner"] == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["domain"] == "TM024.AFFINEMAP.DEV."
        assert runner["skip_eco_spec"] is True
        assert runner["a3_diagnostic_only"] is True
        assert runner["historical_lmm_decision_sha"] == LMM_DEC
        assert runner["shas"]["lmm_prereg"] == LMM_PREREG
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        if MANIFEST_SHA is not None:
            assert sha(man_p) == MANIFEST_SHA
        assert man["expected_n_cells"] == EXPECTED_N_CELLS
        assert len(man["expected_cell_ids"]) == EXPECTED_N_CELLS
        assert len(set(man["expected_cell_ids"])) == EXPECTED_N_CELLS
        assert "eco|" not in json.dumps(man["expected_cell_ids"])
        assert "spec|" not in json.dumps(man["expected_cell_ids"])


def _acq(arm: str, n_cues: int, passed: bool, world: int, order: str) -> dict[str, object]:
    return {
        "arm": arm,
        "kind": "acquire",
        "n_cues": n_cues,
        "order": order,
        "world": world,
        "passed": passed,
        "ranking_ok": passed,
        "id": f"acquire|{arm}|c{n_cues}|{order}|w{world}",
        "ceiling_only": arm in ("A0", "A1"),
        "a3_diagnostic_only": arm == "A3",
        "n_live_reversal_updates": 0,
        "margin_trajectory": {"affine_b": 0.1 if arm == "A0" else 0.0},
        "bounded_match_sanity_used_for_pass": False,
        "domain": "TM024.AFFINEMAP.DEV.",
    }


def _grid(a0_4: bool, a0_8: bool, a1_4: bool, a1_8: bool, a2_4: bool, a2_8: bool, a3_4: bool, a3_8: bool) -> list[dict[str, object]]:
    flags = {
        "A0": {2: True, 4: a0_4, 8: a0_8},
        "A1": {2: True, 4: a1_4, 8: a1_8},
        "A2": {2: True, 4: a2_4, 8: a2_8},
        "A3": {2: True, 4: a3_4, 8: a3_8},
    }
    cells: list[dict[str, object]] = []
    for arm in ARMS:
        for n in (2, 4, 8):
            for wi in range(2):
                for order in ("A_then_B", "B_then_A"):
                    cells.append(_acq(arm, n, bool(flags[arm][n]), wi, order))
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024affinemap import _decision, load_prereg

    p = load_prereg()
    code, then, extra = _decision(_grid(True, True, False, False, False, False, True, True), p)
    assert code == "affine_intercept_required"
    assert then == "affine_intercept_required"
    assert extra["A0_acquire_4"] is True
    assert extra["A1_acquire_4"] is False
    code, _then, _extra = _decision(_grid(True, True, True, True, False, False, False, False), p)
    assert code == "online_optimization_failure"
    code, _then, _extra = _decision(_grid(True, True, True, True, True, True, True, True), p)
    assert code == "apparatus_inconsistency"
    code, _then, _extra = _decision(_grid(True, True, True, True, False, True, True, True), p)
    assert code == "online_optimization_failure"
    code, _then, _extra = _decision(_grid(True, True, True, True, False, False, True, True), p)
    assert code == "online_optimization_failure"
    code, _then, _extra = _decision(_grid(False, False, False, False, False, False, False, False), p)
    assert code == "d1_ceiling_reaudit"
    code, _then, _extra = _decision(_grid(True, False, True, False, False, False, False, False), p)
    assert code == "d1_ceiling_reaudit"
    code, _then, extra = _decision(_grid(True, True, False, True, False, False, True, True), p)
    assert code == "affine_intercept_required"
    assert extra["m1_previously_failed_four_cue_acquire"] is True


def test_homogeneous_solver_vs_affine_intercept() -> None:
    from experiments.run_tm024affinemap import hard_margin_homogeneous, load_prereg
    from experiments.run_tm024discrimmap_r2 import hard_margin_linear

    p = load_prereg()
    aff_spec = p["learner"]["d1_hard_margin_affine"]
    hom_spec = p["learner"]["d1_hard_margin_homogeneous"]
    x = np.zeros((2, 64), dtype=np.float64)
    x[0, 0] = 1.0
    x[1, 0] = 0.4
    y = np.asarray([1.0, -1.0], dtype=np.float64)
    aff = hard_margin_linear(x, y, aff_spec)
    hom = hard_margin_homogeneous(x, y, hom_spec)
    assert aff["status"] == "optimal"
    assert abs(float(aff["b"])) > 1e-6
    assert hom["status"] == "infeasible"

    x2 = np.zeros((2, 64), dtype=np.float64)
    x2[0, 0] = 1.0
    x2[1, 0] = -1.0
    aff2 = hard_margin_linear(x2, y, aff_spec)
    hom2 = hard_margin_homogeneous(x2, y, hom_spec)
    assert aff2["status"] == "optimal"
    assert hom2["status"] == "optimal"
    assert abs(float(hom2["b"])) == 0.0


def test_pa_has_no_bias_a3_starts_at_zero() -> None:
    from experiments.run_tm024affinemap import PassiveAggressiveBias, make_learner, load_prereg
    from experiments.run_tm024convergencemap import PassiveAggressive
    from experiments.run_tm024eligmap import unit_or_zero

    p = load_prereg()
    handles = ["h1", "h2"]
    a2 = make_learner("A2", handles, p)
    a3 = make_learner("A3", handles, p)
    assert isinstance(a2, PassiveAggressive)
    assert isinstance(a3, PassiveAggressiveBias)
    assert a3.bias == {"h1": 0.0, "h2": 0.0}
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    a2.update(x, "h1", 1.0)
    assert a2.n_updates >= 1
    assert not hasattr(a2, "bias")
    a3.update(x, "h1", 1.0)
    assert a3.n_updates >= 1
    assert set(a3.bias) == {"h1", "h2"}


def test_cell_ids() -> None:
    from experiments.run_tm024affinemap import expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert all(not i.startswith("eco|") and not i.startswith("spec|") for i in ids)
    assert "acquire|A0|c4|A_then_B|w0" in ids
    assert "acquire|A1|c8|B_then_A|w1" in ids
    assert "twin|A3|c2|A_then_B|w1" in ids


def test_oracles_on_synthetic_rows() -> None:
    from experiments.run_tm024affinemap import HomogeneousOracle, load_prereg
    from experiments.run_tm024eligmap import unit_or_zero
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle

    p = load_prereg()
    h1, h2 = "h1", "h2"
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(-(np.arange(64, dtype=np.float64) + 1.0))
    rows = [
        {"p1": x, "handle": h1, "adv": 1.0},
        {"p1": y, "handle": h2, "adv": 1.0},
    ]
    aff = HardMarginOracle([h1, h2], p["learner"]["d1_hard_margin_affine"])
    hom = HomogeneousOracle([h1, h2], p["learner"]["d1_hard_margin_homogeneous"])
    aff.fit(rows)
    hom.fit(rows)
    assert aff.status == "optimal"
    assert hom.status == "optimal"
    assert max(aff.scores(x), key=lambda h: aff.scores(x)[h]) == h1
    assert max(hom.scores(x), key=lambda h: hom.scores(x)[h]) == h1


def test_smoke() -> None:
    from experiments.run_tm024affinemap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert "affine_b" in out["a0_traj_keys"]
    assert "pre_rest_min_probe_margin" in out["a0_traj_keys"]
    assert out["skip_eco_spec"] is True
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["a3_diagnostic_only"] is True


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024affinemap import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

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


def test_dev_unopened() -> None:
    assert not (REPO_ROOT / "docs" / "lineage_affinemap.dev.lock").exists()
    assert not (REPO_ROOT / "docs" / "lineage_affinemap.decision.lock").exists()


if __name__ == "__main__":
    test_historical_freeze_preserved()
    test_phase_a_files()
    test_decision_ladder()
    test_homogeneous_solver_vs_affine_intercept()
    test_pa_has_no_bias_a3_starts_at_zero()
    test_cell_ids()
    test_oracles_on_synthetic_rows()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_dev_unopened()
    print("ok")
