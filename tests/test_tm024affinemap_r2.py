"""TM.0.24.AFFINEMAP.R2 provenance. V1 4a5183e preserved. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
V1_GIT = "4a5183eedc45417fd29cd639c3f2fcb4c4a87ad3"
V1_RUNNER_LOCK = "5cd319ecc1872dadcc1e193f05a1794dd850c2d37d8500fa11e2bf96b3577669"
V1_RUNNER_PY = "be7a360ef8b635085b7cd22490812c311ad2335c463054a7471e443dec664eea"
V1_PREREG = "746faa55139c010d265142a055903f898b3c26730e529c93d607a904061017d5"
V1_ISOLATION = "386fe75eece7e8d48c7088de6d4a70f2446e737d7c3aceb670ab795733fc294a"
V1_CONTRACT = "025ffaa181434548d1e29bad4901266f2c0f346e7d8be48671e5f96ce002d8ba"
V1_MANIFEST = "e8acd76acf61c62d487487d94a3ea8acdf8a032a61982a1d7d17629d3c600e59"
LMM_DEC = "851d4a9312a7a8164600f53b857f65d3f50fc22fba136e52f00d3266422ddff0"
LMM_DEV = "57015fef334b533a77173bb06323e3f28e8d9bc5ad41e3453bfe126ee4a34bf8"
LMM_ADD = "d4dd4ca797d4c6c0aff6725fa79723abd870491a59f4cae41f73ca03fd75f794"
NEURAL = "90dae9a652fdefa0e7942b775053e2b765991eb26ee3ee4cdcfdb994b901d9e0"
MEMORY = "fc3942efaffb8b18e891c545510aa4949b52c86c773c707036bbc6d162fe35d7"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "636bd7d68a95b43e0ba9b5d4b4ff0d31c08071bbd03e97603da1ca6cf81b4f46"
RUNNER_SHA = "d6289b9b35da2377fa4ef0ee704ecd2b064044dc9ef0f1b43f7723fcba89c9db"
MANIFEST_SHA = "ce1c83435458da32b4892279204283c2979a55189ee1d280a9b2bfd9cd73cf3c"
DEV_LOCK_SHA = "3ca78f2ec5a383ee38ea644ef476f8308f302b75d59becfe34b861ca0fa761ef"
DECISION_SHA = "dc71c767858cc3fbc982a8fc6a3736c15d33f107087878810942ec4182b323bf"
ADDENDUM_SHA = "d6a2cdbf0869ff7a78b6c2fe2f1d20b6e31011c1333b464f006cfc34b45b7bc4"
DEV_MANIFEST_SHA = "7aedf8c4e7ce1602939e46ca1a7a392de4d0f268f1a92b3f9f2e417a838cb1d1"
FREEZE_GIT = "7f2d8e1b066331bfeb66786f8e7c1e414802491d"
EXPECTED_N_CELLS = 104
RECORDED_FIELDS = (
    "ranking_ok",
    "pairwise_score_gap",
    "normalized_geometric_margin",
    "perturbation_ok",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_freeze_preserved() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.runner.lock") == V1_RUNNER_LOCK
    assert sha(REPO_ROOT / "experiments" / "run_tm024affinemap.py") == V1_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.prereg.lock") == V1_PREREG
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.isolation.lock") == V1_ISOLATION
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.contract.md") == V1_CONTRACT
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.manifest.lock") == V1_MANIFEST
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.lock") == LMM_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock") == LMM_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.addendum.lock") == LMM_ADD
    assert sha(REPO_ROOT / "three_memory" / "cortex_memory.py") == MEMORY
    src = (REPO_ROOT / "three_memory" / "neural_cortex.py").read_text(encoding="utf-8")
    if (REPO_ROOT / "docs" / "cortex_v32.prereg.lock").exists() and "EPISODE_SLOTS" in src:
        devp = REPO_ROOT / "docs" / "lineage_affinemap.r2.dev.lock"
        if devp.exists():
            assert json.loads(devp.read_text(encoding="utf-8"))["shas"]["neural_cortex"] == NEURAL
    else:
        assert sha(REPO_ROOT / "three_memory" / "neural_cortex.py") == NEURAL
    assert not (REPO_ROOT / "docs" / "lineage_affinemap.dev.lock").exists()
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()


def test_phase_a_files() -> None:
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["open_v1_dev"] is False
    assert prereg["rewrite_historical_affinemap_v1"] is False
    assert prereg["historical_affinemap_v1_git_head"] == V1_GIT
    assert prereg["margin"]["pass_statistic"] == "normalized_geometric_margin"
    assert prereg["margin"]["pairwise_score_gap_is_not_pass_statistic"] is True
    assert prereg["common_score"]["homogeneous_geometric_margin_cannot_exceed_1"] is True
    assert prereg["common_score"]["a0_vs_a1"] == "intercept_only"
    assert prereg["common_score"]["a1_vs_a2"] == "same_homogeneous_class_and_common_score_statistic"
    assert prereg["common_score"]["a2_vs_a3"] == "learned_bias_only"
    assert prereg["common_score"]["a3_raw_gap_increase_is_not_bias_support"] is True
    assert prereg["domains"]["DEV"] == "TM024.AFFINEMAP.R2.DEV."
    assert prereg["recorded_fields"] == list(RECORDED_FIELDS)
    assert prereg["phased_contract"]["lifecycle_stability_gate"] == (
        "normalized_geometric_margin_0.01_and_perturbation"
    )
    assert "normalized geometric margin 0.01" in prereg["decision_ladder"][3]["when"]
    assert "raw pairwise-gap increase is not sufficient" in prereg["decision_ladder"][3]["when"]
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    runner_p = REPO_ROOT / "docs" / "lineage_affinemap.r2.runner.lock"
    man_p = REPO_ROOT / "docs" / "lineage_affinemap.r2.manifest.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024affinemap_r2.py") == RUNNER_SHA
            assert runner["shas"]["runner"] == RUNNER_SHA
        assert runner["pass_statistic"] == "normalized_geometric_margin"
        assert runner["pairwise_score_gap_is_not_pass_statistic"] is True
        assert runner["recorded_fields"] == list(RECORDED_FIELDS)
        assert runner["domain"] == "TM024.AFFINEMAP.R2.DEV."
        assert runner["historical_affinemap_v1_git_head"] == V1_GIT
        assert runner["shas"]["v1_runner"] == V1_RUNNER_PY
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        if MANIFEST_SHA is not None:
            assert sha(man_p) == MANIFEST_SHA
        assert man["expected_n_cells"] == EXPECTED_N_CELLS
        assert len(man["expected_cell_ids"]) == EXPECTED_N_CELLS


def test_gap_is_not_gamma() -> None:
    from experiments.run_tm024affinemap_r2 import CommonTwoRow
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle

    spec = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock").read_text())["learner"][
        "d1_hard_margin_affine"
    ]
    handles = ["h1", "h2"]
    oracle = HardMarginOracle(handles, spec)
    w = np.zeros(64, dtype=np.float64)
    w[0] = 1.0
    oracle.w = w
    oracle.b = 0.0
    view = CommonTwoRow(oracle, handles)
    x = w.copy()
    gap = view.pairwise_score_gap(x, "h1")
    gamma = view.geometric_margin(x, "h1")
    assert abs(gap - 2.0) < 1e-12
    assert abs(gamma - 1.0) < 1e-12
    assert gamma <= 1.0 + 1e-12
    native = 2.0 * float(np.dot(w, x) + oracle.b)
    assert abs(native - 2.0) < 1e-12
    assert native > 1.0
    reported_v1_style = 1.996
    converted = reported_v1_style / 2.0
    assert converted <= 1.0
    assert abs(converted - 0.998) < 1e-12


def test_d1_gamma_matches_discrimmap() -> None:
    from experiments.run_tm024affinemap_r2 import CommonTwoRow
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle

    spec = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock").read_text())["learner"][
        "d1_hard_margin_affine"
    ]
    handles = ["h1", "h2"]
    oracle = HardMarginOracle(handles, spec)
    w = np.zeros(64, dtype=np.float64)
    w[0] = 2.0
    oracle.w = w
    oracle.b = 0.4
    x = np.zeros(64, dtype=np.float64)
    x[0] = 1.0
    view = CommonTwoRow(oracle, handles)
    gamma = view.geometric_margin(x, "h1")
    discrim = float((np.dot(w, x) + oracle.b) / np.linalg.norm(w))
    assert abs(gamma - discrim) < 1e-12
    assert abs(view.pairwise_score_gap(x, "h1") - 2.0 * gamma) < 1e-12


def test_a1_and_a2_share_statistic() -> None:
    from experiments.run_tm024affinemap_r2 import CommonTwoRow
    from experiments.run_tm024convergencemap import PassiveAggressive
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle

    spec = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock").read_text())["learner"][
        "d1_hard_margin_homogeneous"
    ]
    handles = ["h1", "h2"]
    w = np.zeros(64, dtype=np.float64)
    w[0] = 3.0
    x = np.zeros(64, dtype=np.float64)
    x[0] = 1.0
    oracle = HardMarginOracle(handles, spec)
    oracle.w = w
    oracle.b = 0.0
    pa = PassiveAggressive(handles, gamma=0.01)
    pa.rows["h1"] = w.copy()
    pa.rows["h2"] = (-w).copy()
    g1 = CommonTwoRow(oracle, handles).geometric_margin(x, "h1")
    g2 = CommonTwoRow(pa, handles).geometric_margin(x, "h1")
    assert abs(g1 - g2) < 1e-12
    assert abs(g1 - 1.0) < 1e-12


def test_decision_requires_robustness_for_bias() -> None:
    from experiments.run_tm024affinemap_r2 import _decision, load_prereg

    p = load_prereg()

    def acq(arm: str, n: int, ok: bool) -> dict[str, object]:
        return {
            "arm": arm,
            "kind": "acquire",
            "n_cues": n,
            "passed": ok,
            "ranking_ok": ok,
            "id": f"acquire|{arm}|c{n}|A_then_B|w0",
        }

    def stab(arm: str, n: int, *, rank: bool, gamma: float, pert: bool) -> dict[str, object]:
        passed = bool(rank and pert and gamma >= 0.01)
        return {
            "arm": arm,
            "kind": "stable",
            "n_cues": n,
            "passed": passed,
            "ranking_ok": rank,
            "perturbation_ok": pert,
            "min_normalized_geometric_margin": gamma,
            "id": f"stable|{arm}|c{n}|A_then_B|w0",
        }

    cells: list[dict[str, object]] = []
    for arm, a4, a8 in (("A0", True, True), ("A1", True, True), ("A2", False, False), ("A3", True, True)):
        for wi in range(2):
            for order in ("A_then_B", "B_then_A"):
                for n, ok in ((2, True), (4, a4), (8, a8)):
                    row = acq(arm, n, ok)
                    row["id"] = f"acquire|{arm}|c{n}|{order}|w{wi}"
                    row["order"] = order
                    row["world"] = wi
                    cells.append(row)
                    s = stab(arm, n, rank=ok, gamma=0.02 if arm == "A3" and n in (4, 8) else 0.001, pert=ok and arm == "A3")
                    if arm != "A3":
                        s = stab(arm, n, rank=ok, gamma=0.001, pert=False)
                    s["id"] = f"stable|{arm}|c{n}|{order}|w{wi}"
                    s["order"] = order
                    s["world"] = wi
                    cells.append(s)
    code, _then, extra = _decision(cells, p)
    assert extra["A1_acquire_4"] is True
    assert extra["A2_acquire_4"] is False
    assert extra["A3_acquire_4"] is True
    assert extra["A3_robust_4"] is True
    assert extra["A2_robust_4"] is False
    assert code == "online_optimization_failure"
    for c in cells:
        if c["arm"] == "A1" and c["kind"] == "acquire" and int(c["n_cues"]) in (4, 8):
            c["passed"] = True
            c["ranking_ok"] = True
        if c["arm"] == "A2" and c["kind"] == "acquire":
            c["passed"] = True
            c["ranking_ok"] = True
    # A1 and A2 both acquire-pass would be apparatus; keep A2 acquire fail and A1 fail acquire
    # Rebuild: A0 fail A1 fail → ceiling reaudit; instead A1 pass A2 pass → apparatus
    cells2: list[dict[str, object]] = []
    for arm, a4, a8 in (("A0", True, True), ("A1", True, True), ("A2", True, True), ("A3", True, True)):
        for n, ok in ((2, True), (4, a4), (8, a8)):
            cells2.append(acq(arm, n, ok))
            cells2.append(stab(arm, n, rank=True, gamma=0.02 if arm == "A3" else 0.001, pert=arm == "A3"))
    code2, _then2, _extra2 = _decision(cells2, p)
    assert code2 == "apparatus_inconsistency"
    cells3: list[dict[str, object]] = []
    for arm, a4, a8 in (("A0", True, True), ("A1", True, True), ("A2", False, False), ("A3", True, True)):
        for n, ok in ((2, True), (4, a4), (8, a8)):
            cells3.append(acq(arm, n, ok))
            cells3.append(stab(arm, n, rank=True, gamma=0.02, pert=True) if arm == "A3" else stab(arm, n, rank=ok, gamma=0.001, pert=False))
    # A1 acquire pass, A2 acquire fail → still optimization, not bias
    code3, _then3, extra3 = _decision(cells3, p)
    assert extra3["A3_robust_4"] is True
    assert extra3["A2_robust_4"] is False
    assert code3 == "online_optimization_failure"
    cells4: list[dict[str, object]] = []
    for arm, a4, a8 in (("A0", False, False), ("A1", False, False), ("A2", False, False), ("A3", True, True)):
        for n, ok in ((2, True), (4, a4), (8, a8)):
            cells4.append(acq(arm, n, ok))
            if arm == "A3":
                cells4.append(stab(arm, n, rank=True, gamma=0.02, pert=True))
            else:
                cells4.append(stab(arm, n, rank=ok, gamma=0.001, pert=False))
    code4, _then4, extra4 = _decision(cells4, p)
    assert extra4["A3_robust_4"] is True
    assert extra4["A3_robust_8"] is True
    assert extra4["A2_robust_4"] is False
    assert extra4["A1_acquire_4"] is False
    assert extra4["A2_acquire_4"] is False
    assert code4 == "learned_local_bias_supported"
    cells5: list[dict[str, object]] = []
    for arm, a4, a8 in (("A0", False, False), ("A1", False, False), ("A2", False, False), ("A3", True, True)):
        for n, ok in ((2, True), (4, a4), (8, a8)):
            cells5.append(acq(arm, n, ok))
            if arm == "A3":
                row = stab(arm, n, rank=True, gamma=0.001, pert=False)
                row["pairwise_score_gap"] = 1.996
                cells5.append(row)
            else:
                cells5.append(stab(arm, n, rank=ok, gamma=0.001, pert=False))
    code5, _then5, extra5 = _decision(cells5, p)
    assert extra5["A3_robust_4"] is False
    assert code5 != "learned_local_bias_supported"
    assert code5 == "d1_ceiling_reaudit"


def test_a0_vs_a1_is_intercept_only() -> None:
    from experiments.run_tm024affinemap_r2 import CommonTwoRow
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle

    prereg = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock").read_text())
    handles = ["h1", "h2"]
    w = np.zeros(64, dtype=np.float64)
    w[0] = 2.0
    x = np.zeros(64, dtype=np.float64)
    x[0] = 1.0
    a0 = HardMarginOracle(handles, prereg["learner"]["d1_hard_margin_affine"])
    a1 = HardMarginOracle(handles, prereg["learner"]["d1_hard_margin_homogeneous"])
    a0.w = w.copy()
    a1.w = w.copy()
    a0.b = 0.4
    a1.b = 0.0
    g0 = CommonTwoRow(a0, handles).geometric_margin(x, "h1")
    g1 = CommonTwoRow(a1, handles).geometric_margin(x, "h1")
    assert abs(g1 - 1.0) < 1e-12
    assert abs(g0 - (1.0 + 0.4 / 2.0)) < 1e-12
    assert g0 != g1


def test_zero_separator_does_not_rank() -> None:
    from experiments.run_tm024affinemap_r2 import CommonTwoRow
    from experiments.run_tm024convergencemap import unique_winner
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle

    spec = json.loads((REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock").read_text())["learner"][
        "d1_hard_margin_homogeneous"
    ]
    handles = ["h1", "h2"]
    oracle = HardMarginOracle(handles, spec)
    oracle.w = np.zeros(64, dtype=np.float64)
    oracle.b = 0.0
    x = np.zeros(64, dtype=np.float64)
    x[0] = 1.0
    view = CommonTwoRow(oracle, handles)
    st = view.margin_state(x, "h1")
    assert st["zero_separator"] is True
    assert st["v_norm"] == 0.0
    assert st["normalized_geometric_margin"] == 0.0
    assert unique_winner(view.scores(x)) is None


def test_metrics_refuse_gap_as_gate() -> None:
    from experiments.run_tm024affinemap_r2 import assert_cell_metrics, expected_stability_pass

    def base(**kwargs: object) -> dict[str, object]:
        row: dict[str, object] = {
            "id": "stable|A2|c4|A_then_B|w0",
            "kind": "stable",
            "arm": "A2",
            "passed": False,
            "ranking_ok": True,
            "pairwise_score_gap": 1.996,
            "normalized_geometric_margin": 0.998,
            "min_normalized_geometric_margin": 0.998,
            "perturbation_ok": True,
            "pass_statistic": "normalized_geometric_margin",
            "geometric_ok": True,
            "domain": "TM024.AFFINEMAP.R2.DEV.",
        }
        row.update(kwargs)
        return row

    high_gap_low_g = base(
        pairwise_score_gap=1.996,
        normalized_geometric_margin=0.005,
        min_normalized_geometric_margin=0.005,
        geometric_ok=False,
        perturbation_ok=True,
        ranking_ok=True,
        passed=False,
    )
    assert expected_stability_pass(ranking_ok=True, gamma=0.005, perturbation_ok=True) is False
    assert_cell_metrics(high_gap_low_g)

    high_gap_high_g_no_pert = base(
        pairwise_score_gap=1.996,
        normalized_geometric_margin=0.998,
        min_normalized_geometric_margin=0.998,
        geometric_ok=True,
        perturbation_ok=False,
        passed=False,
    )
    assert_cell_metrics(high_gap_high_g_no_pert)

    assert_cell_metrics(base(passed=True))

    try:
        assert_cell_metrics(base(passed=True, normalized_geometric_margin=0.005, min_normalized_geometric_margin=0.005, geometric_ok=False))
    except RuntimeError as e:
        assert "pairwise_score_gap_used_as_geometric_gate" in str(e)
    else:
        raise AssertionError("inconsistent pass flag must abort")
    try:
        assert_cell_metrics(base(passed=True, perturbation_ok=False))
    except RuntimeError as e:
        assert "scoring inconsistency" in str(e)
    else:
        raise AssertionError("stable pass must require perturbation")
    try:
        assert_cell_metrics(base(passed=False))
    except RuntimeError as e:
        assert "scoring inconsistency" in str(e)
    else:
        raise AssertionError("omitting a true stability pass is inconsistency")

    premises = {"c": 0.0, "v_norm": 2.0, "features_unit": True}
    assert_cell_metrics(base(arm="A1", normalized_geometric_margin=1.5, min_normalized_geometric_margin=1.5, geometric_ok=True, passed=True, pairwise_score_gap=3.0))
    try:
        assert_cell_metrics(base(arm="A1", normalized_geometric_margin=1.5, min_normalized_geometric_margin=1.5, geometric_ok=True, passed=True, pairwise_score_gap=3.0, **premises))
    except RuntimeError as e:
        assert "homogeneous geometric margin exceeded 1" in str(e)
    else:
        raise AssertionError("unit-row homogeneous gamma must not exceed 1")
    try:
        assert_cell_metrics(base(arm="A2", ranking_ok=True, passed=False, perturbation_ok=False, geometric_ok=False, normalized_geometric_margin=0.0, min_normalized_geometric_margin=0.0, pairwise_score_gap=0.0, c=0.0, v_norm=0.0, features_unit=True))
    except RuntimeError as e:
        assert "zero effective separator cannot rank" in str(e)
    else:
        raise AssertionError("zero separator cannot rank")
    assert_cell_metrics(base(arm="A2", ranking_ok=False, passed=False, perturbation_ok=False, geometric_ok=False, normalized_geometric_margin=0.0, min_normalized_geometric_margin=0.0, pairwise_score_gap=0.0, c=0.0, v_norm=0.0, features_unit=True))


def test_cell_ids() -> None:
    from experiments.run_tm024affinemap_r2 import expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert all(not i.startswith("eco|") and not i.startswith("spec|") for i in ids)
    assert "acquire|A0|c4|A_then_B|w0" in ids
    assert "acquire|A1|c8|B_then_A|w1" in ids
    assert "twin|A3|c2|A_then_B|w1" in ids


def test_smoke() -> None:
    from experiments.run_tm024affinemap_r2 import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert out["a0_fields"] == list(RECORDED_FIELDS)
    assert out["a0_stable_passed_uses_geometric"] is True
    assert out["a1_gamma_le_1"] is True
    assert out["a1_gap"] >= out["a1_gamma"] - 1e-12
    assert out["pass_statistic"] == "normalized_geometric_margin"
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024affinemap_r2 import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_rerun, refuse_score

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
            refuse_rerun()
        except RuntimeError as e:
            assert "again" in str(e)
        else:
            raise AssertionError("same frozen DEV execution must be refused")
        return
    refuse_dev_lock()


def test_dev_opened() -> None:
    assert not (REPO_ROOT / "docs" / "lineage_affinemap.dev.lock").exists()
    p = REPO_ROOT / "docs" / "lineage_affinemap.r2.dev.lock"
    d = REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.lock"
    assert sha(p) == DEV_LOCK_SHA
    assert sha(d) == DECISION_SHA
    dev = json.loads(p.read_text(encoding="utf-8"))
    dec = json.loads(d.read_text(encoding="utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len({c["id"] for c in dev["cells"]}) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.AFFINEMAP.R2.DEV."
    assert sorted({c["domain"] for c in dev["cells"]}) == [
        "TM024.AFFINEMAP.R2.DEV.",
        "TM024.AFFINEMAP.R2.TWIN.",
    ]
    blob = json.dumps(dev)
    assert "TM024.AFFINEMAP.R2.SCORE." not in blob
    assert "TM024.AFFINEMAP.SCORE." not in blob
    assert dev["decision_code"] == "online_optimization_failure"
    assert dev["phase_flags"]["A1_acquire_4"] is True
    assert dev["phase_flags"]["A1_acquire_8"] is True
    assert dev["phase_flags"]["A2_acquire_4"] is False
    assert dev["phase_flags"]["A2_acquire_8"] is False
    assert dev["phase_flags"]["A3_acquire_4"] is False
    assert dev["phase_flags"]["A3_robust_4"] is False
    assert dev["a3_implementation_authorized"] is False
    assert dev["neural_edit"] is False
    assert dev["pass_statistic"] == "normalized_geometric_margin"
    assert dev["manifest_sha"] == DEV_MANIFEST_SHA
    assert dev["git_head"] == FREEZE_GIT
    assert sha(REPO_ROOT / "experiments" / "run_tm024affinemap_r2.py") == RUNNER_SHA
    assert dec["decision"]["code"] == "online_optimization_failure"
    assert dec["a3_implementation_authorized"] is False
    assert dec["implementation_authorized"] is False
    assert dec["dev_lock_sha"] == DEV_LOCK_SHA
    assert dec["n"] == 64
    assert dec["earned_next"] is False


def test_addendum() -> None:
    p = REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.addendum.lock"
    assert sha(p) == ADDENDUM_SHA
    add = json.loads(p.read_text(encoding="utf-8"))
    assert add["historical_code"] == "online_optimization_failure"
    assert add["honest_first_match_on_these_cells"] == "online_optimization_failure"
    assert add["rewrite_historical_decision"] is False
    assert add["rewrite_historical_dev"] is False
    assert add["rewrite_historical_runner"] is False
    assert add["a1_acquire_4"] is True
    assert add["a1_acquire_8"] is True
    assert add["a2_acquire_4"] is False
    assert add["a3_uniquely_helpful"] is False
    assert add["a3_joins_candidate"] is False
    assert add["compact_pa_path"] is False
    assert add["oracle_reaudit"] is False
    assert add["two_timescale_candidate_authorized"] is True
    assert add["no_further_solver_map"] is True
    assert add["next"] == "two_timescale_episodic_consolidation_candidate"
    assert add["historical_decision_sha"] == DECISION_SHA
    assert add["historical_dev_lock_sha"] == DEV_LOCK_SHA
    assert add["historical_runner_lock_sha"] == RUNNER_LOCK_SHA
    assert add["historical_runner_py_sha"] == RUNNER_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm024affinemap_r2.py") == RUNNER_SHA
    assert add["neural_edit_this_addendum"] is False


if __name__ == "__main__":
    test_v1_freeze_preserved()
    test_phase_a_files()
    test_gap_is_not_gamma()
    test_d1_gamma_matches_discrimmap()
    test_a1_and_a2_share_statistic()
    test_decision_requires_robustness_for_bias()
    test_a0_vs_a1_is_intercept_only()
    test_zero_separator_does_not_rank()
    test_metrics_refuse_gap_as_gate()
    test_cell_ids()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_dev_opened()
    test_addendum()
    print("ok")
