"""TM.0.24.LIFECYCLEMARGINMAP provenance. R2 freeze preserved. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
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
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "d0e5eee16752a7ae89bdfc16f3e0294ce14cfd4726aa8e71f7a8eba1c7c848dd"
RUNNER_SHA = "5f7dc1a79e49c42edc45ccd7d12d4c4a8d2989a067071becc433b46c6234ddce"
MANIFEST_SHA = "0048aa21d8512a50923fa0a878dccf3b28e43878d10b9c84b338dec26b21c5f4"
DEV_LOCK_SHA = None
EXPECTED_N_CELLS = 112
ARMS = ("M0", "M1", "M2", "M3")
PHASES_OK = (
    "margin_conditioned_replay_supported",
    "replacement_and_margin_conditioned_replay_jointly_causal",
    "consolidation_margin_loss",
    "compact_margin_replay_insufficient",
    "max_margin_ceiling_only",
    "lifecycle_margin_insufficient",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_freeze_preserved() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.lock") == R2_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.dev.lock") == R2_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.addendum.lock") == R2_ADD
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.runner.lock") == R2_RUNNER_LOCK
    assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap_r2.py") == R2_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.prereg.lock") == R2_PREREG
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock") == V1_RUNNER_LOCK_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py") == V1_RUNNER_PY_SHA
    assert not (REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock").exists()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_lifecyclemarginmap.contract.md",
        "docs/lineage_lifecyclemarginmap.prereg.lock",
        "docs/lineage_lifecyclemarginmap.isolation.lock",
        "docs/lineage_memorylifecyclemap.r2.decision.addendum.lock",
        "docs/lineage_memorylifecyclemap.r2.decision.lock",
        "docs/lineage_convergencemap.decision.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024lifecyclemarginmap.py",
        "experiments/run_tm024memorylifecyclemap_r2.py",
        "experiments/run_tm024convergencemap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["act_score_mode"] == "query"
    assert prereg["no_new_eta_grid"] is True
    assert prereg["no_new_match_radius"] is True
    assert prereg["no_new_replay_budget"] is True
    assert prereg["no_new_margin_threshold"] is True
    assert prereg["domains"]["DEV"] == "TM024.LIFECYCLEMARGINMAP.DEV."
    assert prereg["arms"]["M0"]["replay_learner"] == "error_only"
    assert prereg["arms"]["M1"]["replay_learner"] == "c3_passive_aggressive"
    assert prereg["arms"]["M2"]["policy"] == "retain_stale"
    assert prereg["arms"]["M3"]["ceiling_only"] is True
    assert prereg["arms"]["M1"]["replay_epochs"] == 16
    assert prereg["match"]["radius"] == 0.05
    assert prereg["honest_r2_first_match"] == "memory_lifecycle_insufficient"
    assert prereg["historical_r2_prereg_sha"] == R2_PREREG
    assert "M2 reversal still passes" in prereg["decision_ladder"][0]["when"]
    assert "M3 is not four-phase" in prereg["decision_ladder"][3]["when"]
    assert prereg["memory_lifecycle_insufficient_means"] == "frozen_lifecycle_did_not_pass_all_four_phases"
    assert prereg["does_not_mean_replay_or_replacement_provided_no_benefit"] is True
    assert prereg["ecological_matcher_must_not_redirect_this_diagnosis"] is True
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    assert prereg["r2_frozen_eight_cue_l2"]["n_actual_error_only_updates"] == 14
    assert prereg["r2_frozen_eight_cue_l2"]["first_all_correct_call"] == 20
    assert prereg["r2_frozen_eight_cue_l2"]["rest_reduced_margin"] is False
    iso = json.loads((REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.isolation.lock").read_text(encoding="utf-8"))
    assert iso["rescore_historical_worlds"] is False
    assert iso["honest_r2_first_match"] == "memory_lifecycle_insufficient"
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock") == CVG_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock") == CVG_ADD
    assert sha(REPO_ROOT / "experiments" / "run_tm024convergencemap.py") == CVG_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    runner_p = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.runner.lock"
    man_p = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.manifest.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024lifecyclemarginmap.py") == RUNNER_SHA
            assert runner["shas"]["runner"] == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["domain"] == "TM024.LIFECYCLEMARGINMAP.DEV."
        assert runner["m3_ceiling_only"] is True
        assert runner["no_new_eta_grid"] is True
        assert runner["historical_r2_decision_sha"] == R2_DEC
        assert runner["shas"]["r2_prereg"] == R2_PREREG
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        if MANIFEST_SHA is not None:
            assert sha(man_p) == MANIFEST_SHA
        assert man["expected_n_cells"] == EXPECTED_N_CELLS
        assert len(man["expected_cell_ids"]) == EXPECTED_N_CELLS
        assert len(set(man["expected_cell_ids"])) == EXPECTED_N_CELLS


def _cell(
    arm: str,
    kind: str,
    n_cues: int,
    *,
    passed: bool,
    world: int = 0,
    order: str = "A_then_B",
    min_margin: float = 0.0,
) -> dict[str, object]:
    return {
        "arm": arm,
        "kind": kind,
        "n_cues": n_cues,
        "order": order,
        "world": world,
        "passed": passed,
        "ranking_ok": passed or kind == "acquire",
        "min_probe_margin": float(min_margin),
        "m3_ceiling_only": arm == "M3",
        "id": f"{kind}|{arm}|c{n_cues}|{order}|w{world}",
        "margin_trajectory": {
            "pre_rest_min_probe_margin": float(min_margin) if kind == "acquire" else None,
            "post_rest_min_probe_margin": float(min_margin) if kind == "stable" else None,
        },
    }


def _arm_cells(
    arm: str,
    *,
    acquire_all: bool = False,
    twin: bool = False,
    stable: bool = False,
    plasticity: bool = False,
    specificity: bool = False,
    acquire8_margin: float = 0.0,
    stable8_margin: float = 0.0,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for n in (2, 4, 8):
        acq_m = acquire8_margin if n == 8 else 0.2
        st_m = stable8_margin if n == 8 else 0.2
        for wi in range(2):
            for order in ("A_then_B", "B_then_A"):
                cells.append(
                    _cell(arm, "acquire", n, passed=acquire_all, world=wi, order=order, min_margin=acq_m)
                )
                cells.append(
                    _cell(arm, "stable", n, passed=stable, world=wi, order=order, min_margin=st_m)
                )
    for order in ("A_then_B", "B_then_A"):
        cells.append(_cell(arm, "twin", 2, passed=twin, world=1, order=order, min_margin=0.2))
    cells.append(_cell(arm, "eco", 2, passed=plasticity, min_margin=0.2 if plasticity else 0.0))
    cells.append(_cell(arm, "spec", 4, passed=specificity, min_margin=0.2 if specificity else 0.0))
    return cells


def _full(**arm_kwargs: dict[str, object]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for arm in ARMS:
        cells.extend(_arm_cells(arm, **(arm_kwargs.get(arm) or {})))
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024lifecyclemarginmap import EXPECTED_N_CELLS as N
    from experiments.run_tm024lifecyclemarginmap import _decision, load_prereg

    p = load_prereg()

    def code(**arm_kwargs: dict[str, object]) -> str:
        cells = _full(**arm_kwargs)
        assert len(cells) == N
        c, _then, _extra = _decision(cells, p)
        return c

    four = {
        "acquire_all": True,
        "twin": True,
        "stable": True,
        "plasticity": True,
        "specificity": True,
        "acquire8_margin": 0.02,
        "stable8_margin": 0.02,
    }
    assert code(M1=four, M2={"plasticity": True}) == "margin_conditioned_replay_supported"
    assert code(M1=four) == "replacement_and_margin_conditioned_replay_jointly_causal"
    assert (
        code(M1={**four, "specificity": False}, M2={"plasticity": False})
        == "replacement_and_margin_conditioned_replay_jointly_causal"
    )
    assert (
        code(
            M1={
                "acquire_all": True,
                "acquire8_margin": 0.02,
                "stable8_margin": 0.005,
            }
        )
        == "consolidation_margin_loss"
    )
    assert code() == "compact_margin_replay_insufficient"
    assert code(M3=four) == "max_margin_ceiling_only"
    assert code(M3=four, M1={"acquire8_margin": 0.02, "stable8_margin": 0.02}) == "max_margin_ceiling_only"
    assert (
        code(
            M0={"acquire_all": True, "twin": True, "stable": True, "acquire8_margin": 0.02, "stable8_margin": 0.02},
            M1={"acquire_all": True, "twin": True, "acquire8_margin": 0.02, "stable8_margin": 0.02},
        )
        == "lifecycle_margin_insufficient"
    )


def test_error_only_skips_correct_ranking_pa_does_not() -> None:
    from experiments.run_tm024convergencemap import ErrorOnlyBank, PassiveAggressive
    from experiments.run_tm024eligmap import unit_or_zero

    handles = ["h1", "h2"]
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    e = np.zeros(64, dtype=np.float64)
    e[0] = 1.0
    e = e - float(np.dot(e, x)) * x
    e = unit_or_zero(e)
    w_ch = 0.5 * e + 0.001 * x
    w_ot = np.zeros(64, dtype=np.float64)
    err = ErrorOnlyBank(handles, eta=0.15, c_max=1.0)
    pa = PassiveAggressive(handles, gamma=0.01)
    err.inner.rows["h1"] = w_ch.copy()
    err.inner.rows["h2"] = w_ot.copy()
    pa.rows["h1"] = w_ch.copy()
    pa.rows["h2"] = w_ot.copy()
    assert err.scores(x)["h1"] > err.scores(x)["h2"]
    n0_err, n0_pa = err.n_updates, pa.n_updates
    err.update(x, "h1", 1.0)
    pa.update(x, "h1", 1.0)
    assert err.n_updates == n0_err
    assert pa.n_updates == n0_pa + 1


def test_store_policies_and_oracle() -> None:
    from experiments.run_tm024eligmap import unit_or_zero
    from experiments.run_tm024lifecyclemarginmap import HardMarginOracle, reversal_live_learner
    from experiments.run_tm024memorylifecyclemap_r2 import EpisodeStore, episode_core
    from experiments.run_tm024writegeom import capacity_world

    dummy = object()
    assert reversal_live_learner("M0", dummy) is dummy
    assert reversal_live_learner("M1", dummy) is dummy
    assert reversal_live_learner("M2", dummy) is dummy
    assert reversal_live_learner("M3", dummy) is None

    world = capacity_world(0, "TM024.LIFECYCLEMARGINMAP.TEST.", n_cues=2, n_handles=2)
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(np.arange(64, dtype=np.float64)[::-1] + 7.0)
    h1, h2 = world["handles"][0], world["handles"][1]
    stale = EpisodeStore(policy="retain_stale", match_l2=0.05)
    stale.write(x, h1, 1.0, world=world, cue="a")
    core = episode_core(stale.slots[0])
    q = stale.write(x, h1, -1.0, world=world, cue="a")
    assert q["action"] == "refuse"
    assert episode_core(stale.slots[0]) == core
    repl = EpisodeStore(policy="replace", match_l2=0.05)
    repl.write(x, h1, 1.0, world=world, cue="a")
    wr = repl.write(x, h2, 1.0, world=world, cue="a")
    assert wr["action"] == "replace"

    spec = json.loads((REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.prereg.lock").read_text(encoding="utf-8"))[
        "learner"
    ]["d1_hard_margin"]
    oracle = HardMarginOracle([h1, h2], spec)
    oracle.fit(
        [
            {"p1": x, "handle": h1, "adv": 1.0},
            {"p1": y, "handle": h2, "adv": 1.0},
        ]
    )
    assert oracle.n_updates == 1
    win_x = max(oracle.scores(x), key=lambda h: oracle.scores(x)[h])
    win_y = max(oracle.scores(y), key=lambda h: oracle.scores(y)[h])
    assert win_x == h1
    assert win_y == h2


def test_cell_ids_and_replay_trace() -> None:
    from experiments.run_tm024convergencemap import ErrorOnlyBank
    from experiments.run_tm024eligmap import unit_or_zero
    from experiments.run_tm024lifecyclemarginmap import expected_cell_ids, replay_traced

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    handles = ["h1", "h2"]
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(np.arange(64, dtype=np.float64)[::-1] + 3.0)
    learner = ErrorOnlyBank(handles, eta=0.15, c_max=1.0)
    rows = [
        {"p1": x, "handle": "h1", "adv": 1.0, "age": 1, "version": 1, "valid": True},
        {"p1": y, "handle": "h2", "adv": 1.0, "age": 2, "version": 1, "valid": True},
    ]
    traj = replay_traced(learner, handles, rows, 16, 0.01)
    assert traj["n_replay_calls"] == 32
    assert traj["first_all_correct_call"] is not None
    assert traj["n_replay_after_first_all_correct"] == 32 - int(traj["first_all_correct_call"])
    assert traj["n_actual_updates"] < traj["n_replay_calls"]


def test_rest_null_when_rest_not_run() -> None:
    from experiments.run_tm024lifecyclemarginmap import attach_probe_traj, empty_traj

    traj = empty_traj()
    attach_probe_traj(traj, {"min_probe_margin": 0.02}, None)
    assert traj["pre_rest_min_probe_margin"] == 0.02
    assert traj["post_rest_min_probe_margin"] is None
    assert traj["rest_delta"] is None
    assert traj["rest_reduced_margin"] is None


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.contract.md").read_text(encoding="utf-8")
    assert "memory_lifecycle_insufficient" in text
    assert "did not pass all four phases" in text
    assert "provided no benefit" in text
    assert "margin_conditioned_replay_supported" in text
    assert "M2 reversal still passes" in text
    assert "post-REST fields stay null" in text
    assert "No new η grid" in text or "No new η" in text


def test_smoke() -> None:
    from experiments.run_tm024lifecyclemarginmap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert "first_all_correct_call" in out["m0_traj_keys"]
    assert "pre_rest_min_probe_margin" in out["m0_traj_keys"]
    assert "post_rest_min_probe_margin" in out["m0_traj_keys"]
    assert out["eco_post_rest_min_probe_margin"] is None
    assert out["eco_rest_delta"] is None
    assert out["eco_rest_reduced_margin"] is None
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024lifecyclemarginmap import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

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


def test_dev_absent() -> None:
    assert not (REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock").exists()
    if DEV_LOCK_SHA is not None:
        raise AssertionError("DEV was opened")


def main() -> None:
    test_historical_freeze_preserved()
    test_phase_a_files()
    test_decision_ladder()
    test_error_only_skips_correct_ranking_pa_does_not()
    test_store_policies_and_oracle()
    test_cell_ids_and_replay_trace()
    test_rest_null_when_rest_not_run()
    test_contract_stance()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_dev_absent()
    print("test_tm024lifecyclemarginmap: ok")


if __name__ == "__main__":
    main()
