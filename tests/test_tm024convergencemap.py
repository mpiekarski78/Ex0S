"""TM.0.24.CONVERGENCEMAP provenance. Runner-only. W1 closed. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
TB_DEC = "284968d782c73727a2de8b34e24461bf75a236f7cc6d07770e9d39529fc038a1"
TB_DEV = "b734f0bd45aa52c65ce48fe57fd38b8693ce796d58c38547dc2391e87e49b384"
TB_ADD = "f1bdaa606dc21779bae73b63be139fa8282a788c9a4261c4c87fe9cd961f7eb0"
TB_RUNNER_PY = "db6c4e73cac57dd79cb86fcfa371fee1e2fc5753bf1d03402faea690ff4de551"
WG_RUNNER_PY = "b210cc621ccd93e016483e3d9d8dc8adbc284eb8fabc01a4b15bbb4ecb1f4d31"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "bd876e2c6d4c9fdfa7b77e12c5ad15f0bd32f33ef96c02a3a86d62563afcf7b7"
RUNNER_SHA = "232cffa23619de1fcdbde7b8c82fc3de8e1c2fbe84a014a40bc27f3723cbbcf6"
DEV_LOCK_SHA = "7aa13d1bbd172cadb565d5d6cafd27d91f47b120491069a00a34111a6d505ca6"
DEV_MANIFEST_SHA = "7ad1feb895e6c28897be7dbafe082f5e8a50acbeb0f64695f18f2aea88d9fb7e"
DECISION_SHA = "6de34e295b54ef51c2684b0ce1cf7295064300db75e9e7ee258c7bb95665072d"
ADDENDUM_SHA = "04a5e91cdc23839c9e3c954dc8c921f902c092de14acd906ada7caa678a6b083"
EXPECTED_N_CELLS = 220
ARMS = ("C0", "C1", "C2", "C3", "C4")
PHASES_OK = (
    "compact_error_correcting_write_sufficient",
    "repeated_exposure_suffices",
    "state_reinstatement_unstable_across_learning",
    "covariance_aware_memory_required",
    "oracle_separability_not_operationally_reachable",
)
C1_LIVE = (1, 2, 4, 8, 16)
C3_LIVE = (2, 4, 8, 16)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_convergencemap_contract.md",
        "docs/lineage_convergencemap.prereg.lock",
        "docs/lineage_convergencemap.isolation.lock",
        "docs/lineage_tracebridge.decision.lock",
        "docs/lineage_tracebridge.decision.addendum.lock",
        "docs/lineage_convergencemap.decision.addendum.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024convergencemap.py",
        "experiments/run_tm024tracebridge.py",
        "experiments/run_tm024writegeom.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_convergencemap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["act_score_mode"] == "query"
    assert prereg["w1_resurrected"] is False
    assert prereg["arms"]["C0"]["cycles"] == [1]
    assert prereg["arms"]["C1"]["cycles"] == [1, 2, 4, 8, 16]
    assert prereg["arms"]["C1"]["error_only"] is True
    assert prereg["arms"]["C2"]["learning_rate_grid"] is False
    assert prereg["arms"]["C2"]["geometric_margin_target"] == 0.01
    assert prereg["arms"]["C3"]["cycles"] == [2, 4, 8, 16]
    assert prereg["arms"]["C4"]["lambda"] == 0.01
    assert prereg["arms"]["C4"]["exposure_mode"] == "replay"
    assert prereg["arms"]["C4"]["ceiling_only"] is True
    assert prereg["arms"]["C2"]["status_is_post_update_normalized_geometric"] is True
    assert prereg["arms"]["C2"]["functional_margin_is_not_pass_criterion"] is True
    assert prereg["repetition"]["k_means"] == "complete_passes_through_cue_set"
    assert prereg["repetition"]["exposure_mode_field"] == "exposure_mode"
    assert prereg["retention"]["probes_never_update_weights_or_organism"] is True
    assert prereg["c0_vs_c1_k1"] == "always_update_versus_error_only"
    assert prereg["c4_ceiling_only"] is True
    assert prereg["assert_unique_ids_and_manifest_before_dev"] is True
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    assert prereg["expected_kind_counts"] == {"rank": 168, "twin": 28, "eco": 12, "rest": 12}
    assert prereg["expected_exposure_mode_counts"] == {"live": 176, "replay": 44}
    assert prereg["controls"]["hold"] is False
    assert prereg["domains"]["DEV"] == "TM024.CONVERGENCEMAP.DEV."
    assert prereg["score_reserved_unopened"] is True
    assert prereg["declared_budget_if_later_authorized"]["opened"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_tracebridge.decision.lock") == TB_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_tracebridge.dev.lock") == TB_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_tracebridge.decision.addendum.lock") == TB_ADD
    assert sha(REPO_ROOT / "experiments" / "run_tm024tracebridge.py") == TB_RUNNER_PY
    assert sha(REPO_ROOT / "experiments" / "run_tm024writegeom.py") == WG_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    add = json.loads((REPO_ROOT / "docs" / "lineage_tracebridge.decision.addendum.lock").read_text(encoding="utf-8"))
    assert add["historical_code"] == "p1_not_usable_by_online_class"
    assert add["interpret_as"] == "p1_not_usable_by_frozen_one_pass_online_rules"
    assert add["rewrite_historical_decision"] is False
    assert add["next"] == "TM.0.24.CONVERGENCEMAP"
    assert add["does_not_reject_online_learning_generally"] is True
    cvg_add_p = REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock"
    assert sha(cvg_add_p) == ADDENDUM_SHA
    cvg_add = json.loads(cvg_add_p.read_text(encoding="utf-8"))
    assert cvg_add["historical_code"] == "oracle_separability_not_operationally_reachable"
    assert cvg_add["interpret_as"] == (
        "oracle_separability_not_operationally_reachable_under_frozen_monotonic_retention_and_reversal_contract"
    )
    assert cvg_add["rewrite_historical_decision"] is False
    assert cvg_add["next"] == "TM.0.24.MEMORYLIFECYCLEMAP"
    assert cvg_add["does_not_mean_p1_unlearnable"] is True
    assert cvg_add["c1_exact_replay_8cue"]["ranking_ok"] == 4
    assert cvg_add["c1_exact_replay_8cue"]["passed"] == 0
    assert cvg_add["historical_decision_sha"] == DECISION_SHA
    runner_p = REPO_ROOT / "docs" / "lineage_convergencemap.runner.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024convergencemap.py") == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["act_score_mode"] == "query"
        assert runner["w1_resurrected"] is False
        assert runner["pa_learning_rate_grid"] is False
        assert runner["exposure_mode_field"] == "exposure_mode"
        assert runner["c4_ceiling_only"] is True
        assert runner["pa_status_is_normalized_geometric"] is True


def _cell(
    arm: str,
    kind: str,
    n_cues: int,
    *,
    cycles: int,
    exposure_mode: str,
    passed: bool,
    world: int = 0,
    order: str = "A_then_B",
) -> dict[str, object]:
    return {
        "arm": arm,
        "kind": kind,
        "n_cues": n_cues,
        "cycles": cycles,
        "order": order,
        "world": world,
        "exposure_mode": exposure_mode,
        "passed": passed,
        "retention_ok": passed,
        "c4_ceiling_only": arm == "C4",
        "p1_source": "live_regenerated" if exposure_mode == "live" else "frozen_first_pass",
        "id": f"{kind}|{arm}|c{n_cues}|{order}|w{world}|k{cycles}|{exposure_mode}",
    }


def _spec_cells(arm: str, cycles: int, exposure_mode: str, *, robust: bool, include_eco_rest: bool) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for n in (2, 4, 8):
        for wi in range(2):
            for order in ("A_then_B", "B_then_A"):
                cells.append(
                    _cell(arm, "rank", n, cycles=cycles, exposure_mode=exposure_mode, passed=robust, world=wi, order=order)
                )
    for order in ("A_then_B", "B_then_A"):
        cells.append(_cell(arm, "twin", 2, cycles=cycles, exposure_mode=exposure_mode, passed=robust, world=1, order=order))
    if include_eco_rest:
        cells.append(_cell(arm, "eco", 2, cycles=cycles, exposure_mode=exposure_mode, passed=robust))
        cells.append(_cell(arm, "rest", 2, cycles=cycles, exposure_mode=exposure_mode, passed=robust))
    return cells


def _full_grid(robust_keys: set[tuple[str, int, str]]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    live = [("C0", 1)] + [("C1", k) for k in C1_LIVE] + [("C2", 1)] + [("C3", k) for k in C3_LIVE]
    for arm, k in live:
        cells.extend(
            _spec_cells(arm, k, "live", robust=(arm, k, "live") in robust_keys, include_eco_rest=True)
        )
    for arm, k in (("C4", 16), ("C1", 16), ("C3", 16)):
        cells.extend(
            _spec_cells(
                arm,
                k,
                "replay",
                robust=(arm, k, "replay") in robust_keys,
                include_eco_rest=arm == "C4",
            )
        )
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024convergencemap import _decision, load_prereg

    p = load_prereg()

    def code(keys: set[tuple[str, int, str]]) -> str:
        cells = _full_grid(keys)
        assert len(cells) == EXPECTED_N_CELLS
        c, _then, _extra = _decision(cells, p)
        return c

    assert code({("C2", 1, "live")}) == "compact_error_correcting_write_sufficient"
    assert code({("C1", 1, "live")}) == "compact_error_correcting_write_sufficient"
    assert code({("C1", 8, "live")}) == "repeated_exposure_suffices"
    assert code({("C3", 2, "live")}) == "repeated_exposure_suffices"
    assert code({("C1", 16, "replay")}) == "state_reinstatement_unstable_across_learning"
    assert code({("C3", 16, "replay"), ("C4", 16, "replay")}) == "state_reinstatement_unstable_across_learning"
    assert code({("C4", 16, "replay")}) == "covariance_aware_memory_required"
    assert code(set()) == "oracle_separability_not_operationally_reachable"
    assert code({("C2", 1, "live"), ("C4", 16, "replay")}) == "compact_error_correcting_write_sufficient"


def test_min_tau() -> None:
    import numpy as np

    from experiments.run_tm024convergencemap import geometric_margin, min_tau_for_margin, pa_step
    from experiments.run_tm024eligmap import unit_or_zero

    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    z = np.zeros(64, dtype=np.float64)
    tau = min_tau_for_margin(z, z, x, 0.01)
    assert tau == 0.01
    w_ch = z + tau * x
    w_ot = z - tau * x
    assert geometric_margin(w_ch, w_ot, x) >= 0.01 - 1e-9
    assert min_tau_for_margin(w_ch, w_ot, x, 0.01) == 0.0
    tau2 = min_tau_for_margin(-x, x, x, 0.01)
    assert tau2 > 0.0
    w2_ch = -x + tau2 * x
    w2_ot = x - tau2 * x
    assert geometric_margin(w2_ch, w2_ot, x) >= 0.01 - 1e-6


def test_pa_normalized_margin_not_functional() -> None:
    import numpy as np

    from experiments.run_tm024convergencemap import geometric_margin, pa_step
    from experiments.run_tm024eligmap import unit_or_zero

    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    e = np.zeros(64, dtype=np.float64)
    e[0] = 1.0
    e = e - float(np.dot(e, x)) * x
    e = unit_or_zero(e)
    w = 200.0 * e + 0.5 * x
    tau_f = max(0.0, 1.0 - float(np.dot(w, x)))
    w_f = w + tau_f * x
    geo_f = float(np.dot(w_f, x) / np.linalg.norm(w_f))
    assert geo_f < 0.01
    step = pa_step(w, np.zeros(64), x, 0.01)
    assert step["status"] == "met"
    assert step["geometric"] >= 0.01 - 1e-9
    assert geometric_margin(step["w_ch"], step["w_ot"], x) >= 0.01 - 1e-9
    blocked = pa_step(np.zeros(64), np.zeros(64), x, 0.01, row_c_max=0.0)
    assert blocked["status"] == "infeasible"
    assert blocked["geometric"] < 0.01


def test_c0_c1_differ_only_error_gate() -> None:
    import numpy as np

    from experiments.run_tm024convergencemap import AlwaysBank, ErrorOnlyBank
    from experiments.run_tm024eligmap import unit_or_zero

    handles = ["h1", "h2"]
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    always = AlwaysBank(handles, eta=0.15, c_max=1.0)
    err = ErrorOnlyBank(handles, eta=0.15, c_max=1.0)
    always.update(x, "h1", 1.0)
    err.update(x, "h1", 1.0)
    assert always.n_updates == 1
    assert err.n_updates == 1
    always.update(x, "h1", 1.0)
    err.update(x, "h1", 1.0)
    assert always.n_updates == 2
    assert err.n_updates == 1


def test_exposure_mode_grid() -> None:
    from experiments.run_tm024convergencemap import (
        EXPECTED_N_CELLS,
        assert_cell_coverage,
        cell_id,
        expected_cell_ids,
    )

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    live = cell_id("rank", "C1", 8, "A_then_B", 0, 16, "live")
    replay = cell_id("rank", "C1", 8, "A_then_B", 0, 16, "replay")
    assert live != replay
    assert live.endswith("|live")
    assert replay.endswith("|replay")
    cells = _full_grid(set())
    assert assert_cell_coverage(cells)


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_convergencemap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "C0" in text and "C4" in text
    assert "not a neural amendment" in text.lower()
    assert "0.01" in text
    assert "exposure_mode" in text
    assert "W1" in text


def test_smoke() -> None:
    from experiments.run_tm024convergencemap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["w1_resurrected"] is False
    assert out["act_score_mode"] == "query"
    assert out["product"] == "0.0.004"
    assert out["c0_n_live"] == 2
    assert out["c2_n_live"] == 2
    assert out["c0_exposure_mode"] == "live"
    assert out["c0_p1_source"] == "live_regenerated"
    assert out["c2_exposure_mode"] == "live"
    assert out["tau_zero_init"] == 0.01
    assert out["tau_already_met"] == 0.0
    assert out["pa_zero_status"] == "met"
    assert out["pa_zero_geometric"] >= 0.01 - 1e-9
    assert out["c1_k16_ids_distinct"] is True
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert out["c4_ceiling_only"] is True


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024convergencemap import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

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
            assert "again" in str(e) or "mismatch" in str(e)
        else:
            raise AssertionError("same frozen DEV execution must be refused")
        return
    refuse_dev_lock()


def convergencemap_manifest(dev: dict) -> str:
    rows = []
    for c in dev["cells"]:
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "cycles": c["cycles"],
                "exposure_mode": c.get("exposure_mode") or c.get("exposure"),
                "passed": c["passed"],
                "retention_ok": c.get("retention_ok"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock"
    if not p.exists():
        return
    if DECISION_SHA is not None:
        assert sha(p) == DECISION_SHA
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["neural_edit"] is False
    assert d["w1_resurrected"] is False
    assert d["candidate_v31"] is False
    assert d["decision"]["code"] in PHASES_OK
    assert "TM024.CONVERGENCEMAP.SCORE." not in json.dumps(d)


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_convergencemap.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    raw = p.read_bytes()
    if DEV_LOCK_SHA is not None:
        assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len(set(c["id"] for c in dev["cells"])) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.CONVERGENCEMAP.DEV."
    assert "TM024.CONVERGENCEMAP.SCORE." not in json.dumps(dev)
    assert Counter(c["kind"] for c in dev["cells"]) == {
        "rank": 168,
        "twin": 28,
        "eco": 12,
        "rest": 12,
    }
    assert all("retention_ok" in c for c in dev["cells"])
    assert all(c.get("exposure_mode") in ("live", "replay") for c in dev["cells"])
    assert "manifest_sha" in dev
    assert dev.get("c4_ceiling_only") is True
    assert Counter(c.get("exposure_mode") for c in dev["cells"]) == {"live": 176, "replay": 44}
    if DEV_MANIFEST_SHA is not None:
        assert convergencemap_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_phase_a_files()
    test_decision_ladder()
    test_min_tau()
    test_pa_normalized_margin_not_functional()
    test_c0_c1_differ_only_error_gate()
    test_exposure_mode_grid()
    test_contract_stance()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024convergencemap: ok")


if __name__ == "__main__":
    main()
