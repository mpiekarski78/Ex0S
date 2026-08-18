"""TM032 REST-split diagnostic tests. No DEV execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_restsplit.prereg.lock"
ISO = REPO / "docs" / "lineage_restsplit.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_restsplit_contract.md"
RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM031_RUNNER = REPO / "experiments" / "run_tm031halfspace.py"
TM031_FAILCLASS = REPO / "docs" / "lineage_halfspace.failclass.lock"
TM031_DEC = REPO / "docs" / "lineage_halfspace.decision.lock"
TM031_DEV = REPO / "docs" / "lineage_halfspace.dev.lock"
TM031_ADDENDUM = REPO / "docs" / "lineage_halfspace.decision.addendum.lock"
MANIFEST = "69f9c8e11855c54454bb3e99128761099341ad3d17bcbc9def3a82b75a286bd7"
HISTORICAL_TM031_FAILCLASS_SHA = "7f37c8010685d70b6184cd4242689826081694fd0416747c93002650588a061e"
HISTORICAL_TM031_DEC_SHA = "00d5e289068a68967af2c53669a0f4c2d16abc6617c851014261a1358dea07c6"
HISTORICAL_TM031_DEV_SHA = "56dc67affd8fe5d2bb2263dc617c4d28922bfbac03c40b8bf8f6b7cccad67d9a"
HISTORICAL_TM031_ADDENDUM_SHA = "5be880f3edf1549e75a6929d549d4a8d8d8d493a789dbed197443f25b76abdd1"
HISTORICAL_TM031_RUNNER_SHA = "480f7400ada06143acaa3242e75aed315e941f40ebb253a7d0bf67caaa16f564"
FROZEN_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
HISTORICAL_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_prereg_manifest_arms_and_refuse():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 40
    assert p["expected_n_splits"] == 8
    assert p["neural_edit_authorized"] is False
    assert p["treatment_mode"] == "early_raw_half_spacing"
    assert p["write_match_l2"] == 0.05
    assert p["n_cues"] == 8
    assert p["arms"] == ["none", "awake_only", "replay_no_mix", "mix_only", "full_rest"]
    assert p["act_recall_modes_tuple_untouched"] is True
    assert p["not_a_key_or_radius_experiment"] is True
    assert "modify_R" in p["refuse"]
    assert "add early_raw_half_spacing to ACT_RECALL_MODES" in p["refuse"]
    assert 22222 not in p["registry_seeds"]
    iso = json.loads(ISO.read_text())
    assert iso["neural_edit_authorized"] is False
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_historical_tm031_unedited():
    assert _sha(TM031_FAILCLASS) == HISTORICAL_TM031_FAILCLASS_SHA
    assert _sha(TM031_DEC) == HISTORICAL_TM031_DEC_SHA
    assert _sha(TM031_DEV) == HISTORICAL_TM031_DEV_SHA
    assert _sha(TM031_ADDENDUM) == HISTORICAL_TM031_ADDENDUM_SHA
    assert _sha(TM031_RUNNER) == HISTORICAL_TM031_RUNNER_SHA
    p = json.loads(PREREG.read_text())
    assert p["tm031_failclass_sha"] == HISTORICAL_TM031_FAILCLASS_SHA
    assert p["tm031_decision_sha"] == HISTORICAL_TM031_DEC_SHA
    assert p["tm031_dev_sha"] == HISTORICAL_TM031_DEV_SHA
    assert p["tm031_runner_sha"] == HISTORICAL_TM031_RUNNER_SHA


def test_half_mode_tuple_untouched():
    from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2

    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES


def test_expected_cell_ids():
    from experiments.run_tm032restsplit import expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == 40
    assert "split|c8|A_then_B|reg1|awake_only" in ids
    assert "split|c8|B_then_A|reg3|full_rest" in ids


def test_route_split_ladder():
    from experiments.run_tm032restsplit import arm_fixes, route_split

    def rec(*, viol: int, rank: bool, n_ok: int) -> dict:
        return {"n_store_violations": viol, "live_ranking_ok": rank, "n_probe_correct": n_ok}

    fail = rec(viol=3, rank=False, n_ok=6)
    ok = rec(viol=0, rank=True, n_ok=8)
    assert arm_fixes(ok) is True
    assert arm_fixes(fail) is False
    assert route_split({a: ok for a in ("none", "awake_only", "replay_no_mix", "mix_only", "full_rest")}) == "baseline_already_converged"
    assert (
        route_split({"none": fail, "awake_only": ok, "replay_no_mix": ok, "mix_only": fail, "full_rest": ok})
        == "awake_rehearsal_sufficient"
    )
    assert (
        route_split({"none": fail, "awake_only": fail, "replay_no_mix": fail, "mix_only": ok, "full_rest": ok})
        == "slow_mix_sufficient"
    )
    assert (
        route_split({"none": fail, "awake_only": fail, "replay_no_mix": ok, "mix_only": fail, "full_rest": ok})
        == "rest_replay_without_mix_sufficient"
    )
    assert (
        route_split({"none": fail, "awake_only": fail, "replay_no_mix": fail, "mix_only": fail, "full_rest": ok})
        == "full_rest_interaction_only"
    )
    assert (
        route_split({"none": fail, "awake_only": fail, "replay_no_mix": fail, "mix_only": fail, "full_rest": fail})
        == "compute_matched_wall"
    )


def test_smoke():
    from experiments.run_tm032restsplit import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["parent_unchanged"]
    assert out["n_arms"] == 5
    assert out["treatment_mode"] == "early_raw_half_spacing"


def test_dev_decision_all_arms_remain_visible():
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    dec = json.loads(DEC.read_text())
    assert dec["decision"]["code"] == "restsplit_awake_rehearsal_sufficient"
    assert dec["decision"]["v38_route"] == "adaptive_violation_driven_rehearsal"
    assert dec["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert dec["modify_R"] is False
    assert dec["act_recall_modes_tuple_untouched"] is True
    assert dec["not_v38"] is True
    flags = dec["decision"]["phase_flags"]
    assert flags["n_splits"] == 8
    assert flags["n_diagnostic"] == 2
    assert flags["n_baseline_converged"] == 6
    cells = json.loads(DEV.read_text())["cells"]
    assert len(cells) == 40
    by_split = {}
    for c in cells:
        by_split.setdefault((c["order"], c["reg"]), {})[c["arm"]] = c
    assert len(by_split) == 8
    for arms in by_split.values():
        assert set(arms) == {"none", "awake_only", "replay_no_mix", "mix_only", "full_rest"}
        assert arms["none"]["parent_unchanged"] is True
    diag = [arms for arms in by_split.values() if not arms["none"]["fixed"]]
    assert len(diag) == 2
    for arms in diag:
        assert arms["none"]["n_store_violations"] == 3
        assert arms["none"]["n_probe_correct"] == 6
        assert arms["awake_only"]["fixed"] is True
        assert arms["replay_no_mix"]["fixed"] is True
        assert arms["full_rest"]["fixed"] is True
        assert arms["mix_only"]["fixed"] is False
        assert arms["awake_only"]["split_route"] == "awake_rehearsal_sufficient"
