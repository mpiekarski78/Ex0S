"""TM033 adaptive-rehearsal freeze tests. Neural controller required after freeze push."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    ACT_REHEARSE_ADAPTIVE,
    ACT_REHEARSE_FIXED,
    ACT_REHEARSE_V37,
    EPISODE_MATCH_L2,
    EPISODE_REPLAY_EPOCHS,
    GenomeConfig,
    NeuralCortex,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_adaptrehearse.prereg.lock"
CONTRACT = REPO / "docs" / "lineage_adaptrehearse_contract.md"
RUNNER = REPO / "experiments" / "run_tm033adaptrehearse.py"
V38_PREREG = REPO / "docs" / "cortex_v38.prereg.lock"
V38_ISO = REPO / "docs" / "cortex_v38.isolation.lock"
TM032_DEC = REPO / "docs" / "lineage_restsplit.decision.lock"
TM032_DEV = REPO / "docs" / "lineage_restsplit.dev.lock"
TM032_RUNNER = REPO / "experiments" / "run_tm032restsplit.py"
MANIFEST = "5fa2dac5200a6944b0a31530d57983d7dd61cbe84d69ace45ac361a93079a3ad"
HISTORICAL_TM032_DEC_SHA = "43a2e8e05ce7912420a2f76570b05ef92ea4ff338416ae9cc65e112be4c7a4c7"
HISTORICAL_TM032_DEV_SHA = "7c6a8833f8fbf0f0669825395b5142a082f05de0e5aa9c44df56514974c9644d"
HISTORICAL_TM032_RUNNER_SHA = "bd591d293ba8f4023d5ca89d9f812f58b3afeac662301bb772bad03d99f09503"
V38_ISO_SHA = "73543f2f67f6356e2218aa162cdf00db6beeea033a3cb177299efdf9237af866"
V38_PREREG_SHA = "de924a8df50f0b20c902bf3cd3f689e882a162df0d28fe33e509ac6cb074c510"
FROZEN_RUNNER_SHA = "91d8074d74b3724bd38e69b1d3860ea8ad0ba9835eb615a513cc9fcf620f5a49"
HISTORICAL_DEV_SHA = "a049f5fc44e921e77b762e6da9fd0c19285b0b6143beab758e26a22ace337375"
HISTORICAL_DEC_SHA = "cc22cbe82378164d61793b14727639ace35dca7fc95f5f5fc824bbc27ce081e6"
NEURAL_SHA = "53227d84ee2163c223ded10a1bfbaebe39e665ff5a769007936fcd10d51cfebc"
DEV = REPO / "docs" / "lineage_adaptrehearse.dev.lock"
DEC = REPO / "docs" / "lineage_adaptrehearse.decision.lock"
GENOME_TO_DICT_KEYS = {
    "n",
    "d_sym",
    "k_s",
    "d_body",
    "d_x",
    "p_connect",
    "t_max",
    "tau",
    "cos_thresh",
    "eta_pred",
    "eta_act",
    "beta",
    "clip",
    "seed_birth",
    "seed_registry",
    "seed_source",
    "seed_action",
    "seed_permute",
    "seed_motor",
    "dtype",
    "motor_persist_p",
    "act_score_mode",
    "actuator_proto_h_max",
    "episodic_act_recall",
    "act_recall_mode",
    "separator_matrix_sha",
    "body_setpoint",
    "ops",
    "op_cost",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _stuck_pass(n_viol: int, n_updates: int):
    def pass_fn(eta_a, *, pass_index, skip_p1=None, skip_handle=None):
        return {
            "pass_index": pass_index,
            "violations_before": n_viol,
            "violations_after": n_viol,
            "n_updates": n_updates,
            "n_opportunities": n_viol,
        }

    return pass_fn


def test_prereg_budget_plateau_debt_and_limitation():
    p = json.loads(PREREG.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 24
    assert p["hard_budget_passes"] == 16
    assert p["fit_44_row_updates"] is False
    assert p["work_metric"] == "n_awake_updates"
    assert p["plateau"]["tolerance"] == 0
    assert p["plateau"]["unchanged_passes_to_stop"] == 1
    assert p["debt"]["rest_debit_uses"] == "rehearsal_pass_debt"
    assert p["debt"]["reset"] == "after_rest_replay"
    assert p["fixed_extra_replay"]["targeting"] == "violation_rows"
    assert p["fixed_extra_replay"]["plateau_stop"] is False
    assert p["fixed_extra_replay"]["rest_debit"] is False
    assert p["tm032_n_informative_splits"] == 2
    assert p["tm032_informative_replicate"] == "reg1"
    assert p["neural_edit_before_runner_freeze"] is False
    assert p["arms"] == ["v37_awake_cap", "adaptive_violation", "fixed_extra_replay"]
    assert "fit_tm032_44_row_updates_as_constant" in p["refuse"]
    assert "adaptrehearse_debt_integrity_fail" in {d["code"] for d in p["decision_ladder"]}
    assert "adaptrehearse_adaptive_control" in {d["code"] for d in p["decision_ladder"]}
    assert CONTRACT.is_file()
    assert p["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA


def test_tm032_locks_unedited():
    assert _sha(TM032_DEC) == HISTORICAL_TM032_DEC_SHA
    assert _sha(TM032_DEV) == HISTORICAL_TM032_DEV_SHA
    assert _sha(TM032_RUNNER) == HISTORICAL_TM032_RUNNER_SHA
    p = json.loads(PREREG.read_text())
    assert p["tm032_decision_sha"] == HISTORICAL_TM032_DEC_SHA
    assert p["tm032_dev_sha"] == HISTORICAL_TM032_DEV_SHA
    v38p = json.loads(V38_PREREG.read_text())
    assert v38p["fit_44_row_updates"] is False
    assert v38p["hard_budget_passes"] == 16
    assert v38p["plateau"]["unchanged_passes_to_stop"] == 1
    assert _sha(V38_ISO) == V38_ISO_SHA
    assert _sha(V38_PREREG) == V38_PREREG_SHA


def test_half_mode_and_write_radius_untouched():
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_REPLAY_EPOCHS == 16
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert hasattr(NeuralCortex, "set_act_rehearse_arm")
    g = GenomeConfig().to_dict()
    assert set(g) == GENOME_TO_DICT_KEYS
    assert "act_rehearse_arm" not in g


def test_expected_ids_and_routes():
    from experiments.run_tm033adaptrehearse import _decision, expected_cell_ids, route_triple

    ids = expected_cell_ids()
    assert len(ids) == 24
    fail = {
        "n_store_violations": 3,
        "live_ranking_ok": False,
        "n_probe_correct": 6,
        "n_awake_updates": 0,
        "targeting": "violation_rows",
    }
    ok_cheap = {
        "n_store_violations": 0,
        "live_ranking_ok": True,
        "n_probe_correct": 8,
        "n_awake_updates": 10,
        "targeting": "violation_rows",
    }
    ok_dear = {
        "n_store_violations": 0,
        "live_ranking_ok": True,
        "n_probe_correct": 8,
        "n_awake_updates": 44,
        "targeting": "violation_rows",
    }
    bundle_ad = dict(ok_cheap, targeting="all_rows")
    assert route_triple(ok_cheap, ok_cheap, ok_dear) == "v37_already_converged"
    assert route_triple(fail, ok_cheap, fail) == "adaptive_control"
    assert route_triple(fail, bundle_ad, fail) == "controller_bundle"
    assert route_triple(fail, ok_cheap, ok_dear) == "efficient_scheduling"
    assert route_triple(fail, ok_dear, ok_cheap) == "both_pass_compute_tied"
    assert route_triple(fail, fail, ok_dear) == "extra_compute_not_scheduler"
    assert route_triple(fail, fail, fail) == "both_fail"
    leftover = {
        "id": "acquire|c8|A_then_B|reg1|adaptive_violation",
        "arm": "adaptive_violation",
        "rehearsal_pass_debt_after_rest": 2,
        "rehearsal_update_debt_after_rest": 0,
        "awake_budget_exhausted": False,
    }
    code, _then, flags = _decision(["v37_already_converged"] * 8, [leftover])
    assert code == "adaptrehearse_debt_integrity_fail"
    assert flags["n_debt_remaining_after_rest"] == 1
    code2, _t2, flags2 = _decision(["v37_already_converged"] * 8, [])
    assert code2 == "adaptrehearse_no_v37_acquire_fail"
    assert flags2["n_diagnostic"] == 0
    code3, _t3, flags3 = _decision(["adaptive_control", "both_fail"], [])
    assert code3 == "adaptrehearse_mixed_routes"


def test_smoke_v37_arm():
    from experiments.run_tm033adaptrehearse import smoke

    out = smoke()
    assert out["smoke_ok"]
    assert out["arm"] == "v37_awake_cap"
    assert out["hard_budget_passes"] == 16
    assert out["neural_ready"] is True


def test_credit_one_shot_is_not_debt():
    src = inspect.getsource(NeuralCortex._credit_act_p1_episode)
    assert src.index("_apply_act_query_update") < src.index("_run_awake_rehearsal_burst")
    assert "_rehearsal_pass_debt" not in src
    assert "_rehearsal_update_debt" not in src


def test_adaptive_plateau_is_one_unchanged_pass():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_ADAPTIVE)
    ag._violation_signature = lambda: (3, (0, 1, 2))
    ag._gated_rehearsal_pass = _stuck_pass(3, 4)
    burst = ag._run_awake_rehearsal_burst()
    assert burst["n_passes"] == 1
    assert burst["plateau_stopped"] is True
    assert burst["budget_exhausted"] is False
    assert burst["total_updates"] == 4
    assert ag._rehearsal_pass_debt == 1
    assert ag._rehearsal_update_debt == 4
    assert ag._rehearse_targeting == "violation_rows"


def test_fixed_extra_no_plateau_no_debt():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_FIXED)
    ag._violation_signature = lambda: (3, (0, 1, 2))
    ag._gated_rehearsal_pass = _stuck_pass(3, 2)
    burst = ag._run_awake_rehearsal_burst()
    assert burst["n_passes"] == 16
    assert burst["plateau_stopped"] is False
    assert burst["budget_exhausted"] is True
    assert burst["total_updates"] == 32
    assert ag._rehearsal_pass_debt == 0
    assert ag._rehearsal_update_debt == 0


def test_debt_survives_credits_and_debits_rest():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_ADAPTIVE)
    ag._violation_signature = lambda: (2, (0, 1))
    ag._gated_rehearsal_pass = _stuck_pass(2, 5)
    ag._run_awake_rehearsal_burst()
    ag._run_awake_rehearsal_burst()
    assert ag._rehearsal_pass_debt == 2
    assert ag._rehearsal_update_debt == 10
    rest = ag._replay_episodes()
    assert rest["pass_budget"] == 14
    assert len(rest["epochs"]) == 14
    assert ag._rehearsal_pass_debt == 0
    assert ag._rehearsal_update_debt == 0


def test_zero_budget_rest_resets_debt():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_ADAPTIVE)
    ag._rehearsal_pass_debt = 16
    ag._rehearsal_update_debt = 44
    rest = ag._replay_episodes()
    assert rest["pass_budget"] == 0
    assert rest["epochs"] == []
    assert rest["budget_exhausted"] is False
    assert ag._rehearsal_pass_debt == 0
    assert ag._rehearsal_update_debt == 0


def test_fixed_rest_does_not_debit_or_reset():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_FIXED)
    ag._rehearsal_pass_debt = 7
    ag._rehearsal_update_debt = 9
    rest = ag._replay_episodes()
    assert rest["pass_budget"] == 16
    assert ag._rehearsal_pass_debt == 7
    assert ag._rehearsal_update_debt == 9


def test_v37_burst_does_not_accrue_debt():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_V37)
    burst = ag._run_awake_rehearsal_burst()
    assert burst["n_passes"] == 1
    assert burst["total_updates"] == 0
    assert burst["plateau_stopped"] is False
    assert ag._rehearsal_pass_debt == 0
    assert ag._rehearsal_update_debt == 0


def test_default_instance_does_not_activate_controller():
    ag = NeuralCortex()
    assert ag._act_rehearse_arm == ACT_REHEARSE_V37
    ag._violation_signature = lambda: (3, (0, 1, 2))
    ag._gated_rehearsal_pass = _stuck_pass(3, 2)
    burst = ag._run_awake_rehearsal_burst()
    assert burst["n_passes"] == 16
    assert burst["plateau_stopped"] is False
    assert burst["budget_exhausted"] is True
    assert ag._rehearsal_pass_debt == 0
    assert ag._rehearsal_update_debt == 0
    rest = ag._replay_episodes()
    assert rest["pass_budget"] == 16


def test_explicit_v37_matches_default_and_does_not_plateau():
    default = NeuralCortex()
    explicit = NeuralCortex()
    explicit.set_act_rehearse_arm(ACT_REHEARSE_V37)
    default._violation_signature = lambda: (2, (0, 1))
    explicit._violation_signature = lambda: (2, (0, 1))
    default._gated_rehearsal_pass = _stuck_pass(2, 1)
    explicit._gated_rehearsal_pass = _stuck_pass(2, 1)
    b0 = default._run_awake_rehearsal_burst()
    b1 = explicit._run_awake_rehearsal_burst()
    assert b0["n_passes"] == b1["n_passes"] == 16
    assert b0["total_updates"] == b1["total_updates"] == 16
    assert b0["plateau_stopped"] is False
    assert b1["plateau_stopped"] is False
    assert default._rehearsal_pass_debt == explicit._rehearsal_pass_debt == 0


def test_excess_pass_debt_remains_after_rest():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_ADAPTIVE)
    ag._rehearsal_pass_debt = 20
    ag._rehearsal_update_debt = 50
    rest = ag._replay_episodes()
    assert rest["pass_budget"] == 0
    assert ag._rehearsal_pass_debt == 4
    assert ag._rehearsal_update_debt == 50
    leftover = {
        "id": "acquire|c8|A_then_B|reg1|adaptive_violation",
        "arm": "adaptive_violation",
        "rehearsal_pass_debt_after_rest": ag._rehearsal_pass_debt,
        "rehearsal_update_debt_after_rest": ag._rehearsal_update_debt,
        "awake_budget_exhausted": False,
    }
    from experiments.run_tm033adaptrehearse import _decision

    code, _then, flags = _decision(["efficient_scheduling"] * 8, [leftover])
    assert code == "adaptrehearse_debt_integrity_fail"
    assert flags["n_debt_remaining_after_rest"] == 1


def test_checkpoint_serializes_debt_and_arm():
    ag = NeuralCortex()
    ag.set_act_rehearse_arm(ACT_REHEARSE_ADAPTIVE)
    ag._rehearsal_pass_debt = 3
    ag._rehearsal_update_debt = 11
    snap = ag.checkpoint()
    assert snap["rehearsal_pass_debt"] == 3
    assert snap["rehearsal_update_debt"] == 11
    assert snap["act_rehearse_arm"] == ACT_REHEARSE_ADAPTIVE
    twin = NeuralCortex()
    twin.load_checkpoint(snap)
    assert twin._rehearsal_pass_debt == 3
    assert twin._rehearsal_update_debt == 11
    assert twin._act_rehearse_arm == ACT_REHEARSE_ADAPTIVE
    missing = dict(snap)
    del missing["rehearsal_pass_debt"]
    del missing["rehearsal_update_debt"]
    del missing["act_rehearse_arm"]
    bare = NeuralCortex()
    bare.load_checkpoint(missing)
    assert bare._rehearsal_pass_debt == 0
    assert bare._rehearsal_update_debt == 0
    assert bare._act_rehearse_arm == ACT_REHEARSE_V37


def test_dev_lock_records_implementation_and_first_match():
    assert DEV.is_file()
    assert DEC.is_file()
    assert _sha(DEV) == HISTORICAL_DEV_SHA
    assert _sha(DEC) == HISTORICAL_DEC_SHA
    assert _sha(RUNNER) == FROZEN_RUNNER_SHA
    dev = json.loads(DEV.read_text())
    dec = json.loads(DEC.read_text())
    assert dev["git_head"] == "64883d5c8e21d2121f5f541792218c88b83b3312"
    assert dev["neural_sha"] == NEURAL_SHA
    assert dev["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dev["clean_tree"] is True
    assert dev["clean_tree_status"] == "clean"
    assert dev["fit_44_row_updates"] is False
    assert dev["hard_budget_passes"] == 16
    assert dev["decision_code"] == "adaptrehearse_core_acquire_fail"
    assert dev["phase_flags"]["n_diagnostic"] == 2
    assert dev["phase_flags"]["n_debt_remaining_after_rest"] == 0
    assert set(dev["phase_flags"]["routes"]) == {"v37_already_converged", "both_fail"}
    assert dec["decision"]["code"] == "adaptrehearse_core_acquire_fail"
    assert dec["dev_lock_sha"] == HISTORICAL_DEV_SHA
    assert dec["git_head"] == dev["git_head"]
    assert dec["frozen_runner_sha"] == FROZEN_RUNNER_SHA
    assert dec["fit_44_row_updates"] is False
