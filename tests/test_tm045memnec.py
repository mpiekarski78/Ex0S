"""TM045 memory-necessity freeze tests.

No K/Q/V tuning. Leave TM044 runner, DEV, and decision untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    GenomeConfig,
    MEMORY_PATH_EMPTY,
    MEMORY_PATH_EPISODIC,
    MEMORY_PATH_REJECTED,
    MOTOR_PATH_CORTICAL,
    NeuralCortex,
    SCORE_SRC_LIVE,
    SCORE_SRC_REINSTATED,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_memnec.prereg.lock"
ISO = REPO / "docs" / "lineage_memnec.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_memnec_contract.md"
RUNNER = REPO / "experiments" / "run_tm045memnec.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
LIVE = REPO / "docs" / "cortex.candidate.lock"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM044_RUNNER = REPO / "experiments" / "run_tm044memproj.py"
TM044_DEV = REPO / "docs" / "lineage_memproj.dev.lock"
TM044_DEC = REPO / "docs" / "lineage_memproj.decision.lock"
TM044_ADD = REPO / "docs" / "lineage_memproj.decision.addendum.lock"
TM044_PREREG = REPO / "docs" / "lineage_memproj.prereg.lock"
MANIFEST = "b5a181b72f11c5d7bf37768449ae510982519604ab940e3a098a9e80a69c517a"
NEURAL_SHA = "b0785af069c79c62bd3972a0a3f03f53f9bfbb7221accfb76061b6ee52bb0f1c"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
LIVE_SHA = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
TM044_RUNNER_SHA = "9bbde3eafd7c56ea2a39835405fe78221687a49c01ace0261a41710db7a2cfd0"
TM044_DEV_SHA = "e375a4ae9e19f1697dddc8d1055bd34ead6f667c92db575ed3e6512be4a6fc8e"
TM044_DEC_SHA = "bf3fa56665dfad02657307879a2491e3d1315ecc84024f52e51b782bf0d12efb"
TM044_ADD_SHA = "3b28a031798024f443df88bbc01e10fa8eca30d36ff701b39e69f049a4bc06d1"
TM044_NEURAL_SHA = "20d21f91c275e856ea1ec1faec58e5e1e633c270a79c887afa4b5e41397bb5be"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm044_wall_untouched():
    assert _sha(TM044_RUNNER) == TM044_RUNNER_SHA
    assert _sha(TM044_DEV) == TM044_DEV_SHA
    assert _sha(TM044_DEC) == TM044_DEC_SHA
    assert _sha(TM044_ADD) == TM044_ADD_SHA
    p44 = json.loads(TM044_PREREG.read_text())
    dec = json.loads(TM044_DEC.read_text())
    add = json.loads(TM044_ADD.read_text())
    assert p44["neural_cortex_sha"] == TM044_NEURAL_SHA
    assert p44["frozen_runner_sha"] == TM044_RUNNER_SHA
    assert dec["decision"]["code"] == "memory_not_necessary"
    assert add["rerun_dev"] is False
    assert add["frozen_first_match_unchanged"] is True
    assert add["historical_decision_code"] == "memory_not_necessary"
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_and_no_kqv():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.45.MEMNEC"
    assert p["product"] == "0.0.004"
    assert p["earned_next"] is False
    assert p["eligible_for_000005"] is False
    assert p["neural_edit_authorized"] is False
    assert p["kqv_edit_authorized"] is False
    assert p["solver_edit_authorized"] is False
    assert p["recall_edit_authorized"] is False
    assert p["v41_candidate_authorized"] is False
    assert p["n"] == 64
    assert p["n_handles"] == 2
    assert p["n_worlds"] == 2
    assert p["n_delay_ticks"] == 4
    assert p["n_cues_grid"] == [2, 4, 8]
    assert p["expected_n_cells"] == 72
    assert p["seed_registry"] == 404500045
    assert p["seed_registry"] not in p["forbidden_seeds"]
    assert 22222 in p["forbidden_seeds"]
    assert 404400044 in p["forbidden_seeds"]
    assert p["tm044_decision_code"] == "memory_not_necessary"
    assert p["tm044_decision_sha"] == TM044_DEC_SHA
    assert p["tm044_dev_sha"] == TM044_DEV_SHA
    assert p["tm044_runner_sha"] == TM044_RUNNER_SHA
    assert p["tm044_addendum_sha"] == TM044_ADD_SHA
    assert p["joint_socp_sha"] == JOINT_SOCP_SHA
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert p["reset_rho_before_recall"] is True
    assert p["record_scores_before_after_reinstatement"] is True
    assert p["learned_arm_is_observational"] is True
    assert p["arms"] == ["symbolic_oracle", "learned_projection", "no_persistent_memory"]
    assert p["conditions"] == ["immediate", "delayed", "distractor", "revision"]
    assert [d["code"] for d in p["decision_ladder"]] == [
        "setup_precondition_fail",
        "memory_necessary_at",
        "memory_never_necessary",
    ]
    assert "tune_kqv" in p["refuse"]
    assert "rerun TM044 DEV" in p["refuse"]
    assert "edit experiments/run_tm044memproj.py" in p["refuse"]
    assert "auto_cortex.candidate.v41.lock" in p["refuse"]
    assert "tune_kqv" in iso["refuse"]
    assert "experiments/run_tm044memproj.py" in iso["historical_immutable"]
    assert "docs/lineage_memproj.dev.lock" in iso["historical_immutable"]
    assert "docs/lineage_memproj.decision.lock" in iso["historical_immutable"]
    assert CONTRACT.is_file()
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(LIVE) == LIVE_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert not CANDIDATE_V41.exists()
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(RUNNER)


def test_ids_and_decision_ladder():
    from experiments.run_tm045memnec import _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert len(ids) == 72
    assert ids[0] == "immediate|c2|symbolic_oracle|w0"
    assert ids[1] == "immediate|c2|symbolic_oracle|w1"
    assert ids[2] == "immediate|c2|learned_projection|w0"
    assert ids[4] == "immediate|c2|no_persistent_memory|w0"
    assert ids[-1] == "revision|c8|no_persistent_memory|w1"
    code, _, fl = _decision(synthetic_grid(oracle_c2_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["candidate_v41_lock"] is False
    assert fl["kqv_edited"] is False
    assert fl["learned_arm_is_observational"] is True
    code2, _, fl2 = _decision(synthetic_grid(none_fail_id="immediate|c2|no_persistent_memory|w0"))
    assert code2 == "memory_necessary_at"
    assert fl2["necessary_cell"] == "immediate|c2|no_persistent_memory|w0"
    assert fl2["necessary_oracle_cell"] == "immediate|c2|symbolic_oracle|w0"
    assert fl2["n_cues"] == 2
    assert fl2["condition"] == "immediate"
    code3, _, fl3 = _decision(synthetic_grid(none_fail_id="delayed|c4|no_persistent_memory|w1"))
    assert code3 == "memory_necessary_at"
    assert fl3["necessary_cell"] == "delayed|c4|no_persistent_memory|w1"
    code4, _, fl4 = _decision(synthetic_grid())
    assert code4 == "memory_never_necessary"
    assert fl4["necessary_cell"] is None
    learned_fail = synthetic_grid()
    for c in learned_fail:
        if c["arm"] == "learned_projection":
            c["passed"] = False
    code5, _, _ = _decision(learned_fail)
    assert code5 == "memory_never_necessary"


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm045memnec import refuse_runner_leaks, reset_before_recall, smoke

    src = RUNNER.read_text()
    assert "cortex.candidate.v41.lock" in src
    assert "set_act_socp_arm" in src
    assert "reset_rho" in src
    assert "empty_tick" in src
    assert "event_memory_scores" in src
    assert "scores_before_reinstatement" in src
    assert "early_raw_half_spacing" not in src
    assert refuse_runner_leaks(RUNNER) == []
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    assert "retrieve_by_query" not in names
    assert "rest_epoch" not in names
    assert "event_memory_scores" in names
    assert "empty_tick" in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 72
    assert out["retrieve_leak"] == []
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_necessary"] == "memory_necessary_at"
    assert out["ladder_never"] == "memory_never_necessary"
    assert out["necessary_cell"] == "immediate|c2|no_persistent_memory|w0"
    assert out["candidate_exists"] is False
    assert out["kqv_edit_authorized"] is False
    assert out["memproj_in_genome"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
    ag = NeuralCortex()
    ag._last_p1 = np.ones(ag.genome.n, dtype=np.float64)
    ag.rho[0] = 1.0
    reset_before_recall(ag)
    assert ag._last_p1 is None
    assert float(ag.rho.abs().sum()) == 0.0


def test_canonical_telemetry_fields_are_split():
    assert MEMORY_PATH_EPISODIC == "episodic_completed"
    assert MEMORY_PATH_EMPTY == "empty"
    assert MEMORY_PATH_REJECTED == "rejected"
    assert MOTOR_PATH_CORTICAL == "cortical_scoring"
    assert SCORE_SRC_LIVE == "live_rho"
    assert SCORE_SRC_REINSTATED == "reinstated_value"
    src = NEURAL.read_text()
    assert "memory_path" in src
    assert "motor_path" in src
    assert "scoring_address_source" in src
    assert "scores_before_reinstatement" in src
    assert "scores_after_reinstatement" in src
