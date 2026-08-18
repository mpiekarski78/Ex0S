"""TM046 one-shot fact-retention freeze tests.

No K/Q/V tuning. Leave TM044/TM045 runner, DEV, and decision untouched.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import (
    ACT_RECALL_EARLY_RAW_HALF,
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    GenomeConfig,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_oneshot.prereg.lock"
ISO = REPO / "docs" / "lineage_oneshot.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_oneshot_contract.md"
RUNNER = REPO / "experiments" / "run_tm046oneshot.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
LIVE = REPO / "docs" / "cortex.candidate.lock"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
TM045_RUNNER = REPO / "experiments" / "run_tm045memnec.py"
TM045_DEV = REPO / "docs" / "lineage_memnec.dev.lock"
TM045_DEC = REPO / "docs" / "lineage_memnec.decision.lock"
TM044_RUNNER = REPO / "experiments" / "run_tm044memproj.py"
TM044_DEC = REPO / "docs" / "lineage_memproj.decision.lock"
MANIFEST = "78d7967022b0d898e463b0421c9a3f6ede5495c2af3d561cb46e8f2ebdc070f7"
NEURAL_SHA = "b0785af069c79c62bd3972a0a3f03f53f9bfbb7221accfb76061b6ee52bb0f1c"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
LIVE_SHA = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
TM045_RUNNER_SHA = "10717d65aa7d851a5fe4c880413a3d0b0fc23695dfba38083de6fa3466566212"
TM045_DEV_SHA = "7d45c171d2c2fdf690dcbea83b7f8dc622df6e08927d1931ef3a9e03c825dcb6"
TM045_DEC_SHA = "e5905c04bc0f2ed0f28a3a937167360d32e9402408ed7837a74d9f57130c16f1"
TM044_RUNNER_SHA = "9bbde3eafd7c56ea2a39835405fe78221687a49c01ace0261a41710db7a2cfd0"
TM044_DEC_SHA = "bf3fa56665dfad02657307879a2491e3d1315ecc84024f52e51b782bf0d12efb"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm045_and_tm044_untouched():
    assert _sha(TM045_RUNNER) == TM045_RUNNER_SHA
    assert _sha(TM045_DEV) == TM045_DEV_SHA
    assert _sha(TM045_DEC) == TM045_DEC_SHA
    assert _sha(TM044_RUNNER) == TM044_RUNNER_SHA
    assert _sha(TM044_DEC) == TM044_DEC_SHA
    dec045 = json.loads(TM045_DEC.read_text())
    dec044 = json.loads(TM044_DEC.read_text())
    assert dec045["decision"]["code"] == "memory_never_necessary"
    assert dec044["decision"]["code"] == "memory_not_necessary"
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_and_no_kqv():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["lab"] == "TM.0.46.ONESHOT"
    assert p["product"] == "0.0.004"
    assert p["earned_next"] is False
    assert p["neural_edit_authorized"] is False
    assert p["kqv_edit_authorized"] is False
    assert p["v41_candidate_authorized"] is False
    assert p["n"] == 64
    assert p["n_handles"] == 4
    assert p["n_facts"] == 4
    assert p["n_worlds"] == 2
    assert p["n_delay_ticks"] == 4
    assert p["n_dev_repeats"] == 4
    assert p["n_rest_ticks"] == 8
    assert p["expected_n_cells"] == 34
    assert p["seed_registry"] == 404600046
    assert 404500045 in p["forbidden_seeds"]
    assert p["tm045_decision_code"] == "memory_never_necessary"
    assert p["tm045_decision_sha"] == TM045_DEC_SHA
    assert p["tm045_runner_sha"] == TM045_RUNNER_SHA
    assert p["joint_socp_sha"] == JOINT_SOCP_SHA
    assert p["neural_cortex_sha"] == NEURAL_SHA
    assert p["freeze_slow_cortex_during_facts"] is True
    assert p["slow_cortex_enabled_is_observational"] is True
    assert p["opaque_projection_is_observational"] is True
    assert p["arms"] == [
        "symbolic_oracle",
        "opaque_projection",
        "no_persistent_memory",
        "slow_cortex_enabled",
    ]
    assert p["conditions"] == ["immediate", "delayed", "distractor", "revision"]
    assert [d["code"] for d in p["decision_ladder"]] == [
        "setup_precondition_fail",
        "generic_reinstatement_fail",
        "memory_necessary_at",
        "memory_never_necessary",
    ]
    assert "tune_kqv" in p["refuse"]
    assert "increase_cue_count_grid" in p["refuse"]
    assert "rerun TM045 DEV" in p["refuse"]
    assert "auto_cortex.candidate.v41.lock" in iso["refuse"]
    assert "experiments/run_tm045memnec.py" in iso["historical_immutable"]
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
    from experiments.run_tm046oneshot import _decision, expected_cell_ids, synthetic_grid

    ids = expected_cell_ids()
    assert len(ids) == 34
    assert ids[0] == "decoder|w0"
    assert ids[1] == "decoder|w1"
    assert ids[2] == "immediate|symbolic_oracle|w0"
    assert ids[-1] == "revision|slow_cortex_enabled|w1"
    code, _, fl = _decision(synthetic_grid(decoder_ok=False))
    assert code == "setup_precondition_fail"
    assert fl["earned_kqv"] is False
    code2, _, fl2 = _decision(synthetic_grid(oracle_immediate_ok=False))
    assert code2 == "generic_reinstatement_fail"
    assert fl2["earned_interface"] is True
    assert fl2["earned_kqv"] is False
    code3, _, fl3 = _decision(synthetic_grid(none_fail_id="delayed|no_persistent_memory|w1"))
    assert code3 == "memory_necessary_at"
    assert fl3["necessary_cell"] == "delayed|no_persistent_memory|w1"
    assert fl3["earned_kqv"] is True
    code4, _, fl4 = _decision(synthetic_grid())
    assert code4 == "memory_never_necessary"
    assert fl4["necessary_cell"] is None
    slow_fail = synthetic_grid()
    for c in slow_fail:
        if c["arm"] in ("slow_cortex_enabled", "opaque_projection"):
            c["passed"] = False
    code5, _, _ = _decision(slow_fail)
    assert code5 == "memory_never_necessary"


def test_runner_refuses_v41_and_smoke():
    from experiments.run_tm046oneshot import refuse_runner_leaks, smoke

    src = RUNNER.read_text()
    assert "cortex.candidate.v41.lock" in src
    assert "freeze_plasticity" in src
    assert "rest_epoch" in src
    assert "empty_tick" in src
    assert "event_memory_scores" in src
    assert "early_raw_half_spacing" not in src
    assert refuse_runner_leaks(RUNNER) == []
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    assert "retrieve_by_query" not in names
    assert "event_memory_scores" in names
    assert "empty_tick" in names
    assert "rest_epoch" in names
    assert "freeze_plasticity" in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 34
    assert out["retrieve_leak"] == []
    assert out["ladder_setup"] == "setup_precondition_fail"
    assert out["ladder_interface"] == "generic_reinstatement_fail"
    assert out["ladder_necessary"] == "memory_necessary_at"
    assert out["ladder_never"] == "memory_never_necessary"
    assert out["candidate_exists"] is False
    assert out["kqv_edit_authorized"] is False
    assert out["memproj_in_genome"] is False
    assert "memproj_arm" not in GenomeConfig().to_dict()
