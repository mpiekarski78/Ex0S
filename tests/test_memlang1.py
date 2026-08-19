"""MEMLANG-1 freeze tests.

No TM063 diagnostic. No installed W_star.
Do not retune 0.05. TM062 stays frozen. Product 0.0.004.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from three_memory.neural_cortex import EPISODE_MATCH_L2, EPISODE_SLOTS, MEMPROJ_LEARNED
from three_memory.memlang.adapters import make_adapter, unit
from three_memory.memlang.telemetry import current_identity, skippable
from three_memory.memlang.variants import variants_for

PREREG = REPO / "docs" / "memlang1.prereg.lock"
ISO = REPO / "docs" / "memlang1.isolation.lock"
ISO_ADD = REPO / "docs" / "memlang1.isolation.addendum.lock"
CONTRACT = REPO / "docs" / "memlang1_contract.md"
BUDGET = REPO / "docs" / "memlang1.budget.lock"
STAGE_A = REPO / "experiments" / "run_memlang1_stage_a.py"
ORCH = REPO / "experiments" / "run_memlang1.py"
TM062_RUNNER = REPO / "experiments" / "run_tm062xgen.py"
TM062_DEC = REPO / "docs" / "lineage_xgen.decision.lock"
TM062_ADD = REPO / "docs" / "lineage_xgen.decision.addendum.lock"
TM063 = REPO / "experiments" / "run_tm063.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
OPAQUE = REPO / "three_memory" / "opaque_memory.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
CANDIDATE_V41 = REPO / "docs" / "cortex.candidate.v41.lock"
MANIFEST = "bc09aab32f71e4a32b10436cfe91ab5d31dee8e185a5206fe3d59479dc741f11"
OPAQUE_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM062_RUNNER_SHA = "c62188da88a9dee4f3a75f26c438e82cd7d063185b50c6c5f18b58e444ffaf76"
TM062_DEC_SHA = "bb529975eee280fbf0d0ca688b5bace90f1bd41f36499716b1ee965a6f89880b"
TM062_ADD_SHA = "93fb656b3065f359eff6e03101a6041de8b9bd0fad96b26e9dee8e9486fe946e"
LADDER = [
    "setup_precondition_fail",
    "runner_constructed_value",
    "handle_copied_into_S",
    "observer_used_runner_provenance",
    "ordinary_cortex_broken",
    "offline_reconstruction",
    "checkpoint_restore_fail",
    "feedback_off_fail",
    "permuted_feedback_fail",
    "reward_gate_fail",
    "action_collapse",
    "cue_overfit",
    "fresh_world_fail",
    "later_context_drift",
    "stage_a_integrated_pass",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_tm062_closed_and_narrow():
    assert _sha(TM062_RUNNER) == TM062_RUNNER_SHA
    assert _sha(TM062_DEC) == TM062_DEC_SHA
    assert _sha(TM062_ADD) == TM062_ADD_SHA
    dec = json.loads(TM062_DEC.read_text())
    add = json.loads(TM062_ADD.read_text())
    assert dec["decision"]["code"] == "only_unrestricted_interpolates"
    assert dec["architectural_conclusion"] == "none"
    assert add["rewrite_historical_decision"] is False
    assert add["transport_branch_closed"] is True
    assert add["authorized_next"] == "MEMLANG-1"
    assert EPISODE_MATCH_L2 == 0.05
    assert EPISODE_SLOTS == 8
    assert not CANDIDATE_V41.exists()


def test_prereg_pins_memlang_program():
    from three_memory.cortex_lineage import sha_file

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    add = json.loads(ISO_ADD.read_text())
    b = json.loads(BUDGET.read_text())
    assert p["lab"] == "MEMLANG-1"
    assert p["product"] == "0.0.004"
    assert p["kqv_edit_authorized"] is False
    assert p["tm063_diagnostic_authorized"] is False
    assert add["identity_default_value_hook_authorized"] is True
    assert add["kqv_edit_authorized"] is False
    assert [d["code"] for d in p["stage_a_ladder"]] == LADDER
    assert b["max_variants_per_family"] == 25
    assert CONTRACT.is_file()
    assert iso["tm063_diagnostic_authorized"] is False
    assert "experiments/run_tm062xgen.py" in iso["historical_immutable"]
    assert _sha(OPAQUE) == OPAQUE_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(STAGE_A)
    assert not CANDIDATE_V41.exists()
    assert MEMPROJ_LEARNED not in b["families"]
    src_n = NEURAL.read_text()
    assert "form_write_value" in src_n
    assert "W_k =" not in src_n.split("def form_write_value")[1][:400]


def test_budget_and_locked_stages():
    b = json.loads(BUDGET.read_text())
    for fam in b["families"]:
        vs = variants_for(fam, max_n=int(b["max_variants_per_family"]))
        assert 1 <= len(vs) <= 25
    for stage in ("b", "c", "d", "e"):
        spec = json.loads((REPO / f"docs/memlang1.stage_{stage}.lock").read_text())
        assert spec["executable"] is False
    from experiments.run_memlang1 import refuse_locked_stages

    refuse_locked_stages(stage="A")
    for stage in ("B", "C", "D", "E"):
        try:
            refuse_locked_stages(stage=stage)
            raise AssertionError(stage)
        except RuntimeError as exc:
            assert "locked" in str(exc).lower()


def test_no_motor_pad_and_v2_skip():
    ad = make_adapter("hebbian_delta", 64, {"name": "t", "eta": 0.05})
    ad.observe_motor(np.ones(32), 1.0)
    assert ad.last_motor is not None
    assert int(ad.last_motor.size) == 32
    src = (REPO / "three_memory/memlang/adapters.py").read_text()
    assert "z[:n_copy]" not in src
    ident = current_identity()
    cfg = {"family": "identity", "name": "identity"}
    world = {"seed_registry": 404900100, "domains": {"DEV": "x"}, "n_worlds": 2}
    bad = {"telemetry_schema": "v1", "status": "complete", "n_cells": 10, "decision_code": "later_context_drift", "config": cfg}
    assert skippable(bad, cfg=cfg, ident=ident, world_seed=world) is False
    good = {
        "telemetry_schema": ident["telemetry_schema"],
        "status": "complete",
        "n_cells": 10,
        "decision_code": "later_context_drift",
        "implementation_sha": ident["implementation_sha"],
        "runner_schema_sha": ident["runner_schema_sha"],
        "config": cfg,
        "genome_checkpoint": {"w0_hash": ["a"]},
        "world_seed": world,
    }
    assert skippable(good, cfg=cfg, ident=ident, world_seed=world) is True
    mismatched = dict(good)
    mismatched["implementation_sha"] = "0" * 64
    assert skippable(mismatched, cfg=cfg, ident=ident, world_seed=world) is False


def test_stage_a_smoke_and_receipts():
    from experiments.run_memlang1_stage_a import BEHAVIORAL_LADDER, _decision, expected_cell_ids, refuse_runner_leaks, smoke, synthetic_grid
    from experiments.run_tm059receipt import receipt_contract_violations

    ids = expected_cell_ids()
    assert ids[0] == "decoder|w0"
    assert len(ids) == 10
    assert list(BEHAVIORAL_LADDER) == LADDER[1:]
    for step in LADDER[1:]:
        c, _, flags = _decision(synthetic_grid(code=step))
        assert c == step, (step, c)
        assert flags["install_W_star"] is False
    assert refuse_runner_leaks(STAGE_A) == []
    assert receipt_contract_violations(STAGE_A) == []
    assert receipt_contract_violations(REPO / "three_memory/memlang/capture.py") == []
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 10
    assert out["floor"] == 0.05
    assert out["candidate_exists"] is False
    assert out["ladder"]["later_context_drift"] == "later_context_drift"
    assert out["ladder"]["stage_a_integrated_pass"] == "stage_a_integrated_pass"
    assert out["ladder"]["feedback_off_fail"] == "feedback_off_fail"
    ad = make_adapter("identity", 8, {"name": "identity"})
    rho = unit([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    got = ad.value(rho)
    assert abs(float((got * got).sum()) - 1.0) < 1e-9
