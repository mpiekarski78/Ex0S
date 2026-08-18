"""MEMLANG-1 freeze tests.

No TM063 diagnostic. No neural_cortex.py edit. No installed W_star.
Do not retune 0.05. TM062 stays frozen. Product 0.0.004.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from three_memory.neural_cortex import EPISODE_MATCH_L2, EPISODE_SLOTS, MEMPROJ_LEARNED
from three_memory.memlang.adapters import make_adapter, unit
from three_memory.memlang.variants import variants_for

PREREG = REPO / "docs" / "memlang1.prereg.lock"
ISO = REPO / "docs" / "memlang1.isolation.lock"
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
NEURAL_SHA = "c1ce6f311d2f6958f74e0d55e195d5e1af9130143e06bce149c415396279439b"
OPAQUE_SHA = "30d3adc68286a45756924dc2109a9347ee733bbe7f4817554aa3b5d4969223aa"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM062_RUNNER_SHA = "c62188da88a9dee4f3a75f26c438e82cd7d063185b50c6c5f18b58e444ffaf76"
TM062_DEC_SHA = "bb529975eee280fbf0d0ca688b5bace90f1bd41f36499716b1ee965a6f89880b"
TM062_ADD_SHA = "93fb656b3065f359eff6e03101a6041de8b9bd0fad96b26e9dee8e9486fe946e"
LADDER = [
    "setup_precondition_fail",
    "runner_constructed_value",
    "observer_used_runner_provenance",
    "ordinary_cortex_broken",
    "offline_reconstruction",
    "action_collapse",
    "fresh_world_fail",
    "later_context_drift",
    "stage_a_integrated_gate",
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
    b = json.loads(BUDGET.read_text())
    assert p["lab"] == "MEMLANG-1"
    assert p["product"] == "0.0.004"
    assert p["neural_edit_authorized"] is False
    assert p["value_adapter_bind_authorized"] is True
    assert p["kqv_edit_authorized"] is False
    assert p["tm063_diagnostic_authorized"] is False
    assert p["tm062_transport_branch_closed"] is True
    assert p["memproj_learned_not_a_stage_a_family"] is True
    assert p["n_dev_repeats"] == 4
    assert p["seed_registry"] == 404900100
    assert 404900062 in p["forbidden_seeds"]
    assert [d["code"] for d in p["stage_a_ladder"]] == LADDER
    assert b["n_families"] == 4
    assert b["max_variants_per_family"] == 25
    assert b["n_train_lives_per_variant"] == 4
    assert b["n_val_lives_per_variant"] == 4
    assert b["max_confirmation_runs"] == 1
    assert CONTRACT.is_file()
    assert iso["neural_edit_authorized"] is False
    assert iso["tm063_diagnostic_authorized"] is False
    assert "three_memory/neural_cortex.py" in iso["historical_immutable"]
    assert "experiments/run_tm062xgen.py" in iso["historical_immutable"]
    assert "experiments/run_tm063.py" in iso["historical_immutable"]
    assert _sha(NEURAL) == NEURAL_SHA
    assert _sha(OPAQUE) == OPAQUE_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert p["manifest_sha"] == MANIFEST
    frozen = p["frozen_runner_sha"]
    assert frozen != "PLACEHOLDER"
    assert frozen == sha_file(STAGE_A)
    assert not CANDIDATE_V41.exists()
    assert MEMPROJ_LEARNED not in b["families"]
    assert p["budget_sha"] == _sha(BUDGET)
    assert not list(REPO.glob("docs/lineage_memlang*.decision.lock"))
    assert not list((REPO / "experiments").glob("run_tm063memlang*.py"))


def test_budget_and_locked_stages():
    b = json.loads(BUDGET.read_text())
    for fam in b["families"]:
        vs = variants_for(fam, max_n=int(b["max_variants_per_family"]))
        assert 1 <= len(vs) <= 25
    for stage in ("b", "c", "d", "e"):
        spec = json.loads((REPO / f"docs/memlang1.stage_{stage}.lock").read_text())
        assert spec["executable"] is False
    orch = ORCH.read_text()
    assert "cortex.candidate.v41.lock" in orch
    assert "install_W_star" in orch
    assert "lineage_release" in orch
    assert not list((REPO / "experiments").glob("run_tm063memlang*.py"))
    assert TM063.is_file()
    src063 = TM063.read_text()
    assert "MEMLANG" not in src063
    from experiments.run_memlang1 import refuse_locked_stages

    refuse_locked_stages(stage="A")
    for stage in ("B", "C", "D", "E"):
        try:
            refuse_locked_stages(stage=stage)
            raise AssertionError(stage)
        except RuntimeError as exc:
            assert "locked" in str(exc).lower()


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
    src = STAGE_A.read_text() + (REPO / "three_memory/memlang/capture.py").read_text()
    assert "write_opaque_kv" in src
    assert "MemlangReceipts" in src
    assert "MEMPROJ_LEARNED" not in STAGE_A.read_text()
    assert "learned_projection" not in STAGE_A.read_text()
    assert refuse_runner_leaks(STAGE_A) == []
    assert receipt_contract_violations(STAGE_A) == []
    assert receipt_contract_violations(REPO / "three_memory/memlang/capture.py") == []
    names = []
    for node in ast.walk(ast.parse(STAGE_A.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    assert "_run_joint_socp_consolidation" not in names
    out = smoke()
    assert out["smoke_ok"]
    assert out["n_cells"] == 10
    assert out["floor"] == 0.05
    assert out["candidate_exists"] is False
    assert out["install_W_star"] is False
    assert out["ladder"]["setup_precondition_fail"] == "setup_precondition_fail"
    assert out["ladder"]["later_context_drift"] == "later_context_drift"
    assert out["ladder"]["stage_a_integrated_gate"] == "stage_a_integrated_gate"
    ad = make_adapter("identity", 8, {"name": "identity"})
    rho = unit([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    got = ad.value(rho)
    assert abs(float((got * got).sum()) - 1.0) < 1e-9
    assert abs(float((got - rho).sum())) < 1e-9
