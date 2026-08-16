"""TM.0.23.CORTEX / DEVELOP regression: birth freeze + develop provenance gates."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm023cortex import (
    BIRTH_LOCK,
    CANDIDATE_LOCK,
    CANDIDATE_V1,
    CONTRACT,
    DEV_CONTRACT,
    DEV_LOCK,
    DEV_PREREG,
    DEV_RUNNER_LOCK,
    EVAL_REVEAL_LOCK,
    FIXTURE_DEV,
    FIXTURE_EVAL,
    GEN_LOCK,
    MEMORY_PY,
    NEURAL_PY,
    PREREG,
    PREREG_WALL,
    SANITY_AMENDMENT,
    WALL_LOCK,
    make_cortex,
    run_sanity,
    verify_prereg,
    verify_pre_reveal,
    verify_sanity_amendment,
)
from three_memory.neural_cortex import NeuralCortex


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg() -> None:
    ok, why, lock = verify_prereg()
    assert ok, why
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["stats"]["earn_pairs"] == 16
    assert lock["genome"]["n"] == 64
    assert "agent_sha" not in lock
    assert PREREG_WALL.exists()
    wall = json.loads(PREREG_WALL.read_text(encoding="utf-8"))
    assert wall["mechanism_changes_permitted"] is False
    assert wall["need_not_fully_pass"] is True


def test_contract_and_fixtures() -> None:
    assert CONTRACT.exists()
    assert "sequential" in CONTRACT.read_text(encoding="utf-8").lower()
    assert FIXTURE_DEV.exists()
    assert GEN_LOCK.exists()
    assert DEV_CONTRACT.exists()
    text = DEV_CONTRACT.read_text(encoding="utf-8")
    assert "α = 0.01" in text or "alpha" in text.lower()
    assert "D0" in text and "D12" in text
    dev = json.loads(FIXTURE_DEV.read_text(encoding="utf-8"))
    for w in dev["worlds"]:
        assert "organism_events" in w and "scorer_only" in w
        for ev in w["organism_events"]:
            if ev.get("op") == "observe":
                assert "homeostatic_delta" not in ev["event"]
                assert "body_state" in ev["event"]


def test_factory_isolation() -> None:
    with tempfile.TemporaryDirectory(prefix="tm023_iso_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        assert isinstance(ag, NeuralCortex)
        assert not hasattr(ag, "interpret_message")
        assert not hasattr(ag, "plan_inquiry")
        src = Path(REPO_ROOT / "experiments" / "run_tm023cortex.py").read_text(encoding="utf-8")
        assert "from experiments.run_tm022interpret import" not in src
        assert "from three_memory.agent import" not in src
        assert "make_interpret(" not in src


def test_abi_reject() -> None:
    with tempfile.TemporaryDirectory(prefix="tm023_abi_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        bad = ag.observe({"source_token": "a"})
        assert bad["why"] == "exact_key_reject"
        bad2 = ag.observe(
            {
                "interaction_token": "i",
                "source_token": "s",
                "ordered_symbols": ["a"],
                "observable_state": [],
                "body_state": [0.5, 0.2, 0.5, 0.0],
                "homeostatic_delta": 1.0,
            }
        )
        assert bad2["why"] == "banned_key"


def test_birth_and_candidate_frozen() -> None:
    assert BIRTH_LOCK.exists()
    birth = json.loads(BIRTH_LOCK.read_text(encoding="utf-8"))
    assert birth["learning_law_ok"] is True
    assert birth["earned_next"] is False
    assert birth["ex0s"] is None
    assert CANDIDATE_LOCK.exists()
    assert CANDIDATE_V1.exists()
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand["learning_law_ok"] is True
    assert cand["factory"] == "experiments.run_tm023cortex.make_cortex"
    # neural organism unchanged vs candidate
    assert sha(NEURAL_PY) == cand["neural_cortex_sha"]
    assert sha(MEMORY_PY) == cand["cortex_memory_sha"]


def test_sanity_amendment_append_only() -> None:
    ok, why, am = verify_sanity_amendment()
    assert ok, why
    assert am["neural_mechanism_changed"] is False
    assert am["learning_law_tests"] == [
        "order_ab_ba",
        "prediction",
        "advantage_path",
        "exploration",
        "write_retrieve",
        "checkpoint",
        "rho_reset",
        "scorer_isolation",
    ]
    assert am["accelerator_tests"] == ["cpu_gpu"]
    adv = am["birth_evidence"]["advantage_path"]
    assert adv["beneficial_increases_responsible_action_probability"] is True
    assert adv["harmful_decreases_responsible_action_probability"] is True
    # original locks not rewritten relative to amendment pins
    assert sha(PREREG) == am["original_prereg_sha"]
    assert sha(CANDIDATE_V1) == am["original_candidate_sha"]
    assert sha(BIRTH_LOCK) == am["original_birth_sha"]


def test_v1_architecture_contract_untouched() -> None:
    # pinned birth contract SHA must remain the v1 file
    assert sha(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"


def test_diag_and_v4_gate() -> None:
    diag = REPO_ROOT / "docs" / "cortex_diag.lock"
    assert diag.exists()
    d = json.loads(diag.read_text(encoding="utf-8"))
    assert d["trace_purity_ok"] is True
    assert d["neural_mechanism_changed"] is False
    assert (REPO_ROOT / "docs" / "cortex_diagnosis.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex_v2_architecture_amendment.lock").exists()
    v2_runner = json.loads((REPO_ROOT / "docs" / "cortex_v2_gate.runner.lock").read_text(encoding="utf-8"))
    assert "candidate_sha" not in v2_runner
    assert "candidate_interface" in v2_runner
    v1_commit = json.loads(PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    v4_prereg = json.loads((REPO_ROOT / "docs" / "cortex_v4.prereg.lock").read_text(encoding="utf-8"))
    assert v4_prereg["eval_seed_commitment"] != v1_commit
    gate = json.loads((REPO_ROOT / "docs" / "cortex_v4_gate.lock").read_text(encoding="utf-8"))
    assert gate["sensorimotor_association_gate_clear"] is True
    assert gate["battery"]["n_pair_clear"] >= 13
    assert gate["earned_next"] is False
    assert gate["ex0s"] is None
    assert gate["product"] == "0.0.004"
    # versioned candidates preserved; live points at v4
    assert (REPO_ROOT / "docs" / "cortex.candidate.v1.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v2.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v3.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v4.lock").exists()
    live = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V4"
    # isolation: failed v2/v3 gates frozen; no full D battery on gate worlds
    assert (REPO_ROOT / "docs" / "cortex_v2_gate.failure.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex_v3_gate.failure.lock").exists()
    note = gate.get("note") or ""
    assert "full-development" in note.lower() or "D0-D12" in note or "D0–D12" in note


def test_verify_v4_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v4_gate, write_v4_math_audit

    audit = write_v4_math_audit(write_lock=True)
    assert audit["ok"] is True, audit
    v = verify_v4_gate()
    assert v["ok"] is True, v
    assert v["sensorimotor_association_gate_clear"] is True
    # refuse rewrite: gate lock sha stable across verify
    before = sha(REPO_ROOT / "docs" / "cortex_v4_gate.lock")
    verify_v4_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v4_gate.lock") == before


def test_motor_lexicon_v4() -> None:
    with tempfile.TemporaryDirectory(prefix="tm023_mot_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        assert set(ag.motor_vocab.keys()) == {"press", "harm"}
        from three_memory.neural_cortex import OPS, OP_COST

        assert OP_COST["ACT"] == 0.05
        assert float(ag.b_op[OPS.index("ACT")]) == 0.85
        assert "b_op" not in ag._plastic_names


def test_runner_lock_no_eval_fixture_pin() -> None:
    assert DEV_RUNNER_LOCK.exists()
    runner = json.loads(DEV_RUNNER_LOCK.read_text(encoding="utf-8"))
    assert "eval_fixture_sha" not in runner
    assert runner["eval_revealed"] is False
    assert "d0_chance" in runner
    assert runner["d0_chance"]["n_probes"] == 64
    assert runner["d0_chance"]["alpha"] == 0.01


def test_reveal_and_compose() -> None:
    assert EVAL_REVEAL_LOCK.exists()
    assert FIXTURE_EVAL.exists()
    assert DEV_PREREG.exists()
    reveal = json.loads(EVAL_REVEAL_LOCK.read_text(encoding="utf-8"))
    assert reveal["commitment_verified"] is True
    assert len(bytes.fromhex(reveal["seed_hex"])) == 32
    assert len(bytes.fromhex(reveal["salt_hex"])) == 32
    assert reveal["scorer_only_isolation_ok"] is True
    compose = json.loads(DEV_PREREG.read_text(encoding="utf-8"))
    assert compose["runner_lock_sha"] == sha(DEV_RUNNER_LOCK)
    assert compose["eval_reveal_sha"] == sha(EVAL_REVEAL_LOCK)


def test_develop_results_fields() -> None:
    assert DEV_LOCK.exists()
    lock = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert "development_gate_clear" in lock
    assert "eligible_for_000005" in lock
    assert WALL_LOCK.exists()
    wall = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    assert wall["diagnostic"] is True
    assert wall["cannot_negate_development_gate"] is True


def test_sealed_not_used_in_smoke() -> None:
    src = (REPO_ROOT / "experiments" / "run_tm023cortex.py").read_text(encoding="utf-8")
    assert "SEALED_EVAL.read_text" not in src.split("def run_sanity")[1].split("def _write_results")[0]


def test_sanity_live() -> None:
    summary = run_sanity(write_birth=False, write_candidate=False)
    assert summary["learning_law_ok"] is True, summary["results"]
    assert summary["ok"] is True, summary["results"]
    adv = next(r for r in summary["results"] if r["id"] == "advantage_path")
    assert "beneficial_increases_responsible_action_probability" in adv
    assert "harmful_decreases_responsible_action_probability" in adv


def test_scorer_audit_v1_preserved() -> None:
    v1 = REPO_ROOT / "docs" / "cortex_development.v1.lock"
    assert v1.exists()
    old = json.loads(v1.read_text(encoding="utf-8"))
    # v1 had false-positive all-pass on several stages
    assert old["battery"]["stage_pass_counts_main_and_twin"]["D4"] == 32
    v2 = REPO_ROOT / "docs" / "cortex_development.v2.lock"
    assert v2.exists()
    mid = json.loads(v2.read_text(encoding="utf-8"))
    # v2 still rubber-stamped D11; D9 nearly always passed
    assert mid["battery"]["stage_pass_counts_main_and_twin"]["D11"] == 32
    assert mid["battery"]["stage_pass_counts_main_and_twin"]["D9"] >= 20
    v3 = REPO_ROOT / "docs" / "cortex_development.v3.lock"
    assert v3.exists()
    soft = json.loads(v3.read_text(encoding="utf-8"))
    # v3 still passed D12 without developmental skill
    assert soft["battery"]["stage_pass_counts_main_and_twin"]["D12"] == 32
    cur = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
    # audited scorers no longer rubber-stamp D4/D7/D11/D12
    assert cur["battery"]["stage_pass_counts_main_and_twin"]["D4"] == 0
    assert cur["battery"]["stage_pass_counts_main_and_twin"]["D7"] == 0
    assert cur["battery"]["stage_pass_counts_main_and_twin"]["D11"] < 32
    assert cur["battery"]["stage_pass_counts_main_and_twin"]["D12"] < 32
    assert (REPO_ROOT / "experiments" / "cortex_develop_scorers.py").exists()
    assert (REPO_ROOT / "docs" / "cortex_scorer_audit.amendment.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex_scorer_audit.r2.amendment.lock").exists()


if __name__ == "__main__":
    test_prereg()
    test_contract_and_fixtures()
    test_factory_isolation()
    test_abi_reject()
    test_birth_and_candidate_frozen()
    test_sanity_amendment_append_only()
    test_v1_architecture_contract_untouched()
    test_runner_lock_no_eval_fixture_pin()
    test_reveal_and_compose()
    test_develop_results_fields()
    test_scorer_audit_v1_preserved()
    test_diag_and_v4_gate()
    test_verify_v4_gate_cli()
    test_motor_lexicon_v4()
    test_sealed_not_used_in_smoke()
    test_sanity_live()
    print("test_tm023cortex: ok")
