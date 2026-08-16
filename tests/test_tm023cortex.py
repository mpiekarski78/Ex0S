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
    if cand.get("version") == "TM.0.23.CORTEX.CANDIDATE.V10":
        v10_birth = json.loads((REPO_ROOT / "docs" / "cortex_v10_birth.lock").read_text(encoding="utf-8"))
        assert v10_birth["learning_law_ok"] is True
    elif cand.get("version") == "TM.0.23.CORTEX.CANDIDATE.V9":
        v9_birth = json.loads((REPO_ROOT / "docs" / "cortex_v9_birth.lock").read_text(encoding="utf-8"))
        assert v9_birth["learning_law_ok"] is True
    elif cand.get("version") == "TM.0.23.CORTEX.CANDIDATE.V8":
        v8_birth = json.loads((REPO_ROOT / "docs" / "cortex_v8_birth.lock").read_text(encoding="utf-8"))
        assert v8_birth["learning_law_ok"] is True
    elif cand.get("version") == "TM.0.23.CORTEX.CANDIDATE.V7":
        v7_birth = json.loads((REPO_ROOT / "docs" / "cortex_v7_birth.lock").read_text(encoding="utf-8"))
        assert v7_birth["learning_law_ok"] is True
    else:
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
    # versioned candidates preserved; live points at latest candidate
    assert (REPO_ROOT / "docs" / "cortex.candidate.v1.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v2.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v3.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v4.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v5.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex.candidate.v6.lock").exists()
    live = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    v10 = REPO_ROOT / "docs" / "cortex.candidate.v10.lock"
    v9 = REPO_ROOT / "docs" / "cortex.candidate.v9.lock"
    v8 = REPO_ROOT / "docs" / "cortex.candidate.v8.lock"
    v7 = REPO_ROOT / "docs" / "cortex.candidate.v7.lock"
    if v10.exists():
        assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V10"
    elif v9.exists():
        assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V9"
    elif v8.exists():
        assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V8"
    elif v7.exists():
        assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V7"
    else:
        assert live["version"] == "TM.0.23.CORTEX.CANDIDATE.V6"
    # isolation: failed v2/v3 gates frozen; no full D battery on gate worlds
    assert (REPO_ROOT / "docs" / "cortex_v2_gate.failure.lock").exists()
    assert (REPO_ROOT / "docs" / "cortex_v3_gate.failure.lock").exists()
    note = gate.get("note") or ""
    assert "full-development" in note.lower() or "D0-D12" in note or "D0–D12" in note


def test_verify_v4_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v4_gate

    v = verify_v4_gate()
    assert v["ok"] is True, v
    assert v["sensorimotor_association_gate_clear"] is True
    # refuse rewrite: gate lock sha stable across verify
    before = sha(REPO_ROOT / "docs" / "cortex_v4_gate.lock")
    verify_v4_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v4_gate.lock") == before


def test_motor_abi_v5() -> None:
    from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OPS, OP_COST

    assert list(MOTOR_ACT_TOKENS) == []
    with tempfile.TemporaryDirectory(prefix="tm023_mot_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        assert ag.motor_vocab == {}
        assert OP_COST["ACT"] == 0.05
        assert float(ag.b_op[OPS.index("ACT")]) == 0.85
        assert "b_op" not in ag._plastic_names
        assert hasattr(ag, "bind_actuators")
        ag.bind_actuators(["h_a", "h_b"])
        assert "h_a" not in ag.vocab
        assert abs(float((ag.motor_vocab["h_a"] ** 2).sum() ** 0.5) - 1.0) < 1e-6
        v = ag.motor_vocab["h_a"].copy()
        ag.bind_actuators(["h_b", "h_a"])
        assert (ag.motor_vocab["h_a"] == v).all()


def test_mact_boundary_and_v5_gate_failure() -> None:
    mact_v4 = REPO_ROOT / "docs" / "cortex_mact_boundary.lock"
    mact_v5 = REPO_ROOT / "docs" / "cortex_mact_boundary.v5.lock"
    audit_p = REPO_ROOT / "docs" / "cortex_mact_boundary.v5.audit.lock"
    assert mact_v4.exists()
    assert mact_v5.exists()
    assert audit_p.exists()
    v4 = json.loads(mact_v4.read_text(encoding="utf-8"))
    assert v4["all_controls_green"] is False  # planted dictionary reds
    v5 = json.loads(mact_v5.read_text(encoding="utf-8"))
    # Historical lock is immutable; honesty is in the append-only audit.
    assert v5["all_controls_green"] is True
    assert v5["n_ok"] == 8
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    assert audit["historical_lock_rewritten"] is False
    assert audit["contract_honest_all_green"] is False
    assert audit["mact_v5_lock_sha"] == sha(mact_v5)
    c4 = next(c for c in v5["controls"] if c["id"] == "C4_consequence_swap_timed")
    assert c4.get("stale_held_a_permitted") is False
    gate = json.loads((REPO_ROOT / "docs" / "cortex_v5_gate.lock").read_text(encoding="utf-8"))
    assert gate["sensorimotor_association_gate_clear"] is False
    assert gate["battery"]["n_pair_clear"] < 13
    fail = json.loads((REPO_ROOT / "docs" / "cortex_v5_gate.failure.lock").read_text(encoding="utf-8"))
    assert fail["next"] == "isolated_v6"
    iso = json.loads((REPO_ROOT / "docs" / "cortex_v6.isolation.lock").read_text(encoding="utf-8"))
    assert "DEVELOP.v5 on any worlds" in iso["refuse"]
    assert not (REPO_ROOT / "docs" / "cortex_development.v5.lock").exists()


def test_v6_diagnosis_boundary_and_gate_failure() -> None:
    diag = json.loads((REPO_ROOT / "docs" / "cortex_diagnosis.v5.lock").read_text(encoding="utf-8"))
    assert diag["v6_authorized_only_if_this_lock"] is True
    assert "no_consequence_neutrality" in diag["v6_boundary_must_require"]
    v5p = json.loads((REPO_ROOT / "docs" / "cortex_v5.prereg.lock").read_text(encoding="utf-8"))
    v6p = json.loads((REPO_ROOT / "docs" / "cortex_v6.prereg.lock").read_text(encoding="utf-8"))
    assert v6p["eval_seed_commitment"] != v5p["eval_seed_commitment"]
    mact = json.loads((REPO_ROOT / "docs" / "cortex_mact_boundary.v6.lock").read_text(encoding="utf-8"))
    audit_p = REPO_ROOT / "docs" / "cortex_mact_boundary.v6.audit.lock"
    assert mact["all_controls_green"] is True
    assert mact["n_ok"] == 8
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    assert audit["historical_lock_rewritten"] is False
    assert audit["contract_honest_all_green"] is False
    assert audit["mact_v6_lock_sha"] == sha(REPO_ROOT / "docs" / "cortex_mact_boundary.v6.lock")
    c4 = next(c for c in mact["controls"] if c["id"] == "C4_consequence_swap_timed")
    assert c4["stale_ok"] is True
    assert c4["pref_b"] is True
    assert c4.get("counts_stale") == c4.get("counts_before")
    gate = json.loads((REPO_ROOT / "docs" / "cortex_v6_gate.lock").read_text(encoding="utf-8"))
    assert gate["sensorimotor_association_gate_clear"] is False
    assert gate["battery"]["n_pair_clear"] < 13
    fail = json.loads((REPO_ROOT / "docs" / "cortex_v6_gate.failure.lock").read_text(encoding="utf-8"))
    assert fail["n_pair_clear"] == gate["battery"]["n_pair_clear"]
    iso = json.loads((REPO_ROOT / "docs" / "cortex_v7.isolation.lock").read_text(encoding="utf-8"))
    assert "DEVELOP.v6 on any worlds" in iso["refuse"]
    assert not (REPO_ROOT / "docs" / "cortex_development.v6.lock").exists()


def test_v7_stat_diagnosis_and_no_develop() -> None:
    stat = json.loads((REPO_ROOT / "docs" / "cortex_v7_stat_contract.lock").read_text(encoding="utf-8"))
    assert stat["canonical_baseline"] == "97691cd"
    assert stat["thresholds"]["majority_min"] == 24
    assert stat["thresholds"]["mean_delta_min"] == 0.1
    assert stat["thresholds"]["d1_floors"]["press_min"] == 3
    assert stat["thresholds"]["d2_floors"]["holds_min"] == 5
    assert stat["thresholds"]["gate_clear_min_pairs"] == 13
    diag = json.loads((REPO_ROOT / "docs" / "cortex_diagnosis.v6.lock").read_text(encoding="utf-8"))
    assert diag["neural_mechanism_changed"] is False
    assert diag["c4"]["ok"] is True
    note = json.loads((REPO_ROOT / "docs" / "cortex_diagnosis.v6.note.lock").read_text(encoding="utf-8"))
    assert note["historical_lock_rewritten"] is False
    assert note["contract_honest_c5_population"] is True
    assert note["c4_revision_retained"] is True
    assert not (REPO_ROOT / "docs" / "cortex_development.v6.lock").exists()
    assert not (REPO_ROOT / "docs" / "cortex_development.v7.lock").exists()
    iso8 = REPO_ROOT / "docs" / "cortex_v8.isolation.lock"
    if iso8.exists():
        iso = json.loads(iso8.read_text(encoding="utf-8"))
        assert "DEVELOP.v7 on any worlds" in iso["refuse"]
    audit_p = REPO_ROOT / "docs" / "cortex_v7_gate.audit.lock"
    if audit_p.exists():
        audit = json.loads(audit_p.read_text(encoding="utf-8"))
        assert audit["historical_lock_rewritten"] is False
        assert audit["contract_literal_result_stands"] is True
        assert audit["n_pair_clear"] == 0
        assert audit["gate_sha"] == sha(REPO_ROOT / "docs" / "cortex_v7_gate.lock")
    diag7 = REPO_ROOT / "docs" / "cortex_diagnosis.v7.lock"
    if diag7.exists():
        d7 = json.loads(diag7.read_text(encoding="utf-8"))
        assert d7["neural_mechanism_changed"] is False
        assert d7["c4"]["ok"] is True
        assert d7["v8_authorized_only_if_this_lock"] is True
    from three_memory.neural_cortex import MOTOR_ACT_TOKENS

    assert list(MOTOR_ACT_TOKENS) == []


def test_v9_candidate_boundary_pending_gate() -> None:
    cand_p = REPO_ROOT / "docs" / "cortex.candidate.v9.lock"
    if not cand_p.exists():
        return
    cand = json.loads(cand_p.read_text(encoding="utf-8"))
    assert cand["version"] == "TM.0.23.CORTEX.CANDIDATE.V9"
    assert cand["earned_next"] is False
    assert cand["ex0s"] is None
    assert cand["neural_cortex_sha"] == sha(NEURAL_PY)
    v7 = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v7.lock").read_text(encoding="utf-8"))
    assert cand["neural_cortex_sha"] == v7["neural_cortex_sha"]
    mact = json.loads((REPO_ROOT / "docs" / "cortex_mact_boundary.v9.lock").read_text(encoding="utf-8"))
    assert mact["all_required_green"] is True
    required = {"C4_consequence_swap_timed", "C5_plasticity_necessity", "C6_no_consequence_population"}
    assert required.issubset({c["id"] for c in mact["controls"] if c.get("ok")})
    assert not (REPO_ROOT / "docs" / "cortex_development.v9.lock").exists()
    prereg = json.loads((REPO_ROOT / "docs" / "cortex_v9.prereg.lock").read_text(encoding="utf-8"))
    v8p = json.loads((REPO_ROOT / "docs" / "cortex_v8.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["eval_seed_commitment"] != v8p["eval_seed_commitment"]
    assert prereg["d1_bind"] == ["press", "harm"]
    gate_p = REPO_ROOT / "docs" / "cortex_v9_gate.lock"
    if gate_p.exists():
        gate = json.loads(gate_p.read_text(encoding="utf-8"))
        assert gate["sensorimotor_association_gate_clear"] is False
        assert gate["battery"]["n_pair_clear"] == 6
        assert gate["battery"]["d1_bind"] == ["press", "harm"]
        assert gate["battery"]["stage_ok_counts"]["D0"] == 32
        assert not (REPO_ROOT / "docs" / "cortex_development.v9.lock").exists()
        fail = json.loads((REPO_ROOT / "docs" / "cortex_v9_gate.failure.lock").read_text(encoding="utf-8"))
        assert fail["n_pair_clear"] == 6
        iso = json.loads((REPO_ROOT / "docs" / "cortex_v10.isolation.lock").read_text(encoding="utf-8"))
        assert "DEVELOP.v9 on any worlds" in iso["refuse"]


def test_verify_v7_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v7_gate

    v = verify_v7_gate()
    assert v["ok"] is True, v
    gate_p = REPO_ROOT / "docs" / "cortex_v7_gate.lock"
    if not (REPO_ROOT / "docs" / "cortex.candidate.v7.lock").exists() or not gate_p.exists():
        assert v.get("pending") is True
        return
    assert v.get("pending") is False
    before = sha(gate_p)
    verify_v7_gate()
    assert sha(gate_p) == before
    if v["sensorimotor_association_gate_clear"]:
        assert v["n_pair_clear"] >= 13
        assert not (REPO_ROOT / "docs" / "cortex_v7_gate.failure.lock").exists()
    else:
        assert v["refuse_develop_before_clear"] is True
        assert (REPO_ROOT / "docs" / "cortex_v7_gate.failure.lock").exists()


def test_verify_v8_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v8_gate

    v = verify_v8_gate()
    assert v["ok"] is True, v
    if not (REPO_ROOT / "docs" / "cortex.candidate.v8.lock").exists() or not (
        REPO_ROOT / "docs" / "cortex_v8_gate.lock"
    ).exists():
        assert v.get("pending") is True
        return
    assert v.get("pending") is False
    before = sha(REPO_ROOT / "docs" / "cortex_v8_gate.lock")
    verify_v8_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v8_gate.lock") == before
    if v["sensorimotor_association_gate_clear"]:
        assert v["n_pair_clear"] >= 13
    else:
        assert v["refuse_develop_before_clear"] is True
        assert (REPO_ROOT / "docs" / "cortex_v8_gate.failure.lock").exists()
        assert not (REPO_ROOT / "docs" / "cortex_development.v8.lock").exists()


def test_verify_v9_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v9_gate

    v = verify_v9_gate()
    assert v["ok"] is True, v
    if not (REPO_ROOT / "docs" / "cortex.candidate.v9.lock").exists() or not (
        REPO_ROOT / "docs" / "cortex_v9_gate.lock"
    ).exists():
        assert v.get("pending") is True
        return
    assert v.get("pending") is False
    before = sha(REPO_ROOT / "docs" / "cortex_v9_gate.lock")
    verify_v9_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v9_gate.lock") == before
    if v["sensorimotor_association_gate_clear"]:
        assert v["n_pair_clear"] >= 13
    else:
        assert v["refuse_develop_before_clear"] is True
        assert (REPO_ROOT / "docs" / "cortex_v9_gate.failure.lock").exists()
        assert not (REPO_ROOT / "docs" / "cortex_development.v9.lock").exists()


def test_verify_v10_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v10_gate

    v = verify_v10_gate()
    assert v["ok"] is True, v
    if not (REPO_ROOT / "docs" / "cortex.candidate.v10.lock").exists() or not (
        REPO_ROOT / "docs" / "cortex_v10_gate.lock"
    ).exists():
        assert v.get("pending") is True
        return
    assert v.get("pending") is False
    before = sha(REPO_ROOT / "docs" / "cortex_v10_gate.lock")
    verify_v10_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v10_gate.lock") == before
    if v["sensorimotor_association_gate_clear"]:
        assert v["n_pair_clear"] >= 13
    else:
        assert v["refuse_develop_before_clear"] is True
        assert (REPO_ROOT / "docs" / "cortex_v10_gate.failure.lock").exists()
        assert not (REPO_ROOT / "docs" / "cortex_development.v10.lock").exists()
        gate = json.loads((REPO_ROOT / "docs" / "cortex_v10_gate.lock").read_text(encoding="utf-8"))
        assert gate["battery"]["n_pair_clear"] == 8
        assert gate["battery"]["stage_ok_counts"]["D1"] == 32
        assert gate["battery"]["population_d1"]["ok"] is True
        assert gate["battery"]["population_d2"]["ok"] is True
        iso = json.loads((REPO_ROOT / "docs" / "cortex_v11.isolation.lock").read_text(encoding="utf-8"))
        assert "DEVELOP.v10 on any worlds" in iso["refuse"]


def test_verify_v11_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v11_gate

    v = verify_v11_gate()
    assert v["ok"] is True, v
    if not (REPO_ROOT / "docs" / "cortex.candidate.v11.lock").exists() or not (
        REPO_ROOT / "docs" / "cortex_v11_gate.lock"
    ).exists():
        assert v.get("pending") is True
        return
    assert v.get("pending") is False
    if v["sensorimotor_association_gate_clear"]:
        assert v["n_pair_clear"] >= 13
    else:
        assert v["refuse_develop_before_clear"] is True
        assert (REPO_ROOT / "docs" / "cortex_v11_gate.failure.lock").exists()
        assert not (REPO_ROOT / "docs" / "cortex_development.v11.lock").exists()


def test_verify_v6_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v6_gate

    v = verify_v6_gate()
    assert v["ok"] is True, v
    assert v.get("pending") is False
    assert v["sensorimotor_association_gate_clear"] is False
    assert v["refuse_develop_before_clear"] is True
    assert v["boundary_v6_claimed_green"] is True
    assert v["boundary_v6_contract_honest_green"] is False
    before = sha(REPO_ROOT / "docs" / "cortex_v6_gate.lock")
    verify_v6_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v6_gate.lock") == before


def test_verify_v5_gate_cli() -> None:
    from experiments.run_tm023cortex import verify_v5_gate

    v = verify_v5_gate()
    assert v["ok"] is True, v
    assert v["sensorimotor_association_gate_clear"] is False
    assert v["refuse_develop_v5"] is True
    before = sha(REPO_ROOT / "docs" / "cortex_v5_gate.lock")
    verify_v5_gate()
    assert sha(REPO_ROOT / "docs" / "cortex_v5_gate.lock") == before


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
    test_motor_abi_v5()
    test_mact_boundary_and_v5_gate_failure()
    test_verify_v5_gate_cli()
    test_v6_diagnosis_boundary_and_gate_failure()
    test_verify_v6_gate_cli()
    test_v7_stat_diagnosis_and_no_develop()
    test_v9_candidate_boundary_pending_gate()
    test_verify_v7_gate_cli()
    test_verify_v8_gate_cli()
    test_verify_v9_gate_cli()
    test_verify_v10_gate_cli()
    test_verify_v11_gate_cli()
    test_sealed_not_used_in_smoke()
    test_sanity_live()
    print("test_tm023cortex: ok")
