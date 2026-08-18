"""TM040 measurement addendum + canonical-probe R2 freeze tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from three_memory.neural_cortex import ACT_RECALL_EARLY_RAW_HALF, ACT_RECALL_MODES, EPISODE_MATCH_L2

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "lineage_causalbattery.r2.prereg.lock"
ISO = REPO / "docs" / "lineage_causalbattery.r2.isolation.lock"
CONTRACT = REPO / "docs" / "lineage_causalbattery.r2.contract.md"
MEAS = REPO / "docs" / "lineage_causalbattery.measurement.addendum.lock"
AUDIT = REPO / "docs" / "lineage_probe_boundary.audit.lock"
RUNNER = REPO / "experiments" / "run_tm040causal_r2.py"
CANONICAL = REPO / "experiments" / "canonical_act_probe.py"
NEURAL = REPO / "three_memory" / "neural_cortex.py"
SOLVER = REPO / "three_memory" / "joint_socp.py"
TM040_DEC = REPO / "docs" / "lineage_causalbattery.decision.lock"
TM040_DEV = REPO / "docs" / "lineage_causalbattery.dev.lock"
TM040_RUNNER = REPO / "experiments" / "run_tm040causal.py"
MANIFEST = "1bfe74604d1f0cb26eca700639613aa361c473bcec00e87649dbebffcd16200d"
FROZEN_NEURAL_SHA = "2eb45d8769402330f5ee39a04afffe110a435a0e64a40b12bc2d874b36f5ed59"
JOINT_SOCP_SHA = "ed651a51f8de6cc6ec1d8285c43846c99b47b751ddfea59d3c26db1d63fcc895"
TM040_DEC_SHA = "734204f628362f58e4f3b19237dd82398016544655d2c13800f3408854bd1b99"
TM040_DEV_SHA = "b10865b5f6fea382396db736549488c68dcdc5000932907a3612e29b53354ad7"
TM040_RUNNER_SHA = "0739de3225e36bf88b66001ae9af2c3232a937d0bc6bc3b64eb34bdf7d2f9c6b"
MEAS_SHA = "4eea2517f1b61c39f8615fa434dc4d7c5bae60e3284b5709869eee7591279760"
CANONICAL_SHA = "51e24d272417df5ae689301d9600d49aa86daec6608dfc7ff26f8ad4c2e22aef"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_measurement_addendum_does_not_rewrite_tm040():
    m = json.loads(MEAS.read_text())
    assert m["historical_decision_code"] == "jointsocp_fallback_acquire_fail"
    assert m["interpretation"] == "invalidated_measurement__canonical_path_mismatch"
    assert m["architectural_conclusion"] == "none"
    assert m["rewrite_historical_decision"] is False
    assert m["scientifically_valid_organism_acquire_failure"] is False
    assert m["always_joint_behavioral_acquire_fix"] == "invalidated__fixed_incorrectly_measured_raw_address"
    assert m["tm041_reconstruct"] == "diagnostic_evidence_not_a_replacement_score"
    assert _sha(MEAS) == MEAS_SHA
    assert _sha(TM040_DEC) == TM040_DEC_SHA
    assert _sha(TM040_DEV) == TM040_DEV_SHA
    assert _sha(TM040_RUNNER) == TM040_RUNNER_SHA
    audit = json.loads(AUDIT.read_text())
    leak = [x for x in audit["labs"] if x["lab"] == "TM.0.40.CAUSALBATTERY"][0]
    assert leak["leak"] is True
    assert audit["earlier_decision_addenda_required"] is False


def test_prereg_fresh_domains_and_no_edits():
    from three_memory.cortex_lineage import sha_file
    from experiments.run_tm040causal_r2 import load_prereg

    p = json.loads(PREREG.read_text())
    iso = json.loads(ISO.read_text())
    assert p["manifest_sha"] == MANIFEST
    assert p["expected_n_cells"] == 102
    assert p["domains"]["DEV"] == "TM040.CAUSAL.R2.DEV."
    assert p["seed_registry"] not in (22222, 404000039)
    assert p["v40_candidate_authorized"] is False
    assert p["neural_edit_authorized"] is False
    assert p["behavioral_scorer"] == "actuator_decision_scores"
    assert p["raw_live_scores"] == "diagnostic_only"
    assert "edit_neural_cortex.py" in iso["refuse"]
    assert CONTRACT.is_file()
    assert _sha(NEURAL) == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert sha_file(CANONICAL) == CANONICAL_SHA
    assert p["frozen_runner_sha"] == sha_file(RUNNER)
    assert p["canonical_probe_sha"] == CANONICAL_SHA
    assert EPISODE_MATCH_L2 == 0.05
    assert ACT_RECALL_EARLY_RAW_HALF not in ACT_RECALL_MODES
    assert load_prereg()["historical_decision_code"] == "jointsocp_fallback_acquire_fail"


def test_canonical_probe_matches_motor_and_refuses_raw_gates():
    from experiments.canonical_act_probe import (
        canonical_cue_probe,
        decision_scores,
        motor_loop_uses_decision_scores,
        refuse_raw_behavioral_actuator_scores,
    )
    from experiments.run_tm024writegeom import capacity_world, mapping_pairs
    from experiments.run_tm027gatedrehearsal import teach_pairs
    from experiments.run_tm040causal import _fresh, set_arm
    from experiments.run_tm031halfspace import arr_sha

    assert motor_loop_uses_decision_scores()
    assert refuse_raw_behavioral_actuator_scores() == []
    world = capacity_world(0, "TM040.CAUSAL.R2.DEV.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="r2_id_") as tmp:
        ag = _fresh(tmp, "id", world, seed_registry=404200040)
        set_arm(ag, "v37")
        teach_pairs(ag, world, pairs, tag="id")
        from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue

        probe = clone_frozen(ag)
        observe_cue(probe, world, tag="id_m", body=list(MID_BODY), symbols=[pairs[0][0]])
        p1 = probe._last_p1
        scores_m, addr_m, meta_m = probe.actuator_decision_scores(p1)
        scores_c, addr_c, meta_c = decision_scores(probe, p1)
        assert arr_sha(addr_m) == arr_sha(addr_c)
        assert {k: float(v) for k, v in scores_m.items()} == {k: float(v) for k, v in scores_c.items()}
        assert meta_m.get("path") == meta_c.get("path")
        assert meta_m.get("slot") == meta_c.get("slot")
        live = canonical_cue_probe(ag, world, pairs[0][0], tag="id_m", want=pairs[0][1])
        assert live["scoring_address_hash"] == arr_sha(addr_m)
        assert live["scores"] == {k: float(v) for k, v in scores_m.items()}
        src_motor = NEURAL.read_text()
        assert "self.actuator_decision_scores(addr)" in src_motor
        assert "act_scores, _score_addr, _recall_meta = self.actuator_decision_scores(addr)" in src_motor


def test_counterexample_stored_ranks_raw_live_does_not():
    import tempfile

    from experiments.canonical_act_probe import canonical_cue_probe, diagnostic_raw_live_scores
    from experiments.run_tm024writegeom import capacity_world, mapping_pairs
    from experiments.run_tm027gatedrehearsal import teach_pairs
    from experiments.run_tm040causal import _fresh, _seq, set_arm

    world = capacity_world(0, "TM040.CAUSAL.DEV.", n_cues=8, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory(prefix="r2_cx_") as tmp:
        ag = _fresh(tmp, "cx", world, seed_registry=404000039)
        set_arm(ag, "fallback_joint")
        teach_pairs(ag, world, _seq(pairs, "A_then_B"), tag="cx")
        live = canonical_cue_probe(ag, world, "s_294555646", tag="cx_p", want="h_812030613")
        raw = live["raw_live_diagnostic"]
        assert live["retrieval_path"] == "episodic_completed"
        assert int(live["retrieved_slot"]) == 2
        assert live["winner"] == "h_812030613"
        assert raw["winner"] == "h_679764572"
        assert live["scoring_address_hash"] != live["live_p1_hash"]
        raw2 = diagnostic_raw_live_scores(ag, live["live_p1"])
        assert raw2["winner"] == "h_679764572"
        assert raw2["not_a_behavioral_gate"] is True


def test_perturb_modifies_live_p1_then_canonical():
    import tempfile

    from experiments.canonical_act_probe import canonical_cue_probe, perturb_live_p1_then_canonical
    from experiments.run_tm024writegeom import capacity_world, mapping_pairs
    from experiments.run_tm027gatedrehearsal import teach_pairs
    from experiments.run_tm040causal import _fresh, set_arm

    world = capacity_world(0, "TM040.CAUSAL.R2.DEV.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    with tempfile.TemporaryDirectory(prefix="r2_pt_") as tmp:
        ag = _fresh(tmp, "pt", world, seed_registry=404200040)
        set_arm(ag, "v37")
        teach_pairs(ag, world, pairs, tag="pt")
        live = canonical_cue_probe(ag, world, pairs[0][0], tag="pt_p", want=pairs[0][1])
        stab = perturb_live_p1_then_canonical(
            ag,
            live["live_p1"],
            live["key_rho"],
            pairs[0][1],
            domain=str(world["domain"]),
            key="pt",
            sigma=0.01,
            n=5,
            need=1,
        )
        assert stab["perturbed"] == "live_p1"
        assert stab["then"] == "actuator_decision_scores"
        assert stab["trials"]
        assert stab["trials"][0]["live_p1_hash"] != live["live_p1_hash"]
        assert "retrieval_path" in stab["trials"][0]
        assert "scoring_address_hash" in stab["trials"][0]


def test_ids_and_smoke():
    from experiments.run_tm040causal_r2 import expected_cell_ids, smoke

    ids = expected_cell_ids()
    assert len(ids) == 102
    assert ids[0] == "acquire|c8|A_then_B|w0|v37"
    assert ids[-1] == "contradict|w1|always_joint"
    out = smoke()
    assert out["smoke_ok"]
    assert out["raw_score_leak"] == []
    assert out["probe_identity"]
    row = out["probe_identity"][0]
    assert row["live_p1_hash"]
    assert row["scoring_address_hash"]
    assert "retrieval_path" in row
    assert "retrieved_slot" in row
    src = RUNNER.read_text()
    assert "set_act_proj_arm" not in src
    assert ".actuator_scores(" not in src


def test_dev_lock_canonical_acquire_pass_and_no_candidate():
    devp = REPO / "docs" / "lineage_causalbattery.r2.dev.lock"
    decp = REPO / "docs" / "lineage_causalbattery.r2.decision.lock"
    assert _sha(devp) == "a13838622a76fb3b7f62a73ef3e58001db0a4bf99cb9ede9c575bd7f7c438ab3"
    assert _sha(decp) == "bcd40fba96ff96d90958aaf4c03fd4bb8fa2995dccd313600a09e9fc50124f23"
    assert not (REPO / "docs" / "cortex.candidate.v40.lock").exists()
    assert _sha(NEURAL) == FROZEN_NEURAL_SHA
    assert _sha(SOLVER) == JOINT_SOCP_SHA
    assert _sha(TM040_DEC) == TM040_DEC_SHA
    assert _sha(TM040_DEV) == TM040_DEV_SHA
    dev = json.loads(devp.read_text())
    dec = json.loads(decp.read_text())
    assert dev["clean_tree"] is True
    assert dev["git_head"] == "220d53d76be22cf96058eaa0390212bfa84ab0c4"
    assert dev["decision_code"] == "canonical_r2_later_learning_not_exercised"
    assert dev["historical_decision_code"] == "jointsocp_fallback_acquire_fail"
    assert dev["interpretation"] == "invalidated_measurement__canonical_path_mismatch"
    assert dev["architectural_conclusion"] is None
    assert dev["candidate_v40_lock"] is False
    assert dec["candidate_v40_lock"] is False
    flags = dev["phase_flags"]
    assert flags["fallback_acquire"] is True
    assert flags["fallback_eco"] is True
    assert flags["fallback_spec"] is True
    assert flags["fallback_contradict"] is True
    assert int(flags["n_later_after_socp"]) == 0
    acq = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|fallback_joint"][0]
    assert acq["passed"] is True
    assert int(acq["n_store_violations"]) == 0
    assert int(acq["n_probe_correct"]) == 8
    row = acq["probes"][0]
    assert row["live_p1_hash"]
    assert row["scoring_address_hash"]
    assert row["retrieval_path"]
    assert "retrieved_slot" in row
    v37 = [c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0|v37"][0]
    assert v37["passed"] is True
    assert int(v37["n_probe_correct"]) == 8
