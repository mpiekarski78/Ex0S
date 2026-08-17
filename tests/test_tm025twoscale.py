"""TM.0.25.TWOSCALE / v32 provenance. AFFINEMAP R2 preserved. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
R2_DEC = "dc71c767858cc3fbc982a8fc6a3736c15d33f107087878810942ec4182b323bf"
R2_DEV = "3ca78f2ec5a383ee38ea644ef476f8308f302b75d59becfe34b861ca0fa761ef"
R2_ADD = "d6a2cdbf0869ff7a78b6c2fe2f1d20b6e31011c1333b464f006cfc34b45b7bc4"
V32_PREREG = "110705ba8ee005ed13816dcb592cdf2fc5d9903cb6a3ec91f18a36cd743b77d6"
V32_ISO = "09ae0d9680d62d2a4415205342db421356086d0d17acb849eed5fbcbc323baf2"
V32_AMEND = "6304cb702f3e043d1deb491ec08437e53e212a803ab5ab3045cb87030d201de0"
TWOSCALE_PREREG = "49151e3e4ec1673beb3c8fa2a2210201a772516221c6d032e162eb311c964ff1"
TWOSCALE_ISO = "9243750a1ba2d9652251a71b4d4f9cb04cd2b5722c95f71dedf11b23ce51e01d"
AMEND_MD = "d9a42afa0541c0202181714abc8d853fb5f36279a223a516379e06abcf3b2795"
DEV_LOCK_SHA = "d118d6ee6de95cbc88f05dd051f3c399f2dd2d7c59d796d7e55e9cd27dec0eaf"
DECISION_SHA = "579462b2212cb79646f79b025101c98ea7ad7a0a501ab5827bf3caad27ff4de3"
ADDENDUM_SHA = "6fb1b0b133abcf64f4f55834d531d0ce87f07ae70e4eb5fdf82fc47f04772611"
R1_DEV_SHA = "43a100ed2a45636755e0c957fda188c216afc7246372bf40656a8bbaef29ecfc"
R2_TS_DEV_SHA = "b202cdb23fdb56d406e5bf2bdf2d49ae999b2962451aa9da7d58d3ebaeab28b7"
EXPECTED_N_CELLS = 36
ERRATUM_SHA = "35d9de54ffef1260bc9ea1151e808e7ce1821a38107a42be58e2e528a1c83395"
ERRATUM_MD_SHA = "f42be4b00964b91cd98ebec42fd3a7f0653a9a90a6f95a9e405ef32fe9320bcc"
COMPAT_RUNNER_SHA = "780718e4a6b5c2bc6ef676ede80faf322ac50f4c526e636d870a212977c6ba46"
COMPAT_SHA = "bf74c92e8eaaf115946c74b2e1b030ba61757141c39fb01e757ea6ad2fe9983b"
MEMORY = "fc3942efaffb8b18e891c545510aa4949b52c86c773c707036bbc6d162fe35d7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_files() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.lock") == R2_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.r2.dev.lock") == R2_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.addendum.lock") == R2_ADD
    assert sha(REPO_ROOT / "docs" / "cortex_v32.prereg.lock") == V32_PREREG
    assert sha(REPO_ROOT / "docs" / "cortex_v32.isolation.lock") == V32_ISO
    assert sha(REPO_ROOT / "docs" / "cortex_v32_architecture_amendment.lock") == V32_AMEND
    assert sha(REPO_ROOT / "docs" / "cortex_v32_architecture_amendment.md") == AMEND_MD
    assert sha(REPO_ROOT / "docs" / "lineage_twoscale.prereg.lock") == TWOSCALE_PREREG
    assert sha(REPO_ROOT / "docs" / "lineage_twoscale.isolation.lock") == TWOSCALE_ISO
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert sha(REPO_ROOT / "three_memory" / "cortex_memory.py") == MEMORY
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    r1 = REPO_ROOT / "docs" / "lineage_twoscale.r1.dev.lock"
    assert sha(r1) == R1_DEV_SHA
    r1d = json.loads(r1.read_text(encoding="utf-8"))
    assert r1d["phase_flags"]["acquire_4"] is True
    assert r1d["phase_flags"]["acquire_8"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_twoscale.r2.dev.lock") == R2_TS_DEV_SHA
    src = (REPO_ROOT / "three_memory" / "neural_cortex.py").read_text(encoding="utf-8")
    assert src.count("EPISODE_SLOTS = 8") == 1
    assert "_replay_episodes" in src
    assert "_replay_store_pass" in src
    credit = src.split("def _apply_credit")[1].split("def _clip_and_consolidate")[0]
    assert "_replay_store_pass" not in credit
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v32.lock").exists()
    prereg = json.loads((REPO_ROOT / "docs" / "cortex_v32.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["authorized_law"] == "fast_episodic_p1_memory_plus_slow_cortical_consolidation"
    assert prereg["n"] == 64
    assert prereg["episode_slots"] == 8
    assert prereg["act_score_mode"] == "query"
    assert prereg["a3_bias"] is False
    cells = json.loads((REPO_ROOT / "docs" / "lineage_twoscale.prereg.lock").read_text(encoding="utf-8"))
    assert cells["expected_n_cells"] == EXPECTED_N_CELLS
    assert cells["domains"]["DEV"] == "TM025.TWOSCALE.DEV."
    assert cells["domains"]["SCORE"] == "TM025.TWOSCALE.SCORE."
    assert cells["neural_edit_authorized"] is True
    err = json.loads((REPO_ROOT / "docs" / "cortex_v32.erratum.lock").read_text(encoding="utf-8"))
    assert sha(REPO_ROOT / "docs" / "cortex_v32.erratum.lock") == ERRATUM_SHA
    assert sha(REPO_ROOT / "docs" / "cortex_v32.erratum.md") == ERRATUM_MD_SHA
    assert err["historical_code"] == "architectural_wall_acquire"
    assert err["preserve_missing_p1_fallback"] is True
    assert err["r3_opened"] is False
    assert err["candidate_v32_lock_written"] is False
    assert "competitive_heterosynaptic_rival_depression" in err["refuse"]
    assert "remove_missing_p1_fallback" in err["refuse"]
    assert "execute_replay_before_compat_runner_freeze" in err["refuse"]
    assert err["next_cycle"]["authorized_this_package"] is False
    src = (REPO_ROOT / "three_memory" / "neural_cortex.py").read_text(encoding="utf-8")
    assert "TIE_EPS" in src
    credit = src.split("def _apply_credit")[1].split("def _clip_and_consolidate")[0]
    assert "rho_p1" in credit
    assert "elig_motor" in credit



def test_cell_ids() -> None:
    from experiments.run_tm025twoscale import expected_cell_ids

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert "acquire|c8|A_then_B|w0" in ids
    assert "stable|c4|B_then_A|w1" in ids
    assert "eco|A_then_B|w0" in ids
    assert "spec|B_then_A|w1" in ids
    assert all(not i.startswith("SCORE") for i in ids)


def test_smoke() -> None:
    from experiments.run_tm025twoscale import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert out["n_episodes"] >= 1
    assert out["ranking_ok"] is True
    assert out["winner"] == out["want"]
    assert out["n_replay"] >= 16
    assert out["neural_edit"] is True
    assert out["v31_exists"] is False
    assert out["v32_candidate_exists"] is False
    assert out["tau"] == 1.0
    assert out["n"] == 64
    assert out["act_score_mode"] == "query"


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm025twoscale import DEV_LOCK, refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    if DEV_LOCK.exists():
        try:
            refuse_dev_lock()
        except RuntimeError as e:
            assert "again" in str(e)
        else:
            raise AssertionError("same frozen DEV execution must be refused")
        return
    refuse_dev_lock()


def test_decision_ladder() -> None:
    from experiments.run_tm025twoscale import _decision, load_prereg

    p = load_prereg()

    def cell(kind: str, n: int, ok: bool) -> dict[str, object]:
        return {"kind": kind, "n_cues": n, "passed": ok}

    cells = []
    for kind, ns in (("acquire", (2, 4, 8)), ("stable", (2, 4, 8)), ("twin", (2,)), ("eco", (2,)), ("spec", (2,))):
        for n in ns:
            nrep = 4 if kind in ("acquire", "stable") else 4
            for _i in range(nrep):
                cells.append(cell(kind, n, True))
    code, then, flags = _decision(cells, p)
    assert flags["acquire_8"] is True
    assert flags["acquire_2"] is True
    assert flags["stable_2"] is True
    assert code == "two_timescale_battery_pass"
    assert then == "reopen_lineage_readiness"
    for c in cells:
        if c["kind"] == "acquire" and int(c["n_cues"]) == 2:
            c["passed"] = False
    assert _decision(cells, p)[0] == "architectural_wall_acquire"
    for c in cells:
        if c["kind"] == "acquire":
            c["passed"] = True
        if c["kind"] == "stable" and int(c["n_cues"]) == 2:
            c["passed"] = False
    assert _decision(cells, p)[0] == "architectural_wall_stability"
    for c in cells:
        c["passed"] = True
        if c["kind"] == "acquire" and int(c["n_cues"]) == 4:
            c["passed"] = False
    assert _decision(cells, p)[0] == "architectural_wall_acquire"


def test_dev_opened() -> None:
    p = REPO_ROOT / "docs" / "lineage_twoscale.dev.lock"
    d = REPO_ROOT / "docs" / "lineage_twoscale.decision.lock"
    assert sha(p) == DEV_LOCK_SHA
    assert sha(d) == DECISION_SHA
    dev = json.loads(p.read_text(encoding="utf-8"))
    dec = json.loads(d.read_text(encoding="utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len({c["id"] for c in dev["cells"]}) == EXPECTED_N_CELLS
    assert dev["decision_code"] == "architectural_wall_acquire"
    assert dev["phase_flags"]["acquire_4"] is True
    assert dev["phase_flags"]["acquire_8"] is False
    assert dev["phase_flags"]["stable_4"] is True
    assert dev["phase_flags"]["eco"] is True
    assert dev["phase_flags"]["spec"] is True
    assert dev["n"] == 64
    assert "TM025.TWOSCALE.SCORE." not in json.dumps(dev)
    assert dec["decision"]["code"] == "architectural_wall_acquire"
    assert dec["lineage_reopened"] is False
    assert dec["dev_lock_sha"] == DEV_LOCK_SHA
    assert dec["earned_next"] is False


def test_addendum() -> None:
    p = REPO_ROOT / "docs" / "lineage_twoscale.decision.addendum.lock"
    assert sha(p) == ADDENDUM_SHA
    add = json.loads(p.read_text(encoding="utf-8"))
    assert add["historical_code"] == "architectural_wall_acquire"
    assert add["acquire_4"] is True
    assert add["acquire_8"] is False
    assert add["eco"] is True
    assert add["spec"] is True
    assert add["eight_cue_acquire_probes_correct"] == "7/8"
    assert add["escalation_cleared_eight_cue"] is False
    assert add["lineage_reopened"] is False
    assert add["historical_dev_lock_sha"] == DEV_LOCK_SHA
    assert add["r1_dev_lock_sha"] == R1_DEV_SHA
    assert add["candidate_v32_lock_written"] is False


def test_twoscale_historical_boundary_immutable() -> None:
    """Historical TWOSCALE 7/8 stays pinned; live v33 owns new behavior via TM026."""
    dev = json.loads((REPO_ROOT / "docs" / "lineage_twoscale.dev.lock").read_text(encoding="utf-8"))
    dec = json.loads((REPO_ROOT / "docs" / "lineage_twoscale.decision.lock").read_text(encoding="utf-8"))
    assert dev["decision_code"] == "architectural_wall_acquire"
    assert dec["decision"]["code"] == "architectural_wall_acquire"
    rec = next(c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0")
    n_ok = sum(1 for p in rec["probes"] if p["ranking_ok"])
    assert n_ok == 7
    assert rec["passed"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_twoscale.dev.lock") == DEV_LOCK_SHA
    assert (REPO_ROOT / "docs" / "lineage_competitive.prereg.lock").exists()

def _pending(
    ag,
    handle: str,
    *,
    rho_p1,
    rho_motor,
):
    import numpy as np

    n = int(ag.genome.n)
    z = np.zeros(n, dtype=np.float64)
    motor = z if rho_motor is None else np.asarray(rho_motor, dtype=np.float64)
    return {
        "op": "ACT",
        "token": handle,
        "rho_elig": motor.copy(),
        "rho_op": motor.copy(),
        "rho_motor": motor.copy(),
        "rho_p1": None if rho_p1 is None else np.asarray(rho_p1, dtype=np.float64).copy(),
        "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
        "body": np.zeros(4, dtype=np.float64),
        "cost": 0.0,
        "motor_vec": ag.motor_vocab[handle].copy(),
        "authored": True,
        "clamped": True,
        "t": 0,
        "interaction_token": "erratum",
    }


def _fresh_bound():
    from experiments.run_tm023cortex import make_cortex

    ag = make_cortex(None, device="cpu")
    ag.bind_actuators(["h_a", "h_b"])
    return ag


def test_tie_band_unique_winner() -> None:
    import numpy as np
    from three_memory.neural_cortex import TIE_EPS

    ag = _fresh_bound()
    scores = {"h_a": 0.1, "h_b": 0.1 + TIE_EPS * 0.5}
    assert ag._unique_act_winner(scores) is None
    orig = ag.actuator_scores
    ag.actuator_scores = lambda rho: scores  # type: ignore[method-assign]
    pick = ag._choose_actuator(np.zeros(ag.genome.n))
    ag.actuator_scores = orig
    assert pick in ("h_a", "h_b")


def test_p1_credit_three_cases() -> None:
    import numpy as np
    from three_memory.neural_cortex import BODY_SETPOINT, ELIG_EPS

    n_ok = 0
    p1 = np.zeros(64, dtype=np.float64)
    p1[0] = 1.0
    motor = np.zeros(64, dtype=np.float64)
    motor[1] = 1.0
    z = np.zeros(64, dtype=np.float64)

    ag = _fresh_bound()
    h = "h_a"
    w0 = float(ag.W_act_query.abs().max().item())
    ag._pending = _pending(ag, h, rho_p1=p1, rho_motor=z)
    ag._apply_credit(np.zeros(ag.genome.d_sym), BODY_SETPOINT)
    assert float(ag.W_act_query.abs().max().item()) > w0 + ELIG_EPS
    assert len(ag._episodes) >= 1
    n_ok += 1

    ag = _fresh_bound()
    ag._pending = _pending(ag, h, rho_p1=z, rho_motor=motor)
    ag._apply_credit(np.zeros(ag.genome.d_sym), BODY_SETPOINT)
    assert float(ag.W_act_query.abs().max().item()) <= ELIG_EPS
    assert len(ag._episodes) == 0
    n_ok += 1

    ag = _fresh_bound()
    ag._pending = _pending(ag, h, rho_p1=None, rho_motor=motor)
    ag._apply_credit(np.zeros(ag.genome.d_sym), BODY_SETPOINT)
    assert float(ag.W_act_query.abs().max().item()) > ELIG_EPS
    assert len(ag._episodes) >= 1
    n_ok += 1
    assert n_ok == 3


def test_checkpoint_dev_epoch_defaults_and_age_scale() -> None:
    import numpy as np
    from three_memory.neural_cortex import GenomeConfig, NeuralCortex

    lp = {
        "age.birth.eta_act_scale": 1.0,
        "age.high_plasticity.eta_act_scale": 0.25,
    }
    g = GenomeConfig(lineage_params=dict(lp))
    ag = NeuralCortex(None, genome=g, device="cpu")
    ag.bind_actuators(["h_a", "h_b"])
    ag.rest_epoch(1)
    assert ag.dev_epoch == 1
    assert ag._age_scale("eta_act_scale") == 0.25
    snap = ag.checkpoint()
    twin_g = GenomeConfig(lineage_params=dict(lp))
    twin = NeuralCortex(None, genome=twin_g, device="cpu")
    twin.bind_actuators(["h_a", "h_b"])
    assert twin.dev_epoch == 0
    assert twin._age_scale("eta_act_scale") == 1.0
    twin.load_checkpoint(snap)
    assert twin.dev_epoch == 1
    assert twin._age_scale("eta_act_scale") == 0.25
    p1 = np.zeros(ag.genome.n, dtype=np.float64)
    p1[0] = 1.0
    eta_live = float(ag.genome.eta_act) * ag._age_scale("eta_act_scale")
    eta_twin = float(twin.genome.eta_act) * twin._age_scale("eta_act_scale")
    assert abs(eta_live - eta_twin) < 1e-12
    w0 = twin.W_act_query.detach().clone()
    twin._apply_act_query_update(p1, "h_a", 1.0, eta_twin, mix_slow=False)
    live_w0 = ag.W_act_query.detach().clone()
    ag._apply_act_query_update(p1, "h_a", 1.0, eta_live, mix_slow=False)
    assert float((ag.W_act_query - live_w0).abs().max().item()) > 0.0
    assert abs(
        float((ag.W_act_query - live_w0).abs().max().item())
        - float((twin.W_act_query - w0).abs().max().item())
    ) < 1e-12

    missing = ag.checkpoint()
    for k in ("dev_epoch", "episode_n_inserts", "episode_n_replaced", "n_rest_replay", "n_rest_strengthen"):
        missing.pop(k, None)
    blank = NeuralCortex(None, genome=GenomeConfig(lineage_params=dict(lp)), device="cpu")
    blank.bind_actuators(["h_a", "h_b"])
    blank.dev_epoch = 7
    blank.load_checkpoint(missing)
    assert blank.dev_epoch == 0
    assert blank._episode_n_inserts == 0
    assert blank._episode_n_replaced == 0
    assert blank._n_rest_replay == 0
    assert blank._n_rest_strengthen == 0


def test_compat_runner_schema_and_gate() -> None:
    from experiments.run_tm025twoscale import (
        COMPAT_LOCK,
        COMPAT_RUNNER,
        MISMATCH_LOCK,
        comparison_schema,
    )

    p = REPO_ROOT / "docs" / "lineage_twoscale.compat.runner.lock"
    if not p.exists():
        raise AssertionError("compat runner lock must be frozen before replay")
    if COMPAT_RUNNER_SHA:
        assert sha(p) == COMPAT_RUNNER_SHA
    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["historical_dev_lock_sha"] == DEV_LOCK_SHA
    assert lock["expected_n_cells"] == EXPECTED_N_CELLS
    assert lock["n_unique_cell_ids"] == EXPECTED_N_CELLS
    assert len(lock["expected_cell_ids"]) == EXPECTED_N_CELLS
    assert len(set(lock["expected_cell_ids"])) == EXPECTED_N_CELLS
    assert lock["rewrite_historical_dev"] is False
    assert lock["replay_writes_dev"] is False
    assert lock["competitive_plasticity_authorized"] is False
    schema = lock["comparison_schema"]
    assert schema["mode"] == "recursive_semantic_payload"
    assert schema["float_atol"] == 1e-12
    assert "git_head" in schema["exclude_provenance_prefixes"]
    assert "shas" in schema["exclude_provenance_prefixes"]
    assert comparison_schema()["exclude_provenance_prefixes"] == schema["exclude_provenance_prefixes"]
    assert COMPAT_LOCK.exists()
    assert not MISMATCH_LOCK.exists()
    assert sha(COMPAT_LOCK) == COMPAT_SHA
    compat = json.loads(COMPAT_LOCK.read_text(encoding="utf-8"))
    assert compat["compatible"] is True
    assert compat["changed_fields"] == []
    assert compat["max_abs_float_delta"] == 0.0
    assert compat["exact_boolean_string_equality"] is True
    assert compat["n_expected_cells"] == EXPECTED_N_CELLS
    assert compat["n_unique_cell_ids"] == EXPECTED_N_CELLS
    assert compat["decision_code"] == "architectural_wall_acquire"
    assert compat["rewrite_historical_dev"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_twoscale.dev.lock") == DEV_LOCK_SHA


def test_semantic_compare_excludes_provenance() -> None:
    from experiments.run_tm025twoscale import compare_semantic_payload, expected_cell_ids

    ids = expected_cell_ids()
    cells = [{"id": i, "passed": True, "n_cues": 2} for i in ids]
    hist = {
        "decision_code": "architectural_wall_acquire",
        "git_head": "aaa",
        "shas": {"runner": "old", "neural_cortex": "oldn"},
        "env": {"python": "3"},
        "cells": cells,
        "phase_flags": {"acquire_8": False},
    }
    live = {
        "decision_code": "architectural_wall_acquire",
        "git_head": "bbb",
        "shas": {"runner": "new", "neural_cortex": "newn"},
        "env": {"python": "9"},
        "cells": cells,
        "phase_flags": {"acquire_8": False},
    }
    out = compare_semantic_payload(hist, live)
    assert out["compatible"] is True
    assert out["changed_fields"] == []
    live2 = json.loads(json.dumps(live))
    live2["phase_flags"]["acquire_8"] = True
    bad = compare_semantic_payload(hist, live2)
    assert bad["compatible"] is False
    assert bad["changed_fields"]
    assert bad["exact_boolean_string_equality"] is False


if __name__ == "__main__":
    test_freeze_files()
    test_cell_ids()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_ladder()
    test_dev_opened()
    test_addendum()
    test_scored_eight_cue_cell_reproduces()
    test_tie_band_unique_winner()
    test_p1_credit_three_cases()
    test_checkpoint_dev_epoch_defaults_and_age_scale()
    test_compat_runner_schema_and_gate()
    test_semantic_compare_excludes_provenance()
    print("ok")
