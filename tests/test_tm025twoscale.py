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


def test_scored_eight_cue_cell_reproduces() -> None:
    from experiments.run_tm024writegeom import capacity_world, mapping_pairs
    from experiments.run_tm025twoscale import eval_acquire_stable

    dev = json.loads((REPO_ROOT / "docs" / "lineage_twoscale.dev.lock").read_text(encoding="utf-8"))
    rec = next(c for c in dev["cells"] if c["id"] == "acquire|c8|A_then_B|w0")
    world = capacity_world(0, "TM025.TWOSCALE.DEV.", n_cues=8, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    live = eval_acquire_stable(
        kind="acquire",
        world=world,
        pairs=pairs,
        order="A_then_B",
        tag="audit_c8",
        rest=False,
    )
    assert live["passed"] is False
    assert live["n_episodes"] == 8
    n_ok = sum(1 for p in live["probes"] if p["ranking_ok"])
    assert n_ok == 7
    assert abs(float(live["min_normalized_geometric_margin"]) - float(rec["min_normalized_geometric_margin"])) < 1e-12


if __name__ == "__main__":
    test_freeze_files()
    test_cell_ids()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_ladder()
    test_dev_opened()
    test_addendum()
    test_scored_eight_cue_cell_reproduces()
    print("ok")
