"""TM.0.16.LIFEWALL: continuous-lifetime integration wall regression."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016lifewall import (
    AGENT_PY,
    CLAIM,
    FIXTURE_JSON,
    LIFE_WALL_LOCK,
    PERSIST_LOCK,
    PREREG_LOCK,
    run_lifewall,
    verify_life_wall_lock,
    verify_prereg_lock,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg_and_fixture_pins() -> None:
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["lab"] == "TM.0.16.LIFEWALL"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["preregistered_claim"] == CLAIM
    assert lock["organism"]["agent_edits_permitted"] is False
    assert lock["organism"]["factory"].endswith("make_persist")
    assert lock["fixture_sha"] == sha(FIXTURE_JSON)
    persist = json.loads(PERSIST_LOCK.read_text(encoding="utf-8"))
    # Historical: LIFEWALL froze the PERSIST-era agent; later labs may extend agent.py.
    assert lock["frozen_agent_sha"] == persist["agent_sha"]
    assert lock["frozen_agent_sha"] != ""
    assert "agent_sha" not in lock
    assert "run_tm016lifewall_sha" not in lock


def test_fixture_disjoint_and_alias_complete() -> None:
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    reserved = set(fix["reserved_alias"])
    obj_toks = set()
    for o in fix["objects"]:
        for k in ("P", "dest", "Q", "R", "foil", "dist"):
            assert o[k] not in reserved, o[k]
            assert o[k] not in obj_toks, o[k]
            obj_toks.add(o[k])
    assert len(fix["objects"]) == 32
    alias = fix["alias_lane"]
    assert alias["query_cue"] == "kelm"
    assert alias["query_target"] == "wift"
    assert len(alias["kill_table"]) == 3
    assert len(alias["kill_schedule"]) == 3
    # complete A2 schedule present in both scripts
    for script_name in ("script_main", "script_twin"):
        fps = [op for op in fix[script_name] if op["op"] == "alias_probe"]
        assert len(fps) == 18  # 9 aliases × 2 contexts
        assert any(op["op"] == "checkpoint" and op["source"] == "fingerprint" for op in fix[script_name])


def test_expected_state_machine() -> None:
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    by = {op["id"]: op for op in fix["script_main"] if op["op"] == "checkpoint"}
    assert by["K_cont_r4b_3"]["expect"] == "HOLD"
    assert by["K_cont_r8b_3"]["expect"] == "lorb"
    assert by["K_skel_r8c_0"]["expect"] == "HOLD"
    assert by["K_cont_r8c_0"]["expect"] == "HOLD"
    assert by["K_cont_r16c_1"]["expect"] == "HOLD"
    assert by["K_cont_r16c_2"]["expect"] == "HOLD"
    assert by["K_skel_r16c_1"]["expect"] == "daft"
    assert by["K_cont_r32a_31"]["source"] == "continuity"
    twin = {op["id"]: op for op in fix["script_twin"] if op["op"] == "checkpoint"}
    # Owner 0 withdrawn → matching read Q[vp[0]] HOLD (checkpoint id still *_0)
    assert twin["K_cont_r8c_0"]["expect"] == "HOLD"
    assert twin["K_cont_r8c_0"]["object_index"] == 5  # vp[0]
    # Delayed object-3 earn on twin is via Q[vp[3]], not renamed Q_3
    assert twin["K_cont_r8b_3"]["expect"] != "HOLD"
    assert twin["K_cont_r8b_3"]["object_index"] == 8  # vp[3]
    assert twin["K_alias_r4e"]["expect"] != "wift"  # renamed


def test_twin_verify_perm_coverage() -> None:
    """Every main continuity earn must have a twin probe on rename(Q[vp[i]])→rename(dest_i)."""
    fix = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    objs = fix["objects"]
    orename = fix["twin"]["object_rename"]
    vp = {int(k): int(v) for k, v in fix["twin"]["verify_perm"].items()}
    twin_by = {
        (op["rung"], op["phase"], op["cue"]): op
        for op in fix["script_twin"]
        if op["op"] == "checkpoint" and op["id"].startswith("K_cont_")
    }
    for op in fix["script_main"]:
        if op["op"] != "checkpoint" or not op["id"].startswith("K_cont_"):
            continue
        if op["expect"] == "HOLD":
            continue
        i = op["object_index"]
        cue = orename[objs[vp[i]]["Q"]]
        expect = orename[objs[i]["dest"]]
        top = twin_by.get((op["rung"], op["phase"], cue))
        assert top is not None, f"missing twin earn for {op['id']}"
        assert top["expect"] == expect
        assert top["source"] == "continuity"


def test_lifewall_battery_and_freeze() -> None:
    assert LIFE_WALL_LOCK.exists(), "life_wall.lock must exist after scored run"
    summary = run_lifewall(seed=12345, write_lock=False)
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["fixture_sha"] == sha(FIXTURE_JSON)
    # Wall outcomes must match freeze; agent_sha on freeze is historical (PERSIST-era).
    ok, why, freeze = verify_life_wall_lock(summary)
    assert ok, why
    assert freeze["last_ok_rung"] == summary["last_ok_rung"]
    assert freeze["first_fail_rung"] == summary["first_fail_rung"]
    assert freeze["main_ok"] == summary["main"]["ok"]
    assert freeze["twin_ok"] == summary["twin"]["ok"]
    # This recorded run cleared through 32
    assert summary["ok"] is True
    assert summary["last_ok_rung"] == 32
    assert summary["first_fail"] is None


if __name__ == "__main__":
    test_prereg_and_fixture_pins()
    test_fixture_disjoint_and_alias_complete()
    test_expected_state_machine()
    test_twin_verify_perm_coverage()
    test_lifewall_battery_and_freeze()
    print("ok")
