"""TM.0.16.ALIASWALL: alias-equivalence wall on frozen RELATE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016aliaswall import (
    ALIAS_WALL_LOCK,
    CELL_IDS,
    KILL_ALIAS_TABLE,
    PREREG_LOCK,
    WALL_CLAIM,
    run_alias_wall,
    verify_alias_wall_lock,
    verify_prereg_lock,
)
from experiments.run_tm016relate import GENOME_016_LOCK, RELATE_LOCK, verify_genome_016


def test_prereg_intact():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["preregistered_claim"] == WALL_CLAIM
    assert lock["not_tm017"] is True
    assert lock["cell_ids"] == list(CELL_IDS)
    assert "run_tm016aliaswall_sha" not in lock
    assert "agent_sha" not in lock
    assert len(lock["kill_alias_table"]) == 3
    assert lock["kill_alias_table"][0]["origin"] == "kelm"


def test_alias_wall_battery_and_lock():
    summary = run_alias_wall(seed=12345, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 6
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["W0_control"]["lived_bind"] == "y"
    assert by["W1_kill"]["motor"] == "hold"
    assert by["W1_kill"]["cue"] == "kelm"
    assert by["W1_kill"]["cue"] != "x"
    assert by["W1_kill"]["lived_bind"] != "wift"
    assert all(v == 1 for v in by["W1_kill"]["supports"].values())
    assert by["W0_control"]["s_dir"] != by["W1_kill"]["s_dir"]
    assert by["W2_schedule_twin"]["canon_match"] is True
    assert by["W4_map_isolation"]["has_alias_visible"] is True
    assert by["W4_map_isolation"]["no_latent_in_s"] is True

    ok, why, _ = verify_genome_016()
    assert ok, why
    ok2, why2, _ = verify_alias_wall_lock(summary["rows"])
    assert ok2, why2
    lock = json.loads(ALIAS_WALL_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert lock["kill_alias_table"] == [dict(r) for r in KILL_ALIAS_TABLE]


def test_prior_016_pins():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert lock["genome_016_lock_sha"] == sha(GENOME_016_LOCK)
    assert lock["relate_016_lock_sha"] == sha(RELATE_LOCK)


if __name__ == "__main__":
    test_prereg_intact()
    test_alias_wall_battery_and_lock()
    test_prior_016_pins()
    print("ok")
