"""Smoke tests for three-memory v0 pieces."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.store import FactRecord


def test_weights_frozen_after_life():
    a = ThreeMemoryAgent(store_enabled=True)
    h0 = a.weight_hash()
    world = KeyDoorWorld(0)
    obs = world.reset("experience_teach")
    for _ in range(5):
        act, _ = a.act(obs)
        r = world.step(act)
        a.observe_outcome(r.obs, r.success, r.info)
        obs = r.obs
    assert a.weight_hash() == h0


def test_rho_reset_clears_trace():
    a = ThreeMemoryAgent()
    world = KeyDoorWorld(1)
    obs = world.reset("experience_teach")
    for _ in range(4):
        act, _ = a.act(obs)
        r = world.step(act)
        a.observe_outcome(r.obs, r.success, r.info)
        obs = r.obs
    assert float(np.linalg.norm(a.rho.rho)) > 0
    a.reset_rho()
    assert float(np.linalg.norm(a.rho.rho)) == 0.0


def test_store_survives_rho_reset():
    a = ThreeMemoryAgent(store_enabled=True)
    a.store.write(
        FactRecord(
            fact_id=KeyDoorWorld.FACT_ID,
            what=KeyDoorWorld.FACT_TEXT,
            when=1,
            drive_scores={"novelty": 0.9, "integrity": 1.0},
            tags={"door": "red", "requires": "key"},
        )
    )
    a.rho.update(np.ones(32))
    a.reset_rho()
    assert a.store.has_fact(KeyDoorWorld.FACT_ID)
    assert float(np.linalg.norm(a.rho.rho)) == 0.0


def test_store_disabled_blocks_writes():
    a = ThreeMemoryAgent(store_enabled=False)
    ok = a.store.write(
        FactRecord(
            fact_id="x",
            what="y",
            when=0,
            drive_scores={},
            tags={},
        )
    )
    assert ok is False
    assert len(a.store) == 0


if __name__ == "__main__":
    test_weights_frozen_after_life()
    test_rho_reset_clears_trace()
    test_store_survives_rho_reset()
    test_store_disabled_blocks_writes()
    print("ok")
