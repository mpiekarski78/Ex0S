"""Unit and checkpoint tests for the TM058 opaque-store write path.

Does not edit the frozen TM058 runner. Does not retune 0.05.
Product 0.0.004.
"""

from __future__ import annotations

import copy

import numpy as np

from three_memory.neural_cortex import (
    ACT_RECALL_MODES,
    EPISODE_MATCH_L2,
    GenomeConfig,
    NeuralCortex,
)
from three_memory.opaque_memory import OPAQUE_KV_SLOTS, OpaqueMemory


def _unit(n: int, i: int) -> np.ndarray:
    v = np.zeros(int(n), dtype=np.float64)
    v[int(i) % int(n)] = 1.0
    return v


def _ag() -> NeuralCortex:
    return NeuralCortex(None, genome=GenomeConfig(), device="cpu")


def test_flag_default_off_not_genome_not_recall_mode():
    ag = _ag()
    assert ag._opaque_store_enabled is False
    g = ag.genome.to_dict()
    assert "opaque_store_enabled" not in g
    assert "_opaque_store_enabled" not in g
    assert "opaque_store_enabled" not in ACT_RECALL_MODES
    assert EPISODE_MATCH_L2 == 0.05
    assert OPAQUE_KV_SLOTS == 8


def test_flag_off_rejects_and_leaves_historical_store_untouched():
    ag = _ag()
    n = int(ag.genome.n)
    before_ep = list(ag._episodes)
    rec = ag.write_opaque_kv(_unit(n, 0), _unit(n, 1), handle="h", provenance_id="runner")
    assert rec["outcome"] == "reject"
    assert rec["reason"] == "flag_off"
    assert ag.opaque.rows() == []
    assert ag._episodes == before_ep
    assert ag._opaque_kv_seq == 0


def test_flag_on_does_not_call_episode_write_or_store_runner_provenance():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    rec = ag.write_opaque_kv(_unit(n, 0), _unit(n, 1), handle="h0", provenance_id="runner_meta")
    assert rec["outcome"] == "append"
    assert rec["provenance_id"] == "1"
    rows = ag.opaque.rows()
    assert len(rows) == 1
    assert rows[0].provenance_id == "1"
    assert rows[0].when == 1
    assert ag._episodes == []
    assert rec["provenance_id"] != "runner_meta"


def test_copies_isolate_residents_from_later_mutation():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    key = _unit(n, 2)
    val = _unit(n, 3)
    ag.write_opaque_kv(key, val, handle="h", provenance_id="x")
    key[0] = 0.0
    val[0] = 0.0
    row = ag.opaque.rows()[0]
    assert float(row.key[2]) == 1.0
    assert float(row.value[3]) == 1.0


def test_invalid_write_is_atomic_and_does_not_advance_seq():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    ag.write_opaque_kv(_unit(n, 0), _unit(n, 1), handle="h", provenance_id="a")
    bad = np.array([np.nan, 1.0], dtype=np.float64)
    rec = ag.write_opaque_kv(bad, _unit(n, 1), handle="h", provenance_id="b")
    assert rec["outcome"] == "reject"
    assert rec["reason"] == "invalid_arrays"
    assert ag._opaque_kv_seq == 1
    assert len(ag.opaque.rows()) == 1


def test_distinct_events_coexist_including_identical_key_value():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    k = _unit(n, 4)
    v = _unit(n, 5)
    ag.write_opaque_kv(k, v, handle="h0", provenance_id="a")
    ag.write_opaque_kv(k, v, handle="h0", provenance_id="b")
    nearby = v.copy()
    nearby[6] = 0.01
    nearby = nearby / np.linalg.norm(nearby)
    ag.write_opaque_kv(_unit(n, 7), nearby, handle="h1", provenance_id="c")
    rows = ag.opaque.rows()
    assert len(rows) == 3
    assert [r.provenance_id for r in rows] == ["1", "2", "3"]


def test_capacity_evicts_oldest_when_then_appends():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    recs = []
    for i in range(OPAQUE_KV_SLOTS + 1):
        recs.append(ag.write_opaque_kv(_unit(n, i), _unit(n, i + 1), handle="h", provenance_id=f"r{i}"))
    assert recs[-1]["outcome"] == "evict_append"
    assert recs[-1]["evicted_provenance_id"] == "1"
    rows = ag.opaque.rows()
    assert len(rows) == 8
    assert [r.provenance_id for r in rows] == [str(i) for i in range(2, 10)]
    assert [r.when for r in rows] == list(range(2, 10))


def test_checkpoint_restores_flag_seq_and_rows_fail_closed():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    ag.write_opaque_kv(_unit(n, 0), _unit(n, 1), handle="h", provenance_id="x")
    snap = ag.checkpoint()
    assert snap["opaque_store_enabled"] is True
    assert int(snap["opaque_kv_seq"]) == 1
    twin = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device="cpu")
    twin.load_checkpoint(snap)
    assert twin._opaque_store_enabled is True
    assert twin._opaque_kv_seq == 1
    assert twin.opaque.rows()[0].provenance_id == "1"
    bare = {k: v for k, v in snap.items() if k not in ("opaque_store_enabled", "opaque_kv_seq")}
    closed = NeuralCortex(None, genome=copy.deepcopy(ag.genome), device="cpu")
    closed.load_checkpoint(bare)
    assert closed._opaque_store_enabled is False
    assert closed._opaque_kv_seq == 0


def test_historical_episode_write_unchanged_with_flag_on():
    ag = _ag()
    ag.set_opaque_store_enabled(True)
    n = int(ag.genome.n)
    a = _unit(n, 0)
    b = a.copy()
    b[1] = 0.01
    b = b / np.linalg.norm(b)
    assert float(np.linalg.norm(a - b)) <= EPISODE_MATCH_L2
    ag._episode_write(a, "h0", 1.0, key_rho=_unit(n, 8))
    ag._episode_write(b, "h1", 1.0, key_rho=_unit(n, 9))
    assert len(ag._episodes) == 1
    assert ag.opaque.rows() == []


def test_historical_opaque_write_still_key_replaces():
    mem = OpaqueMemory()
    n = 8
    k = _unit(n, 0)
    mem.write(k, _unit(n, 1), provenance_id="a")
    mem.write(k, _unit(n, 2), provenance_id="b")
    assert len(mem.rows()) == 1
    assert mem.rows()[0].provenance_id == "b"
