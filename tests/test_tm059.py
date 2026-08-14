"""TM.0.5.9: correct the dirty store — stop appending, drop junk."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm058 import N_CLUTTER, wiki_prose
from experiments.run_tm059 import (
    MAX_TRAIN_S_FILES,
    _c_life_on_s,
    _copy_s,
    _probe_s,
    _s_snapshot,
    _train_keep,
    _w_flags,
    make,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def _forced_policy(*, revise: bool = True) -> UsePolicy:
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_write = np.array([3.0, 3.0], dtype=np.float64)
    policy.b_write = np.array(3.0, dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    if revise:
        policy.w_revise = np.array([3.0, -3.0], dtype=np.float64)
        policy.b_revise = np.array(-1.0, dtype=np.float64)
    return policy


def test_flags_require_correct_dirty(tmp_path: Path):
    w = tmp_path / "W"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    flags = _w_flags(sorted(p.name for p in w.glob("*.md")), w)
    assert flags["w_n_distinct_clutter"] == N_CLUTTER
    assert flags["correct_dirty_s"]
    assert flags["use_revise_head"]
    assert flags["use_commit_here_only"]
    assert flags["max_train_s_files"] == MAX_TRAIN_S_FILES == 8


def test_train_keep_press_and_cleans(tmp_path: Path):
    w = tmp_path / "W"
    work = tmp_path / "work"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = _forced_policy(revise=True)
    _train_keep(policy, w, work, n=4, seed=3, split=True, max_steps=40)
    s = work / "ep"
    snap = _s_snapshot(s)
    assert snap["found_press"] and snap["found_cha"]
    assert snap["n"] <= MAX_TRAIN_S_FILES
    extra = (work / "tm059_train.json").read_text(encoding="utf-8")
    assert '"n_revised"' in extra
    p_a, p_c = _probe_s(s, policy, 3)
    assert p_a["action_name"] == "press"
    assert p_c["action_name"] == "hold"


def test_revise_off_keeps_more_files(tmp_path: Path):
    w = tmp_path / "W"
    write_prose_notes(w, wiki_prose(include_a=True))
    on = tmp_path / "on"
    off = tmp_path / "off"
    _train_keep(_forced_policy(revise=True), w, on, n=4, seed=3, split=True, max_steps=40)
    # Same here-only make, but revise weights stay untrained-off.
    import experiments.run_tm059 as tm059

    saved = tm059.make

    def make_off(*args, **kw):
        kw["use_revise_head"] = False
        return saved(*args, **kw)

    tm059.make = make_off
    try:
        _train_keep(_forced_policy(revise=False), w, off, n=4, seed=3, split=True, max_steps=40)
    finally:
        tm059.make = saved
    n_on = _s_snapshot(on / "ep")["n"]
    n_off = _s_snapshot(off / "ep")["n"]
    assert n_on <= MAX_TRAIN_S_FILES
    assert n_off >= n_on


def test_c_life_on_corrected_s(tmp_path: Path):
    w_a = tmp_path / "Wa"
    w_both = tmp_path / "Wb"
    work = tmp_path / "work"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    policy = _forced_policy(revise=True)
    _train_keep(policy, w_a, work, n=3, seed=3, split=True, max_steps=40)
    acc = tmp_path / "acc"
    _copy_s(work / "ep", acc)
    _, c_live, both_a, both_c, _ = _c_life_on_s(
        acc,
        w_both,
        policy,
        6,
        max_steps=40,
        explore_epsilon=1.0,
        rng=np.random.default_rng(3),
    )
    assert c_live["found_tune"] and c_live["found_chc"]
    assert c_live["found_press"] and c_live["found_cha"]
    assert both_a["action_name"] == "press"
    assert both_c["action_name"] == "tune"


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm059.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "use_commit_here_only" in src_agent
    assert 'flags["correct_dirty_s"] = True' in src_exp
    assert "MAX_TRAIN_S_FILES = 8" in src_exp
    assert "KeyDoorWorld" not in src_exp
    assert "Working motor was press" not in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_here_match and not a.use_event_annotate
    assert not a.use_commit_rare_only
    assert not a.use_commit_here_only
    assert not a.use_revise_head
    b = make054(
        Path("/tmp/tm059_skip_s"),
        None,
        UsePolicy(seed=1),
        enabled=False,
    )
    assert not b.use_revise_head and not b.use_commit_here_only


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src


if __name__ == "__main__":
    import tempfile

    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_flags_require_correct_dirty,
        test_train_keep_press_and_cleans,
        test_revise_off_keeps_more_files,
        test_c_life_on_corrected_s,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
