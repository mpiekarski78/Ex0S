"""TM.0.5.6: never-wipe train — dirty S survives training."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import wiki_prose
from experiments.run_tm055 import _accumulate_lives
from experiments.run_tm056 import _c_life_on_s, _copy_s, _probe_s, _s_snapshot, _train_keep
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def _forced_policy() -> UsePolicy:
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_write = np.array([3.0, 3.0], dtype=np.float64)
    policy.b_write = np.array(3.0, dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    return policy


def test_train_keep_does_not_wipe(tmp_path: Path):
    w = tmp_path / "W"
    work = tmp_path / "work"
    write_prose_notes(w, wiki_prose(include_a=True))
    policy = _forced_policy()
    _train_keep(policy, w, work, n=3, seed=3, split=True, max_steps=40)
    s = work / "ep"
    snap = _s_snapshot(s)
    assert snap["n"] >= 1
    assert snap["found_press"] and snap["found_cha"]
    p_a, p_c = _probe_s(s, policy, 3)
    assert p_a["action_name"] == "press"
    assert p_c["action_name"] == "hold"


def test_c_life_on_dirty_train_s(tmp_path: Path):
    w_a = tmp_path / "Wa"
    w_both = tmp_path / "Wb"
    work = tmp_path / "work"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    policy = _forced_policy()
    _train_keep(policy, w_a, work, n=2, seed=3, split=True, max_steps=40)
    acc = tmp_path / "acc"
    _copy_s(work / "ep", acc)
    before = _s_snapshot(acc)
    assert before["found_press"] and before["found_cha"]
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


def test_wipe_loses_train_s(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    _, _, _, _, _, _, wipe_a, wipe_c, _ = _accumulate_lives(
        s,
        w,
        _forced_policy(),
        3,
        max_steps=40,
        explore_epsilon=1.0,
        rng=np.random.default_rng(0),
        wipe_between=True,
    )
    assert wipe_a["action_name"] != "press"
    assert wipe_c["action_name"] == "tune"


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm056.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert 'flags["train_wipe_s"] = False' in src_exp
    assert 'flags["accumulate_s"] = True' in src_exp
    assert 'flags["open_w"] = True' in src_exp
    assert "KeyDoorWorld" not in src_exp
    assert "Working motor was press" not in src_exp
    assert "def _train_keep" in src_exp
    body = src_exp.split("def _train_keep", 1)[1].split("def classify_common", 1)[0]
    assert "shutil.rmtree(s_dir)" not in body
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_here_match and not a.use_event_annotate
    assert not a.use_commit_rare_only


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src


if __name__ == "__main__":
    import tempfile

    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_train_keep_does_not_wipe,
        test_c_life_on_dirty_train_s,
        test_wipe_loses_train_s,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
