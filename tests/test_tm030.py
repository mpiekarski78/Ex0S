"""TM.0.3.0: free life find/commit/use; probe after ρ reset with W gone."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012 import _has_field
from experiments.run_tm030 import live_free, make, wiki_notes
from three_memory import agent as agent_mod
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_agent_has_no_where():
    src = inspect.getsource(agent_mod)
    assert '"where"' not in src and "'where'" not in src


def test_wiki_is_messy_not_d0():
    notes = wiki_notes(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    by = dict(notes)
    assert by["p99.tag"]["where"] == 0
    assert by["p99.tag"]["action"] == 2
    for _, tags in notes:
        assert "when" not in tags
        assert "loc" not in tags
        assert "here" not in tags


def test_make_is_life_not_write_author(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    assert a.write_from_events is False
    assert a.collect_mode == "commit"
    assert a.use_search_head is True
    assert a.record_search_on_explore is True
    assert a.force_use is False


def test_live_free_n_forced_zero(tmp_path: Path):
    w = tmp_path / "W"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(tmp_path / "S", w, UsePolicy(seed=7), explore_epsilon=0.5, rng=np.random.default_rng(0))
    live = live_free(a, "experience_teach", 1, max_steps=8)
    assert live["n_forced"] == 0
    assert live["n_steps"] >= 1


def test_record_search_on_explore_leaves_traces(tmp_path: Path):
    w = tmp_path / "W"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(
        tmp_path / "S",
        w,
        UsePolicy(seed=7),
        epsilon=0.5,
        explore_epsilon=0.5,
        rng=np.random.default_rng(1),
        record_search_on_explore=True,
    )
    a.policy_traces = []
    live_free(a, "experience_teach", 2, max_steps=6)
    kinds = {t.get("kind") for t in a.policy_traces}
    assert "search" in kinds
    assert "use" not in kinds  # use traces only on greedy probe


def test_prefer_rare_life_commits_p99(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_teach", 3, max_steps=4)
    assert "p99.tag" in live["files"]
    assert live["found_action2"]
    assert _has_field(live["tag"], "where")
    a.world = None
    a.reset_rho()
    action, _ = _probe(a, "probe_red_with_key")
    assert action == Action.USE_KEY


def test_untrained_life_does_not_solve_probe(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=0.5, rng=np.random.default_rng(9))
    live_free(a, "experience_teach", 4, max_steps=16)
    a.world = None
    a.reset_rho()
    action, _ = _probe(a, "probe_red_with_key")
    assert action != Action.USE_KEY


def test_clone_keeps_record_search_flag(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    b = a.clone_empty()
    assert b.record_search_on_explore is True
    assert b.use_search_head is True


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_where()
    test_wiki_is_messy_not_d0()
    fns = [
        test_make_is_life_not_write_author,
        test_live_free_n_forced_zero,
        test_record_search_on_explore_leaves_traces,
        test_prefer_rare_life_commits_p99,
        test_untrained_life_does_not_solve_probe,
        test_clone_keeps_record_search_flag,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
