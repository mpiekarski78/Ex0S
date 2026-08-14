"""TM.0.1.2: messy retrieve without exact loc=/door= query."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012 import WIKI_RED, make, wiki_notes
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


def test_wiki_notes_are_messy():
    notes = wiki_notes(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    by = dict(notes)
    assert by["p99.tag"]["where"] == 0
    assert by["p99.tag"]["action"] == 2
    assert by["p99.tag"]["pad"] == 7
    assert "loc" not in by["p99.tag"]
    assert "door" not in by["p99.tag"]
    assert "here" not in by["p99.tag"]
    for _, tags in notes:
        assert "when" not in tags
        assert "loc" not in tags
        assert "here" not in tags


def test_where_is_not_here_field():
    from experiments.run_tm012 import _has_field

    assert _has_field("where=0\naction=2\n", "where")
    assert not _has_field("where=0\naction=2\n", "here")
    assert not _has_field("where=0\naction=2\n", "loc")


def test_untrained_takes_first_code_file(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7))
    action, meta = _probe(a, "probe_red_with_key")
    assert a.use_search_head is True
    assert a.use_match_head is False
    assert meta["policy"].get("has_code") is True
    assert "p99.tag" not in a.store.list_files()
    assert action != Action.USE_KEY


def test_prefer_rare_keeps_messy_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, _ = _probe(a, "probe_red_with_key")
    assert "p99.tag" in a.store.list_files()
    assert "where=0" in _tags(s)
    assert "pad=7" in _tags(s)
    assert "loc=" not in _tags(s)
    assert "\nhere=" not in _tags(s)
    assert action == Action.USE_KEY


def test_exact_match_misses_messy_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, use_search_head=False, use_match_head=True)
    action, _ = _probe(a, "probe_red_with_key")
    assert "p99.tag" not in a.store.list_files()
    assert action != Action.USE_KEY


def test_search_and_match_conflict():
    try:
        agent_mod.ThreeMemoryAgent(
            native=True,
            use_policy=UsePolicy(seed=7),
            use_search_head=True,
            use_match_head=True,
        )
    except ValueError as e:
        assert "search" in str(e)
    else:
        raise AssertionError("expected conflict")


def test_clone_empty_search_flags(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    b = a.clone_empty()
    assert b.use_search_head is True
    assert b.use_match_head is False
    assert b.use_qname_head is False
    assert b.force_use is False


def test_default_search_prefers_code():
    p = UsePolicy(seed=7)
    assert float(p.w_search[0]) > 0.0
    assert abs(float(p.w_search[1])) < 1e-12


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_where()
    test_wiki_notes_are_messy()
    test_where_is_not_here_field()
    test_search_and_match_conflict()
    test_default_search_prefers_code()
    fns = [
        test_untrained_takes_first_code_file,
        test_prefer_rare_keeps_messy_page,
        test_exact_match_misses_messy_page,
        test_clone_empty_search_flags,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
