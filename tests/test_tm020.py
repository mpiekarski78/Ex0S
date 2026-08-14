"""TM.0.2.0: scale of W — messy retrieve among hundreds of unread files."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm020 import N_W, WIKI_RED, _has_field, make, scale_clutter, wiki_notes
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


def test_pool_is_256():
    red = wiki_notes(include_red=True)
    green = wiki_notes(include_green=True)
    assert len(red) == N_W == 256
    assert len(green) == N_W
    assert len(scale_clutter()) == N_W - 1


def test_wiki_notes_are_messy_scale():
    notes = wiki_notes(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    assert "d1.tag" not in names
    assert "d2.tag" not in names
    assert WIKI_RED[0] in names
    assert "p98.tag" in names
    by = dict(notes)
    assert by["p99.tag"]["where"] == 0
    assert by["p99.tag"]["action"] == 2
    assert by["p99.tag"]["pad"] == 7
    for name, tags in notes:
        assert "when" not in tags
        assert "loc" not in tags
        assert "here" not in tags
        assert "door" not in tags
        if name.startswith("c"):
            assert tags.get("action") == 1
            assert tags.get("action") not in (0, 2)


def test_where_is_not_here_field():
    assert _has_field("where=0\naction=2\n", "where")
    assert not _has_field("where=0\naction=2\n", "here")
    assert not _has_field("where=0\naction=2\n", "loc")


def test_clutter_sorts_before_p99():
    names = [n[0] for n in wiki_notes(include_red=True)]
    assert names.index("c000.tag") < names.index("p99.tag")
    assert sorted(names)[0].startswith("c")


def test_untrained_does_not_commit_p99(tmp_path: Path):
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
    assert not _has_field(_tags(s), "here")
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


def test_default_search_prefers_code():
    p = UsePolicy(seed=7)
    assert float(p.w_search[0]) > 0.0
    assert abs(float(p.w_search[1])) < 1e-12


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_where()
    test_pool_is_256()
    test_wiki_notes_are_messy_scale()
    test_where_is_not_here_field()
    test_clutter_sorts_before_p99()
    test_default_search_prefers_code()
    fns = [
        test_untrained_does_not_commit_p99,
        test_prefer_rare_keeps_messy_page,
        test_exact_match_misses_messy_page,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
