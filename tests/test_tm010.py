"""TM.0.1.0: query names come from files, not a {door, here} menu."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm010 import WIKI_RED, make, wiki_notes
from three_memory import agent as agent_mod
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_agent_has_no_loc_menu():
    src = inspect.getsource(agent_mod)
    assert '"loc"' not in src and "'loc'" not in src


def test_wiki_notes_use_open_name_not_here():
    notes = wiki_notes(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    assert "p99.tag" in names
    by = dict(notes)
    assert "loc" in by["p99.tag"]
    assert "here" not in by["p99.tag"]
    assert "door" not in by["p99.tag"]
    assert by["p99.tag"]["action"] == 2
    assert "loc" in by["p98.tag"]
    for _, tags in notes:
        assert "when" not in tags
        assert "here" not in tags


def test_untrained_queries_a_file_key_not_use_key(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7))
    action, meta = _probe(a, "probe_red_with_key")
    assert a.use_qname_head is True
    assert a.use_match_head is False
    q = meta["policy"].get("qname")
    assert q in meta["policy"].get("qnames", [])
    assert q != "here"
    assert action != Action.USE_KEY
    assert "loc=" not in _tags(s)


def test_prefer_uncommon_keeps_loc_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.w_qname = np.array([1.2, -3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("qname") == "loc"
    assert "p99.tag" in a.store.list_files()
    assert "loc=0" in _tags(s)
    assert "action=2" in _tags(s)
    assert "here=" not in _tags(s)
    assert action == Action.USE_KEY


def test_match_menu_misses_loc_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, use_qname_head=False, use_match_head=True)
    action, _ = _probe(a, "probe_red_with_key")
    assert "p99.tag" not in a.store.list_files()
    assert action != Action.USE_KEY


def test_qname_and_match_conflict():
    try:
        agent_mod.ThreeMemoryAgent(
            native=True,
            use_policy=UsePolicy(seed=7),
            use_qname_head=True,
            use_match_head=True,
        )
    except ValueError as e:
        assert "qname" in str(e)
    else:
        raise AssertionError("expected conflict")


def test_clone_empty_qname_flags(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    b = a.clone_empty()
    assert b.use_qname_head is True
    assert b.use_match_head is False
    assert b.force_use is False
    assert b.write_from_events is False


def test_default_qname_prefers_hit():
    p = UsePolicy(seed=7)
    assert float(p.w_qname[0]) > 0.0
    assert abs(float(p.w_qname[1])) < 1e-12


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_loc_menu()
    test_wiki_notes_use_open_name_not_here()
    test_qname_and_match_conflict()
    test_default_qname_prefers_hit()
    fns = [
        test_untrained_queries_a_file_key_not_use_key,
        test_prefer_uncommon_keeps_loc_page,
        test_match_menu_misses_loc_page,
        test_clone_empty_qname_flags,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
