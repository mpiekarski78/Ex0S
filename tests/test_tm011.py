"""TM.0.1.1: copy names come from the hit file, not an {action, do} menu."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011 import WIKI_RED, make, wiki_notes
from three_memory import agent as agent_mod
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_agent_has_no_act_menu():
    src = inspect.getsource(agent_mod)
    assert '"act"' not in src and "'act'" not in src


def test_wiki_notes_use_open_value_name():
    notes = wiki_notes(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    by = dict(notes)
    assert by["p99.tag"]["act"] == 2
    assert "action" not in by["p99.tag"]
    assert "do" not in by["p99.tag"]
    assert by["p99.tag"]["loc"] == 0
    assert by["p98.tag"]["act"] == 0
    for _, tags in notes:
        assert "when" not in tags
        assert "do" not in tags


def test_untrained_copies_query_key_not_use_key(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7))
    action, meta = _probe(a, "probe_red_with_key")
    assert a.use_vname_head is True
    assert a.use_key_head is False
    assert a.place_key == "loc"
    assert meta["policy"].get("vname") == "loc"
    assert action != Action.USE_KEY


def test_prefer_not_query_copies_act(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.w_vname = np.array([-3.0, 0.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("vname") == "act"
    assert "p99.tag" in a.store.list_files()
    assert "act=2" in _tags(s)
    assert "action=" not in _tags(s)
    assert action == Action.USE_KEY


def test_green_place_code_is_use_key_if_copied(tmp_path: Path):
    """Copying loc=2 on green would use_key. The motor field is act=0 (wait)."""
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_green=True))
    policy = UsePolicy(seed=7)
    policy.w_vname = np.array([-3.0, 0.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, meta = _probe(a, "probe_green")
    assert meta["policy"].get("vname") == "act"
    assert action == Action.WAIT


def test_copy_menu_misses_act_page(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, use_vname_head=False, use_key_head=True)
    action, _ = _probe(a, "probe_red_with_key")
    assert "p99.tag" in a.store.list_files()
    assert action != Action.USE_KEY


def test_vname_and_key_conflict():
    try:
        agent_mod.ThreeMemoryAgent(
            native=True,
            use_policy=UsePolicy(seed=7),
            use_vname_head=True,
            use_key_head=True,
        )
    except ValueError as e:
        assert "vname" in str(e)
    else:
        raise AssertionError("expected conflict")


def test_clone_empty_vname_flags(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    b = a.clone_empty()
    assert b.use_vname_head is True
    assert b.use_key_head is False
    assert b.use_qname_head is False
    assert b.place_key == "loc"
    assert b.force_use is False


def test_default_vname_prefers_query():
    p = UsePolicy(seed=7)
    assert float(p.w_vname[0]) > 0.0
    assert abs(float(p.w_vname[1])) < 1e-12


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_act_menu()
    test_wiki_notes_use_open_value_name()
    test_vname_and_key_conflict()
    test_default_vname_prefers_query()
    fns = [
        test_untrained_copies_query_key_not_use_key,
        test_prefer_not_query_copies_act,
        test_green_place_code_is_use_key_if_copied,
        test_copy_menu_misses_act_page,
        test_clone_empty_vname_flags,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
