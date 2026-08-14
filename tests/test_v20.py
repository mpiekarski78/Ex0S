"""v20: find unread W by here=; junk on door= must not leak use_key."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v20 import JUNK_RED, WIKI_GREEN, WIKI_RED, make, wiki_notes
from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent: ThreeMemoryAgent, scenario: str) -> tuple[int, dict]:
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_wiki_notes_are_not_d0():
    notes = wiki_notes(include_red=True, include_green=True, junk=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    assert "d2.tag" not in names
    assert WIKI_RED[0] in names
    assert WIKI_GREEN[0] in names
    assert JUNK_RED[0] in names
    assert WIKI_RED[1]["here"] == 0
    assert WIKI_RED[1]["action"] == 2
    assert JUNK_RED[1]["door"] == 0
    assert JUNK_RED[1]["action"] == 0


def test_match_before_collect_commits_here_page(tmp_path: Path):
    """Same-step match here= must see W. Stale door= query would miss p99.tag."""
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, [WIKI_RED])
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, force_use=True)
    action, meta = _probe(a, "probe_red_with_key")
    text = _tags(s)
    assert "here=0" in text
    assert "p99.tag" in a.store.list_files()
    assert meta["policy"].get("match_alt") is True
    assert action == Action.USE_KEY


def test_collect_keeps_match_alt(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, [WIKI_RED])
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, force_use=True)
    _, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is True
    assert a.collect_mode == "commit"


def test_untrained_misses_here_wiki(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True, junk=False))
    policy = UsePolicy(seed=7)
    a = make(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is False
    assert action == Action.OPEN
    assert "here=" not in _tags(s)


def test_untrained_junk_is_not_use_key(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True, junk=True))
    policy = UsePolicy(seed=7)
    a = make(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is False
    assert action != Action.USE_KEY


def test_door_match_applies_junk_wait(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True, junk=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(-3.0, dtype=np.float64)
    a = make(s, w, policy, force_use=True)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is False
    assert "door=0" in _tags(s)
    assert "here=" not in _tags(s)
    assert action == Action.WAIT


def test_here_match_ignores_junk(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True, junk=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, force_use=True)
    action, _ = _probe(a, "probe_red_with_key")
    text = _tags(s)
    assert "here=0" in text
    assert "action=2" in text
    assert action == Action.USE_KEY


def test_peek_unmount_is_not_memory(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, [WIKI_RED])
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="peek",
        use_policy=policy,
        use_read=True,
        use_match_head=True,
        force_use=True,
        write_from_events=False,
        store=TagStore(s),
        world=TagLibrary(w),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    action_peek, _ = _probe(a, "probe_red_with_key")
    assert action_peek == Action.USE_KEY
    assert _tags(s) == ""
    b = make(s, None, policy, force_use=True)
    action_unmount, _ = _probe(b, "probe_red_with_key")
    assert action_unmount == Action.OPEN


def test_green_here_wait(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, [WIKI_GREEN])
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, force_use=True)
    action, _ = _probe(a, "probe_green")
    assert action == Action.WAIT
    assert "here=2" in _tags(s)
    assert "action=0" in _tags(s)


def test_clone_empty_keeps_find_flags(tmp_path: Path):
    policy = UsePolicy(seed=7)
    a = make(tmp_path / "S", None, policy)
    b = a.clone_empty()
    assert b.use_match_head is True
    assert b.use_read is True
    assert b.collect_mode == "commit"
    assert b.write_from_events is False


def test_default_match_still_door():
    p = UsePolicy(seed=7)
    assert float(p.b_match) < 0.0
    assert float(p.b_use) < 0.0


def test_no_write_from_events(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    assert a.write_from_events is False
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    info = a.observe_outcome(obs, True, {"opened": True, "action": "use_key"})
    assert info["wrote"] is False
    assert _tags(tmp_path / "S") == ""


if __name__ == "__main__":
    import tempfile

    test_wiki_notes_are_not_d0()
    test_default_match_still_door()
    fns = [
        test_match_before_collect_commits_here_page,
        test_collect_keeps_match_alt,
        test_untrained_misses_here_wiki,
        test_untrained_junk_is_not_use_key,
        test_door_match_applies_junk_wait,
        test_here_match_ignores_junk,
        test_peek_unmount_is_not_memory,
        test_green_here_wait,
        test_clone_empty_keeps_find_flags,
        test_no_write_from_events,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
