"""v23: joint match+complete+use with no when=. Split vs shared return."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v23 import make, wiki_notes
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_wiki_notes_have_no_when():
    notes = wiki_notes(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "d0.tag" not in names
    assert "junk.tag" in names
    assert "aaa.tag" in names
    assert "p99.tag" in names
    for _, tags in notes:
        assert "when" not in tags


def test_untrained_not_use_key(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7))
    action, meta = _probe(a, "probe_red_with_key")
    assert a.force_use is False
    assert a.place_key == "door"
    assert meta["policy"].get("match_alt") is False
    assert action != Action.USE_KEY


def test_forced_three_heads_keep_complete(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_wcomp = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is True
    assert meta["policy"].get("wcomp_alt") is True
    assert "p99.tag" in a.store.list_files()
    assert "aaa.tag" not in a.store.list_files()
    assert "junk.tag" not in a.store.list_files()
    assert "when=" not in _tags(s)
    assert action == Action.USE_KEY


def test_door_match_keeps_junk_wait(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(-3.0, dtype=np.float64)
    policy.b_wcomp = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, _ = _probe(a, "probe_red_with_key")
    assert "junk.tag" in a.store.list_files()
    assert action == Action.WAIT


def test_stub_first_opens(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_wcomp = np.array(-3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, _ = _probe(a, "probe_red_with_key")
    assert a.store.list_files() == ["aaa.tag"]
    assert "action=" not in _tags(s)
    assert action == Action.OPEN


def test_green_complete_waits(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_green=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_wcomp = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy)
    action, _ = _probe(a, "probe_green")
    assert "p98.tag" in a.store.list_files()
    assert action == Action.WAIT


def test_clone_empty_joint_flags(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7))
    b = a.clone_empty()
    assert b.use_match_head is True
    assert b.use_wcomp_head is True
    assert b.use_wsel_head is False
    assert b.force_use is False
    assert b.place_key == "door"
    assert b.write_from_events is False


def test_default_wcomp_off():
    p = UsePolicy(seed=7)
    assert float(p.b_wcomp) < 0.0


if __name__ == "__main__":
    import tempfile

    test_wiki_notes_have_no_when()
    test_default_wcomp_off()
    fns = [
        test_untrained_not_use_key,
        test_forced_three_heads_keep_complete,
        test_door_match_keeps_junk_wait,
        test_stub_first_opens,
        test_green_complete_waits,
        test_clone_empty_joint_flags,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
