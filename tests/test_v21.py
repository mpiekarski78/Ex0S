"""v21: among unread W pages that share here=, keep newest when= not first/dump."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v21 import GREEN_JUNK, GREEN_WAIT, RED_JUNK, RED_USE, make, wiki_notes
from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent: ThreeMemoryAgent, scenario: str) -> tuple[int, dict]:
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_wiki_notes_filename_first_is_junk():
    notes = wiki_notes(include_red=True, include_green=True)
    names = [n[0] for n in notes]
    assert "d0.tag" not in names
    assert "d2.tag" not in names
    red = [n for n in notes if n[0] in (RED_JUNK[0], RED_USE[0])]
    assert red[0][0] == RED_JUNK[0] or sorted(n[0] for n in red)[0] == RED_JUNK[0]
    assert RED_JUNK[1]["when"] < RED_USE[1]["when"]
    assert RED_JUNK[1]["action"] == 0
    assert RED_USE[1]["action"] == 2
    assert GREEN_JUNK[1]["action"] == 1
    assert GREEN_WAIT[1]["action"] == 0


def test_untrained_first_file_waits(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7), dump=False)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("wsel_alt") is False
    assert "aaa.tag" in a.store.list_files()
    assert "p99.tag" not in a.store.list_files()
    assert action == Action.WAIT


def test_newest_keeps_useful(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_wsel = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, dump=False)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("wsel_alt") is True
    assert "p99.tag" in a.store.list_files()
    assert "aaa.tag" not in a.store.list_files()
    assert action == Action.USE_KEY


def test_untrained_dump_mixes_wait(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = make(s, w, UsePolicy(seed=7), dump=True)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("wsel_alt") is False
    files = set(a.store.list_files())
    assert "aaa.tag" in files and "p99.tag" in files
    assert action == Action.WAIT


def test_dump_green_opens(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_green=True))
    a = make(s, w, UsePolicy(seed=7), dump=True)
    action, _ = _probe(a, "probe_green")
    assert "aag.tag" in a.store.list_files()
    assert action == Action.OPEN


def test_newest_green_waits(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_green=True))
    policy = UsePolicy(seed=7)
    policy.b_wsel = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, dump=True)
    action, _ = _probe(a, "probe_green")
    assert "p98.tag" in a.store.list_files()
    assert "aag.tag" not in a.store.list_files()
    assert action == Action.WAIT


def test_recency_swap_newest_is_junk(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(
        w,
        [
            ("aaa.tag", {"here": 0, "action": 0, "when": 9}),
            ("p99.tag", {"here": 0, "action": 2, "when": 1}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.b_wsel = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, dump=False)
    action, _ = _probe(a, "probe_red_with_key")
    assert "aaa.tag" in a.store.list_files()
    assert action == Action.WAIT


def test_v20_still_takes_first_without_wsel(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, wiki_notes(include_red=True))
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="commit",
        use_policy=UsePolicy(seed=7),
        use_read=True,
        place_key="here",
        force_use=True,
        write_from_events=False,
        store=TagStore(s),
        world=TagLibrary(w),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    action, _ = _probe(a, "probe_red_with_key")
    assert a.store.list_files() == ["aaa.tag"]
    assert action == Action.WAIT


def test_clone_empty_keeps_wsel_flags(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7), dump=True)
    b = a.clone_empty()
    assert b.use_wsel_head is True
    assert b.wsel_dump is True
    assert b.place_key == "here"
    assert b.force_use is True
    assert b.write_from_events is False


def test_default_wsel_stays_off():
    p = UsePolicy(seed=7)
    assert float(p.b_wsel) < 0.0


def test_no_write_from_events(tmp_path: Path):
    a = make(tmp_path / "S", None, UsePolicy(seed=7), dump=False)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    info = a.observe_outcome(obs, True, {"opened": True, "action": "use_key"})
    assert info["wrote"] is False


if __name__ == "__main__":
    import tempfile

    test_wiki_notes_filename_first_is_junk()
    test_default_wsel_stays_off()
    fns = [
        test_untrained_first_file_waits,
        test_newest_keeps_useful,
        test_untrained_dump_mixes_wait,
        test_dump_green_opens,
        test_newest_green_waits,
        test_recency_swap_newest_is_junk,
        test_v20_still_takes_first_without_wsel,
        test_clone_empty_keeps_wsel_flags,
        test_no_write_from_events,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
