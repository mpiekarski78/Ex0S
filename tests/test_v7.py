"""v7 native integer tag files. No English prose."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.symbols import ACT_USE_KEY, DOOR_RED, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, all_tag_notes, write_tag_notes


def test_tagfile_has_no_english(tmp_path: Path):
    write_tag_notes(tmp_path, all_tag_notes(include_red=True))
    text = (tmp_path / f"{RED_FACT_ID}.tag").read_text(encoding="utf-8")
    assert "door=0" in text
    assert "action=2" in text
    assert "opens" not in text
    assert "love" not in text
    assert "NOTE" not in text


def test_select_uses_red_not_clutter(tmp_path: Path):
    write_tag_notes(tmp_path, all_tag_notes(include_red=True))
    a = ThreeMemoryAgent(native=True, store=TagStore(tmp_path), cortex_seed=1337)
    world = KeyDoorWorld(0)
    obs = world.reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False)
    assert action == Action.USE_KEY


def test_dump_prefers_open_from_clutter(tmp_path: Path):
    write_tag_notes(tmp_path, all_tag_notes(include_red=True))
    a = ThreeMemoryAgent(
        native=True, retrieve_policy="dump", store=TagStore(tmp_path), cortex_seed=1337
    )
    world = KeyDoorWorld(0)
    obs = world.reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False)
    assert action == Action.OPEN


def test_commit_copies_only_red(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, all_tag_notes(include_red=True))
    s.mkdir()
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="commit",
        store=TagStore(s),
        world=TagLibrary(w),
        cortex_seed=1337,
    )
    world = KeyDoorWorld(0)
    obs = world.reset("probe_red_with_key")
    a.act(obs, update_rho=False)
    assert TagStore(s).list_files() == [f"{RED_FACT_ID}.tag"]
    assert (w / f"{RED_FACT_ID}.tag").is_file()


def test_peek_unmount_loses_fact(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, all_tag_notes(include_red=True))
    s.mkdir()
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="peek",
        store=TagStore(s),
        world=TagLibrary(w),
        cortex_seed=1337,
    )
    world = KeyDoorWorld(0)
    obs = world.reset("probe_red_with_key")
    act_peek, _ = a.act(obs, update_rho=False)
    assert act_peek == Action.USE_KEY
    assert TagStore(s).list_files() == []
    b = ThreeMemoryAgent(native=True, collect_mode="off", store=TagStore(s), world=None, cortex_seed=1337)
    act_after, _ = b.act(obs, update_rho=False)
    assert act_after == Action.OPEN


def test_v0_still_string_tags():
    a = ThreeMemoryAgent(store_enabled=True, native=False)
    world = KeyDoorWorld(0)
    obs = world.reset("experience_teach")
    for _ in range(8):
        act, _ = a.act(obs)
        r = world.step(act)
        a.observe_outcome(r.obs, r.success, r.info)
        obs = r.obs
        if r.done:
            break
    assert a.weights_unchanged()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_tagfile_has_no_english(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_select_uses_red_not_clutter(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_dump_prefers_open_from_clutter(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_commit_copies_only_red(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_peek_unmount_loses_fact(Path(d))
    test_v0_still_string_tags()
    print("ok")
