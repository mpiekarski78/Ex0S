"""v9: write from events; W has no answer; features have no door id."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, all_tag_notes, write_tag_notes


def test_write_features_ignore_door():
    a = UsePolicy.features(False, True)
    b = UsePolicy.features(False, True)
    assert np.allclose(a, b)
    assert a.tolist() == [0.0, 1.0]


def test_untrained_does_not_write(tmp_path: Path):
    (tmp_path / "S").mkdir()
    policy = UsePolicy(seed=7)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        use_policy=policy,
        write_from_events=True,
        store=TagStore(tmp_path / "S"),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    world = KeyDoorWorld(0)
    world.reset("experience_teach")
    for act in (Action.OPEN, Action.PICK_KEY, Action.USE_KEY):
        result = world.step(int(act), "experience_teach")
        a.observe_outcome(result.obs, result.success, result.info)
    assert TagStore(tmp_path / "S").list_files() == []
    a.reset_rho()
    obs = KeyDoorWorld(1).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False)
    assert action == Action.OPEN


def test_forced_write_authors_action_from_success(tmp_path: Path):
    (tmp_path / "S").mkdir()
    policy = UsePolicy(seed=7)
    policy.b_write = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        use_policy=policy,
        write_from_events=True,
        store=TagStore(tmp_path / "S"),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    world = KeyDoorWorld(0)
    world.reset("experience_teach")
    for act in (Action.OPEN, Action.PICK_KEY, Action.USE_KEY):
        result = world.step(int(act), "experience_teach")
        a.observe_outcome(result.obs, result.success, result.info)
    text = (tmp_path / "S" / f"{RED_FACT_ID}.tag").read_text(encoding="utf-8")
    assert "door=0" in text
    assert "action=2" in text
    assert "opens" not in text
    a.reset_rho()
    obs = KeyDoorWorld(1).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False)
    assert action == Action.USE_KEY


def test_green_success_authors_wait(tmp_path: Path):
    (tmp_path / "S").mkdir()
    policy = UsePolicy(seed=7)
    policy.b_write = np.array(3.0, dtype=np.float64)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        use_policy=policy,
        write_from_events=True,
        store=TagStore(tmp_path / "S"),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    world = KeyDoorWorld(0)
    world.reset("experience_green")
    for act in (Action.OPEN, Action.WAIT):
        result = world.step(int(act), "experience_green")
        a.observe_outcome(result.obs, result.success, result.info)
    text = (tmp_path / "S" / f"{GREEN_FACT_ID}.tag").read_text(encoding="utf-8")
    assert "door=2" in text
    assert "action=0" in text
    a.reset_rho()
    obs = KeyDoorWorld(1).reset("probe_green")
    action, _ = a.act(obs, update_rho=False)
    assert action == Action.WAIT


def test_clutter_w_has_no_answer(tmp_path: Path):
    write_tag_notes(tmp_path / "W", all_tag_notes(include_red=False, include_green=False))
    names = TagLibrary(tmp_path / "W").list_files()
    assert f"{RED_FACT_ID}.tag" not in names
    assert f"{GREEN_FACT_ID}.tag" not in names


def test_write_update_changes_hash():
    p = UsePolicy(seed=1)
    h0 = p.weight_hash()
    feat = UsePolicy.features(False, True)
    tr = p.decide_write(feat, epsilon=0.0)
    p.update([tr], 1.0)
    assert p.weight_hash() != h0


if __name__ == "__main__":
    import tempfile

    test_write_features_ignore_door()
    test_write_update_changes_hash()
    test_clutter_w_has_no_answer(Path(tempfile.mkdtemp()))
    with tempfile.TemporaryDirectory() as d:
        test_untrained_does_not_write(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_forced_write_authors_action_from_success(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_green_success_authors_wait(Path(d))
    print("ok")
