"""v10: free life, no forced curriculum; probe is greedy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import TagStore


def test_probe_without_explore_stays_open(tmp_path: Path):
    (tmp_path / "S").mkdir()
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        use_policy=UsePolicy(seed=7),
        write_from_events=True,
        store=TagStore(tmp_path / "S"),
        cortex_seed=1337,
        explore_epsilon=0.9,
        policy_epsilon=0.0,
    )
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False, explore=False)
    assert action == Action.OPEN
    assert not meta["explored"]


def test_affordances_hide_key_use_until_held():
    a = ThreeMemoryAgent(native=True, cortex_seed=1337)
    world = KeyDoorWorld(0)
    obs = world.reset("experience_teach")
    assert Action.USE_KEY not in a._affordances(obs)
    assert Action.PICK_KEY in a._affordances(obs)
    result = world.step(int(Action.PICK_KEY), "experience_teach")
    obs = result.obs
    assert Action.USE_KEY in a._affordances(obs)
    assert Action.PICK_KEY not in a._affordances(obs)


def test_free_red_life_opens_without_script(tmp_path: Path):
    (tmp_path / "S").mkdir()
    rng = np.random.default_rng(0)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        use_policy=UsePolicy(seed=7),
        write_from_events=False,
        store=TagStore(tmp_path / "S"),
        cortex_seed=1337,
        explore_epsilon=1.0,
        policy_rng=rng,
    )
    world = KeyDoorWorld(0)
    obs = world.reset("experience_teach")
    opened = False
    forced = 0
    actions: list[int] = []
    for _ in range(40):
        action, meta = a.act(obs, update_rho=True, explore=True)
        result = world.step(int(action), "experience_teach")
        actions.append(int(action))
        forced += int(bool(meta.get("forced")))
        if result.info.get("opened"):
            opened = True
            break
        obs = result.obs
    assert forced == 0
    assert opened
    assert Action.PICK_KEY in actions or Action.USE_KEY in actions


def test_free_green_life_can_wait(tmp_path: Path):
    (tmp_path / "S").mkdir()
    rng = np.random.default_rng(1)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        use_policy=UsePolicy(seed=7),
        write_from_events=False,
        store=TagStore(tmp_path / "S"),
        cortex_seed=1337,
        explore_epsilon=1.0,
        policy_rng=rng,
    )
    world = KeyDoorWorld(0)
    obs = world.reset("experience_green")
    opened = False
    used_wait = False
    for _ in range(40):
        action, _ = a.act(obs, update_rho=True, explore=True)
        result = world.step(int(action), "experience_green")
        if action == Action.WAIT:
            used_wait = True
        if result.info.get("opened"):
            opened = True
            break
        obs = result.obs
    assert used_wait
    assert opened


if __name__ == "__main__":
    import tempfile

    test_affordances_hide_key_use_until_held()
    with tempfile.TemporaryDirectory() as d:
        test_probe_without_explore_stays_open(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_free_red_life_opens_without_script(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_free_green_life_can_wait(Path(d))
    print("ok")
