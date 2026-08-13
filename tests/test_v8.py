"""v8 boxed policy: features have no door id; cortex frozen."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import GREEN_NOTE, TagLibrary, TagStore, all_tag_notes, write_tag_notes


def test_features_ignore_door():
    a = UsePolicy.features(False, True, novelty=0.9)
    b = UsePolicy.features(False, True, novelty=0.1)
    assert np.allclose(a, b)
    assert a.tolist() == [0.0, 1.0]


def test_untrained_apply_off_prefers_open(tmp_path: Path):
    write_tag_notes(tmp_path / "W", all_tag_notes(include_red=True))
    (tmp_path / "S").mkdir()
    policy = UsePolicy(seed=7)
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="policy",
        use_policy=policy,
        write_from_events=False,
        store=TagStore(tmp_path / "S"),
        world=TagLibrary(tmp_path / "W"),
        cortex_seed=1337,
        policy_epsilon=0.0,
    )
    world = KeyDoorWorld(0)
    obs = world.reset("probe_red_with_key")
    action, meta = a.act(obs, update_rho=False)
    assert action == Action.OPEN
    assert a.weight_hash() == ThreeMemoryAgent(cortex_seed=1337).weight_hash()


def test_green_wait_with_applied_file(tmp_path: Path):
    write_tag_notes(tmp_path, [GREEN_NOTE])
    a = ThreeMemoryAgent(
        native=True,
        collect_mode="off",
        store=TagStore(tmp_path),
        cortex_seed=1337,
    )
    world = KeyDoorWorld(0)
    obs = world.reset("probe_green")
    action, _ = a.act(obs, update_rho=False)
    assert action == Action.WAIT


def test_policy_update_changes_hash():
    p = UsePolicy(seed=1)
    h0 = p.weight_hash()
    feat = UsePolicy.features(False, True)
    tr = p.decide(feat, epsilon=0.0)
    p.update([tr], 1.0)
    assert p.weight_hash() != h0
    assert p.changed()


if __name__ == "__main__":
    import tempfile

    test_features_ignore_door()
    test_policy_update_changes_hash()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_apply_off_prefers_open(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_green_wait_with_applied_file(Path(d))
    print("ok")
