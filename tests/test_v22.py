"""v22: complete vs stub; joint match+wsel+use without clamps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v22 import (
    GREEN_COMPLETE,
    GREEN_STUB,
    RED_COMPLETE,
    RED_STUB,
    joint_notes,
    make_a,
    make_b,
    stub_notes,
)
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_tag_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_stub_notes_have_no_when():
    notes = stub_notes(include_red=True, include_green=True)
    for name, tags in notes:
        assert "when" not in tags
        assert name not in ("d0.tag", "d2.tag")
    assert RED_STUB[1] == {"here": 0}
    assert RED_COMPLETE[1]["action"] == 2
    assert GREEN_STUB[1] == {"here": 2}
    assert GREEN_COMPLETE[1]["action"] == 0


def test_untrained_stub_opens(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, stub_notes(include_red=True))
    a = make_a(s, w, UsePolicy(seed=7))
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("wcomp_alt") is False
    assert a.store.list_files() == ["aaa.tag"]
    assert "action=" not in _tags(s)
    assert action == Action.OPEN


def test_complete_keeps_payload(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, stub_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_wcomp = np.array(3.0, dtype=np.float64)
    a = make_a(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("wcomp_alt") is True
    assert "p99.tag" in a.store.list_files()
    assert "aaa.tag" not in a.store.list_files()
    assert action == Action.USE_KEY


def test_complete_green_waits(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, stub_notes(include_green=True))
    policy = UsePolicy(seed=7)
    policy.b_wcomp = np.array(3.0, dtype=np.float64)
    a = make_a(s, w, policy)
    action, _ = _probe(a, "probe_green")
    assert action == Action.WAIT
    assert "p98.tag" in a.store.list_files()


def test_complete_swap_is_wait(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(
        w,
        [
            ("aaa.tag", {"here": 0, "action": 0}),
            ("p99.tag", {"here": 0}),
        ],
    )
    policy = UsePolicy(seed=7)
    policy.b_wcomp = np.array(3.0, dtype=np.float64)
    a = make_a(s, w, policy)
    action, _ = _probe(a, "probe_red_with_key")
    assert "aaa.tag" in a.store.list_files()
    assert action == Action.WAIT


def test_joint_untrained_not_use_key(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, joint_notes(include_red=True))
    a = make_b(s, w, UsePolicy(seed=7))
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is False
    assert a.force_use is False
    assert action != Action.USE_KEY


def test_joint_forced_heads_use_key(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_tag_notes(w, joint_notes(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_wsel = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make_b(s, w, policy)
    action, meta = _probe(a, "probe_red_with_key")
    assert meta["policy"].get("match_alt") is True
    assert meta["policy"].get("wsel_alt") is True
    assert "p99.tag" in a.store.list_files()
    assert action == Action.USE_KEY


def test_clone_empty_keeps_v22_flags(tmp_path: Path):
    a = make_a(tmp_path / "S", None, UsePolicy(seed=7))
    b = a.clone_empty()
    assert b.use_wcomp_head is True
    assert b.place_key == "here"
    c = make_b(tmp_path / "T", None, UsePolicy(seed=7)).clone_empty()
    assert c.use_match_head is True
    assert c.use_wsel_head is True
    assert c.force_use is False
    assert c.place_key == "door"


def test_default_wcomp_off():
    p = UsePolicy(seed=7)
    assert float(p.b_wcomp) < 0.0


if __name__ == "__main__":
    import tempfile

    test_stub_notes_have_no_when()
    test_default_wcomp_off()
    fns = [
        test_untrained_stub_opens,
        test_complete_keeps_payload,
        test_complete_green_waits,
        test_complete_swap_is_wait,
        test_joint_untrained_not_use_key,
        test_joint_forced_heads_use_key,
        test_clone_empty_keeps_v22_flags,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
