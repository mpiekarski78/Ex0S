"""v11: select among two authored .tag files; dump-all mixes them."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.symbols import GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagStore, write_tag_notes


def _notes() -> list[tuple[str, dict]]:
    return [
        (f"{RED_FACT_ID}.tag", {"door": 0, "action": 2}),
        (f"{GREEN_FACT_ID}.tag", {"door": 2, "action": 0}),
    ]


def test_select_red_ignores_green_note(tmp_path: Path):
    write_tag_notes(tmp_path, _notes())
    a = ThreeMemoryAgent(native=True, retrieve_policy="select", store=TagStore(tmp_path), cortex_seed=1337)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.USE_KEY


def test_select_green_ignores_red_note(tmp_path: Path):
    write_tag_notes(tmp_path, _notes())
    a = ThreeMemoryAgent(native=True, retrieve_policy="select", store=TagStore(tmp_path), cortex_seed=1337)
    obs = KeyDoorWorld(0).reset("probe_green")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action == Action.WAIT


def test_dump_red_does_not_use_key(tmp_path: Path):
    write_tag_notes(tmp_path, _notes())
    a = ThreeMemoryAgent(native=True, retrieve_policy="dump", store=TagStore(tmp_path), cortex_seed=1337)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    action, _ = a.act(obs, update_rho=False, explore=False)
    assert action != Action.USE_KEY


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_select_red_ignores_green_note(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_select_green_ignores_red_note(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_dump_red_does_not_use_key(Path(d))
    print("ok")
