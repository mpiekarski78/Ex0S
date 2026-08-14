"""TM.0.3.2: prose retrieve — no filed where=/action=; free life."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012 import _has_field
from experiments.run_tm032 import live_free, make, wiki_prose
from three_memory import agent as agent_mod
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import ProseLibrary, extract_prose_ints, prose_to_record, write_prose_notes


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_agent_has_no_where():
    src = inspect.getsource(agent_mod)
    assert '"where"' not in src and "'where'" not in src


def test_wiki_is_pure_prose():
    notes = wiki_prose(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert "p99.md" in names and "p98.md" in names
    for name, body in notes:
        assert name.endswith(".md")
        assert "where=" not in body
        assert "action=" not in body
        assert "loc=" not in body
        assert "door=" not in body
        assert "\nhere=" not in body
        assert "place" in body or "Hallway" in body or "Staff" in body or "Clutter" in body
        assert "p99" not in body and "p98" not in body  # no filename digits in body
    red = dict(notes)["p99.md"]
    assert 0 in extract_prose_ints(red) and 2 in extract_prose_ints(red)
    # Heading digits must not pollute ints used as world content.
    from three_memory.tag_store import prose_to_record
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_prose_notes(Path(d), [("p99.md", red)])
        rec = prose_to_record(Path(d) / "p99.md")
        assert rec is not None
        assert list(rec.tags[k] for k in sorted(rec.tags) if k.startswith("n")) == [0, 2]


def test_prose_to_record_rejects_filed_tags(tmp_path: Path):
    p = tmp_path / "x.md"
    p.write_text("# x\n\nHello\nwhere=0\naction=2\n", encoding="utf-8")
    assert prose_to_record(p) is None
    write_prose_notes(tmp_path, [("y.md", "At place 0 the working motor was 2.\n")])
    rec = prose_to_record(tmp_path / "y.md")
    assert rec is not None
    assert rec.tags["n0"] == 0 and rec.tags["n1"] == 2
    assert "action" not in rec.tags and "where" not in rec.tags


def test_prefer_rare_prose_life(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_red=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_vname = np.array([-2.0, 0.0], dtype=np.float64)  # prefer non-code int
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    assert a.use_prose_ints is True
    assert isinstance(a.world, ProseLibrary)
    live = live_free(a, "experience_teach", 3, max_steps=6)
    assert live["found_red_pair"]
    assert "p99.tag" in live["files"]
    assert not _has_field(live["tag"], "action")
    assert not _has_field(live["tag"], "where")
    a.world = None
    a.reset_rho()
    action, _ = _probe(a, "probe_red_with_key")
    assert action == Action.USE_KEY


def test_untrained_does_not_solve(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_red=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=0.5, rng=np.random.default_rng(9))
    live_free(a, "experience_teach", 4, max_steps=16)
    a.world = None
    a.reset_rho()
    action, _ = _probe(a, "probe_red_with_key")
    assert action != Action.USE_KEY


def test_exact_match_misses_prose(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_red=True))
    a = make(
        s,
        w,
        UsePolicy(seed=7),
        use_search_head=False,
        use_match_head=True,
        use_vname_head=False,
        use_prose_ints=False,
        explore_epsilon=0.0,
    )
    live_free(a, "experience_teach", 5, max_steps=4)
    assert a.store.list_files() == [] or "p99.tag" not in a.store.list_files()


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_where()
    test_wiki_is_pure_prose()
    fns = [
        test_prose_to_record_rejects_filed_tags,
        test_prefer_rare_prose_life,
        test_untrained_does_not_solve,
        test_exact_match_misses_prose,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
