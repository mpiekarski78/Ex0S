"""TM.0.3.1: free life over unread .md documents, not tidy .tag W."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm012 import _has_field
from experiments.run_tm031 import live_free, make, wiki_docs
from three_memory import agent as agent_mod
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import DocLibrary, write_doc_notes


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _probe(agent, scenario: str):
    obs = KeyDoorWorld(0).reset(scenario)
    return agent.act(obs, update_rho=False, explore=False)


def test_agent_has_no_where():
    src = inspect.getsource(agent_mod)
    assert '"where"' not in src and "'where'" not in src


def test_wiki_is_md_documents():
    notes = wiki_docs(include_red=True, include_green=True)
    names = {n[0] for n in notes}
    assert all(n.endswith(".md") for n in names)
    assert "d0.tag" not in names and "d0.md" not in names
    assert "p99.md" in names and "p98.md" in names
    by = {n[0]: n for n in notes}
    assert by["p99.md"][2]["where"] == 0
    assert by["p99.md"][2]["action"] == 2
    assert "Staff scrap" in by["p99.md"][1]
    for name, prose, tags in notes:
        assert prose.strip()
        assert "when" not in tags
        assert "loc" not in tags
        assert "here" not in tags


def test_doc_library_loads_md_not_tag(tmp_path: Path):
    write_doc_notes(tmp_path, wiki_docs(include_red=True))
    (tmp_path / "ignored.tag").write_text("# x\naction=2\n", encoding="utf-8")
    lib = DocLibrary(tmp_path)
    files = lib.list_files()
    assert "ignored.tag" not in files
    assert "p99.md" in files
    assert all(f.endswith(".md") for f in files)
    recs = {r.fact_id: r for r in lib.records()}
    assert "p99" in recs
    assert recs["p99"].tags["where"] == 0
    assert recs["p99"].tags["action"] == 2


def test_make_uses_doc_library(tmp_path: Path):
    w = tmp_path / "W"
    write_doc_notes(w, wiki_docs(include_red=True))
    a = make(tmp_path / "S", w, UsePolicy(seed=7))
    assert isinstance(a.world, DocLibrary)
    assert a.write_from_events is False
    assert a.use_search_head is True
    assert a.record_search_on_explore is True


def test_prefer_rare_life_commits_from_md(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_doc_notes(w, wiki_docs(include_red=True))
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, explore_epsilon=0.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_teach", 3, max_steps=4)
    assert "p99.tag" in live["files"]  # committed into TagStore S
    assert live["found_action2"]
    assert _has_field(live["tag"], "where")
    a.world = None
    a.reset_rho()
    action, _ = _probe(a, "probe_red_with_key")
    assert action == Action.USE_KEY


def test_exact_match_misses_md_docs(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_doc_notes(w, wiki_docs(include_red=True))
    policy = UsePolicy(seed=7)
    policy.b_match = np.array(3.0, dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    a = make(s, w, policy, use_search_head=False, use_match_head=True, explore_epsilon=0.0)
    live_free(a, "experience_teach", 4, max_steps=4)
    assert "p99.tag" not in a.store.list_files()


def test_untrained_life_does_not_solve_probe(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_doc_notes(w, wiki_docs(include_red=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=0.5, rng=np.random.default_rng(9))
    live_free(a, "experience_teach", 5, max_steps=16)
    a.world = None
    a.reset_rho()
    action, _ = _probe(a, "probe_red_with_key")
    assert action != Action.USE_KEY


if __name__ == "__main__":
    import tempfile

    test_agent_has_no_where()
    test_wiki_is_md_documents()
    fns = [
        test_doc_library_loads_md_not_tag,
        test_make_uses_doc_library,
        test_prefer_rare_life_commits_from_md,
        test_exact_match_misses_md_docs,
        test_untrained_life_does_not_solve_probe,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
