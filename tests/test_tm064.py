"""TM.0.6.4: English find without a unique rare token."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import _n_paragraphs, _rare_words
from experiments.run_tm054 import clutter_prose as clutter_closed
from experiments.run_tm054 import make as make054
from experiments.run_tm060 import make as make060
from experiments.run_tm061 import make as make061
from experiments.run_tm061 import wiki_prose as wiki_closed
from experiments.run_tm062 import make as make062
from experiments.run_tm063 import make as make063
from experiments.run_tm064 import clutter_prose, make, wiki_prose
from three_memory import agent as agent_mod
from three_memory.dial_env import STATION_NAMES, ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import extract_prose_ints, prose_token_stream, prose_tokens, write_prose_notes

_MOTOR = {a.name.lower() for a in DialAction}
_STATIONS = set(STATION_NAMES.values())


def test_wiki_has_several_rare_clutter_pages():
    rare = _rare_words(wiki_prose(include_a=True, include_c=True))
    clutter_rare = [n for n, ws in rare.items() if n.startswith("c") and ws]
    assert len(clutter_rare) >= 3
    assert set(rare["p99.md"]) >= {"push", "argon"}
    assert set(rare["p98.md"]) >= {"adjust", "alpha"}
    assert "xenon" in set(rare["c08.md"])
    assert "neon" in set(rare["c09.md"])
    assert "krypton" in set(rare["c10.md"])
    assert "argon" not in set(rare.get("c09.md") or [])
    a_body = wiki_prose(include_a=True)[-1][1]
    assert prose_token_stream(a_body).index("push") < prose_token_stream(a_body).index("argon")
    closed = _rare_words(wiki_closed(include_a=True))
    assert not any(closed[n] for n in closed if n.startswith("c"))


def test_hapax_clutter_is_still_english_documents():
    for name, body in clutter_prose():
        assert _n_paragraphs(body) >= 2
        assert not extract_prose_ints(body)
        toks = prose_tokens(body)
        assert not (toks & _MOTOR)
        assert not (toks & _STATIONS)
    closed_names = {n for n, _ in clutter_closed()}
    hapax_names = {n for n, _ in clutter_prose()}
    assert closed_names == hapax_names


def test_untrained_holds(tmp_path: Path):
    import numpy as np
    from experiments.run_tm052 import live_free

    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(s, w, UsePolicy(seed=7), explore_epsilon=1.0, rng=np.random.default_rng(0))
    live = live_free(a, "experience_channel_a", 3, max_steps=32)
    assert live["n_annotated"] == 0
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_a == int(DialAction.HOLD)
    assert act_c == int(DialAction.HOLD)


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm064.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert 'domain="dial"' in src_exp
    assert "KeyDoorWorld" not in src_exp
    assert 'flags["find_without_unique_rare"] = True' in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_stamp_new_here and not a.use_one_bind
    b = make054(Path("/tmp/tm064_skip_s"), None, UsePolicy(seed=1), enabled=False)
    assert not b.use_stamp_new_here
    c = make060(Path("/tmp/tm064_skip_060"), None, UsePolicy(seed=1), enabled=False)
    assert not c.use_one_bind
    d = make061(Path("/tmp/tm064_skip_061"), None, UsePolicy(seed=1), enabled=False)
    assert d.use_one_bind and not d.use_stamp_new_here
    e = make062(Path("/tmp/tm064_skip_062"), None, UsePolicy(seed=1), enabled=False)
    assert not e.use_stamp_new_here
    f = make063(Path("/tmp/tm064_skip_063"), None, UsePolicy(seed=1), enabled=False)
    assert f.use_stamp_new_here
    g = make(Path("/tmp/tm064_skip_064"), None, UsePolicy(seed=1), enabled=False)
    assert g.use_stamp_new_here and g.use_one_bind


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src
    assert '"xenon"' not in src and "'xenon'" not in src
    assert '"krypton"' not in src and "'krypton'" not in src


if __name__ == "__main__":
    import tempfile

    test_wiki_has_several_rare_clutter_pages()
    test_hapax_clutter_is_still_english_documents()
    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    with tempfile.TemporaryDirectory() as d:
        test_untrained_holds(Path(d))
    print("ok")
