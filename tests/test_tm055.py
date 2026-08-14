"""TM.0.5.5: accumulate S — two lives, same store."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm052 import live_free
from experiments.run_tm054 import _n_paragraphs, _rare_words, clutter_prose, make, wiki_prose
from experiments.run_tm055 import _accumulate_lives
from three_memory import agent as agent_mod
from three_memory.dial_env import STATION_NAMES, ChannelDialWorld, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import extract_prose_ints, prose_tokens, write_prose_notes

_MOTOR = {a.name.lower() for a in DialAction}
_STATIONS = set(STATION_NAMES.values())


def _forced_policy() -> UsePolicy:
    policy = UsePolicy(seed=7)
    policy.w_search = np.array([1.2, 3.0], dtype=np.float64)
    policy.w_write = np.array([3.0, 3.0], dtype=np.float64)
    policy.b_write = np.array(3.0, dtype=np.float64)
    policy.w_vname = np.array([-2.0, 3.0], dtype=np.float64)
    policy.b_use = np.array(3.0, dtype=np.float64)
    return policy


def test_both_pages_open_and_rare():
    notes = wiki_prose(include_a=True, include_c=True)
    clutter = clutter_prose()
    assert len(set(b for _, b in clutter)) == len(clutter) >= 11
    for _, body in notes:
        assert _n_paragraphs(body) >= 2
        toks = prose_tokens(body)
        assert not extract_prose_ints(body)
        assert not (toks & _MOTOR)
        assert not (toks & _STATIONS)
    rare = _rare_words(notes)
    assert not any(rare[n] for n in rare if n.startswith("c"))
    assert "krypton" in set(rare["p99.md"])
    assert "helium" in set(rare["p98.md"])


def test_accumulate_a_press_c_tune(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    ag, a_live, a_after, a_foil, _, c_live, both_a, both_c, both_tag = _accumulate_lives(
        s,
        w,
        _forced_policy(),
        3,
        max_steps=40,
        explore_epsilon=1.0,
        rng=np.random.default_rng(0),
        wipe_between=False,
    )
    assert a_live["found_press"] and a_live["found_cha"]
    assert a_after["action_name"] == "press"
    assert a_foil["action_name"] == "hold"
    assert c_live["found_tune"] and c_live["found_chc"]
    assert c_live["found_press"] and c_live["found_cha"]
    assert both_a["action_name"] == "press" and both_a["correct"]
    assert both_c["action_name"] == "tune" and both_c["correct"]
    assert "press" in both_tag and "tune" in both_tag
    assert ag.weight_hash()


def test_wipe_between_loses_a(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True, include_c=True))
    _, _, _, _, _, _, wipe_a, wipe_c, _ = _accumulate_lives(
        s,
        w,
        _forced_policy(),
        3,
        max_steps=40,
        explore_epsilon=1.0,
        rng=np.random.default_rng(0),
        wipe_between=True,
    )
    assert wipe_a["action_name"] != "press"
    assert wipe_c["action_name"] == "tune"


def test_copy_only_presses_on_c(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_prose_notes(w, wiki_prose(include_a=True))
    a = make(
        s, w, _forced_policy(), use_here_match=False, explore_epsilon=1.0, rng=np.random.default_rng(0)
    )
    live_free(a, "experience_channel_a", 3, max_steps=40)
    a.world = None
    a.reset_rho()
    act_a, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_a"), update_rho=False, explore=False)
    a.reset_rho()
    act_c, _ = a.act(ChannelDialWorld(seed=3).reset("probe_channel_c"), update_rho=False, explore=False)
    assert act_a == int(DialAction.PRESS)
    assert act_c == int(DialAction.PRESS)


def test_skips_stay_skipped():
    src_agent = Path(REPO_ROOT / "three_memory" / "agent.py").read_text(encoding="utf-8")
    src_exp = Path(REPO_ROOT / "experiments" / "run_tm055.py").read_text(encoding="utf-8")
    assert "has_code" in src_agent
    assert "domain=\"dial\"" in Path(REPO_ROOT / "experiments" / "run_tm054.py").read_text(encoding="utf-8")
    assert "KeyDoorWorld" not in src_exp
    assert "Working motor was press" not in src_exp
    assert "shutil.rmtree(s_dir)" in src_exp
    assert 'flags["accumulate_s"] = True' in src_exp
    assert 'flags["train_wipe_s"] = True' in src_exp
    assert 'flags["open_w"] = True' in src_exp
    a = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert a.domain == "door" and not a.use_here_match and not a.use_event_annotate


def test_agent_source_no_synonym_table():
    src = inspect.getsource(agent_mod)
    assert '"push"' not in src and "'push'" not in src


if __name__ == "__main__":
    import tempfile

    test_both_pages_open_and_rare()
    test_skips_stay_skipped()
    test_agent_source_no_synonym_table()
    fns = [
        test_accumulate_a_press_c_tune,
        test_wipe_between_loses_a,
        test_copy_only_presses_on_c,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
