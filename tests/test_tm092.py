"""TM.0.9.2: antecedent MATCH — unit tests."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm090 import classify_common as classify090
from experiments.run_tm091 import make as make091
from experiments.run_tm092 import (
    classify_match_battery,
    make,
    permute_pair,
    run_match_battery,
    write_relation_s,
)
from three_memory import agent as agent_mod
from three_memory.dial_env import DialObs
from three_memory.policy import UsePolicy


def test_bind_match_on_092_off_before():
    assert make(Path("/tmp/tm092_092"), None, UsePolicy(seed=1), enabled=False).use_bind_match
    assert not make091(Path("/tmp/tm092_091"), None, UsePolicy(seed=1), enabled=False).use_bind_match
    assert not make054(Path("/tmp/tm092_054"), None, UsePolicy(seed=1), enabled=False).use_bind_match
    door = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert door.domain == "door" and not door.use_bind_match
    src = inspect.getsource(agent_mod)
    for word in ("push", "flim", "zorg", "blen", "nork", "wibble", "tork"):
        assert f'"{word}"' not in src
    assert "bind_token_id" not in src
    pol = inspect.getsource(__import__("three_memory.policy", fromlist=["UsePolicy"]))
    assert "bind_token_id" not in pol
    assert UsePolicy.n_feat == 2


def test_090_confounds_bind_match_smuggle():
    import experiments.run_tm090 as tm090

    saved = tm090._classify_common082
    tm090._classify_common082 = lambda m: None
    try:
        label, why = classify090(
            {
                "use_count_search": True,
                "count_search": True,
                "use_bind_match": True,
            }
        )
    finally:
        tm090._classify_common082 = saved
    assert label == "Confound"
    assert "bind" in why.lower()


def test_obs_tokens_not_in_cortex_vector():
    a = DialObs(at_b=True)
    b = DialObs(at_b=True, tokens=frozenset({"flim"}))
    assert list(a.vector()) == list(b.vector())


def test_match_battery_force_use_permuted(tmp_path: Path):
    policy = UsePolicy(seed=7, lr=0.2)
    for seed in (11, 22, 33):
        spec = permute_pair(seed)
        assert spec["x"] != spec["y"]
        assert {spec["m1"], spec["m2"]} == {"press", "tune"}
        bat = run_match_battery(policy, spec, tmp_path / f"s{seed}", force_use=True)
        assert bat["classification"] == "Store-works", bat["rationale"]


def test_match_gate_is_boolean_not_token(tmp_path: Path):
    write_relation_s(tmp_path / "s", [("a", "quop", "press"), ("b", "daff", "tune")])
    ag = make(tmp_path / "s", None, UsePolicy(seed=1), enabled=True, force_use=True)
    obs = DialObs(at_b=True, tokens=frozenset({"quop"}))
    present = any(ag._bind_in_stream(r, obs) for r in ag.store.records())
    assert present is True
    obs_y = DialObs(at_b=True, tokens=frozenset({"daff"}))
    assert ag._bind_in_stream(next(r for r in ag.store.records() if r.tags.get("bind") == "daff"), obs_y)
    assert not ag._bind_in_stream(next(r for r in ag.store.records() if r.tags.get("bind") == "quop"), obs_y)
    feat_name = "bind_present_in_current_stream"
    assert feat_name in inspect.getsource(agent_mod)


def test_classify_match_rejects_blind_copy():
    spec = {"x": "aa", "y": "bb", "m1": "press", "m2": "tune"}
    cells = {
        "same_x": {"action_name": "press"},
        "same_y": {"action_name": "tune"},
        "cross_x": {"action_name": "tune"},
        "cross_y": {"action_name": "press"},
        "empty": {"action_name": "hold"},
    }
    label, why = classify_match_battery(cells, spec)
    assert label == "Fail"
    assert "hold" in why.lower() or "only" in why.lower()


if __name__ == "__main__":
    test_bind_match_on_092_off_before()
    test_090_confounds_bind_match_smuggle()
    test_obs_tokens_not_in_cortex_vector()
    test_classify_match_rejects_blind_copy()
    with tempfile.TemporaryDirectory() as d:
        test_match_battery_force_use_permuted(Path(d))
        test_match_gate_is_boolean_not_token(Path(d) / "gate")
    print("ok")
