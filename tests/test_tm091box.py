"""TM.0.9.BOX: boxed-policy leakage control — unit tests."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import make as make054
from experiments.run_tm090 import make as make090
from experiments.run_tm091 import make as make091
from experiments.run_tm091box import (
    READOUT_FACT_ID,
    classify_separation,
    classify_transfer,
    clone_policy,
    birth_policy,
    project_to_readout,
    score_box_measures,
    verify_genome_lock,
    verify_protocol_lock,
    write_neutral_readout,
    wiki_nonce,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy
from three_memory.symbols import record_to_tagfile


def test_genome_and_protocol_locks():
    ok, why = verify_genome_lock()
    assert ok, why
    ok, why = verify_protocol_lock(n_train=500, max_steps=32)
    assert ok, why
    lock = json.loads((REPO_ROOT / "docs" / "genome_091.lock").read_text(encoding="utf-8"))
    assert lock["cortex_weight_hash"].startswith("a485b26b")
    assert "n_train" not in lock  # childhood is protocol, not genome


def test_091_make_unchanged_and_flags():
    assert make091(Path("/tmp/tm091box_091"), None, UsePolicy(seed=1), enabled=False).use_hyp_survive
    assert not make090(Path("/tmp/tm091box_090"), None, UsePolicy(seed=1), enabled=False).use_hyp_survive
    assert not make054(Path("/tmp/tm091box_054"), None, UsePolicy(seed=1), enabled=False).use_hyp_survive
    door = agent_mod.ThreeMemoryAgent(use_policy=UsePolicy(seed=1), store_enabled=False, cortex_seed=1337)
    assert door.domain == "door" and not door.use_hyp_survive
    src = inspect.getsource(agent_mod)
    for word in ("push", "flim", "zorg", "blen", "nork", "wibble", "tork"):
        assert f'"{word}"' not in src


def test_paired_birth_clone():
    b = birth_policy(seed=7, lr=0.2)
    p1 = clone_policy(b)
    p2 = clone_policy(b)
    assert p1.weight_hash() == p2.weight_hash() == b.weight_hash()
    assert p1.n_updates == p2.n_updates == 0
    assert p1.lr == p2.lr == 0.2


def test_projection_canonical_and_same_filename(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    tags = {
        "bind": "flim",
        "did": "press",
        "hyp": "supported",
        "trials": 3,
        "wins": 3,
        "losses": 0,
        "source": "W->S",
        "w0": "argon",
        "w1": "flim",
        "w2": "cha",
        "when": 99,
    }
    (raw / "p99.tag").write_text(record_to_tagfile("p99", tags), encoding="utf-8")
    tags2 = {
        "bind": "flim",
        "did": "tune",
        "hyp": "supported",
        "trials": 1,
        "source": "W->S",
        "w0": "flim",
        "w1": "chc",
        "when": 12,
    }
    raw2 = tmp_path / "raw2"
    raw2.mkdir()
    (raw2 / "p98.tag").write_text(record_to_tagfile("p98", tags2), encoding="utf-8")

    d1 = tmp_path / "s1"
    d2 = tmp_path / "s2"
    r1 = project_to_readout(raw, d1, bind="flim", did="press")
    r2 = project_to_readout(raw2, d2, bind="flim", did="tune")
    assert r1["ok"] and r2["ok"]
    t1 = (d1 / f"{READOUT_FACT_ID}.tag").read_text(encoding="utf-8")
    t2 = (d2 / f"{READOUT_FACT_ID}.tag").read_text(encoding="utf-8")
    assert (d1 / f"{READOUT_FACT_ID}.tag").name == (d2 / f"{READOUT_FACT_ID}.tag").name
    assert "bind=flim" in t1 and "did=press" in t1 and "here=chb" in t1 and "w0=flim" in t1
    assert "bind=flim" in t2 and "did=tune" in t2 and "here=chb" in t2
    for bad in ("hyp=", "trials=", "source=", "when=", "argon", "cha", "p99", "p98"):
        assert bad not in t1
        assert bad not in t2
    # bind and did unchanged from donor selection
    assert r1["bind"] == "flim" and r1["did"] == "press"
    assert r2["bind"] == "flim" and r2["did"] == "tune"


def test_worlds_differ():
    w1 = dict(wiki_nonce("W1", include_a=True, include_c=True))
    w2 = dict(wiki_nonce("W2", include_a=True, include_c=True))
    w3 = dict(wiki_nonce("W3", include_a=True, include_c=True))
    assert "flim" in w1["p99.md"].lower() and "zorg" in w1["p98.md"].lower()
    assert "zorg" in w2["p99.md"].lower() and "flim" in w2["p98.md"].lower()
    assert "blen" in w3["q17.md"].lower() and "nork" in w3["r43.md"].lower()
    assert "p99.md" not in w3 and "q17.md" in w3


def _cell(**kwargs):
    base = {
        "genome_ok": True,
        "protocol_ok": True,
        "cortex_unchanged": True,
        "acquisition_ok": True,
        "P1": {
            "empty": {"action_name": "hold"},
            "neutral_PRESS": {"action_name": "hold"},
            "neutral_TUNE": {"action_name": "hold"},
            "S1": {"action_name": "press"},
            "S2": {"action_name": "tune"},
        },
        "P2": {
            "empty": {"action_name": "hold"},
            "neutral_PRESS": {"action_name": "hold"},
            "neutral_TUNE": {"action_name": "hold"},
            "S1": {"action_name": "press"},
            "S2": {"action_name": "tune"},
        },
    }
    base.update(kwargs)
    return base


def test_classify_store_works():
    label, why = classify_separation(_cell())
    assert label == "Store-works"
    assert "separation" in why.lower() or "HOLD" in why or "frozen" in why.lower()


def test_classify_confound_leak():
    c = _cell()
    c["P1"]["empty"] = {"action_name": "press"}
    c["P2"]["empty"] = {"action_name": "tune"}
    label, why = classify_separation(c)
    assert label == "Confound"
    assert "covaries" in why.lower() or "mapping" in why.lower()


def test_classify_control_fail_drift():
    c = _cell()
    c["P1"]["empty"] = {"action_name": "press"}
    c["P2"]["empty"] = {"action_name": "press"}
    label, why = classify_separation(c)
    assert label == "Control Fail"
    assert "bias" in why.lower() or "independent" in why.lower()


def test_classify_inconclusive():
    c = _cell()
    c["P1"]["empty"] = {"action_name": "press"}
    c["P2"]["empty"] = {"action_name": "hold"}
    label, why = classify_separation(c)
    assert label == "Inconclusive"


def test_classify_confound_follows_training_world():
    c = _cell()
    c["P1"]["S2"] = {"action_name": "press"}  # kept training PRESS instead of donor TUNE
    c["P2"]["S1"] = {"action_name": "tune"}
    label, why = classify_separation(c)
    assert label == "Confound"
    assert "donor" in why.lower() or "training" in why.lower()


def test_classify_fail_cannot_use_s():
    c = _cell()
    c["P1"]["S1"] = {"action_name": "hold"}
    label, why = classify_separation(c)
    assert label == "Fail"


def test_transfer_separate_from_separation():
    label, why = classify_transfer(
        {
            "transfer": {
                "P1_S3": {"action_name": "hold"},
                "P2_S3": {"action_name": "hold"},
                "P3_S1": {"action_name": "press"},
                "P3_S2": {"action_name": "tune"},
            }
        }
    )
    assert label == "Fail"
    # Separation still Store-works on a clean lethal pair
    sep, _ = classify_separation(_cell())
    assert sep == "Store-works"


def test_missing_s3_is_unevaluable_not_transfer_fail():
    label, why = classify_transfer(
        {
            "s3_ok": False,
            "transfer": {
                "P1_S3": {"action_name": "hold"},
                "P2_S3": {"action_name": "hold"},
                "P3_S1": {"action_name": "press"},
                "P3_S2": {"action_name": "tune"},
            },
        }
    )
    assert label == "Unevaluable"
    assert "s3" in why.lower() or "acquisition" in why.lower()


def test_measures_split_leakage_from_relevance():
    c = _cell()
    c["P1"]["neutral_PRESS"] = {"action_name": "press"}
    c["P2"]["neutral_PRESS"] = {"action_name": "press"}
    c["P1"]["neutral_TUNE"] = {"action_name": "tune"}
    c["P2"]["neutral_TUNE"] = {"action_name": "tune"}
    c["s3_ok"] = True
    c["transfer"] = {
        "P1_S3": {"action_name": "press"},
        "P2_S3": {"action_name": "press"},
        "P3_S1": {"action_name": "press"},
        "P3_S2": {"action_name": "tune"},
    }
    m = score_box_measures(c)
    assert m["world_fact_leakage"] == "Not observed"
    assert m["counterfactual_donor"] == "Pass"
    assert m["neutral_relevance"] == "Fail"
    assert m["transfer"] == "Pass"
    assert m["w3_acquired"] is True
    sep, _ = classify_separation(c)
    assert sep == "Control Fail"


def test_p0_press_not_fail_for_separation():
    # P0 is not part of classify_separation; only P1/P2.
    sep, _ = classify_separation(_cell())
    assert sep == "Store-works"


if __name__ == "__main__":
    test_genome_and_protocol_locks()
    test_091_make_unchanged_and_flags()
    test_paired_birth_clone()
    test_worlds_differ()
    test_classify_store_works()
    test_classify_confound_leak()
    test_classify_control_fail_drift()
    test_classify_inconclusive()
    test_classify_confound_follows_training_world()
    test_classify_fail_cannot_use_s()
    test_transfer_separate_from_separation()
    test_missing_s3_is_unevaluable_not_transfer_fail()
    test_measures_split_leakage_from_relevance()
    test_p0_press_not_fail_for_separation()
    with tempfile.TemporaryDirectory() as d:
        test_projection_canonical_and_same_filename(Path(d))
    print("ok")
