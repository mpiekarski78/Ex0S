"""TM.0.11.FAMILY: freeze, first-hop D/F, transitive shortcuts, G upstream hash."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make as make011
from experiments.run_tm011family import (
    DEVELOP,
    HOLDOUT,
    generate_family_C,
    generate_family_D,
    generate_family_F,
    generate_world,
    has_edge,
    no_transitive_shortcuts,
    run_family,
    run_one,
    transitive_forbidden,
    verify_freeze,
    write_freeze_lock,
    write_world_s,
)
from experiments.run_tm094 import make as make094
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy


def test_frozen_011_compose_genome():
    write_freeze_lock()
    ok, why, snap = verify_freeze()
    assert ok, why
    assert "family_E_generator_sha" in snap
    assert "family_F_generator_sha" in snap
    assert "family_G_generator_sha" in snap
    assert "scorer_sha" in snap
    assert "seed_list_sha" in snap
    ag = make011(Path("/tmp/tm011family_011"), None, UsePolicy(seed=1), enabled=False)
    assert ag.use_compose and ag.use_evidence and ag.use_bind_match
    assert not make094(Path("/tmp/tm011family_094"), None, UsePolicy(seed=1), enabled=False).use_compose
    assert UsePolicy.n_feat == 2
    src = inspect.getsource(agent_mod)
    for banned in ("use_two_hop", "use_three_hop", "MAX_HOPS", "use_lookahead"):
        assert banned not in src


def test_holdout_split_and_depths():
    assert DEVELOP == ("A", "B", "C", "D")
    assert HOLDOUT == ("E", "F", "G")
    depths = {fam: generate_world(fam, 12345, 0).depth for fam in DEVELOP + HOLDOUT}
    assert depths["A"] == 2
    assert depths["B"] == 3
    assert depths["E"] == 4
    for fam in HOLDOUT:
        assert generate_world(fam, 1, 0).holdout


def test_transitive_forbidden_3hop():
    chain = ["x", "a", "b", "press"]
    forbid = set(transitive_forbidden(chain))
    assert ("x", "b") in forbid
    assert ("x", "press") in forbid
    assert ("a", "press") in forbid
    assert ("x", "a") not in forbid
    assert ("a", "b") not in forbid


def test_first_hop_df_not_lookahead():
    d = generate_family_D(99, 0)
    by_role = {r.role: r for r in d.relations}
    assert by_role["xy"].init[0] > by_role["xz"].init[0]
    assert by_role["zm"].init[0] > by_role["ym"].init[0]  # downstream trap
    f = generate_family_F(99, 0)
    by_f = {r.role: r for r in f.relations}
    assert by_f["xy"].init == by_f["xz"].init == (1, 0)
    assert by_f["zm"].init[0] > by_f["ym"].init[0]


def test_no_shortcut_helper(tmp_path: Path):
    w = generate_world("B", 7, 0)
    write_world_s(tmp_path, w.relations)
    assert no_transitive_shortcuts(tmp_path, w.chain)
    # Plant a forbidden skip edge.
    skip = w.chain[0], w.chain[2]
    from three_memory.symbols import record_to_tagfile

    (tmp_path / "bad.tag").write_text(
        record_to_tagfile(
            "bad",
            {
                "bind": skip[0],
                "did": skip[1],
                "here": "chb",
                "support": 1,
                "contradiction": 0,
                "wins": 1,
                "losses": 0,
                "trials": 1,
                "hyp": "supported",
            },
        ),
        encoding="utf-8",
    )
    assert has_edge(tmp_path, skip[0], skip[1])
    assert not no_transitive_shortcuts(tmp_path, w.chain)


def test_c_junk_wrong_motor():
    w = generate_family_C(12345, 0)
    by = {r.role: r for r in w.relations}
    assert by["irr"].did != by["ym"].did


def test_smoke_does_not_stamp_ex0s():
    summary = run_family(seed=11, per_family=1, births=1, workers=1)
    assert summary["n_worlds"] == 7
    assert summary["genome_ok"]
    assert summary["solved_frac"] == 1.0, summary["families"]
    assert summary["ex0s"] is None
    assert not summary["earned_frozen_composition"]
    assert summary["intervention"]["full_battery"] is False


def test_lookahead_trap_holds():
    """Equal first-hop + strong downstream must HOLD — not planning."""
    from experiments.run_tm011family import Rel, probe_cue, write_world_s
    from three_memory.policy import UsePolicy

    with tempfile.TemporaryDirectory() as d:
        dest = Path(d)
        write_world_s(
            dest,
            [
                Rel("n01", "aa", "bb", "xy", (1, 0)),
                Rel("n02", "aa", "cc", "xz", (1, 0)),
                Rel("n03", "bb", "press", "ym", (1, 0)),
                Rel("n04", "cc", "tune", "zm", (1000, 0)),
            ],
        )
        out = probe_cue(UsePolicy(seed=1), dest, 1, "aa")
        assert out["action_name"] == "hold"
        assert out.get("evidence_tie") or out.get("compose_hold")


def test_g_upstream_hash_stable(tmp_path: Path):
    job = {
        "family": "G",
        "seed": 42,
        "birth": 0,
        "dest": str(tmp_path / "g"),
        "genome_ok": True,
    }
    row = run_one(job)
    assert row["solved"], row.get("missing")
    assert row["upstream_before"] and row["upstream_before"] == row["upstream_after"]
    assert row["measures"]["upstream_stability"] is True
    assert row["measures"]["revise_downstream"] is True


if __name__ == "__main__":
    test_frozen_011_compose_genome()
    test_holdout_split_and_depths()
    test_transitive_forbidden_3hop()
    test_first_hop_df_not_lookahead()
    test_c_junk_wrong_motor()
    test_lookahead_trap_holds()
    with tempfile.TemporaryDirectory() as d:
        test_no_shortcut_helper(Path(d))
        test_g_upstream_hash_stable(Path(d) / "gjob")
    test_smoke_does_not_stamp_ex0s()
    print("ok")
