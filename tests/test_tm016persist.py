"""TM.0.16.PERSIST: opt-in continuity candidate under frozen continuity-evidence contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016persist import (
    CANDIDATE_LOCK,
    CELL_IDS,
    CLAIM,
    CONTRACT_PREREG,
    PERSIST_LOCK,
    PREREG_LOCK,
    make_persist,
    run_persist,
    smoke_compat,
    verify_candidate_lock,
    verify_persist_lock,
    verify_prereg_lock,
)
from experiments.run_tm016relate import empty_birth, make_relate
from three_memory.policy import UsePolicy


def test_prereg_and_contract():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["lab"] == "TM.0.16.PERSIST"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["preregistered_claim"] == CLAIM
    assert lock["cell_ids"] == list(CELL_IDS)
    assert "agent_sha" not in lock
    assert "run_tm016persist_sha" not in lock
    assert CONTRACT_PREREG.exists()
    contract = json.loads(CONTRACT_PREREG.read_text(encoding="utf-8"))
    assert contract["earned_next"] is False
    assert contract["ex0s"] is None
    assert contract.get("lab") is None
    assert lock["observation_abi"] == "observe_continuity_mark"
    assert lock["source"] == "experience_continuity"
    assert "exactly one pre-gap apply" in lock["uniqueness"]["rule"]
    assert lock["contradiction"]["reprobe_after"] is True
    assert lock["projection"]["kind"] == "one_hop_P_to_Q_recomputed_at_use_time"


def test_smoke_abi_and_opt_in():
    smoke = smoke_compat()
    assert smoke["genome_016"] is True, smoke
    assert smoke["abi_all"] is True, smoke
    assert smoke["smoke_wrote_nothing"] is True
    assert smoke["flag_default_false"] is True
    assert smoke["finger_continuity_off"] is True
    assert smoke["clone_copies_flag"] is True
    assert smoke["fingerprint_inaccessible"] is True


def test_flag_requires_relate():
    import tempfile

    from experiments.run_tm054 import make
    from three_memory.agent import ThreeMemoryAgent

    with tempfile.TemporaryDirectory() as tmp:
        s = Path(tmp) / "s"
        s.mkdir()
        try:
            make(
                s,
                None,
                UsePolicy(seed=1),
                use_event_annotate=True,
                use_here_match=True,
                use_alias_bind=True,
                use_did_stamp=True,
                use_hyp_survive=True,
                use_bind_match=True,
                use_evidence=True,
                use_compose=True,
                use_context_kappa=True,
                use_acquire_ctx=True,
                use_acquire_skel=True,
                use_acquire_relate=False,
                use_continuity_mark=True,
            )
            raise AssertionError("expected TypeError or ValueError")
        except TypeError:
            # Historical run_tm054.make must not grow this kwarg.
            pass
        except ValueError as e:
            assert "use_continuity_mark requires use_acquire_relate" in str(e)
        try:
            ThreeMemoryAgent(use_continuity_mark=True)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "use_continuity_mark requires use_acquire_relate" in str(e)


def test_make_persist_flag():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_persist(s, UsePolicy(seed=1))
        assert ag.use_continuity_mark is True
        assert ag.use_alias_fingerprint is True
        assert ag.use_acquire_relate is True
        s2 = Path(tmp) / "s2"
        empty_birth(s2)
        off = make_relate(s2, UsePolicy(seed=1))
        assert off.use_continuity_mark is False
        nb = ag.clone_empty()
        assert nb.use_continuity_mark is True


def test_prereg_fixtures_are_wired():
    lock = json.loads(PREREG_LOCK.read_text(encoding="utf-8"))
    c2 = lock["fixtures"]["C2_mark"]["continuity_observations"]
    assert [r["token"] for r in c2] == ["kelm", "norb", "jasp"]
    assert c2[0]["operation"] == "apply"
    assert c2[1]["observed_state"] == "on"
    assert c2[2]["observed_state"] == "off"
    c5 = lock["fixtures"]["C5_contradiction"]["subcases"]
    assert c5["C5_later_state"]["later_observation"]["observed_state"] == "off"
    assert c5["C5_second_read"]["later_observation"]["token"] == "jasp"
    assert c5["C5_second_apply"]["later_observation"]["operation"] == "apply"
    assert lock["uniqueness"]["count_unit"] == "observation_rows"
    assert "same_as" in lock["projection"]["forbidden"]


def test_prior_pins():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    pins = lock["prior_lock_shas"]
    assert pins["continuity_evidence.prereg.lock"] == sha(CONTRACT_PREREG)
    assert pins["gap_wall.lock"] == sha(REPO_ROOT / "docs" / "gap_wall.lock")
    assert pins["alias_finger.lock"] == sha(REPO_ROOT / "docs" / "alias_finger.lock")


def test_persist_battery_and_locks():
    assert CANDIDATE_LOCK.exists(), "candidate.lock must exist before scored CI run"
    v1 = REPO_ROOT / "docs" / "persist.candidate.v1.lock"
    assert v1.exists(), "v1 candidate must remain as the superseded pre-audit pin"
    ok_c, why_c, _ = verify_candidate_lock()
    assert ok_c, why_c
    assert json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8")).get("supersedes") == (
        "docs/persist.candidate.v1.lock"
    )

    summary = run_persist(seed=12345, write_candidate=False, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 9
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["C0_gapwall"]["object_continuity_claim"] is False
    assert by["C2_mark"]["lived_norb"] == "wift"
    assert by["C2_mark"]["skel_untouched"] is True
    assert by["C4_swap"]["behavior_follows_swapped_evidence"] is True
    assert by["C5_contradiction"]["later_state"] is True
    assert by["C5_contradiction"]["second_read"] is True
    assert by["C5_contradiction"]["second_apply"] is True
    assert by["C6_causality"]["reset_rho_retains"] is True
    assert by["C6_causality"]["strip_continuity_only_hold"] is True
    assert by["C6_causality"]["donate_onto_skel_follows_s"] is True

    assert PERSIST_LOCK.exists()
    ok2, why2, _ = verify_persist_lock(summary["rows"])
    assert ok2, why2
    lock = json.loads(PERSIST_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["lab"] == "TM.0.16.PERSIST"


if __name__ == "__main__":
    test_prereg_and_contract()
    test_smoke_abi_and_opt_in()
    test_flag_requires_relate()
    test_make_persist_flag()
    test_prereg_fixtures_are_wired()
    test_prior_pins()
    if CANDIDATE_LOCK.exists():
        test_persist_battery_and_locks()
    print("ok")
