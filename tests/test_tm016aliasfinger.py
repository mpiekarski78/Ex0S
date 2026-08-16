"""TM.0.16.ALIASFINGER: fingerprint candidate under frozen alias-evidence contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016aliasfinger import (
    ALIAS_FINGER_LOCK,
    CANDIDATE_LOCK,
    CELL_IDS,
    CLAIM,
    EVIDENCE_PREREG,
    PREREG_LOCK,
    make_finger,
    run_alias_finger,
    smoke_compat,
    verify_alias_finger_lock,
    verify_prereg_lock,
)
from experiments.run_tm016relate import GENOME_016_LOCK, RELATE_LOCK, verify_genome_016
from experiments.run_tm016aliaswall import ALIAS_WALL_LOCK
from three_memory.policy import UsePolicy


def test_prereg_and_contract():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["lab"] == "TM.0.16.ALIASFINGER"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["preregistered_claim"] == CLAIM
    assert lock["cell_ids"] == list(CELL_IDS)
    assert "agent_sha" not in lock
    assert "run_tm016aliasfinger_sha" not in lock
    assert EVIDENCE_PREREG.exists()
    contract = json.loads(EVIDENCE_PREREG.read_text(encoding="utf-8"))
    assert contract["earned_next"] is False
    assert contract["ex0s"] is None
    assert contract.get("lab") is None
    keys = (contract.get("channel_contract") or {}).get("allowed_keys") or contract.get(
        "channel_allow_keys"
    )
    assert set(keys) == {"alias", "probe_context", "action", "observed_outcome"}
    assert (contract.get("evidence_rule") or {}).get(
        "minimum_independent_probe_contexts",
        (contract.get("evidence_rule") or {}).get("min_independent_contexts"),
    ) == 2


def test_smoke_compat_and_exact_keys():
    smoke = smoke_compat()
    assert smoke["genome_016"] is True, smoke
    assert smoke["relate_16"] is True, smoke
    assert smoke["aliaswall_6"] is True, smoke
    assert smoke["exact_reject_missing"] is True
    assert smoke["exact_reject_extra"] is True
    assert smoke["off_is_noop"] is True
    assert smoke["flag_default_false"] is True
    assert smoke["ambiguous_pair_xp1"] is True
    assert smoke["ambiguous_pair_xp2"] is True
    assert smoke["ambiguous_pair_p1p2"] is False
    assert smoke["ambiguous_clique_none"] is True
    assert smoke["ambiguous_holds"] is True, smoke
    ok, why, _ = verify_genome_016()
    assert ok, why


def test_flag_requires_relate():
    import tempfile

    from experiments.run_tm054 import make
    from three_memory.policy import UsePolicy

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
                use_alias_fingerprint=True,
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "use_alias_fingerprint requires use_acquire_relate" in str(e)


def test_alias_finger_battery_and_locks():
    assert CANDIDATE_LOCK.exists(), "candidate.lock must exist before scored CI run"
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    assert cand["lab"] == "TM.0.16.ALIASFINGER"
    assert cand["earned_next"] is False
    assert cand["ex0s"] is None
    assert cand["cell_ids"] == list(CELL_IDS)
    # Live agent.py may grow later opt-in surfaces; do not require byte-identity
    # with the historical candidate pin. The freeze lock still pins this file.

    summary = run_alias_finger(seed=12345, write_candidate=False, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 7
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["A0_wall"]["motor"] == "hold"
    assert by["A1_weak"]["motor"] == "hold"
    assert by["A1_weak"]["no_pair"] is True
    assert by["A2_convergent"]["lived_bind"] == "wift"
    assert by["A2_convergent"]["strip_hold"] is True
    assert by["A2_convergent"]["skel_untouched"] is True
    assert by["A3_collision"]["no_pair_origins"] is True
    assert by["A4_swap"]["behavior_follows_swapped_evidence"] is True
    assert by["A4_swap"]["pair_kelm_zzzz"] is True
    assert by["A4_swap"]["pair_kelm_jasp"] is False
    assert by["A5_contradiction"]["conflicted_context"] is True
    assert by["A5_contradiction"]["retained_both"] is True
    assert by["A6_causality"]["reset_rho_retains_completion"] is True
    assert by["A6_causality"]["strip_fingerprint_only_hold"] is True
    assert by["A6_causality"]["swap_fingerprint_donors_follow_s"] is True

    assert ALIAS_FINGER_LOCK.exists()
    ok2, why2, _ = verify_alias_finger_lock(summary["rows"])
    assert ok2, why2
    lock = json.loads(ALIAS_FINGER_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["lab"] == "TM.0.16.ALIASFINGER"


def test_prior_pins():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert lock["alias_evidence_prereg_sha"] == sha(EVIDENCE_PREREG)
    assert lock["alias_wall_lock_sha"] == sha(ALIAS_WALL_LOCK)
    assert lock["genome_016_lock_sha"] == sha(GENOME_016_LOCK)
    assert lock["relate_016_lock_sha"] == sha(RELATE_LOCK)


def test_make_finger_flag():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        from experiments.run_tm016relate import empty_birth, make_relate

        s = Path(tmp) / "s"
        empty_birth(s)
        ag = make_finger(s, UsePolicy(seed=1))
        assert ag.use_alias_fingerprint is True
        assert ag.use_acquire_relate is True
        s2 = Path(tmp) / "s2"
        empty_birth(s2)
        off = make_relate(s2, UsePolicy(seed=1))
        assert off.use_alias_fingerprint is False
        nb = ag.clone_empty()
        assert nb.use_alias_fingerprint is True


if __name__ == "__main__":
    test_prereg_and_contract()
    test_smoke_compat_and_exact_keys()
    test_flag_requires_relate()
    test_make_finger_flag()
    test_prior_pins()
    test_alias_finger_battery_and_locks()
    print("ok")
