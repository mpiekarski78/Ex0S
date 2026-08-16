"""Continuity-evidence contract prereg integrity (no mechanism / no battery)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PREREG = REPO_ROOT / "docs" / "continuity_evidence.prereg.lock"
CLAIM = (
    "An opaque post-gap token may be linked provisionally to a pre-gap token only when "
    "an observable pre-gap intervention placed a specified mark or state on the latter "
    "and exactly one post-gap candidate produces the corresponding observable readout. "
    "Token spelling, route position, empty-event skip, episode structure, or sole "
    "candidacy contribute no identity evidence. The resulting continuity hypothesis is "
    "defeasible and must be withdrawn when later observations contradict it."
)
CELL_IDS = [
    "C0_gapwall",
    "C1_weak",
    "C2_mark",
    "C3a_both",
    "C3b_neither",
    "C3c_conflict",
    "C4_swap",
    "C5_contradiction",
    "C6_causality",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_continuity_evidence_prereg():
    assert PREREG.exists()
    lock = json.loads(PREREG.read_text(encoding="utf-8"))
    assert lock["kind"] == "contract_prereg_only"
    assert lock["lab"] is None
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["results"] is None
    assert lock["ex0s_under_test"] == "0.0.004"
    assert lock["not_tm017"] is True
    assert lock["claims_any_organism_satisfies_C2_through_C6"] is False
    assert lock["claim"] == CLAIM
    assert lock["cell_ids"] == CELL_IDS
    assert [c["cell"] for c in lock["battery_contract"]] == CELL_IDS

    ch = lock["channel_contract"]
    assert set(ch["allowed_keys"]) == {
        "token",
        "mark_id",
        "phase",
        "operation",
        "observed_state",
    }
    assert ch["key_rule"] == "exact_set_equality"
    assert ch["operation_phase_lock"] == {"pre_gap": "apply", "post_gap": "read"}
    assert ch["insufficient_without_apply_read_sequence"] is True

    er = lock["evidence_rule"]
    assert er["contradiction_requires_withdrawal"] is True
    assert er["stale_merge_after_contradiction_forbidden"] is True
    assert er["both_verify_refuses_unique"] is True
    assert er["neither_verify_refuses_unique"] is True
    assert er["mutually_conflicting_readouts_refuse_unique"] is True
    assert er["same_property_observed_twice_without_apply_sufficient"] is False
    assert er["bare_mark_id_without_apply_read_sufficient"] is False
    assert "always-HOLD fails earn" in er["admissibility_vs_earn"]["future_earn"]
    assert "randomized independently of marks" in ch["field_semantics"]["token"]

    c2 = next(c for c in lock["battery_contract"] if c["cell"] == "C2_mark")
    assert "admissibility" in c2["honest_outcome"]
    assert "always-HOLD fails earn" in c2["future_earn"]
    c5 = next(c for c in lock["battery_contract"] if c["cell"] == "C5_contradiction")
    assert "withdraw" in c5["honest_outcome"].lower()

    banned = (
        "agent_sha",
        "run_tm016gapwall_sha",
        "make_finger_sha",
        "observe_alias_probe_sha",
        "candidate_lock_sha",
    )
    assert not any(k in lock for k in banned)

    pins = lock["prior_lock_shas"]
    assert pins["gap_wall.lock"] == _sha(REPO_ROOT / "docs" / "gap_wall.lock")
    assert pins["alias_finger.lock"] == _sha(REPO_ROOT / "docs" / "alias_finger.lock")
    assert pins["alias_evidence.prereg.lock"] == _sha(
        REPO_ROOT / "docs" / "alias_evidence.prereg.lock"
    )


if __name__ == "__main__":
    test_continuity_evidence_prereg()
    print("ok")
