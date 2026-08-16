"""TM.0.16.GAPWALL: continuity capacity wall regression and freeze checks."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016aliasfinger import (
    ALIAS_FINGER_LOCK,
    CANDIDATE_LOCK as ALIAS_FINGER_CANDIDATE_LOCK,
    EVIDENCE_PREREG as ALIAS_EVIDENCE_PREREG,
)
from experiments.run_tm016gapwall import (
    AGENT_PY,
    CELL_IDS,
    GAP_WALL_LOCK,
    PREREG_LOCK,
    fresh_world,
    run_gap_wall,
    verify_gap_wall_lock,
    verify_prereg_lock,
)
from experiments.run_tm016relate import GENOME_016_LOCK, RELATE_LOCK
from three_memory.policy import UsePolicy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg_and_prior_pins() -> None:
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["lab"] == "TM.0.16.GAPWALL"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["not_tm017"] is True
    assert lock["cell_ids"] == list(CELL_IDS)
    assert lock["organism"]["factory"].endswith("make_finger")
    assert lock["organism"]["use_alias_fingerprint"] is True
    assert lock["organism"]["agent_edits_permitted"] is False
    serialized = json.dumps(lock, sort_keys=True)
    assert '"run_tm016gapwall_sha"' not in serialized
    assert '"agent_sha"' not in serialized

    paths = {
        "alias_finger.lock": ALIAS_FINGER_LOCK,
        "alias_finger.candidate.lock": ALIAS_FINGER_CANDIDATE_LOCK,
        "alias_evidence.prereg.lock": ALIAS_EVIDENCE_PREREG,
        "genome_016.lock": GENOME_016_LOCK,
        "relate_016.lock": RELATE_LOCK,
    }
    for name, path in paths.items():
        assert lock["prior_lock_shas"][name] == sha(path)


def test_fixture_contracts() -> None:
    lock = json.loads(PREREG_LOCK.read_text(encoding="utf-8"))
    fixtures = lock["fixtures"]
    assert fixtures["G1_empty_skip"]["episodes"] == [
        [["a"], [], ["b"], ["c"]]
    ]
    assert fixtures["G1_empty_skip"]["scorer"]["empty_event_skip_semantics"] is True
    assert fixtures["G1_empty_skip"]["scorer"]["object_continuity_claim"] is False
    assert fixtures["G2_episode_gap"]["episodes"] == [[["a"]], [["a"]]]
    assert fixtures["G4_one_reappear"]["scorer"]["measurement_only"] is True
    assert fixtures["G5_two_reappear"]["episodes"] == [
        [["a"], [], ["u1", "u2"]]
    ]
    assert fixtures["G5_two_reappear"]["scorer"]["fail_closed_on_unique_peer"] is True


def test_isolation_and_frozen_finger_factory() -> None:
    with tempfile.TemporaryDirectory(prefix="tm016gapwall_test_") as tmp:
        root = Path(tmp)
        s0, ag0 = fresh_world(root, "first", UsePolicy(seed=1))
        s1, ag1 = fresh_world(root, "second", UsePolicy(seed=1))
        assert s0 != s1
        assert not list(s0.glob("*.tag"))
        assert not list(s1.glob("*.tag"))
        assert ag0 is not ag1
        assert ag0.use_alias_fingerprint is True
        assert ag1.use_alias_fingerprint is True
        assert ag0.use_acquire_relate is True


def test_gapwall_battery_and_freeze() -> None:
    summary = run_gap_wall(seed=12345, write_lock=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 6
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None

    by = {row["cell"]: row for row in summary["rows"]}
    assert by["G0_adjacent"]["unique_route_completion"] is True
    assert by["G0_adjacent"]["continuity_not_at_issue"] is True

    g1 = by["G1_empty_skip"]
    assert g1["gap_why"] == "empty_visible"
    assert g1["gap_wrote"] == g1["gap_updated"] == 0
    assert g1["empty_event_skip_semantics"] is True
    assert g1["object_continuity_claim"] is False

    g2 = by["G2_episode_gap"]
    assert g2["support_a_a"] == 0
    assert g2["cross_episode_bridge"] is False
    assert g2["inherited_episode_frontier"] is False

    g3 = by["G3_distractor"]
    assert g3["support_a_d"] == g3["support_d_a"] == 1
    assert g3["support_a_a"] == 0
    assert g3["distractor_edges_authored"] is True
    assert g3["direct_continuity_privileged"] is False

    g4 = by["G4_one_reappear"]
    assert g4["measurement_only"] is True
    assert g4["measured_outcome"] == "unique"
    assert g4["honest_label"] == "single_candidate_selected_by_empty_event_skip"
    assert g4["object_continuity_claim"] is False

    g5 = by["G5_two_reappear"]
    assert g5["motor"] == "hold"
    assert g5["winner_a"] is None
    assert g5["lived_bind"] is None
    assert g5["evidence_tie"] is True
    assert g5["unique_assignment"] is False
    assert g5["fail_closed_on_unique_peer"] is True

    assert GAP_WALL_LOCK.exists()
    ok, why, freeze = verify_gap_wall_lock(summary["rows"])
    assert ok, why
    assert freeze["earned_next"] is False
    assert freeze["ex0s"] is None
    assert freeze["observed"]["G4_measured_outcome"] == "unique"
    assert freeze["observed"]["G5_irreducible_ambiguity"] is True


def test_no_agent_or_prior_freeze_drift() -> None:
    alias_freeze = json.loads(ALIAS_FINGER_LOCK.read_text(encoding="utf-8"))
    gap_freeze = json.loads(GAP_WALL_LOCK.read_text(encoding="utf-8"))
    assert alias_freeze["agent_sha"] == sha(AGENT_PY)
    assert gap_freeze["agent_sha"] == sha(AGENT_PY)
    assert gap_freeze["prior_lock_shas"]["alias_finger.lock"] == sha(ALIAS_FINGER_LOCK)
    assert gap_freeze["gap_wall_prereg_sha"] == sha(PREREG_LOCK)


if __name__ == "__main__":
    test_prereg_and_prior_pins()
    test_fixture_contracts()
    test_isolation_and_frozen_finger_factory()
    test_gapwall_battery_and_freeze()
    test_no_agent_or_prior_freeze_drift()
    print("ok")
