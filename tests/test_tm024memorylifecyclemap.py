"""TM.0.24.MEMORYLIFECYCLEMAP provenance. Runner-only. W1 closed. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
CVG_DEC = "6de34e295b54ef51c2684b0ce1cf7295064300db75e9e7ee258c7bb95665072d"
CVG_DEV = "7aa13d1bbd172cadb565d5d6cafd27d91f47b120491069a00a34111a6d505ca6"
CVG_ADD = "04a5e91cdc23839c9e3c954dc8c921f902c092de14acd906ada7caa678a6b083"
CVG_RUNNER_PY = "232cffa23619de1fcdbde7b8c82fc3de8e1c2fbe84a014a40bc27f3723cbbcf6"
TB_RUNNER_PY = "db6c4e73cac57dd79cb86fcfa371fee1e2fc5753bf1d03402faea690ff4de551"
WG_RUNNER_PY = "b210cc621ccd93e016483e3d9d8dc8adbc284eb8fabc01a4b15bbb4ecb1f4d31"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593"
RUNNER_SHA = "edec8809938f3f1ab77948feb3661bea4fc3e6bb1abf573a81089ae628dfc974"
DEV_LOCK_SHA = None
DEV_MANIFEST_SHA = None
DECISION_SHA = None
EXPECTED_N_CELLS = 140
ARMS = ("L0", "L1", "L2", "L3", "L4")
PHASES_OK = (
    "live_repetition_sufficient",
    "fifo_replay_sufficient",
    "content_addressed_invalidation_supported",
    "replacement_not_causal",
    "episode_reinstatement_match_failure",
    "covariance_memory_ceiling_only",
    "memory_lifecycle_insufficient",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_memorylifecyclemap_contract.md",
        "docs/lineage_memorylifecyclemap.prereg.lock",
        "docs/lineage_memorylifecyclemap.isolation.lock",
        "docs/lineage_convergencemap.decision.lock",
        "docs/lineage_convergencemap.decision.addendum.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024memorylifecyclemap.py",
        "experiments/run_tm024convergencemap.py",
        "experiments/run_tm024tracebridge.py",
        "experiments/run_tm024writegeom.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_memorylifecyclemap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["act_score_mode"] == "query"
    assert prereg["w1_resurrected"] is False
    assert prereg["stop_isolated_write_formulas"] is True
    assert prereg["match"]["radius"] == 0.05
    assert prereg["match"]["threshold"] == 0.05
    assert prereg["match"]["role"] == "preregistered_match_radius"
    assert prereg["match"]["not_same_cue_ceiling"] is True
    assert prereg["match"]["distributions_overlap"] is True
    assert prereg["match"]["not_cue_name"] is True
    assert prereg["match"]["tie_break"] == ["distance", "age", "slot_index"]
    assert prereg["episode_store"]["n_slots"] == 8
    assert prereg["episode_store"]["max_state_scalars"] == 512
    assert prereg["episode_store"]["no_cue_string"] is True
    assert prereg["arms"]["L0"]["store"] is False
    assert prereg["arms"]["L1"]["policy"] == "fifo"
    assert prereg["arms"]["L2"]["policy"] == "replace"
    assert prereg["arms"]["L3"]["policy"] == "retain_stale"
    assert prereg["arms"]["L3"]["refuse_contradictory_replacement"] is True
    assert prereg["contradiction"]["match_then_refuse_replacement_on_L3"] is True
    assert prereg["arms"]["L4"]["ceiling_only"] is True
    assert prereg["phased_contract"]["monotonic_retention_not_acquire_fail"] is True
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    assert prereg["expected_kind_counts"] == {
        "acquire": 60,
        "stable": 60,
        "twin": 10,
        "eco": 5,
        "spec": 5,
    }
    assert prereg["domains"]["DEV"] == "TM024.MEMORYLIFECYCLEMAP.DEV."
    assert prereg["score_reserved_unopened"] is True
    assert prereg["declared_budget_if_later_authorized"]["opened"] is False
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock") == CVG_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.dev.lock") == CVG_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock") == CVG_ADD
    assert sha(REPO_ROOT / "experiments" / "run_tm024convergencemap.py") == CVG_RUNNER_PY
    assert sha(REPO_ROOT / "experiments" / "run_tm024tracebridge.py") == TB_RUNNER_PY
    assert sha(REPO_ROOT / "experiments" / "run_tm024writegeom.py") == WG_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    add = json.loads((REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock").read_text(encoding="utf-8"))
    assert add["historical_code"] == "oracle_separability_not_operationally_reachable"
    assert add["interpret_as"] == (
        "oracle_separability_not_operationally_reachable_under_frozen_monotonic_retention_and_reversal_contract"
    )
    assert add["rewrite_historical_decision"] is False
    assert add["next"] == "TM.0.24.MEMORYLIFECYCLEMAP"
    runner_p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py") == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["act_score_mode"] == "query"
        assert runner["w1_resurrected"] is False
        assert runner["l4_ceiling_only"] is True
        assert runner["match_radius"] == 0.05
        assert runner["match_role"] == "preregistered_match_radius"
        assert runner["l3_refuse_contradictory_replacement"] is True
        assert runner["max_state_scalars"] == 512


def _cell(
    arm: str,
    kind: str,
    n_cues: int,
    *,
    passed: bool,
    world: int = 0,
    order: str = "A_then_B",
    reversal_miss: bool = False,
) -> dict[str, object]:
    out: dict[str, object] = {
        "arm": arm,
        "kind": kind,
        "n_cues": n_cues,
        "order": order,
        "world": world,
        "passed": passed,
        "ranking_ok": passed,
        "l4_ceiling_only": arm == "L4",
        "id": f"{kind}|{arm}|c{n_cues}|{order}|w{world}",
    }
    if reversal_miss:
        out["matcher"] = {"reversal_matcher_miss": True, "reversal_match_recall": 0.0}
    return out


def _arm_cells(
    arm: str,
    *,
    acquire_all: bool = False,
    acquire8: bool = False,
    twin: bool = False,
    stable: bool = False,
    plasticity: bool = False,
    specificity: bool = False,
    reversal_miss: bool = False,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for n in (2, 4, 8):
        acq = bool(acquire_all or (acquire8 and n == 8))
        for wi in range(2):
            for order in ("A_then_B", "B_then_A"):
                cells.append(_cell(arm, "acquire", n, passed=acq, world=wi, order=order))
                cells.append(_cell(arm, "stable", n, passed=stable, world=wi, order=order))
    for order in ("A_then_B", "B_then_A"):
        cells.append(_cell(arm, "twin", 2, passed=twin, world=1, order=order))
    cells.append(_cell(arm, "eco", 2, passed=plasticity, reversal_miss=reversal_miss))
    cells.append(_cell(arm, "spec", 4, passed=specificity, reversal_miss=reversal_miss))
    return cells


def _full(**arm_kwargs: dict[str, bool]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for arm in ARMS:
        cells.extend(_arm_cells(arm, **(arm_kwargs.get(arm) or {})))
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024memorylifecyclemap import EXPECTED_N_CELLS as N
    from experiments.run_tm024memorylifecyclemap import _decision, load_prereg

    p = load_prereg()

    def code(**arm_kwargs: dict[str, bool]) -> str:
        cells = _full(**arm_kwargs)
        assert len(cells) == N
        c, _then, _extra = _decision(cells, p)
        return c

    four = {
        "acquire_all": True,
        "acquire8": True,
        "twin": True,
        "stable": True,
        "plasticity": True,
        "specificity": True,
    }
    assert code(L0=four) == "live_repetition_sufficient"
    assert code(L0=four, L1=four) == "live_repetition_sufficient"
    assert code(L1=four) == "fifo_replay_sufficient"
    assert code(L1=four, L2=four, L3=four) == "fifo_replay_sufficient"
    assert (
        code(L2=four, L3={"plasticity": False}) == "content_addressed_invalidation_supported"
    )
    assert code(L2=four, L3=four) == "replacement_not_causal"
    assert (
        code(L2={"reversal_miss": True}) == "episode_reinstatement_match_failure"
    )
    assert code(L4=four, L2={"reversal_miss": True}) == "episode_reinstatement_match_failure"
    assert code(L4=four) == "covariance_memory_ceiling_only"
    assert code() == "memory_lifecycle_insufficient"
    assert code(L4={"acquire8": True}) == "memory_lifecycle_insufficient"


def test_episode_store_match_and_no_cue() -> None:
    from experiments.run_tm024eligmap import unit_or_zero
    from experiments.run_tm024memorylifecyclemap import (
        ALLOWED_EPISODE_FIELDS,
        EpisodeStore,
        assert_episode_legal,
    )
    from experiments.run_tm024writegeom import capacity_world

    world = capacity_world(0, "TM024.MEMORYLIFECYCLEMAP.TEST.", n_cues=2, n_handles=2)
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(np.arange(64, dtype=np.float64)[::-1] + 7.0)
    h1, h2 = world["handles"][0], world["handles"][1]
    legal = {"p1": x, "handle": h1, "adv": 1.0, "age": 1, "version": 1, "valid": True}
    assert set(legal) == set(ALLOWED_EPISODE_FIELDS)
    assert_episode_legal(legal, world)
    try:
        assert_episode_legal({**legal, "cue": world["cues"][0]}, world)
    except RuntimeError as e:
        assert "illegal" in str(e) or "cue" in str(e).lower()
    else:
        raise AssertionError("cue field must be refused")

    repl = EpisodeStore(policy="replace", match_l2=0.05)
    repl.write(x, h1, 1.0, world=world)
    assert repl.nearest(x) == 0
    assert repl.nearest(y) is None
    repl.write(x, h1, -1.0, world=world)
    assert repl.stats()["n_invalidated"] == 1
    assert repl.stats()["n_valid"] == 1
    repl.write(y, h2, 1.0, world=world)
    assert repl.stats()["n_valid"] == 2

    fifo = EpisodeStore(policy="fifo", match_l2=0.05)
    fifo.write(x, h1, 1.0, world=world)
    fifo.write(x, h1, -1.0, world=world)
    assert fifo.stats()["n_invalidated"] == 0
    assert fifo.stats()["n_occupied"] == 2

    stale = EpisodeStore(policy="retain_stale", match_l2=0.05)
    stale.write(x, h1, 1.0, world=world, cue="a")
    stale.write(x, h1, -1.0, world=world, cue="a", reversal=True)
    assert stale.stats()["n_invalidated"] == 0
    assert stale.stats()["n_refused"] == 1
    assert stale.stats()["n_evicted"] == 0
    assert stale.stats()["n_valid"] == 1
    assert stale.stats()["n_p1_scalars"] == 64
    assert stale.ledger.summary()["reversal_matcher_miss"] is False
    assert stale.ledger.summary()["n_unique_match"] == 1
    q = fifo.query(x)
    assert q["kind"] == "multiple_match"
    assert q["match_i"] == 0
    assert q["n_ties"] == 2
    assert "cue" not in stale.slots[0]


def test_exposure_grid() -> None:
    from experiments.run_tm024memorylifecyclemap import (
        EXPECTED_N_CELLS,
        assert_cell_coverage,
        cell_id,
        expected_cell_ids,
    )

    ids = expected_cell_ids()
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS
    assert cell_id("acquire", "L2", 8, "A_then_B", 0) == "acquire|L2|c8|A_then_B|w0"
    cells = _full()
    for c in cells:
        c.setdefault("domain", "TM024.MEMORYLIFECYCLEMAP.DEV.")
        c.setdefault("store", None)
    assert assert_cell_coverage(cells)


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_memorylifecyclemap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "L0" in text and "L4" in text
    assert "not a neural amendment" in text.lower()
    assert "0.05" in text
    assert "preregistered match radius" in text
    assert "512" in text
    assert "W1" in text


def test_smoke() -> None:
    from experiments.run_tm024memorylifecyclemap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["w1_resurrected"] is False
    assert out["act_score_mode"] == "query"
    assert out["product"] == "0.0.004"
    assert out["match_radius"] == 0.05
    assert out["replace_n_invalidated"] == 1
    assert out["stale_n_invalidated"] == 0
    assert out["stale_n_valid"] == 1
    assert out["stale_n_refused"] == 1
    assert out["stale_n_evicted"] == 0
    assert out["fifo_query_kind"] == "multiple_match"
    assert out["fifo_query_match_i"] == 0
    assert out["l1_acquire_n_probe"] == 2
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert out["l4_ceiling_only"] is True


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024memorylifecyclemap import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")
    if not RUNNER_LOCK.exists():
        try:
            refuse_dev_lock()
        except RuntimeError as e:
            assert "runner.lock" in str(e)
        else:
            raise AssertionError("DEV lock must wait for runner.lock")
        return
    if DEV_LOCK.exists():
        try:
            refuse_dev_lock()
        except RuntimeError as e:
            assert "again" in str(e)
        else:
            raise AssertionError("same frozen DEV execution must be refused")
        return
    refuse_dev_lock()


def memorylifecyclemap_manifest(dev: dict) -> str:
    rows = []
    for c in dev["cells"]:
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "passed": c["passed"],
                "ranking_ok": c.get("ranking_ok"),
                "reversal_matcher_miss": (c.get("matcher") or {}).get("reversal_matcher_miss"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.decision.lock"
    if not p.exists():
        return
    if DECISION_SHA is not None:
        assert sha(p) == DECISION_SHA
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["neural_edit"] is False
    assert d["w1_resurrected"] is False
    assert d["candidate_v31"] is False
    assert d["decision"]["code"] in PHASES_OK
    assert "TM024.MEMORYLIFECYCLEMAP.SCORE." not in json.dumps(d)


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    raw = p.read_bytes()
    if DEV_LOCK_SHA is not None:
        assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len(set(c["id"] for c in dev["cells"])) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.MEMORYLIFECYCLEMAP.DEV."
    assert "TM024.MEMORYLIFECYCLEMAP.SCORE." not in json.dumps(dev)
    assert Counter(c["kind"] for c in dev["cells"]) == {
        "acquire": 60,
        "stable": 60,
        "twin": 10,
        "eco": 5,
        "spec": 5,
    }
    assert all(c.get("l4_ceiling_only") is (c["arm"] == "L4") for c in dev["cells"])
    if DEV_MANIFEST_SHA is not None:
        assert memorylifecyclemap_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_phase_a_files()
    test_decision_ladder()
    test_episode_store_match_and_no_cue()
    test_exposure_grid()
    test_contract_stance()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024memorylifecyclemap: ok")


if __name__ == "__main__":
    main()
