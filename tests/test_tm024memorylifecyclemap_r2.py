"""TM.0.24.MEMORYLIFECYCLEMAP.R2 provenance. V1 freeze preserved. SCORE unopened."""

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
V1_RUNNER_LOCK_SHA = "28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593"
V1_RUNNER_PY_SHA = "edec8809938f3f1ab77948feb3661bea4fc3e6bb1abf573a81089ae628dfc974"
V1_PREREG_SHA = "dc2e7f09e80576b65a355448c8ab90a7055086db91cb47825768add5fa0fb247"
V1_CONTRACT_SHA = "909fbd708e816d6298fd9599e1d39788df9a11f43d20ee123d7b9521b8df18f6"
V1_ISOLATION_SHA = "dec9d1125d70342b14f9d1a556af58e88e95cb239b37849614fe2cde57fda825"
RUNNER_LOCK_SHA = "c05c9254e9e1b1d6b6039d7cee43b487f83a53b2d1c0b46b60feb039dd6a1077"
RUNNER_SHA = "30f3c4ee67fb4e6524088cc545232682ea7c189758c06d11db0a80428af825a2"
MANIFEST_SHA = "4dfffce9dc423f72fc136b38996b2e32f8156ab25c3316b957a3ba37d8c4feb5"
DEV_LOCK_SHA = "9321e57bb4f3bd1f4fe108c8fcb7751eca4fdb9da3d23401da5e5e2abd09eaed"
DEV_MANIFEST_SHA = "1c031296210424cc3593a320cc862c9fd1b68cc24f4909218394023890464d92"
DECISION_SHA = "484c38d90582b650633e76a9a92481022a5d3c97308c72e8d51d30d6c9b266dd"
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


def test_v1_freeze_preserved() -> None:
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock") == V1_RUNNER_LOCK_SHA
    assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py") == V1_RUNNER_PY_SHA
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.prereg.lock") == V1_PREREG_SHA
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap_contract.md") == V1_CONTRACT_SHA
    assert sha(REPO_ROOT / "docs" / "lineage_memorylifecyclemap.isolation.lock") == V1_ISOLATION_SHA
    assert not (REPO_ROOT / "docs" / "lineage_memorylifecyclemap.dev.lock").exists()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_memorylifecyclemap.r2.contract.md",
        "docs/lineage_memorylifecyclemap.r2.prereg.lock",
        "docs/lineage_memorylifecyclemap.r2.isolation.lock",
        "docs/lineage_memorylifecyclemap.runner.addendum.lock",
        "docs/lineage_memorylifecyclemap.runner.lock",
        "docs/lineage_memorylifecyclemap.prereg.lock",
        "docs/lineage_convergencemap.decision.lock",
        "docs/lineage_convergencemap.decision.addendum.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024memorylifecyclemap_r2.py",
        "experiments/run_tm024memorylifecyclemap.py",
        "experiments/run_tm024convergencemap.py",
        "experiments/run_tm024tracebridge.py",
        "experiments/run_tm024writegeom.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["act_score_mode"] == "query"
    assert prereg["w1_resurrected"] is False
    assert prereg["rewrite_historical_runner"] is False
    assert prereg["domains"]["DEV"] == "TM024.MEMORYLIFECYCLEMAP.R2.DEV."
    assert prereg["domains"]["TWIN"] == "TM024.MEMORYLIFECYCLEMAP.R2.TWIN."
    assert prereg["matched_live_reversal_arms"] == ["L1", "L2", "L3"]
    assert prereg["arms"]["L1"]["matched_live_reversal"] is True
    assert prereg["arms"]["L2"]["matched_live_reversal"] is True
    assert prereg["arms"]["L3"]["matched_live_reversal"] is True
    assert prereg["arms"]["L4"]["live_reversal_trains_learner"] is False
    assert prereg["phased_contract"]["bounded_match_sanity_cannot_satisfy_stability_gate"] is True
    assert prereg["match"]["perturbation"]["ecological_match_stability"]["sigma"] == 0.01
    assert prereg["match"]["ranking_ties_are_non_unique"] is True
    assert prereg["expected_n_cells"] == EXPECTED_N_CELLS
    assert prereg["historical_runner_lock_sha"] == V1_RUNNER_LOCK_SHA
    add = json.loads((REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.addendum.lock").read_text(encoding="utf-8"))
    assert add["rewrite_historical_runner"] is False
    assert add["historical_runner_lock_sha"] == V1_RUNNER_LOCK_SHA
    assert add["next"] == "TM.0.24.MEMORYLIFECYCLEMAP.R2"
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock") == CVG_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.dev.lock") == CVG_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock") == CVG_ADD
    assert sha(REPO_ROOT / "experiments" / "run_tm024convergencemap.py") == CVG_RUNNER_PY
    assert sha(REPO_ROOT / "experiments" / "run_tm024tracebridge.py") == TB_RUNNER_PY
    assert sha(REPO_ROOT / "experiments" / "run_tm024writegeom.py") == WG_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    runner_p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.runner.lock"
    man_p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.manifest.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap_r2.py") == RUNNER_SHA
            assert runner["shas"]["runner"] == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["domain"] == "TM024.MEMORYLIFECYCLEMAP.R2.DEV."
        assert runner["matched_live_reversal_arms"] == ["L1", "L2", "L3"]
        assert runner["l4_no_live_reversal"] is True
        assert runner["perturbation_modes"] == ["bounded_match_sanity", "ecological_match_stability"]
        assert runner["lifecycle_stability_gate"] == "ranking_perturb_sigma_0.01"
        assert runner["bounded_cannot_satisfy_stability_gate"] is True
        assert runner["historical_runner_lock_sha"] == V1_RUNNER_LOCK_SHA
        assert runner["shas"]["historical_runner_lock"] == V1_RUNNER_LOCK_SHA
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        if MANIFEST_SHA is not None:
            assert sha(man_p) == MANIFEST_SHA
        assert man["expected_n_cells"] == EXPECTED_N_CELLS
        assert len(man["expected_cell_ids"]) == EXPECTED_N_CELLS
        assert len(set(man["expected_cell_ids"])) == EXPECTED_N_CELLS


def _cell(
    arm: str,
    kind: str,
    n_cues: int,
    *,
    passed: bool,
    world: int = 0,
    order: str = "A_then_B",
    reversal_miss: bool = False,
    ecological_fail: bool = False,
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
    matcher: dict[str, object] = {}
    if reversal_miss:
        matcher["reversal_matcher_miss"] = True
        matcher["reversal_match_recall"] = 0.0
    if ecological_fail:
        matcher["ecological_match_failure"] = True
        matcher["ecological_match_stability"] = 0.0
    if matcher:
        out["matcher"] = matcher
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
    ecological_fail: bool = False,
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
    cells.append(
        _cell(arm, "eco", 2, passed=plasticity, reversal_miss=reversal_miss, ecological_fail=ecological_fail)
    )
    cells.append(
        _cell(arm, "spec", 4, passed=specificity, reversal_miss=reversal_miss, ecological_fail=ecological_fail)
    )
    return cells


def _full(**arm_kwargs: dict[str, bool]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for arm in ARMS:
        cells.extend(_arm_cells(arm, **(arm_kwargs.get(arm) or {})))
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024memorylifecyclemap_r2 import EXPECTED_N_CELLS as N
    from experiments.run_tm024memorylifecyclemap_r2 import _decision, load_prereg

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
    assert code(L1=four) == "fifo_replay_sufficient"
    assert code(L2=four, L3={"plasticity": False}) == "content_addressed_invalidation_supported"
    assert code(L2=four, L3=four) == "replacement_not_causal"
    assert code(L2={"reversal_miss": True}) == "episode_reinstatement_match_failure"
    assert code(L2={"ecological_fail": True}) == "episode_reinstatement_match_failure"
    assert code(L4=four) == "covariance_memory_ceiling_only"
    assert code() == "memory_lifecycle_insufficient"


def test_matched_live_and_store_policies() -> None:
    from experiments.run_tm024convergencemap import unique_winner
    from experiments.run_tm024eligmap import unit_or_zero
    from experiments.run_tm024memorylifecyclemap_r2 import (
        EpisodeStore,
        checkpoint_error,
        episode_core,
        ingest,
        reversal_live_learner,
    )
    from experiments.run_tm024writegeom import capacity_world

    dummy = object()
    assert reversal_live_learner("L1", dummy) is dummy
    assert reversal_live_learner("L2", dummy) is dummy
    assert reversal_live_learner("L3", dummy) is dummy
    assert reversal_live_learner("L4", dummy) is None
    assert reversal_live_learner("L0", dummy) is None

    world = capacity_world(0, "TM024.MEMORYLIFECYCLEMAP.R2.TEST.", n_cues=2, n_handles=2)
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(np.arange(64, dtype=np.float64)[::-1] + 7.0)
    h1, h2 = world["handles"][0], world["handles"][1]

    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def update(self, *_a, **_k) -> None:
            self.n += 1

    for arm in ("L1", "L2", "L3"):
        c = Counter()
        assert reversal_live_learner(arm, c) is c
        captured: list[dict] = []
        store = EpisodeStore(policy={"L1": "fifo", "L2": "replace", "L3": "retain_stale"}[arm], match_l2=0.05)
        ingest(store, captured, x, h1, 1.0, world, cue="a", tag="p")
        ingest(store, captured, x, h1, -1.0, world, cue="a", reversal=True, tag="n", live_learner=c)
        ingest(store, captured, x, h2, 1.0, world, cue="a", reversal=True, tag="r", live_learner=c)
        assert c.n == 2

    c4 = Counter()
    cap4: list[dict] = []
    ingest(None, cap4, x, h1, -1.0, world, cue="a", reversal=True, tag="n", live_learner=reversal_live_learner("L4", c4))
    assert c4.n == 0
    assert len(cap4) == 1

    stale = EpisodeStore(policy="retain_stale", match_l2=0.05)
    captured_l3: list[dict] = []
    ingest(stale, captured_l3, x, h1, 1.0, world, cue="a", tag="p")
    core = episode_core(stale.slots[0])
    age = int(stale.slots[0]["age"])
    before = list(captured_l3)
    qn = ingest(stale, captured_l3, x, h1, -1.0, world, cue="a", reversal=True, tag="n")
    qr = ingest(stale, captured_l3, x, h2, 1.0, world, cue="a", reversal=True, tag="r")
    assert qn["action"] == "refuse"
    assert qr["action"] == "refuse"
    assert episode_core(stale.slots[0]) == core
    assert int(stale.slots[0]["age"]) == age
    assert captured_l3 == before
    assert stale.stats()["n_valid"] == 1
    assert stale.stats()["n_evicted"] == 0
    assert stale.stats()["n_refused"] == 2

    repl = EpisodeStore(policy="replace", match_l2=0.05)
    w1 = repl.write(x, h1, 1.0, world=world, cue="a")
    w2 = repl.write(y, h2, 1.0, world=world, cue="b")
    assert w1["action"] == "insert"
    assert w2["action"] == "insert"
    core_y = episode_core(repl.slots[1])
    wr = repl.write(x, h1, -1.0, world=world, cue="a")
    assert wr["action"] == "replace"
    assert wr["match_i"] == 0
    assert repl.stats()["n_replaced"] == 1
    assert repl.stats()["n_valid"] == 2
    assert episode_core(repl.slots[1]) == core_y
    assert str(repl.slots[0]["handle"]) == h1
    assert float(repl.slots[0]["adv"]) < 0.0

    class TieBank:
        def scores(self, _p1):
            return {h1: 1.0, h2: 1.0}

    assert unique_winner({h1: 1.0, h2: 1.0}) is None
    assert unique_winner({h1: 1.0, h2: 0.9}) == h1
    assert checkpoint_error(TieBank(), [h1, h2], [{"p1": x, "handle": h1, "adv": 1.0}]) == 1


def test_dual_perturbation_disjoint() -> None:
    from experiments.run_tm024eligmap import unit_or_zero
    from experiments.run_tm024memorylifecyclemap_r2 import (
        EpisodeStore,
        bounded_match_sigma,
    )
    from experiments.run_tm024writegeom import capacity_world

    world = capacity_world(0, "TM024.MEMORYLIFECYCLEMAP.R2.TEST.", n_cues=2, n_handles=2)
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(np.arange(64, dtype=np.float64)[::-1] + 7.0)
    h1, h2 = world["handles"][0], world["handles"][1]
    store = EpisodeStore(policy="fifo", match_l2=0.05)
    store.write(x, h1, 1.0, world=world, cue="a")
    store.write(y, h2, 1.0, world=world, cue="b")
    store.score_perturbation(domain=world["domain"], tag="dual")
    s = store.ledger.summary()
    assert "bounded_match_sanity" in s and "ecological_match_stability" in s
    assert s["bounded_match_sigma"] == bounded_match_sigma(0.05)
    assert s["ecological_match_sigma"] == 0.01
    assert s["bounded_match_sigma"] != s["ecological_match_sigma"]
    assert s["bounded_match_sanity"] == 1.0
    assert s["ecological_match_stability"] is not None
    assert s["ecological_match_stability"] < s["bounded_match_sanity"]
    assert s["ecological_match_failure"] is True
    assert store.ledger.n_bounded_perturb == 40
    assert store.ledger.n_eco_perturb == 40


def test_cell_ids_and_manifest_hash() -> None:
    from experiments.run_tm024memorylifecyclemap_r2 import (
        EXPECTED_N_CELLS as N,
        assert_cell_coverage,
        cell_id,
        cell_manifest_hash,
        expected_cell_ids,
        expected_ids_sha,
    )

    ids = expected_cell_ids()
    assert len(ids) == N == EXPECTED_N_CELLS
    assert len(set(ids)) == N
    assert cell_id("acquire", "L2", 8, "A_then_B", 0) == "acquire|L2|c8|A_then_B|w0"
    assert ids[0] == "acquire|L0|c2|A_then_B|w0"
    cells = _full()
    for c in cells:
        c.setdefault("domain", "TM024.MEMORYLIFECYCLEMAP.R2.DEV.")
        c.setdefault("store", None)
    assert assert_cell_coverage(cells)
    man = cell_manifest_hash(cells)
    assert len(man) == 64
    assert expected_ids_sha(ids) == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    man_p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.manifest.lock"
    if man_p.exists():
        locked = json.loads(man_p.read_text(encoding="utf-8"))
        assert locked["expected_cell_ids"] == ids
        assert locked["expected_ids_sha"] == expected_ids_sha(ids)
        assert locked["expected_n_cells"] == N


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "matched contrast" in text.lower() or "matched live" in text.lower()
    assert "bounded_match_sanity" in text
    assert "ecological_match_stability" in text
    assert "replacement_not_causal" in text
    assert "ec317ba" in text


def test_smoke() -> None:
    from experiments.run_tm024memorylifecyclemap_r2 import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["act_score_mode"] == "query"
    assert out["product"] == "0.0.004"
    assert out["match_radius"] == 0.05
    assert out["stale_n_refused"] == 1
    assert out["stale_n_evicted"] == 0
    assert out["l1_acquire_n_probe"] == 2
    assert out["expected_id_count"] == EXPECTED_N_CELLS
    assert out["stability_gate"] == "ranking_perturb_sigma_0.01"
    assert out["bounded_match_sanity_used_for_pass"] is False


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024memorylifecyclemap_r2 import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

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
            raise AssertionError("DEV lock must wait for r2.runner.lock")
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


def memorylifecyclemap_r2_manifest(dev: dict) -> str:
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
                "ecological_match_failure": (c.get("matcher") or {}).get("ecological_match_failure"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.lock"
    if not p.exists():
        return
    if DECISION_SHA is not None:
        assert sha(p) == DECISION_SHA
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["neural_edit"] is False
    assert d["decision"]["code"] in PHASES_OK
    assert "TM024.MEMORYLIFECYCLEMAP.R2.SCORE." not in json.dumps(d)
    assert "TM024.MEMORYLIFECYCLEMAP.SCORE." not in json.dumps(d)


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    raw = p.read_bytes()
    if DEV_LOCK_SHA is not None:
        assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len(set(c["id"] for c in dev["cells"])) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.MEMORYLIFECYCLEMAP.R2.DEV."
    assert "TM024.MEMORYLIFECYCLEMAP.R2.SCORE." not in json.dumps(dev)
    assert Counter(c["kind"] for c in dev["cells"]) == {
        "acquire": 60,
        "stable": 60,
        "twin": 10,
        "eco": 5,
        "spec": 5,
    }
    live = {
        (c["arm"], c["kind"]): c.get("n_live_reversal_updates")
        for c in dev["cells"]
        if c["kind"] in ("eco", "spec") and c["arm"] in ("L1", "L2", "L3")
    }
    eco_live = {v for (arm, kind), v in live.items() if kind == "eco"}
    spec_live = {v for (arm, kind), v in live.items() if kind == "spec"}
    assert len(eco_live) == 1 and 0 not in eco_live
    assert len(spec_live) == 1 and 0 not in spec_live
    assert all(c.get("n_live_reversal_updates") == 0 for c in dev["cells"] if c["arm"] == "L4")
    assert all(c.get("bounded_match_sanity_used_for_pass") is False for c in dev["cells"])
    assert all(c.get("stability_gate") == "ranking_perturb_sigma_0.01" for c in dev["cells"])
    if DEV_MANIFEST_SHA is not None:
        assert memorylifecyclemap_r2_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_v1_freeze_preserved()
    test_phase_a_files()
    test_decision_ladder()
    test_matched_live_and_store_policies()
    test_dual_perturbation_disjoint()
    test_cell_ids_and_manifest_hash()
    test_contract_stance()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024memorylifecyclemap_r2: ok")


if __name__ == "__main__":
    main()
