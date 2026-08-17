"""TM.0.24.TRACEBRIDGE provenance. Runner-only. W1 closed. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
PM_DEC = "55807477c1a16157a890f2ecb6c2e7fbc305905b7cc04a1ead48b70bc91e3852"
PM_DEV = "fff8040db45958a325cd6ff4dcf5468b63961d547c62c0cff321beedb88ff5d9"
R2_RUNNER_PY = "06f5f2c6edc0dffef570e75295708ea2816ea737cd0af9dab157cd94f4c26b41"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "7905af2382133119c010b800107ec13992cafea653de0720da4ac8c8b83628b8"
RUNNER_SHA = "db6c4e73cac57dd79cb86fcfa371fee1e2fc5753bf1d03402faea690ff4de551"
DEV_LOCK_SHA = "b734f0bd45aa52c65ce48fe57fd38b8693ce796d58c38547dc2391e87e49b384"
DEV_MANIFEST_SHA = "4901cdf22223b8cc0ea4be35a08d4f65ce98d8877a7381f903ef4e6c5de2ebca"
DECISION_SHA = "284968d782c73727a2de8b34e24461bf75a236f7cc6d07770e9d39529fc038a1"
EXPECTED_N_CELLS = 85
ARMS = ("B0", "B1", "B2", "B3", "B4")
PHASES_OK = (
    "revisit_phasemap_apparatus",
    "p1_transport_interpolation_only",
    "authorize_trace_only_keep_v29_write",
    "p1_sufficient_trace_dynamics_fail",
    "transport_and_discriminative_write_required",
    "p1_not_usable_by_online_class",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_tracebridge_contract.md",
        "docs/lineage_tracebridge.prereg.lock",
        "docs/lineage_tracebridge.isolation.lock",
        "docs/lineage_phasemap.decision.lock",
        "docs/lineage_phasemap.decision.addendum.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024tracebridge.py",
        "experiments/run_tm024discrimmap_r2.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_tracebridge.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["act_score_mode"] == "query"
    assert prereg["w1_resurrected"] is False
    assert prereg["arms"]["B1"]["ideal_causal_upper_bound"] is True
    assert prereg["arms"]["B2"]["lambda"] == 0.0
    assert prereg["arms"]["B2"]["rows"] == 8
    assert prereg["declared_budget_if_later_authorized"]["event_end_trace_rows"] == 512
    assert prereg["declared_budget_if_later_authorized"]["opened"] is False
    assert prereg["domains"]["DEV"] == "TM024.TRACEBRIDGE.DEV."
    assert prereg["score_reserved_unopened"] is True
    assert sha(REPO_ROOT / "docs" / "lineage_phasemap.decision.lock") == PM_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_phasemap.dev.lock") == PM_DEV
    assert sha(REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py") == R2_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    add = json.loads((REPO_ROOT / "docs" / "lineage_phasemap.decision.addendum.lock").read_text(encoding="utf-8"))
    assert add["last_robust_phase"] == "P1"
    assert add["next"] == "TM.0.24.TRACEBRIDGE"
    runner_p = REPO_ROOT / "docs" / "lineage_tracebridge.runner.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024tracebridge.py") == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["act_score_mode"] == "query"
        assert runner["w1_resurrected"] is False


def _cell(arm: str, kind: str, n_cues: int, *, passed: bool, train_ok: bool = False) -> dict[str, object]:
    return {
        "arm": arm,
        "kind": kind,
        "n_cues": n_cues,
        "passed": passed,
        "train_ranking_ok": train_ok or passed,
        "id": f"{kind}|{arm}|c{n_cues}|A_then_B|w0",
    }


def _arm_cells(arm: str, *, robust: bool, train_only: bool = False) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    rank_pass = robust or train_only
    full = robust
    for n in (2, 4, 8):
        for _wi in range(2):
            for _ord in range(2):
                cells.append(_cell(arm, "rank", n, passed=full if n == 8 else full, train_ok=rank_pass if n == 8 else full))
    # overwrite 8-cue train_only: ranking train ok, probes fail
    if train_only:
        for c in cells:
            if c["kind"] == "rank" and c["n_cues"] == 8:
                c["passed"] = False
                c["train_ranking_ok"] = True
    for _ord in range(2):
        cells.append(_cell(arm, "twin", 2, passed=full))
    for kind in ("eco", "hold", "rest"):
        cells.append(_cell(arm, kind, 2, passed=full if arm != "B4" else True))
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024tracebridge import _decision, load_prereg

    p = load_prereg()

    def code(**arms: bool) -> str:
        cells: list[dict[str, object]] = []
        train_only = arms.pop("b1_train_only", False)
        for arm in ARMS:
            rob = bool(arms.get(arm, False))
            cells.extend(_arm_cells(arm, robust=rob, train_only=bool(arm == "B1" and train_only)))
        c, _then, _extra = _decision(cells, p)
        return c

    assert code(B0=True, B1=True, B2=True, B3=True, B4=True) == "authorize_trace_only_keep_v29_write"
    assert code(B4=False, B1=True, B2=True, B3=True) == "revisit_phasemap_apparatus"
    assert code(B4=True, b1_train_only=True, B2=False, B3=False) == "p1_transport_interpolation_only"
    assert code(B4=True, B1=True, B2=False, B3=True) == "p1_sufficient_trace_dynamics_fail"
    assert code(B4=True, B1=False, B2=False, B3=True) == "transport_and_discriminative_write_required"
    assert code(B4=True, B1=False, B2=False, B3=False) == "p1_not_usable_by_online_class"


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_tracebridge_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "B1" in text and "B2" in text
    assert "not a neural amendment" in text.lower()
    assert "W1" in text


def test_smoke() -> None:
    from experiments.run_tm024tracebridge import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["w1_resurrected"] is False
    assert out["act_score_mode"] == "query"
    assert out["product"] == "0.0.004"
    assert out["b1_n_train"] == 2


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024tracebridge import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

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


def tracebridge_manifest(dev: dict) -> str:
    rows = []
    for c in dev["cells"]:
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "passed": c["passed"],
                "train_ranking_ok": c.get("train_ranking_ok"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_tracebridge.decision.lock"
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
    assert "TM024.TRACEBRIDGE.SCORE." not in json.dumps(d)


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_tracebridge.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    raw = p.read_bytes()
    if DEV_LOCK_SHA is not None:
        assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len(set(c["id"] for c in dev["cells"])) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.TRACEBRIDGE.DEV."
    assert "TM024.TRACEBRIDGE.SCORE." not in json.dumps(dev)
    assert Counter(c["arm"] for c in dev["cells"]) == {a: 17 for a in ARMS}
    assert Counter(c["kind"] for c in dev["cells"]) == {
        "rank": 60,
        "twin": 10,
        "eco": 5,
        "hold": 5,
        "rest": 5,
    }
    if DEV_MANIFEST_SHA is not None:
        assert tracebridge_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_phase_a_files()
    test_decision_ladder()
    test_contract_stance()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024tracebridge: ok")


if __name__ == "__main__":
    main()
