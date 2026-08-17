"""TM.0.24.PHASEMAP provenance. Runner-only. Write-geometry closed. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
R2_DEC = "a888a8af5495e19139f8691725241c2b31322f192218983d02c131ebf6c84675"
R2_DEV = "99f9097a3f30df7b69d46d5c20f858e35f540faa387c8e9bcb9062db07ab3831"
R2_RUNNER_PY = "06f5f2c6edc0dffef570e75295708ea2816ea737cd0af9dab157cd94f4c26b41"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"
RUNNER_LOCK_SHA = "1fb87558052b6cf41dedf958729334e487a922580965dbfdb60f3d077822b3bc"
RUNNER_SHA = "716b726af3fec2167c061b804964b3b0f755685dd959b4c202edfa746455a50b"
DEV_LOCK_SHA = "fff8040db45958a325cd6ff4dcf5468b63961d547c62c0cff321beedb88ff5d9"
DEV_MANIFEST_SHA = "692f00b27832f238d00fb82abd45ee23f9a1f79313d620d606a7a5188ec6a35b"
DECISION_SHA = "55807477c1a16157a890f2ecb6c2e7fbc305905b7cc04a1ead48b70bc91e3852"
EXPECTED_N_RANK = 72
EXPECTED_N_TWIN = 12
EXPECTED_N_CELLS = 84
PHASES = ("P0", "P1", "P2", "P3", "P4", "P5")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_phasemap_contract.md",
        "docs/lineage_phasemap.prereg.lock",
        "docs/lineage_phasemap.isolation.lock",
        "docs/lineage_discrimmap.r2.decision.lock",
        "docs/lineage_discrimmap.r2.decision.addendum.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024phasemap.py",
        "experiments/run_tm024discrimmap_r2.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_phasemap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["n"] == 64
    assert prereg["neural_edit"] is False
    assert prereg["write_geometry_branch_closed"] is True
    assert prereg["phases"] == list(PHASES)
    assert prereg["d1_oracle"]["no_new_solver"] is True
    assert prereg["d1_oracle"]["no_soft_margin"] is True
    assert prereg["margin"]["geometric_margin_min"] == 0.01
    assert prereg["n_repeats"] == 3
    assert prereg["domains"]["DEV"] == "TM024.PHASEMAP.DEV."
    assert prereg["score_reserved_unopened"] is True
    assert prereg["declared_budget_remains_closed"] == 1536
    assert "D5" in prereg["refuse"]
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.r2.decision.lock") == R2_DEC
    assert sha(REPO_ROOT / "docs" / "lineage_discrimmap.r2.dev.lock") == R2_DEV
    assert sha(REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py") == R2_RUNNER_PY
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    add = json.loads((REPO_ROOT / "docs" / "lineage_discrimmap.r2.decision.addendum.lock").read_text(encoding="utf-8"))
    assert add["write_geometry_branch_closed"] is True
    assert add["next"] == "TM.0.24.PHASEMAP"
    assert add["d3_failure_does_not_authorize_better_online_rule"] is True
    assert add["d4_does_not_affect_linear_refusal"] is True
    runner_p = REPO_ROOT / "docs" / "lineage_phasemap.runner.lock"
    if runner_p.exists():
        runner = json.loads(runner_p.read_text(encoding="utf-8"))
        if RUNNER_LOCK_SHA is not None:
            assert sha(runner_p) == RUNNER_LOCK_SHA
        if RUNNER_SHA is not None:
            assert sha(REPO_ROOT / "experiments" / "run_tm024phasemap.py") == RUNNER_SHA
        assert runner["n"] == 64
        assert runner["expected_n_cells"] == EXPECTED_N_CELLS
        assert runner["write_geometry_branch_closed"] is True
        assert runner["phases"] == list(PHASES)


def _synth_cells(phase_spec: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for ph in PHASES:
        spec = phase_spec.get(ph, {})
        passed = bool(spec.get("passed", False))
        train_clean = bool(spec.get("train_clean", passed))
        ranking_ok = bool(spec.get("ranking_ok", passed))
        probe_g = float(spec.get("probe_g", 0.02 if ranking_ok else 0.0))
        cells.append(
            {
                "phase": ph,
                "kind": "rank",
                "n_cues": 8,
                "passed": passed,
                "train_clean": train_clean,
                "ranking_ok": ranking_ok,
                "probe_geometric_margin": probe_g,
            }
        )
        cells.append(
            {
                "phase": ph,
                "kind": "twin",
                "n_cues": 2,
                "passed": bool(spec.get("twin_passed", passed)),
                "train_clean": True,
                "ranking_ok": True,
                "probe_geometric_margin": 0.02,
            }
        )
    return cells


def test_decision_ladder() -> None:
    from experiments.run_tm024phasemap import _decision, load_prereg

    p = load_prereg()
    ok = {"passed": True}
    dirty = {"passed": False, "train_clean": False, "ranking_ok": False, "probe_g": 0.0}
    reinst = {"passed": False, "train_clean": True, "ranking_ok": False, "probe_g": 0.0}
    perturb_only = {"passed": False, "train_clean": True, "ranking_ok": True, "probe_g": 0.02}

    def code(spec: dict[str, dict[str, object]]) -> tuple[str, object]:
        c, _then, extra = _decision(_synth_cells(spec), p)
        return c, extra["first_fail"]

    all_ok = {ph: ok for ph in PHASES}
    assert code(all_ok) == ("revisit_discrimmap_apparatus", None)
    assert code({"P0": reinst, **{ph: ok for ph in PHASES if ph != "P0"}}) == (
        "state_reinstatement_instability",
        "P0",
    )
    assert code({**{ph: ok for ph in ("P0", "P1", "P2", "P3", "P4")}, "P5": dirty}) == (
        "eligibility_snapshot_mismatch",
        "P5",
    )
    assert code({**{ph: ok for ph in ("P0", "P1", "P2", "P3")}, "P4": dirty, "P5": dirty}) == (
        "motor_transition_address_collapse",
        "P4",
    )
    assert code({**{ph: ok for ph in ("P0", "P1")}, "P2": dirty, "P3": dirty, "P4": dirty, "P5": dirty}) == (
        "post_cue_dynamics_destroy_robust_address",
        "P2",
    )
    assert code({"P0": ok, "P1": dirty, "P2": dirty, "P3": dirty, "P4": dirty, "P5": dirty}) == (
        "post_cue_dynamics_destroy_robust_address",
        "P1",
    )
    assert code({ph: dirty for ph in PHASES}) == ("cue_representation_capacity_absent", "P0")
    assert code({"P0": perturb_only, **{ph: dirty for ph in PHASES if ph != "P0"}}) == (
        "cue_representation_capacity_absent",
        "P0",
    )


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_phasemap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "P0" in text and "P5" in text
    assert "contraction" in text.lower()
    assert "not a neural amendment" in text.lower()
    assert "write-geometry" in text.lower()


def test_smoke() -> None:
    from experiments.run_tm024phasemap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["neural_edit"] is False
    assert out["write_geometry_branch_closed"] is True
    assert out["v31_exists"] is False
    assert out["product"] == "0.0.004"


def test_score_and_dev_lock_gate() -> None:
    from experiments.run_tm024phasemap import DEV_LOCK, RUNNER_LOCK, refuse_dev_lock, refuse_score

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


def phasemap_manifest(dev: dict) -> str:
    rows = []
    for c in dev["cells"]:
        rows.append(
            {
                "id": c["id"],
                "phase": c["phase"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "order": c["order"],
                "world": c["world"],
                "passed": c["passed"],
                "solver_status": c["solver_status"],
                "train_clean": c["train_clean"],
                "train_geometric_margin": round(float(c["train_geometric_margin"]), 12),
                "probe_geometric_margin": round(float(c["probe_geometric_margin"]), 12),
                "perturb_stable": c["perturb_stable"],
                "contraction_mean": None
                if c.get("contraction_mean") is None
                else round(float(c["contraction_mean"]), 12),
            }
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_phasemap.decision.lock"
    if not p.exists():
        return
    if DECISION_SHA is not None:
        assert sha(p) == DECISION_SHA
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["neural_edit"] is False
    assert d["implementation_authorized"] is False
    assert d["candidate_v31"] is False
    assert d["write_geometry_branch_closed"] is True
    assert d["decision"]["code"] in {
        "revisit_discrimmap_apparatus",
        "state_reinstatement_instability",
        "eligibility_snapshot_mismatch",
        "motor_transition_address_collapse",
        "post_cue_dynamics_destroy_robust_address",
        "cue_representation_capacity_absent",
    }
    assert "TM024.PHASEMAP.SCORE." not in json.dumps(d)


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_phasemap.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    raw = p.read_bytes()
    if DEV_LOCK_SHA is not None:
        assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    assert dev["n_cells"] == EXPECTED_N_CELLS
    assert len(set(c["id"] for c in dev["cells"])) == EXPECTED_N_CELLS
    assert dev["domain"] == "TM024.PHASEMAP.DEV."
    assert "TM024.PHASEMAP.SCORE." not in json.dumps(dev)
    assert Counter(c["phase"] for c in dev["cells"]) == {ph: 14 for ph in PHASES}
    assert Counter(c["kind"] for c in dev["cells"]) == {"rank": EXPECTED_N_RANK, "twin": EXPECTED_N_TWIN}
    gmin = 0.01
    for c in dev["cells"]:
        assert c["n_train"] == c["n_cues"] and c["n_probe"] == c["n_cues"]
        assert c["solver_status"] in ("optimal", "infeasible")
        expected = bool(
            c["solver_status"] == "optimal"
            and c["train_ranking_ok"]
            and c["ranking_ok"]
            and float(c["train_geometric_margin"]) >= gmin
            and float(c["probe_geometric_margin"]) >= gmin
            and c["perturb_stable"]
        )
        assert bool(c["passed"]) is expected, c["id"]
        if c["phase"] != "P0":
            assert "contraction_mean" in c
    if DEV_MANIFEST_SHA is not None:
        assert phasemap_manifest(dev) == DEV_MANIFEST_SHA


def main() -> None:
    test_phase_a_files()
    test_decision_ladder()
    test_contract_stance()
    test_smoke()
    test_score_and_dev_lock_gate()
    test_decision_if_present()
    test_dev_coverage_if_present()
    print("test_tm024phasemap: ok")


if __name__ == "__main__":
    main()
