"""TM.0.24.MOTORPERSIST provenance. Phase A: no neural edit. Scoring requires runner.lock."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

V29_NEURAL = "d75b8da7f251378c9638cf9a0c4a859f12b0215d9f6f7b1623e704d831f86d03"
COLLISIONMAP_DECISION = "b439a2ba1c5eb92c38fabb4724eafd9f134395ea7e093e61ea68c4f4ef01b134"
PERSISTGATE_PREREG = "39179151be07a73c4c8fff506b84490af7ae21f2b24c2070638a93de9dd648c5"
AMEND_MD = "0e0015791c868d941f482d0f17ab68c53be5e33f06b50150d8843083a9042254"
V30_ISO = "039b80892a2bd44a226f901e0881d1779ce618caa3cc14b094af67cb18b6dcc1"
MP_ISO = "a4c7aabdb1e42d5387d702091fe4ca7abbc687053fda4cadf493fd4764e5985d"
AMEND_LOCK = "09f18b1965383d165b7ed0cc2c6bea19565e76651debbde101a6a352cc3c3e91"
ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_motorpersist_contract.md",
        "docs/lineage_motorpersist.prereg.lock",
        "docs/lineage_motorpersist.isolation.lock",
        "docs/cortex_v30.prereg.lock",
        "docs/cortex_v30.isolation.lock",
        "docs/cortex_v30_architecture_amendment.md",
        "docs/cortex_v30_architecture_amendment.lock",
        "docs/cortex.candidate.v29.lock",
        "docs/lineage_collisionmap.decision.lock",
        "docs/lineage_persistgate.prereg.lock",
        "experiments/run_tm024motorpersist.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_motorpersist.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["candidate_name"] == "v30"
    assert prereg["neural_edit_authorized_after_this_freeze"] is True
    assert prereg["implementation_on_scored_cells_before_p_freeze"] is False
    assert prereg["p_grid"] == [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    assert prereg["select"] == "smallest_p_meeting_all_dev_criteria"
    assert prereg["domains"]["DEV"] == "TM024.MOTORPERSIST.DEV."
    assert prereg["domains"]["SCORE"] == "TM024.MOTORPERSIST.SCORE."
    assert prereg["domains"]["TWIN"] == "TM024.MOTORPERSIST.TWIN."
    assert prereg["norm"] == "convex_combination_no_extra_l2"
    assert prereg["p0_is_v29"] is True
    assert prereg["lineage_after_v30_pass"] is False
    iso = json.loads((REPO_ROOT / "docs" / "lineage_motorpersist.isolation.lock").read_text(encoding="utf-8"))
    assert iso["collisionmap_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_collisionmap.decision.lock")
    assert iso["persistgate_prereg_sha"] == sha(REPO_ROOT / "docs" / "lineage_persistgate.prereg.lock")
    assert iso["implementation_authorized"] is True
    assert iso["n"] == 64
    v30p = json.loads((REPO_ROOT / "docs" / "cortex_v30.prereg.lock").read_text(encoding="utf-8"))
    assert v30p["architecture_amendment_md_sha"] == sha(REPO_ROOT / "docs" / "cortex_v30_architecture_amendment.md")
    assert v30p["architecture_amendment_sha"] == sha(REPO_ROOT / "docs" / "cortex_v30_architecture_amendment.lock")
    assert v30p["isolation_sha"] == sha(REPO_ROOT / "docs" / "cortex_v30.isolation.lock")
    amend = json.loads((REPO_ROOT / "docs" / "cortex_v30_architecture_amendment.lock").read_text(encoding="utf-8"))
    assert amend["amendment_md_sha"] == AMEND_MD
    assert amend["isolation_sha"] == V30_ISO
    assert amend["motorpersist_isolation_sha"] == MP_ISO
    assert amend["neural_sha_at_freeze"] == V29_NEURAL
    assert amend["n"] == 64
    assert iso["collisionmap_decision_sha"] == COLLISIONMAP_DECISION
    assert iso["persistgate_prereg_sha"] == PERSISTGATE_PREREG
    assert sha(REPO_ROOT / "docs" / "cortex_v30_architecture_amendment.lock") == AMEND_LOCK


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_motorpersist_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "zero-input motor" in text
    assert "p=0" in text
    assert "lineage stays closed" in text.lower()
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    cand = json.loads((REPO_ROOT / "docs" / "cortex.candidate.v29.lock").read_text(encoding="utf-8"))
    v30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
    live_neural = sha(REPO_ROOT / "three_memory" / "neural_cortex.py")
    if v30.exists():
        live = json.loads(v30.read_text(encoding="utf-8"))
        assert live["neural_cortex_sha"] == live_neural
        assert live["neural_cortex_sha"] != cand["neural_cortex_sha"]
        assert live["genome"]["n"] == 64
    else:
        assert cand["neural_cortex_sha"] == V29_NEURAL
        assert cand["genome"]["n"] == 64
    assert cand["genome"]["n"] == 64


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_motorpersist.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024motorpersist import motorpersist_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["n"] == 64
    assert lock["shas"] == motorpersist_shas()
    assert lock["domain"] == "TM024.MOTORPERSIST.SCORE."
    assert "p" in lock


def test_opposing_handles_positive_advantage() -> None:
    from experiments.run_tm024motorpersist import (
        DEV_DOMAIN,
        POS_DELTA,
        POS_DELTA_H1,
        POS_DELTA_H2,
        handles,
        make_cell_world,
        mid_adv,
        opposing_world,
    )

    assert POS_DELTA_H1 == POS_DELTA
    assert POS_DELTA_H2 == POS_DELTA
    w = opposing_world(make_cell_world(0, DEV_DOMAIN))
    h1, h2 = handles(w)
    a1, a2 = mid_adv(w, h1), mid_adv(w, h2)
    assert a1 > 0.0
    assert a2 > 0.0
    assert abs(a1 - a2) <= 1e-12


def test_last_write_wins_equal_advantage() -> None:
    """Live regression: equal consequences, both orders, last write wins at p=0."""
    import tempfile

    from experiments.run_tm023cortex import make_cortex
    from experiments.run_tm024motorpersist import (
        DEV_DOMAIN,
        apply_persist,
        make_cell_world,
        teach_opposing,
    )

    world = make_cell_world(0, DEV_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="mp_lww_") as tmp:
        for order, want_h1 in (("A_then_B", False), ("B_then_A", True)):
            ag = make_cortex(Path(tmp) / order, device="cpu")
            ag.bind_actuators(list(world["handles"]))
            apply_persist(ag, 0.0)
            out = teach_opposing(ag, world, tag=f"lww_{order}", order=order)
            assert out["equal_adv"] is True
            assert abs(out["mid_adv"]["h1"] - out["mid_adv"]["h2"]) <= 1e-12
            assert all(float(t["adv"]) > 0.0 for t in out["taught"])
            assert out["opposing"] is False
            assert out["last_write_wins"] is True
            if want_h1:
                assert out["live_a"]["prefer_h1"] and out["live_b"]["prefer_h1"]
            else:
                assert out["live_a"]["prefer_h2"] and out["live_b"]["prefer_h2"]
            assert out["teach_rho_cosine"] > 0.99


def test_smoke() -> None:
    from experiments.run_tm024motorpersist import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["product"] == "0.0.004"
    assert out["earned_next"] is False


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_motorpersist.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["n"] == 64
    assert d["usable_p_exists"] is False
    assert d["selected_p"] is None
    assert d["scored_worlds"] is False
    assert d["lineage_reopened"] is False
    assert d["q3"] is False
    assert d["decision"]["code"] == "identity_survives_opposing_learning_fails"
    assert d["decision"]["next"] == "plastic_write_geometry_or_connection_local_state"
    assert d["decision"]["passed"] is False
    plock = json.loads((REPO_ROOT / "docs" / "lineage_motorpersist.p.lock").read_text(encoding="utf-8"))
    assert plock["usable_p_exists"] is False
    assert plock["module_p"] == 0.0
    opp_rows = [r for r in plock.get("rows") or [] if r.get("opposing_detail")]
    assert opp_rows
    for r in opp_rows:
        advs = [float(t["adv"]) for t in r["opposing_detail"]["taught"]]
        assert all(a > 0.0 for a in advs), advs


def test_reaudit_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_motorpersist.reaudit.lock"
    if not p.exists():
        return
    from experiments.run_tm024motorpersist import (
        DEV_DOMAIN,
        HISTORICAL_DECISION_SHA,
        HISTORICAL_P_LOCK_SHA,
        SCORE_DOMAIN,
    )

    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["n"] == 64
    assert d["n_sequences"] == 24
    assert len(d["rows"]) == 24
    assert d["domain"] == DEV_DOMAIN
    assert d["score_domain_opened"] is False
    assert SCORE_DOMAIN not in json.dumps(d)
    assert d["equal_positive_consequences"] is True
    assert d["last_write_wins_all"] is True
    assert d["a_then_b_both_select_b"] is True
    assert d["b_then_a_both_select_a"] is True
    assert d["opposing_any"] is False
    assert d["decision_code"] == "identity_survives_opposing_learning_fails"
    assert sha(REPO_ROOT / "docs" / "lineage_motorpersist.decision.lock") == HISTORICAL_DECISION_SHA
    assert sha(REPO_ROOT / "docs" / "lineage_motorpersist.p.lock") == HISTORICAL_P_LOCK_SHA
    assert d["historical_decision_sha"] == HISTORICAL_DECISION_SHA
    assert d["historical_p_lock_sha"] == HISTORICAL_P_LOCK_SHA
    orders = {r["order"] for r in d["rows"]}
    assert orders == {"A_then_B", "B_then_A"}
    assert all(r["domain"] == DEV_DOMAIN for r in d["rows"])
    for r in d["rows"]:
        assert r["last_write_wins"] is True
        assert r["opposing"] is False
        assert r["equal_adv"] is True
        assert "teach_rho_cosine" in r and "teach_rho_l2" in r


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_opposing_handles_positive_advantage()
    test_last_write_wins_equal_advantage()
    test_smoke()
    test_decision_if_present()
    test_reaudit_if_present()
    print("test_tm024motorpersist: ok")


if __name__ == "__main__":
    main()
