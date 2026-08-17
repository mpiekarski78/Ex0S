"""TM.0.24.ELIGMAP provenance. Runner-only. No neural edit. SCORE unopened."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
WG_DECISION = "babde6761df7c7aca4edbd88daa415a242f6c881b5fdcd021d0d43f9a9a5e538"
WG_DEV = "311fd79b9cf3e31c191c31677ca7707256bd81a6739e8a7bca2f44c457dcd78f"
WG_ADDENDUM = "60973f27cc133310c9bffac91aec732d8ac80d7735120c79dc6c49c70914ce0c"
V30_CAND = "4992ad0206916c17d7723fcbf22d9f8e1ad7e90d55497d80ee791d16c559856c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_eligmap_contract.md",
        "docs/lineage_eligmap.prereg.lock",
        "docs/lineage_eligmap.isolation.lock",
        "docs/lineage_writegeom.decision.lock",
        "docs/lineage_writegeom.decision.addendum.lock",
        "docs/lineage_eligmap.decision.addendum.lock",
        "docs/cortex.candidate.v30.lock",
        "experiments/run_tm024eligmap.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_eligmap.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["neural_edit"] is False
    assert prereg["implementation_authorized"] is False
    assert prereg["investigation_justified"] is True
    assert prereg["H_max"] == 8
    assert prereg["declared_budget_if_later_authorized"]["total"] == 1536
    assert prereg["declared_budget_if_later_authorized"]["eligibility_rows"] == 512
    assert prereg["domains"]["DEV"] == "TM024.ELIGMAP.DEV."
    assert prereg["domains"]["SCORE"] == "TM024.ELIGMAP.SCORE."
    assert prereg["score_reserved_unopened"] is True
    assert prereg["elam"]["lambda_grid"] == [0.0, 0.5, 0.9, 0.95, 0.99]
    assert prereg["margin"]["cosine_margin_min"] == 0.01
    assert prereg["margin"]["frozen_before_dev"] is True
    assert prereg["negative_write"]["c_max"] == 1.0
    assert prereg["negative_write"]["ranking_battery_law"] == "N0"
    caps = {(c["n_cues"], c["n_handles"]): c for c in prereg["capacity"]}
    assert caps[(2, 2)]["required"] is True
    assert caps[(8, 2)]["required"] is True
    assert (8, 8) not in caps
    iso = json.loads((REPO_ROOT / "docs" / "lineage_eligmap.isolation.lock").read_text(encoding="utf-8"))
    assert iso["writegeom_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_writegeom.decision.lock")
    assert iso["writegeom_addendum_sha"] == sha(REPO_ROOT / "docs" / "lineage_writegeom.decision.addendum.lock")
    assert iso["implementation_authorized"] is False
    assert iso["neural_edit"] is False
    assert iso["n"] == 64
    assert sha(REPO_ROOT / "docs" / "lineage_writegeom.decision.lock") == WG_DECISION
    assert sha(REPO_ROOT / "docs" / "lineage_writegeom.dev.lock") == WG_DEV
    assert sha(REPO_ROOT / "docs" / "lineage_writegeom.decision.addendum.lock") == WG_ADDENDUM
    assert sha(REPO_ROOT / "docs" / "cortex.candidate.v30.lock") == V30_CAND
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v32.lock").exists()


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_eligmap_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "runner-only" in text.lower()
    assert "E0" in text and "E1" in text
    assert "Eλ" in text or "Elam" in text
    assert "clipnorm" in text
    assert "1536" in text
    assert "do not install another local learner" in text.lower()
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    add = json.loads((REPO_ROOT / "docs" / "lineage_writegeom.decision.addendum.lock").read_text(encoding="utf-8"))
    assert add["refined_code"] == "w1_query_margin_insufficient__unit_norm_negative_inert"
    assert add["historical_code"] == "w1_ranking_crumb_margin_ecological_fail"
    assert add["preserved"]["candidate_v31"] is False
    assert add["rewrite_historical_decision"] is False
    hist = json.loads((REPO_ROOT / "docs" / "lineage_writegeom.decision.lock").read_text(encoding="utf-8"))
    assert hist["decision"]["code"] == "w1_ranking_crumb_margin_ecological_fail"


def test_unit_norm_identity() -> None:
    from experiments.run_tm024eligmap import n1_negative_shrinks, unit_norm_negative_inert

    assert unit_norm_negative_inert() is True
    assert n1_negative_shrinks() is True


def test_smoke() -> None:
    from experiments.run_tm024eligmap import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["unit_norm_negative_inert"] is True
    assert out["n1_negative_shrinks"] is True
    assert out["neural_edit"] is False
    assert out["v31_exists"] is False
    assert out["v32_exists"] is False
    assert out["product"] == "0.0.004"
    assert out["earned_next"] is False


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_eligmap.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["n"] == 64
    assert d["neural_edit"] is False
    assert d["candidate_v31"] is False
    assert d["candidate_v32"] is False
    assert d["scored_worlds"] is False
    assert d["lineage_reopened"] is False
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists()
    assert not (REPO_ROOT / "docs" / "cortex.candidate.v32.lock").exists()
    assert "TM024.ELIGMAP.SCORE." not in json.dumps(d)
    devp = REPO_ROOT / "docs" / "lineage_eligmap.dev.lock"
    if devp.exists():
        dev = json.loads(devp.read_text(encoding="utf-8"))
        assert dev["score_domain_opened"] is False
        assert dev["neural_edit"] is False


def test_score_refused() -> None:
    from experiments.run_tm024eligmap import refuse_score

    try:
        refuse_score()
    except RuntimeError as e:
        assert "SCORE" in str(e)
    else:
        raise AssertionError("SCORE must be refused")


DEV_LOCK_SHA = "33f79b6b83fb5b7e33b452019e010b5a05a8b0cd762b8ed6f355346b6a4a7578"
DEV_MANIFEST_SHA = "f0c08596637d1ef4b14da0cc08f2f463377a59501ae7d9904360427023615441"
EXPECTED_N_CELLS = 113
ADDRESSES = ["E0", "E1", "Edelta", "Elam_0.0", "Elam_0.5", "Elam_0.9", "Elam_0.95", "Elam_0.99"]
LAMBDAS = ["0.0", "0.5", "0.9", "0.95", "0.99"]
CUE_COUNTS = (2, 4, 8)
ORDERS = ("A_then_B", "B_then_A")
WORLDS = (0, 1)
NEG_ADDR = ("E0", "E1", "Elam_0.9")
NEG_LAWS = ("N0", "N1", "N2")
TICKS_PER_ONE_SYMBOL_OBSERVE = 5
TEACH_OBSERVES = 2


def _cid_rank(c: dict) -> str:
    return f"rank|{c['address']}|c{c['n_cues']}|h{c['n_handles']}|{c['order']}|w{c['world']}|{c['law']}"


def _cid_surv(s: dict) -> str:
    return f"surv|{s['address']}"


def _cid_neg(n: dict) -> str:
    return f"neg|{n['law']}|{n['address']}"


def eligmap_manifest(dev: dict) -> str:
    rows: list[dict] = []
    for c in dev["cells"]:
        rows.append(
            {
                "id": _cid_rank(c),
                "passed": c["passed"],
                "ranking_ok": c["ranking_ok"],
                "ok": [p["ok"] for p in c["probes"]],
                "margins": [round(float(p["margin"]), 12) for p in c["probes"]],
            }
        )
    for s in dev["survival"]:
        rows.append(
            {
                "id": _cid_surv(s),
                "passed": s["passed"],
                "rest": s["rest"],
                "twin": s["rename_twin"],
            }
        )
    for n in dev["negative"]:
        rows.append(
            {
                "id": _cid_neg(n),
                "passed": n["passed"],
                "ok": [p["ok"] for p in n["probes"]],
                "margins": [round(float(p["margin"]), 12) for p in n["probes"]],
            }
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_dev_coverage_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_eligmap.dev.lock"
    if not p.exists():
        return
    from collections import Counter

    from experiments.run_tm024eligmap import (
        DEV_DOMAIN,
        TWIN_DOMAIN,
        _fresh,
        address_ids,
        capacity_world,
        load_prereg,
        teach_with_adv,
    )

    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DEV_LOCK_SHA
    dev = json.loads(raw.decode("utf-8"))
    prereg = load_prereg()
    n_rank = len(dev["cells"])
    n_surv = len(dev["survival"])
    n_neg = len(dev["negative"])
    assert n_rank + n_surv + n_neg == EXPECTED_N_CELLS
    assert dev["n_cells"] == EXPECTED_N_CELLS
    ids = [_cid_rank(c) for c in dev["cells"]]
    ids += [_cid_surv(s) for s in dev["survival"]]
    ids += [_cid_neg(n) for n in dev["negative"]]
    assert len(ids) == EXPECTED_N_CELLS
    assert len(set(ids)) == EXPECTED_N_CELLS

    aids = address_ids(prereg)
    assert aids == ADDRESSES
    addr_c = Counter(c["address"] for c in dev["cells"])
    for a in ADDRESSES:
        assert addr_c[a] == len(CUE_COUNTS) * len(ORDERS) * len(WORLDS), a
    lam_c = Counter(c["address"] for c in dev["cells"] if c["address"].startswith("Elam_"))
    for lam in LAMBDAS:
        assert lam_c[f"Elam_{lam}"] == 12, lam
    assert Counter(c["n_cues"] for c in dev["cells"]) == {2: 32, 4: 32, 8: 32}
    assert Counter(c["order"] for c in dev["cells"]) == {o: 48 for o in ORDERS}
    assert Counter(c["world"] for c in dev["cells"]) == {w: 48 for w in WORLDS}
    assert set(s["address"] for s in dev["survival"]) == set(ADDRESSES)
    assert {(n["law"], n["address"]) for n in dev["negative"]} == {
        (law, a) for a in NEG_ADDR for law in NEG_LAWS
    }
    assert all("rename_twin" in s for s in dev["survival"])
    assert n_surv == 8
    assert n_neg == 9

    blob = json.dumps(dev)
    assert "TM024.ELIGMAP.SCORE." not in blob
    assert "TM024.WRITEGEOM.SCORE." not in blob
    assert dev["domain"] == DEV_DOMAIN
    assert TWIN_DOMAIN.startswith("TM024.ELIGMAP.TWIN")
    assert dev["score_domain_opened"] is False

    for c in dev["cells"]:
        assert c.get("probes"), _cid_rank(c)
        assert c.get("taught_adv"), _cid_rank(c)
        assert len(c["probes"]) == int(c["n_cues"]), _cid_rank(c)
        assert len(c["taught_adv"]) == int(c["n_cues"]), _cid_rank(c)
        assert all(abs(float(a)) > 1e-12 for a in c["taught_adv"]), _cid_rank(c)
        ok_all = all(bool(p["ok"]) for p in c["probes"])
        assert bool(c["passed"]) is ok_all, _cid_rank(c)
        rank = all(p["winner"] == p["want"] for p in c["probes"])
        assert bool(c["ranking_ok"]) is rank, _cid_rank(c)
        for p in c["probes"]:
            assert "margin" in p and "perturb_stable" in p and "scores" in p
            assert p["scores"], _cid_rank(c)
    for s in dev["survival"]:
        flags = [s["rest"], s["distractor"], s["event"], s["permutation"], s["rename_twin"]]
        assert bool(s["passed"]) is all(flags), _cid_surv(s)
        assert "rest_ranking" in s and "twin_ranking" in s
    for n in dev["negative"]:
        assert n.get("probes") and n.get("adv"), _cid_neg(n)
        assert len(n["adv"]) == 3
        assert all(abs(float(a)) > 1e-12 for a in n["adv"]), _cid_neg(n)
        ok_all = all(bool(p["ok"]) for p in n["probes"])
        assert bool(n["passed"]) is ok_all, _cid_neg(n)

    assert eligmap_manifest(dev) == DEV_MANIFEST_SHA
    dec = json.loads((REPO_ROOT / "docs" / "lineage_eligmap.decision.lock").read_text(encoding="utf-8"))
    assert dec["dev_lock_sha"] == DEV_LOCK_SHA
    assert dec["decision"]["code"] == "trace_separates_linear_fails"
    assert dec["implementation_authorized"] is False

    import tempfile

    w = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    cue, handle = w["cue_handle"][0]["cue"], w["cue_handle"][0]["handle"]
    with tempfile.TemporaryDirectory(prefix="em_cov_") as tmp:
        ag = _fresh(tmp, "s", w)
        ticks, adv = teach_with_adv(ag, w, cue, handle, tag="cov")
    assert len(ticks) == TEACH_OBSERVES
    assert [t["kind"] for t in ticks] == ["select", "credit"]
    assert abs(float(adv)) > 1e-12
    for t in ticks:
        n_ticks = len(t["trajectory"])
        assert n_ticks == TICKS_PER_ONE_SYMBOL_OBSERVE, n_ticks
        assert t["rho_elig"].shape == (64,)
        assert t["observable"].shape == (64,)
        assert float(abs(t["rho_elig"]).sum()) > 0.0


def test_decision_addendum_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_eligmap.decision.addendum.lock"
    if not p.exists():
        return
    add = json.loads(p.read_text(encoding="utf-8"))
    hist = json.loads((REPO_ROOT / "docs" / "lineage_eligmap.decision.lock").read_text(encoding="utf-8"))
    assert add["historical_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_eligmap.decision.lock")
    assert add["historical_code"] == hist["decision"]["code"]
    assert add["next"] == "TM.0.24.DISCRIMMAP"
    assert add["next_kind"] == "runner_only_separability_diagnostic"
    assert add["not_neural_amendment"] is True
    assert add["eligibility_budget_installed"] is False
    assert add["n1n2_secondary"] is True
    assert add["rewrite_historical_decision"] is False


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_unit_norm_identity()
    test_smoke()
    test_decision_if_present()
    test_dev_coverage_if_present()
    test_decision_addendum_if_present()
    test_score_refused()
    print("test_tm024eligmap: ok")


if __name__ == "__main__":
    main()
