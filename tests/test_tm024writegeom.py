"""TM.0.24.WRITEGEOM provenance. Phase 1: no neural edit. Scoring requires runner.lock."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

V30_NEURAL = "cc22cf381839049246776d2c223683078f8c13abf00cbd8e99ab2554206538b5"
MP_DECISION = "75d0ab57de8c4e394d42f78fa2af0494722b85348f123b9796de3e9b74af93b9"
MP_REAUDIT = "a076fc5a4f1199e0463756d66e9686e889ba5d200ecf63131387c440e6094a30"
ARCH_CONTRACT = "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"
AMEND_MD = "0436dfbc63e819aa74c5b20bfd31b7c68cad3b50fb915e85835c2e8d808d77de"
AMEND_LOCK = "d52bdad3f2a94909cddeae565440c3b22199e47be25f7662fb0d92746345e3f0"
V31_ISO = "9e463501b6608c0774b3a28c8aeae99a992e66473e07ca1f95bdc1b8258ce7ef"
WG_ISO = "fbb1c760acb3be452a60608cdc0d9ecd22ab54e50006c3499c4bfdd645296e34"
WG_CONTRACT = "da1b539f4e2c91f8cb08570d6c33d896e0bd79902acae594efcb325cf4695865"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_writegeom_contract.md",
        "docs/lineage_writegeom.prereg.lock",
        "docs/lineage_writegeom.isolation.lock",
        "docs/cortex_v31.prereg.lock",
        "docs/cortex_v31.isolation.lock",
        "docs/cortex_v31_architecture_amendment.md",
        "docs/cortex_v31_architecture_amendment.lock",
        "docs/cortex.candidate.v30.lock",
        "docs/lineage_motorpersist.decision.lock",
        "docs/lineage_motorpersist.reaudit.lock",
        "experiments/run_tm024writegeom.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    prereg = json.loads((REPO_ROOT / "docs" / "lineage_writegeom.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["n"] == 64
    assert prereg["increase_n"] is False
    assert prereg["candidate_name"] == "v31"
    assert prereg["neural_edit_authorized_after_this_freeze"] is True
    assert prereg["implementation_on_scored_cells_before_w1_freeze"] is False
    assert prereg["H_max"] == 8
    assert prereg["state_budget"] == 2 * 8 * 64
    assert prereg["proto_dim"] == 64
    assert prereg["no_confidence_representation"] is True
    assert prereg["arms"]["W1"]["v31_eligible"] is True
    assert prereg["arms"]["W2"]["v31_eligible"] is False
    assert prereg["arms"]["W2"]["lambda"] == 0.01
    assert prereg["domains"]["DEV"] == "TM024.WRITEGEOM.DEV."
    assert prereg["domains"]["SCORE"] == "TM024.WRITEGEOM.SCORE."
    assert prereg["margin"]["cosine_margin_min"] == 0.01
    assert prereg["margin"]["rho_perturb_sigma"] == 0.01
    assert prereg["margin"]["perturb_n"] == 20
    assert prereg["margin"]["perturb_stable_min"] == 19
    assert prereg["margin"]["frozen_before_dev"] is True
    assert prereg["reversal"]["ecological"]["required_w1_pass"] is True
    assert prereg["reversal"]["positive_only_reassignment"]["required_w1_pass"] is False
    assert prereg["lifecycle"]["same_handle_rebound"] == "retain"
    assert prereg["lifecycle"]["removed_handle"] == "dormant_not_scored"
    assert prereg["lineage_after_v31_pass"] is False
    caps = {(c["n_cues"], c["n_handles"]): c for c in prereg["capacity"]}
    assert caps[(2, 2)]["required"] is True
    assert caps[(8, 8)]["required"] is False
    iso = json.loads((REPO_ROOT / "docs" / "lineage_writegeom.isolation.lock").read_text(encoding="utf-8"))
    assert iso["motorpersist_decision_sha"] == sha(REPO_ROOT / "docs" / "lineage_motorpersist.decision.lock")
    assert iso["motorpersist_reaudit_sha"] == sha(REPO_ROOT / "docs" / "lineage_motorpersist.reaudit.lock")
    assert iso["implementation_authorized"] is True
    assert iso["n"] == 64
    v31p = json.loads((REPO_ROOT / "docs" / "cortex_v31.prereg.lock").read_text(encoding="utf-8"))
    assert v31p["architecture_amendment_md_sha"] == sha(REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.md")
    assert v31p["architecture_amendment_sha"] == sha(REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.lock")
    assert v31p["isolation_sha"] == sha(REPO_ROOT / "docs" / "cortex_v31.isolation.lock")
    amend = json.loads((REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.lock").read_text(encoding="utf-8"))
    assert amend["amendment_md_sha"] == AMEND_MD
    assert amend["isolation_sha"] == V31_ISO
    assert amend["writegeom_isolation_sha"] == WG_ISO
    assert amend["writegeom_contract_sha"] == WG_CONTRACT
    assert amend["neural_sha_at_freeze"] == V30_NEURAL
    assert amend["n"] == 64
    assert amend["H_max"] == 8
    assert sha(REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.lock") == AMEND_LOCK
    assert sha(REPO_ROOT / "docs" / "lineage_motorpersist.decision.lock") == MP_DECISION
    assert sha(REPO_ROOT / "docs" / "lineage_motorpersist.reaudit.lock") == MP_REAUDIT


def test_contract_stance() -> None:
    text = (REPO_ROOT / "docs" / "lineage_writegeom_contract.md").read_text(encoding="utf-8")
    assert "0.0.004" in text
    assert "**64**" in text
    assert "actuator-local" in text
    assert "H_max" in text
    assert "no confidence representation" in text.lower() or "no confidence representation" in text
    assert "ecological reversal" in text.lower()
    assert "positive-only reassignment" in text.lower()
    assert "lineage stays closed" in text.lower()
    assert sha(REPO_ROOT / "docs" / "cortex_architecture_contract.md") == ARCH_CONTRACT
    live_neural = sha(REPO_ROOT / "three_memory" / "neural_cortex.py")
    v31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
    if v31.exists():
        live = json.loads(v31.read_text(encoding="utf-8"))
        assert live["neural_cortex_sha"] == live_neural
        assert live["genome"]["n"] == 64
    else:
        assert live_neural == V30_NEURAL


def test_runner_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_writegeom.runner.lock"
    if not p.exists():
        return
    from experiments.run_tm024writegeom import writegeom_shas

    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["product"] == "0.0.004"
    assert lock["earned_next"] is False
    assert lock["n"] == 64
    assert lock["shas"] == writegeom_shas()
    assert lock["domain"] == "TM024.WRITEGEOM.SCORE."


def test_dev_refused_before_neural() -> None:
    from experiments.run_tm024writegeom import neural_has_proto, refuse_dev

    if neural_has_proto():
        return
    try:
        refuse_dev()
    except RuntimeError as e:
        assert "W1 neural law" in str(e)
    else:
        raise AssertionError("DEV must refuse before W1 neural law")


def test_smoke() -> None:
    from experiments.run_tm024writegeom import smoke

    out = smoke()
    assert out["smoke_ok"] is True
    assert out["n"] == 64
    assert out["H_max"] == 8
    assert out["state_budget"] == 1024
    assert out["product"] == "0.0.004"
    assert out["earned_next"] is False


def test_decision_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_writegeom.decision.lock"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["product"] == "0.0.004"
    assert d["earned_next"] is False
    assert d["ex0s"] is None
    assert d["n"] == 64
    assert d["lineage_reopened"] is False
    assert d["q3"] is False


def main() -> None:
    test_phase_a_files()
    test_contract_stance()
    test_runner_lock_if_present()
    test_dev_refused_before_neural()
    test_smoke()
    test_decision_if_present()
    print("test_tm024writegeom: ok")


if __name__ == "__main__":
    main()
