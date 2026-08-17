"""TM.0.24.LINEAGE Phase A provenance. CPU only. No capability claim."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lineage_genome_layout_spec import OUT as LAYOUT_OUT, build as build_layout


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_files() -> None:
    for rel in (
        "docs/lineage_architecture_contract.md",
        "docs/lineage_fitness_contract.md",
        "docs/lineage_genome_layout.json",
        "docs/lineage_world_generator.prereg.lock",
        "docs/lineage.prereg.lock",
        "docs/lineage_wall.prereg.lock",
        "docs/lineage_phase0a.lock",
        "experiments/lineage_genome_layout_spec.py",
        "experiments/run_tm024lineage_phase0a.py",
        "experiments/run_tm024lineage.py",
        "three_memory/cortex_lineage.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    gen_lock = REPO_ROOT / "docs" / "lineage_world_generator.lock"
    if gen_lock.exists():
        lock = json.loads(gen_lock.read_text(encoding="utf-8"))
        assert lock["generator_sha"] == sha(REPO_ROOT / "experiments" / "run_tm024lineage.py")
    cand_p = REPO_ROOT / "docs" / "lineage_engine.candidate.lock"
    if cand_p.exists():
        from experiments.run_tm024lineage import engine_shas

        cand = json.loads(cand_p.read_text(encoding="utf-8"))
        pre = json.loads((REPO_ROOT / "docs" / "lineage_engine.preflight.lock").read_text(encoding="utf-8"))
        assert cand["shas"] == pre["shas"]
        assert cand["shas"] == engine_shas()
        assert cand["product"] == "0.0.004"
        assert cand["earned_next"] is False
        assert cand["ex0s"] is None
        assert cand["eligible_for_000005"] is False


def test_layout_rebuild() -> None:
    expected = build_layout()
    on_disk = json.loads(LAYOUT_OUT.read_text(encoding="utf-8"))
    assert on_disk == expected
    assert expected["arms"]["D"]["dim"] == 134
    assert expected["arms"]["C"]["dim"] == 38092
    assert expected["topology_fixed"]["n"] == 64
    dumped = json.dumps(expected["arms"]["D"]["slices"])
    for banned in ("hello", "english", "phrase_program"):
        assert banned not in dumped


def test_prereg_stance() -> None:
    prereg = json.loads((REPO_ROOT / "docs" / "lineage.prereg.lock").read_text(encoding="utf-8"))
    assert prereg["product"] == "0.0.004"
    assert prereg["earned_next"] is False
    assert prereg["ex0s"] is None
    assert prereg["eligible_for_000005"] is False
    assert prereg["compute_frozen"] is False
    assert prereg["ancestor"]["neural_cortex_sha"] == "71bece5917893fae03c3a95c276cf93bc0e34fce6a7bfb6a99adf093bb7ebc08"
    assert prereg["ancestor"]["cortex_memory_sha"] == "fc3942efaffb8b18e891c545510aa4949b52c86c773c707036bbc6d162fe35d7"
    assert prereg["qual_seed_commitment"] == "b3cbb4fe0222f9973f1e74b1cb525eab9ee6003ada5ab88f948224b46489c23d"
    assert prereg["eval_seed_commitment"] == "b57363135bc8986811c21c5aaeaa4097e9c0c40e7527d61e586c3c422a757a38"
    assert prereg["qual_seed_commitment"] != prereg["eval_seed_commitment"]
    for key in ("P", "B", "W", "E", "N_wake", "N_replay", "gpu_hours"):
        assert key not in prereg.get("frozen_compute", {})


def test_fitness_gk_not_cliff() -> None:
    text = (REPO_ROOT / "docs" / "lineage_fitness_contract.md").read_text(encoding="utf-8")
    assert "F_search" in text
    assert "adult_lower_quartile" in text
    assert "δ_B = 0.05" in text
    assert "Larger causal margins" in text and "add fitness" in text
    assert "C6 is a remain-green constraint" in text
    assert "20260817" in text


def test_secrets_gitignored_and_commitments() -> None:
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docs/lineage_qual_secrets.sealed.json" in gi
    assert "docs/lineage_eval_secrets.sealed.json" in gi
    for name, commit in (
        ("lineage_qual_secrets.sealed.json", "b3cbb4fe0222f9973f1e74b1cb525eab9ee6003ada5ab88f948224b46489c23d"),
        ("lineage_eval_secrets.sealed.json", "b57363135bc8986811c21c5aaeaa4097e9c0c40e7527d61e586c3c422a757a38"),
    ):
        p = REPO_ROOT / "docs" / name
        if p.exists():
            sealed = json.loads(p.read_text(encoding="utf-8"))
            raw = bytes.fromhex(sealed["seed_hex"]) + bytes.fromhex(sealed["salt_hex"])
            assert hashlib.sha256(raw).hexdigest() == commit
            assert len(bytes.fromhex(sealed["seed_hex"])) == 32


def test_architecture_body_and_channels() -> None:
    text = (REPO_ROOT / "docs" / "lineage_architecture_contract.md").read_text(encoding="utf-8")
    assert "blind and deaf" in text
    assert "Never **copied**" in text
    assert "Not inherited in TM.0.24" in text
    assert "regardless of whether the scorer considers it useful" in text
    assert "FULLDEV.R7" in text
    assert "pair_seeds()" in text
    contract_v1 = REPO_ROOT / "docs" / "cortex_architecture_contract.md"
    assert sha(contract_v1) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2"


def test_phase0a_no_claim() -> None:
    lock = json.loads((REPO_ROOT / "docs" / "lineage_phase0a.lock").read_text(encoding="utf-8"))
    assert lock["capability_claim"] is False
    assert lock["neural_edit"] is False
    assert lock["product"] == "0.0.004"


def test_prereg_file_shas() -> None:
    prereg = json.loads((REPO_ROOT / "docs" / "lineage.prereg.lock").read_text(encoding="utf-8"))
    files = prereg["phase_a_file_shas"]
    for rel, digest in files.items():
        assert sha(REPO_ROOT / rel) == digest, rel


def test_codec_refuse_and_antithetic() -> None:
    import tempfile

    import numpy as np

    from three_memory.cortex_lineage import (
        antithetic_children,
        defaults_theta,
        f_search,
        g_k,
        refuse_audit,
        sample_birth_from_arm_d,
    )

    theta = defaults_theta("D")
    assert refuse_audit(theta, "D")["ok"]
    plus, minus = antithetic_children(theta, np.ones_like(theta), 0.01)
    assert np.allclose(theta, 0.5 * (plus + minus))
    with tempfile.TemporaryDirectory(prefix="lin_c_") as tmp:
        a = sample_birth_from_arm_d(theta, life_seed=9, s_dir=Path(tmp) / "a")
        b = sample_birth_from_arm_d(theta, life_seed=10, s_dir=Path(tmp) / "b")
        assert not torch_equal(a.W_rec, b.W_rec)
    assert f_search([0.2, 0.4, 0.6], 0.1, 0.2)[0] <= 0.4
    assert g_k(0.7, 0.5, 0.5, 0.6, 0.05, 0.05)
    assert not g_k(0.7, 0.68, 0.5, 0.6, 0.05, 0.05)


def torch_equal(x, y) -> bool:  # noqa: ANN001
    return bool((x - y).abs().max().item() < 1e-12)


def test_teacher_and_rest() -> None:
    import tempfile

    from experiments.run_tm024lineage import (
        live_once,
        make_synthetic_world,
        teacher_audit_identical_histories,
    )
    from three_memory.cortex_lineage import defaults_theta, sample_birth_from_arm_d

    assert teacher_audit_identical_histories()
    theta = defaults_theta("D")
    world = make_synthetic_world(11, teacher_convention=0)
    twin = make_synthetic_world(11, teacher_convention=1)
    assert world["teacher_pair"] != twin["teacher_pair"]
    assert world["handles"] == twin["handles"]
    with tempfile.TemporaryDirectory(prefix="lin_r_") as tmp:
        ag = sample_birth_from_arm_d(theta, life_seed=5, s_dir=Path(tmp) / "s")
        out = live_once(ag, world, n_wake=8, n_replay=3, teacher_seed=2)
        assert out["dev_epoch"] >= 1
        assert "st_idle" in ["st_idle"]


def test_no_stage_in_observe() -> None:
    from experiments.run_tm023cortex import build_observe

    ev = build_observe(
        interaction_token="i",
        source_token="s",
        ordered_symbols=["a"],
        observable_state=["st_idle"],
        body_state=[1.0, 0.0, 1.0, 0.0],
    )
    assert "stage" not in ev and "correct" not in ev and "homeostatic_delta" not in ev


def test_dev_triplets_disjoint_and_qual_refuse() -> None:
    from experiments.run_tm024lineage import next_dev_triplet, refuse_shared_qual

    t0 = next_dev_triplet(0, n_worlds=2)
    t1 = next_dev_triplet(1, n_worlds=2)
    seeds0 = set(t0["A"] + t0["B"] + t0["C"])
    seeds1 = set(t1["A"] + t1["B"] + t1["C"])
    assert len(seeds0) == 6
    assert seeds0.isdisjoint(seeds1)
    assert next_dev_triplet(0, n_worlds=2) == t0
    qual = refuse_shared_qual()
    assert qual["refuse"] is True
    assert qual["superiority_claim"] is False


def test_cluster_bootstrap_method() -> None:
    from three_memory.cortex_lineage import cluster_bootstrap_lower

    cells = [(0, 0, 0.7), (0, 1, 0.6), (1, 0, 0.65), (1, 1, 0.55)]
    lo = cluster_bootstrap_lower(cells, n_boot=200, seed=20260817)
    assert 0.0 <= lo <= 1.0
    assert lo <= 0.7


def test_compat_lock_if_present() -> None:
    p = REPO_ROOT / "docs" / "lineage_v27_default_compat.lock"
    if not p.exists():
        return
    lock = json.loads(p.read_text(encoding="utf-8"))
    assert lock["ancestor_neural_sha"] == "71bece5917893fae03c3a95c276cf93bc0e34fce6a7bfb6a99adf093bb7ebc08"
    assert sha(REPO_ROOT / "three_memory" / "neural_cortex.py") == lock["neural_cortex_sha"]
    assert lock["C4"]["ok"] and lock["C5"]["ok"] and lock["C6"]["ok"]
    assert lock["earned_next"] is False
    assert "_phrase" not in (REPO_ROOT / "three_memory" / "neural_cortex.py").read_text(encoding="utf-8")
    assert "phrase_program" not in (REPO_ROOT / "three_memory" / "neural_cortex.py").read_text(encoding="utf-8")


def test_scored_wall_if_present() -> None:
    wall_p = REPO_ROOT / "docs" / "lineage_wall.lock"
    scored_p = REPO_ROOT / "docs" / "lineage_scored.lock"
    if not wall_p.exists() or not scored_p.exists():
        return
    wall = json.loads(wall_p.read_text(encoding="utf-8"))
    scored = json.loads(scored_p.read_text(encoding="utf-8"))
    assert wall["L0_unlocked"] is False
    assert wall["product"] == "0.0.004"
    assert wall["earned_next"] is False
    assert wall["ex0s"] is None
    assert scored["qual_revealed"] is False
    assert scored["eval_revealed"] is False
    assert scored["failed_panels_reused"] is False


def main() -> None:
    test_phase_a_files()
    test_layout_rebuild()
    test_prereg_stance()
    test_fitness_gk_not_cliff()
    test_secrets_gitignored_and_commitments()
    test_architecture_body_and_channels()
    test_phase0a_no_claim()
    test_prereg_file_shas()
    test_codec_refuse_and_antithetic()
    test_teacher_and_rest()
    test_no_stage_in_observe()
    test_dev_triplets_disjoint_and_qual_refuse()
    test_cluster_bootstrap_method()
    test_compat_lock_if_present()
    test_scored_wall_if_present()
    print("test_tm024lineage: ok")


if __name__ == "__main__":
    main()
