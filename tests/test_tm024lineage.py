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
    ):
        assert (REPO_ROOT / rel).is_file(), rel
    assert not (REPO_ROOT / "docs" / "lineage_world_generator.lock").exists()
    assert not (REPO_ROOT / "docs" / "lineage_engine.candidate.lock").exists()


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
    assert prereg["ancestor"]["neural_cortex_sha"] == sha(REPO_ROOT / "three_memory" / "neural_cortex.py")
    assert prereg["ancestor"]["cortex_memory_sha"] == sha(REPO_ROOT / "three_memory" / "cortex_memory.py")
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


def main() -> None:
    test_phase_a_files()
    test_layout_rebuild()
    test_prereg_stance()
    test_fitness_gk_not_cliff()
    test_secrets_gitignored_and_commitments()
    test_architecture_body_and_channels()
    test_phase0a_no_claim()
    test_prereg_file_shas()
    print("test_tm024lineage: ok")


if __name__ == "__main__":
    main()
