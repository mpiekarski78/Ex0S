from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.optimizers.latent_r1_2 import (
    LATENT_PARAM_ORDER,
    WORST_FITNESS,
    RewardBasedMetaGradientR12,
    apply_latent_to_genome,
    latent_from_genome,
    phenotype_from_latent,
    validate_phenotype,
)
from experiments.dev1.search_r1_2 import run_stage_a_r1_2_search
from three_memory.dev1.genome import DevGenome


def test_r1_2_transforms_enforce_domains():
    genome = DevGenome.default()
    latent = latent_from_genome(genome)
    params = phenotype_from_latent(latent)
    ok, reason = validate_phenotype(params)
    assert ok, reason
    assert 0.0 <= params["gamma"] < 1.0
    assert 0.0 <= params["eligibility_decay"] < 1.0
    assert 0.0 <= params["baseline_decay"] < 1.0
    assert params["learning_rate"] > 0.0
    assert params["hebbian_lr"] > 0.0


def test_r1_2_invalid_candidate_rejected_without_crashing():
    genome = DevGenome.default()
    opt = RewardBasedMetaGradientR12()
    proposed, meta = opt.propose(genome)
    assert proposed is not None
    ok = opt.update_after_training_lives(meta, 0.4)
    assert ok
    telem = opt.telemetry()
    assert telem["gradient_norm"] is not None
    assert telem["genome_step_norm"] is not None
    assert telem["genome_step_norm"] <= 1.0 + 1e-6


def test_r1_2_run_failed_json_survives_exception(tmp_path):
    with pytest.raises(RuntimeError, match="injected_r1_2_exception"):
        run_stage_a_r1_2_search(
            run_id="r12_fail_test",
            world_seeds=[f"r12_fail_world_{i:03d}" for i in range(1, 7)],
            confirmation_seeds=[f"r12_fail_conf_{i:03d}" for i in range(1, 5)],
            output_dir=str(tmp_path / "run"),
            inject_exception_after_started=True,
        )
    assert (tmp_path / "run" / "run_started.json").exists()
    assert (tmp_path / "run" / "run_failed.json").exists()


def test_r1_2_run_completed_json_exists_on_success(tmp_path):
    summary = run_stage_a_r1_2_search(
        run_id="r12_complete_test",
        world_seeds=[f"r12_ok_world_{i:03d}" for i in range(1, 7)],
        confirmation_seeds=[f"r12_ok_conf_{i:03d}" for i in range(1, 5)],
        meta_updates=1,
        evo_generations=1,
        output_dir=str(tmp_path / "run"),
    )
    assert (tmp_path / "run" / "run_started.json").exists()
    assert (tmp_path / "run" / "run_completed.json").exists()
    assert "outcome" in summary


def test_r1_2_same_latent_surface_for_both_optimizers():
    genome = DevGenome.default()
    latent = latent_from_genome(genome)
    assert latent.numel() == len(LATENT_PARAM_ORDER)
    g2 = apply_latent_to_genome(genome, latent)
    assert set(g2.credit_parameter_dict().keys()) == set(genome.credit_parameter_dict().keys())
