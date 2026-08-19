from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.scaffold_r2 import (
    ContinuousScaffoldPhenotype,
    TopologyScaffoldPhenotype,
    apply_scaffold_to_organism,
    run_scaffold_sensitivity_preflight,
    scaffold_extremes,
)
from experiments.dev1.search_r2 import run_stage_a_r2_search
from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism


def test_r2_scaffold_preflight_detects_phenotype_motion():
    genome = DevGenome.default()
    result = run_scaffold_sensitivity_preflight(genome, "reward_baseline_three_factor")
    assert result.decision_code in {"scaffold_sensitivity_pass", "scaffold_sensitivity_fail"}
    assert "recurrent_dynamics_delta" in result.metrics
    assert "motor_basis_separability_delta" in result.metrics
    assert "phenotype_delta_from_genome_delta" in result.metrics


def test_r2_scaffold_extremes_change_birth_phenotype():
    genome = DevGenome.default()
    low, high = scaffold_extremes()
    org_low = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    org_high = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    apply_scaffold_to_organism(org_low, low, TopologyScaffoldPhenotype("dense"))
    apply_scaffold_to_organism(org_high, high, TopologyScaffoldPhenotype("dense"))
    assert org_low._r2_scaffold_hash != org_high._r2_scaffold_hash
    assert float(org_low.action_ctx.W_motor.weight.norm().item()) != float(org_high.action_ctx.W_motor.weight.norm().item())


def test_r2_topology_motif_changes_weight_structure():
    genome = DevGenome.default()
    continuous = ContinuousScaffoldPhenotype()
    dense = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    block = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
    apply_scaffold_to_organism(dense, continuous, TopologyScaffoldPhenotype("dense"))
    apply_scaffold_to_organism(block, continuous, TopologyScaffoldPhenotype("block"))
    dense_zeros = (dense.action_ctx.W_motor.weight == 0).sum().item()
    block_zeros = (block.action_ctx.W_motor.weight == 0).sum().item()
    assert block_zeros > dense_zeros


def test_r2_runner_writes_completed_ledger(tmp_path):
    summary = run_stage_a_r2_search(
        run_id="r2_smoke",
        world_seeds=[f"r2_smoke_world_{i:03d}" for i in range(1, 7)],
        confirmation_seeds=[f"r2_smoke_conf_{i:03d}" for i in range(1, 5)],
        output_dir=str(tmp_path / "run"),
        meta_updates=1,
        evo_generations=1,
        population_size=2,
    )
    assert (tmp_path / "run" / "run_started.json").exists()
    assert (tmp_path / "run" / "run_completed.json").exists()
    assert "outcome" in summary
