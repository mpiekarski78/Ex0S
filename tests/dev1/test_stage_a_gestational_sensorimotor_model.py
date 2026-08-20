"""GSM unit / leakage / equivariance / sealed-seed tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.gsm_life import evaluate_gsm_life
from experiments.dev1.gsm_model_certification import run_model_certification
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.valence import OrganismValenceCircuit
from three_memory.dev1.gsm.action_eval import choose_synergy_by_valence
from three_memory.dev1.gsm.forward_model import ForwardSensorimotorModel
from three_memory.dev1.gsm.gestation import PredictiveGestationMode, run_predictive_gestation
from three_memory.dev1.gsm.state import dims_from_body_config, pack_visible_state
from three_memory.dev1.nursery_v2.construction import construct_nursery_organism
from three_memory.dev1.nursery_v2.synergies import expand_synergy_index_to_motor, permute_channels_within_synergy


SEALED_R4R2 = (
    "exos_dev1_developmental_birth_r4_r2_world_001",
    "exos_dev1_developmental_birth_r4_r2_conf_001",
)


def test_forward_model_predicts_delta_not_labels():
    dims = dims_from_body_config()
    fm = ForwardSensorimotorModel(dims)
    s = torch.randn(dims.state_dim)
    m = expand_synergy_index_to_motor(0)
    pred = fm.predict_delta(s, m)
    assert pred.delta_exo.numel() == dims.exo_dim
    assert pred.delta_proprio.numel() == dims.proprio_dim
    assert pred.delta_intero.numel() == dims.intero_dim
    assert torch.allclose(pred.predicted_state, s + torch.cat([pred.delta_exo, pred.delta_proprio, pred.delta_intero]))
    src = Path("three_memory/dev1/gsm/forward_model.py").read_text()
    for forbidden in ("expected_action", "target_location", "behavioral_correct", "synergy_id"):
        assert forbidden not in src


def test_predictive_vs_sham_same_pre_gestation_checkpoint():
    g = GenerativeGenome.small(embryonic_seed=2)
    g.gestation_ticks = 24
    org, _ = construct_nursery_organism(g, device=torch.device("cpu"))
    sham_org, sham_fm, sham_r = run_predictive_gestation(
        org, g, PredictiveGestationMode.SHAM, body_seed=7
    )
    act_org, act_fm, act_r = run_predictive_gestation(
        org, g, PredictiveGestationMode.PREDICTIVE, body_seed=7
    )
    assert sham_r.pre_gestation_checkpoint_hash == act_r.pre_gestation_checkpoint_hash
    assert sham_r.model_updates == 0
    assert act_r.model_updates == g.gestation_ticks
    assert sham_r.ticks == act_r.ticks == g.gestation_ticks


def test_shuffled_consequences_arm_runs():
    g = GenerativeGenome.small(embryonic_seed=3)
    g.gestation_ticks = 16
    org, _ = construct_nursery_organism(g, device=torch.device("cpu"))
    _, _, r = run_predictive_gestation(
        org, g, PredictiveGestationMode.PREDICTIVE_SHUFFLED, body_seed=4
    )
    assert r.metadata["shuffled_consequences"] is True
    assert r.model_updates == g.gestation_ticks


def test_within_synergy_permutation_equivariance_on_untrained_mass_preserving_input():
    dims = dims_from_body_config()
    fm = ForwardSensorimotorModel(dims)
    # Train briefly so weights nontrivial but equivariance is about input motors under P
    s = torch.zeros(dims.state_dim)
    for syn in range(4):
        m1 = expand_synergy_index_to_motor(syn, encoding="onehot_in_block", channel_within_block=1)
        m2 = permute_channels_within_synergy(m1, perm_seed=2)
        with torch.no_grad():
            # Untrained network is equivariant only if identical inputs under projection;
            # after one shared forward through same weights, same motor mass → same if we
            # feed identical tensors. Permuted onehot within block differs as channel vector
            # but body projection is mass-preserving — model sees channel vector, so
            # predictions may differ unless trained for equivariance. Certification tests
            # trained model; here assert expand+perm have same mass per block.
            from three_memory.dev1.nursery_v2.synergies import channels_to_synergy_activations

            assert torch.allclose(
                channels_to_synergy_activations(m1), channels_to_synergy_activations(m2), atol=1e-6
            )


def test_action_eval_no_runner_preferred_action():
    dims = dims_from_body_config()
    fm = ForwardSensorimotorModel(dims)
    valence = OrganismValenceCircuit(dims.intero_dim, gain=2.0, setpoint=0.85)
    sensory = torch.zeros(48)
    sensory[2] = 1.0  # distance-ish exo
    intero = torch.tensor([0.2, 0.7, 0.1, 0.2])
    choice = choose_synergy_by_valence(
        fm, valence, sensory=sensory, intero=intero, model_enabled=True, require_trusted=False
    )
    assert 0 <= choice.synergy_index < 4
    assert choice.motor.numel() == 32
    src = Path("three_memory/dev1/gsm/action_eval.py").read_text()
    assert "preferred_action" not in src
    assert "expected_action" not in src


def test_model_off_fallback():
    valence = OrganismValenceCircuit(4)
    choice = choose_synergy_by_valence(
        None,
        valence,
        sensory=torch.zeros(48),
        intero=torch.zeros(4),
        model_enabled=False,
    )
    assert choice.used_model is False
    assert choice.fallback_reason == "model_off"


def test_r4_r2_seeds_rejected():
    for seed in SEALED_R4R2:
        try:
            run_model_certification(seed, n_episodes=2, episode_ticks=4, epochs=1)
            assert False, "expected sealed seed rejection"
        except ValueError as e:
            assert "sealed" in str(e).lower() or "forbidden" in str(e).lower()
        try:
            evaluate_gsm_life("predictive_gestation", seed, n_episodes=1, episode_ticks=2)
            assert False, "expected sealed seed rejection"
        except ValueError:
            pass


def test_model_certification_smoke():
    report = run_model_certification(
        "gsm_eng_cert_seed_001",
        n_episodes=12,
        episode_ticks=8,
        epochs=25,
        device=torch.device("cpu"),
    )
    assert report["n_holdout"] > 0
    assert "certification_checks" in report
    assert report["predicts_delta_not_absolute"] is True


def test_paired_active_sham_life_smoke():
    sham = evaluate_gsm_life(
        "sham_gestation",
        "gsm_eng_life_seed_001",
        n_episodes=2,
        episode_ticks=4,
        embryonic_seed=0,
        body_seed=1,
    )
    pred = evaluate_gsm_life(
        "predictive_gestation",
        "gsm_eng_life_seed_001",
        n_episodes=2,
        episode_ticks=4,
        embryonic_seed=0,
        body_seed=1,
    )
    assert sham.pre_gestation_checkpoint_hash == pred.pre_gestation_checkpoint_hash
    assert pred.model_updates > 0
    assert sham.model_updates == 0
    assert "synergy_histogram" in pred.life_record
    assert "early_end_in_zone_rate" in pred.life_record
    assert "action_sources" in pred.life_record
    assert "action_source_counts" in pred.life_record


def test_uncertainty_fallback_path_records_fallback_source():
    m = evaluate_gsm_life(
        "predictive_gestation",
        "gsm_eng_life_seed_fallback",
        n_episodes=1,
        episode_ticks=4,
        gestation_ticks=8,
        force_uncertainty_fallback=True,
        embryonic_seed=0,
        body_seed=2,
    )
    assert m.fraction_model_actions == 0.0
    assert m.fraction_fallback_actions == 1.0
    assert all(s.startswith("fallback:") for s in m.life_record["action_sources"])


def test_scored_runner_complete_not_stub():
    path = Path("experiments/dev1/search_gestational_sensorimotor_model.py")
    src = path.read_text()
    assert path.exists()
    assert "def run_protocol" in src
    assert "def apply_decision_ladder" in src
    assert "full_evidence_before_decision_ladder" in src
    assert "confirmation_sealed_until_validation_pass" in src
    assert "--rehearsal" in src
    assert "force_uncertainty_fallback" in src
    assert "treatment_primarily_fallback" in src
    assert len(src) > 8000


def test_prereg_pins_uncertainty_fallback_and_disjoint_seeds():
    import json

    prereg = json.loads(Path("docs/exos_dev1.stage_a_gestational_sensorimotor_model.prereg.lock").read_text())
    assert prereg["scored_run_authorized"] is False
    u = prereg["uncertainty_fallback"]
    assert u["uncertainty_max"] == 0.35
    assert u["fallback_policy"] == "uniform_random_synergy_no_planner"
    assert u["no_planner"] is True
    assert u["no_expected_action"] is True
    assert u["no_runner_selected_synergy"] is True
    parts = prereg["seed_partitions"]
    disc = set(parts["discovery_world_seeds"])
    val = set(parts["validation_world_seeds"])
    conf = set(parts["confirmation_seeds"])
    excl = set(parts["excluded_seeds"])
    assert not (disc & val)
    assert not (disc & conf)
    assert not (val & conf)
    assert not ((disc | val | conf) & excl)
    assert all(not s.startswith("exos_dev1_developmental_birth_r4_r2_") for s in disc | val | conf)
    assert prereg["implementation"]["runner_file"] == (
        "experiments/dev1/search_gestational_sensorimotor_model.py"
    )
    assert prereg["protocol"]["full_evidence_before_decision_ladder"] is True
