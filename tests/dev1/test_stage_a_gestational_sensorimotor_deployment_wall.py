"""GSM deployment-wall protocol tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dev1.gsm_deployment_wall_controllers import (
    HORIZON_TICKS,
    UNCERTAINTY_MAX_PIN,
    choose_exact_one_step_valence,
    choose_exact_receding_horizon,
    controller_invariants,
)
from experiments.dev1.gsm_deployment_wall_life import (
    REQUIRED_ARMS,
    TELEMETRY_FIELDS,
    evaluate_deployment_wall_life,
    paired_life_rng_seed,
)
from experiments.dev1.search_gestational_sensorimotor_deployment_wall import apply_decision_ladder
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.nursery_v2.construction import construct_nursery_organism
from three_memory.dev1.nursery_v2.world import NurseryWorldV2

MECHANISM_SHA = "8129ccff11177263159f3d76342317288886c481"
PREREG = Path("docs/exos_dev1.stage_a_gestational_sensorimotor_deployment_wall.prereg.lock")


def test_mechanism_package_immutable_at_8129ccf():
    diff = subprocess.check_output(
        ["git", "diff", "--stat", "8129ccf", "--", "three_memory/dev1/gsm/"], text=True
    ).strip()
    assert diff == "", f"mechanism package drifted: {diff}"
    prereg = json.loads(PREREG.read_text())
    assert prereg["provenance"]["mechanism_sha"] == MECHANISM_SHA
    assert prereg["scored_run_authorized"] is False


def test_pins_horizon_and_uncertainty():
    inv = controller_invariants()
    assert inv["horizon_ticks_frozen"] == 3
    assert HORIZON_TICKS == 3
    assert UNCERTAINTY_MAX_PIN == 0.35
    prereg = json.loads(PREREG.read_text())
    assert prereg["thresholds"]["receding_horizon_ticks"] == 3
    assert prereg["thresholds"]["uncertainty_max"] == 0.35
    assert prereg["uncertainty_fallback"]["do_not_lower"] is True


def test_exact_controllers_measurement_only_no_live_mutation_no_expected_action():
    g = GenerativeGenome.small(embryonic_seed=0)
    org, _ = construct_nursery_organism(g, device=torch.device("cpu"))
    world = NurseryWorldV2(generative=g, world_seed="gsm_dw_exact_probe", episode_ticks=4)
    step = world.reset_episode(0)
    pos0 = world.body.state.position.detach().clone()
    tick0 = int(world.body.state.tick)
    c1 = choose_exact_one_step_valence(world.body, org.valence_circuit, device=torch.device("cpu"), world=world)
    assert c1.measurement_only is True
    assert c1.wrote_weights is False
    assert c1.accessed_expected_action is False
    assert torch.equal(pos0, world.body.state.position.detach())
    assert int(world.body.state.tick) == tick0
    c2 = choose_exact_receding_horizon(
        world.body, org.valence_circuit, device=torch.device("cpu"), horizon_ticks=3, world=world
    )
    assert c2.horizon_ticks == 3
    assert torch.equal(pos0, world.body.state.position.detach())
    assert int(world.body.state.tick) == tick0
    # Applying chosen motor is the world's job; choice itself must not step live body.
    _ = step


def test_exact_controller_refuses_horizon_sweep():
    g = GenerativeGenome.small(embryonic_seed=1)
    org, _ = construct_nursery_organism(g, device=torch.device("cpu"))
    world = NurseryWorldV2(generative=g, world_seed="gsm_dw_h_refuse", episode_ticks=4)
    world.reset_episode(0)
    try:
        choose_exact_receding_horizon(
            world.body, org.valence_circuit, device=torch.device("cpu"), horizon_ticks=4
        )
        assert False, "expected horizon pin refusal"
    except RuntimeError as e:
        assert "frozen" in str(e).lower() or "H=3" in str(e)


def test_exact_controller_cannot_write_weights_and_no_future_state_api():
    src = Path("experiments/dev1/gsm_deployment_wall_controllers.py").read_text()
    assert "optim" not in src.lower()
    assert "expected_action" in src  # refusal path
    assert "teacher_action" in src
    assert "backward(" not in src
    assert ".step(" in src  # probe only
    # Live body clone pattern required
    assert "_clone_probe" in src


def test_paired_randomness_and_matched_learned_checkpoints():
    seed = "gsm_dw_unit_matched_world"
    rng_a = paired_life_rng_seed(seed)
    rng_b = paired_life_rng_seed(seed)
    assert rng_a == rng_b
    gated = evaluate_deployment_wall_life(
        "learned_gated",
        f"{seed}:arm:learned_gated",
        n_episodes=1,
        episode_ticks=4,
        gestation_ticks=8,
        embryonic_seed=0,
        body_seed=1,
        life_rng_seed=rng_a,
    )
    forced = evaluate_deployment_wall_life(
        "learned_forced",
        f"{seed}:arm:learned_forced",
        n_episodes=1,
        episode_ticks=4,
        gestation_ticks=8,
        embryonic_seed=0,
        body_seed=1,
        life_rng_seed=rng_b,
    )
    assert gated.pre_gestation_checkpoint_hash == forced.pre_gestation_checkpoint_hash
    assert gated.life_record["paired_life_rng_seed"] == forced.life_record["paired_life_rng_seed"]
    assert gated.life_record["world_hash"] != forced.life_record["world_hash"]  # arm-tagged worlds
    assert forced.fraction_model_actions == 1.0
    assert forced.measurement_only is False
    assert gated.organism_candidate is True


def test_telemetry_fields_present_on_learned_and_exact():
    for arm in ("learned_gated", "exact_one_step_valence", "random_fallback"):
        m = evaluate_deployment_wall_life(
            arm,
            f"gsm_dw_telem_{arm}",
            n_episodes=2,
            episode_ticks=4,
            gestation_ticks=8,
            embryonic_seed=0,
            body_seed=1,
        )
        for field in TELEMETRY_FIELDS:
            if field == "synergy_histograms":
                assert m.life_record.get("synergy_histogram") is not None
            elif field == "synergy_histogram_early_to_late_change":
                assert m.life_record.get("synergy_histogram_early_to_late_change") is not None
            else:
                assert field in m.life_record, field
        assert m.life_record["uncertainty_max"] == 0.35
        assert m.plasticity_updates == 0
        assert m.life_record["weights_frozen_during_life"] is True
        if arm.startswith("exact") or arm == "random_fallback":
            assert m.measurement_only is True
            assert m.organism_candidate is False


def test_ladder_routing_decisive_order():
    thresh = {
        "predictive_min_train_end_in_zone": 0.25,
        "predictive_min_distance_reduction": 0.0,
        "active_beats_control_epsilon": 0.05,
        "fresh_world_min_validation_end_in_zone": 0.2,
        "min_fraction_model_actions": 0.5,
    }

    def arm(eiz, dd=0.1, frac=1.0, val=0.5):
        return {
            "train_end_in_zone_mean": eiz,
            "train_distance_reduction_mean": dd,
            "fraction_model_actions_mean": frac,
            "validation_end_in_zone_mean": val,
            "systematic_misprediction_risk": False,
        }

    setup = {"setup_reference_pass": True}
    cert = {"all_certified": True}
    # missing horizon
    arms = {
        "exact_one_step_valence": arm(0.1),
        "exact_receding_horizon": arm(0.5),
        "learned_forced": arm(0.5),
        "learned_gated": arm(0.5, frac=0.6),
        "random_fallback": arm(0.0),
    }
    assert (
        apply_decision_ladder(setup=setup, cert=cert, arms=arms, thresh=thresh, evidence_complete=True)
        == "missing_value_or_horizon_machinery"
    )
    # transfer fail
    arms["exact_one_step_valence"] = arm(0.5)
    arms["learned_forced"] = arm(0.1)
    assert (
        apply_decision_ladder(setup=setup, cert=cert, arms=arms, thresh=thresh, evidence_complete=True)
        == "learned_representation_does_not_transfer_online"
    )
    # trust/coverage fail
    arms["learned_forced"] = arm(0.5)
    arms["learned_gated"] = arm(0.5, frac=0.2)
    assert (
        apply_decision_ladder(setup=setup, cert=cert, arms=arms, thresh=thresh, evidence_complete=True)
        == "uncertainty_calibration_or_coverage_failure"
    )
    # causal fail vs random
    arms["learned_gated"] = arm(0.3, frac=0.6)
    arms["random_fallback"] = arm(0.3)
    assert (
        apply_decision_ladder(setup=setup, cert=cert, arms=arms, thresh=thresh, evidence_complete=True)
        == "model_not_causally_useful"
    )
    # pass
    arms["learned_gated"] = arm(0.5, frac=0.6, val=0.4)
    arms["random_fallback"] = arm(0.1)
    assert (
        apply_decision_ladder(setup=setup, cert=cert, arms=arms, thresh=thresh, evidence_complete=True)
        == "deployment_wall_pass"
    )


def test_fresh_partitions_and_required_arms():
    prereg = json.loads(PREREG.read_text())
    parts = prereg["seed_partitions"]
    scored = set(parts["discovery_world_seeds"]) | set(parts["validation_world_seeds"]) | set(
        parts["confirmation_seeds"]
    )
    excl = set(parts["excluded_seeds"])
    assert not (scored & excl)
    assert all("deployment_wall" in s for s in scored)
    assert "exos_dev1_gestational_sensorimotor_model_r1_world_001" in excl
    assert "exos_dev1_gestational_sensorimotor_model_r1_conf_001" in excl
    assert set(REQUIRED_ARMS) == {
        "learned_gated",
        "learned_forced",
        "exact_one_step_valence",
        "exact_receding_horizon",
        "random_fallback",
    }
