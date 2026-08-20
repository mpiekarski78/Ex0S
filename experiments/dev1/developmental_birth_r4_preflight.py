"""
Developmental Birth R4 preflight (unit / ownership / leakage / permutation / scale).

Excluded-seed scored preflight and numeric preregistration are later steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.developmental_birth_r4_life import (
    evaluate_ceiling_on_body_world,
    evaluate_matched_factorial,
    evaluate_r4_life,
)
from experiments.dev1.developmental_birth_r4_outer import MatchedOuterBudget, run_matched_es_smoke
from three_memory.dev1.development.construction import construct_post_growth_organism
from three_memory.dev1.development.generative_genome import GenerativeGenome
from three_memory.dev1.development.gestation import GestationMode, run_gestation


def ownership_leakage_check(tag: str = "r4_leak") -> dict[str, Any]:
    """Genome/gestation must not contain cue names or fixture answers."""
    g = GenerativeGenome.small(embryonic_seed=1)
    blob = json.dumps(g.to_dict())
    forbidden = ["cue_", "fixture", "expected_action", "validation_seed", "confirm_"]
    hits = [f for f in forbidden if f in blob]
    org, receipt = construct_post_growth_organism(g, device=torch.device("cpu"))
    # Runner behavioral score must not equal learning reward injection
    assert org.r4_use_organism_valence is True
    return {
        "tag": tag,
        "forbidden_hits": hits,
        "ok": len(hits) == 0,
        "genome_hash": receipt.generative_genome_hash,
        "pre_gestation_checkpoint_hash": receipt.pre_gestation_checkpoint_hash,
    }


def sham_vs_active_compute_match_check() -> dict[str, Any]:
    g = GenerativeGenome.small(embryonic_seed=2)
    g.gestation_ticks = 16
    org, _ = construct_post_growth_organism(g, device=torch.device("cpu"))
    sham_org, sham_r = run_gestation(org, g, GestationMode.SHAM, body_seed=7)
    act_org, act_r = run_gestation(org, g, GestationMode.ACTIVE, body_seed=7)
    return {
        "same_ticks": sham_r.ticks == act_r.ticks,
        "sham_plasticity_updates": sham_r.plasticity_updates,
        "active_plasticity_updates": act_r.plasticity_updates,
        "same_transcript_length_ticks": sham_r.ticks == act_r.ticks == g.gestation_ticks,
        "sham_no_credit": sham_r.plasticity_updates == 0,
        "active_has_credit": act_r.plasticity_updates > 0,
        "ok": (
            sham_r.ticks == act_r.ticks == g.gestation_ticks
            and sham_r.plasticity_updates == 0
            and act_r.plasticity_updates > 0
        ),
        "sham_post_hash": sham_r.post_gestation_checkpoint_hash,
        "active_post_hash": act_r.post_gestation_checkpoint_hash,
    }


def gestation_effect_audit(seeds: tuple[int, ...] = (3, 7, 11, 13, 17)) -> dict[str, Any]:
    """
    Confirm active gestation changes weights and lifetime learning, not merely hashes.
    Preferred-action divergence is seed-dependent; require it on at least one seed.
    """
    from three_memory.dev1.body.world import ClosedLoopGroundingWorld
    from three_memory.dev1.development.construction import construct_post_growth_organism
    from three_memory.dev1.development.generative_genome import GenerativeGenome
    from three_memory.dev1.development.gestation import GestationMode, run_gestation
    from experiments.dev1.developmental_birth_r4_life import evaluate_r4_life
    import torch.nn.functional as F

    rows = []
    for seed in seeds:
        g = GenerativeGenome.small(7)
        sham, _ = run_gestation(
            construct_post_growth_organism(g, device=torch.device("cpu"))[0],
            g,
            GestationMode.SHAM,
            body_seed=seed,
        )
        act, _ = run_gestation(
            construct_post_growth_organism(g, device=torch.device("cpu"))[0],
            g,
            GestationMode.ACTIVE,
            body_seed=seed,
        )
        dW = float((act.action_ctx.W_motor.weight.data - sham.action_ctx.W_motor.weight.data).norm())
        world = ClosedLoopGroundingWorld(g, world_seed=f"gest_audit_{seed}", device=torch.device("cpu"))

        def _argmax(org):
            org.valence_circuit.reset()
            org.episode_reset()
            step = world.reset_episode(0)
            org.observe(world.observation_from_step(step))
            _, logits = org.action_ctx(org.rho.relational_repr, org.rho.action_repr)
            return int(logits.argmax())

        as_ = _argmax(sham)
        aa = _argmax(act)
        ls = evaluate_r4_life(
            "sham_gestation",
            "r2_fixed_eprop_baseline",
            f"gest_audit_sham_{seed}",
            generative=GenerativeGenome.small(7),
            n_episodes=4,
            episode_ticks=6,
            embryonic_seed=7,
            body_seed=seed,
            life_rng_seed=11,
            use_teacher=False,
        )
        la = evaluate_r4_life(
            "active_gestation",
            "r2_fixed_eprop_baseline",
            f"gest_audit_act_{seed}",
            generative=GenerativeGenome.small(7),
            n_episodes=4,
            episode_ticks=6,
            embryonic_seed=7,
            body_seed=seed,
            life_rng_seed=11,
            use_teacher=False,
        )
        rows.append(
            {
                "seed": seed,
                "dW": dW,
                "argmax_diff": as_ != aa,
                "score_delta": la.mean_behavioral_score - ls.mean_behavioral_score,
                "acc_delta": la.treatment_accuracy - ls.treatment_accuracy,
            }
        )
    ok = (
        all(r["dW"] > 1e-4 for r in rows)
        and any(r["argmax_diff"] for r in rows)
        and any(abs(r["score_delta"]) > 1e-3 or abs(r["acc_delta"]) > 1e-6 for r in rows)
    )
    return {"ok": ok, "rows": rows}


def run_preflight(device: torch.device | None = None) -> dict[str, Any]:
    dev = device or torch.device("cpu")
    out: dict[str, Any] = {}
    out["ownership"] = ownership_leakage_check()
    out["sham_active"] = sham_vs_active_compute_match_check()
    out["gestation_effect"] = gestation_effect_audit()
    out["factorial"] = {
        k: {"acc": v.treatment_accuracy, "credit": v.credit, "development": v.development}
        for k, v in evaluate_matched_factorial(
            "r4_preflight_world", n_episodes=2, episode_ticks=4, embryonic_seed=0, device=dev
        ).items()
    }
    out["lifetime_plasticity_off"] = evaluate_r4_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "r4_preflight_lpo",
        n_episodes=2,
        episode_ticks=4,
        lifetime_plasticity_off=True,
        device=dev,
    ).plasticity_updates
    out["lsg_off"] = evaluate_r4_life(
        "active_gestation",
        "inherited_learning_signal_generator",
        "r4_preflight_lsg_off",
        n_episodes=2,
        episode_ticks=4,
        lsg_off=True,
        device=dev,
    ).treatment_accuracy
    out["ceiling"] = evaluate_ceiling_on_body_world(
        GenerativeGenome.small(), "r4_preflight_ceiling", n_episodes=2, episode_ticks=4, device=dev
    )
    out["matched_es"] = run_matched_es_smoke(
        "r4_preflight_es",
        MatchedOuterBudget(population=2, generations=1, n_episodes=2, episode_ticks=4),
        device=dev,
    )
    out["scale_smoke"] = evaluate_r4_life(
        "active_gestation",
        "r2_fixed_eprop_baseline",
        "r4_preflight_scale",
        generative=GenerativeGenome.small().with_size(2),
        n_episodes=1,
        episode_ticks=2,
        device=dev,
    ).generative_genome_hash
    out["ok"] = (
        out["ownership"]["ok"]
        and out["sham_active"]["ok"]
        and out["gestation_effect"]["ok"]
        and out["matched_es"]["matched"]
    )
    return out


def main() -> None:
    report = run_preflight()
    out_dir = Path("runs/exos_dev1/stage_a_developmental_birth_r4")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preflight_smoke.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"ok": report["ok"], "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
