"""GSM multi-seed model-certification preflight (excluded from scored partitions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.dev1.gsm_life import evaluate_gsm_life
from experiments.dev1.gsm_model_certification import run_model_certification

PREFLIGHT_SEED = "gestational_sensorimotor_model_excluded_preflight_20260820"
ENG_SEEDS = (
    "gsm_eng_cert_seed_001",
    "gsm_eng_cert_seed_002",
    "gsm_eng_cert_seed_003",
)

SEALED_PREFIX = "exos_dev1_developmental_birth_r4_r2_"


def run_preflight(device: torch.device | None = None) -> dict[str, Any]:
    dev = device or torch.device("cpu")
    out: dict[str, Any] = {"preflight_seed": PREFLIGHT_SEED, "engineering_seeds": list(ENG_SEEDS)}
    certs = []
    for seed in ENG_SEEDS:
        assert not seed.startswith(SEALED_PREFIX)
        c = run_model_certification(
            seed, device=dev, n_episodes=16, episode_ticks=8, epochs=40
        )
        certs.append(c)
    out["certifications"] = [
        {"world_seed": c["world_seed"], "certified": c["certified"], "checks": c["certification_checks"]}
        for c in certs
    ]
    out["all_certified"] = all(c["certified"] for c in certs)

    # Paired sham/active predictive on preflight seed
    sham = evaluate_gsm_life(
        "sham_gestation",
        PREFLIGHT_SEED + ":sham",
        n_episodes=4,
        episode_ticks=8,
        body_seed=3,
        device=dev,
    )
    pred = evaluate_gsm_life(
        "predictive_gestation",
        PREFLIGHT_SEED + ":pred",
        n_episodes=4,
        episode_ticks=8,
        body_seed=3,
        device=dev,
    )
    shuffled = evaluate_gsm_life(
        "predictive_gestation_shuffled_consequences",
        PREFLIGHT_SEED + ":shuf",
        n_episodes=2,
        episode_ticks=8,
        body_seed=3,
        device=dev,
    )
    model_off = evaluate_gsm_life(
        "learned_model_off_at_action_selection",
        PREFLIGHT_SEED + ":off",
        n_episodes=2,
        episode_ticks=8,
        body_seed=3,
        device=dev,
    )
    out["paired"] = {
        "same_pre_gestation": sham.pre_gestation_checkpoint_hash == pred.pre_gestation_checkpoint_hash,
        "sham_model_updates": sham.model_updates,
        "pred_model_updates": pred.model_updates,
        "shuffled_model_updates": shuffled.model_updates,
        "model_off_fraction_model_actions": model_off.fraction_model_actions,
    }
    out["sealed_r4_r2_not_used"] = True
    out["ok"] = (
        out["all_certified"]
        and out["paired"]["same_pre_gestation"]
        and out["paired"]["sham_model_updates"] == 0
        and out["paired"]["pred_model_updates"] > 0
        and out["paired"]["model_off_fraction_model_actions"] == 0.0
    )
    return out


def main() -> None:
    report = run_preflight()
    out_dir = Path("runs/exos_dev1/stage_a_gestational_sensorimotor_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "excluded_model_certification_preflight.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"ok": report["ok"], "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
