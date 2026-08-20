"""Developmental Body Reference Wall search / scored entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.dev1.developmental_body_reference_wall import run_body_reference_wall
from three_memory.dev1.device import cuda_utilization_sample, dev1_device

PREREG = Path("docs/exos_dev1.stage_a_developmental_body_reference_wall.prereg.lock")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", action="store_true")
    p.add_argument("--world-seed", default="developmental_body_wall_unscored")
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--episode-ticks", type=int, default=16)
    p.add_argument(
        "--out",
        default="runs/exos_dev1/stage_a_developmental_body_reference_wall/unscored_smoke.json",
    )
    args = p.parse_args()

    thresholds = None
    out_path = Path(args.out)
    if args.scored:
        if not PREREG.exists():
            raise SystemExit("Wall prereg missing")
        prereg = json.loads(PREREG.read_text())
        if not prereg.get("scored_run_authorized"):
            raise SystemExit("Wall scored run not authorized")
        thresholds = {
            "min_final_comfort_rate": float(prereg["thresholds"]["min_final_comfort_rate"]),
            "min_distance_reduction": float(prereg["thresholds"]["min_distance_reduction"]),
            "min_margin_over_random": float(prereg["thresholds"]["min_margin_over_random"]),
        }
        args.world_seed = prereg["run_identity"]["primary_world_seed"]
        args.n_episodes = int(prereg["budget"]["n_episodes"])
        args.episode_ticks = int(prereg["budget"]["episode_ticks"])
        horizon = int(prereg["budget"]["model_based_horizon"])
        beam = int(prereg["budget"]["model_based_beam"])
        out_path = Path(prereg["run_identity"]["output_dir"]) / "wall_summary.json"
    else:
        horizon, beam = 16, 32

    dev = dev1_device(require_cuda=bool(args.scored))
    t0 = time.perf_counter()
    if args.scored:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        started = {
            "revision": "DevelopmentalBodyReferenceWall",
            "world_seed": args.world_seed,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(dev),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_utilization": cuda_utilization_sample(),
            "r4_r1_decision_unchanged": True,
            "db69d10_stands": True,
        }
        (out_path.parent / "run_started.json").write_text(json.dumps(started, indent=2))

    report = run_body_reference_wall(
        args.world_seed,
        n_episodes=args.n_episodes,
        episode_ticks=args.episode_ticks,
        device=dev,
        thresholds=thresholds,
        horizon=horizon,
        beam=beam,
    )
    report["wall_s"] = time.perf_counter() - t0
    report["device"] = str(dev)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    if args.scored:
        (out_path.parent / "run_completed.json").write_text(
            json.dumps(
                {
                    "decision_code": report["decision_code"],
                    "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "summary_path": str(out_path),
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(out_path),
                "decision_code": report["decision_code"],
                "wall_s": report["wall_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
