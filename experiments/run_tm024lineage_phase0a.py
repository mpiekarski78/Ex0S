"""TM.0.24.LINEAGE Phase 0A — v27 throughput baseline. No neural edit. No capability claim.

REST is absent on v27; empty-symbol observes are a rest-opportunity proxy only.
This is a planning estimate, not a compute freeze.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.cortex_develop_life import (  # noqa: E402
    BODY0,
    LifeSeeds,
    apply_event,
    bind_life_actuators,
    curriculum_tokens,
    motor_latent,
)
from experiments.run_tm023cortex import make_cortex, torch_env  # noqa: E402
from three_memory.neural_cortex import NeuralCortex  # noqa: E402

OUT = REPO_ROOT / "docs" / "lineage_phase0a.lock"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.lock"


def _life_seeds(tag: int) -> LifeSeeds:
    base = 24_000_000 + tag * 17
    return LifeSeeds(
        pair_id=tag,
        role="main",
        seed_birth=base + 1,
        seed_registry=base + 2,
        seed_source=base + 3,
        seed_action=base + 4,
        seed_permute=base + 5,
        seed_motor=base + 6,
    )


def _make(seeds: LifeSeeds, tmp: Path, device: str) -> tuple[NeuralCortex, dict[str, str], dict[str, Any]]:
    ag = make_cortex(tmp / "s", genome=seeds.genome(), device=device)
    toks = curriculum_tokens(seeds)
    bind_life_actuators(ag, toks, seeds)
    return ag, toks, motor_latent(toks)


def _wake_ticks(ag: NeuralCortex, toks: dict[str, str], latent: dict[str, Any], n: int, prefix: str) -> dict[str, Any]:
    body = list(BODY0)
    state = ["st_idle"]
    ops: dict[str, int] = {}
    t0 = time.perf_counter()
    for i in range(n):
        out, state, body = apply_event(
            ag,
            ix=f"{prefix}_{i}",
            source="src_wake",
            symbols=[toks["a"], toks["b"]] if i % 3 else [toks["a"]],
            state=state,
            body=body,
            latent=latent,
        )
        op = str((out.get("action") or {}).get("op") or "?")
        ops[op] = ops.get(op, 0) + 1
    elapsed = time.perf_counter() - t0
    return {
        "n": n,
        "seconds": elapsed,
        "ticks_per_second": (n / elapsed) if elapsed > 0 else 0.0,
        "op_counts": ops,
    }


def _rest_proxy_ticks(ag: NeuralCortex, latent: dict[str, Any], n: int, prefix: str) -> dict[str, Any]:
    """Empty-symbol observe: rest-opportunity proxy. Not REST/replay (absent on v27)."""
    body = list(BODY0)
    state = ["st_idle"]
    t0 = time.perf_counter()
    for i in range(n):
        _, state, body = apply_event(
            ag,
            ix=f"{prefix}_{i}",
            source="src_rest",
            symbols=[],
            state=state,
            body=body,
            latent=latent,
        )
    elapsed = time.perf_counter() - t0
    return {
        "n": n,
        "seconds": elapsed,
        "ticks_per_second": (n / elapsed) if elapsed > 0 else 0.0,
        "note": "empty-symbol observe proxy; REST/replay not implemented on v27",
    }


def _life_seconds(device: str, n_wake: int, tag: int) -> float:
    seeds = _life_seeds(tag)
    with tempfile.TemporaryDirectory(prefix="tm024_0a_life_") as tmp:
        ag, toks, lat = _make(seeds, Path(tmp), device)
        t0 = time.perf_counter()
        _wake_ticks(ag, toks, lat, n_wake, prefix=f"life{tag}")
        return time.perf_counter() - t0


def _s_overhead(device: str) -> dict[str, Any]:
    seeds = _life_seeds(99)
    n = 200
    with tempfile.TemporaryDirectory(prefix="tm024_0a_s_") as tmp:
        p = Path(tmp)
        ag_disk, toks, lat = _make(seeds, p / "disk", device)
        disk = _wake_ticks(ag_disk, toks, lat, n, prefix="sdisk")
        ag_mem = make_cortex(None, genome=seeds.genome(), device=device)
        bind_life_actuators(ag_mem, toks, seeds)
        mem = _wake_ticks(ag_mem, toks, lat, n, prefix="smem")
    return {
        "n": n,
        "disk_s_ticks_per_second": disk["ticks_per_second"],
        "memory_only_ticks_per_second": mem["ticks_per_second"],
        "disk_over_memory_ratio": (
            disk["seconds"] / mem["seconds"] if mem["seconds"] > 0 else None
        ),
    }


def _vram() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.reset_peak_memory_stats()
    seeds = _life_seeds(7)
    with tempfile.TemporaryDirectory(prefix="tm024_0a_vram_") as tmp:
        ag, toks, lat = _make(seeds, Path(tmp), "cuda")
        _wake_ticks(ag, toks, lat, 80, prefix="vram")
        allocated = int(torch.cuda.memory_allocated())
        peak = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        name = torch.cuda.get_device_name(0)
    return {
        "cuda_available": True,
        "device_name": name,
        "allocated_bytes": allocated,
        "peak_allocated_bytes": peak,
        "reserved_bytes": reserved,
    }


def _cpu_gpu_divergence() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "skipped": True}
    seeds = _life_seeds(3)
    n = 40
    with tempfile.TemporaryDirectory(prefix="tm024_0a_div_") as tmp:
        cpu, toks, lat = _make(seeds, Path(tmp) / "cpu", "cpu")
        gpu, _, _ = _make(seeds, Path(tmp) / "gpu", "cuda")
        cpu_ops: list[str] = []
        gpu_ops: list[str] = []
        cpu_body = list(BODY0)
        gpu_body = list(BODY0)
        cpu_state = ["st_idle"]
        gpu_state = ["st_idle"]
        rho_gaps: list[float] = []
        for i in range(n):
            syms = [toks["a"], toks["b"]]
            cout, cpu_state, cpu_body = apply_event(
                cpu, ix=f"divc_{i}", source="src_div", symbols=syms, state=cpu_state, body=cpu_body, latent=lat
            )
            gout, gpu_state, gpu_body = apply_event(
                gpu, ix=f"divg_{i}", source="src_div", symbols=syms, state=gpu_state, body=gpu_body, latent=lat
            )
            cpu_ops.append(str((cout.get("action") or {}).get("op")))
            gpu_ops.append(str((gout.get("action") or {}).get("op")))
            cr = cpu.rho.detach().cpu().numpy()
            gr = gpu.rho.detach().cpu().numpy()
            rho_gaps.append(float(np.max(np.abs(cr - gr))))
    agree = sum(a == b for a, b in zip(cpu_ops, gpu_ops, strict=True))
    return {
        "n": n,
        "op_agree": agree,
        "op_agree_rate": agree / n,
        "rho_maxabs_p50": float(np.median(rho_gaps)),
        "rho_maxabs_p95": float(np.percentile(rho_gaps, 95)),
        "rho_maxabs_max": float(np.max(rho_gaps)),
        "note": "float64 CPU gold vs CUDA; not a capability score",
    }


def _analytical(t_tick: float) -> dict[str, Any]:
    """T_generation ≈ 2P × B × W × E × (N_wake + c_r N_replay) × t_tick. Not frozen."""

    def tgen(p: int, b: int, w: int, e: int, n_wake: int, n_replay: int, c_r: float) -> dict[str, Any]:
        steps = 2 * p * b * w * e * (n_wake + c_r * n_replay)
        seconds = steps * t_tick
        return {
            "P_antithetic_pairs": p,
            "B_births": b,
            "W_worlds": w,
            "E_epochs": e,
            "N_wake": n_wake,
            "N_replay": n_replay,
            "c_r": c_r,
            "steps": steps,
            "seconds": seconds,
            "hours": seconds / 3600.0,
            "days": seconds / 86400.0,
        }

    return {
        "formula": "T_generation ≈ 2P*B*W*E*(N_wake + c_r*N_replay)*t_tick",
        "t_tick_seconds_gpu_wake": t_tick,
        "illustrative_128x4x4x1_500plus25": tgen(128, 4, 4, 1, 500, 25, 1.0),
        "illustrative_four_epochs": tgen(128, 4, 4, 4, 500, 25, 1.0),
        "illustrative_smoke_16x1x1x1_80plus10": tgen(16, 1, 1, 1, 80, 10, 1.0),
        "illustrative_pilot_32x2x2x1_200plus15": tgen(32, 2, 2, 1, 200, 15, 1.0),
        "note": "Planning estimate only. Compute freeze is Phase 0B after the real engine exists.",
    }


def main() -> dict[str, Any]:
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    env = torch_env()
    cuda = bool(torch.cuda.is_available())
    gpu_device = "cuda" if cuda else "cpu"

    seeds = _life_seeds(1)
    with tempfile.TemporaryDirectory(prefix="tm024_0a_gpu_") as tmp:
        ag, toks, lat = _make(seeds, Path(tmp), gpu_device)
        gpu_wake = _wake_ticks(ag, toks, lat, 400 if cuda else 80, prefix="gwake")
        gpu_rest = _rest_proxy_ticks(ag, lat, 200 if cuda else 40, prefix="grest")

    cpu_wake: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="tm024_0a_cpu_") as tmp:
        ag, toks, lat = _make(_life_seeds(2), Path(tmp), "cpu")
        cpu_wake = _wake_ticks(ag, toks, lat, 80, prefix="cwake")

    life_n = [80, 200, 500]
    life_times = [_life_seconds(gpu_device, n, 10 + i) for i, n in enumerate(life_n)]
    complete_lives = [
        _life_seconds(gpu_device, 200, 20 + i) for i in range(5 if cuda else 3)
    ]
    complete_lives.sort()

    s_ov = _s_overhead(gpu_device)
    vram = _vram()
    div = _cpu_gpu_divergence()
    t_tick = 1.0 / gpu_wake["ticks_per_second"] if gpu_wake["ticks_per_second"] else None
    analytical = _analytical(t_tick) if t_tick else {}

    out = {
        "version": "TM.0.24.LINEAGE.PHASE0A",
        "lab": "TM.0.24.LINEAGE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "neural_edit": False,
        "rest_implemented": False,
        "ancestor": {
            "candidate": "docs/cortex.candidate.lock",
            "version": cand.get("version"),
            "neural_cortex_sha": cand.get("neural_cortex_sha"),
            "cortex_memory_sha": cand.get("cortex_memory_sha"),
            "git": subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            ).strip(),
        },
        "env": env,
        "gpu_wake": gpu_wake,
        "gpu_rest_proxy": gpu_rest,
        "cpu_wake": cpu_wake,
        "life_duration_seconds_by_n_wake": dict(zip([str(n) for n in life_n], life_times, strict=True)),
        "complete_life_n_wake": 200,
        "complete_life_seconds": complete_lives,
        "complete_life_p50": statistics.median(complete_lives),
        "complete_life_p95": (
            complete_lives[int(round(0.95 * (len(complete_lives) - 1)))]
        ),
        "s_overhead": s_ov,
        "vram": vram,
        "cpu_gpu_divergence": div,
        "analytical": analytical,
        "note": (
            "Phase 0A baseline on immutable v27. REST is a no-op/proxy. "
            "Dummy Arm D codec does not exist. Do not freeze P/B/W/E from this lock. "
            "Not a capability score. Product remains 0.0.004."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gpu_wake", "cpu_wake", "analytical", "vram")}, indent=2, default=str))
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
