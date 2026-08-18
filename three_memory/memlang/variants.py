"""MEMLANG-1 family variant grids. Hard-capped at 25 per family."""

from __future__ import annotations

from typing import Any


def variants_for(family: str, *, max_n: int = 25) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    etas = (0.01, 0.03, 0.05, 0.1, 0.2)
    if family == "slow_target":
        for i, eta in enumerate(etas * 5):
            out.append({"family": family, "name": f"slow_target_{i:02d}", "eta": float(eta), "seed": 1000 + i})
    elif family == "hebbian_delta":
        for i, eta in enumerate(etas * 5):
            out.append({"family": family, "name": f"hebbian_delta_{i:02d}", "eta": float(eta), "seed": 2000 + i})
    elif family == "lowrank_adapter":
        ranks = (1, 2, 4, 8, 16)
        i = 0
        for rank in ranks:
            for eta in etas:
                out.append(
                    {
                        "family": family,
                        "name": f"lowrank_r{rank}_{i:02d}",
                        "eta": float(eta),
                        "rank": int(rank),
                        "seed": 3000 + i,
                    }
                )
                i += 1
    elif family == "recurrent_consistency":
        leaks = (0.1, 0.2, 0.3, 0.5, 0.8)
        i = 0
        for leak in leaks:
            for eta in etas:
                out.append(
                    {
                        "family": family,
                        "name": f"recurrent_l{leak}_{i:02d}",
                        "eta": float(eta),
                        "leak": float(leak),
                        "seed": 4000 + i,
                    }
                )
                i += 1
    else:
        raise RuntimeError(family)
    return out[: int(max_n)]
