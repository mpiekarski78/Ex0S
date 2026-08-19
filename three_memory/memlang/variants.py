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
    elif family == "feedback_invariance":
        i = 0
        for max_k in (4, 8):
            for spawn in (0.2, 0.5):
                for mix in (0.5, 0.75, 0.9):
                    for eta in (0.05, 0.15):
                        if i >= 25:
                            break
                        out.append(
                            {
                                "family": family,
                                "name": f"finv_{i:02d}",
                                "eta": float(eta),
                                "sep": 0.08,
                                "mix": float(mix),
                                "max_k": int(max_k),
                                "spawn": float(spawn),
                                "seed": 5000 + i,
                            }
                        )
                        i += 1
                    if i >= 25:
                        break
                if i >= 25:
                    break
            if i >= 25:
                break
        while i < 25:
            out.append(
                {
                    "family": family,
                    "name": f"finv_{i:02d}",
                    "eta": 0.1,
                    "sep": 0.15,
                    "mix": 0.9,
                    "max_k": 6,
                    "spawn": 0.35,
                    "seed": 5000 + i,
                }
            )
            i += 1
    elif family == "dual_timescale":
        i = 0
        for mix in (0.3, 0.7, 0.9):
            for sl in (0.005, 0.02, 0.08):
                for eta in (0.03, 0.1, 0.2):
                    if i >= 25:
                        break
                    out.append(
                        {
                            "family": family,
                            "name": f"dual_{i:02d}",
                            "eta_fast": float(eta),
                            "eta_slow": float(sl),
                            "mix": float(mix),
                            "seed": 6000 + i,
                        }
                    )
                    i += 1
                if i >= 25:
                    break
            if i >= 25:
                break
        while i < 25:
            out.append({"family": family, "name": f"dual_{i:02d}", "eta_fast": 0.05, "eta_slow": 0.01, "mix": 0.85, "seed": 6000 + i})
            i += 1
    elif family == "prediction_error":
        stores = ("pred", "target", "mix")
        i = 0
        for store in stores:
            for eta in etas:
                out.append({"family": family, "name": f"perr_{i:02d}", "eta": float(eta), "store": store, "seed": 7000 + i})
                i += 1
        while i < 25:
            out.append({"family": family, "name": f"perr_{i:02d}", "eta": 0.05, "store": "target", "seed": 7000 + i})
            i += 1
    elif family == "evolved_plasticity":
        specs = [
            {"hebb": 1.0, "anti": 0.0, "oja": 0.0, "bcm": 0.0, "mix": 0.5},
            {"hebb": 1.0, "anti": 0.3, "oja": 0.0, "bcm": 0.0, "mix": 0.5},
            {"hebb": 1.0, "anti": 0.0, "oja": 0.2, "bcm": 0.0, "mix": 0.5},
            {"hebb": 0.5, "anti": 0.0, "oja": 0.0, "bcm": 0.2, "mix": 0.5},
            {"hebb": 1.0, "anti": 0.2, "oja": 0.1, "bcm": 0.1, "mix": 0.8},
        ]
        i = 0
        for spec in specs:
            for eta in etas:
                row = {"family": family, "name": f"evo_{i:02d}", "eta": float(eta), "norm": 1.0, "seed": 8000 + i}
                row.update(spec)
                out.append(row)
                i += 1
    elif family == "latent_manifold":
        i = 0
        for k in (2, 4, 8, 16, 32):
            for mix in (0.2, 0.5, 0.9):
                if i >= 25:
                    break
                for eta in (0.05,):
                    out.append({"family": family, "name": f"lat_k{k}_{i:02d}", "eta": float(eta), "k": int(k), "mix": float(mix), "seed": 9000 + i})
                    i += 1
            if i >= 25:
                break
        while i < 25:
            out.append({"family": family, "name": f"lat_k4_{i:02d}", "eta": 0.1, "k": 4, "mix": 0.9, "seed": 9000 + i})
            i += 1
    elif family == "slow_feature":
        i = 0
        for sep in (0.02, 0.1, 0.3):
            for eta in etas:
                out.append({"family": family, "name": f"sfa_{i:02d}", "eta": float(eta), "sep": float(sep), "seed": 10000 + i})
                i += 1
        while i < 25:
            out.append({"family": family, "name": f"sfa_{i:02d}", "eta": 0.05, "sep": 0.1, "seed": 10000 + i})
            i += 1
    elif family == "efference_copy":
        i = 0
        for pure in (True, False):
            for eta in (0.05, 0.2, 0.5, 1.0):
                for spawn in (0.2, 0.5, 0.8):
                    if i >= 25:
                        break
                    out.append(
                        {
                            "family": family,
                            "name": f"eff_{i:02d}",
                            "eta": float(eta),
                            "pure": bool(pure),
                            "max_k": 8,
                            "spawn": float(spawn),
                            "sep": 0.08,
                            "seed": 11000 + i,
                        }
                    )
                    i += 1
                if i >= 25:
                    break
            if i >= 25:
                break
        while i < 25:
            out.append({"family": family, "name": f"eff_{i:02d}", "eta": 1.0, "pure": True, "max_k": 4, "spawn": 0.35, "sep": 0.05, "seed": 11000 + i})
            i += 1
    elif family == "rho_cluster":
        i = 0
        for mix in (0.3, 0.5, 0.7, 0.85, 0.92):
            for eta in etas:
                out.append(
                    {
                        "family": family,
                        "name": f"rho_{i:02d}",
                        "eta": float(eta),
                        "mix": float(mix),
                        "max_k": 8,
                        "spawn": 0.35,
                        "sep": 0.08,
                        "seed": 12000 + i,
                    }
                )
                i += 1
    elif family == "contrastive_rho":
        i = 0
        for mix in (0.2, 0.4, 0.6, 0.8, 0.9):
            for eta in etas:
                out.append({"family": family, "name": f"crho_{i:02d}", "eta": float(eta), "sep": 0.05, "mix": float(mix), "seed": 13000 + i})
                i += 1
    elif family == "cluster_sfa":
        i = 0
        for mix in (0.2, 0.4, 0.6, 0.8, 0.9):
            for eta in etas:
                out.append({"family": family, "name": f"csfa_{i:02d}", "eta": float(eta), "mix": float(mix), "seed": 14000 + i})
                i += 1
    elif family == "kmeans_rho":
        i = 0
        for spawn in (0.5, 0.75, 0.9, 0.95, 0.99):
            for mix in (0.4, 0.6, 0.8, 0.9, 0.92):
                out.append(
                    {
                        "family": family,
                        "name": f"km_{i:02d}",
                        "eta": 0.15,
                        "mix": float(mix),
                        "k": 4,
                        "spawn": float(spawn),
                        "seed": 15000 + i,
                    }
                )
                i += 1
    elif family == "sep_cluster":
        i = 0
        for spawn in (0.7, 0.85, 0.92, 0.97, 0.99):
            for mix in (0.4, 0.6, 0.8, 0.9, 0.92):
                out.append(
                    {
                        "family": family,
                        "name": f"sep_{i:02d}",
                        "eta": 0.15,
                        "mix": float(mix),
                        "max_k": 4,
                        "spawn": float(spawn),
                        "sep": 0.12,
                        "seed": 16000 + i,
                    }
                )
                i += 1
    elif family == "sticky_sep":
        i = 0
        for eta in (0.05, 0.1, 0.2, 0.35, 0.5):
            for mix in (0.35, 0.5, 0.65, 0.8, 0.92):
                out.append(
                    {
                        "family": family,
                        "name": f"sticky_{i:02d}",
                        "eta": float(eta),
                        "mix": float(mix),
                        "max_k": 4,
                        "spawn": 0.99,
                        "sep": 0.0,
                        "seed": 17000 + i,
                    }
                )
                i += 1
    elif family == "motor_cluster":
        i = 0
        for spawn in (0.2, 0.4, 0.6, 0.8, 0.9):
            for mix in (0.35, 0.5, 0.65, 0.8, 0.92):
                out.append(
                    {
                        "family": family,
                        "name": f"mcl_{i:02d}",
                        "eta": 0.25,
                        "mix": float(mix),
                        "spawn": float(spawn),
                        "max_k": 8,
                        "seed": 18000 + i,
                    }
                )
                i += 1
    elif family == "whitening":
        i = 0
        for eps in (1e-4, 1e-3, 1e-2, 0.05, 0.1):
            for eta in etas:
                out.append({"family": family, "name": f"zca_{i:02d}", "eta": float(eta), "eps": float(eps), "seed": 19000 + i})
                i += 1
    else:
        raise RuntimeError(family)
    return out[: int(max_n)]
