"""Independent behavioral gates for Nursery Body v2 / R4-R2.

Tick-fraction comfort is retired as a primary scored gate. Longer successful
approaches must not be penalized by spending more ticks outside the zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BehavioralEpisodeGates:
    """Per-episode independent gates (not derived from tick-fraction comfort)."""

    ever_reached: bool
    end_in_zone: bool
    start_distance: float
    end_distance: float

    @property
    def distance_reduction(self) -> float:
        return float(self.start_distance) - float(self.end_distance)


def aggregate_behavioral_gates(episodes: list[BehavioralEpisodeGates]) -> dict[str, float]:
    n = max(1, len(episodes))
    return {
        "n_episodes": float(len(episodes)),
        "ever_reached_rate": sum(1.0 for e in episodes if e.ever_reached) / n,
        "end_in_zone_rate": sum(1.0 for e in episodes if e.end_in_zone) / n,
        "mean_start_distance": sum(e.start_distance for e in episodes) / n,
        "mean_end_distance": sum(e.end_distance for e in episodes) / n,
        "distance_reduction": sum(e.distance_reduction for e in episodes) / n,
        "tick_fraction_comfort_retired": 1.0,
    }


def best_of_ac_initializations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregation: best of frozen initializations by end_in_zone, then distance_reduction.

    Does not retry until a ceiling passes — caller must supply a fixed row set.
    """
    if not rows:
        raise ValueError("AC initialization battery produced no rows")
    ranked = sorted(
        rows,
        key=lambda r: (
            float(r.get("end_in_zone_rate", r.get("episode_success_rate", 0.0))),
            float(r.get("distance_reduction", 0.0)),
        ),
        reverse=True,
    )
    best = dict(ranked[0])
    best["aggregation"] = "best_of_three_by_end_in_zone_then_distance_reduction"
    best["n_initializations_run"] = len(rows)
    best["initialization_seeds_run"] = [int(r.get("init_seed", i)) for i, r in enumerate(rows)]
    return best


# Frozen AC retrain tolerance (certification + R4-R2 reference ceiling).
AC_RETRAIN_N_INITIALIZATIONS = 3
AC_RETRAIN_INITIALIZATION_SEEDS: tuple[int, ...] = (0, 1, 2)
AC_RETRAIN_AGGREGATION = "best_of_three_by_end_in_zone_then_distance_reduction"
AC_RETRAIN_STOP = "always_run_exactly_three_initializations_then_stop"
