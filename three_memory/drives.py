"""Innate drives and write/retrieve rules. Frozen; not 'survive/reproduce' slogans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriveThresholds:
    novelty_write: float = 0.35
    integrity_write: float = 0.5
    retrieve_min_novelty: float = 0.0


class InnateDrives:
    """
    Proximate drives (species prior):
    - novelty / prediction-error: how surprising is this embed vs ρ
    - integrity-cost: how costly was the last outcome (failure hurts)
    """

    def __init__(self, thresholds: DriveThresholds | None = None):
        self.thresholds = thresholds or DriveThresholds()

    def novelty(self, embed: np.ndarray, predicted: np.ndarray) -> float:
        e = np.asarray(embed, dtype=np.float64)
        p = np.asarray(predicted, dtype=np.float64)
        err = float(np.linalg.norm(e - p))
        # Soft saturate so thresholds are stable across scales.
        return float(np.tanh(err))

    def integrity_cost(self, success: bool | None, harm: float = 0.0) -> float:
        if success is None:
            return float(harm)
        return float(0.0 if success else 1.0) + float(harm)

    def should_write(self, novelty: float, integrity: float) -> bool:
        return novelty >= self.thresholds.novelty_write or integrity >= self.thresholds.integrity_write

    def should_collect(self, n_store_hits: int, n_world_hits: int) -> bool:
        """Take from W only on an S miss. Frozen; not learn-to-learn."""
        return n_store_hits == 0 and n_world_hits > 0
