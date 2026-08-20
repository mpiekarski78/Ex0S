"""Nursery Body v2 package — engineering surface until certification freeze."""

from three_memory.dev1.nursery_v2.physics import BodyConfig, NurseryBodyV2
from three_memory.dev1.nursery_v2.synergies import (
    SYNERGY_REPORT_NAMES,
    expand_synergy_index_to_motor,
    synergy_projection_matrix,
)
from three_memory.dev1.nursery_v2.world import (
    NurseryWorldV2,
    analytic_reachability_report,
    max_useful_travel,
    reachability_chi,
)

__all__ = [
    "BodyConfig",
    "NurseryBodyV2",
    "NurseryWorldV2",
    "SYNERGY_REPORT_NAMES",
    "analytic_reachability_report",
    "expand_synergy_index_to_motor",
    "max_useful_travel",
    "reachability_chi",
    "synergy_projection_matrix",
]
