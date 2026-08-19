"""
EX0S-DEV1 transient working state (ρ).

ρ is cleared by EpisodeReset; it does not persist facts across resets.
It holds the current context, active inference state, and retrieved
content from H — all transient.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional

from three_memory.dev1.genome import DevGenome


class WorkingState:
    """
    Transient recurrent context state.

    May contain:
    - current context and active inference
    - retrieved content currently active from H

    Must not contain:
    - persistent facts
    - organism-lifetime memory
    """

    def __init__(self, genome: DevGenome, device: torch.device):
        self.device = device
        self.n_rel = genome.relational_ctx.n_units
        self.n_sensory = genome.sensory_ctx.n_units
        self.n_action = genome.action_ctx.n_units
        self.n_neuromod = genome.neuromod_dim

        self.reset()

    def reset(self) -> None:
        """Clear all transient state (EpisodeReset boundary)."""
        self.sensory_repr = torch.zeros(self.n_sensory, device=self.device)
        self.relational_repr = torch.zeros(self.n_rel, device=self.device)
        self.action_repr = torch.zeros(self.n_action, device=self.device)
        self.retrieved_content: Optional[torch.Tensor] = None
        self.step: int = 0

    def state_dict(self) -> dict:
        d: dict = {
            "sensory_repr": self.sensory_repr.cpu(),
            "relational_repr": self.relational_repr.cpu(),
            "action_repr": self.action_repr.cpu(),
            "step": self.step,
        }
        if self.retrieved_content is not None:
            d["retrieved_content"] = self.retrieved_content.cpu()
        return d

    def load_state_dict(self, d: dict) -> None:
        self.sensory_repr = d["sensory_repr"].to(self.device)
        self.relational_repr = d["relational_repr"].to(self.device)
        self.action_repr = d["action_repr"].to(self.device)
        self.step = d["step"]
        self.retrieved_content = d.get("retrieved_content")
        if self.retrieved_content is not None:
            self.retrieved_content = self.retrieved_content.to(self.device)

    def cleared_snapshot(self) -> dict:
        """Return a snapshot of state just before clearing (for EpisodeReset record)."""
        return {
            "working_state": {
                "sensory_repr": self.sensory_repr.cpu().tolist(),
                "relational_repr": self.relational_repr.cpu().tolist(),
                "action_repr": self.action_repr.cpu().tolist(),
                "step": self.step,
            }
        }
