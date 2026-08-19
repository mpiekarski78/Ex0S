"""
EX0S-DEV1 neuromodulatory system.

Computes reward/prediction-error gating signals and eligibility traces
used by the cortical plasticity rules.

Allowed signals produced here:
- Scalar reward/advantage (gate only, never an answer identifier)
- Prediction error (temporal-difference or direct)
- Novelty / surprise estimate
- Conflict / ambiguity signal
- Replay gate (consolidation trigger)

These are passed to plasticity rules; they never write directly to W or H.
No gradient updates are applied during an evaluated life.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from three_memory.dev1.genome import DevGenome


class NeuromodController(nn.Module):
    """
    Lightweight neuromodulatory controller.

    Input: concatenated population activation (sensory + relational + action)
    Output: modulation signals for plasticity gating.
    """

    def __init__(self, genome: DevGenome, device: torch.device):
        super().__init__()
        in_dim = genome.sensory_ctx.n_units + genome.relational_ctx.n_units + genome.action_ctx.n_units
        self.fc = nn.Linear(in_dim, genome.neuromod_dim, bias=False)
        self.reward_gate = nn.Linear(genome.neuromod_dim, 1, bias=True)
        self.novelty_gate = nn.Linear(genome.neuromod_dim, 1, bias=True)
        self.consolidation_gate = nn.Linear(genome.neuromod_dim, 1, bias=True)
        self.to(device)
        self.device = device

        self._baseline = torch.zeros(1, device=device)
        self._last_pred = torch.zeros(1, device=device)

    def forward(
        self,
        sensory: torch.Tensor,
        relational: torch.Tensor,
        action: torch.Tensor,
        reward: float,
    ) -> dict:
        x = torch.cat([sensory, relational, action]).unsqueeze(0)
        h = torch.relu(self.fc(x))

        reward_t = torch.tensor([reward], device=self.device, dtype=torch.float32)
        pred_error = (reward_t - self._baseline).detach()
        self._baseline = 0.9 * self._baseline + 0.1 * reward_t

        reward_gate = torch.sigmoid(self.reward_gate(h)).squeeze()
        novelty = torch.sigmoid(self.novelty_gate(h)).squeeze()
        consolidation_trigger = torch.sigmoid(self.consolidation_gate(h)).squeeze()

        return {
            "reward_gate": reward_gate,
            "prediction_error": pred_error.squeeze(),
            "novelty": novelty,
            "consolidation_trigger": consolidation_trigger,
            "h_hidden": h.squeeze(),
        }

    def state_dict_serialisable(self) -> dict:
        sd = {k: v.cpu() for k, v in self.state_dict().items()}
        sd["_baseline"] = self._baseline.cpu()
        sd["_last_pred"] = self._last_pred.cpu()
        return sd

    def load_state_dict_serialisable(self, d: dict) -> None:
        self._baseline = d.pop("_baseline").to(self.device)
        self._last_pred = d.pop("_last_pred").to(self.device)
        super().load_state_dict(d)


class EligibilityTrace:
    """
    Population-level eligibility trace for three-factor plasticity.

    e_t = decay * e_{t-1} + pre ⊗ post
    """

    def __init__(self, n_pre: int, n_post: int, decay: float, device: torch.device):
        self.decay = decay
        self.device = device
        self.trace = torch.zeros(n_pre, n_post, device=device)

    def update(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        self.trace = self.decay * self.trace + torch.outer(pre, post)
        return self.trace.clone()

    def reset_transient(self) -> None:
        """Reset at EpisodeReset boundary (transient eligibility only)."""
        self.trace.zero_()

    def state_dict(self) -> dict:
        return {"trace": self.trace.cpu(), "decay": self.decay}

    def load_state_dict(self, d: dict) -> None:
        self.trace = d["trace"].to(self.device)
        self.decay = d["decay"]
