"""
EX0S-DEV1 slow modular cortical populations.

Three separate population modules:
- SensoryCortex:    encodes raw observations into a contextual representation.
- RelationalCortex: active variable binding, reusable relational structure.
- ActionCortex:     opaque motor-channel competition.

During an evaluated life, within-life updates use ONLY the frozen local
plasticity law (pre/post activity, eligibility, neuromodulation, prediction
errors, and organism-sampled replay). No gradient-based update is applied
to W, H, or ρ.

Cortical weights W are updated by the plasticity family attached from
three_memory/dev1/plasticity/cortical_plasticity/. This module contains
only the forward computation and state management.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from three_memory.dev1.genome import DevGenome, PopulationSpec


class CorticalPopulation(nn.Module):
    """Generic recurrent cortical population."""

    def __init__(self, spec: PopulationSpec, in_dim: int, device: torch.device):
        super().__init__()
        self.n = spec.n_units
        self.device = device
        self.W_in = nn.Linear(in_dim, spec.n_units, bias=False)
        if spec.recurrent:
            self.W_rec = nn.Linear(spec.n_units, spec.n_units, bias=False)
        else:
            self.W_rec = None
        self.to(device)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h = self.W_in(x)
        if self.W_rec is not None:
            h = h + self.W_rec(state)
        return F.relu(h)


class SensoryCortex(nn.Module):
    """
    Sensory/context cortical population.

    Input:  raw sensory vector from OrganismObservation.sensory_vector
    Output: contextual population state
    """

    def __init__(self, genome: DevGenome, device: torch.device):
        super().__init__()
        self.pop = CorticalPopulation(genome.sensory_ctx, genome.sensory_dim, device)
        self.n = genome.sensory_ctx.n_units
        self.device = device
        self.to(device)

    def forward(self, sensory_input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.pop(sensory_input, state)


class RelationalCortex(nn.Module):
    """
    Relational/working cortical population.

    Receives: sensory population state + retrieved H content (when available)
    Produces:  relational context used by action cortex and H key
    """

    def __init__(self, genome: DevGenome, device: torch.device):
        super().__init__()
        in_dim = genome.sensory_ctx.n_units + genome.hippocampus.ca1_n_units
        self.pop = CorticalPopulation(genome.relational_ctx, in_dim, device)
        self.n = genome.relational_ctx.n_units
        self.device = device
        self.to(device)

    def forward(
        self,
        sensory_state: torch.Tensor,
        retrieved: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([sensory_state, retrieved])
        return self.pop(x, state)


class ActionCortex(nn.Module):
    """
    Action cortical population.

    Produces opaque motor-channel competition scores.
    The organism — not the environment or runner — owns the meaning of channels.
    ASK, HOLD, and answer actions are motor channels; no op/operand structure.
    """

    def __init__(self, genome: DevGenome, device: torch.device):
        super().__init__()
        in_dim = genome.relational_ctx.n_units
        self.pop = CorticalPopulation(genome.action_ctx, in_dim, device)
        self.W_motor = nn.Linear(genome.action_ctx.n_units, genome.n_motor_channels, bias=False)
        self.n = genome.action_ctx.n_units
        self.device = device
        self.to(device)

    def forward(self, relational_state: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_pop_state, motor_logits)."""
        action_state = self.pop(relational_state, state)
        motor_logits = self.W_motor(action_state)
        return action_state, motor_logits

    def competition(
        self,
        motor_logits: torch.Tensor,
        policy_mode: str = "hard",
        generator: torch.Generator | None = None,
    ) -> tuple[int, torch.Tensor, float]:
        """
        Motor competition.
        Returns (channel_idx, scores, confidence).
        Confidence is the top-1 minus top-2 softmax margin.
        `policy_mode="hard"` uses canonical argmax evaluation.
        `policy_mode="stochastic"` samples from the softmax policy.
        """
        scores = F.softmax(motor_logits, dim=-1)
        if policy_mode == "hard":
            channel = int(scores.argmax().item())
        elif policy_mode == "stochastic":
            channel = int(torch.multinomial(scores, 1, generator=generator).item())
        else:
            raise ValueError(f"unknown policy_mode: {policy_mode}")
        top2 = scores.topk(min(2, scores.numel())).values
        confidence = float((top2[0] - top2[-1]).item())
        return channel, scores, confidence
