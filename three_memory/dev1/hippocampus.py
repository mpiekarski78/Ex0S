"""
EX0S-DEV1 fast memory H (EC→DG/CA3→CA1 computational abstraction).

Design principles
─────────────────
- Cortical state enters a learned/evolved encoder (EC analogue).
- A sparse competitive population forms a conjunctive episode code (DG/CA3).
- A fast Hebbian recurrent matrix binds episode structure and content.
- A learned/evolved readout reinstates content into relational/working cortex (CA1).
- The action cortex — NOT H — chooses the motor response.
- Revision appends a new evidence record; previous episodes are never silently
  erased or rewritten. Reconsolidation may change active H contents, but every
  change is attributable to experienced events and preserved in S_log.
- Capacity is bounded; eviction and decay are reported explicitly in telemetry.
- H write/read is disabled during Stage A via the `h_disabled` flag.

Organism-owned addressing
─────────────────────────
H is written using keys derived from the organism's own cortical state.
No programmer key, logical slot, or runner-generated address enters H.
The retrieval path uses the same EC encoder — not an external query.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from three_memory.dev1.genome import DevGenome, HippocampalSpec


class ECEncoder(nn.Module):
    """Entorhinal cortex analogue: compresses relational context into EC code."""

    def __init__(self, in_dim: int, ec_dim: int, device: torch.device):
        super().__init__()
        self.proj = nn.Linear(in_dim, ec_dim, bias=False)
        self.to(device)

    def forward(self, relational: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(relational))


class DGEncoder(nn.Module):
    """
    Dentate gyrus / CA3 analogue: produces sparse conjunctive code.
    Winner-take-all sparsification makes each episode maximally orthogonal.
    """

    def __init__(self, ec_dim: int, dg_n: int, sparsity: float, device: torch.device):
        super().__init__()
        self.proj = nn.Linear(ec_dim, dg_n, bias=False)
        self.k = max(1, int(dg_n * sparsity))
        self.to(device)

    def forward(self, ec: torch.Tensor) -> torch.Tensor:
        h = self.proj(ec)
        # k-winners-take-all sparsification
        topk = h.topk(self.k)
        sparse = torch.zeros_like(h)
        sparse[topk.indices] = topk.values
        return F.relu(sparse)


class CA1Readout(nn.Module):
    """CA1 analogue: reinstates content into relational/working cortex."""

    def __init__(self, ca3_n: int, ca1_n: int, out_dim: int, device: torch.device):
        super().__init__()
        self.proj = nn.Linear(ca3_n, ca1_n, bias=False)
        self.readout = nn.Linear(ca1_n, out_dim, bias=False)
        self.to(device)

    def forward(self, ca3: torch.Tensor) -> torch.Tensor:
        return F.relu(self.readout(F.relu(self.proj(ca3))))


class FastHippocampus(nn.Module):
    """
    EC→DG/CA3→CA1 fast memory module.

    Fast Hebbian writes allow one-shot retention; the learned encoder and
    readout are inherited parameters shaped by G and the research optimizer.
    """

    def __init__(self, genome: DevGenome, device: torch.device):
        super().__init__()
        spec: HippocampalSpec = genome.hippocampus
        self.spec = spec
        self.device = device
        self.h_disabled: bool = False  # set True during Stage A

        # Encoder pathway
        self.ec = ECEncoder(genome.relational_ctx.n_units, spec.ec_dim, device)
        self.dg = DGEncoder(spec.ec_dim, spec.dg_n_units, spec.dg_sparsity, device)

        # Fast Hebbian recurrent matrix: shape (dg_n, dg_n)
        self.W_hebb = torch.zeros(spec.dg_n_units, spec.dg_n_units, device=device)
        self.hebbian_lr: float = spec.hebbian_lr

        # CA1 readout into relational cortex
        self.ca1 = CA1Readout(
            spec.dg_n_units,
            spec.ca1_n_units,
            genome.relational_ctx.n_units,
            device,
        )

        # Episodic store: list of (key: Tensor, value: Tensor) pairs
        self._store: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._evictions: int = 0
        self._writes_this_episode: int = 0

        self.to(device)

    # ── Write path ─────────────────────────────────────────────────────────────

    def write(self, relational: torch.Tensor, content: torch.Tensor) -> bool:
        """
        Organism-owned write. Key derived from relational cortex state only.
        Returns True if write occurred, False if H is disabled.
        """
        if self.h_disabled:
            return False

        ec = self.ec(relational)
        key = self.dg(ec)

        # Fast Hebbian update to recurrent matrix
        key_norm = F.normalize(key, dim=0)
        self.W_hebb = self.W_hebb + self.hebbian_lr * torch.outer(key_norm, key_norm)

        # Store episode record
        if len(self._store) >= self.spec.capacity:
            self._evict()

        self._store.append((key.detach().clone(), content.detach().clone()))
        self._writes_this_episode += 1
        return True

    def _evict(self) -> None:
        """Remove one episode according to eviction policy."""
        if self.spec.eviction_policy == "lru":
            self._store.pop(0)
        else:
            import random
            idx = random.randrange(len(self._store))
            self._store.pop(idx)
        self._evictions += 1

    # ── Read path ──────────────────────────────────────────────────────────────

    def read(self, relational: torch.Tensor) -> torch.Tensor:
        """
        Associative completion.
        Returns a zero vector if H is disabled or store is empty.
        Action cortex — NOT H — makes the motor decision.
        """
        out_dim = self.ca1.readout.out_features
        null = torch.zeros(out_dim, device=self.device)
        if self.h_disabled or not self._store:
            return null

        ec = self.ec(relational)
        query = self.dg(ec)
        query_norm = F.normalize(query, dim=0)

        # Pattern completion via W_hebb
        completed = torch.mv(self.W_hebb, query_norm)
        completed = F.relu(completed)

        # Similarity-weighted content retrieval
        best_sim = -1.0
        best_content = self._store[0][1]
        for key, content in self._store:
            sim = float(F.cosine_similarity(key.unsqueeze(0), query.unsqueeze(0)))
            if sim > best_sim:
                best_sim = sim
                best_content = content

        reinstated = self.ca1(completed)
        return reinstated

    # ── Revision (append, never erase) ─────────────────────────────────────────

    def revise(self, relational: torch.Tensor, new_content: torch.Tensor) -> None:
        """
        Append a new evidence record for an existing conceptual key.
        Previous records are never silently erased or overwritten.
        The S_log entry is created by organism.py, not here.
        """
        self.write(relational, new_content)

    # ── Wipe (Stage B gate) ────────────────────────────────────────────────────

    def wipe(self) -> None:
        """Delete all stored episodes and reset Hebbian matrix."""
        self._store.clear()
        self.W_hebb.zero_()
        self._evictions = 0
        self._writes_this_episode = 0

    # ── Episode reset (does NOT wipe H) ────────────────────────────────────────

    def reset_episode_counter(self) -> None:
        """Called at EpisodeReset; H itself is NOT cleared."""
        self._writes_this_episode = 0

    # ── Telemetry ──────────────────────────────────────────────────────────────

    def capacity_telemetry(self) -> dict:
        return {
            "capacity_used": len(self._store),
            "capacity_max": self.spec.capacity,
            "evictions_total": self._evictions,
            "writes_this_episode": self._writes_this_episode,
        }

    # ── Checkpointing ──────────────────────────────────────────────────────────

    def hippocampus_state_dict(self) -> dict:
        return {
            "W_hebb": self.W_hebb.cpu(),
            "store_keys": [k.cpu() for k, _ in self._store],
            "store_contents": [c.cpu() for _, c in self._store],
            "evictions": self._evictions,
            "writes_this_episode": self._writes_this_episode,
            "h_disabled": self.h_disabled,
        }

    def load_hippocampus_state_dict(self, d: dict) -> None:
        self.W_hebb = d["W_hebb"].to(self.device)
        self._store = list(zip(
            [k.to(self.device) for k in d["store_keys"]],
            [c.to(self.device) for c in d["store_contents"]],
        ))
        self._evictions = d["evictions"]
        self._writes_this_episode = d["writes_this_episode"]
        self.h_disabled = d.get("h_disabled", False)

    def hippocampus_plasticity_state_dict(self) -> dict:
        return {
            "ec": self.ec.state_dict(),
            "dg": self.dg.state_dict(),
            "ca1": self.ca1.state_dict(),
            "hebbian_lr": self.hebbian_lr,
        }

    def load_hippocampus_plasticity_state_dict(self, d: dict) -> None:
        self.ec.load_state_dict(d["ec"])
        self.dg.load_state_dict(d["dg"])
        self.ca1.load_state_dict(d["ca1"])
        self.hebbian_lr = d["hebbian_lr"]
