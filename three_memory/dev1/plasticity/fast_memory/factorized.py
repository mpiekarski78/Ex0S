"""
Factorized structure/content fast memory (Axis 2, candidate B).

Inspired by the Tolman-Eichenbaum Machine (TEM): separates structural
(relational/positional) encoding from content (sensory/item) encoding.
This factorization allows abstract relational structure to generalize
across different content.

Reference: Whittington et al., "The Tolman-Eichenbaum Machine:
Unifying Space and Relational Memory through Generalisation in the
Hippocampal Formation", Neuron 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedMemory(nn.Module):
    """
    Factorized hippocampal memory.

    Writes two separate codes:
    - structural code: abstract relational context
    - content code:   current sensory item / fact

    Retrieval uses structural key to recover content.
    This allows composition of known structural patterns
    with novel content at test time.
    """

    def __init__(self, relational_dim: int, content_dim: int, capacity: int, device: torch.device):
        super().__init__()
        self.capacity = capacity
        self.device = device

        # Structure encoder (learned/evolved; frozen during evaluated life)
        self.struct_enc = nn.Linear(relational_dim, relational_dim // 2, bias=False)
        # Content encoder (learned/evolved; frozen during evaluated life)
        self.content_enc = nn.Linear(content_dim, content_dim // 2, bias=False)

        self._store: list[tuple[torch.Tensor, torch.Tensor]] = []  # (struct_key, content)
        self.to(device)

    def write(self, relational: torch.Tensor, content: torch.Tensor) -> None:
        """Factorized write: separate struct and content encoding."""
        struct_key = F.normalize(F.relu(self.struct_enc(relational)), dim=0)
        content_val = F.relu(self.content_enc(content))
        if len(self._store) >= self.capacity:
            self._store.pop(0)
        self._store.append((struct_key.detach(), content_val.detach()))

    def read(self, relational: torch.Tensor) -> torch.Tensor:
        """Retrieve content by structural similarity."""
        if not self._store:
            return torch.zeros(self.content_enc.out_features, device=self.device)
        query = F.normalize(F.relu(self.struct_enc(relational)), dim=0)
        best_sim = -1.0
        best_content = self._store[0][1]
        for sk, cv in self._store:
            sim = float(F.cosine_similarity(sk.unsqueeze(0), query.unsqueeze(0)))
            if sim > best_sim:
                best_sim = sim
                best_content = cv
        return best_content

    def wipe(self) -> None:
        self._store.clear()

    def capacity_used(self) -> int:
        return len(self._store)

    def name(self) -> str:
        return "factorized"
