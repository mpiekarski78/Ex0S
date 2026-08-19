"""
EX0S-DEV1 interaction world generator.

Generates deterministic sequences of sensory events and reward signals
for organism interaction.

Boundary invariants
───────────────────
- No internal addresses, cue IDs, logical slot indices, or runner-generated
  keys are ever passed to the organism.
- Symbols are opaque vectors; the organism must learn their meaning from
  consequences (reward, feedback through the motor-observation channel).
- World seeds determine all symbol/role assignments; these are never
  disclosed to the organism.
- Renamed symbol and new-world variants are used for generalization probes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class WorldConfig:
    seed: str = "dev1_world_v1"
    n_symbols: int = 16               # opaque sensory symbol count
    n_roles: int = 4                  # motor role count (Stage A has 4)
    sensory_dim: int = 64             # must match genome.sensory_dim
    reward_on_correct: float = 1.0
    reward_on_incorrect: float = -0.1
    episode_length: int = 32
    n_episodes: int = 64
    renamed_symbol_offset: int = 100  # offset to generate novel symbol variants


def _rng(seed: str) -> random.Random:
    h = hashlib.sha256(seed.encode()).digest()
    return random.Random(int.from_bytes(h[:4], "big"))


def _symbol_vector(symbol_id: int, dim: int, rng: random.Random) -> np.ndarray:
    """Deterministic fixed random vector for a symbol ID. Opaque to organism."""
    r = random.Random(symbol_id * 1337 + 17)
    v = np.array([r.gauss(0, 1) for _ in range(dim)], dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v


@dataclass
class WorldEvent:
    sensory_vector: np.ndarray
    reward: float = 0.0
    is_terminal: bool = False
    _symbol_id: int = -1              # internal; NOT passed to organism
    _correct_channel: int = -1        # internal; NOT passed to organism


class InteractionWorld:
    """
    One world instance with fixed symbol→role mapping.

    Generates sensory events as opaque vectors. The organism must discover
    which motor channel corresponds to which symbol through trial and feedback.
    The runner never passes symbol IDs or role labels to the organism.
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = _rng(cfg.seed)
        self._build_mapping()

    def _build_mapping(self) -> None:
        """Assign symbols to motor roles. Assignment is hidden from organism."""
        symbol_ids = list(range(self.cfg.n_symbols))
        self.rng.shuffle(symbol_ids)
        # Each symbol maps to one role (role = correct motor channel)
        self.symbol_to_role: dict[int, int] = {}
        for i, sid in enumerate(symbol_ids):
            self.symbol_to_role[sid] = i % self.cfg.n_roles

    def generate_episode(self) -> list[WorldEvent]:
        """Generate one episode of interaction events."""
        events: list[WorldEvent] = []
        for _ in range(self.cfg.episode_length):
            sid = self.rng.randint(0, self.cfg.n_symbols - 1)
            vec = _symbol_vector(sid, self.cfg.sensory_dim, self.rng)
            events.append(WorldEvent(
                sensory_vector=vec,
                reward=0.0,
                _symbol_id=sid,
                _correct_channel=self.symbol_to_role[sid],
            ))
        return events

    def reward_for_action(self, event: WorldEvent, motor_channel: int) -> float:
        """Environment-side reward computation. Never exposes correct_channel to organism."""
        if motor_channel == event._correct_channel:
            return self.cfg.reward_on_correct
        return self.cfg.reward_on_incorrect

    def world_hash(self) -> str:
        payload = {
            "seed": self.cfg.seed,
            "n_symbols": self.cfg.n_symbols,
            "n_roles": self.cfg.n_roles,
            "sensory_dim": self.cfg.sensory_dim,
            "symbol_to_role": {str(k): v for k, v in sorted(self.symbol_to_role.items())},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def renamed(cls, base_cfg: WorldConfig, rename_seed: str = "renamed_v1") -> "InteractionWorld":
        """
        World with same structure but fresh symbol vectors.
        Used for generalization probes. Symbol IDs in the new world
        are offset so no original vector appears.
        """
        new_cfg = WorldConfig(
            seed=rename_seed,
            n_symbols=base_cfg.n_symbols,
            n_roles=base_cfg.n_roles,
            sensory_dim=base_cfg.sensory_dim,
            reward_on_correct=base_cfg.reward_on_correct,
            reward_on_incorrect=base_cfg.reward_on_incorrect,
            episode_length=base_cfg.episode_length,
            n_episodes=base_cfg.n_episodes,
            renamed_symbol_offset=base_cfg.renamed_symbol_offset,
        )
        return cls(new_cfg)
