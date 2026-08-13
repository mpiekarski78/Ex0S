"""Three-memory agent: frozen cortex + ρ + S + innate write/retrieve rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cortex import CortexConfig, FrozenCortex
from .drives import InnateDrives
from .env import Action, KeyDoorWorld, Obs
from .rho import RhoConfig, WorkingTrace
from .store import FactRecord, WorldStore


class ThreeMemoryAgent:
    def __init__(
        self,
        *,
        store_enabled: bool = True,
        cortex_seed: int = 1337,
        obs_dim: int = 16,
        embed_dim: int = 32,
    ):
        cfg = CortexConfig(obs_dim=obs_dim, embed_dim=embed_dim, n_actions=4, seed=cortex_seed)
        self.cortex = FrozenCortex(cfg)
        self.rho = WorkingTrace(RhoConfig(embed_dim=embed_dim))
        self.store = WorldStore(enabled=store_enabled)
        self.drives = InnateDrives()
        self.t = 0
        self._weight_hash0 = self.cortex.weight_hash()

    def weight_hash(self) -> str:
        return self.cortex.weight_hash()

    def weights_unchanged(self) -> bool:
        return self.weight_hash() == self._weight_hash0

    def reset_rho(self) -> None:
        self.rho.reset()

    def reset_store(self) -> None:
        self.store.reset()

    def reset_life(self) -> None:
        self.reset_rho()
        self.reset_store()
        self.t = 0

    def clone_empty(self, store_enabled: bool | None = None) -> "ThreeMemoryAgent":
        """Same frozen cortex seed, empty ρ and S."""
        en = self.store.enabled if store_enabled is None else store_enabled
        return ThreeMemoryAgent(
            store_enabled=en,
            cortex_seed=self.cortex.config.seed,
            obs_dim=self.cortex.config.obs_dim,
            embed_dim=self.cortex.config.embed_dim,
        )

    def _store_bias(self, obs: Obs) -> np.ndarray:
        """Retrieve facts and bias action logits. Knowledge lives in S, not ρ."""
        logits = np.zeros(4, dtype=np.float64)
        hits = self.store.retrieve({"door": "red"}) if obs.at_red_door else []
        if not hits and obs.at_red_door:
            hits = [r for r in self.store.records() if r.fact_id == KeyDoorWorld.FACT_ID]
        for rec in hits:
            # Fact: red door opens only with key → prefer USE_KEY, avoid bare OPEN.
            logits[Action.USE_KEY] += 3.0
            logits[Action.OPEN] -= 2.0
            if not obs.has_key and obs.key_visible:
                logits[Action.PICK_KEY] += 2.0
        if obs.at_blue_door:
            blue = self.store.retrieve({"door": "blue"})
            for _ in blue:
                logits[Action.OPEN] += 2.0
        return logits

    def _rho_bias(self) -> np.ndarray:
        """Session residue only: recent embed + last successful action (cleared on reset)."""
        logits = 0.35 * self.cortex.baseline_logits(self.rho.predict())
        if self.rho.last_success_action is not None:
            logits[self.rho.last_success_action] += 2.5
        return logits

    def act(self, obs: Obs, *, update_rho: bool = True) -> tuple[int, dict[str, Any]]:
        vec = obs.vector(self.cortex.config.obs_dim)
        predicted = self.rho.predict()
        embed = self.cortex.encode(vec)
        novelty = self.drives.novelty(embed, predicted)

        logits = self.cortex.baseline_logits(embed)
        logits = logits + self._rho_bias()
        # Species prior: at a door, try OPEN (does not know about keys).
        # Life knowledge must come from S (or fragile ρ after recent success).
        if obs.at_red_door or obs.at_blue_door:
            logits[Action.OPEN] += 1.5
            logits[Action.USE_KEY] -= 0.5
        logits = logits + self._store_bias(obs)

        # Hard constraints from current percept (not knowledge): can't use key without holding it.
        if not obs.has_key:
            logits[Action.USE_KEY] -= 5.0
        if obs.has_key or not obs.key_visible:
            logits[Action.PICK_KEY] -= 3.0
        if not (obs.at_red_door or obs.at_blue_door):
            logits[Action.OPEN] -= 5.0
            logits[Action.USE_KEY] -= 5.0

        action = int(np.argmax(logits))
        if update_rho:
            self.rho.update(embed)
        self.t += 1
        meta = {
            "novelty": novelty,
            "logits": logits.tolist(),
            "action": action,
            "store_hits": len(self.store.retrieve({"door": "red"})) if obs.at_red_door else 0,
            "rho_l2": float(np.linalg.norm(self.rho.rho)),
            "last_success_action": self.rho.last_success_action,
        }
        return action, meta

    def observe_outcome(self, obs: Obs, success: bool | None, info: dict[str, Any]) -> dict[str, Any]:
        """Apply innate write rules. Facts go to S, never into cortex weights."""
        vec = obs.vector(self.cortex.config.obs_dim)
        embed = self.cortex.encode(vec)
        novelty = self.drives.novelty(embed, self.rho.predict())
        integrity = self.drives.integrity_cost(success)
        wrote = False
        record = None

        lesson = info.get("lesson")
        if lesson and self.drives.should_write(novelty, integrity):
            if "red door" in lesson and "key" in lesson:
                record = FactRecord(
                    fact_id=KeyDoorWorld.FACT_ID,
                    what=KeyDoorWorld.FACT_TEXT,
                    when=self.t,
                    drive_scores={"novelty": novelty, "integrity": integrity},
                    tags={"door": "red", "requires": "key", "action": "use_key"},
                )
                wrote = self.store.write(record)
            elif obs.at_blue_door and success:
                record = FactRecord(
                    fact_id="blue_door_opens",
                    what="blue door opens with open",
                    when=self.t,
                    drive_scores={"novelty": novelty, "integrity": integrity},
                    tags={"door": "blue", "action": "open"},
                )
                wrote = self.store.write(record)

        # Session residue: remember which action just worked (cleared on ρ reset).
        action_name = info.get("action")
        if success and action_name:
            name_to_id = {a.name.lower(): int(a) for a in Action}
            if action_name in name_to_id:
                self.rho.note_success(name_to_id[action_name])

        # Always update ρ with post-outcome observation (session residue).
        self.rho.update(embed)
        return {
            "novelty": novelty,
            "integrity": integrity,
            "wrote": wrote,
            "record": record.to_dict() if record and wrote else None,
            "weights_unchanged": self.weights_unchanged(),
        }
