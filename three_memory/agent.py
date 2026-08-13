"""Three-memory agent: frozen cortex + ρ + S + innate write/retrieve rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cortex import CortexConfig, FrozenCortex
from .drives import InnateDrives
from .env import Action, KeyDoorWorld, Obs
from .rho import RhoConfig, WorkingTrace
from .store import FactRecord, WorldStore
from .policy import UsePolicy
from .symbols import (
    ACT_OPEN,
    ACT_PICK_KEY,
    ACT_USE_KEY,
    ACT_WAIT,
    DOOR_BLUE,
    DOOR_GREEN,
    DOOR_RED,
    RED_FACT_ID,
    REQ_KEY,
    encode_tags,
)
from .tag_store import TagLibrary


class ThreeMemoryAgent:
    def __init__(
        self,
        *,
        store_enabled: bool = True,
        cortex_seed: int = 1337,
        obs_dim: int = 16,
        embed_dim: int = 32,
        native: bool = False,
        retrieve_policy: str = "select",
        collect_mode: str = "off",
        store: WorldStore | None = None,
        world: TagLibrary | None = None,
        use_policy: UsePolicy | None = None,
        write_from_events: bool = True,
        policy_epsilon: float = 0.0,
        policy_rng: np.random.Generator | None = None,
    ):
        if retrieve_policy not in ("select", "dump"):
            raise ValueError(retrieve_policy)
        if collect_mode not in ("off", "commit", "peek", "policy"):
            raise ValueError(collect_mode)
        if collect_mode == "policy" and use_policy is None:
            raise ValueError("collect_mode=policy requires use_policy")
        cfg = CortexConfig(obs_dim=obs_dim, embed_dim=embed_dim, n_actions=4, seed=cortex_seed)
        self.cortex = FrozenCortex(cfg)
        self.rho = WorkingTrace(RhoConfig(embed_dim=embed_dim))
        self.store = store if store is not None else WorldStore(enabled=store_enabled)
        self.world = world
        self.drives = InnateDrives()
        self.t = 0
        self.native = native
        self.retrieve_policy = retrieve_policy
        self.collect_mode = collect_mode
        self.use_policy = use_policy
        self.write_from_events = write_from_events
        self.policy_epsilon = policy_epsilon
        self.policy_rng = policy_rng or np.random.default_rng(0)
        self._peek: list[FactRecord] = []
        self.last_policy: dict[str, Any] = {}
        self.policy_traces: list[dict[str, Any]] = []
        self._weight_hash0 = self.cortex.weight_hash()

    def weight_hash(self) -> str:
        return self.cortex.weight_hash()

    def weights_unchanged(self) -> bool:
        return self.weight_hash() == self._weight_hash0

    def reset_rho(self) -> None:
        self.rho.reset()
        self._peek = []

    def reset_store(self) -> None:
        self.store.reset()
        self._peek = []

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
            native=self.native,
            retrieve_policy=self.retrieve_policy,
            collect_mode="off",
        )

    def _obs_query(self, obs: Obs) -> dict[str, Any] | None:
        if obs.at_red_door:
            return {"door": DOOR_RED} if self.native else {"door": "red"}
        if obs.at_blue_door:
            return {"door": DOOR_BLUE} if self.native else {"door": "blue"}
        if obs.at_green_door:
            return {"door": DOOR_GREEN} if self.native else {"door": "green"}
        return None

    def collect(self, obs: Obs, *, novelty: float = 0.0) -> dict[str, Any]:
        query = self._obs_query(obs)
        info: dict[str, Any] = {"taken": 0, "committed": 0, "mode": self.collect_mode}
        if query is None:
            return info
        s_hits = self.store.retrieve(query)
        w_hits = self.world.match(query) if self.world is not None else []
        info["n_store_hits"] = len(s_hits)
        info["n_world_hits"] = len(w_hits)
        chosen = self.collect_mode
        if self.collect_mode == "policy":
            feat = UsePolicy.features(bool(s_hits), bool(w_hits), novelty)
            dec = self.use_policy.decide(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            chosen = dec["collect_mode"]
            info["policy"] = dec
            self.last_policy = dec
            self.policy_traces.append(dec)
        elif self.collect_mode == "off" or self.world is None:
            return info
        elif not self.drives.should_collect(len(s_hits), len(w_hits)):
            return info
        if chosen in ("off", "ignore") or self.world is None or not w_hits:
            return info
        rec = w_hits[0]
        info["taken"] = 1
        if chosen == "commit":
            copied = FactRecord(
                fact_id=rec.fact_id,
                what=rec.what,
                when=self.t,
                drive_scores={"collect": 1.0},
                tags={k: v for k, v in rec.tags.items() if k not in ("source_file", "source")},
            )
            copied.tags["source"] = "W->S"
            if self.store.write(copied):
                info["committed"] = 1
        elif chosen == "peek":
            self._peek = [rec]
        return info

    def _pool(self) -> list[FactRecord]:
        recs = list(self.store.records()) if self.store.enabled else []
        return recs + list(self._peek)

    def _hits_for(self, obs: Obs) -> list[FactRecord]:
        pool = self._pool()
        if self.retrieve_policy == "dump":
            return pool
        query = self._obs_query(obs)
        if not query:
            return []
        out = []
        for r in pool:
            if all(r.tags.get(k) == v for k, v in query.items()):
                out.append(r)
        if not self.native and not out and obs.at_red_door:
            out = [r for r in pool if r.fact_id == KeyDoorWorld.FACT_ID]
        return out

    def _apply_record_bias(self, logits: np.ndarray, rec: FactRecord, obs: Obs) -> None:
        act = rec.tags.get("action")
        if self.native:
            if act == ACT_USE_KEY:
                logits[Action.USE_KEY] += 3.0
                logits[Action.OPEN] -= 2.0
                if not obs.has_key and obs.key_visible:
                    logits[Action.PICK_KEY] += 2.0
            elif act == ACT_OPEN:
                logits[Action.OPEN] += 2.0
            elif act == ACT_PICK_KEY:
                logits[Action.PICK_KEY] += 2.0
            elif act == ACT_WAIT:
                logits[Action.WAIT] += 3.0
                logits[Action.OPEN] -= 2.0
            return
        if rec.tags.get("door") == "red" or rec.fact_id == KeyDoorWorld.FACT_ID:
            logits[Action.USE_KEY] += 3.0
            logits[Action.OPEN] -= 2.0
            if not obs.has_key and obs.key_visible:
                logits[Action.PICK_KEY] += 2.0
        if rec.tags.get("door") == "blue":
            logits[Action.OPEN] += 2.0

    def _store_bias(self, obs: Obs, *, novelty: float = 0.0) -> np.ndarray:
        """Retrieve facts and bias action logits. Knowledge lives in S, not ρ."""
        self.collect(obs, novelty=novelty)
        logits = np.zeros(4, dtype=np.float64)
        apply = True
        if self.collect_mode == "policy":
            apply = bool(self.last_policy.get("apply", False))
        if apply:
            for rec in self._hits_for(obs):
                self._apply_record_bias(logits, rec, obs)
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
        at_door = obs.at_red_door or obs.at_blue_door or obs.at_green_door
        if at_door:
            logits[Action.OPEN] += 1.5
            logits[Action.USE_KEY] -= 0.5
        logits = logits + self._store_bias(obs, novelty=novelty)

        # Hard constraints from current percept (not knowledge): can't use key without holding it.
        if not obs.has_key:
            logits[Action.USE_KEY] -= 5.0
        if obs.has_key or not obs.key_visible:
            logits[Action.PICK_KEY] -= 3.0
        if not at_door:
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
            "store_hits": len(self._hits_for(obs)),
            "rho_l2": float(np.linalg.norm(self.rho.rho)),
            "last_success_action": self.rho.last_success_action,
            "policy": dict(self.last_policy),
        }
        return action, meta

    def observe_outcome(self, obs: Obs, success: bool | None, info: dict[str, Any]) -> dict[str, Any]:
        """Apply innate write rules. Facts go to S, never into cortex weights.

        The world reports events (open_failed / key_worked), not a fact string.
        Write rules build the inspectable record from observation + event.
        """
        vec = obs.vector(self.cortex.config.obs_dim)
        embed = self.cortex.encode(vec)
        novelty = self.drives.novelty(embed, self.rho.predict())
        integrity = self.drives.integrity_cost(success)
        wrote = False
        record = None

        event = None
        if obs.event_open_failed:
            event = "open_failed"
        elif obs.event_key_worked:
            event = "key_worked"
        elif obs.last_succeeded and obs.at_blue_door:
            event = "blue_opened"

        if self.write_from_events and self.drives.should_write(novelty, integrity):
            if obs.at_red_door and event in ("open_failed", "key_worked"):
                if self.native:
                    tags = {"door": DOOR_RED, "requires": REQ_KEY, "action": ACT_USE_KEY}
                    record = FactRecord(
                        fact_id=RED_FACT_ID,
                        what=encode_tags(tags),
                        when=self.t,
                        drive_scores={"novelty": novelty, "integrity": integrity},
                        tags=tags,
                    )
                else:
                    record = FactRecord(
                        fact_id=KeyDoorWorld.FACT_ID,
                        what=KeyDoorWorld.FACT_TEXT,
                        when=self.t,
                        drive_scores={"novelty": novelty, "integrity": integrity},
                        tags={"door": "red", "requires": "key", "action": "use_key"},
                    )
                wrote = self.store.write(record)
            elif obs.at_blue_door and event == "blue_opened":
                if self.native:
                    tags = {"door": DOOR_BLUE, "action": ACT_OPEN}
                    record = FactRecord(
                        fact_id="d1",
                        what=encode_tags(tags),
                        when=self.t,
                        drive_scores={"novelty": novelty, "integrity": integrity},
                        tags=tags,
                    )
                else:
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
            "event": event,
            "record": record.to_dict() if record and wrote else None,
            "weights_unchanged": self.weights_unchanged(),
        }
