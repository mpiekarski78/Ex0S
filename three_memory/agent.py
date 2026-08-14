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
        explore_epsilon: float = 0.0,
        use_read: bool = False,
        unique_writes: bool = False,
        use_pick: bool = False,
        write_schema: bool = False,
        force_use: bool = False,
        force_write: bool = False,
    ):
        if retrieve_policy not in ("select", "dump", "policy"):
            raise ValueError(retrieve_policy)
        if collect_mode not in ("off", "commit", "peek", "policy"):
            raise ValueError(collect_mode)
        if collect_mode == "policy" and use_policy is None:
            raise ValueError("collect_mode=policy requires use_policy")
        if retrieve_policy == "policy" and use_policy is None:
            raise ValueError("retrieve_policy=policy requires use_policy")
        if use_read and use_policy is None:
            raise ValueError("use_read requires use_policy")
        if use_pick and use_policy is None:
            raise ValueError("use_pick requires use_policy")
        if write_schema and use_policy is None:
            raise ValueError("write_schema requires use_policy")
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
        self.explore_epsilon = explore_epsilon
        self.use_read = use_read
        self.unique_writes = unique_writes
        self.use_pick = use_pick
        self.write_schema = write_schema
        self.force_use = force_use
        self.force_write = force_write
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

    def _affordances(self, obs: Obs) -> list[int]:
        """Percept-legal acts. Not knowledge: cannot use a key you do not hold."""
        acts = [int(Action.WAIT), int(Action.OPEN)]
        if obs.key_visible and not obs.has_key:
            acts.append(int(Action.PICK_KEY))
        if obs.has_key:
            acts.append(int(Action.USE_KEY))
        return acts

    def _door_code(self, obs: Obs) -> int | None:
        if obs.at_red_door:
            return DOOR_RED
        if obs.at_blue_door:
            return DOOR_BLUE
        if obs.at_green_door:
            return DOOR_GREEN
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

    def _select_hits(self, obs: Obs, pool: list) -> list:
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

    def _choose_retrieve(self, obs: Obs) -> None:
        if self.retrieve_policy != "policy" or self.use_policy is None:
            return
        pool = self._pool()
        hits = self._select_hits(obs, pool)
        n_store = len(self.store) if self.store.enabled else 0
        feat = UsePolicy.retrieve_features(n_store, len(hits))
        dec = self.use_policy.decide_retrieve(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
        self.last_policy = {**self.last_policy, **dec}
        self.policy_traces.append(dec)

    @staticmethod
    def _note_when(rec: FactRecord) -> int:
        w = rec.tags.get("when")
        if isinstance(w, (int, np.integer)):
            return int(w)
        return int(rec.when)

    def _newest(self, hits: list) -> object:
        return max(hits, key=self._note_when)

    def _hits_for(self, obs: Obs) -> list:
        pool = self._pool()
        mode = self.retrieve_policy
        if mode == "policy":
            mode = str(self.last_policy.get("retrieve_mode") or "select")
        if mode == "dump":
            return pool
        return self._select_hits(obs, pool)

    def _apply_record_bias(self, logits: np.ndarray, rec: FactRecord, obs: Obs) -> None:
        act = rec.tags.get("action")
        if self.use_read:
            # Generic copy: the file's integer is the motor index. No USE_KEY/WAIT table.
            if isinstance(act, (int, np.integer)) and 0 <= int(act) < logits.shape[0]:
                logits[int(act)] += 3.0
            return
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

    def _store_bias(self, obs: Obs, *, novelty: float = 0.0, record_use: bool = True) -> np.ndarray:
        """Retrieve facts and bias action logits. Knowledge lives in S, not ρ."""
        self.collect(obs, novelty=novelty)
        if self.retrieve_policy == "policy":
            self._choose_retrieve(obs)
        logits = np.zeros(4, dtype=np.float64)
        apply = True
        chosen = None
        if self.collect_mode == "policy":
            apply = bool(self.last_policy.get("apply", False))
        elif self.use_pick and self.use_policy is not None:
            hits = self._hits_for(obs)
            feat = UsePolicy.pick_features(len(hits))
            dec = self.use_policy.decide_pick(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            self.last_policy = {**self.last_policy, **dec}
            if record_use:
                self.policy_traces.append(dec)
            chosen = hits
            if dec["one"] and hits:
                chosen = [self._newest(hits)]
            apply = True
        elif self.force_use:
            apply = True
        elif self.use_read and self.use_policy is not None:
            hits = self._hits_for(obs)
            feat = UsePolicy.features(bool(hits), False)
            dec = self.use_policy.decide_use(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            self.last_policy = {**self.last_policy, **dec}
            if record_use:
                self.policy_traces.append(dec)
            apply = bool(dec["use"])
        if apply:
            recs = chosen if chosen is not None else self._hits_for(obs)
            for rec in recs:
                self._apply_record_bias(logits, rec, obs)
        return logits

    def _rho_bias(self) -> np.ndarray:
        """Session residue only: recent embed + last successful action (cleared on reset)."""
        logits = 0.35 * self.cortex.baseline_logits(self.rho.predict())
        if self.rho.last_success_action is not None:
            logits[self.rho.last_success_action] += 2.5
        return logits

    def act(self, obs: Obs, *, update_rho: bool = True, explore: bool = False) -> tuple[int, dict[str, Any]]:
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
        # Probe (explore=False) records the use-gate trace; life does not (write traces only).
        logits = logits + self._store_bias(obs, novelty=novelty, record_use=not explore)

        # Hard constraints from current percept (not knowledge): can't use key without holding it.
        if not obs.has_key:
            logits[Action.USE_KEY] -= 5.0
        if obs.has_key or not obs.key_visible:
            logits[Action.PICK_KEY] -= 3.0
        if not at_door:
            logits[Action.OPEN] -= 5.0
            logits[Action.USE_KEY] -= 5.0

        explored = False
        action = int(np.argmax(logits))
        if explore and self.explore_epsilon > 0.0 and float(self.policy_rng.random()) < self.explore_epsilon:
            afford = self._affordances(obs)
            action = int(afford[int(self.policy_rng.integers(0, len(afford)))])
            explored = True
        if update_rho:
            self.rho.update(embed)
        self.t += 1
        meta = {
            "novelty": novelty,
            "logits": logits.tolist(),
            "action": action,
            "explored": explored,
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
        Policy-gated writes (v9) author {door, action} from a door-opening success.
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
        elif obs.last_succeeded and obs.at_green_door:
            event = "green_opened"

        action_name = info.get("action")
        name_to_id = {a.name.lower(): int(a) for a in Action}
        opened = bool(info.get("opened"))
        door = self._door_code(obs) if self.native else None

        if self.use_policy is not None and self.write_from_events:
            # v9: frozen WHAT = {here, the act that opened}. Learned WHEN.
            # v14 B: schema head may omit action=; integer still comes from the event.
            if opened and door is not None and action_name in name_to_id:
                query = {"door": door}
                s_hit = bool(self.store.retrieve(query)) if self.store.enabled else False
                feat = UsePolicy.features(s_hit, True)
                do_write = self.force_write
                if not do_write:
                    dec = self.use_policy.decide_write(
                        feat, epsilon=self.policy_epsilon, rng=self.policy_rng
                    )
                    self.last_policy = dec
                    self.policy_traces.append(dec)
                    do_write = bool(dec["write"])
                complete = True
                if do_write and self.write_schema:
                    sch = self.use_policy.decide_schema(
                        feat, epsilon=self.policy_epsilon, rng=self.policy_rng
                    )
                    self.last_policy = {**self.last_policy, **sch}
                    self.policy_traces.append(sch)
                    complete = bool(sch["complete"])
                if do_write:
                    act = name_to_id[action_name]
                    tags: dict[str, Any] = {"door": door}
                    if complete:
                        tags["action"] = act
                    if self.unique_writes:
                        tags["when"] = int(self.t)
                        fact_id = f"d{door}_t{self.t}_{len(self.store)}"
                    else:
                        fact_id = f"d{door}"
                    record = FactRecord(
                        fact_id=fact_id,
                        what=encode_tags(tags),
                        when=self.t,
                        drive_scores={"novelty": novelty, "integrity": integrity},
                        tags=tags,
                    )
                    wrote = self.store.write(record)
        elif self.write_from_events and self.drives.should_write(novelty, integrity):
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
        if success and action_name:
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
            "policy": dict(self.last_policy) if self.use_policy is not None else {},
        }
