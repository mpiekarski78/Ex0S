"""Three-memory agent: frozen cortex + ρ + S + innate write/retrieve rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cortex import CortexConfig, FrozenCortex
from .drives import InnateDrives
from .env import Action, KeyDoorWorld, Obs
from .dial_env import CH_A, CH_B, CH_C, DialAction, STATION_NAMES
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
from .tag_store import DocLibrary, ProseLibrary, TagLibrary, prose_token_stream, prose_tokens

# Bookkeeping tags, not a place-name menu. Query candidates are whatever else the files have.
# `did` is the act the body just did (not a copy token). `bind` is the one page-word aliased to it.
_QNAME_SKIP = frozenset({"source", "source_file", "when", "ok", "what", "did", "bind"})


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
        world: TagLibrary | DocLibrary | ProseLibrary | None = None,
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
        use_rank: bool = False,
        mark_ok: bool = False,
        value_key: str = "action",
        place_key: str = "door",
        use_key_head: bool = False,
        use_match_head: bool = False,
        use_wkey_head: bool = False,
        use_wplace_head: bool = False,
        use_wsel_head: bool = False,
        wsel_dump: bool = False,
        use_wcomp_head: bool = False,
        use_qname_head: bool = False,
        use_vname_head: bool = False,
        use_search_head: bool = False,
        record_search_on_explore: bool = False,
        use_prose_ints: bool = False,
        use_prose_tokens: bool = False,
        use_revise_head: bool = False,
        use_event_annotate: bool = False,
        use_here_match: bool = False,
        use_commit_rare_only: bool = False,
        use_commit_here_only: bool = False,
        use_alias_bind: bool = False,
        use_did_stamp: bool = False,
        use_one_bind: bool = False,
        use_stamp_new_here: bool = False,
        use_block_here: bool = False,
        domain: str = "door",
    ):
        if retrieve_policy not in ("select", "dump", "policy"):
            raise ValueError(retrieve_policy)
        if collect_mode not in ("off", "commit", "peek", "policy"):
            raise ValueError(collect_mode)
        if domain not in ("door", "dial"):
            raise ValueError(domain)
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
        if use_rank and use_policy is None:
            raise ValueError("use_rank requires use_policy")
        if use_key_head and use_policy is None:
            raise ValueError("use_key_head requires use_policy")
        if use_match_head and use_policy is None:
            raise ValueError("use_match_head requires use_policy")
        if use_wkey_head and use_policy is None:
            raise ValueError("use_wkey_head requires use_policy")
        if use_wplace_head and use_policy is None:
            raise ValueError("use_wplace_head requires use_policy")
        if use_wsel_head and use_policy is None:
            raise ValueError("use_wsel_head requires use_policy")
        if use_wcomp_head and use_policy is None:
            raise ValueError("use_wcomp_head requires use_policy")
        if use_qname_head and use_policy is None:
            raise ValueError("use_qname_head requires use_policy")
        if use_qname_head and use_match_head:
            raise ValueError("use_qname_head cannot run with the {door, here} match menu")
        if use_vname_head and use_policy is None:
            raise ValueError("use_vname_head requires use_policy")
        if use_vname_head and use_key_head:
            raise ValueError("use_vname_head cannot run with the {action, do} copy menu")
        if use_search_head and use_policy is None:
            raise ValueError("use_search_head requires use_policy")
        if use_search_head and (use_match_head or use_qname_head):
            raise ValueError("use_search_head cannot run with exact query match")
        if use_prose_ints and use_policy is None:
            raise ValueError("use_prose_ints requires use_policy")
        if use_prose_ints and not (use_search_head and use_vname_head and use_read):
            raise ValueError("use_prose_ints requires search + vname + use_read")
        if use_prose_tokens and use_policy is None:
            raise ValueError("use_prose_tokens requires use_policy")
        if use_prose_tokens and not (use_search_head and use_vname_head and use_read):
            raise ValueError("use_prose_tokens requires search + vname + use_read")
        if use_prose_ints and use_prose_tokens:
            raise ValueError("use_prose_ints and use_prose_tokens cannot run together")
        if use_revise_head and use_policy is None:
            raise ValueError("use_revise_head requires use_policy")
        if use_event_annotate and use_policy is None:
            raise ValueError("use_event_annotate requires use_policy")
        if use_event_annotate and write_from_events:
            raise ValueError("use_event_annotate cannot run with write_from_events")
        if use_event_annotate and not (use_search_head and use_vname_head and use_read and use_prose_tokens):
            raise ValueError("use_event_annotate requires search + vname + use_read + use_prose_tokens")
        if use_here_match and not use_event_annotate:
            raise ValueError("use_here_match requires use_event_annotate")
        if use_alias_bind and not use_event_annotate:
            raise ValueError("use_alias_bind requires use_event_annotate")
        if use_did_stamp and not use_event_annotate:
            raise ValueError("use_did_stamp requires use_event_annotate")
        if use_alias_bind and not use_did_stamp:
            raise ValueError("use_alias_bind requires use_did_stamp")
        if use_one_bind and not use_alias_bind:
            raise ValueError("use_one_bind requires use_alias_bind")
        if use_stamp_new_here and not use_here_match:
            raise ValueError("use_stamp_new_here requires use_here_match")
        if use_block_here and not use_here_match:
            raise ValueError("use_block_here requires use_here_match")
        if use_block_here and not use_event_annotate:
            raise ValueError("use_block_here requires use_event_annotate")
        if value_key not in ("action", "do"):
            raise ValueError(value_key)
        if not isinstance(place_key, str) or not place_key:
            raise ValueError(place_key)
        self.domain = domain
        self.n_actions = 5 if domain == "dial" else 4
        cfg = CortexConfig(
            obs_dim=obs_dim, embed_dim=embed_dim, n_actions=self.n_actions, seed=cortex_seed
        )
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
        self.use_rank = use_rank
        self.mark_ok = mark_ok
        self.value_key = value_key
        self.place_key = place_key
        self.use_key_head = use_key_head
        self.use_match_head = use_match_head
        self.use_wkey_head = use_wkey_head
        self.use_wplace_head = use_wplace_head
        self.use_wsel_head = use_wsel_head
        self.wsel_dump = wsel_dump
        self.use_wcomp_head = use_wcomp_head
        self.use_qname_head = use_qname_head
        self.use_vname_head = use_vname_head
        self.use_search_head = use_search_head
        # Free life: motor explores, but search commits must still leave learnable traces.
        self.record_search_on_explore = record_search_on_explore
        # Prose pages: digits → anonymous n* tags; vname picks among values, not filed action=.
        self.use_prose_ints = use_prose_ints
        # Prose pages with no answer ints: words → w* tags; vname picks a token, not n*.
        self.use_prose_tokens = use_prose_tokens
        self.use_revise_head = use_revise_head
        self.use_event_annotate = use_event_annotate
        self.use_here_match = use_here_match
        self.use_commit_rare_only = use_commit_rare_only
        self.use_commit_here_only = use_commit_here_only
        self.use_alias_bind = use_alias_bind
        self.use_did_stamp = use_did_stamp
        self.use_one_bind = use_one_bind
        self.use_stamp_new_here = use_stamp_new_here
        self.use_block_here = use_block_here
        self._peek: list[FactRecord] = []
        self._search_chosen: list = []
        self._in_hand_id: str | None = None
        self._w_skip: set[str] = set()
        self.n_revised = 0
        self.n_annotated = 0
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
        self._w_skip = set()
        self._in_hand_id = None

    def reset_store(self) -> None:
        self.store.reset()
        self._peek = []

    def reset_life(self) -> None:
        self.reset_rho()
        self.reset_store()
        self.t = 0

    def clone_empty(self, store_enabled: bool | None = None) -> "ThreeMemoryAgent":
        """Same frozen cortex seed and boxed heads, empty ρ and S."""
        en = self.store.enabled if store_enabled is None else store_enabled
        return ThreeMemoryAgent(
            store_enabled=en,
            cortex_seed=self.cortex.config.seed,
            obs_dim=self.cortex.config.obs_dim,
            embed_dim=self.cortex.config.embed_dim,
            native=self.native,
            retrieve_policy=self.retrieve_policy,
            collect_mode=self.collect_mode,
            use_policy=self.use_policy,
            write_from_events=self.write_from_events,
            policy_epsilon=self.policy_epsilon,
            explore_epsilon=self.explore_epsilon,
            use_read=self.use_read,
            unique_writes=self.unique_writes,
            use_pick=self.use_pick,
            write_schema=self.write_schema,
            force_use=self.force_use,
            force_write=self.force_write,
            use_rank=self.use_rank,
            mark_ok=self.mark_ok,
            value_key=self.value_key,
            place_key=self.place_key,
            use_key_head=self.use_key_head,
            use_match_head=self.use_match_head,
            use_wkey_head=self.use_wkey_head,
            use_wplace_head=self.use_wplace_head,
            use_wsel_head=self.use_wsel_head,
            wsel_dump=self.wsel_dump,
            use_wcomp_head=self.use_wcomp_head,
            use_qname_head=self.use_qname_head,
            use_vname_head=self.use_vname_head,
            use_search_head=self.use_search_head,
            record_search_on_explore=self.record_search_on_explore,
            use_prose_ints=self.use_prose_ints,
            use_prose_tokens=self.use_prose_tokens,
            use_revise_head=self.use_revise_head,
            use_event_annotate=self.use_event_annotate,
            use_here_match=self.use_here_match,
            use_commit_rare_only=self.use_commit_rare_only,
            use_commit_here_only=self.use_commit_here_only,
            use_alias_bind=self.use_alias_bind,
            use_did_stamp=self.use_did_stamp,
            use_one_bind=self.use_one_bind,
            use_stamp_new_here=self.use_stamp_new_here,
            use_block_here=self.use_block_here,
            domain=self.domain,
        )

    def _obs_query(self, obs: Obs) -> dict[str, Any] | None:
        if not self.native:
            if obs.at_red_door:
                return {"door": "red"}
            if obs.at_blue_door:
                return {"door": "blue"}
            if obs.at_green_door:
                return {"door": "green"}
            return None
        code = self._door_code(obs)
        if code is None:
            return None
        if self.use_qname_head:
            key = self.last_policy.get("qname")
            if not key:
                return None
            return {str(key): code}
        key = self.place_key
        if self.use_match_head:
            key = "here" if bool(self.last_policy.get("match_alt")) else "door"
        return {key: code}

    def _affordances(self, obs) -> list[int]:
        """Percept-legal acts. Not knowledge."""
        if self.domain == "dial":
            return [int(a) for a in DialAction]
        acts = [int(Action.WAIT), int(Action.OPEN)]
        if obs.key_visible and not obs.has_key:
            acts.append(int(Action.PICK_KEY))
        if obs.has_key:
            acts.append(int(Action.USE_KEY))
        return acts

    def _door_code(self, obs) -> int | None:
        """Place code for the current station (door code or dial channel)."""
        if self.domain == "dial":
            if getattr(obs, "at_a", False):
                return CH_A
            if getattr(obs, "at_b", False):
                return CH_B
            if getattr(obs, "at_c", False):
                return CH_C
            return None
        if obs.at_red_door:
            return DOOR_RED
        if obs.at_blue_door:
            return DOOR_BLUE
        if obs.at_green_door:
            return DOOR_GREEN
        return None

    def _qname_pool(self) -> list:
        if self.world is not None:
            return list(self.world.records())
        if self.store.enabled:
            return list(self.store.records())
        return []

    def _query_keys(self, pool: list) -> list[str]:
        keys: set[str] = set()
        for rec in pool:
            keys.update(k for k in rec.tags if k not in _QNAME_SKIP)
        return sorted(keys)

    def _choose_qname(self, obs: Obs, *, record: bool) -> None:
        if not self.use_qname_head or self.use_policy is None:
            return
        code = self._door_code(obs)
        pool = self._qname_pool()
        keys = self._query_keys(pool)
        if not keys or code is None:
            self.last_policy = {k: v for k, v in self.last_policy.items() if k != "qname"}
            return
        items = []
        for k in keys:
            has_hit = any(r.tags.get(k) == code for r in pool)
            key_common = sum(1 for r in pool if k in r.tags) >= 3
            items.append((has_hit, key_common))
        dec = self.use_policy.decide_qname(
            items, epsilon=self.policy_epsilon, rng=self.policy_rng
        )
        dec["qname"] = keys[int(dec["idx"])]
        dec["qnames"] = keys
        self.last_policy = {**self.last_policy, **dec}
        if record:
            self.policy_traces.append(dec)

    def collect(self, obs: Obs, *, novelty: float = 0.0, record: bool = True) -> dict[str, Any]:
        info: dict[str, Any] = {"taken": 0, "committed": 0, "mode": self.collect_mode}
        if self.use_search_head:
            return self._collect_search(obs, record=record)
        query = self._obs_query(obs)
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
            self.last_policy = {**self.last_policy, **dec}
            self.policy_traces.append(dec)
        elif self.collect_mode == "off" or self.world is None:
            return info
        elif not self.drives.should_collect(len(s_hits), len(w_hits)):
            return info
        if chosen in ("off", "ignore") or self.world is None or not w_hits:
            return info
        picks = self._wsel_picks(w_hits)
        info["taken"] = 1
        if chosen == "commit":
            n = 0
            for rec in picks:
                if self._commit_w_record(rec):
                    n += 1
            info["committed"] = n
        elif chosen == "peek":
            self._peek = list(picks)
        return info

    def _tag_vals(self, rec) -> list:
        return [v for k, v in rec.tags.items() if k not in _QNAME_SKIP]

    def _search_picks(self, pool: list, obs: Obs, *, record: bool) -> list:
        if not pool or self.use_policy is None:
            return []
        pool = [r for r in pool if getattr(r, "fact_id", None) not in self._w_skip]
        if not pool:
            return []
        if self.use_here_match:
            here = [r for r in pool if self._rec_names_here(r, obs)]
            if here:
                pool = here
        code = self._door_code(obs)
        if code is None:
            return []
        pool_words = [prose_tokens(getattr(o, "what", "") or "") for o in pool]
        items = []
        for i, rec in enumerate(pool):
            ints = [int(v) for v in self._tag_vals(rec) if isinstance(v, (int, np.integer))]
            has_code = code in ints
            has_rare = any(
                sum(1 for o in pool if k in o.tags) < 3
                for k in rec.tags
                if k not in _QNAME_SKIP
            )
            if (self.use_prose_ints or self.use_prose_tokens) and not has_rare:
                words = pool_words[i]
                has_rare = any(sum(1 for ws in pool_words if w in ws) < 3 for w in words)
            items.append((has_code, has_rare))
        dec = self.use_policy.decide_search(
            items, epsilon=self.policy_epsilon, rng=self.policy_rng
        )
        idx = int(dec["idx"])
        dec["file"] = getattr(pool[idx], "fact_id", str(idx))
        self.last_policy = {**self.last_policy, **dec}
        if record:
            self.policy_traces.append(dec)
        return [pool[idx]]

    def _collect_search(self, obs: Obs, *, record: bool) -> dict[str, Any]:
        info: dict[str, Any] = {"taken": 0, "committed": 0, "mode": self.collect_mode}
        if self.collect_mode in ("off", "ignore") or self.world is None:
            return info
        if self._door_code(obs) is None:
            return info
        wpool = list(self.world.records())
        if self.use_here_match and self.store.enabled:
            here = self._station_name(obs)
            stations = set(STATION_NAMES.values())
            recs = list(self.store.records())
            other = False
            for rec in recs:
                vals = {str(v).lower() for v in rec.tags.values() if isinstance(v, str)}
                if (vals & stations) - ({here} if here else set()):
                    other = True
                    break
            if other:
                owned = {r.fact_id for r in recs}
                wpool = [r for r in wpool if getattr(r, "fact_id", None) not in owned]
        if (
            self.use_commit_here_only
            and self.use_here_match
            and self.store.enabled
            and self.collect_mode == "commit"
            and any(self._rec_names_here(r, obs) for r in self.store.records())
        ):
            return info
        picks = self._search_picks(wpool, obs, record=record)
        if not picks:
            return info
        self._in_hand_id = getattr(picks[0], "fact_id", None)
        if (
            self.use_commit_rare_only
            and self.use_here_match
            and self.store.enabled
            and self.collect_mode == "commit"
            and any(self._rec_names_here(r, obs) for r in self.store.records())
        ):
            pool_words = [prose_tokens(getattr(o, "what", "") or "") for o in wpool]
            rare_picks = []
            for rec in picks:
                words = prose_tokens(getattr(rec, "what", "") or "")
                if any(sum(1 for ws in pool_words if w in ws) < 3 for w in words):
                    rare_picks.append(rec)
            picks = rare_picks
            if not picks:
                return info
        info["taken"] = 1
        if self.collect_mode == "commit":
            n = 0
            for rec in picks:
                if self._commit_w_record(rec):
                    n += 1
            info["committed"] = n
        elif self.collect_mode == "peek":
            self._peek = list(picks)
        return info

    @staticmethod
    def _has_payload(rec: FactRecord) -> bool:
        return "action" in rec.tags or "do" in rec.tags

    def _wsel_picks(self, w_hits: list) -> list:
        """Which unread W hits to take. Untrained: first file, or all if dump.

        wcomp alt: first hit with action=/do=. wsel alt: newest when=.
        """
        if not w_hits:
            return []
        if self.use_wcomp_head and self.use_policy is not None:
            any_payload = any(self._has_payload(r) for r in w_hits)
            feat = UsePolicy.wcomp_features(any_payload, len(w_hits))
            dec = self.use_policy.decide_wcomp(
                feat, epsilon=self.policy_epsilon, rng=self.policy_rng
            )
            self.last_policy = {**self.last_policy, **dec}
            self.policy_traces.append(dec)
            if dec["wcomp_alt"]:
                complete = [r for r in w_hits if self._has_payload(r)]
                return [complete[0]] if complete else [w_hits[0]]
            return [w_hits[0]]
        if not self.use_wsel_head or self.use_policy is None:
            return [w_hits[0]]
        feat = UsePolicy.pick_features(len(w_hits))
        dec = self.use_policy.decide_wsel(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
        self.last_policy = {**self.last_policy, **dec}
        self.policy_traces.append(dec)
        if dec["wsel_alt"]:
            return [max(w_hits, key=self._note_when)]
        if self.wsel_dump:
            return list(w_hits)
        return [w_hits[0]]

    def _commit_w_record(self, rec: FactRecord) -> bool:
        if rec.fact_id in self._w_skip:
            return False
        if self.use_event_annotate:
            # Do not replace a committed note with the unread original (would wipe did=).
            if any(r.fact_id == rec.fact_id for r in self.store.records()):
                return False
        copied = FactRecord(
            fact_id=rec.fact_id,
            what=rec.what,
            when=self.t,
            drive_scores={"collect": 1.0},
            tags={k: v for k, v in rec.tags.items() if k not in ("source_file", "source")},
        )
        copied.tags["source"] = "W->S"
        return self.store.write(copied)

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

    def _rank_hits(self, hits: list, *, record: bool) -> list:
        if not hits or self.use_policy is None:
            return hits
        newest_t = max(self._note_when(r) for r in hits)
        items = [
            (self._note_when(r) == newest_t, int(r.tags.get("ok", 0)) == 1) for r in hits
        ]
        dec = self.use_policy.decide_rank(items, epsilon=self.policy_epsilon, rng=self.policy_rng)
        self.last_policy = {**self.last_policy, **dec}
        if record:
            self.policy_traces.append(dec)
        return [hits[int(dec["idx"])]]

    def _hits_for(self, obs: Obs) -> list:
        if self.use_search_head:
            return list(self._search_chosen)
        pool = self._pool()
        mode = self.retrieve_policy
        if mode == "policy":
            mode = str(self.last_policy.get("retrieve_mode") or "select")
        if mode == "dump":
            return pool
        return self._select_hits(obs, pool)

    def _value_tag(self) -> str:
        if self.use_vname_head:
            return str(self.last_policy.get("vname") or "")
        if self.use_key_head:
            return "do" if bool(self.last_policy.get("key_alt")) else "action"
        return self.value_key

    def _act_names(self) -> set[str]:
        """Innate motor names (body vocabulary). Not an English lexicon."""
        if self.domain == "dial":
            return {a.name.lower() for a in DialAction}
        return {a.name.lower() for a in Action}

    def _act_map(self) -> dict[str, int]:
        if self.domain == "dial":
            innate = {a.name.lower(): int(a) for a in DialAction}
        else:
            innate = {a.name.lower(): int(a) for a in Action}
        if not self.use_alias_bind:
            return innate
        m = dict(innate)
        stations = set(STATION_NAMES.values())
        for rec in self.store.records():
            did = rec.tags.get("did")
            if not isinstance(did, str):
                continue
            aid = innate.get(did.lower())
            if aid is None:
                continue
            if self.use_one_bind:
                bound = rec.tags.get("bind")
                if isinstance(bound, str):
                    vl = bound.lower()
                    if vl not in innate and vl not in stations:
                        m[vl] = aid
                continue
            for k, v in rec.tags.items():
                if not str(k).startswith("w") or not isinstance(v, str):
                    continue
                vl = v.lower()
                if vl in innate or vl in stations:
                    continue
                m[vl] = aid
        return m

    def _station_name(self, obs) -> str | None:
        """Innate station name for the current percept. Body vocabulary, not English."""
        if self.domain != "dial":
            return None
        if getattr(obs, "at_a", False):
            return STATION_NAMES[CH_A]
        if getattr(obs, "at_b", False):
            return STATION_NAMES[CH_B]
        if getattr(obs, "at_c", False):
            return STATION_NAMES[CH_C]
        return None

    def _rec_names_here(self, rec, obs) -> bool:
        station = self._station_name(obs)
        if not station:
            return False
        return any(
            isinstance(v, str) and str(v).lower() == station
            for k, v in rec.tags.items()
            if k not in _QNAME_SKIP
        )

    def _chosen_has_act_name(self) -> bool:
        fid = str(self.last_policy.get("file") or "")
        if not fid:
            return False
        acts = self._act_names()
        for rec in self.store.records():
            if rec.fact_id != fid:
                continue
            did = rec.tags.get("did")
            if isinstance(did, str) and did.lower() in acts:
                return True
            for k, v in rec.tags.items():
                if k in _QNAME_SKIP:
                    continue
                if isinstance(v, str) and v.lower() in acts:
                    return True
        return False

    def _maybe_revise(self, *, failed: bool) -> None:
        if self.use_policy is None or not self.store.enabled:
            return
        fid = str(self.last_policy.get("file") or "")
        has_act = self._chosen_has_act_name()
        feat = UsePolicy.revise_features(failed, has_act)
        dec = self.use_policy.decide_revise(
            feat, epsilon=self.policy_epsilon, rng=self.policy_rng
        )
        dec["file"] = fid
        dec["has_act_name"] = has_act
        self.last_policy = {**self.last_policy, **dec}
        self.policy_traces.append(dec)
        if not dec["revise"] or not fid:
            return
        if has_act:
            return
        if self.store.delete(fid):
            self._w_skip.add(fid)
            self.n_revised += 1

    def _sweep_unstamped(self) -> None:
        """Drop committed pages that never received an act name. Not a subject."""
        if not self.store.enabled:
            return
        acts = self._act_names()
        for rec in list(self.store.records()):
            vals = {
                str(v).lower()
                for v in rec.tags.values()
                if isinstance(v, str)
            }
            if vals & acts:
                continue
            if self.store.delete(rec.fact_id):
                self._w_skip.add(rec.fact_id)
                self.n_revised += 1

    def _rec_words(self, rec) -> set[str]:
        words = {
            str(v).lower()
            for k, v in rec.tags.items()
            if str(k).startswith("w") and isinstance(v, str)
        }
        if not words and getattr(rec, "what", ""):
            words = prose_tokens(rec.what)
        return words

    def _is_rare_in_world(self, rec) -> bool:
        return bool(self._rare_page_tokens(rec))

    def _rare_page_tokens(self, rec) -> set[str]:
        """Tokens on a note that are rare in unread W. Not a lexicon."""
        if self.world is None:
            return set()
        pool = list(self.world.records())
        pool_words = [prose_tokens(getattr(o, "what", "") or "") for o in pool]
        words = self._rec_words(rec)
        return {w for w in words if sum(1 for ws in pool_words if w in ws) < 3}

    def _keep_rare_words(self, rec, station: str | None) -> None:
        """Drop common page words so copy cannot treat a closed-lexicon token as a motor name."""
        rare = self._rare_page_tokens(rec)
        stations = set(STATION_NAMES.values())
        keep: list[str] = []
        seen: set[str] = set()
        for k, v in list(rec.tags.items()):
            if not str(k).startswith("w") or not isinstance(v, str):
                continue
            vl = v.lower()
            if vl in rare or vl in stations or (station and vl == station):
                if vl not in seen:
                    keep.append(vl)
                    seen.add(vl)
            del rec.tags[k]
        for i, vl in enumerate(keep):
            rec.tags[f"w{i}"] = vl

    def _bound_here_note(self, station: str | None):
        """The note that already binds an act at this station. One CS per place."""
        if not station or not self.store.enabled:
            return None
        for rec in self.store.records():
            vals = {str(v).lower() for v in rec.tags.values() if isinstance(v, str)}
            bind = rec.tags.get("bind")
            did = rec.tags.get("did")
            if station in vals and (isinstance(bind, str) or isinstance(did, str)):
                return rec
        return None

    def _in_hand_note(self):
        """The page attended this step, if it already lives in S."""
        fid = self._in_hand_id
        if not fid or not self.store.enabled:
            return None
        for rec in self.store.records():
            if rec.fact_id == fid:
                return rec
        return None

    def _commit_in_hand(self):
        """Commit the attended unread page. Coincidence, not a librarian leftover."""
        fid = self._in_hand_id
        if not fid or self.world is None or not self.store.enabled:
            return None
        if any(r.fact_id == fid for r in self.store.records()):
            return self._in_hand_note()
        for rec in self.world.records():
            if getattr(rec, "fact_id", None) == fid:
                if self._commit_w_record(rec):
                    return self._in_hand_note()
                return None
        return None

    def _unknown_here(self, obs) -> bool:
        """S already names some other station, but not this one. Clutter-only S is not 'unknown'."""
        if not self.use_stamp_new_here or not self.use_here_match or not self.store.enabled:
            return False
        station = self._station_name(obs)
        if not station:
            return False
        stations = set(STATION_NAMES.values())
        named: set[str] = set()
        for rec in self.store.records():
            vals = {str(v).lower() for v in rec.tags.values() if isinstance(v, str)}
            named |= vals & stations
        if not named:
            return False
        return station not in named

    def _commit_rare_unmarked(self) -> bool:
        """Commit one rare unread page that does not already live in S."""
        if self.world is None or not self.store.enabled:
            return False
        owned = {r.fact_id for r in self.store.records()}
        stations = set(STATION_NAMES.values())
        for rec in self.world.records():
            if getattr(rec, "fact_id", None) in owned:
                continue
            if not self._is_rare_in_world(rec):
                continue
            vals = {str(v).lower() for v in rec.tags.values() if isinstance(v, str)}
            if vals & stations:
                continue
            return self._commit_w_record(rec)
        return False

    def _pick_stamp_note(self, rare_recs, station: str | None):
        """Prefer an unmarked rare page when this station is new."""
        stations = set(STATION_NAMES.values())
        unmarked = []
        named_here = []
        for cand in rare_recs:
            vals = {str(v).lower() for v in cand.tags.values() if isinstance(v, str)}
            other = (vals & stations) - ({station} if station else set())
            if other:
                continue
            did = str(cand.tags.get("did") or "").lower()
            if not (vals & stations) and not did:
                unmarked.append(cand)
            else:
                named_here.append(cand)
        if self.use_stamp_new_here and unmarked:
            return unmarked[0]
        pool = unmarked + named_here
        return pool[0] if pool else None

    def _page_stream(self, rec) -> list[str]:
        text = ""
        if self.world is not None:
            fid = rec.fact_id
            for o in self.world.records():
                if getattr(o, "fact_id", None) == fid:
                    text = getattr(o, "what", "") or ""
                    break
        if not text:
            text = getattr(rec, "what", "") or ""
        return prose_token_stream(text)

    def _stamp_one_bind(self, rec) -> None:
        """Alias the first rare page word in stream order. Not a lexicon."""
        if rec.tags.get("bind"):
            return
        rare = {
            str(v).lower()
            for k, v in rec.tags.items()
            if str(k).startswith("w") and isinstance(v, str)
        }
        stations = set(STATION_NAMES.values())
        innate = self._act_names()
        for w in self._page_stream(rec):
            if w in rare and w not in stations and w not in innate:
                rec.tags["bind"] = w
                return

    def _maybe_annotate(self, action_name: str, obs=None) -> None:
        """Write the act (and station, if here-match) the body just did onto a rare note."""
        if self.use_policy is None or not self.store.enabled:
            return
        if action_name not in self._act_names():
            return
        recs = list(self.store.records())
        if not recs and not (self.use_block_here and self._in_hand_id):
            return
        rare_recs = [r for r in recs if self._is_rare_in_world(r)]
        rare = bool(rare_recs)
        feat = UsePolicy.features(rare, True)
        do_write = self.force_write
        if not do_write:
            dec = self.use_policy.decide_write(
                feat, epsilon=self.policy_epsilon, rng=self.policy_rng
            )
            self.last_policy = {**self.last_policy, **dec}
            self.policy_traces.append(dec)
            do_write = bool(dec["write"])
        station = self._station_name(obs) if self.use_here_match else None
        new_here = bool(self.use_stamp_new_here and station and self._unknown_here(obs))
        if not do_write and not new_here:
            return
        if not rare and not new_here and not (self.use_block_here and self._in_hand_id):
            return
        rec = None
        if self.use_block_here and self.use_here_match:
            rec = self._bound_here_note(station)
            if rec is None:
                rec = self._in_hand_note() or self._commit_in_hand()
                if rec is not None and not self._is_rare_in_world(rec) and not new_here:
                    rec = None
                if rec is None and new_here:
                    if self._commit_rare_unmarked():
                        rare_recs = [r for r in self.store.records() if self._is_rare_in_world(r)]
                        rec = self._pick_stamp_note(rare_recs, station)
        else:
            rec = self._pick_stamp_note(rare_recs, station)
            if rec is None and new_here:
                if self._commit_rare_unmarked():
                    rare_recs = [r for r in self.store.records() if self._is_rare_in_world(r)]
                    rec = self._pick_stamp_note(rare_recs, station)
        if rec is None:
            return
        vals = {str(v).lower() for v in rec.tags.values() if isinstance(v, str)}
        did = str(rec.tags.get("did") or "").lower()
        need_act = action_name not in vals and did != action_name
        need_st = bool(station) and station not in vals
        if not need_act and not need_st:
            return
        n = sum(1 for k in rec.tags if str(k).startswith("w"))
        if need_act:
            if self.use_did_stamp:
                rec.tags["did"] = action_name
                self._keep_rare_words(rec, station)
                n = sum(1 for k in rec.tags if str(k).startswith("w"))
            else:
                rec.tags[f"w{n}"] = action_name
                n += 1
        if need_st:
            rec.tags[f"w{n}"] = station
        if self.use_one_bind and rec.tags.get("did"):
            self._stamp_one_bind(rec)
        if self.store.write(rec):
            self.n_annotated += 1
            if self.use_revise_head:
                self._sweep_unstamped()

    def _choose_vname(self, recs: list, obs: Obs, *, record: bool) -> None:
        if not self.use_vname_head or self.use_policy is None:
            return
        q = self._obs_query(obs)
        qkey = next(iter(q)) if q else ""
        pool = self._qname_pool()
        keys = sorted({k for r in recs for k in r.tags if k not in _QNAME_SKIP})
        if self.use_prose_tokens:
            keys = [
                k
                for k in keys
                if any(isinstance(r.tags.get(k), str) and r.tags.get(k) for r in recs)
            ]
        elif self.use_prose_ints:
            keys = [
                k
                for k in keys
                if any(
                    isinstance(r.tags.get(k), (int, np.integer))
                    and 0 <= int(r.tags[k]) < self.n_actions
                    for r in recs
                )
            ]
        if not keys:
            return
        items = []
        code = self._door_code(obs)
        for k in keys:
            if self.use_prose_tokens:
                val = next((str(r.tags.get(k)) for r in recs if isinstance(r.tags.get(k), str)), "")
                val = val.lower()
                is_act = val in self._act_map()
                is_common = (
                    sum(
                        1
                        for r in pool
                        if any(
                            isinstance(r.tags.get(kk), str) and str(r.tags.get(kk)).lower() == val
                            for kk in r.tags
                            if kk not in _QNAME_SKIP
                        )
                    )
                    >= 3
                )
                # Untrained prefers common words (first feat). Trained must copy the act-name token.
                items.append((is_common, is_act))
            elif self.use_prose_ints:
                val = next((r.tags.get(k) for r in recs if k in r.tags), None)
                is_code = code is not None and val == code
                val_common = (
                    sum(
                        1
                        for r in pool
                        if any(
                            r.tags.get(kk) == val
                            for kk in r.tags
                            if kk not in _QNAME_SKIP
                        )
                    )
                    >= 3
                )
                items.append((is_code, val_common))
            else:
                is_query = k == qkey
                key_common = sum(1 for r in pool if k in r.tags) >= 3
                items.append((is_query, key_common))
        dec = self.use_policy.decide_vname(
            items, epsilon=self.policy_epsilon, rng=self.policy_rng
        )
        dec["vname"] = keys[int(dec["idx"])]
        dec["vnames"] = keys
        if self.use_prose_tokens:
            dec["is_common"] = bool(items[int(dec["idx"])][0])
            dec["is_act"] = bool(items[int(dec["idx"])][1])
        elif self.use_prose_ints:
            dec["is_code"] = bool(items[int(dec["idx"])][0])
            dec["val_common"] = bool(items[int(dec["idx"])][1])
        self.last_policy = {**self.last_policy, **dec}
        if record:
            self.policy_traces.append(dec)

    def _apply_record_bias(self, logits: np.ndarray, rec: FactRecord, obs: Obs) -> None:
        if self.use_prose_tokens:
            key = self._value_tag() if self.use_read else ""
            val = rec.tags.get(key)
            if isinstance(val, str):
                act = self._act_map().get(val.lower())
                if act is not None:
                    logits[int(act)] += 3.0
            return
        act = rec.tags.get(self._value_tag() if self.use_read else "action")
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

    def _store_bias(
        self,
        obs: Obs,
        *,
        novelty: float = 0.0,
        record_use: bool = True,
        record_search: bool | None = None,
    ) -> np.ndarray:
        """Retrieve facts and bias action logits. Knowledge lives in S, not ρ."""
        if record_search is None:
            record_search = record_use
        self._search_chosen = []
        self._in_hand_id = None
        # Match / open-name first so collect's W lookup uses this step's query name, not a stale one.
        if self.use_qname_head:
            self._choose_qname(obs, record=record_use)
        elif self.use_match_head and self.use_policy is not None:
            at = self._door_code(obs) is not None
            feat = UsePolicy.features(at, False)
            dec = self.use_policy.decide_match(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            self.last_policy = {**self.last_policy, **dec}
            if record_use:
                self.policy_traces.append(dec)
        self.collect(obs, novelty=novelty, record=record_search)
        if self.use_search_head:
            self._search_chosen = self._search_picks(self._pool(), obs, record=record_search)
        if self.retrieve_policy == "policy":
            self._choose_retrieve(obs)
        logits = np.zeros(self.n_actions, dtype=np.float64)
        apply = True
        hits = self._hits_for(obs)
        chosen = hits
        if self.use_pick and self.use_policy is not None:
            feat = UsePolicy.pick_features(len(hits))
            dec = self.use_policy.decide_pick(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            self.last_policy = {**self.last_policy, **dec}
            if record_use:
                self.policy_traces.append(dec)
            if dec["one"] and hits:
                chosen = self._rank_hits(hits, record=record_use) if self.use_rank else [self._newest(hits)]
        elif self.use_rank and self.use_policy is not None and hits:
            chosen = self._rank_hits(hits, record=record_use)
        if self.use_vname_head and self.use_policy is not None and chosen:
            self._choose_vname(chosen, obs, record=record_use)
        elif self.use_key_head and self.use_policy is not None:
            feat = UsePolicy.features(bool(chosen), False)
            dec = self.use_policy.decide_key(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            self.last_policy = {**self.last_policy, **dec}
            if record_use:
                self.policy_traces.append(dec)
        if self.collect_mode == "policy":
            apply = bool(self.last_policy.get("apply", False))
        elif self.force_use:
            apply = True
        elif self.use_read and self.use_policy is not None:
            feat = UsePolicy.features(bool(chosen), False)
            dec = self.use_policy.decide_use(feat, epsilon=self.policy_epsilon, rng=self.policy_rng)
            self.last_policy = {**self.last_policy, **dec}
            if record_use:
                self.policy_traces.append(dec)
            apply = bool(dec["use"])
            if self.use_event_annotate and bool(self.last_policy.get("is_act")):
                # Selected innate act name: copy is frozen grammar. A non-act token
                # does not bias the motor, so untrained (common word) stays HOLD.
                apply = True
                if self.use_here_match:
                    # File is a fact about *this* station, not a global motor.
                    apply = any(self._rec_names_here(r, obs) for r in chosen)
                    self.last_policy["here_match"] = apply
        if apply:
            for rec in chosen:
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
        # Species prior: try a default motor at a station (OPEN on doors, HOLD on dial).
        # Dial HOLD is wrong on A and C so empty S cannot look like Store-works.
        # Life knowledge must come from S (or fragile ρ after recent success).
        if self.domain == "dial":
            at_station = bool(
                getattr(obs, "at_a", False)
                or getattr(obs, "at_b", False)
                or getattr(obs, "at_c", False)
            )
            if at_station and not (explore and self._unknown_here(obs)):
                logits[DialAction.HOLD] += 1.5
                logits[DialAction.PRESS] -= 0.3
                logits[DialAction.TUNE] -= 0.3
        else:
            at_door = obs.at_red_door or obs.at_blue_door or obs.at_green_door
            if at_door:
                logits[Action.OPEN] += 1.5
                logits[Action.USE_KEY] -= 0.5
        # Probe records use. Life usually does not (write traces only), unless a free-life
        # search head must leave traces while the motor still explores.
        record_use = not explore
        record_search = (not explore) or self.record_search_on_explore
        logits = logits + self._store_bias(
            obs, novelty=novelty, record_use=record_use, record_search=record_search
        )

        if self.domain == "dial":
            at_station = bool(
                getattr(obs, "at_a", False)
                or getattr(obs, "at_b", False)
                or getattr(obs, "at_c", False)
            )
            if not at_station:
                for a in (DialAction.PRESS, DialAction.HOLD, DialAction.TUNE, DialAction.FLIP):
                    logits[int(a)] -= 5.0
        else:
            at_door = obs.at_red_door or obs.at_blue_door or obs.at_green_door
            # Hard constraints from current percept (not knowledge).
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
        if getattr(obs, "event_open_failed", False):
            event = "open_failed"
        elif getattr(obs, "event_key_worked", False):
            event = "key_worked"
        elif getattr(obs, "last_succeeded", False) and getattr(obs, "at_blue_door", False):
            event = "blue_opened"
        elif getattr(obs, "last_succeeded", False) and getattr(obs, "at_green_door", False):
            event = "green_opened"
        elif self.domain == "dial" and getattr(obs, "last_ok", False):
            event = "dial_ok"
        elif self.domain == "dial" and getattr(obs, "last_failed", False):
            event = "dial_failed"

        action_name = info.get("action")
        if self.domain == "dial":
            name_to_id = {a.name.lower(): int(a) for a in DialAction}
        else:
            name_to_id = {a.name.lower(): int(a) for a in Action}
        opened = bool(info.get("opened"))
        door = self._door_code(obs) if self.native else None

        if self.use_revise_head and success is False:
            self._maybe_revise(failed=True)

        if self.use_event_annotate and opened and action_name:
            self._maybe_annotate(str(action_name).lower(), obs)

        if self.use_policy is not None and self.write_from_events:
            # v9: frozen WHAT = {here, the act that opened}. Learned WHEN.
            # v14 B: schema head may omit action=; integer still comes from the event.
            if opened and door is not None and action_name in name_to_id:
                query = {self.place_key: door}
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
                    pkey = self.place_key
                    vkey = self.value_key
                    if self.use_wplace_head:
                        wpl = self.use_policy.decide_wplace(
                            feat, epsilon=self.policy_epsilon, rng=self.policy_rng
                        )
                        self.last_policy = {**self.last_policy, **wpl}
                        self.policy_traces.append(wpl)
                        pkey = "here" if bool(wpl["wplace_alt"]) else "door"
                    if complete and self.use_wkey_head:
                        wk = self.use_policy.decide_wkey(
                            feat, epsilon=self.policy_epsilon, rng=self.policy_rng
                        )
                        self.last_policy = {**self.last_policy, **wk}
                        self.policy_traces.append(wk)
                        vkey = "do" if bool(wk["wkey_alt"]) else "action"
                    tags: dict[str, Any] = {pkey: door}
                    if complete:
                        tags[vkey] = act
                        if self.mark_ok:
                            tags["ok"] = 1
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
