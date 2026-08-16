"""Three-memory agent: frozen cortex + ρ + S + innate write/retrieve rules."""

from __future__ import annotations

from typing import Any, Sequence

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
_QNAME_SKIP = frozenset(
    {
        "source",
        "source_file",
        "when",
        "ok",
        "what",
        "did",
        "bind",
        "hyp",
        "trials",
        "wins",
        "losses",
        "support",
        "contradiction",
    }
)


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
        use_in_hand_new_here: bool = False,
        use_find_novel: bool = False,
        use_retry_novel: bool = False,
        use_local_alias: bool = False,
        use_keep_steerer: bool = False,
        use_count_search: bool = False,
        use_hyp_survive: bool = False,
        use_bind_match: bool = False,
        use_evidence: bool = False,
        use_compose: bool = False,
        use_context_kappa: bool = False,
        use_acquire_ctx: bool = False,
        use_acquire_skel: bool = False,
        use_acquire_relate: bool = False,
        use_alias_fingerprint: bool = False,
        use_continuity_mark: bool = False,
        use_symbol_ground: bool = False,
        use_symbol_sequence: bool = False,
        use_inquire: bool = False,
        use_source_reliability: bool = False,
        use_source_perspective: bool = False,
        use_source_interpretation: bool = False,
        n_actions: int = 4,
        domain: str = "door",
    ):
        if retrieve_policy not in ("select", "dump", "policy"):
            raise ValueError(retrieve_policy)
        if collect_mode not in ("off", "commit", "peek", "policy"):
            raise ValueError(collect_mode)
        if n_actions not in (4, 5):
            raise ValueError(n_actions)
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
        if use_in_hand_new_here and not use_stamp_new_here:
            raise ValueError("use_in_hand_new_here requires use_stamp_new_here")
        if use_find_novel and not use_search_head:
            raise ValueError("use_find_novel requires use_search_head")
        if use_retry_novel and not use_find_novel:
            raise ValueError("use_retry_novel requires use_find_novel")
        if use_local_alias and not use_alias_bind:
            raise ValueError("use_local_alias requires use_alias_bind")
        if use_keep_steerer and not use_here_match:
            raise ValueError("use_keep_steerer requires use_here_match")
        if use_count_search and not use_search_head:
            raise ValueError("use_count_search requires use_search_head")
        if use_hyp_survive and not use_here_match:
            raise ValueError("use_hyp_survive requires use_here_match")
        if use_bind_match and not use_alias_bind:
            raise ValueError("use_bind_match requires use_alias_bind")
        if use_evidence and not use_bind_match:
            raise ValueError("use_evidence requires use_bind_match")
        if use_evidence and not use_hyp_survive:
            raise ValueError("use_evidence requires use_hyp_survive")
        if use_compose and not use_evidence:
            raise ValueError("use_compose requires use_evidence")
        if use_context_kappa and not use_compose:
            raise ValueError("use_context_kappa requires use_compose")
        if use_acquire_ctx and not use_context_kappa:
            raise ValueError("use_acquire_ctx requires use_context_kappa")
        if use_acquire_skel and not use_acquire_ctx:
            raise ValueError("use_acquire_skel requires use_acquire_ctx")
        if use_acquire_relate and not use_acquire_skel:
            raise ValueError("use_acquire_relate requires use_acquire_skel")
        if use_acquire_relate and not use_hyp_survive:
            raise ValueError("use_acquire_relate requires use_hyp_survive")
        if use_acquire_relate and not use_evidence:
            raise ValueError("use_acquire_relate requires use_evidence")
        if use_alias_fingerprint and not use_acquire_relate:
            raise ValueError("use_alias_fingerprint requires use_acquire_relate")
        if use_continuity_mark and not use_acquire_relate:
            raise ValueError("use_continuity_mark requires use_acquire_relate")
        if use_symbol_ground and not use_acquire_relate:
            raise ValueError("use_symbol_ground requires use_acquire_relate")
        if use_symbol_sequence and not use_symbol_ground:
            raise ValueError("use_symbol_sequence requires use_symbol_ground")
        if use_inquire and not use_symbol_sequence:
            raise ValueError("use_inquire requires use_symbol_sequence")
        if use_source_reliability and not use_inquire:
            raise ValueError("use_source_reliability requires use_inquire")
        if use_source_perspective and not use_source_reliability:
            raise ValueError("use_source_perspective requires use_source_reliability")
        if use_source_interpretation and not use_source_perspective:
            raise ValueError("use_source_interpretation requires use_source_perspective")
        if value_key not in ("action", "do"):
            raise ValueError(value_key)
        if not isinstance(place_key, str) or not place_key:
            raise ValueError(place_key)
        self.domain = domain
        self.n_actions = n_actions
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
        self.use_in_hand_new_here = use_in_hand_new_here
        self.use_find_novel = use_find_novel
        self.use_retry_novel = use_retry_novel
        self.use_local_alias = use_local_alias
        self.use_keep_steerer = use_keep_steerer
        self.use_count_search = use_count_search
        self.use_hyp_survive = use_hyp_survive
        self.use_bind_match = use_bind_match
        self.use_evidence = use_evidence
        self.use_compose = use_compose
        self.use_context_kappa = use_context_kappa
        self.use_acquire_ctx = use_acquire_ctx
        self.use_acquire_skel = use_acquire_skel
        self.use_acquire_relate = use_acquire_relate
        self.use_alias_fingerprint = use_alias_fingerprint
        self.use_continuity_mark = use_continuity_mark
        self.use_symbol_ground = use_symbol_ground
        self.use_symbol_sequence = use_symbol_sequence
        self.use_inquire = use_inquire
        self.use_source_reliability = use_source_reliability
        self.use_inquire_liveness = bool(use_source_reliability)
        self.use_source_perspective = use_source_perspective
        self.use_source_interpretation = use_source_interpretation
        self.reliability_lambda = 4
        self.reliability_n_min = 2
        self.reliability_jaccard = 0.5
        # Perspective: alignment margin uses same λ/n_min; strong exposure atoms scaffolded
        self.perspective_lambda = 4
        self.perspective_n_min = 2
        self.perspective_jaccard = 0.5
        self.perspective_strong_exposure = frozenset(
            {"exp_delivered", "exp_ack_read", "exp_receipt"}
        )
        self.perspective_weak_exposure = frozenset(
            {"exp_present", "exp_absent", "exp_sensor_connected"}
        )
        # Interpretation: factorized reconstruct; no Jaccard; independent anchors
        self.interpret_lambda = 4
        self.interpret_n_min = 2
        self.inquire_budget = 8
        self.inquire_cost_ask = 2
        self.inquire_cost_experiment = 5
        self._inquire_probes_used = 0
        self.first_internal_fail: dict[str, Any] | None = None
        self._last_chosen_ids: list[str] = []
        self._peek: list[FactRecord] = []
        self._search_chosen: list = []
        self._in_hand_id: str | None = None
        self._w_skip: set[str] = set()
        self._lived_kappa: str | None = None
        self._lived_bind: str | None = None
        self._lived_pending: bool = False
        self._skel_prev: str | None = None
        self._rel_prev_visible: list[str] | None = None
        self.n_revised = 0
        self.n_annotated = 0
        self.last_policy: dict[str, Any] = {}
        self.policy_traces: list[dict[str, Any]] = []
        self._weight_hash0 = self.cortex.weight_hash()

    def weight_hash(self) -> str:
        return self.cortex.weight_hash()

    def weights_unchanged(self) -> bool:
        return self.weight_hash() == self._weight_hash0

    def _clear_lived_context(self) -> None:
        """One-shot lived (κ, bind) from compose — never survive a new act/outcome."""
        self._lived_kappa = None
        self._lived_bind = None
        self._lived_pending = False

    def _clear_skel_prev(self) -> None:
        """Transient symbol adjacency — cleared on ρ reset / newborn (fresh agent)."""
        self._skel_prev = None

    def _clear_rel_prev_visible(self) -> None:
        """RELATE episode transient — cleared by end_event_episode / reset_rho / newborn."""
        self._rel_prev_visible = None

    def end_event_episode(self) -> dict[str, Any]:
        """Clear only RELATE transient event state. Not S, ρ, or κ."""
        had = self._rel_prev_visible is not None
        self._clear_rel_prev_visible()
        return {"ok": True, "cleared": had}

    def reset_rho(self) -> None:
        self.rho.reset()
        self._peek = []
        self._w_skip = set()
        self._in_hand_id = None
        self._clear_lived_context()
        self._clear_skel_prev()
        self._clear_rel_prev_visible()

    def reset_store(self) -> None:
        self.store.reset()
        self._peek = []
        self._clear_lived_context()
        self._clear_skel_prev()
        self._clear_rel_prev_visible()

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
            use_in_hand_new_here=self.use_in_hand_new_here,
            use_find_novel=self.use_find_novel,
            use_retry_novel=self.use_retry_novel,
            use_local_alias=self.use_local_alias,
            use_keep_steerer=self.use_keep_steerer,
            use_count_search=self.use_count_search,
            use_hyp_survive=self.use_hyp_survive,
            use_bind_match=self.use_bind_match,
            use_evidence=self.use_evidence,
            use_compose=self.use_compose,
            use_context_kappa=self.use_context_kappa,
            use_acquire_ctx=self.use_acquire_ctx,
            use_acquire_skel=self.use_acquire_skel,
            use_acquire_relate=self.use_acquire_relate,
            use_alias_fingerprint=self.use_alias_fingerprint,
            use_continuity_mark=self.use_continuity_mark,
            use_symbol_ground=self.use_symbol_ground,
            use_symbol_sequence=self.use_symbol_sequence,
            use_inquire=self.use_inquire,
            use_source_reliability=self.use_source_reliability,
            use_source_perspective=self.use_source_perspective,
            use_source_interpretation=self.use_source_interpretation,
            n_actions=self.n_actions,
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

    def _body_enum(self):
        """Motors from body size, not a domain= label."""
        return DialAction if self.n_actions == 5 else Action

    def _affordances(self, obs) -> list[int]:
        """Percept-legal acts. Not knowledge."""
        if self.n_actions == 5:
            return [int(a) for a in DialAction]
        acts = [int(Action.WAIT), int(Action.OPEN)]
        if getattr(obs, "key_visible", False) and not getattr(obs, "has_key", False):
            acts.append(int(Action.PICK_KEY))
        if getattr(obs, "has_key", False):
            acts.append(int(Action.USE_KEY))
        return acts

    def _door_code(self, obs) -> int | None:
        """Place code for the current station (door code or dial channel)."""
        if getattr(obs, "at_a", False):
            return CH_A
        if getattr(obs, "at_b", False):
            return CH_B
        if getattr(obs, "at_c", False):
            return CH_C
        if getattr(obs, "at_red_door", False):
            return DOOR_RED
        if getattr(obs, "at_blue_door", False):
            return DOOR_BLUE
        if getattr(obs, "at_green_door", False):
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

    def _s_token_set(self) -> set[str]:
        """Tokens already committed. Find-novel treats these as known, not new rares."""
        out: set[str] = set()
        if not self.store.enabled:
            return out
        for rec in self.store.records():
            out |= prose_tokens(getattr(rec, "what", "") or "")
            for v in rec.tags.values():
                if isinstance(v, str):
                    out.add(v.lower())
        return out

    def _filter_find_novel(self, pool: list) -> list:
        """Keep unread pages that would add the most rare tokens S does not already have.

        Frozen grammar, not a filename ranker and not unique-rare restored: clutter hapax
        stay rare; they lose only when some other unread page has more novel rares.
        """
        if not self.use_find_novel or not pool:
            return pool
        pool_words = [prose_tokens(getattr(o, "what", "") or "") for o in pool]
        known = self._s_token_set()
        scores = []
        for words in pool_words:
            rare = {w for w in words if sum(1 for ws in pool_words if w in ws) < 3}
            scores.append(len(rare - known))
        m = max(scores)
        if m <= 0:
            return pool
        return [r for r, n in zip(pool, scores) if n == m]

    def _full_world_novel_count(self, rec) -> int:
        """Rare tokens on rec vs the whole unread library, minus S. Not leftover-pile rarity."""
        if self.world is None:
            return 0
        base_words = [prose_tokens(getattr(o, "what", "") or "") for o in self.world.records()]
        words = prose_tokens(getattr(rec, "what", "") or "")
        rare = {w for w in words if sum(1 for ws in base_words if w in ws) < 3}
        return len(rare - self._s_token_set())

    def _unused_novel_remain(self, obs: Obs, wpool: list | None = None) -> bool:
        """True when an unowned page still adds a full-W rare token S lacks."""
        if not self.use_retry_novel or not self.use_find_novel or self.world is None:
            return False
        owned = {r.fact_id for r in self.store.records()} if self.store.enabled else set()
        pool = list(wpool) if wpool is not None else list(self.world.records())
        return any(
            getattr(r, "fact_id", None) not in owned
            and getattr(r, "fact_id", None) not in self._w_skip
            and self._full_world_novel_count(r) > 0
            for r in pool
        )

    def _in_hand_is_novel(self) -> bool:
        if not self._in_hand_id or self.world is None:
            return False
        owned = {r.fact_id for r in self.store.records()} if self.store.enabled else set()
        if self._in_hand_id in owned or self._in_hand_id in self._w_skip:
            return False
        rec = next((r for r in self.world.records() if getattr(r, "fact_id", None) == self._in_hand_id), None)
        return rec is not None and self._full_world_novel_count(rec) > 0

    def _search_picks(self, pool: list, obs: Obs, *, record: bool) -> list:
        if not pool or self.use_policy is None:
            return []
        pool = [r for r in pool if getattr(r, "fact_id", None) not in self._w_skip]
        if not pool:
            return []
        pool = self._filter_find_novel(pool)
        if not pool:
            return []
        if self.use_here_match:
            here = [r for r in pool if self._rec_names_here(r, obs)]
            if here:
                pool = here
                pool = self._prefer_untried(pool)
        if self.use_bind_match:
            matched = self._match_applicable(pool, obs)
            if not matched:
                return []
            pool = matched
        code = self._door_code(obs)
        if code is None:
            return []
        pool_words = [prose_tokens(getattr(o, "what", "") or "") for o in pool]
        items = []
        for i, rec in enumerate(pool):
            ints = [int(v) for v in self._tag_vals(rec) if isinstance(v, (int, np.integer))]
            has_code = code in ints
            has_novel = self._full_world_novel_count(rec) > 0
            has_rare = any(
                sum(1 for o in pool if k in o.tags) < 3
                for k in rec.tags
                if k not in _QNAME_SKIP
            )
            if (self.use_prose_ints or self.use_prose_tokens) and not has_rare:
                words = pool_words[i]
                has_rare = any(sum(1 for ws in pool_words if w in ws) < 3 for w in words)
            # Count of unread rares is cardinality, not + in cortex.
            items.append((has_novel if self.use_count_search else has_code, has_rare))
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
            if self.use_retry_novel and self._unused_novel_remain(obs, wpool):
                owned = {r.fact_id for r in self.store.records()}
                wpool = [r for r in wpool if getattr(r, "fact_id", None) not in owned]
            else:
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
        if self.use_find_novel:
            # Attend the novel page; stamp commits it. Do not keep copying leftover hapax into S.
            return info
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
        return {a.name.lower() for a in self._body_enum()}

    def _act_map(self, recs=None) -> dict[str, int]:
        innate = {a.name.lower(): int(a) for a in self._body_enum()}
        if not self.use_alias_bind:
            return innate
        m = dict(innate)
        stations = set(STATION_NAMES.values())
        if self.use_local_alias:
            sources = list(recs) if recs is not None else []
        else:
            sources = list(self.store.records()) if self.store.enabled else []
        for rec in sources:
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

    def _current_stream(self, obs) -> set[str]:
        """Tokens in the current observation. Never S note text."""
        out: set[str] = set()
        raw = getattr(obs, "tokens", None)
        if raw:
            out |= {str(t).lower() for t in raw if t}
        if self.world is not None and self._in_hand_id:
            rec = next(
                (r for r in self.world.records() if getattr(r, "fact_id", None) == self._in_hand_id),
                None,
            )
            if rec is not None:
                out |= {w.lower() for w in prose_tokens(getattr(rec, "what", "") or "")}
        return out

    def _bind_in_stream(self, rec, obs) -> bool:
        bind = rec.tags.get("bind")
        if not isinstance(bind, str) or not bind:
            return False
        return bind.lower() in self._current_stream(obs)

    def _match_applicable(self, recs: list, obs) -> list:
        """Boolean gate: bind_present_in_current_stream. No token identity."""
        if not self.use_bind_match:
            return recs
        return [r for r in recs if self._bind_in_stream(r, obs)]

    def _match_frontier(
        self, recs: list, frontier: set[str], *, kappa: str | None = None
    ) -> list:
        """MATCH against a derived frontier only. Not observation ∪ derived.

        When use_context_kappa and kappa is set: if any eligible fact has ctx,
        discard all untagged facts and keep only ctx == kappa. Zero matches → [].
        Caller must pass only unvisited facts — a consumed ctx row must not poison
        later untagged matches at the same bind. Observation MATCH never uses this.
        """
        out = []
        for r in recs:
            bind = r.tags.get("bind")
            if isinstance(bind, str) and bind.lower() in frontier:
                out.append(r)
        if not self.use_context_kappa or kappa is None:
            return out
        tagged = [r for r in out if isinstance(r.tags.get("ctx"), str) and r.tags.get("ctx")]
        if not tagged:
            return out
        return [r for r in tagged if str(r.tags.get("ctx")) == kappa]

    def _compose_choose(self, obs) -> list:
        """Compose along chosen relations: non-motor did becomes the next frontier.

        Act-local only. A fact_id is consumed at most once. No hop cap — bound by |S|.
        Does not write shortcuts into S.

        With use_context_kappa: after each selected non-motor hop, carry κ and
        match derived facts by ctx. Motor hops never step κ. Hop-1 motor returns
        without initializing κ.

        With use_acquire_ctx: retain the compose-local (κ, frontier bind) when a
        non-motor hop advances κ so observe_outcome can author a contextual
        continuation after a HOLD. No second κ engine.
        """
        if self.use_acquire_ctx:
            self._clear_lived_context()
        if not self.use_compose or not self.store.enabled:
            return []
        from three_memory.kappa import edge_sem, kappa_seed, kappa_step

        store_recs = list(self.store.records())
        motors = self._act_names()
        visited: set[str] = set()
        frontier: set[str] | None = None
        kappa: str | None = None
        hops = 0

        def _hold(*, evidence_tie: bool = False) -> list:
            self.last_policy["compose_hold"] = True
            self.last_policy["evidence_resolved"] = False
            self.last_policy["evidence_tie"] = evidence_tie
            self.last_policy["context_kappa"] = kappa
            self.last_policy["compose_hops"] = hops
            return []

        # Termination: each fact at most once ⇒ ≤ |S| iterations.
        for _ in range(max(len(store_recs), 1)):
            # Visited exclusion before ctx filter: a consumed ctx fact must not
            # poison "any eligible has ctx" on a later revisit of the same bind.
            unvisited = [
                r
                for r in store_recs
                if str(getattr(r, "fact_id", "") or "") not in visited
            ]
            if frontier is None:
                eligible = self._match_applicable(unvisited, obs)
                projected = self._continuity_first_hop(obs)
                if projected is None:
                    projected = self._grounding_first_hop(obs)
                if projected is not None:
                    if not projected:
                        return _hold()
                    winners = projected
                else:
                    if not eligible:
                        return _hold()
                    winners = self._compose_select(eligible, frontier)
            else:
                eligible = self._match_frontier(unvisited, frontier, kappa=kappa)
                if not eligible:
                    return _hold()
                winners = self._compose_select(eligible, frontier)
            if not winners:
                return _hold(evidence_tie=True)
            rec = winners[0]
            fid = str(getattr(rec, "fact_id", "") or "")
            if fid:
                visited.add(fid)
            hops += 1
            did = rec.tags.get("did")
            bind = rec.tags.get("bind")
            if not isinstance(did, str) or not did:
                return _hold()
            did_l = did.lower()
            if did_l in motors:
                self.last_policy["compose_hold"] = False
                self.last_policy["evidence_resolved"] = True
                self.last_policy["evidence_tie"] = False
                self.last_policy["compose_hops"] = hops
                self.last_policy["context_kappa"] = kappa
                if self.use_acquire_ctx:
                    # Motor selected — acquisition is not pending from this act.
                    self._clear_lived_context()
                return [rec]
            # Selected non-motor consequent: advance κ (if enabled), then frontier.
            if self.use_context_kappa:
                if not isinstance(bind, str) or not bind:
                    return _hold()
                if kappa is None:
                    kappa = kappa_seed(bind)
                kappa = kappa_step(kappa, edge_sem(bind, did))
                if self.use_acquire_ctx:
                    self._lived_kappa = kappa
                    self._lived_bind = did_l
                    self._lived_pending = True
            frontier = {did_l}
        return _hold()

    def _compose_select(self, eligible: list, frontier: set[str] | None) -> list:
        """Evidence choose, optionally via pairwise fingerprint projection.

        When fingerprint rows exist, projection failure is HOLD — never fall back
        to raw evidence (that would complete Control routes under ambiguous cliques).
        With the flag on but no fingerprint rows yet, raw evidence still applies (A0).
        """
        if self.use_alias_fingerprint and self._fingerprint_rows():
            return self._fingerprint_project_eligible(eligible, frontier)
        return self._evidence_choose(eligible)

    def _fingerprint_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_fingerprint"
        ]

    def _fingerprint_witnesses(self, alias: str) -> dict[str, list[tuple[str, str]]]:
        """probe_context → list of (action, outcome) witnesses for alias."""
        al = alias.lower()
        out: dict[str, list[tuple[str, str]]] = {}
        for rec in self._fingerprint_rows():
            if str(rec.tags.get("alias") or "").lower() != al:
                continue
            ctx = str(rec.tags.get("probe_context") or "")
            act = str(rec.tags.get("action") or "").lower()
            outcome = str(rec.tags.get("observed_outcome") or "").lower()
            if not ctx or not act or not outcome:
                continue
            out.setdefault(ctx, []).append((act, outcome))
        return out

    def _fingerprint_conflicted_contexts(self, alias: str) -> set[str]:
        wit = self._fingerprint_witnesses(alias)
        return {ctx for ctx, rows in wit.items() if len(set(rows)) > 1}

    def _fingerprint_context_signature(
        self, alias: str, ctx: str
    ) -> tuple[str, str] | None:
        """Unique (action, outcome) for a non-conflicted context, else None."""
        rows = self._fingerprint_witnesses(alias).get(ctx) or []
        uniq = set(rows)
        if len(uniq) != 1:
            return None
        return next(iter(uniq))

    def _fingerprint_pair_ok(self, a: str, b: str) -> bool:
        """≥2 non-conflicted shared contexts with identical (action, outcome)."""
        if a.lower() == b.lower():
            return True
        wa = self._fingerprint_witnesses(a)
        wb = self._fingerprint_witnesses(b)
        ca = self._fingerprint_conflicted_contexts(a)
        cb = self._fingerprint_conflicted_contexts(b)
        shared = 0
        for ctx in set(wa) & set(wb):
            if ctx in ca or ctx in cb:
                continue
            sa = self._fingerprint_context_signature(a, ctx)
            sb = self._fingerprint_context_signature(b, ctx)
            if sa is None or sb is None or sa != sb:
                continue
            shared += 1
            if shared >= 2:
                return True
        return False

    def _fingerprint_clique(self, alias: str) -> set[str] | None:
        """Pairwise-qualified clique containing alias, or None if ambiguous.

        Peers that pair with alias must also pair with each other. Overlapping
        non-clique structure → None (HOLD).
        """
        u = alias.lower()
        aliases = {
            str(r.tags.get("alias") or "").lower()
            for r in self._fingerprint_rows()
            if isinstance(r.tags.get("alias"), str) and r.tags.get("alias")
        }
        aliases.add(u)
        peers = {v for v in aliases if v != u and self._fingerprint_pair_ok(u, v)}
        clique = {u} | peers
        for a in clique:
            for b in clique:
                if a >= b:
                    continue
                if not self._fingerprint_pair_ok(a, b):
                    return None
        return clique

    def _fingerprint_dest_class_key(self, dest: str) -> frozenset[str] | None:
        cl = self._fingerprint_clique(dest)
        if cl is None:
            return None
        return frozenset(cl)

    def _skel_out_edges(self, bind: str) -> list[FactRecord]:
        bl = bind.lower()
        motors = self._act_names()
        out: list[FactRecord] = []
        for rec in self.store.records():
            if str(rec.tags.get("source") or "") != "experience_skel":
                continue
            if isinstance(rec.tags.get("ctx"), str) and rec.tags.get("ctx"):
                continue
            if str(rec.tags.get("bind") or "").lower() != bl:
                continue
            did = rec.tags.get("did")
            if not isinstance(did, str) or not did:
                continue
            if did.lower() in motors:
                continue
            out.append(rec)
        return out

    def _project_from_alias(self, alias: str) -> FactRecord | None:
        """Aggregate clique support → unique dest class → unique raw edge from alias."""
        clique = self._fingerprint_clique(alias)
        if clique is None:
            return None
        # dest_class_key → aggregated (support, -contradiction)
        class_scores: dict[frozenset[str], tuple[int, int]] = {}
        class_members: dict[frozenset[str], set[str]] = {}
        for peer in clique:
            for rec in self._skel_out_edges(peer):
                did = str(rec.tags.get("did")).lower()
                dkey = self._fingerprint_dest_class_key(did)
                if dkey is None:
                    return None
                w, neg_c = self._evidence_score(rec)
                prev = class_scores.get(dkey)
                if prev is None:
                    class_scores[dkey] = (w, neg_c)
                else:
                    class_scores[dkey] = (prev[0] + w, prev[1] + neg_c)
                class_members.setdefault(dkey, set()).add(did)
        if not class_scores:
            return None
        best = max(class_scores.values())
        winners = [k for k, sc in class_scores.items() if sc == best]
        if len(winners) != 1:
            return None
        dest_class = class_members[winners[0]]
        raw = [
            rec
            for rec in self._skel_out_edges(alias)
            if str(rec.tags.get("did")).lower() in dest_class
        ]
        if len(raw) != 1:
            return None
        return raw[0]

    def _fingerprint_project_eligible(
        self, eligible: list, frontier: set[str] | None
    ) -> list:
        """Return projected winners, or [] for HOLD. Never invites raw fallback."""
        if not eligible:
            return []
        if frontier is not None:
            if len(frontier) != 1:
                return []
            u = next(iter(frontier))
            rec = self._project_from_alias(u)
            if rec is None:
                return []
            fid = str(getattr(rec, "fact_id", "") or "")
            elig_ids = {str(getattr(r, "fact_id", "") or "") for r in eligible}
            if fid and fid not in elig_ids:
                return []
            return [rec]

        binds = {
            str(r.tags.get("bind")).lower()
            for r in eligible
            if isinstance(r.tags.get("bind"), str) and r.tags.get("bind")
        }
        if not binds:
            return []
        projected: list = []
        for u in sorted(binds):
            rec = self._project_from_alias(u)
            if rec is None:
                # Ambiguous clique / non-unique dest → HOLD (do not skip to another bind).
                return []
            fid = str(getattr(rec, "fact_id", "") or "")
            elig_ids = {str(getattr(r, "fact_id", "") or "") for r in eligible}
            if fid and fid not in elig_ids:
                return []
            projected.append(rec)
        if not projected:
            return []
        return self._evidence_choose(projected)

    def _acquire_motors(self) -> set[str]:
        """Motors that may author contextual continuations (not HOLD/IDLE/WAIT)."""
        skip = {"hold", "idle", "wait"}
        return {m for m in self._act_names() if m not in skip}

    def _find_experience_ctx(
        self, bind: str, did: str, ctx: str
    ) -> FactRecord | None:
        bl, dl = bind.lower(), did.lower()
        for rec in self.store.records():
            if str(rec.tags.get("source") or "") != "experience_ctx":
                continue
            if str(rec.tags.get("bind") or "").lower() != bl:
                continue
            if str(rec.tags.get("did") or "").lower() != dl:
                continue
            if str(rec.tags.get("ctx") or "") != ctx:
                continue
            return rec
        return None

    def _apply_acquire_ctx(self, *, success: bool | None, action_name: str | None) -> None:
        """Author/revise contextual continuation from one-shot lived compose state.

        Consumes lived context at most once. Harness must not supply bind/κ.
        """
        if not self.use_acquire_ctx or not self.store.enabled:
            return
        pending = self._lived_pending
        bind = self._lived_bind
        kappa = self._lived_kappa
        if not pending or not isinstance(bind, str) or not bind:
            return
        if not isinstance(kappa, str) or not kappa:
            return
        if success is None or not isinstance(action_name, str) or not action_name:
            return
        motor = action_name.lower()
        if motor not in self._acquire_motors():
            return
        existing = self._find_experience_ctx(bind, motor, kappa)
        if success is True:
            if existing is None:
                tags: dict[str, Any] = {
                    "bind": bind,
                    "did": motor,
                    "ctx": kappa,
                    "source": "experience_ctx",
                    "here": "chb",
                    "w0": bind,
                    "hyp": "supported",
                    "trials": 1,
                    "wins": 1,
                    "losses": 0,
                    "support": 1,
                    "contradiction": 0,
                }
                n = len(self.store.records())
                fid = f"acq_{n:04d}_{bind}_{motor}"
                rec = FactRecord(
                    fact_id=fid,
                    what=encode_tags(tags),
                    when=int(self.t),
                    drive_scores={},
                    tags=tags,
                )
                self.store.write(rec)
            else:
                self._mark_hyp(existing, success=True)
            return
        # Failure: revise existing matching hyp only — never manufacture negatives.
        if existing is not None:
            self._mark_hyp(existing, success=False)

    def observe_symbol(self, token: str) -> dict[str, Any]:
        """Observed-transition channel: temporal symbols → durable skeleton adjacency.

        Apparatus emits a symbol sequence. Organism keeps one transient prev and
        authors prev→token as source=experience_skel (no ctx; did must not be motor).
        Separate sensory channel from motor-outcome teaching. Cleared by reset_rho.
        """
        out: dict[str, Any] = {
            "ok": False,
            "wrote": False,
            "updated": False,
            "prev": self._skel_prev,
            "token": None,
            "why": "",
        }
        if not self.use_acquire_skel or not self.store.enabled:
            out["why"] = "skel_off"
            return out
        if not isinstance(token, str) or not token.strip():
            out["why"] = "empty_token"
            return out
        tok = token.strip().lower()
        out["token"] = tok
        motors = self._acquire_motors() | {"hold", "idle", "wait"}
        prev = self._skel_prev
        if prev is None:
            self._skel_prev = tok
            out["ok"] = True
            out["why"] = "seed_prev"
            return out
        # Adjacency write only when both ends are non-motor relational symbols.
        if prev in motors or tok in motors:
            self._skel_prev = tok
            out["ok"] = True
            out["why"] = "skip_motor_symbol"
            return out
        existing = self._find_experience_skel(prev, tok)
        if existing is None:
            tags: dict[str, Any] = {
                "bind": prev,
                "did": tok,
                "source": "experience_skel",
                "here": "chb",
                "w0": prev,
                "hyp": "supported",
                "trials": 1,
                "wins": 1,
                "losses": 0,
                "support": 1,
                "contradiction": 0,
            }
            if "ctx" in tags:
                raise ValueError("experience_skel refuses ctx")
            n = len(self.store.records())
            fid = f"skel_{n:04d}_{prev}_{tok}"
            rec = FactRecord(
                fact_id=fid,
                what=encode_tags(tags),
                when=int(self.t),
                drive_scores={},
                tags=tags,
            )
            self.store.write(rec)
            out["wrote"] = True
        else:
            self._mark_hyp(existing, success=True)
            out["updated"] = True
        self._skel_prev = tok
        out["ok"] = True
        out["why"] = "authored" if out["wrote"] else "bumped"
        return out

    def _normalize_visible(self, visible: Any) -> list[str]:
        """Strip/lower/dedupe/sort; drop motors and empties. Never reads focus."""
        motors = self._acquire_motors() | {"hold", "idle", "wait"}
        out: set[str] = set()
        if not isinstance(visible, (list, tuple, set, frozenset)):
            return []
        for x in visible:
            if not isinstance(x, str):
                continue
            tok = x.strip().lower()
            if not tok or tok in motors:
                continue
            out.add(tok)
        return sorted(out)

    def observe_event(self, event: dict[str, Any] | None) -> dict[str, Any]:
        """Ambiguous multi-symbol events → candidate experience_skel cloud.

        Reads event['visible'] only. MUST NOT read event['focus'].
        All-pairs prev_visible × curr_visible (incl. self-pairs). No prune.
        """
        result: dict[str, Any] = {
            "ok": False,
            "wrote": 0,
            "updated": 0,
            "visible": [],
            "why": "",
        }
        if not self.use_acquire_relate or not self.store.enabled:
            result["why"] = "relate_off"
            return result
        if not isinstance(event, dict):
            result["why"] = "bad_event"
            return result
        # Binding: visible only — never touch focus.
        curr = self._normalize_visible(event.get("visible"))
        result["visible"] = list(curr)
        if not curr:
            result["ok"] = True
            result["why"] = "empty_visible"
            # Still advance? Plan: empty visible no write. Do not update prev with empty
            # (keeps prior episode state until end_event_episode / non-empty event).
            return result
        prev = self._rel_prev_visible
        if prev is None:
            self._rel_prev_visible = list(curr)
            result["ok"] = True
            result["why"] = "seed_prev_visible"
            return result
        wrote = 0
        updated = 0
        for a in prev:
            for b in curr:
                existing = self._find_experience_skel(a, b)
                if existing is None:
                    tags: dict[str, Any] = {
                        "bind": a,
                        "did": b,
                        "source": "experience_skel",
                        "here": "chb",
                        "w0": a,
                        "hyp": "supported",
                        "trials": 1,
                        "wins": 1,
                        "losses": 0,
                        "support": 1,
                        "contradiction": 0,
                    }
                    n = len(self.store.records())
                    fid = f"skel_{n:04d}_{a}_{b}"
                    rec = FactRecord(
                        fact_id=fid,
                        what=encode_tags(tags),
                        when=int(self.t),
                        drive_scores={},
                        tags=tags,
                    )
                    self.store.write(rec)
                    wrote += 1
                else:
                    self._mark_hyp(existing, success=True)
                    updated += 1
        self._rel_prev_visible = list(curr)
        result["ok"] = True
        result["wrote"] = wrote
        result["updated"] = updated
        result["why"] = "authored" if wrote or updated else "noop"
        return result

    def observe_alias_probe(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Behavioral fingerprint channel. Exact key set only; never writes roles/maps.

        Duplicate exact (alias, probe_context, action, observed_outcome) bumps support;
        that context still counts once toward pairing. Different action/outcome in the
        same (alias, probe_context) retains both and marks the context conflicted.
        """
        required = {"alias", "probe_context", "action", "observed_outcome"}
        out: dict[str, Any] = {
            "ok": False,
            "wrote": False,
            "updated": False,
            "why": "",
        }
        if not self.use_alias_fingerprint or not self.store.enabled:
            out["why"] = "fingerprint_off"
            return out
        if not isinstance(info, dict):
            out["why"] = "bad_info"
            return out
        if set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        alias_raw = info["alias"]
        ctx_raw = info["probe_context"]
        act_raw = info["action"]
        out_raw = info["observed_outcome"]
        if not all(isinstance(x, str) and x.strip() for x in (alias_raw, ctx_raw, act_raw, out_raw)):
            out["why"] = "empty_field"
            return out
        alias = alias_raw.strip().lower()
        ctx = ctx_raw.strip()
        action = act_raw.strip().lower()
        outcome = out_raw.strip().lower()
        # Exact duplicate → bump support.
        existing = None
        for rec in self._fingerprint_rows():
            if str(rec.tags.get("alias") or "").lower() != alias:
                continue
            if str(rec.tags.get("probe_context") or "") != ctx:
                continue
            if str(rec.tags.get("action") or "").lower() != action:
                continue
            if str(rec.tags.get("observed_outcome") or "").lower() != outcome:
                continue
            existing = rec
            break
        if existing is not None:
            self._mark_hyp(existing, success=True)
            out["ok"] = True
            out["updated"] = True
            out["why"] = "bumped"
            return out
        tags: dict[str, Any] = {
            "alias": alias,
            "probe_context": ctx,
            "action": action,
            "observed_outcome": outcome,
            "source": "experience_fingerprint",
            "hyp": "supported",
            "trials": 1,
            "wins": 1,
            "losses": 0,
            "support": 1,
            "contradiction": 0,
        }
        # Refuse role / map contamination in authored tags.
        for banned in ("role", "equivalence_class", "canonical_token", "role_alias_map", "bind", "did", "ctx"):
            if banned in tags:
                raise ValueError(f"experience_fingerprint refuses {banned}")
        n = len(self.store.records())
        fid = f"fp_{n:04d}_{alias}_{ctx}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def observe_continuity_mark(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Causal mark channel. Exact key set only; writes experience_continuity rows.

        Never stores same_as, canonical tokens, bind/did identity, or a persistent
        permission. Uniqueness is recomputed at use time from raw rows.
        """
        required = {"token", "mark_id", "phase", "operation", "observed_state"}
        phase_op = {"pre_gap": "apply", "post_gap": "read"}
        out: dict[str, Any] = {
            "ok": False,
            "wrote": False,
            "updated": False,
            "why": "",
        }
        if not self.use_continuity_mark or not self.store.enabled:
            out["why"] = "continuity_off"
            return out
        if not isinstance(info, dict):
            out["why"] = "bad_info"
            return out
        if set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        token_raw = info["token"]
        mark_raw = info["mark_id"]
        phase_raw = info["phase"]
        op_raw = info["operation"]
        state_raw = info["observed_state"]
        if not all(
            isinstance(x, str) and x.strip()
            for x in (token_raw, mark_raw, phase_raw, op_raw, state_raw)
        ):
            out["why"] = "empty_field"
            return out
        token = token_raw.strip().lower()
        mark_id = mark_raw.strip()
        phase = phase_raw.strip()
        operation = op_raw.strip()
        observed_state = state_raw.strip()
        if phase not in phase_op or phase_op[phase] != operation:
            out["why"] = "phase_op_reject"
            return out
        tags: dict[str, Any] = {
            "token": token,
            "mark_id": mark_id,
            "phase": phase,
            "operation": operation,
            "observed_state": observed_state,
            "source": "experience_continuity",
            "hyp": "supported",
            "trials": 1,
            "wins": 1,
            "losses": 0,
            "support": 1,
            "contradiction": 0,
        }
        for banned in (
            "object_id",
            "same_as",
            "continuity_class",
            "canonical_token",
            "canonical_id",
            "latent_map",
            "role",
            "route_position_identity",
            "bind",
            "did",
            "ctx",
        ):
            if banned in tags:
                raise ValueError(f"experience_continuity refuses {banned}")
        n = len(self.store.records())
        fid = f"cont_{n:04d}_{phase}_{token}_{mark_id}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def _continuity_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_continuity"
        ]

    def _continuity_recompute(self) -> list[tuple[str, str]]:
        """Derived (P, Q) permissions from raw rows. Never stored.

        For each mark_id: conflicting states on the same token refuse that mark.
        Any row that violates the phase–operation lock refuses that mark.
        For each (mark_id, observed_state): exactly one apply row and exactly one
        read row produce one (P, Q). A Q with two distinct P sources is dropped.
        """
        phase_op = {"pre_gap": "apply", "post_gap": "read"}
        rows = self._continuity_rows()
        by_mark: dict[str, list[FactRecord]] = {}
        for rec in rows:
            mid = str(rec.tags.get("mark_id") or "")
            if not mid:
                continue
            by_mark.setdefault(mid, []).append(rec)
        pairs: list[tuple[str, str]] = []
        for _mid, recs in by_mark.items():
            apply_states: dict[str, set[str]] = {}
            read_states: dict[str, set[str]] = {}
            malformed = False
            for rec in recs:
                tok = str(rec.tags.get("token") or "").lower()
                state = str(rec.tags.get("observed_state") or "")
                phase = str(rec.tags.get("phase") or "")
                op = str(rec.tags.get("operation") or "")
                if not tok or not state or phase not in phase_op or phase_op[phase] != op:
                    malformed = True
                    break
                if op == "apply":
                    apply_states.setdefault(tok, set()).add(state)
                else:
                    read_states.setdefault(tok, set()).add(state)
            if malformed:
                continue
            if any(len(s) > 1 for s in apply_states.values()):
                continue
            if any(len(s) > 1 for s in read_states.values()):
                continue
            matching: dict[str, tuple[list[str], list[str]]] = {}
            for rec in recs:
                tok = str(rec.tags.get("token") or "").lower()
                state = str(rec.tags.get("observed_state") or "")
                op = str(rec.tags.get("operation") or "")
                slot = matching.setdefault(state, ([], []))
                if op == "apply":
                    slot[0].append(tok)
                elif op == "read":
                    slot[1].append(tok)
            for _state, (applies, reads) in matching.items():
                if len(applies) == 1 and len(reads) == 1:
                    pairs.append((applies[0], reads[0]))
        by_q: dict[str, set[str]] = {}
        for p, q in pairs:
            by_q.setdefault(q, set()).add(p)
        return [(next(iter(ps)), q) for q, ps in by_q.items() if len(ps) == 1]

    def _continuity_source_for(self, q: str) -> str | None:
        ql = q.lower()
        ps = {p for p, qq in self._continuity_recompute() if qq == ql}
        if len(ps) != 1:
            return None
        return next(iter(ps))

    def _continuity_project_token(self, q: str) -> FactRecord | None:
        """One-hop: unique P for Q, then the unique existing skel out-edge from P."""
        p = self._continuity_source_for(q)
        if p is None:
            return None
        edges = self._skel_out_edges(p)
        dests = {
            str(rec.tags.get("did") or "").lower()
            for rec in edges
            if isinstance(rec.tags.get("did"), str) and rec.tags.get("did")
        }
        dests.discard("")
        if len(dests) != 1:
            return None
        dest = next(iter(dests))
        raw = [rec for rec in edges if str(rec.tags.get("did") or "").lower() == dest]
        if len(raw) != 1:
            return None
        return raw[0]

    def _continuity_first_hop(self, obs) -> list | None:
        """None = continuity not in play; [] = HOLD; [rec] = project existing edge.

        Continuity is in play only when rows exist and the current stream token
        is a unique Q. Fingerprint rows are not read.
        """
        if not self.use_continuity_mark or not self._continuity_rows():
            return None
        stream = self._current_stream(obs)
        permitted = [t for t in stream if self._continuity_source_for(t) is not None]
        if len(stream) != 1:
            # A Q in a multi-token cue must not fall through to raw compose.
            return [] if permitted else None
        if not permitted:
            return None
        t = next(iter(stream))
        rec = self._continuity_project_token(t)
        if rec is None:
            return []
        return [rec]

    def observe_symbol_ground(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Evidence-weighted symbol↔world co-occurrence. Exact keys only.

        Writes experience_grounding rows. Bindings are recomputed at use time —
        never stored as meanings, POS tags, or synonym maps.
        """
        required = {"symbol", "paired", "trial_id", "result"}
        allowed_results = {"success", "failure", "correction"}
        out: dict[str, Any] = {
            "ok": False,
            "wrote": False,
            "updated": False,
            "why": "",
        }
        if not self.use_symbol_ground or not self.store.enabled:
            out["why"] = "grounding_off"
            return out
        if not isinstance(info, dict):
            out["why"] = "bad_info"
            return out
        if set(info.keys()) != required:
            # Reliability path: optional provenance for verification-eligible rows
            if (
                self.use_source_reliability
                and set(info.keys()) == required | {"provenance"}
            ):
                pass
            else:
                out["why"] = "exact_key_reject"
                return out
        symbol_raw = info["symbol"]
        paired_raw = info["paired"]
        trial_raw = info["trial_id"]
        result_raw = info["result"]
        if not all(
            isinstance(x, str) and x.strip()
            for x in (symbol_raw, paired_raw, trial_raw, result_raw)
        ):
            out["why"] = "empty_field"
            return out
        symbol = symbol_raw.strip().lower()
        paired = paired_raw.strip().lower()
        trial_id = trial_raw.strip()
        result = result_raw.strip().lower()
        if result not in allowed_results:
            out["why"] = "result_reject"
            return out
        provenance = None
        if "provenance" in info:
            provenance = str(info["provenance"]).strip().lower()
            if provenance not in {"direct", "experiment", "state_read", "testimony_derived"}:
                out["why"] = "provenance_reject"
                return out
        tags: dict[str, Any] = {
            "symbol": symbol,
            "paired": paired,
            "trial_id": trial_id,
            "result": result,
            "source": "experience_grounding",
            "hyp": "supported",
            "trials": 1,
            "wins": 1,
            "losses": 0,
            "support": 1,
            "contradiction": 0,
        }
        if provenance is not None:
            tags["provenance"] = provenance
        for banned in (
            "noun",
            "verb",
            "color",
            "pos",
            "meaning",
            "same_as",
            "bind",
            "did",
            "ctx",
            "role",
            "english",
            "gloss",
        ):
            if banned in tags:
                raise ValueError(f"experience_grounding refuses {banned}")
        n = len(self.store.records())
        fid = f"ground_{n:04d}_{symbol}_{paired}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        if self.use_source_reliability and provenance in {
            "direct",
            "experiment",
            "state_read",
        }:
            self._reliability_reconcile_from_ground(rec)
        return out

    def _grounding_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_grounding"
        ]

    def _grounding_support(self) -> dict[str, dict[str, int]]:
        """symbol → paired → net support. success/correction +1; failure −1.

        Rows marked provenance=testimony_derived never count as world evidence
        (anti-circular: teacher claim must not become inquire/ground support).
        """
        scores: dict[str, dict[str, int]] = {}
        for rec in self._grounding_rows():
            if str(rec.tags.get("provenance") or "") == "testimony_derived":
                continue
            sym = str(rec.tags.get("symbol") or "").lower()
            paired = str(rec.tags.get("paired") or "").lower()
            result = str(rec.tags.get("result") or "").lower()
            if not sym or not paired:
                continue
            if result == "failure":
                delta = -1
            else:
                # success and correction both reinforce the paired token
                delta = 1
            bucket = scores.setdefault(sym, {})
            bucket[paired] = int(bucket.get(paired, 0)) + delta
        return scores

    def _grounding_binding(self, symbol: str, *, min_support: int = 2) -> str | None:
        """Unique evidence-weighted paired token for symbol, else None (HOLD)."""
        sym = symbol.lower()
        scores = self._grounding_support().get(sym) or {}
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_tok, best = ranked[0]
        if best < min_support:
            return None
        if len(ranked) > 1 and ranked[1][1] == best:
            return None
        return best_tok

    def select_grounded(
        self,
        utterance: Sequence[str],
        choices: Sequence[str],
        *,
        min_support: int = 2,
        expression: bool = False,
    ) -> dict[str, Any]:
        """Choose among world-offered options from recomputed grounding evidence.

        expression=True: utterance holds a referent cue; choices are word candidates.
        """
        out: dict[str, Any] = {
            "ok": True,
            "selected": None,
            "why": "hold",
        }
        if not self.use_symbol_ground:
            out["why"] = "grounding_off"
            return out
        if not self._grounding_rows():
            out["why"] = "no_rows"
            return out
        utt = [str(x).strip().lower() for x in utterance if str(x).strip()]
        opts = [str(x).strip().lower() for x in choices if str(x).strip()]
        if not utt or not opts:
            out["why"] = "empty_probe"
            return out
        if expression:
            # Invert: choice words whose direct support for the referent cue is unique max.
            cue = utt[0]
            scored: list[tuple[int, str]] = []
            for w in opts:
                support = int((self._grounding_support().get(w) or {}).get(cue, 0))
                if support >= min_support:
                    scored.append((support, w))
            if not scored:
                out["why"] = "expression_none"
                return out
            scored.sort(key=lambda kv: (-kv[0], kv[1]))
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                out["why"] = "expression_tie"
                return out
            out["selected"] = scored[0][1]
            out["why"] = "expression"
            return out

        # Ordered multi-symbol phrase: treat "a b" as its own opaque symbol when
        # evidence exists (word order without POS tags).
        if len(utt) > 1:
            phrase = " ".join(utt)
            phrase_scores = self._grounding_support().get(phrase) or {}
            if phrase_scores:
                ranked = sorted(
                    ((ch, phrase_scores.get(ch, 0)) for ch in opts),
                    key=lambda kv: (-kv[1], kv[0]),
                )
                if ranked and ranked[0][1] >= min_support:
                    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                        out["selected"] = ranked[0][0]
                        out["why"] = "ordered_phrase"
                        return out
                    out["why"] = "phrase_tie"
                    return out

        # Score each choice by how many utterance tokens uniquely support a
        # fragment inside the choice (compound choices use '+').
        scored_c: list[tuple[int, str]] = []
        for ch in opts:
            parts = set(ch.split("+")) | {ch}
            hit = 0
            blocked = False
            for w in utt:
                b = self._grounding_binding(w, min_support=min_support)
                if b is None:
                    blocked = True
                    break
                b_parts = set(b.split("+")) | {b}
                if parts & b_parts or b in parts or ch in b_parts:
                    hit += 1
                else:
                    # Direct support row for (w, ch) even without unique global binding
                    support = (self._grounding_support().get(w) or {}).get(ch, 0)
                    if support >= min_support:
                        hit += 1
                    else:
                        blocked = True
                        break
            if not blocked and hit == len(utt):
                scored_c.append((hit, ch))
        if not scored_c:
            # Fallback: among choices, unique max direct support summed over words
            totals: dict[str, int] = {ch: 0 for ch in opts}
            for w in utt:
                bag = self._grounding_support().get(w) or {}
                for ch in opts:
                    parts = set(ch.split("+")) | {ch}
                    for tok, sc in bag.items():
                        if tok in parts or tok == ch:
                            totals[ch] = int(totals[ch]) + max(sc, 0)
            ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
            if ranked and ranked[0][1] >= min_support * max(len(utt), 1):
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    out["selected"] = ranked[0][0]
                    out["why"] = "support_sum"
                    return out
            out["why"] = "insufficient_or_tie"
            return out
        scored_c.sort(key=lambda kv: (-kv[0], kv[1]))
        if len(scored_c) > 1 and scored_c[0][0] == scored_c[1][0]:
            out["why"] = "choice_tie"
            return out
        out["selected"] = scored_c[0][1]
        out["why"] = "grounded"
        return out

    def _grounding_first_hop(self, obs) -> list | None:
        """None = not in play; [] = HOLD; [rec] = project bound referent as did."""
        if not self.use_symbol_ground or not self._grounding_rows():
            return None
        stream = list(self._current_stream(obs))
        if len(stream) != 1:
            # Multi-token cues use select_grounded via the world harness.
            return None
        bound = self._grounding_binding(stream[0])
        if bound is None:
            return []
        tags = {
            "bind": stream[0],
            "did": bound,
            "source": "experience_skel",
            "hyp": "supported",
            "support": 1,
        }
        rec = FactRecord(
            fact_id=f"ground_proj_{stream[0]}_{bound}",
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        return [rec]

    def observe_sequence_step(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Author one next-operation demonstration into experience_sequence.

        Exact keys only. Factorized context_atoms — no scene IDs / grammar slots.
        """
        required = {
            "context_atoms",
            "input_symbols",
            "prefix",
            "next_operation",
            "next_symbol",
            "result",
        }
        allowed_ops = {"emit", "stop"}
        allowed_results = {"success", "failure", "correction"}
        banned_keys = {
            "scene_id",
            "scene",
            "noun_slot",
            "verb_slot",
            "subject",
            "object",
            "role",
            "grammar",
            "expect",
            "answer",
            "english",
            "gloss",
        }
        out: dict[str, Any] = {
            "ok": False,
            "wrote": False,
            "why": "",
        }
        if not self.use_symbol_sequence or not self.store.enabled:
            out["why"] = "sequence_off"
            return out
        if not isinstance(info, dict):
            out["why"] = "bad_info"
            return out
        if set(info.keys()) & banned_keys:
            out["why"] = "banned_field"
            return out
        if set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        ctx_raw = info["context_atoms"]
        inp_raw = info["input_symbols"]
        pref_raw = info["prefix"]
        op_raw = info["next_operation"]
        sym_raw = info["next_symbol"]
        result_raw = info["result"]
        if not isinstance(ctx_raw, (list, tuple)) or not isinstance(inp_raw, (list, tuple)):
            out["why"] = "bad_list"
            return out
        if not isinstance(pref_raw, (list, tuple)):
            out["why"] = "bad_prefix"
            return out
        if not isinstance(op_raw, str) or not isinstance(result_raw, str):
            out["why"] = "bad_scalar"
            return out
        if not isinstance(sym_raw, str):
            out["why"] = "bad_symbol"
            return out
        op = op_raw.strip().lower()
        result = result_raw.strip().lower()
        if op not in allowed_ops or result not in allowed_results:
            out["why"] = "op_or_result_reject"
            return out
        # Forbid scene-handle smuggling inside atoms
        atoms = [str(x).strip().lower() for x in ctx_raw if str(x).strip()]
        inputs = [str(x).strip().lower() for x in inp_raw if str(x).strip()]
        prefix = [str(x).strip().lower() for x in pref_raw if str(x).strip()]
        if any(a.startswith("scene_") or a.startswith("hash(") for a in atoms):
            out["why"] = "scene_id_refuse"
            return out
        if not atoms or not inputs:
            out["why"] = "empty_context_or_input"
            return out
        symbol = sym_raw.strip().lower()
        if op == "emit":
            if not symbol:
                out["why"] = "emit_needs_symbol"
                return out
        else:
            symbol = ""
        if any("|" in x for x in atoms + inputs + prefix + ([symbol] if symbol else [])):
            out["why"] = "pipe_in_token"
            return out
        tags: dict[str, Any] = {
            "context_atoms": "|".join(sorted(atoms)),
            "input_symbols": "|".join(inputs),
            "prefix": "|".join(prefix),
            "next_operation": op,
            "next_symbol": symbol,
            "result": result,
            "source": "experience_sequence",
            "hyp": "supported",
            "support": 1,
        }
        for banned in banned_keys:
            if banned in tags:
                raise ValueError(f"experience_sequence refuses {banned}")
        n = len(self.store.records())
        fid = f"seq_{n:04d}_{op}_{symbol or 'stop'}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def _sequence_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_sequence"
        ]

    @staticmethod
    def _seq_split(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(x).strip().lower() for x in raw if str(x).strip()]
        text = str(raw).strip()
        if not text or text == "[]":
            return []
        # Tolerate legacy stringified lists from accidental list writes
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            parts = []
            for chunk in inner.split(","):
                parts.append(chunk.strip().strip("'").strip('"').lower())
            return [p for p in parts if p]
        return [p for p in text.split("|") if p]

    def _sequence_atom_to_symbol(self, *, min_support: int = 2) -> dict[str, str]:
        """paired atom → unique symbol via inverted grounding support."""
        scores = self._grounding_support()
        atom_votes: dict[str, dict[str, int]] = {}
        for sym, bag in scores.items():
            for atom, sc in bag.items():
                if sc <= 0:
                    continue
                atom_votes.setdefault(atom, {})[sym] = int(
                    atom_votes.setdefault(atom, {}).get(sym, 0)
                ) + int(sc)
        out: dict[str, str] = {}
        for atom, bag in atom_votes.items():
            ranked = sorted(bag.items(), key=lambda kv: (-kv[1], kv[0]))
            if not ranked or ranked[0][1] < min_support:
                continue
            if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
                continue
            out[atom] = ranked[0][0]
        return out

    def _sequence_symbol_to_atom(self, *, min_support: int = 2) -> dict[str, str]:
        out: dict[str, str] = {}
        for sym, bag in self._grounding_support().items():
            ranked = sorted(bag.items(), key=lambda kv: (-kv[1], kv[0]))
            if not ranked or ranked[0][1] < min_support:
                continue
            if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
                continue
            out[sym] = ranked[0][0]
        return out

    def _sequence_next_votes(
        self,
        context_atoms: Sequence[str],
        input_symbols: Sequence[str],
        prefix: Sequence[str],
    ) -> dict[tuple[str, str], int]:
        ctx = "|".join(sorted(str(x).strip().lower() for x in context_atoms if str(x).strip()))
        inp = "|".join(str(x).strip().lower() for x in input_symbols if str(x).strip())
        pref = "|".join(str(x).strip().lower() for x in prefix if str(x).strip())
        votes: dict[tuple[str, str], int] = {}
        for rec in self._sequence_rows():
            rctx = str(rec.tags.get("context_atoms") or "")
            rinp = str(rec.tags.get("input_symbols") or "")
            rpref = str(rec.tags.get("prefix") or "")
            # normalize list-shaped leftovers
            if rctx.startswith("["):
                rctx = "|".join(self._seq_split(rctx))
            if rinp.startswith("["):
                rinp = "|".join(self._seq_split(rinp))
            if rpref.startswith("["):
                rpref = "|".join(self._seq_split(rpref))
            if rctx != ctx or rinp != inp or rpref != pref:
                continue
            op = str(rec.tags.get("next_operation") or "").lower()
            sym = str(rec.tags.get("next_symbol") or "").lower()
            result = str(rec.tags.get("result") or "success").lower()
            if result == "failure":
                delta = -1
            else:
                delta = 1
            key = (op, sym)
            votes[key] = int(votes.get(key, 0)) + delta
        return votes

    def _sequence_unique_next(
        self,
        context_atoms: Sequence[str],
        input_symbols: Sequence[str],
        prefix: Sequence[str],
        *,
        min_support: int,
    ) -> tuple[str, str] | None:
        votes = self._sequence_next_votes(context_atoms, input_symbols, prefix)
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        if not ranked or ranked[0][1] < min_support:
            return None
        if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
            return None
        return ranked[0][0]

    def _sequence_complete_demos(
        self, input_symbols: Sequence[str], *, min_support: int
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Reconstruct complete (context, sequence) demos with enough STOP support."""
        inp = [str(x).strip().lower() for x in input_symbols if str(x).strip()]
        # Collect contexts that have stop votes
        contexts: set[tuple[str, ...]] = set()
        for rec in self._sequence_rows():
            if self._seq_split(rec.tags.get("input_symbols")) != inp:
                continue
            contexts.add(tuple(self._seq_split(rec.tags.get("context_atoms"))))
        demos: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for ctx in contexts:
            seq: list[str] = []
            ok = True
            for _ in range(64):
                nxt = self._sequence_unique_next(ctx, inp, seq, min_support=min_support)
                if nxt is None:
                    ok = False
                    break
                op, sym = nxt
                if op == "stop":
                    break
                if op != "emit" or not sym:
                    ok = False
                    break
                seq.append(sym)
            else:
                ok = False
            if ok and seq:
                key = (ctx, tuple(seq))
                if key not in seen:
                    seen.add(key)
                    demos.append(key)
        return demos

    def _sequence_compose(
        self,
        context_atoms: Sequence[str],
        input_symbols: Sequence[str],
        *,
        min_support: int,
    ) -> list[str] | None:
        """Nearest same-size single-atom context substitution; unique template only.

        Multi-atom remaps are refused (sorted zip would be an arbitrary grammar).
        Requires overlap == |C|-1 (exactly one differing atom).
        """
        C = tuple(sorted(str(x).strip().lower() for x in context_atoms if str(x).strip()))
        demos = self._sequence_complete_demos(input_symbols, min_support=min_support)
        if not demos:
            return None
        sym2atom = self._sequence_symbol_to_atom(min_support=min_support)
        atom2sym = self._sequence_atom_to_symbol(min_support=min_support)
        Cset = set(C)
        best_overlap = -1
        candidates: list[list[str]] = []
        for ctx, seq in demos:
            cset = set(ctx)
            if len(ctx) != len(C):
                continue
            overlap = len(cset & Cset)
            only_d = sorted(cset - Cset)
            only_q = sorted(Cset - cset)
            # Exactly one atom substitution (or identical — unused because exact path wins).
            if len(only_d) != len(only_q):
                continue
            if len(only_d) > 1:
                continue
            if overlap != len(C) - len(only_d):
                continue
            if only_d and overlap != len(C) - 1:
                continue
            mapping = {a: a for a in (cset & Cset)}
            for a, b in zip(only_d, only_q):
                mapping[a] = b
            out_seq: list[str] = []
            ok = True
            for tok in seq:
                atom = sym2atom.get(tok)
                if atom is None:
                    ok = False
                    break
                mapped = mapping.get(atom)
                if mapped is None:
                    ok = False
                    break
                word = atom2sym.get(mapped)
                if word is None:
                    ok = False
                    break
                out_seq.append(word)
            if not ok or not out_seq:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                candidates = [out_seq]
            elif overlap == best_overlap:
                if out_seq not in candidates:
                    candidates.append(out_seq)
        if best_overlap < 0 or len(candidates) != 1:
            return None
        return candidates[0]

    def emit_sequence(
        self,
        context_atoms: Sequence[str],
        input_symbols: Sequence[str],
        *,
        min_support: int = 2,
        cap: int = 64,
    ) -> dict[str, Any]:
        """Construct a full utterance or atomic HOLD. Records first_internal_fail."""
        out: dict[str, Any] = {
            "ok": True,
            "sequence": None,
            "why": "hold",
            "first_internal_fail": None,
        }
        self.first_internal_fail = None
        if not self.use_symbol_sequence:
            out["why"] = "sequence_off"
            return out
        if not self._sequence_rows():
            out["why"] = "no_rows"
            return out
        # Grounding gate: sequence alone is not enough once grounding channel exists.
        if not self._grounding_rows():
            out["why"] = "no_grounding"
            return out
        atoms = [str(x).strip().lower() for x in context_atoms if str(x).strip()]
        inputs = [str(x).strip().lower() for x in input_symbols if str(x).strip()]
        if not atoms or not inputs:
            out["why"] = "empty_probe"
            return out
        if any(a.startswith("scene_") for a in atoms):
            out["why"] = "scene_id_refuse"
            return out
        # Every context atom must be attested as a grounded paired token.
        paired_ok = set()
        for bag in self._grounding_support().values():
            for atom, sc in bag.items():
                if sc > 0:
                    paired_ok.add(atom)
        missing = [a for a in atoms if a not in paired_ok]
        if missing:
            out["why"] = "ungrounded_context"
            return out

        prefix: list[str] = []
        used_compose = False
        for step in range(int(cap)):
            nxt = self._sequence_unique_next(atoms, inputs, prefix, min_support=min_support)
            if nxt is None:
                if step == 0 and not prefix:
                    composed = self._sequence_compose(
                        atoms, inputs, min_support=min_support
                    )
                    if composed is None:
                        fail = {
                            "step": step,
                            "expected_op": "unique",
                            "actual": None,
                            "why": "no_unique_next",
                        }
                        self.first_internal_fail = fail
                        out["first_internal_fail"] = fail
                        out["why"] = "no_unique_next"
                        return out
                    # Validate composed sequence has evidenced STOP via remapped exact path
                    # by accepting composition atomically when unique.
                    used_compose = True
                    out["sequence"] = list(composed)
                    out["why"] = "compose"
                    return out
                fail = {
                    "step": step,
                    "prefix": list(prefix),
                    "expected_op": "unique",
                    "actual": None,
                    "why": "ambiguous_or_missing",
                }
                self.first_internal_fail = fail
                out["first_internal_fail"] = fail
                out["why"] = "atomic_hold"
                return out
            op, sym = nxt
            if op == "stop":
                out["sequence"] = list(prefix)
                out["why"] = "exact" if not used_compose else "compose"
                return out
            if op != "emit" or not sym:
                fail = {
                    "step": step,
                    "expected_op": "emit",
                    "actual": op,
                    "symbol": sym,
                }
                self.first_internal_fail = fail
                out["first_internal_fail"] = fail
                out["why"] = "bad_op"
                return out
            prefix.append(sym)
        fail = {
            "step": int(cap),
            "prefix": list(prefix),
            "why": "cap_without_stop",
        }
        self.first_internal_fail = fail
        out["first_internal_fail"] = fail
        out["why"] = "cap_hold"
        return out

    def observe_inquire_trace(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Author experience_inquire plan/trace rows only (not world consequences)."""
        required = {
            "context_atoms",
            "input_symbols",
            "probe_kind",
            "probe_atoms",
            "predicted_partition",
            "cost",
            "phase",
        }
        allowed_kinds = {"ask", "experiment"}
        allowed_phases = {"propose", "observed"}
        out: dict[str, Any] = {"ok": False, "wrote": False, "why": ""}
        if not self.use_inquire or not self.store.enabled:
            out["why"] = "inquire_off"
            return out
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        kind = str(info["probe_kind"]).strip().lower()
        phase = str(info["phase"]).strip().lower()
        if kind not in allowed_kinds or phase not in allowed_phases:
            out["why"] = "kind_or_phase_reject"
            return out
        ctx = [str(x).strip().lower() for x in info["context_atoms"] if str(x).strip()]
        inp = [str(x).strip().lower() for x in info["input_symbols"] if str(x).strip()]
        atoms = [str(x).strip().lower() for x in info["probe_atoms"] if str(x).strip()]
        part = str(info["predicted_partition"]).strip().lower()
        try:
            cost = int(info["cost"])
        except (TypeError, ValueError):
            out["why"] = "bad_cost"
            return out
        if not ctx or not inp or not atoms or not part or cost < 0:
            out["why"] = "empty_field"
            return out
        if any("|" in x for x in ctx + inp + atoms):
            out["why"] = "pipe_in_token"
            return out
        tags: dict[str, Any] = {
            "context_atoms": "|".join(sorted(ctx)),
            "input_symbols": "|".join(inp),
            "probe_kind": kind,
            "probe_atoms": "|".join(atoms),
            "predicted_partition": part,
            "cost": cost,
            "phase": phase,
            "source": "experience_inquire",
            "hyp": "supported",
            "support": 1,
        }
        n = len(self.store.records())
        fid = f"inquire_{n:04d}_{kind}_{phase}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def _inquire_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_inquire"
        ]

    def _inquire_cue(self, input_symbols: Sequence[str]) -> str | None:
        toks = [str(x).strip().lower() for x in input_symbols if str(x).strip()]
        if not toks:
            return None
        # Last contentful token is the cue (e.g. what/is/dax → dax; dax → dax)
        return toks[-1]

    def _inquire_hypotheses(self, cue: str, *, min_support: int) -> list[str]:
        scores = self._grounding_support().get(cue) or {}
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best = ranked[0][1]
        if best < min_support:
            return []
        return [tok for tok, sc in ranked if sc == best and sc >= min_support]

    def _inquire_factors(self, referent: str, *, min_support: int) -> set[str]:
        """Factors attested for a referent: grounding rows with symbol=referent."""
        out: set[str] = set()
        bag = self._grounding_support().get(referent) or {}
        for paired, sc in bag.items():
            if sc >= min_support:
                out.add(paired)
        return out

    def _inquire_predicted_outcome(self, hyp: str, factor: str, *, min_support: int) -> str:
        factors = self._inquire_factors(hyp, min_support=min_support)
        return "yes" if factor in factors else "no"

    def _inquire_probe_candidates(
        self, hyps: Sequence[str], *, min_support: int
    ) -> list[dict[str, Any]]:
        """Generate ask/experiment probes from factor disagreement + S-justified predictions."""
        if len(hyps) < 2:
            return []
        factor_sets = {h: self._inquire_factors(h, min_support=min_support) for h in hyps}
        all_factors: set[str] = set()
        for fs in factor_sets.values():
            all_factors |= fs
        # Disagreement: factor not shared by all hyps
        disagree = [
            f
            for f in sorted(all_factors)
            if any(f in factor_sets[h] for h in hyps)
            and any(f not in factor_sets[h] for h in hyps)
        ]
        cands: list[dict[str, Any]] = []
        for factor in disagree:
            # Justified prediction: every hyp has a definite yes/no from S factors
            partition: dict[str, list[str]] = {}
            justified = True
            for h in hyps:
                # Require positive factor evidence somewhere for this factor among hyps
                # Outcome is yes/no from membership; always justified if factor appears in union
                o = self._inquire_predicted_outcome(h, factor, min_support=min_support)
                partition.setdefault(o, []).append(h)
            if not justified:
                continue
            value = len(hyps) - max(len(v) for v in partition.values())
            if value <= 0:
                continue
            part_key = "|".join(
                f"{o}:{','.join(sorted(hs))}" for o, hs in sorted(partition.items())
            )
            cands.append(
                {
                    "kind": "ask",
                    "atoms": ["ask", factor],
                    "factor": factor,
                    "value": value,
                    "cost": int(self.inquire_cost_ask),
                    "partition": part_key,
                }
            )
            # Experiment variant: same partition, different cost (from S: if act_* linked to factor)
            act = f"do_{factor}"
            # Only offer experiment if S has grounding for the action token
            act_scores = self._grounding_support().get(act) or {}
            if any(sc >= min_support for sc in act_scores.values()) or any(
                int((self._grounding_support().get(h) or {}).get(act, 0)) >= min_support
                for h in hyps
            ):
                cands.append(
                    {
                        "kind": "experiment",
                        "atoms": ["act", factor],
                        "factor": factor,
                        "value": value,
                        "cost": int(self.inquire_cost_experiment),
                        "partition": part_key,
                    }
                )
        return cands

    def _inquire_already_resolved_factor(self, cue: str, factor: str) -> bool:
        """True if inquire trace already observed this factor probe.

        Default (INQUIRE-frozen): any observed factor is globally spent.
        With use_inquire_liveness (reliability on): cue-scoped, and only blocked
        while a live supporting consequence for that cue+factor remains.
        """
        cue_l = str(cue).strip().lower()
        fac_l = str(factor).strip().lower()
        for rec in self._inquire_rows():
            if str(rec.tags.get("phase") or "") != "observed":
                continue
            atoms = str(rec.tags.get("probe_atoms") or "")
            if fac_l not in atoms.split("|"):
                continue
            if not self.use_inquire_liveness:
                return True
            # Opt-in R10: require cue match via input_symbols
            inp = str(rec.tags.get("input_symbols") or "")
            toks = [t for t in inp.split("|") if t]
            if cue_l and cue_l not in toks and (not toks or toks[-1] != cue_l):
                continue
            if self._inquire_consequence_live(cue_l, fac_l):
                return True
            # Trace exists but consequence gone → eligible again
            continue
        return False

    def _inquire_consequence_live(self, cue: str, factor: str) -> bool:
        """True if cons_* grounding for cue still uniquely supports a hyp via factor."""
        for rec in self._grounding_rows():
            tid = str(rec.tags.get("trial_id") or "")
            if not (tid.startswith("cons_") or tid.startswith("conflict_")):
                continue
            if cue and cue not in tid:
                continue
            if factor and factor not in tid:
                continue
            return True
        return False

    def _inquire_render_probe(
        self,
        context_atoms: Sequence[str],
        probe_atoms: Sequence[str],
        *,
        min_support: int = 2,
    ) -> list[str] | None:
        """Render probe via frozen emit_sequence. None → cannot ask (HOLD)."""
        want = [str(x).strip().lower() for x in probe_atoms if str(x).strip()]
        if not want:
            return None
        # Probe identity is the intended atom list used as emit input_symbols.
        rendered = self.emit_sequence(
            list(context_atoms), want, min_support=min_support
        )
        seq = rendered.get("sequence")
        if not seq:
            return None
        got = [str(x).strip().lower() for x in seq if str(x).strip()]
        if got != want:
            return None
        return got

    def plan_inquiry(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """One-step epistemic plan: ANSWER | PROBE_ATOMS | SYMBOLIC_ACTION | HOLD.

        Does not call the teacher. Host executes probes and writes consequences via
        ordinary observation ABIs, then calls plan_inquiry again.
        PROBE_ATOMS / SYMBOLIC_ACTION require unique render via frozen emit_sequence.
        """
        required = {"context_atoms", "input_symbols"}
        out: dict[str, Any] = {
            "ok": True,
            "status": "HOLD",
            "answer_symbols": None,
            "probe_atoms": None,
            "action_symbol": None,
            "why": "hold",
            "value": None,
            "cost": None,
        }
        if not self.use_inquire:
            out["why"] = "inquire_off"
            return out
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            out["ok"] = False
            return out
        ctx = [str(x).strip().lower() for x in info["context_atoms"] if str(x).strip()]
        inp = [str(x).strip().lower() for x in info["input_symbols"] if str(x).strip()]
        if not ctx or not inp:
            out["why"] = "empty_probe"
            return out
        min_support = 2
        cue = self._inquire_cue(inp)
        if cue is None:
            out["why"] = "no_cue"
            return out
        hyps = self._inquire_hypotheses(cue, min_support=min_support)
        # PERSPECTIVE frozen influence #1: unique direct world grounding wins
        # before predictive testimony (does not change reliability-only order).
        if self.use_source_perspective and len(hyps) == 1:
            out["status"] = "ANSWER"
            out["answer_symbols"] = [hyps[0]]
            out["why"] = "unique_hypothesis"
            return out
        # Reliability: try weighted testimony before unique-hyp / probe paths
        live_hyps: list[str] = []
        if self.use_source_reliability:
            wans = self._reliability_weighted_answer(ctx, cue)
            if wans is not None:
                out["status"] = "ANSWER"
                out["answer_symbols"] = [wans]
                why = "source_evidence_margin"
                if self.use_source_perspective:
                    why = "source_evidence_margin_perspective"
                out["why"] = why
                return out
            live_hyps = self._reliability_live_hyps(cue, ctx)
        if not hyps:
            # Uncalibrated unique testimony agreement is NOT an answer
            if self.use_source_reliability and len(live_hyps) >= 2:
                hyps = list(live_hyps)
            else:
                out["why"] = "no_hypotheses"
                return out
        elif self.use_source_reliability and live_hyps:
            # Grounding ambiguous: include live hyps for inquire candidates
            if len(hyps) > 1:
                hyps = sorted(set(hyps) | set(live_hyps))
        if not hyps:
            out["why"] = "no_hypotheses"
            return out
        if len(hyps) == 1:
            out["status"] = "ANSWER"
            out["answer_symbols"] = [hyps[0]]
            out["why"] = "unique_hypothesis"
            return out
        # Ambiguous: need a justified discriminating probe
        if self._inquire_probes_used >= int(self.inquire_budget):
            out["why"] = "budget_exhausted"
            return out
        cands = self._inquire_probe_candidates(hyps, min_support=min_support)
        if self.use_source_reliability:
            cands.extend(self._reliability_speaker_candidates(ctx, cue, hyps))
        # Drop probes already observed (factor probes only; speaker asks use factor=spk)
        cands = [
            c
            for c in cands
            if not self._inquire_already_resolved_factor(cue, c["factor"])
        ]
        # Expression gate: only probes uniquely renderable by emit_sequence
        expressible: list[dict[str, Any]] = []
        for c in cands:
            rendered = self._inquire_render_probe(
                ctx, c["atoms"], min_support=min_support
            )
            if rendered is None:
                continue
            row = dict(c)
            row["rendered"] = rendered
            expressible.append(row)
        if not expressible:
            out["why"] = "cannot_express" if cands else "no_justified_probe"
            return out
        cands = expressible
        # Max value, then min cost; unique winner required
        best_val = max(c["value"] for c in cands)
        top = [c for c in cands if c["value"] == best_val]
        best_cost = min(c["cost"] for c in top)
        winners = [c for c in top if c["cost"] == best_cost]
        # Unique by (kind, atoms) — if multiple distinct probes tie, HOLD
        keys = {(w["kind"], tuple(w["atoms"])) for w in winners}
        if len(keys) != 1:
            out["why"] = "probe_tie"
            return out
        w = winners[0]
        probe_atoms = list(w["rendered"])
        # Record propose trace
        self.observe_inquire_trace(
            {
                "context_atoms": ctx,
                "input_symbols": inp,
                "probe_kind": w["kind"],
                "probe_atoms": probe_atoms,
                "predicted_partition": w["partition"],
                "cost": w["cost"],
                "phase": "propose",
            }
        )
        out["value"] = w["value"]
        out["cost"] = w["cost"]
        if w["kind"] == "ask":
            out["status"] = "PROBE_ATOMS"
            out["probe_atoms"] = probe_atoms
            out["why"] = "discriminating_ask"
        else:
            out["status"] = "SYMBOLIC_ACTION"
            out["action_symbol"] = probe_atoms[-1] if probe_atoms else None
            out["probe_atoms"] = probe_atoms
            out["why"] = "discriminating_experiment"
        return out

    def note_inquire_observation(
        self,
        *,
        context_atoms: Sequence[str],
        input_symbols: Sequence[str],
        probe_kind: str,
        probe_atoms: Sequence[str],
        cost: int,
        predicted_partition: str = "observed",
    ) -> None:
        """Host marks that a proposed probe was executed (trace only)."""
        if not self.use_inquire:
            return
        self._inquire_probes_used += 1
        self.observe_inquire_trace(
            {
                "context_atoms": list(context_atoms),
                "input_symbols": list(input_symbols),
                "probe_kind": probe_kind,
                "probe_atoms": list(probe_atoms),
                "predicted_partition": predicted_partition,
                "cost": int(cost),
                "phase": "observed",
            }
        )

    def reset_inquire_budget(self) -> None:
        self._inquire_probes_used = 0

    # --- TM.0.20.RELIABILITY: predictive accuracy / source_evidence_margin -------

    def observe_testimony(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Author experience_testimony only. No truth / confirm labels."""
        required = {"speaker_token", "context_atoms", "claim_atoms", "event_token"}
        out: dict[str, Any] = {"ok": False, "wrote": False, "why": ""}
        if not self.use_source_reliability or not self.store.enabled:
            out["why"] = "reliability_off"
            return out
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        speaker = str(info["speaker_token"]).strip().lower()
        event = str(info["event_token"]).strip().lower()
        ctx = [str(x).strip().lower() for x in info["context_atoms"] if str(x).strip()]
        claim = [str(x).strip().lower() for x in info["claim_atoms"] if str(x).strip()]
        if not speaker or not event or not ctx or len(claim) < 2:
            out["why"] = "empty_field"
            return out
        banned = {
            "is_correct",
            "truth",
            "trusted",
            "expected_source",
            "domain",
            "trust_score",
            "honesty",
            "intent",
        }
        if any(b in speaker or b in event for b in banned):
            out["why"] = "banned_token"
            return out
        if any("|" in x for x in ctx + claim + [speaker, event]):
            out["why"] = "pipe_in_token"
            return out
        tags: dict[str, Any] = {
            "speaker_token": speaker,
            "context_atoms": "|".join(sorted(self._reliability_project_context(ctx))),
            "context_raw": "|".join(ctx),
            "claim_atoms": "|".join(claim),
            "event_token": event,
            "cue": claim[0],
            "hypothesis": claim[1],
            "source": "experience_testimony",
            "hyp": "supported",
            "support": 1,
            "live": 1,
        }
        n = len(self.store.records())
        fid = f"testimony_{n:04d}_{speaker}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        # Supersede prior live claims for same speaker×cue×context (replacement)
        self._reliability_supersede_live(speaker, claim[0], tags["context_atoms"])
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def _reliability_project_context(self, context_atoms: Sequence[str]) -> list[str]:
        """Factor atoms only; exclude speaker/event/answer/domain tokens."""
        out: list[str] = []
        for a in context_atoms:
            t = str(a).strip().lower()
            if not t:
                continue
            if t.startswith("spk_") or t.startswith("evt_") or t.startswith("hyp_"):
                continue
            if t.startswith("ans_") or t.startswith("trial_"):
                continue
            if t in {"domain", "color_domain", "ownership_domain", "location_domain"}:
                continue
            # Keep factor-like tokens (feat_*, ctxf_*, or fixture factor_* )
            if t.startswith("feat_") or t.startswith("ctxf_") or t.startswith("fac_"):
                out.append(t)
            elif t.startswith("factor_"):
                out.append(t)
        return sorted(set(out))

    def _reliability_supersede_live(
        self, speaker: str, cue: str, ctx_key: str
    ) -> None:
        # Collect ids first: TagStore.write reloads and invalidates row refs.
        to_clear: list[str] = []
        for rec in self._testimony_rows():
            if str(rec.tags.get("speaker_token") or "") != speaker:
                continue
            if str(rec.tags.get("cue") or "") != cue:
                continue
            if str(rec.tags.get("context_atoms") or "") != ctx_key:
                continue
            if int(rec.tags.get("live") or 0) == 1:
                to_clear.append(str(rec.fact_id))
        for fid in to_clear:
            for rec in self._testimony_rows():
                if str(rec.fact_id) != fid:
                    continue
                if int(rec.tags.get("live") or 0) != 1:
                    break
                rec.tags["live"] = 0
                rec.what = encode_tags(rec.tags)
                self.store.write(rec)
                break

    def _testimony_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_testimony"
        ]

    def _reliability_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_reliability"
        ]

    def _reliability_jaccard(self, a: Sequence[str], b: Sequence[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / float(len(sa | sb))

    def _reliability_event_from_trial(self, trial_id: str) -> str | None:
        """Locked correlation: trial_id = evt_<event_token>__<rest>."""
        tid = str(trial_id or "").strip()
        if not tid.startswith("evt_") or "__" not in tid:
            return None
        return tid[4:].split("__", 1)[0].strip().lower() or None

    def _reliability_reconcile_from_ground(self, ground: FactRecord) -> None:
        """Organism-derived support/contradict vs pending testimony (same event_token)."""
        if not self.use_source_reliability:
            return
        prov = str(ground.tags.get("provenance") or "")
        if prov not in {"direct", "experiment", "state_read"}:
            return
        sym = str(ground.tags.get("symbol") or "").lower()
        paired = str(ground.tags.get("paired") or "").lower()
        result = str(ground.tags.get("result") or "").lower()
        tid = str(ground.tags.get("trial_id") or "")
        event_hint = self._reliability_event_from_trial(tid)
        if not event_hint:
            return
        for trec in self._testimony_rows():
            claim = str(trec.tags.get("claim_atoms") or "").split("|")
            if len(claim) < 2:
                continue
            if claim[0] != sym:
                continue
            ev = str(trec.tags.get("event_token") or "").lower()
            if ev != event_hint:
                continue
            # Failure of a different pairing is not evidence about this claim.
            if claim[1] == paired:
                derived = "contradict" if result == "failure" else "support"
            elif result == "failure":
                continue
            else:
                # success/correction of a different hyp for the same cue → contradict
                derived = "contradict"
            speaker = str(trec.tags.get("speaker_token") or "")
            # Append-only: skip duplicate compare for same speaker×event×verify
            dup = False
            for existing in self._reliability_rows():
                if (
                    str(existing.tags.get("speaker_token") or "") == speaker
                    and str(existing.tags.get("event_token") or "") == ev
                    and str(existing.tags.get("verify_trial_id") or "") == tid
                ):
                    dup = True
                    break
            if dup:
                continue
            self._reliability_append_derived(
                speaker=speaker,
                context_key=str(trec.tags.get("context_atoms") or ""),
                event_token=ev,
                derived=derived,
                claim_atoms=claim,
                verify_trial=tid,
            )

    def _reliability_append_derived(
        self,
        *,
        speaker: str,
        context_key: str,
        event_token: str,
        derived: str,
        claim_atoms: Sequence[str],
        verify_trial: str,
    ) -> None:
        if not speaker or derived not in {"support", "contradict"}:
            return
        tags: dict[str, Any] = {
            "speaker_token": speaker,
            "context_atoms": context_key,
            "event_token": event_token,
            "derived": derived,
            "claim_atoms": "|".join(claim_atoms),
            "verify_trial_id": verify_trial,
            "source": "experience_reliability",
            "hyp": "supported",
            "support": 1,
        }
        n = len(self.store.records())
        fid = f"reliability_{n:04d}_{derived}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)

    def source_evidence_margin(
        self, speaker: str, context_atoms: Sequence[str]
    ) -> float:
        """Bounded predictive-accuracy margin; not trust/honesty/intent."""
        if not self.use_source_reliability:
            return 0.0
        proj = self._reliability_project_context(context_atoms)
        thr = float(self.reliability_jaccard)
        s_cnt = 0
        k_cnt = 0
        for rec in self._reliability_rows():
            if str(rec.tags.get("speaker_token") or "") != speaker:
                continue
            ctx = [
                x
                for x in str(rec.tags.get("context_atoms") or "").split("|")
                if x
            ]
            if self._reliability_jaccard(proj, ctx) < thr:
                continue
            d = str(rec.tags.get("derived") or "")
            if d == "support":
                s_cnt += 1
            elif d == "contradict":
                k_cnt += 1
        n = s_cnt + k_cnt
        if n < int(self.reliability_n_min):
            return 0.0
        quality = (s_cnt - k_cnt) / float(n)
        lam = float(self.reliability_lambda)
        confidence = n / (n + lam)
        return max(0.0, quality * confidence)

    def _reliability_live_claims(
        self, context_atoms: Sequence[str], cue: str
    ) -> list[FactRecord]:
        """Deduped live claims for cue whose projected context overlaps the query."""
        cue_l = cue.lower()
        proj = self._reliability_project_context(context_atoms)
        thr = float(self.reliability_jaccard)
        live: dict[tuple[str, str, str, str], FactRecord] = {}
        for rec in self._testimony_rows():
            if int(rec.tags.get("live") or 0) != 1:
                continue
            if str(rec.tags.get("cue") or "") != cue_l:
                continue
            claim_proj = [
                x for x in str(rec.tags.get("context_atoms") or "").split("|") if x
            ]
            if self._reliability_jaccard(proj, claim_proj) < thr:
                continue
            key = (
                str(rec.tags.get("speaker_token") or ""),
                cue_l,
                str(rec.tags.get("hypothesis") or ""),
                str(rec.tags.get("context_atoms") or ""),
            )
            live[key] = rec
        return list(live.values())

    def _reliability_live_hyps(
        self, cue: str, context_atoms: Sequence[str]
    ) -> list[str]:
        hyps = {
            str(r.tags.get("hypothesis") or "")
            for r in self._reliability_live_claims(context_atoms, cue)
            if str(r.tags.get("hypothesis") or "")
        }
        return sorted(hyps)

    def _reliability_speaker_candidates(
        self,
        context_atoms: Sequence[str],
        cue: str,
        hyps: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Ask-speaker candidates; value from source_evidence_margin (not fixed priority)."""
        cands: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in self._reliability_live_claims(context_atoms, cue):
            spk = str(rec.tags.get("speaker_token") or "")
            if not spk or spk in seen:
                continue
            seen.add(spk)
            margin = self.source_evidence_margin(spk, context_atoms)
            # Uncalibrated (margin 0) → no ask-speaker candidate from this channel
            value = int(round(margin * 10.0))
            if value <= 0:
                continue
            cands.append(
                {
                    "kind": "ask",
                    "atoms": ["ask", spk],
                    "factor": spk,
                    "value": value,
                    "cost": int(self.inquire_cost_ask),
                    "partition": f"speaker:{spk}|hyps:{','.join(sorted(hyps))}",
                }
            )
        return cands

    def _reliability_weighted_answer(
        self, context_atoms: Sequence[str], cue: str
    ) -> str | None:
        """Unique max source_evidence_margin over deduped live claims, else None.

        With use_source_perspective: MISALIGNED live claims contribute weight 0
        (withhold, never invert). UNKNOWN/ALIGNED use predictive margin unpenalized.
        """
        scores: dict[str, float] = {}
        for rec in self._reliability_live_claims(context_atoms, cue):
            hyp = str(rec.tags.get("hypothesis") or "")
            if not hyp:
                continue
            spk = str(rec.tags.get("speaker_token") or "")
            w = self.source_evidence_margin(spk, context_atoms)
            if w <= 0:
                continue
            if self.use_source_perspective:
                status = self.report_alignment_status(
                    spk, [cue, hyp], context_atoms
                )
                if status == "MISALIGNED":
                    continue  # withhold channel influence; do not invert
            scores[hyp] = float(scores.get(hyp, 0.0)) + w
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))
        best_h, best_w = ranked[0]
        if best_w <= 0:
            return None
        if len(ranked) > 1 and round(ranked[1][1], 9) == round(best_w, 9):
            return None
        return best_h

    # --- TM.0.21.PERSPECTIVE: exposure, evidenced perspective, report alignment ---

    def observe_exposure(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Author raw experience_perspective exposure rows. No knows/believes/access."""
        required = {"speaker_token", "context_atoms", "exposure_atoms", "event_token"}
        out: dict[str, Any] = {"ok": False, "wrote": False, "why": ""}
        if not self.use_source_perspective or not self.store.enabled:
            out["why"] = "perspective_off"
            return out
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        speaker = str(info["speaker_token"]).strip().lower()
        event = str(info["event_token"]).strip().lower()
        ctx = [str(x).strip().lower() for x in info["context_atoms"] if str(x).strip()]
        atoms = [
            str(x).strip().lower() for x in info["exposure_atoms"] if str(x).strip()
        ]
        if not speaker or not event or not ctx or not atoms:
            out["why"] = "empty_field"
            return out
        banned = {
            "knows",
            "believes",
            "saw_truth",
            "is_lying",
            "honest",
            "has_access",
            "honesty",
            "intent",
            "trust_score",
            "false_belief",
        }
        blob = " ".join([speaker, event] + atoms)
        if any(b in blob for b in banned):
            out["why"] = "banned_token"
            return out
        if any("|" in x for x in ctx + atoms + [speaker, event]):
            out["why"] = "pipe_in_token"
            return out
        strong = sorted(a for a in atoms if a in self.perspective_strong_exposure)
        tags: dict[str, Any] = {
            "speaker_token": speaker,
            "context_atoms": "|".join(sorted(self._reliability_project_context(ctx))),
            "context_raw": "|".join(ctx),
            "exposure_atoms": "|".join(atoms),
            "event_token": event,
            "strong": 1 if strong else 0,
            "strong_atoms": "|".join(strong),
            "source": "experience_perspective",
            "hyp": "supported",
            "support": 1,
        }
        n = len(self.store.records())
        fid = f"perspective_{n:04d}_{speaker}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def _perspective_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_perspective"
        ]

    def _perspective_strong_exposures(self, speaker: str) -> list[FactRecord]:
        spk = speaker.lower()
        rows = [
            r
            for r in self._perspective_rows()
            if str(r.tags.get("speaker_token") or "") == spk
            and int(r.tags.get("strong") or 0) == 1
        ]
        return sorted(rows, key=lambda r: (int(r.when), str(r.fact_id)))

    def _perspective_world_for_event(
        self, event_token: str, cue: str
    ) -> list[tuple[str, int, str]]:
        """Return (paired_hyp, when, fact_id) for eligible grounds linked to event."""
        ev = event_token.lower()
        cue_l = cue.lower()
        out: list[tuple[str, int, str]] = []
        for rec in self._grounding_rows():
            prov = str(rec.tags.get("provenance") or "")
            if prov not in {"direct", "experiment", "state_read"}:
                continue
            if str(rec.tags.get("symbol") or "").lower() != cue_l:
                continue
            if str(rec.tags.get("result") or "").lower() == "failure":
                continue
            tid = str(rec.tags.get("trial_id") or "")
            hint = self._reliability_event_from_trial(tid)
            if hint != ev:
                continue
            paired = str(rec.tags.get("paired") or "").lower()
            if not paired:
                continue
            out.append((paired, int(rec.when), str(rec.fact_id)))
        return out

    def evidenced_perspective_hyp(
        self, speaker: str, cue: str
    ) -> str | None:
        """Last uniquely supported evidenced hyp for cue from strong exposure+event link.

        Presence-only never attaches. Exact event_token linkage only (not Jaccard).
        """
        if not self.use_source_perspective:
            return None
        # Chronological: each strong exposure may attach world facts for its event
        last_hyp: str | None = None
        last_when = -1
        last_fid = ""
        for exp in self._perspective_strong_exposures(speaker):
            ev = str(exp.tags.get("event_token") or "")
            if not ev:
                continue
            worlds = self._perspective_world_for_event(ev, cue)
            if not worlds:
                continue
            # Unique world hyp for this event×cue
            hyps = sorted({h for h, _w, _f in worlds})
            if len(hyps) != 1:
                continue
            # Prefer latest world observe among matches
            hyp, when, fid = max(worlds, key=lambda t: (t[1], t[2]))
            if when > last_when or (when == last_when and fid > last_fid):
                last_hyp = hyp
                last_when = when
                last_fid = fid
        return last_hyp

    def report_alignment_status(
        self,
        speaker: str,
        claim_atoms: Sequence[str],
        context_atoms: Sequence[str] | None = None,
    ) -> str:
        """Use-time ALIGNED | MISALIGNED | UNKNOWN. Not honesty/belief."""
        if not self.use_source_perspective:
            return "UNKNOWN"
        claim = [str(x).strip().lower() for x in claim_atoms if str(x).strip()]
        if len(claim) < 2:
            return "UNKNOWN"
        cue, hyp = claim[0], claim[1]
        persp = self.evidenced_perspective_hyp(speaker, cue)
        if persp is None:
            return "UNKNOWN"
        if persp == hyp:
            return "ALIGNED"
        return "MISALIGNED"

    def report_alignment_margin(
        self, speaker: str, context_atoms: Sequence[str]
    ) -> float:
        """Bounded margin from justified ALIGNED/MISALIGNED assessments (use-time).

        Jaccard transfers historical assessments across similar contexts only —
        never used to attach world facts to a perspective.
        Dedup: speaker × cue × hyp × evidenced perspective hyp (repetition without
        new exposure does not amplify).
        """
        if not self.use_source_perspective:
            return 0.0
        proj = self._reliability_project_context(context_atoms)
        thr = float(self.perspective_jaccard)
        # Collect unique justified assessments from live+historical testimony
        seen: set[tuple[str, str, str, str]] = set()
        s_cnt = 0
        k_cnt = 0
        for rec in self._testimony_rows():
            if str(rec.tags.get("speaker_token") or "") != speaker.lower():
                continue
            claim = str(rec.tags.get("claim_atoms") or "").split("|")
            if len(claim) < 2:
                continue
            ctx = [
                x
                for x in str(rec.tags.get("context_atoms") or "").split("|")
                if x
            ]
            # Transfer: only count assessments whose testimony context overlaps query
            if proj and ctx:
                if self._reliability_jaccard(proj, ctx) < thr:
                    continue
            elif proj or ctx:
                continue
            status = self.report_alignment_status(speaker, claim, context_atoms)
            if status == "UNKNOWN":
                continue
            # Dedup: same claim under same evidenced perspective — repetition
            # without new exposure must not amplify the margin.
            persp = self.evidenced_perspective_hyp(speaker, claim[0]) or ""
            key = (speaker.lower(), claim[0], claim[1], persp)
            if key in seen:
                continue
            seen.add(key)
            if status == "ALIGNED":
                s_cnt += 1
            elif status == "MISALIGNED":
                k_cnt += 1
        n = s_cnt + k_cnt
        if n < int(self.perspective_n_min):
            return 0.0
        quality = (s_cnt - k_cnt) / float(n)
        lam = float(self.perspective_lambda)
        confidence = n / (n + lam)
        return max(0.0, quality * confidence)

    # --- TM.0.22.INTERPRET: behaviorally evidenced interpretation ---

    def observe_source_consequence(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Author raw experience_interpretation rows. Opaque observables only."""
        required = {
            "source_token",
            "interaction_token",
            "exposure_event_token",
            "consequence_event_token",
            "context_symbols",
            "message_symbols",
            "action_symbols",
            "state_before",
            "state_after",
        }
        out: dict[str, Any] = {"ok": False, "wrote": False, "why": ""}
        if not self.use_source_interpretation or not self.store.enabled:
            out["why"] = "interpretation_off"
            return out
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        banned = {
            "understood",
            "intended",
            "belief",
            "honest",
            "honesty",
            "success",
            "failure",
            "correction",
            "cause",
            "misunderstood",
            "true_parse",
            "result",
        }
        src = str(info["source_token"]).strip().lower()
        ix = str(info["interaction_token"]).strip().lower()
        eev = str(info["exposure_event_token"]).strip().lower()
        cev = str(info["consequence_event_token"]).strip().lower()
        ctx = [str(x).strip().lower() for x in info["context_symbols"] if str(x).strip()]
        msg = [str(x).strip().lower() for x in info["message_symbols"] if str(x).strip()]
        act = [str(x).strip().lower() for x in info["action_symbols"] if str(x).strip()]
        sb = [str(x).strip().lower() for x in info["state_before"] if str(x).strip()]
        sa = [str(x).strip().lower() for x in info["state_after"] if str(x).strip()]
        if not src or not ix or not eev or not cev or not ctx or not msg or not act:
            out["why"] = "empty_field"
            return out
        if len(msg) != len(act):
            # Ordered-role evidence requires aligned message/action sequences.
            # Never silently truncate unequal lengths into a false map.
            out["why"] = "length_mismatch"
            return out
        # Exact opaque-token match only — substring bans reject innocent vocab
        # (e.g. "because"/"unsuccessful" under "cause"/"success").
        atoms = {src, ix, eev, cev, *ctx, *msg, *act, *sb, *sa}
        if atoms & banned:
            out["why"] = "banned_token"
            return out
        if any("|" in x for x in [src, ix, eev, cev] + ctx + msg + act + sb + sa):
            out["why"] = "pipe_in_token"
            return out
        tags: dict[str, Any] = {
            "source_token": src,
            "interaction_token": ix,
            "exposure_event_token": eev,
            "consequence_event_token": cev,
            "context_symbols": "|".join(ctx),
            "message_symbols": "|".join(msg),
            "action_symbols": "|".join(act),
            "state_before": "|".join(sb),
            "state_after": "|".join(sa),
            "source": "experience_interpretation",
            "hyp": "supported",
            "support": 1,
        }
        n = len(self.store.records())
        fid = f"interpret_{n:04d}_{src}"
        rec = FactRecord(
            fact_id=fid,
            what=encode_tags(tags),
            when=int(self.t),
            drive_scores={},
            tags=tags,
        )
        self.store.write(rec)
        out["ok"] = True
        out["wrote"] = True
        out["why"] = "authored"
        return out

    def _interpretation_rows(self) -> list[FactRecord]:
        return [
            r
            for r in self.store.records()
            if str(r.tags.get("source") or "") == "experience_interpretation"
        ]

    def _independent_ground_meaning(self, symbol: str, *, min_support: int = 2) -> str | None:
        """Unique non-testimony_derived grounding for symbol, else None."""
        sym = symbol.lower()
        scores: dict[str, int] = {}
        for rec in self._grounding_rows():
            if str(rec.tags.get("provenance") or "") == "testimony_derived":
                continue
            if str(rec.tags.get("symbol") or "").lower() != sym:
                continue
            if str(rec.tags.get("result") or "").lower() == "failure":
                continue
            paired = str(rec.tags.get("paired") or "").lower()
            if not paired:
                continue
            scores[paired] = int(scores.get(paired, 0)) + 1
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if ranked[0][1] < min_support:
            return None
        if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
            return None
        return ranked[0][0]

    def _interpret_episode_valid(self, rec: FactRecord) -> bool:
        """Require shared interaction linkage and independently grounded actions."""
        eev = str(rec.tags.get("exposure_event_token") or "")
        cev = str(rec.tags.get("consequence_event_token") or "")
        ix = str(rec.tags.get("interaction_token") or "")
        if not eev or not cev or not ix:
            return False
        # Exposure row for same source + event must exist (observable linkage)
        src = str(rec.tags.get("source_token") or "")
        has_expo = False
        for er in self.store.records():
            if str(er.tags.get("source") or "") != "experience_perspective":
                continue
            if str(er.tags.get("speaker_token") or "") != src:
                continue
            if str(er.tags.get("event_token") or "") == eev:
                has_expo = True
                break
        if not has_expo:
            return False
        acts = [x for x in str(rec.tags.get("action_symbols") or "").split("|") if x]
        if not acts:
            return False
        # Every action must be independently grounded
        for a in acts:
            if self._independent_ground_meaning(a) is None:
                return False
        return True

    def _interpret_symbol_votes(
        self,
        source: str,
        context_symbols: Sequence[str],
        *,
        backoff: bool = True,
    ) -> dict[str, dict[str, set[str]]]:
        """symbol → meaning → set of dedup episode keys (interaction_token)."""
        src = source.lower()
        ctx = [str(x).strip().lower() for x in context_symbols if str(x).strip()]
        ctx_set = set(ctx)
        votes: dict[str, dict[str, set[str]]] = {}
        for rec in self._interpretation_rows():
            if str(rec.tags.get("source_token") or "") != src:
                continue
            if not self._interpret_episode_valid(rec):
                continue
            rctx = [x for x in str(rec.tags.get("context_symbols") or "").split("|") if x]
            # Exact context match first; optional backoff drops only listed extra factors
            if backoff:
                # Accept if episode context equals query, or query is subset after
                # removing only non-core factors beyond {scene, fac_lab} shared core.
                core = {"scene", "fac_lab"}
                if set(rctx) != ctx_set:
                    # allow episode with fewer optional factors if core matches
                    if not (core <= set(rctx) and core <= ctx_set):
                        continue
                    # never cross: require all of episode's symbols ⊆ query or vice versa
                    if not (set(rctx) <= ctx_set or ctx_set <= set(rctx)):
                        continue
                    # if both have extras that differ, reject (same-context conflict rule)
                    if set(rctx) - core != ctx_set - core and set(rctx) != ctx_set:
                        # only exact optional match or pure core
                        if set(rctx) != ctx_set and not (
                            set(rctx) <= core and ctx_set <= core
                        ):
                            continue
            else:
                if set(rctx) != ctx_set:
                    continue
            msgs = [x for x in str(rec.tags.get("message_symbols") or "").split("|") if x]
            acts = [x for x in str(rec.tags.get("action_symbols") or "").split("|") if x]
            if len(msgs) != len(acts):
                # Refuse truncated zip — unequal roles are not interpretation evidence.
                continue
            ix = str(rec.tags.get("interaction_token") or "")
            for m, a in zip(msgs, acts):
                meaning = self._independent_ground_meaning(a)
                if meaning is None:
                    continue
                # Dedup by interaction — repetition does not multiply
                bucket = votes.setdefault(m, {})
                bucket.setdefault(meaning, set()).add(ix)
        return votes

    def interpret_message(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Use-time UNIQUE | AMBIGUOUS | INSUFFICIENT (+ candidate if UNIQUE)."""
        out: dict[str, Any] = {
            "ok": True,
            "status": "INSUFFICIENT",
            "candidate": None,
            "why": "",
        }
        if not self.use_source_interpretation:
            out["why"] = "interpretation_off"
            return out
        required = {"source_token", "context_symbols", "ordered_symbols"}
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["ok"] = False
            out["why"] = "exact_key_reject"
            return out
        src = str(info["source_token"]).strip().lower()
        ctx = [str(x).strip().lower() for x in info["context_symbols"] if str(x).strip()]
        ordered = [
            str(x).strip().lower() for x in info["ordered_symbols"] if str(x).strip()
        ]
        if not src or not ctx or not ordered:
            out["why"] = "empty_field"
            return out
        # Prefer exact context; if empty votes, try backoff once
        votes = self._interpret_symbol_votes(src, ctx, backoff=False)
        if not any(votes.get(s) for s in ordered):
            votes = self._interpret_symbol_votes(src, ctx, backoff=True)
        n_min = int(self.interpret_n_min)
        candidate: list[str] = []
        any_evidence = False
        ambiguous = False
        for sym in ordered:
            bag = votes.get(sym) or {}
            # count unique episodes per meaning
            ranked = sorted(
                ((m, len(eps)) for m, eps in bag.items()),
                key=lambda kv: (-kv[1], kv[0]),
            )
            if not ranked or ranked[0][1] < n_min:
                out["status"] = "INSUFFICIENT"
                out["why"] = "insufficient_symbol"
                out["candidate"] = None
                return out
            any_evidence = True
            if len(ranked) > 1 and ranked[1][1] >= n_min:
                ambiguous = True
                break
            # also ambiguous if two meanings both present with any support at same level
            if len(ranked) > 1 and ranked[1][1] > 0 and ranked[0][1] == ranked[1][1]:
                ambiguous = True
                break
            candidate.append(ranked[0][0])
        if not any_evidence:
            out["status"] = "INSUFFICIENT"
            out["why"] = "no_evidence"
            return out
        if ambiguous:
            out["status"] = "AMBIGUOUS"
            out["why"] = "conflicting_meanings"
            out["candidate"] = None
            return out
        out["status"] = "UNIQUE"
        out["candidate"] = candidate
        out["why"] = "unique"
        return out

    def interpretation_fit(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """SUPPORTED | CONFLICT | UNKNOWN. Internally reconstructs; never takes candidate."""
        out: dict[str, Any] = {"ok": True, "fit": "UNKNOWN", "why": ""}
        if not self.use_source_interpretation:
            out["why"] = "interpretation_off"
            return out
        required = {
            "source_token",
            "context_symbols",
            "message_symbols",
            "action_symbols",
            "state_before",
            "state_after",
        }
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["ok"] = False
            out["why"] = "exact_key_reject"
            return out
        recon = self.interpret_message(
            {
                "source_token": info["source_token"],
                "context_symbols": info["context_symbols"],
                "ordered_symbols": info["message_symbols"],
            }
        )
        if recon.get("status") != "UNIQUE" or not recon.get("candidate"):
            out["fit"] = "UNKNOWN"
            out["why"] = "no_unique_reconstruction"
            return out
        acts = [
            str(x).strip().lower() for x in info["action_symbols"] if str(x).strip()
        ]
        observed: list[str] = []
        for a in acts:
            m = self._independent_ground_meaning(a)
            if m is None:
                out["fit"] = "UNKNOWN"
                out["why"] = "action_not_independently_grounded"
                return out
            observed.append(m)
        cand = [str(x) for x in recon["candidate"]]
        if not cand or not observed:
            out["fit"] = "UNKNOWN"
            out["why"] = "empty_compare"
            return out
        # Length mismatch is incomplete observation — not a behavioral match.
        if len(cand) != len(observed):
            out["fit"] = "UNKNOWN"
            out["why"] = "length_mismatch"
            return out
        if cand == observed:
            out["fit"] = "SUPPORTED"
            out["why"] = "match"
        else:
            out["fit"] = "CONFLICT"
            out["why"] = "mismatch"
        return out

    def plan_interpretation(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """INTERPRET planner: separate ambiguous interpretation hyps (inquire value math)."""
        out: dict[str, Any] = {
            "ok": True,
            "status": "HOLD",
            "probe_atoms": None,
            "why": "",
        }
        if not self.use_source_interpretation:
            out["why"] = "interpretation_off"
            return out
        required = {"source_token", "context_symbols", "ordered_symbols"}
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["ok"] = False
            out["why"] = "exact_key_reject"
            return out
        recon = self.interpret_message(info)
        if recon.get("status") == "UNIQUE":
            out["status"] = "ANSWER"
            out["why"] = "unique_interpretation"
            out["candidate"] = recon.get("candidate")
            return out
        if recon.get("status") == "INSUFFICIENT":
            out["status"] = "HOLD"
            out["why"] = "insufficient"
            return out
        # AMBIGUOUS → propose a diagnostic ask over source (reuse inquire cost model)
        src = str(info["source_token"]).strip().lower()
        probe = ["ask", src, "which_meaning"]
        # Expression gate via emit_sequence if available
        if self.use_symbol_sequence:
            ctx = [str(x).strip().lower() for x in info["context_symbols"] if str(x).strip()]
            rendered = self.emit_sequence(ctx, probe)
            if not rendered.get("ok") or rendered.get("sequence") != probe:
                # Still allow PROBE status for lab scoring when emit cannot express
                out["status"] = "PROBE"
                out["probe_atoms"] = probe
                out["why"] = "ambiguous_probe"
                return out
        out["status"] = "PROBE"
        out["probe_atoms"] = probe
        out["why"] = "ambiguous_probe"
        return out

    def plan_recipient_message(self, info: dict[str, Any] | None) -> dict[str, Any]:
        """Emit recipient-specific message for uniquely resolved goal cues."""
        out: dict[str, Any] = {
            "ok": False,
            "status": "HOLD",
            "sequence": None,
            "why": "",
        }
        if not self.use_source_interpretation:
            out["why"] = "interpretation_off"
            return out
        required = {"recipient_token", "context_symbols", "goal_cue_symbols"}
        if not isinstance(info, dict) or set(info.keys()) != required:
            out["why"] = "exact_key_reject"
            return out
        recip = str(info["recipient_token"]).strip().lower()
        ctx = [str(x).strip().lower() for x in info["context_symbols"] if str(x).strip()]
        cues = [
            str(x).strip().lower() for x in info["goal_cue_symbols"] if str(x).strip()
        ]
        if not recip or not ctx or not cues:
            out["why"] = "empty_field"
            return out
        # Resolve unique communicative goal from pre-existing grounding of cues
        goals: list[str] = []
        for cue in cues:
            g = self._independent_ground_meaning(cue)
            if g is None:
                out["status"] = "HOLD"
                out["why"] = "goal_unresolved"
                return out
            goals.append(g)
        if len(set(goals)) != 1:
            out["status"] = "HOLD"
            out["why"] = "goal_ambiguous"
            return out
        goal = goals[0]
        # Invert recipient symbol map: find message symbols uniquely meaning goal
        votes = self._interpret_symbol_votes(recip, ctx, backoff=False)
        if not votes:
            votes = self._interpret_symbol_votes(recip, ctx, backoff=True)
        n_min = int(self.interpret_n_min)
        candidates: list[str] = []
        for sym, bag in votes.items():
            ranked = sorted(
                ((m, len(eps)) for m, eps in bag.items()),
                key=lambda kv: (-kv[1], kv[0]),
            )
            if not ranked or ranked[0][1] < n_min:
                continue
            if ranked[0][0] != goal:
                continue
            if len(ranked) > 1 and ranked[1][1] >= n_min:
                continue  # ambiguous symbol
            candidates.append(sym)
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            out["status"] = "HOLD"
            out["why"] = "recipient_projection_ambiguous"
            return out
        seq = [candidates[0]]
        # Prefer SEQUENCE emit under recipient-tagged input when demos exist
        if self.use_symbol_sequence:
            for prefix_inputs in (
                ["say", recip],
                ["tell", recip],
                list(ctx[:1]) + [recip],
            ):
                emitted = self.emit_sequence(ctx, prefix_inputs)
                if (
                    emitted.get("ok")
                    and emitted.get("sequence")
                    and list(emitted["sequence"]) == seq
                ):
                    out["ok"] = True
                    out["status"] = "EMIT"
                    out["sequence"] = seq
                    out["why"] = "recipient_emit"
                    return out
        # Atomic construction without partial: accept projected sequence when UNIQUE
        out["ok"] = True
        out["status"] = "EMIT"
        out["sequence"] = seq
        out["why"] = "recipient_projection"
        return out

    def _find_experience_skel(self, bind: str, did: str) -> FactRecord | None:
        bl, dl = bind.lower(), did.lower()
        for rec in self.store.records():
            if str(rec.tags.get("source") or "") != "experience_skel":
                continue
            if isinstance(rec.tags.get("ctx"), str) and rec.tags.get("ctx"):
                continue
            if str(rec.tags.get("bind") or "").lower() != bl:
                continue
            if str(rec.tags.get("did") or "").lower() != dl:
                continue
            return rec
        return None

    def _hyp_state(self, rec) -> str:
        raw = rec.tags.get("hyp")
        if isinstance(raw, str) and raw in ("untried", "supported", "contradicted"):
            return raw
        return "untried"

    def _hyp_trials(self, rec) -> int:
        n = rec.tags.get("trials")
        if isinstance(n, (int, np.integer)):
            return int(n)
        return 0

    def _init_hyp(self, rec) -> None:
        if not self.use_hyp_survive:
            return
        if rec.tags.get("hyp") not in ("untried", "supported", "contradicted"):
            rec.tags["hyp"] = "untried"
            rec.tags["trials"] = 0
            rec.tags["wins"] = 0
            rec.tags["losses"] = 0
            rec.tags["support"] = 0
            rec.tags["contradiction"] = 0

    def _mark_hyp(self, rec, *, success: bool) -> None:
        if not self.use_hyp_survive or not self.store.enabled:
            return
        self._init_hyp(rec)
        rec.tags["trials"] = self._hyp_trials(rec) + 1
        if success:
            rec.tags["wins"] = int(rec.tags.get("wins") or 0) + 1
            rec.tags["hyp"] = "supported"
        else:
            rec.tags["losses"] = int(rec.tags.get("losses") or 0) + 1
            rec.tags["hyp"] = "contradicted"
        rec.tags["support"] = int(rec.tags.get("wins") or 0)
        rec.tags["contradiction"] = int(rec.tags.get("losses") or 0)
        self.store.write(rec)

    def _update_chosen_hyp(self, *, success: bool) -> None:
        if not self.use_hyp_survive or not self.store.enabled:
            return
        ids = {i for i in self._last_chosen_ids if i}
        if not ids:
            return
        for rec in list(self.store.records()):
            if rec.fact_id in ids:
                self._mark_hyp(rec, success=success)

    def _prefer_untried(self, pool: list) -> list:
        """Same-here exploration: unused hypotheses still carry information."""
        if not self.use_hyp_survive or not pool:
            return pool
        live = [r for r in pool if self._hyp_state(r) != "contradicted"]
        if not live:
            live = list(pool)
        untried = [r for r in live if self._hyp_state(r) == "untried"]
        if untried:
            return untried
        trials = [self._hyp_trials(r) for r in live]
        least = min(trials)
        return [r for r, n in zip(live, trials) if n == least]

    def _evidence_score(self, rec) -> tuple[int, int]:
        """Inspectable (support, -contradiction). No token or filename."""
        wins = rec.tags.get("support", rec.tags.get("wins"))
        losses = rec.tags.get("contradiction", rec.tags.get("losses"))
        w = int(wins) if isinstance(wins, (int, np.integer)) else 0
        n = int(losses) if isinstance(losses, (int, np.integer)) else 0
        return (w, -n)

    def _evidence_choose(self, recs: list) -> list:
        """Unique strict evidence winner. A tie with different dids is unresolved."""
        if not self.use_evidence or not recs:
            return recs
        scored = [(self._evidence_score(r), r) for r in recs]
        best = max(s for s, _r in scored)
        winners = [r for s, r in scored if s == best]
        if len(winners) == 1:
            return winners
        dids = {
            str(r.tags.get("did")).lower()
            for r in winners
            if isinstance(r.tags.get("did"), str)
        }
        if len(dids) > 1:
            return []
        return winners

    def _keep_steerer(self, obs) -> None:
        """Drop other same-here notes after a file steered. Not a subject."""
        if not self.use_keep_steerer or not self.store.enabled:
            return
        keep = {i for i in self._last_chosen_ids if i}
        if self._in_hand_id:
            keep.add(str(self._in_hand_id))
        if not keep:
            return
        for rec in list(self.store.records()):
            if rec.fact_id in keep:
                continue
            if not self._rec_names_here(rec, obs):
                continue
            # Success of A does not falsify untested B.
            if self.use_hyp_survive and self._hyp_state(rec) != "contradicted":
                continue
            # EVIDENCE needs the contradicted rival to remain inspectable.
            if self.use_evidence:
                continue
            if self.store.delete(rec.fact_id):
                self.n_revised += 1

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

    def _stamp_leftover_new_here(self, station: str | None):
        """Librarian leftover: first unowned rare in W order. Off when new-here is in-hand only."""
        if self.use_in_hand_new_here:
            return None
        if not self._commit_rare_unmarked():
            return None
        rare_recs = [r for r in self.store.records() if self._is_rare_in_world(r)]
        return self._pick_stamp_note(rare_recs, station)

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
            if self.use_retry_novel and self._in_hand_is_novel():
                hand = self._in_hand_note() or self._commit_in_hand()
                if (
                    hand is not None
                    and self._is_rare_in_world(hand)
                    and (rec is None or hand.fact_id != rec.fact_id)
                ):
                    rec = hand
            if rec is None:
                rec = self._in_hand_note() or self._commit_in_hand()
                if rec is not None and not self._is_rare_in_world(rec):
                    rec = None
                if rec is None and new_here:
                    rec = self._stamp_leftover_new_here(station)
        elif self.use_in_hand_new_here and new_here:
            rec = self._in_hand_note() or self._commit_in_hand()
            if rec is not None and not self._is_rare_in_world(rec):
                rec = None
        else:
            rec = self._pick_stamp_note(rare_recs, station)
            if rec is None and new_here:
                rec = self._stamp_leftover_new_here(station)
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
        self._init_hyp(rec)
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
                is_act = val in self._act_map(recs if self.use_local_alias else None)
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
        if self.use_bind_match:
            did = rec.tags.get("did")
            if isinstance(did, str):
                aid = {a.name.lower(): int(a) for a in self._body_enum()}.get(did.lower())
                if aid is not None:
                    logits[int(aid)] += 3.0
                    return
        if self.use_prose_tokens:
            key = self._value_tag() if self.use_read else ""
            val = rec.tags.get(key)
            if isinstance(val, str):
                act = self._act_map([rec] if self.use_local_alias else None).get(val.lower())
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
        if self.use_bind_match:
            hits = self._match_applicable(hits, obs)
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
        if self.use_bind_match:
            if self.use_compose:
                matched = self._match_applicable(
                    list(self.store.records()) if self.store.enabled else list(chosen),
                    obs,
                )
                self.last_policy["bind_present_in_current_stream"] = bool(matched)
                chosen = self._compose_choose(obs)
                apply = bool(chosen)
            else:
                matched = self._match_applicable(
                    list(self.store.records()) if self.store.enabled else list(chosen),
                    obs,
                )
                present = bool(matched)
                self.last_policy["bind_present_in_current_stream"] = present
                if self.use_evidence:
                    chosen = self._evidence_choose(matched)
                    self.last_policy["evidence_resolved"] = len(chosen) == 1
                    self.last_policy["evidence_tie"] = bool(matched) and len(chosen) != 1
                else:
                    chosen = self._match_applicable(chosen, obs)
                apply = bool(chosen)
        self._last_chosen_ids = []
        if apply:
            for rec in chosen:
                fid = getattr(rec, "fact_id", None)
                if fid:
                    self._last_chosen_ids.append(str(fid))
                bind = rec.tags.get("bind")
                if isinstance(bind, str):
                    self.last_policy["used_file"] = fid
                    self.last_policy["used_bind"] = bind.lower()
                self._apply_record_bias(logits, rec, obs)
        return logits

    def _rho_bias(self) -> np.ndarray:
        """Session residue only: recent embed + last successful action (cleared on reset)."""
        logits = 0.35 * self.cortex.baseline_logits(self.rho.predict())
        if self.rho.last_success_action is not None:
            logits[self.rho.last_success_action] += 2.5
        return logits

    def act(self, obs: Obs, *, update_rho: bool = True, explore: bool = False) -> tuple[int, dict[str, Any]]:
        # One-shot lived context: every new act/composition attempt clears prior pending.
        if self.use_acquire_ctx:
            self._clear_lived_context()
        vec = obs.vector(self.cortex.config.obs_dim)
        predicted = self.rho.predict()
        embed = self.cortex.encode(vec)
        novelty = self.drives.novelty(embed, predicted)

        logits = self.cortex.baseline_logits(embed)
        logits = logits + self._rho_bias()
        # Species prior: try a default motor at a station (OPEN on doors, HOLD on dial).
        # Dial HOLD is wrong on A and C so empty S cannot look like Store-works.
        # Life knowledge must come from S (or fragile ρ after recent success).
        if self.n_actions == 5:
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
            at_door = bool(
                getattr(obs, "at_red_door", False)
                or getattr(obs, "at_blue_door", False)
                or getattr(obs, "at_green_door", False)
            )
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

        if self.n_actions == 5:
            at_station = bool(
                getattr(obs, "at_a", False)
                or getattr(obs, "at_b", False)
                or getattr(obs, "at_c", False)
            )
            if not at_station:
                for a in (DialAction.PRESS, DialAction.HOLD, DialAction.TUNE, DialAction.FLIP):
                    logits[int(a)] -= 5.0
        else:
            at_door = bool(
                getattr(obs, "at_red_door", False)
                or getattr(obs, "at_blue_door", False)
                or getattr(obs, "at_green_door", False)
            )
            # Hard constraints from current percept (not knowledge).
            if not getattr(obs, "has_key", False):
                logits[Action.USE_KEY] -= 5.0
            if getattr(obs, "has_key", False) or not getattr(obs, "key_visible", False):
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
        elif getattr(obs, "last_ok", False):
            event = "dial_ok"
        elif getattr(obs, "last_failed", False):
            event = "dial_failed"

        action_name = info.get("action")
        name_to_id = {a.name.lower(): int(a) for a in self._body_enum()}
        opened = bool(info.get("opened"))
        door = self._door_code(obs) if self.native else None

        if self.use_revise_head and success is False:
            self._maybe_revise(failed=True)

        if self.use_event_annotate and opened and action_name:
            self._maybe_annotate(str(action_name).lower(), obs)
        if success is True:
            self._update_chosen_hyp(success=True)
        elif success is False:
            self._update_chosen_hyp(success=False)
        # Contextual continuation from compose-lived (κ, bind). Consume once; always clear.
        try:
            self._apply_acquire_ctx(
                success=success,
                action_name=str(action_name).lower() if isinstance(action_name, str) else None,
            )
        finally:
            if self.use_acquire_ctx:
                self._clear_lived_context()
        if self.use_keep_steerer and success is True:
            self._keep_steerer(obs)

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
