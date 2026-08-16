"""TM.0.6.2: never-wipe English life on the TM.0.6.1 one-bind recipe.

Same tiny English W and stream-first bind as TM.0.6.1, but train does not rmtree S.
After train, probe that dirty store (no fresh A life). Then a C life on the same S.
Not math, not shared-return rescue, not dropping has_code or domain="dial",
not turning on 0.5.9 revise/here-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm010 import _copy_heads, _head_fp
from experiments.run_tm012 import _has_field
from experiments.run_tm040 import probe
from experiments.run_tm052 import live_free
from experiments.run_tm054 import _MOTOR_WORDS
from experiments.run_tm055 import _accumulate_lives
from experiments.run_tm056 import _copy_s
from experiments.run_tm061 import (
    _enrich,
    _life_then_two_probes,
    _w_flags as _w_flags061,
    _write_nonce_s,
    clutter_prose,
    make as _make061,
    wiki_prose,
)
from experiments.run_v22 import _tags
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes

LIVE_KEYS = (
    "opened",
    "files",
    "n_forced",
    "n_steps",
    "n_explored",
    "n_annotated",
    "found_press",
    "found_tune",
    "found_push",
    "found_adjust",
    "found_argon",
    "found_alpha",
    "found_did_press",
    "found_did_tune",
    "found_bind_push",
    "found_bind_adjust",
    "found_bind_argon",
    "found_bind_alpha",
    "found_krypton",
    "found_helium",
    "found_cha",
    "found_chc",
    "actions",
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm062"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(
    *args,
    use_alias_bind: bool = True,
    use_did_stamp: bool = True,
    use_one_bind: bool = True,
    use_commit_rare_only: bool = True,
    use_revise_head: bool = False,
    use_commit_here_only: bool = False,
    use_event_annotate: bool = True,
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
    **kwargs,
):
    if not use_event_annotate:
        use_alias_bind = False
        use_did_stamp = False
        use_one_bind = False
        use_stamp_new_here = False
        use_block_here = False
        use_in_hand_new_here = False
        use_find_novel = False
        use_retry_novel = False
        use_local_alias = False
        use_keep_steerer = False
        use_count_search = False
        use_hyp_survive = False
        use_bind_match = False
        use_evidence = False
        use_compose = False
        use_context_kappa = False
        use_acquire_ctx = False
        use_acquire_skel = False
        use_acquire_relate = False
        use_alias_fingerprint = False
    if kwargs.get("use_here_match") is False:
        use_stamp_new_here = False
        use_block_here = False
        use_in_hand_new_here = False
        use_keep_steerer = False
        use_hyp_survive = False
        use_bind_match = False
        use_evidence = False
        use_compose = False
        use_context_kappa = False
        use_acquire_ctx = False
        use_acquire_skel = False
        use_acquire_relate = False
        use_alias_fingerprint = False
    if kwargs.get("use_search_head") is False:
        use_find_novel = False
        use_retry_novel = False
        use_count_search = False
    if not use_find_novel:
        use_retry_novel = False
    if not use_alias_bind:
        use_local_alias = False
        use_bind_match = False
    if not use_bind_match or not use_hyp_survive:
        use_evidence = False
    if not use_evidence:
        use_compose = False
    if not use_compose:
        use_context_kappa = False
    if not use_context_kappa:
        use_acquire_ctx = False
    if not use_acquire_ctx:
        use_acquire_skel = False
    if not use_acquire_skel:
        use_acquire_relate = False
    if not use_acquire_relate:
        use_alias_fingerprint = False
    if not use_stamp_new_here:
        use_in_hand_new_here = False
    return _make061(
        *args,
        use_alias_bind=use_alias_bind,
        use_did_stamp=use_did_stamp,
        use_one_bind=use_one_bind,
        use_commit_rare_only=use_commit_rare_only,
        use_revise_head=use_revise_head,
        use_commit_here_only=use_commit_here_only,
        use_event_annotate=use_event_annotate,
        use_stamp_new_here=use_stamp_new_here,
        use_block_here=use_block_here,
        use_in_hand_new_here=use_in_hand_new_here,
        use_find_novel=use_find_novel,
        use_retry_novel=use_retry_novel,
        use_local_alias=use_local_alias,
        use_keep_steerer=use_keep_steerer,
        use_count_search=use_count_search,
        use_hyp_survive=use_hyp_survive,
        use_bind_match=use_bind_match,
        use_evidence=use_evidence,
        use_compose=use_compose,
        use_context_kappa=use_context_kappa,
        use_acquire_ctx=use_acquire_ctx,
        use_acquire_skel=use_acquire_skel,
        use_acquire_relate=use_acquire_relate,
        use_alias_fingerprint=use_alias_fingerprint,
        **kwargs,
    )


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags061(w_files, w_dir)
    flags["english_life"] = True
    flags["use_alias_bind"] = True
    flags["use_one_bind"] = True
    flags["accumulate_s"] = True
    flags["train_wipe_s"] = False
    flags["eval_s_wiped_between"] = False
    flags["open_w"] = True
    flags["use_commit_rare_only"] = True
    flags["use_revise_head"] = False
    flags["use_commit_here_only"] = False
    flags["use_stamp_new_here"] = False
    flags["use_block_here"] = False
    flags["use_in_hand_new_here"] = False
    flags["use_find_novel"] = False
    flags["use_retry_novel"] = False
    flags["use_local_alias"] = False
    flags["use_keep_steerer"] = False
    flags["use_count_search"] = False
    flags["use_hyp_survive"] = False
    flags["use_bind_match"] = False
    flags["use_evidence"] = False
    flags["use_compose"] = False
    flags["w_has_p98"] = "p98.md" in w_files
    return flags


def _live_slice(live: dict[str, Any]) -> dict[str, Any]:
    return {k: live.get(k) for k in LIVE_KEYS}


def _s_snapshot(s_dir: Path) -> dict[str, Any]:
    tag = _tags(s_dir) if s_dir.exists() else ""
    files = sorted(p.name for p in s_dir.glob("*.tag")) if s_dir.exists() else []
    live = {
        "tag": tag,
        "files": files,
        "opened": False,
        "n_forced": 0,
        "n_steps": 0,
        "n_explored": 0,
        "n_annotated": 0,
        "found_press": False,
        "found_tune": False,
        "found_krypton": False,
        "found_helium": False,
        "actions": [],
    }
    snap = _enrich(live)
    snap["n"] = len(files)
    snap["tag"] = tag
    snap["files"] = files
    return snap


def _probe_s(
    s_dir: Path,
    policy: UsePolicy,
    seed: int,
    *,
    rng: np.random.Generator | None = None,
    **make_kw: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ag = make(s_dir, None, policy, explore_epsilon=0.0, rng=rng, **make_kw)
    ag.reset_rho()
    p_a = probe(ag, "probe_channel_a", seed)
    ag.reset_rho()
    p_c = probe(ag, "probe_channel_c", seed)
    return p_a, p_c


def _c_life_on_s(
    s_dir: Path,
    w_dir: Path,
    policy: UsePolicy,
    seed: int,
    *,
    max_steps: int,
    explore_epsilon: float,
    rng: np.random.Generator | None = None,
    **make_kw: Any,
):
    ag = make(s_dir, w_dir, policy, explore_epsilon=explore_epsilon, rng=rng, **make_kw)
    ag.reset_rho()
    ag.policy_traces = []
    c_live = _enrich(live_free(ag, "experience_channel_c", seed, max_steps=max_steps))
    ag.world = None
    ag.reset_rho()
    both_a = probe(ag, "probe_channel_a", seed)
    ag.reset_rho()
    both_c = probe(ag, "probe_channel_c", seed)
    return ag, c_live, both_a, both_c, c_live["tag"]


def _wipe_lives(*args, **kwargs):
    import experiments.run_tm055 as tm055

    saved = tm055.make
    tm055.make = make
    try:
        return _accumulate_lives(*args, **kwargs)
    finally:
        tm055.make = saved


def _train_keep(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    *,
    split: bool,
    max_steps: int,
) -> tuple[list[float], Path]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_m = b_u = 0.0
    s_dir = work / "ep"
    s_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        snap = _s_snapshot(s_dir)
        ag = make(s_dir, w_dir, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live = _enrich(live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps))
        tr_life = list(ag.policy_traces)
        r_find = (
            1.0
            if (live["found_push"] and live["found_argon"])
            or (snap["found_push"] and snap["found_argon"])
            else 0.0
        )
        wrote = any(t.get("kind") == "write" and t.get("write") for t in tr_life)
        r_mark = (
            1.0
            if (
                live["found_bind_push"]
                and live["found_argon"]
                and live["found_cha"]
                and live["found_did_press"]
                and not live["found_bind_argon"]
            )
            or (
                snap["found_bind_push"]
                and snap["found_cha"]
                and snap["found_argon"]
                and not snap["found_bind_argon"]
                and wrote
            )
            else 0.0
        )
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_channel_a", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        if split:
            b_f = 0.9 * b_f + 0.1 * r_find
            b_m = 0.9 * b_m + 0.1 * r_mark
            b_u = 0.9 * b_u + 0.1 * r_use
            policy.update([t for t in tr if t.get("kind") == "search"], r_find - b_f)
            policy.update([t for t in tr if t.get("kind") == "write"], r_mark - b_m)
            policy.update([t for t in tr if t.get("kind") == "vname"], r_use - b_u)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") in ("search", "write")], adv)
            policy.update([t for t in tr if t.get("kind") == "vname"], adv)
        rewards.append(r_use)
    return rewards, s_dir


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m.get("door_world"):
        return "Confound", "Door world was restored."
    if m.get("world") != "channel_dial":
        return "Confound", "Not the channel dial world."
    if m["w_has_tag_files"] or not m["w_all_md"]:
        return "Confound", "W still used .tag files."
    if not m["w_has_prose"]:
        return "Confound", "W .md files have no prose."
    if m["w_has_where"] or m["w_has_action"] or m["w_has_loc"] or m["w_has_here"] or m["w_has_door"]:
        return "Confound", "Filed k=v motor/place tags were planted."
    if not m["w_has_p99"] or not m.get("w_has_p98"):
        return "Confound", "Eval W must have both useful pages p99.md and p98.md."
    if m.get("w_body_ints") or m.get("w_a_ints"):
        return "Confound", "Answer integers were restored in W."
    if m.get("w_has_motor_name") or any(t in m.get("w_a_tokens", []) for t in _MOTOR_WORDS):
        return "Confound", "Unread W still names an innate motor."
    if m.get("w_has_station_name") or m.get("w_a_has_station"):
        return "Confound", "Unread W still names an innate station."
    if not m.get("english_life"):
        return "Fail", "English life was frozen off."
    if m.get("w_a_has_nonce"):
        return "Confound", "Useful page is still a nonce scrap, not an English word."
    if not m.get("w_a_has_synonym"):
        return "Confound", "Useful page never used an English corpus word."
    if not m.get("use_alias_bind"):
        return "Fail", "Alias bind was frozen off."
    if not m.get("use_one_bind"):
        return "Fail", "One-bind was frozen off."
    if not m.get("w_a_has_distractor"):
        return "Confound", "Useful page has no distractor hapax."
    if (m.get("w_useful_n_rare") or 0) < 2:
        return "Confound", "Unique-rare needle was restored."
    if not m.get("open_w"):
        return "Fail", "W was not document-shaped Open W."
    if m.get("w_clutter_cloned") or (m.get("w_n_distinct_clutter") or 0) < 11:
        return "Confound", "W clutter is still cloned one-liners."
    if (m.get("w_min_paragraphs") or 0) < 2:
        return "Confound", "W pages are not multi-paragraph documents."
    if m.get("w_clutter_has_rare"):
        return "Confound", "Clutter pages are rare-word pages; search is not load-bearing."
    if not m.get("w_useful_has_rare"):
        return "Confound", "Useful page has no rare word."
    if not m.get("accumulate_s") or m.get("eval_s_wiped_between"):
        return "Fail", "S was still wiped between eval lives."
    if m.get("train_wipe_s"):
        return "Fail", "Train still wiped S every episode."
    if not m.get("use_commit_rare_only"):
        return "Fail", "Rare-only commit was frozen off."
    if m.get("use_revise_head") or m.get("use_commit_here_only"):
        return "Confound", "Dirty-store correct flags were smuggled onto this English never-wipe slice."
    if m.get("use_stamp_new_here"):
        return "Confound", "New-here stamp was smuggled onto this slice."
    if not m.get("has_code_in_search"):
        return "Confound", "has_code was dropped from search (not this jump)."
    if not m.get("domain_switch"):
        return "Confound", "domain= switch was removed (not this jump)."
    if m["write_from_events"]:
        return "Confound", "v9 write_from_events was restored."
    if m.get("use_prose_ints"):
        return "Confound", "Digit-copy use_prose_ints was restored."
    if not m.get("use_event_annotate"):
        return "Fail", "Event annotate was frozen off."
    if not m.get("use_here_match"):
        return "Fail", "Here-match was frozen off."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue the dial."
    if m["use_match_head"] or m["use_qname_head"]:
        return "Fail", "Exact query match was restored."
    if not (m["use_search_head"] and m["use_vname_head"] and m["use_read"] and m["use_prose_tokens"]):
        return "Fail", "Prose token search/vname path was frozen off."
    if not m["trained_life"]:
        return "Fail", "Training was not a free life."
    if any(lv.get("n_forced", 0) for lv in (m["untrained_live"], m["a_live"], m["c_live"])):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_a"]["correct"]:
        return "Confound", "disable-S still solved A; fact leaked outside S."
    if any(
        p.get("explored")
        for p in (m["untrained_probe"], m["train_s_probe"], m["train_s_foil"], m["both_after_a"], m["both_after_c"])
    ):
        return "Confound", "Probe used exploration."
    if m["empty_S_c"]["correct"] or m["empty_S_c"]["action_name"] == "tune":
        return "Confound", "Empty S already TUNE on held-out C."
    if m.get("domain") != "dial" or m.get("n_actions") != 5:
        return "Confound", "Agent was not dial domain / 5 actions."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "press":
        return "Fail", "Untrained already PRESS-correct on A."
    if m["untrained_foil_c"]["action_name"] == "press":
        return "Fail", "Untrained already PRESS on foil C."
    if m["untrained_live"].get("found_did_press") or m["untrained_live"].get("n_annotated", 0):
        return "Fail", "Untrained already bound a motor onto S."
    if not m["train_s_probe"]["correct"] or m["train_s_probe"]["action_name"] != "press":
        return "Fail", "After never-wipe train, dirty S was not PRESS."
    if m["train_s_foil"]["action_name"] == "press" or m["train_s_foil"]["correct"]:
        return "Fail", "Train S still fired PRESS on channel C."
    if m["train_s_foil"]["action_name"] != "hold":
        return "Fail", "Train S on channel C was not HOLD."
    if not m["both_after_a"]["correct"] or m["both_after_a"]["action_name"] != "press":
        return "Fail", "C life on dirty S lost A's PRESS."
    if not m["both_after_c"]["correct"] or m["both_after_c"]["action_name"] != "tune":
        return "Fail", "C life on dirty S was not TUNE."
    if m["wipe_ctrl_a"]["action_name"] == "press" or m["wipe_ctrl_a"]["correct"]:
        return "Fail", "Wiping dirty S still PRESS on A; keep-S was not load-bearing."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if m["menu_control"]["correct"]:
        return "Fail", "Exact match still solved A."
    if m["search_control"]["correct"]:
        return "Fail", "Untrained search still solved A."
    if m["write_control"]["correct"]:
        return "Fail", "Annotate-off still solved A."
    if m["clutter_control"]["correct"]:
        return "Fail", "Clutter-only W still solved A."
    if m["bind_control"]["correct"]:
        return "Fail", "Bind-off still solved A; English bind was not load-bearing."
    if m["nonce_control"]["correct"] or m["nonce_control"]["action_name"] == "press":
        return "Fail", "Nonce-only S still PRESS; one-bind was not load-bearing."
    if not m["bindall_nonce"]["correct"] or m["bindall_nonce"]["action_name"] != "press":
        return "Fail", "Bind-all on nonce S did not PRESS; the distractor test is not load-bearing."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    train = m["train_s"]
    if train.get("found_press") or m["c_live"].get("found_press") or m["c_live"].get("found_tune"):
        return "Confound", "S still has an innate motor name as a copy token."
    if not train.get("found_push") or not train.get("found_cha") or not train.get("found_did_press"):
        return "Fail", "Never-wipe train never bound a page word to press+cha."
    if not train.get("found_argon") or not train.get("found_bind_push"):
        return "Fail", "Train S did not keep the distractor and bind the stream-first word."
    if train.get("found_bind_argon"):
        return "Fail", "Train S aliased the distractor hapax."
    if not m["c_live"].get("found_adjust") or not m["c_live"].get("found_chc") or not m["c_live"].get("found_did_tune"):
        return "Fail", "C life on dirty S never bound a page word to tune+chc."
    if not m["c_live"].get("found_alpha") or not m["c_live"].get("found_bind_adjust"):
        return "Fail", "C life did not keep the distractor and bind the stream-first word."
    if m["c_live"].get("found_bind_alpha"):
        return "Fail", "C life aliased the distractor hapax."
    if not m["c_live"].get("found_bind_push") or not m["c_live"].get("found_cha"):
        return "Fail", "C life clobbered train's bind=push+cha."
    tag = m.get("both_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "door") or _has_field(tag, "where"):
        return "Fail", "S restored filed tag names."
    if _has_field(tag, "n0") or _has_field(tag, "n1"):
        return "Fail", "S still has n* digit tags."
    return (
        "Store-works",
        "Never-wipe English train S still PRESS from push; C life on that S: A PRESS, C TUNE. Cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["train_s_probe"]["correct"] or m["train_s_probe"]["action_name"] != "press":
        return "Fail", "Shared return never-wipe train S was not PRESS."
    if not m["both_after_c"]["correct"] or m["both_after_c"]["action_name"] != "tune":
        return "Fail", "Shared return C on dirty S was not TUNE."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    if m["train_s"].get("found_press"):
        return "Confound", "S still has an innate motor name as a copy token."
    if m["nonce_control"]["action_name"] == "press":
        return "Fail", "Shared return nonce-only S still PRESS."
    return (
        "Store-works",
        "Shared return never-wipe English one-bind; dirty S two facts; cortex frozen.",
    )


def run_arm(
    *,
    arm: str,
    split: bool,
    run_dir: Path,
    w_a: Path,
    w_both: Path,
    w_clutter: Path,
    w_files: list[str],
    seed: int,
    n_train: int,
    train_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    work = run_dir / f"{arm}_train"
    dirs = {
        k: run_dir / f"{arm}_{k}"
        for k in (
            "untrained",
            "trains",
            "acc",
            "wipe",
            "menuctrl",
            "searchctrl",
            "writectrl",
            "clutter",
            "bindoff",
            "nonce",
            "bindall",
            "copyonly",
            "copydirty",
            "off",
            "empty",
        )
    }
    for d in (work, *dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    search0 = _head_fp(policy, "search")
    vname0 = _head_fp(policy, "vname")
    use0 = _head_fp(policy, "use")
    write0 = _head_fp(policy, "write")
    dummy = make(run_dir / f"{arm}_hash", None, policy, explore_epsilon=0.0)
    cortex0 = dummy.weight_hash()

    _, untrained_live, untrained_probe, untrained_foil, _ = _life_then_two_probes(
        make,
        dirs["untrained"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 99),
    )

    rewards, train_s = _train_keep(policy, w_a, work, n_train, train_seed, split=split, max_steps=max_steps)
    search1 = _head_fp(policy, "search")
    vname1 = _head_fp(policy, "vname")
    use1 = _head_fp(policy, "use")
    write1 = _head_fp(policy, "write")

    _copy_s(train_s, dirs["trains"])
    train_snap = _s_snapshot(dirs["trains"])
    train_s_probe, train_s_foil = _probe_s(
        dirs["trains"], policy, seed + 10, rng=np.random.default_rng(seed + 1)
    )

    _copy_s(train_s, dirs["acc"])
    acc_ag, c_live, both_a, both_c, both_tag = _c_life_on_s(
        dirs["acc"],
        w_both,
        policy,
        seed + 20,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
    )

    _, _, wipe_a_after, _, _, _, wipe_a, wipe_c, wipe_tag = _wipe_lives(
        dirs["wipe"],
        w_both,
        policy,
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 2),
        wipe_between=True,
    )
    _, _, menu_control, _, _ = _life_then_two_probes(
        lambda s, w, p, **kw: make(
            s,
            w,
            p,
            use_search_head=False,
            use_match_head=True,
            use_vname_head=False,
            use_prose_tokens=False,
            use_prose_ints=False,
            use_event_annotate=False,
            use_here_match=False,
            use_alias_bind=False,
            use_did_stamp=False,
            use_one_bind=False,
            **kw,
        ),
        dirs["menuctrl"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 4),
    )
    _, _, search_control, _, _ = _life_then_two_probes(
        make,
        dirs["searchctrl"],
        w_a,
        _copy_heads(policy, "use", "vname", "write"),
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 5),
    )
    _, _, write_control, _, _ = _life_then_two_probes(
        lambda s, w, p, **kw: make(
            s,
            w,
            p,
            use_event_annotate=False,
            use_here_match=False,
            use_alias_bind=False,
            use_did_stamp=False,
            use_one_bind=False,
            **kw,
        ),
        dirs["writectrl"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 7),
    )
    _, _, clutter_control, _, clutter_tag = _life_then_two_probes(
        make,
        dirs["clutter"],
        w_clutter,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 8),
    )
    bind_a, bind_c = _probe_s(
        dirs["trains"],
        policy,
        seed + 10,
        rng=np.random.default_rng(seed + 12),
        use_alias_bind=False,
        use_one_bind=False,
        use_did_stamp=True,
    )
    bind_control = bind_a
    bind_tag = train_snap["tag"]
    bind_live = _live_slice(train_snap)

    _write_nonce_s(dirs["trains"], dirs["nonce"], nonce="argon", station="cha")
    nonce_ag = make(dirs["nonce"], None, policy, explore_epsilon=0.0)
    nonce_ag.reset_rho()
    nonce_control = probe(nonce_ag, "probe_channel_a", seed + 10)
    _write_nonce_s(dirs["trains"], dirs["bindall"], nonce="argon", station="cha")
    bindall_ag = make(dirs["bindall"], None, policy, explore_epsilon=0.0, use_one_bind=False)
    bindall_ag.reset_rho()
    bindall_nonce = probe(bindall_ag, "probe_channel_a", seed + 10)

    _copy_s(train_s, dirs["copydirty"])
    copy_snap = _s_snapshot(dirs["copydirty"])
    copy_a, copy_c = _probe_s(
        dirs["copydirty"],
        policy,
        seed + 10,
        rng=np.random.default_rng(seed + 11),
        use_here_match=False,
    )
    copy_tag = copy_snap["tag"]
    copy_live = _live_slice(copy_snap)

    empty = make(dirs["empty"], None, policy, explore_epsilon=0.0)
    empty.reset_rho()
    empty_p = probe(empty, "probe_channel_a", seed + 10)
    empty_c = probe(empty, "probe_channel_c", seed + 20)
    off = make(
        dirs["off"], None, policy, enabled=False, explore_epsilon=0.5, rng=np.random.default_rng(seed + 9)
    )
    off.reset_rho()
    live_free(off, "experience_channel_a", seed + 10, max_steps=max_steps)
    off.reset_rho()
    disable_a = probe(off, "probe_channel_a", seed + 10)

    train_live = _live_slice(train_snap)
    metrics: dict[str, Any] = {
        "arm": arm,
        "trained_split": split,
        "trained_life": True,
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_event_annotate": dummy.use_event_annotate,
        "use_here_match": dummy.use_here_match,
        "use_alias_bind": dummy.use_alias_bind,
        "use_one_bind": dummy.use_one_bind,
        "use_commit_rare_only": dummy.use_commit_rare_only,
        "use_revise_head": dummy.use_revise_head,
        "use_commit_here_only": dummy.use_commit_here_only,
        "use_stamp_new_here": dummy.use_stamp_new_here,
        "use_block_here": dummy.use_block_here,
        "use_in_hand_new_here": dummy.use_in_hand_new_here,
        "use_find_novel": dummy.use_find_novel,
        "use_retry_novel": dummy.use_retry_novel,
        "use_local_alias": dummy.use_local_alias,
        "use_keep_steerer": dummy.use_keep_steerer,
        "use_count_search": dummy.use_count_search,
        "use_hyp_survive": dummy.use_hyp_survive,
        "use_bind_match": dummy.use_bind_match,
        "use_evidence": dummy.use_evidence,
        "use_compose": dummy.use_compose,
        "use_search_head": dummy.use_search_head,
        "use_match_head": dummy.use_match_head,
        "use_qname_head": dummy.use_qname_head,
        "use_vname_head": dummy.use_vname_head,
        "use_read": dummy.use_read,
        "use_prose_ints": dummy.use_prose_ints,
        "use_prose_tokens": dummy.use_prose_tokens,
        "domain": dummy.domain,
        "n_actions": dummy.n_actions,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == acc_ag.weight_hash(),
        "search_changed": search0 != search1,
        "vname_changed": vname0 != vname1,
        "use_changed": use0 != use1,
        "write_changed": write0 != write1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_both),
        "untrained_live": _live_slice(untrained_live),
        "untrained_probe": untrained_probe,
        "untrained_foil_c": untrained_foil,
        "a_live": train_live,
        "train_s": train_snap,
        "train_s_probe": train_s_probe,
        "train_s_foil": train_s_foil,
        "a_after_reset": train_s_probe,
        "a_foil_c": train_s_foil,
        "a_tag": train_snap["tag"],
        "c_live": _live_slice(c_live),
        "both_after_a": both_a,
        "both_after_c": both_c,
        "both_tag": both_tag,
        "c_after_reset": both_c,
        "c_foil_a": both_a,
        "wipe_ctrl_a": wipe_a,
        "wipe_ctrl_c": wipe_c,
        "wipe_ctrl_tag": wipe_tag,
        "wipe_ctrl_after_a": wipe_a_after,
        "menu_control": menu_control,
        "search_control": search_control,
        "write_control": write_control,
        "clutter_control": clutter_control,
        "clutter_tag": clutter_tag,
        "bind_control": bind_control,
        "bind_foil_c": bind_c,
        "bind_tag": bind_tag,
        "bind_live": bind_live,
        "nonce_control": nonce_control,
        "bindall_nonce": bindall_nonce,
        "copy_only": {
            "a_after_reset": copy_a,
            "a_foil_c": copy_c,
            "tag": copy_tag,
            "live": copy_live,
        },
        "empty_S": empty_p,
        "empty_S_c": empty_c,
        "disable_S_a": disable_a,
    }
    label, rationale = (classify_a if split else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_tm062(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_prose())
    w_files = sorted(p.name for p in w_both.glob("*.md"))
    a = run_arm(
        arm="A",
        split=True,
        run_dir=run_dir,
        w_a=w_a,
        w_both=w_both,
        w_clutter=w_clutter,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed,
        max_steps=max_steps,
    )
    b = run_arm(
        arm="B",
        split=False,
        run_dir=run_dir,
        w_a=w_a,
        w_both=w_both,
        w_clutter=w_clutter,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
        max_steps=max_steps,
    )
    out = {
        "version": "TM.0.6.2",
        "seed": seed,
        "n_train": n_train,
        "max_steps": max_steps,
        "world": "channel_dial",
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.6.2 A never-wipe English vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split never-wipe English | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Wipe-between: A / C | {a['wipe_ctrl_a']['action_name']} / {a['wipe_ctrl_c']['action_name']} | {b['wipe_ctrl_a']['action_name']} / {b['wipe_ctrl_c']['action_name']} |
| Bind-off A | {a['bind_control']['action_name']} | {b['bind_control']['action_name']} |
| Nonce-only A | {a['nonce_control']['action_name']} | {b['nonce_control']['action_name']} |
| Bind-all nonce A | {a['bindall_nonce']['action_name']} | {b['bindall_nonce']['action_name']} |
| Copy-only foil C | {a['copy_only']['a_foil_c']['action_name']} | {b['copy_only']['a_foil_c']['action_name']} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.6.2 never-wipe English life on one-bind")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm062(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {"A": m["A"]["classification"], "B": m["B"]["classification"], "world": m["world"], "run_dir": m["run_dir"]},
            indent=2,
        )
    )
    print(
        "A",
        "trainS",
        m["A"]["train_s_probe"]["action_name"],
        m["A"]["train_s_foil"]["action_name"],
        "dirtyC",
        m["A"]["both_after_a"]["action_name"],
        m["A"]["both_after_c"]["action_name"],
        "n",
        m["A"]["train_s"]["n"],
        m["A"]["a_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["train_s_probe"]["action_name"],
        m["B"]["both_after_c"]["action_name"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
