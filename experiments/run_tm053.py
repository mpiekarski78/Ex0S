"""TM.0.5.3: use the fact, not pick-a-motor.

Same S after A life: probe A PRESS, probe C HOLD.
Copy-only (no here-match) fires PRESS on C — Fail if that is the trained agent.
Not Open W, not English, not shared-return rescue, not accumulate S, not domain= drop.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from experiments.run_tm052 import (
    _s_has_token,
    _w_flags as _w_flags052,
    clutter_prose,
    live_free,
    wiki_prose,
)
from three_memory.agent import ThreeMemoryAgent
from three_memory.dial_env import STATION_NAMES, DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import ProseLibrary, TagStore, prose_tokens, write_prose_notes

_MOTOR_WORDS = {a.name.lower() for a in DialAction}
_STATION_WORDS = set(STATION_NAMES.values())


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm053"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags052(w_files, w_dir)
    body_toks = set(flags.get("w_a_tokens") or [])
    texts = "".join(p.read_text(encoding="utf-8") for p in w_dir.glob("*.md"))
    flags["w_has_station_name"] = bool(prose_tokens(texts) & _STATION_WORDS)
    flags["w_a_has_station"] = bool(set(body_toks) & _STATION_WORDS)
    flags["has_code_in_search"] = True
    flags["domain_switch"] = True
    flags["open_w"] = False
    flags["english_life"] = False
    flags["accumulate_s"] = False
    return flags


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    explore_epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    force_use: bool = False,
    force_write: bool = False,
    use_search_head: bool = True,
    use_match_head: bool = False,
    use_vname_head: bool = True,
    record_search_on_explore: bool = True,
    use_prose_tokens: bool = True,
    use_prose_ints: bool = False,
    use_event_annotate: bool = True,
    use_here_match: bool = True,
) -> ThreeMemoryAgent:
    world = ProseLibrary(w_dir) if w_dir is not None else None
    return ThreeMemoryAgent(
        store_enabled=enabled,
        cortex_seed=1337,
        native=True,
        domain="dial",
        retrieve_policy="select",
        collect_mode="commit",
        store=TagStore(s_dir, enabled=enabled),
        world=world,
        use_policy=policy,
        write_from_events=False,
        policy_epsilon=epsilon,
        policy_rng=rng,
        explore_epsilon=explore_epsilon,
        use_read=True,
        use_search_head=use_search_head,
        use_match_head=use_match_head,
        use_vname_head=use_vname_head,
        force_use=force_use,
        force_write=force_write,
        record_search_on_explore=record_search_on_explore,
        use_prose_ints=use_prose_ints,
        use_prose_tokens=use_prose_tokens,
        use_event_annotate=use_event_annotate,
        use_here_match=use_here_match,
    )


def _live_extra(live: dict[str, Any]) -> dict[str, Any]:
    tag = live.get("tag") or ""
    live["found_cha"] = _s_has_token(tag, "cha")
    live["found_chc"] = _s_has_token(tag, "chc")
    return live


def _life_then_two_probes(
    make_fn,
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    life_scenario: str,
    here_probe: str,
    foil_probe: str,
    seed: int,
    *,
    max_steps: int,
    explore_epsilon: float,
    rng: np.random.Generator | None = None,
    **make_kw: Any,
) -> tuple[ThreeMemoryAgent, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make_fn(s_dir, w_dir, policy, explore_epsilon=explore_epsilon, rng=rng, **make_kw)
    ag.reset_rho()
    ag.policy_traces = []
    live = _live_extra(live_free(ag, life_scenario, seed, max_steps=max_steps))
    tag = live["tag"]
    ag.world = None
    ag.reset_rho()
    p_here = probe(ag, here_probe, seed)
    ag.reset_rho()
    p_foil = probe(ag, foil_probe, seed)
    return ag, live, p_here, p_foil, tag


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
    if not m["w_has_p99"]:
        return "Confound", "W must have useful prose p99.md."
    if m.get("w_body_ints") or m.get("w_a_ints"):
        return "Confound", "Answer integers were restored in W."
    if m.get("w_has_motor_name") or any(t in m.get("w_a_tokens", []) for t in _MOTOR_WORDS):
        return "Confound", "Unread W still names an innate motor."
    if m.get("w_has_station_name") or m.get("w_a_has_station"):
        return "Confound", "Unread W still names an innate station."
    if "krypton" not in m.get("w_a_tokens", []):
        return "Confound", "Useful page must still be a rare-word scrap."
    if m.get("open_w") or m.get("english_life"):
        return "Confound", "Open W / English life was smuggled into this jump."
    if m.get("accumulate_s"):
        return "Confound", "S was accumulated across lives."
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
        for p in (m["untrained_probe"], m["a_after_reset"], m["a_foil_c"], m["c_after_reset"], m["c_foil_a"])
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
    if m["untrained_live"].get("found_press") or m["untrained_live"].get("n_annotated", 0):
        return "Fail", "Untrained already annotated a motor onto S."
    if not m["a_after_reset"]["correct"] or m["a_after_reset"]["action_name"] != "press":
        return "Fail", "After A life, probe A was not PRESS."
    if m["a_foil_c"]["action_name"] == "press" or m["a_foil_c"]["correct"]:
        return "Fail", "A's press stamp still fired on channel C (pick-a-motor)."
    if m["a_foil_c"]["action_name"] != "hold":
        return "Fail", "A's S on channel C was not HOLD."
    if not m["c_after_reset"]["correct"] or m["c_after_reset"]["action_name"] != "tune":
        return "Fail", "After C life, probe C was not TUNE."
    if m["c_foil_a"]["action_name"] == "tune" or m["c_foil_a"]["action_name"] == "press":
        return "Fail", "C's stamp still fired on channel A."
    if m["c_foil_a"]["action_name"] != "hold":
        return "Fail", "C's S on channel A was not HOLD."
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
    if not (
        m["copy_only"]["a_after_reset"]["action_name"] == "press"
        and m["copy_only"]["a_foil_c"]["action_name"] == "press"
    ):
        return "Fail", "Copy-only control did not fire PRESS on C; here-match was not load-bearing."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    if not m["a_live"]["found_press"] or not m["a_live"].get("found_cha"):
        return "Fail", "Free A life never stamped press+cha."
    if not m["c_live"]["found_tune"] or not m["c_live"].get("found_chc"):
        return "Fail", "Held-out C life never stamped tune+chc."
    tag = m.get("a_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "door") or _has_field(tag, "where"):
        return "Fail", "S restored filed tag names."
    if _has_field(tag, "n0") or _has_field(tag, "n1"):
        return "Fail", "S still has n* digit tags."
    return (
        "Store-works",
        "Same S: A PRESS, C HOLD. Fact is about this station, not a global motor. Cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["a_after_reset"]["correct"] or m["a_after_reset"]["action_name"] != "press":
        return "Fail", "Shared return did not solve A after a free life."
    if m["a_foil_c"]["action_name"] == "press":
        return "Fail", "Shared return still fired A's press on C."
    if not m["c_after_reset"]["correct"] or m["c_after_reset"]["action_name"] != "tune":
        return "Fail", "Shared return C was not TUNE."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return on use-the-fact; cortex frozen; foil C HOLD.",
    )


def _train(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    *,
    split: bool,
    max_steps: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_m = b_u = 0.0
    s_dir = work / "ep"
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_dir, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live = _live_extra(live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps))
        r_find = 1.0 if live["found_krypton"] else 0.0
        r_mark = 1.0 if live["found_press"] and live["found_cha"] else 0.0
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
    return rewards


def run_arm(
    *,
    arm: str,
    split: bool,
    run_dir: Path,
    w_a: Path,
    w_c: Path,
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
            "a",
            "c",
            "menuctrl",
            "searchctrl",
            "writectrl",
            "clutter",
            "copyonly",
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

    rewards = _train(policy, w_a, work, n_train, train_seed, split=split, max_steps=max_steps)
    search1 = _head_fp(policy, "search")
    vname1 = _head_fp(policy, "vname")
    use1 = _head_fp(policy, "use")
    write1 = _head_fp(policy, "write")

    a_ag, a_live, a_after, a_foil, a_tag = _life_then_two_probes(
        make,
        dirs["a"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
    )
    _, c_live, c_after, c_foil, c_tag = _life_then_two_probes(
        make,
        dirs["c"],
        w_c,
        policy,
        "experience_channel_c",
        "probe_channel_c",
        "probe_channel_a",
        seed + 20,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 3),
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
        lambda s, w, p, **kw: make(s, w, p, use_event_annotate=False, use_here_match=False, **kw),
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
    _, copy_live, copy_a, copy_c, copy_tag = _life_then_two_probes(
        lambda s, w, p, **kw: make(s, w, p, use_here_match=False, **kw),
        dirs["copyonly"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        "probe_channel_c",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 11),
    )

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

    live_keys = (
        "opened",
        "files",
        "n_forced",
        "n_steps",
        "n_explored",
        "n_annotated",
        "found_press",
        "found_tune",
        "found_krypton",
        "found_helium",
        "found_cha",
        "found_chc",
        "actions",
    )
    metrics: dict[str, Any] = {
        "arm": arm,
        "trained_split": split,
        "trained_life": True,
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_event_annotate": dummy.use_event_annotate,
        "use_here_match": dummy.use_here_match,
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
        "cortex_unchanged": cortex0 == a_ag.weight_hash(),
        "search_changed": search0 != search1,
        "vname_changed": vname0 != vname1,
        "use_changed": use0 != use1,
        "write_changed": write0 != write1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_a),
        "untrained_live": {k: untrained_live[k] for k in live_keys},
        "untrained_probe": untrained_probe,
        "untrained_foil_c": untrained_foil,
        "a_live": {k: a_live[k] for k in live_keys},
        "a_after_reset": a_after,
        "a_foil_c": a_foil,
        "a_tag": a_tag,
        "c_live": {k: c_live[k] for k in live_keys},
        "c_after_reset": c_after,
        "c_foil_a": c_foil,
        "c_tag": c_tag,
        "menu_control": menu_control,
        "search_control": search_control,
        "write_control": write_control,
        "clutter_control": clutter_control,
        "clutter_tag": clutter_tag,
        "copy_only": {
            "a_after_reset": copy_a,
            "a_foil_c": copy_c,
            "tag": copy_tag,
            "live": {k: copy_live[k] for k in live_keys},
        },
        "empty_S": empty_p,
        "empty_S_c": empty_c,
        "disable_S_a": disable_a,
    }
    label, rationale = (classify_a if split else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_tm053(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_c = run_dir / "W_c"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_c, wiki_prose(include_c=True))
    write_prose_notes(w_clutter, clutter_prose())
    w_files = sorted(p.name for p in w_a.glob("*.md"))
    a = run_arm(
        arm="A",
        split=True,
        run_dir=run_dir,
        w_a=w_a,
        w_c=w_c,
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
        w_c=w_c,
        w_clutter=w_clutter,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
        max_steps=max_steps,
    )
    out = {
        "version": "TM.0.5.3",
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
        f"""# TM.0.5.3 A use-the-fact vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split here-match | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After A life: probe A / foil C | {a['a_after_reset']['action_name']} / {a['a_foil_c']['action_name']} | {b['a_after_reset']['action_name']} / {b['a_foil_c']['action_name']} |
| After C life: probe C / foil A | {a['c_after_reset']['action_name']} / {a['c_foil_a']['action_name']} | {b['c_after_reset']['action_name']} / {b['c_foil_a']['action_name']} |
| Copy-only foil C | {a['copy_only']['a_foil_c']['action_name']} | {b['copy_only']['a_foil_c']['action_name']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.3 use the fact, not pick-a-motor")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm053(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {"A": m["A"]["classification"], "B": m["B"]["classification"], "world": m["world"], "run_dir": m["run_dir"]},
            indent=2,
        )
    )
    print(
        "A",
        m["A"]["a_after_reset"]["action_name"],
        m["A"]["a_foil_c"]["action_name"],
        m["A"]["c_after_reset"]["action_name"],
        m["A"]["c_foil_a"]["action_name"],
        m["A"]["a_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["a_after_reset"]["action_name"],
        m["B"]["a_foil_c"]["action_name"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
