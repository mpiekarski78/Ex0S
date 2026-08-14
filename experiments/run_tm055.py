"""TM.0.5.5: accumulate S — two lives, same store, both facts remain.

One unread W with both useful pages. Life A then life C, no rmtree.
After both: probe A PRESS, probe C TUNE. Train still wipes.
Not English, not shared-return rescue, not domain= drop, not dropping has_code.
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
from experiments.run_tm052 import live_free
from experiments.run_tm054 import (
    _MOTOR_WORDS,
    _STATION_WORDS,
    _life_then_two_probes,
    _live_extra,
    _n_paragraphs,
    _rare_words,
    _train,
    _w_flags as _w_flags054,
    clutter_prose,
    make,
    wiki_prose,
)
from three_memory.policy import UsePolicy
from three_memory.tag_store import prose_tokens, write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm055"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags054(w_files, w_dir)
    flags["accumulate_s"] = True
    flags["train_wipe_s"] = True
    flags["eval_s_wiped_between"] = False
    flags["w_has_p98"] = "p98.md" in w_files
    flags["open_w"] = True
    flags["english_life"] = False
    both = wiki_prose(include_a=True, include_c=True)
    flags["w_min_paragraphs"] = min((_n_paragraphs(b) for _, b in both), default=0)
    rare = _rare_words(both)
    flags["w_clutter_has_rare"] = any(rare[n] for n in rare if n.startswith("c"))
    flags["w_useful_has_rare"] = bool(rare.get("p99.md")) and bool(rare.get("p98.md"))
    texts = "".join(p.read_text(encoding="utf-8") for p in w_dir.glob("*.md"))
    flags["w_has_helium"] = "helium" in prose_tokens(texts)
    flags["w_has_krypton"] = "krypton" in prose_tokens(texts)
    return flags


def _probes_restore_world(ag, here: str, foil: str, seed: int):
    world = ag.world
    ag.world = None
    ag.reset_rho()
    p_here = probe(ag, here, seed)
    ag.reset_rho()
    p_foil = probe(ag, foil, seed)
    ag.world = world
    ag.reset_rho()
    return p_here, p_foil


def _accumulate_lives(
    s_dir: Path,
    w_dir: Path,
    policy: UsePolicy,
    seed: int,
    *,
    max_steps: int,
    explore_epsilon: float,
    rng: np.random.Generator | None = None,
    wipe_between: bool = False,
    **make_kw: Any,
):
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make(s_dir, w_dir, policy, explore_epsilon=explore_epsilon, rng=rng, **make_kw)
    ag.reset_rho()
    ag.policy_traces = []
    a_live = _live_extra(live_free(ag, "experience_channel_a", seed, max_steps=max_steps))
    a_tag = a_live["tag"]
    a_after, a_foil = _probes_restore_world(ag, "probe_channel_a", "probe_channel_c", seed)
    if wipe_between:
        shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_dir, policy, explore_epsilon=explore_epsilon, rng=rng, **make_kw)
        ag.reset_rho()
        ag.policy_traces = []
    c_live = _live_extra(live_free(ag, "experience_channel_c", seed + 10, max_steps=max_steps))
    both_tag = c_live["tag"]
    ag.world = None
    ag.reset_rho()
    both_a = probe(ag, "probe_channel_a", seed)
    ag.reset_rho()
    both_c = probe(ag, "probe_channel_c", seed)
    return ag, a_live, a_after, a_foil, a_tag, c_live, both_a, both_c, both_tag


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
    if "krypton" not in m.get("w_a_tokens", []) or not m.get("w_has_helium"):
        return "Confound", "Both useful rare-word scraps must be in eval W."
    if m.get("english_life"):
        return "Confound", "English life was smuggled into this jump."
    if not m.get("open_w"):
        return "Fail", "Open W was frozen off."
    if m.get("w_clutter_cloned") or (m.get("w_n_distinct_clutter") or 0) < 11:
        return "Confound", "W clutter is still cloned one-liners."
    if (m.get("w_min_paragraphs") or 0) < 2:
        return "Confound", "W pages are not multi-paragraph documents."
    if m.get("w_clutter_has_rare"):
        return "Confound", "Clutter pages are rare-word pages; search is not load-bearing."
    if not m.get("w_useful_has_rare"):
        return "Confound", "A useful page has no rare word."
    if not m.get("accumulate_s") or m.get("eval_s_wiped_between"):
        return "Fail", "S was still wiped between eval lives."
    if not m.get("train_wipe_s"):
        return "Confound", "Train also stopped wiping (not this slice)."
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
        for p in (
            m["untrained_probe"],
            m["a_after_reset"],
            m["a_foil_c"],
            m["both_after_a"],
            m["both_after_c"],
            m["wipe_ctrl_a"],
            m["wipe_ctrl_c"],
        )
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
    if not m["both_after_a"]["correct"] or m["both_after_a"]["action_name"] != "press":
        return "Fail", "After both lives, probe A was not PRESS (first fact lost)."
    if not m["both_after_c"]["correct"] or m["both_after_c"]["action_name"] != "tune":
        return "Fail", "After both lives, probe C was not TUNE."
    if m["wipe_ctrl_a"]["action_name"] == "press" or m["wipe_ctrl_a"]["correct"]:
        return "Fail", "Wipe-between still kept A's PRESS; accumulate was not load-bearing."
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
        return "Fail", "C life never stamped tune+chc."
    if not m["c_live"]["found_press"] or not m["c_live"].get("found_cha"):
        return "Fail", "C life clobbered A's press+cha."
    tag = m.get("both_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "door") or _has_field(tag, "where"):
        return "Fail", "S restored filed tag names."
    if _has_field(tag, "n0") or _has_field(tag, "n1"):
        return "Fail", "S still has n* digit tags."
    return (
        "Store-works",
        "Same S after two lives: A PRESS, C TUNE. First fact kept. Cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["both_after_a"]["correct"] or m["both_after_a"]["action_name"] != "press":
        return "Fail", "Shared return did not keep A PRESS after two lives."
    if not m["both_after_c"]["correct"] or m["both_after_c"]["action_name"] != "tune":
        return "Fail", "Shared return C was not TUNE after two lives."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return accumulated two facts; cortex frozen.",
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
            "acc",
            "wipe",
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
        w_both,
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

    acc_ag, a_live, a_after, a_foil, a_tag, c_live, both_a, both_c, both_tag = _accumulate_lives(
        dirs["acc"],
        w_both,
        policy,
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
        wipe_between=False,
    )
    _, _, wipe_a_after, _, _, _, wipe_a, wipe_c, wipe_tag = _accumulate_lives(
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
        "cortex_unchanged": cortex0 == acc_ag.weight_hash(),
        "search_changed": search0 != search1,
        "vname_changed": vname0 != vname1,
        "use_changed": use0 != use1,
        "write_changed": write0 != write1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_both),
        "untrained_live": {k: untrained_live[k] for k in live_keys},
        "untrained_probe": untrained_probe,
        "untrained_foil_c": untrained_foil,
        "a_live": {k: a_live[k] for k in live_keys},
        "a_after_reset": a_after,
        "a_foil_c": a_foil,
        "a_tag": a_tag,
        "c_live": {k: c_live[k] for k in live_keys},
        "both_after_a": both_a,
        "both_after_c": both_c,
        "both_tag": both_tag,
        "wipe_ctrl_a": wipe_a,
        "wipe_ctrl_c": wipe_c,
        "wipe_ctrl_tag": wipe_tag,
        "wipe_ctrl_after_a": wipe_a_after,
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


def run_tm055(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.5.5",
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
        f"""# TM.0.5.5 A accumulate S vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split accumulate | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After A life: A / foil C | {a['a_after_reset']['action_name']} / {a['a_foil_c']['action_name']} | {b['a_after_reset']['action_name']} / {b['a_foil_c']['action_name']} |
| After both lives: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Wipe-between: A / C | {a['wipe_ctrl_a']['action_name']} / {a['wipe_ctrl_c']['action_name']} | {b['wipe_ctrl_a']['action_name']} / {b['wipe_ctrl_c']['action_name']} |
| Copy-only foil C | {a['copy_only']['a_foil_c']['action_name']} | {b['copy_only']['a_foil_c']['action_name']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.5 accumulate S — two lives, same store")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm055(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {"A": m["A"]["classification"], "B": m["B"]["classification"], "world": m["world"], "run_dir": m["run_dir"]},
            indent=2,
        )
    )
    print(
        "A",
        "afterA",
        m["A"]["a_after_reset"]["action_name"],
        m["A"]["a_foil_c"]["action_name"],
        "both",
        m["A"]["both_after_a"]["action_name"],
        m["A"]["both_after_c"]["action_name"],
        "wipe",
        m["A"]["wipe_ctrl_a"]["action_name"],
        m["A"]["wipe_ctrl_c"]["action_name"],
    )
    print(
        "B",
        m["B"]["both_after_a"]["action_name"],
        m["B"]["both_after_c"]["action_name"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
