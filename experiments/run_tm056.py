"""TM.0.5.6: never-wipe train — dirty S survives the whole training life.

Same two-fact eval as TM.0.5.5, but train does not rmtree S each episode.
After train, probe that store (no fresh A life). Then C life on the same dirty S.
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
from experiments.run_tm052 import _s_has_token, live_free
from experiments.run_tm054 import _life_then_two_probes, make as _make054

def make(*args, use_commit_rare_only: bool = True, **kwargs):
    return _make054(*args, use_commit_rare_only=use_commit_rare_only, **kwargs)
from experiments.run_tm055 import (
    _accumulate_lives,
    _live_extra,
    _w_flags as _w_flags055,
    clutter_prose,
    wiki_prose,
)
from experiments.run_v22 import _tags
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm056"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags055(w_files, w_dir)
    flags["accumulate_s"] = True
    flags["train_wipe_s"] = False
    flags["eval_s_wiped_between"] = False
    flags["open_w"] = True
    flags["english_life"] = False
    return flags


def _copy_s(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _s_snapshot(s_dir: Path) -> dict[str, Any]:
    tag = _tags(s_dir) if s_dir.exists() else ""
    files = sorted(p.name for p in s_dir.glob("*.tag")) if s_dir.exists() else []
    return {
        "tag": tag,
        "files": files,
        "n": len(files),
        "found_press": _s_has_token(tag, "press"),
        "found_tune": _s_has_token(tag, "tune"),
        "found_krypton": _s_has_token(tag, "krypton"),
        "found_helium": _s_has_token(tag, "helium"),
        "found_cha": _s_has_token(tag, "cha"),
        "found_chc": _s_has_token(tag, "chc"),
    }


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
    c_live = _live_extra(live_free(ag, "experience_channel_c", seed, max_steps=max_steps))
    ag.world = None
    ag.reset_rho()
    both_a = probe(ag, "probe_channel_a", seed)
    ag.reset_rho()
    both_c = probe(ag, "probe_channel_c", seed)
    return ag, c_live, both_a, both_c, c_live["tag"]


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
        live = _live_extra(live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps))
        tr_life = list(ag.policy_traces)
        r_find = 1.0 if any(t.get("kind") == "search" and t.get("has_rare") for t in tr_life) else 0.0
        wrote = any(t.get("kind") == "write" and t.get("write") for t in tr_life)
        r_mark = (
            1.0
            if live["n_annotated"] > 0
            or (snap["found_press"] and snap["found_cha"] and wrote)
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
    from experiments.run_tm055 import classify_common as _c055

    # Reuse 0.5.5 checks, then invert the train-wipe slice.
    saved = m.get("train_wipe_s")
    m["train_wipe_s"] = True
    early = _c055(m)
    m["train_wipe_s"] = saved
    if early:
        return early
    if m.get("train_wipe_s"):
        return "Fail", "Train still wiped S every episode."
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
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    if not m["train_s"]["found_press"] or not m["train_s"]["found_cha"]:
        return "Fail", "Never-wipe train never stamped press+cha."
    if not m["c_live"]["found_tune"] or not m["c_live"].get("found_chc"):
        return "Fail", "C life on dirty S never stamped tune+chc."
    if not m["c_live"]["found_press"] or not m["c_live"].get("found_cha"):
        return "Fail", "C life clobbered train's press+cha."
    tag = m.get("both_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "door") or _has_field(tag, "where"):
        return "Fail", "S restored filed tag names."
    if _has_field(tag, "n0") or _has_field(tag, "n1"):
        return "Fail", "S still has n* digit tags."
    return (
        "Store-works",
        "Never-wipe train S still PRESS; C life on that dirty S: A PRESS, C TUNE. Cortex frozen.",
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
    return (
        "Store-works",
        "Shared return never-wipe train; dirty S two facts; cortex frozen.",
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
    copy_live = {
        "opened": False,
        "files": copy_snap["files"],
        "n_forced": 0,
        "n_steps": 0,
        "n_explored": 0,
        "n_annotated": 0,
        "found_press": copy_snap["found_press"],
        "found_tune": copy_snap["found_tune"],
        "found_krypton": copy_snap["found_krypton"],
        "found_helium": copy_snap["found_helium"],
        "found_cha": copy_snap["found_cha"],
        "found_chc": copy_snap["found_chc"],
        "actions": [],
    }

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
    train_live = {
        "opened": False,
        "files": train_snap["files"],
        "n_forced": 0,
        "n_steps": 0,
        "n_explored": 0,
        "n_annotated": 0,
        "found_press": train_snap["found_press"],
        "found_tune": train_snap["found_tune"],
        "found_krypton": train_snap["found_krypton"],
        "found_helium": train_snap["found_helium"],
        "found_cha": train_snap["found_cha"],
        "found_chc": train_snap["found_chc"],
        "actions": [],
    }
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
        "a_live": train_live,
        "train_s": train_snap,
        "train_s_probe": train_s_probe,
        "train_s_foil": train_s_foil,
        "a_after_reset": train_s_probe,
        "a_foil_c": train_s_foil,
        "a_tag": train_snap["tag"],
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


def run_tm056(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.5.6",
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
        f"""# TM.0.5.6 A never-wipe train vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split never-wipe | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Wipe-between: A / C | {a['wipe_ctrl_a']['action_name']} / {a['wipe_ctrl_c']['action_name']} | {b['wipe_ctrl_a']['action_name']} / {b['wipe_ctrl_c']['action_name']} |
| Copy-only foil C | {a['copy_only']['a_foil_c']['action_name']} | {b['copy_only']['a_foil_c']['action_name']} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.6 never-wipe train — dirty S survives training")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm056(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
