"""v11: two free lives, one S. Select the matching authored note; dump-all is the control."""

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

from experiments.run_v10 import live_free, probe, train_policy
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.symbols import GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, all_tag_notes, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v11"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    explore_epsilon: float = 0.0,
    retrieve_policy: str = "select",
    rng: np.random.Generator | None = None,
) -> ThreeMemoryAgent:
    world = TagLibrary(w_dir) if w_dir is not None else None
    return ThreeMemoryAgent(
        store_enabled=enabled,
        cortex_seed=1337,
        native=True,
        retrieve_policy=retrieve_policy,
        collect_mode="off",
        store=TagStore(s_dir, enabled=enabled),
        world=world,
        use_policy=policy,
        write_from_events=True,
        policy_epsilon=epsilon,
        policy_rng=rng,
        explore_epsilon=explore_epsilon,
    )


def _slim_live(lv: dict[str, Any]) -> dict[str, Any]:
    return {k: lv[k] for k in ("wrote", "opened", "files", "n_forced", "n_steps", "n_explored", "actions")}


def two_lives(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    explore_epsilon: float = 0.5,
    rng: np.random.Generator | None = None,
    max_steps: int = 32,
    seed: int = 12345,
) -> tuple[ThreeMemoryAgent, dict[str, Any], dict[str, Any]]:
    ag = make(
        s_dir,
        w_dir,
        policy,
        enabled=enabled,
        epsilon=0.0,
        explore_epsilon=explore_epsilon,
        rng=rng,
    )
    ag.reset_rho()
    red = live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
    ag.reset_rho()
    green = live_free(ag, "experience_green", seed + 20, max_steps=max_steps)
    ag.reset_rho()
    return ag, red, green


def classify(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"]:
        return "Confound", "Answer file was in W; this is copy-from-library, not a life."
    lives = (m["red_live"], m["green_live"], m["untrained_red"], m["untrained_green"])
    if any(lv["n_forced"] for lv in lives):
        return "Confound", "A forced curriculum ran; this is not a free life."
    if m["disable_S_red"]["correct"] or m["disable_S_green"]["correct"]:
        return "Confound", "disable-S still correct; fact leaked into policy/cortex."
    probes = (m["select_red"], m["select_green"], m["dump_red"], m["dump_green"])
    if any(p.get("explored") for p in probes):
        return "Confound", "Probe used exploration; greedy prior/S must decide."
    if not m["policy_changed"]:
        return "Fail", "Policy weights did not change (no write skill learned)."
    if not m["red_live"]["opened"] or not m["green_live"]["opened"]:
        return "Fail", "A free life never opened the door."
    files = set(m["s_files"])
    if f"{RED_FACT_ID}.tag" not in files or f"{GREEN_FACT_ID}.tag" not in files:
        return "Fail", "S did not contain both authored notes after two lives."
    if not m["select_red"]["correct"] or not m["select_green"]["correct"]:
        return "Fail", "Select missed the matching authored note."
    dump_same = m["dump_red"]["correct"] and m["dump_green"]["correct"]
    if dump_same:
        return "Fail", "Dump-all matched select on both probes; N is too small to show growth needs pick."
    if m["empty_S_red"]["correct"] or m["empty_S_green"]["correct"]:
        return "Fail", "Empty S still correct."
    return (
        "Store-works",
        "Two free lives authored two notes; select uses the match; dump-all does not.",
    )


def run_v11(seed: int = 12345, n_train: int = 400, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    work = run_dir / "train"
    s_both = run_dir / "S_both"
    s_select = run_dir / "S_select"
    s_dump = run_dir / "S_dump"
    s_off = run_dir / "S_off"
    s_empty = run_dir / "S_empty"
    s_untrained = run_dir / "S_untrained"
    write_tag_notes(w_clutter, all_tag_notes(include_red=False, include_green=False))
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))
    w_has_red = f"{RED_FACT_ID}.tag" in w_files
    w_has_green = f"{GREEN_FACT_ID}.tag" in w_files
    for d in (s_both, s_select, s_dump, s_off, s_empty, s_untrained, work):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    policy_hash0 = policy.weight_hash()
    dummy = make(run_dir / "empty_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_ag, un_red, un_green = two_lives(
        s_untrained, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 50), seed=seed, max_steps=max_steps
    )
    un_ag.world = None
    probe_un_red = probe(un_ag, "probe_red_with_key", seed + 10)
    probe_un_green = probe(un_ag, "probe_green", seed + 20)

    rewards = train_policy(policy, w_clutter, work, n_train, seed, max_steps)
    policy_hash1 = policy.weight_hash()

    shutil.rmtree(s_both, ignore_errors=True)
    s_both.mkdir()
    both_a, live_red, live_green = two_lives(
        s_both,
        w_clutter,
        policy,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
        seed=seed,
        max_steps=max_steps,
    )
    s_files = sorted(both_a.store.list_files()) if hasattr(both_a.store, "list_files") else []
    shutil.copytree(s_both, s_select, dirs_exist_ok=True)
    shutil.copytree(s_both, s_dump, dirs_exist_ok=True)

    sel = make(s_select, None, policy, explore_epsilon=0.0, retrieve_policy="select")
    sel.reset_rho()
    select_red = probe(sel, "probe_red_with_key", seed + 10)
    select_green = probe(sel, "probe_green", seed + 20)

    dump = make(s_dump, None, policy, explore_epsilon=0.0, retrieve_policy="dump")
    dump.reset_rho()
    dump_red = probe(dump, "probe_red_with_key", seed + 10)
    dump_green = probe(dump, "probe_green", seed + 20)

    empty = make(s_empty, None, policy, explore_epsilon=0.0)
    empty.reset_rho()
    empty_red = probe(empty, "probe_red_with_key", seed + 10)
    empty_green = probe(empty, "probe_green", seed + 20)

    off_a, off_red_live, off_green_live = two_lives(
        s_off,
        None,
        policy,
        enabled=False,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 2),
        seed=seed,
        max_steps=max_steps,
    )
    probe_off_red = probe(off_a, "probe_red_with_key", seed + 10)
    probe_off_green = probe(off_a, "probe_green", seed + 20)

    cortex1 = sel.weight_hash()
    red_text = (s_both / f"{RED_FACT_ID}.tag").read_text(encoding="utf-8") if (s_both / f"{RED_FACT_ID}.tag").is_file() else ""
    green_text = (
        (s_both / f"{GREEN_FACT_ID}.tag").read_text(encoding="utf-8") if (s_both / f"{GREEN_FACT_ID}.tag").is_file() else ""
    )

    metrics: dict[str, Any] = {
        "seed": seed,
        "n_train": n_train,
        "max_steps": max_steps,
        "train_return_mean_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "train_return_mean": float(np.mean(rewards)) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == cortex1,
        "policy_hash_before": policy_hash0,
        "policy_hash_after": policy_hash1,
        "policy_changed": policy_hash0 != policy_hash1,
        "w_files": w_files,
        "w_has_red": w_has_red,
        "w_has_green": w_has_green,
        "untrained_red": _slim_live(un_red),
        "untrained_green": _slim_live(un_green),
        "untrained_probe_red": probe_un_red,
        "untrained_probe_green": probe_un_green,
        "red_live": _slim_live(live_red),
        "green_live": _slim_live(live_green),
        "s_files": s_files,
        "red_tag": red_text,
        "green_tag": green_text,
        "select_red": select_red,
        "select_green": select_green,
        "dump_red": dump_red,
        "dump_green": dump_green,
        "empty_S_red": empty_red,
        "empty_S_green": empty_green,
        "disable_S_red": probe_off_red,
        "disable_S_green": probe_off_green,
        "disable_lives": {"red": _slim_live(off_red_live), "green": _slim_live(off_green_live)},
        "select_n_files": select_red.get("store_len"),
        "dump_n_hits_red": dump_red.get("store_len"),
    }
    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v11 select among authored notes

Classification: **{label}**

{rationale}

| Check | Result |
|-------|--------|
| Cortex unchanged | {metrics['cortex_unchanged']} |
| Policy changed | {metrics['policy_changed']} |
| n_forced red/green | {live_red['n_forced']} / {live_green['n_forced']} |
| S files after two lives | {s_files} |
| Select red | {select_red['correct']} ({select_red['action_name']}) |
| Select green | {select_green['correct']} ({select_green['action_name']}) |
| Dump-all red | {dump_red['correct']} ({dump_red['action_name']}) |
| Dump-all green | {dump_green['correct']} ({dump_green['action_name']}) |
| Empty S red / green | {empty_red['correct']} ({empty_red['action_name']}) / {empty_green['correct']} ({empty_green['action_name']}) |
| disable-S red / green | {probe_off_red['correct']} ({probe_off_red['action_name']}) / {probe_off_green['correct']} ({probe_off_green['action_name']}) |
| Train return last 50 | {metrics['train_return_mean_last50']:.2f} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v11 two lives, select vs dump-all")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v11(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                k: m[k]
                for k in (
                    "classification",
                    "rationale",
                    "run_dir",
                    "s_files",
                    "cortex_unchanged",
                    "policy_changed",
                )
            },
            indent=2,
        )
    )
    print("select red", m["select_red"]["action_name"], m["select_red"]["correct"])
    print("select green", m["select_green"]["action_name"], m["select_green"]["correct"])
    print("dump red", m["dump_red"]["action_name"], m["dump_red"]["correct"])
    print("dump green", m["dump_green"]["action_name"], m["dump_green"]["correct"])
    print("files", m["s_files"])
    print("n_forced", m["red_live"]["n_forced"], m["green_live"]["n_forced"])


if __name__ == "__main__":
    main()
