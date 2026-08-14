"""v9: boxed policy learns when to author a note from a life; W has no answer."""

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

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, all_tag_notes, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v9"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    world = KeyDoorWorld(seed=seed)
    obs = world.reset(scenario)
    action, meta = agent.act(obs, update_rho=False)
    result = world.step(action, scenario)
    if scenario == "probe_red_with_key":
        correct = action == Action.USE_KEY and bool(result.info.get("opened"))
    elif scenario == "probe_green":
        correct = action == Action.WAIT and bool(result.info.get("opened"))
    else:
        correct = False
    return {
        "scenario": scenario,
        "action": int(action),
        "action_name": Action(action).name.lower(),
        "correct": correct,
        "opened": bool(result.info.get("opened")),
        "store_len": len(agent.store),
        "policy": meta.get("policy") or {},
        "files": agent.store.list_files() if hasattr(agent.store, "list_files") else [],
    }


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ThreeMemoryAgent:
    world = TagLibrary(w_dir) if w_dir is not None else None
    return ThreeMemoryAgent(
        store_enabled=enabled,
        cortex_seed=1337,
        native=True,
        retrieve_policy="select",
        collect_mode="off",
        store=TagStore(s_dir, enabled=enabled),
        world=world,
        use_policy=policy,
        write_from_events=True,
        policy_epsilon=epsilon,
        policy_rng=rng,
    )


def live(
    agent: ThreeMemoryAgent,
    scenario: str,
    forced: tuple[Action, ...],
    seed: int,
) -> dict[str, Any]:
    """Scripted life so a door-opening event fires. Policy only chooses whether to write."""
    world = KeyDoorWorld(seed=seed)
    obs = world.reset(scenario)
    log: list[dict[str, Any]] = []
    wrote = False
    for act in forced:
        result = world.step(int(act), scenario)
        w = agent.observe_outcome(result.obs, result.success, result.info)
        log.append(
            {
                "action": int(act),
                "success": result.success,
                "opened": bool(result.info.get("opened")),
                "wrote": w["wrote"],
                "policy": w.get("policy") or {},
            }
        )
        wrote = wrote or bool(w["wrote"])
        obs = result.obs
        if result.done:
            break
    files = agent.store.list_files() if hasattr(agent.store, "list_files") else []
    return {"wrote": wrote, "files": files, "n_forced": len(log), "steps": log}


def train_policy(policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_clutter, policy, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live(ag, "experience_teach", (Action.OPEN, Action.PICK_KEY, Action.USE_KEY), seed + 10)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r
        policy.update(ag.policy_traces, r - baseline)
        rewards.append(r)
    return rewards


def classify(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"]:
        return "Confound", "Answer file was in W; this is copy-from-library, not a life."
    if m["disable_S_after_reset"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked into policy/cortex."
    if not m["policy_changed"]:
        return "Fail", "Policy weights did not change (no write skill learned)."
    if not m["red_authored"]:
        return "Fail", "Trained policy did not author d0.tag from the red life."
    red_ok = m["red_after_reset"]["correct"]
    green_ok = m["green_heldout_after_reset"]["correct"]
    empty_ok = m["empty_S"]["correct"]
    if red_ok and green_ok and not empty_ok:
        return (
            "Store-works",
            "Policy learned when to author a note from a door-opening; cortex frozen; held-out green authored, not copied.",
        )
    if red_ok and not green_ok:
        return "Fail", "Red worked but held-out green life did not; policy/template learned that door."
    if not red_ok:
        return "Fail", "Authored red note did not steer the probe after ρ reset."
    return "Fail", "Empty S still correct or other control failed."


def run_v9(seed: int = 12345, n_train: int = 400) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    work = run_dir / "train"
    s_red = run_dir / "S_red"
    s_green = run_dir / "S_green"
    s_off = run_dir / "S_off"
    s_empty = run_dir / "S_empty"
    s_reload = run_dir / "S_reload"
    write_tag_notes(w_clutter, all_tag_notes(include_red=False, include_green=False))
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))
    w_has_red = f"{RED_FACT_ID}.tag" in w_files
    w_has_green = f"{GREEN_FACT_ID}.tag" in w_files
    for d in (s_red, s_green, s_off, s_empty, s_reload, work):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    policy_hash0 = policy.weight_hash()
    dummy = make(run_dir / "empty_hash", None, policy)
    cortex0 = dummy.weight_hash()

    untrained = make(run_dir / "S_untrained", w_clutter, policy, epsilon=0.0)
    untrained.reset_rho()
    live_untrained = live(
        untrained, "experience_teach", (Action.OPEN, Action.PICK_KEY, Action.USE_KEY), seed + 10
    )
    untrained.world = None
    untrained.reset_rho()
    probe_untrained = probe(untrained, "probe_red_with_key", seed + 10)

    rewards = train_policy(policy, w_clutter, work, n_train, seed)
    policy_hash1 = policy.weight_hash()

    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    red_a = make(s_red, w_clutter, policy, epsilon=0.0)
    red_a.reset_rho()
    live_red = live(
        red_a, "experience_teach", (Action.OPEN, Action.PICK_KEY, Action.USE_KEY), seed + 10
    )
    shutil.copytree(s_red, s_reload, dirs_exist_ok=True)
    red_only = make(s_reload, None, policy, epsilon=0.0)
    red_only.reset_rho()
    probe_red = probe(red_only, "probe_red_with_key", seed + 10)

    empty = make(s_empty, None, policy, epsilon=0.0)
    empty.reset_rho()
    probe_empty = probe(empty, "probe_red_with_key", seed + 10)

    off = make(s_off, None, policy, enabled=False, epsilon=0.0)
    off.reset_rho()
    live(off, "experience_teach", (Action.OPEN, Action.PICK_KEY, Action.USE_KEY), seed + 10)
    off.reset_rho()
    probe_off = probe(off, "probe_red_with_key", seed + 10)

    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    green_a = make(s_green, w_clutter, policy, epsilon=0.0)
    green_a.reset_rho()
    live_green = live(green_a, "experience_green", (Action.OPEN, Action.WAIT), seed + 20)
    s_green_reload = run_dir / "S_green_reload"
    shutil.copytree(s_green, s_green_reload, dirs_exist_ok=True)
    green_only = make(s_green_reload, None, policy, epsilon=0.0)
    green_only.reset_rho()
    probe_green = probe(green_only, "probe_green", seed + 20)

    cortex1 = red_only.weight_hash()
    red_text = ""
    red_path = s_red / f"{RED_FACT_ID}.tag"
    if red_path.is_file():
        red_text = red_path.read_text(encoding="utf-8")
    green_text = ""
    green_path = s_green / f"{GREEN_FACT_ID}.tag"
    if green_path.is_file():
        green_text = green_path.read_text(encoding="utf-8")

    metrics: dict[str, Any] = {
        "seed": seed,
        "n_train": n_train,
        "train_return_mean_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "train_return_mean": float(np.mean(rewards)) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == cortex1,
        "policy_hash_before": policy_hash0,
        "policy_hash_after": policy_hash1,
        "policy_changed": policy_hash0 != policy_hash1,
        "policy_n_updates": policy.n_updates,
        "w_files": w_files,
        "w_has_red": w_has_red,
        "w_has_green": w_has_green,
        "untrained_live": {k: live_untrained[k] for k in ("wrote", "files", "n_forced")},
        "untrained_after_reset": probe_untrained,
        "red_live": {k: live_red[k] for k in ("wrote", "files", "n_forced")},
        "red_authored": f"{RED_FACT_ID}.tag" in live_red["files"],
        "red_tag": red_text,
        "red_after_reset": probe_red,
        "empty_S": probe_empty,
        "disable_S_after_reset": probe_off,
        "green_live": {k: live_green[k] for k in ("wrote", "files", "n_forced")},
        "green_authored": f"{GREEN_FACT_ID}.tag" in live_green["files"],
        "green_tag": green_text,
        "green_heldout_after_reset": probe_green,
        "english_in_red_tag": any(s in red_text.lower() for s in ("open", "key", "love", "note", "red")),
    }
    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v9 write from a life

Classification: **{label}**

{rationale}

| Check | Result |
|-------|--------|
| Cortex unchanged | {metrics['cortex_unchanged']} |
| Policy changed | {metrics['policy_changed']} |
| W has d0.tag / d2.tag | {w_has_red} / {w_has_green} |
| Untrained after ρ reset | {probe_untrained['correct']} ({probe_untrained['action_name']}) |
| Red authored from events | {metrics['red_authored']} ({live_red['files']}) |
| Red after ρ reset, W unmounted | {probe_red['correct']} ({probe_red['action_name']}) |
| Empty S | {probe_empty['correct']} ({probe_empty['action_name']}) |
| disable-S | {probe_off['correct']} ({probe_off['action_name']}) |
| Held-out green authored | {metrics['green_authored']} ({live_green['files']}) |
| Green after ρ reset, W unmounted | {probe_green['correct']} ({probe_green['action_name']}) |
| Train return last 50 | {metrics['train_return_mean_last50']:.2f} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v9 boxed write-from-life, frozen cortex")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    args = p.parse_args()
    m = run_v9(seed=args.seed, n_train=args.n_train)
    print(
        json.dumps(
            {
                k: m[k]
                for k in (
                    "classification",
                    "rationale",
                    "run_dir",
                    "policy_changed",
                    "cortex_unchanged",
                    "w_has_red",
                    "red_authored",
                    "green_authored",
                )
            },
            indent=2,
        )
    )
    print("untrained", m["untrained_after_reset"]["action_name"], m["untrained_after_reset"]["correct"])
    print("red after reset", m["red_after_reset"]["action_name"], m["red_after_reset"]["correct"], m["red_tag"].strip())
    print("disable-S", m["disable_S_after_reset"]["action_name"], m["disable_S_after_reset"]["correct"])
    print("green after reset", m["green_heldout_after_reset"]["action_name"], m["green_heldout_after_reset"]["correct"], m["green_tag"].strip())
    print("empty S", m["empty_S"]["action_name"], m["empty_S"]["correct"])


if __name__ == "__main__":
    main()
