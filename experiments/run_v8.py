"""v8: boxed use-policy may learn; cortex frozen. Facts stay in .tag files."""

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
    d = REPO_ROOT / "runs" / f"{stamp}_v8"
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
    elif scenario == "probe_blue":
        correct = action == Action.OPEN and bool(result.info.get("opened"))
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
        collect_mode="policy",
        store=TagStore(s_dir, enabled=enabled),
        world=world,
        use_policy=policy,
        write_from_events=False,
        policy_epsilon=epsilon,
        policy_rng=rng,
    )


def train_policy(policy: UsePolicy, w_red: Path, work: Path, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_red, policy, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        probe(ag, "probe_red_with_key", seed + 10)
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
    if m["disable_S_after_reset"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked into policy/cortex."
    if not m["policy_changed"]:
        return "Fail", "Policy weights did not change (no skill learned)."
    red_ok = m["red_commit_unmount"]["correct"]
    green_ok = m["green_heldout_unmount"]["correct"]
    peek_bad = not m["red_peek_unmount"]["correct"]
    if red_ok and green_ok and peek_bad:
        return (
            "Store-works",
            "Policy learned when to commit/apply; cortex frozen; held-out tag used; peek is not memory.",
        )
    if red_ok and not green_ok:
        return "Fail", "Red worked but held-out door did not; policy learned that door, not use."
    if not red_ok:
        return "Fail", "Trained policy did not use the red tag after ρ reset / unmount W."
    return "Fail", "Peek-unmount still correct or other control failed."


def run_v8(seed: int = 12345, n_train: int = 400) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    work = run_dir / "train"
    s_red = run_dir / "S_red"
    s_green = run_dir / "S_green"
    s_peek = run_dir / "S_peek"
    s_off = run_dir / "S_off"
    s_reload = run_dir / "S_reload"
    write_tag_notes(w_red, all_tag_notes(include_red=True, include_green=False))
    write_tag_notes(w_green, all_tag_notes(include_red=False, include_green=True))
    for d in (s_red, s_green, s_peek, s_off, s_reload, work):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    policy_hash0 = policy.weight_hash()
    dummy = make(run_dir / "empty", None, policy)
    cortex0 = dummy.weight_hash()

    untrained = make(run_dir / "S_untrained", w_red, policy, epsilon=0.0)
    untrained.reset_rho()
    probe_untrained = probe(untrained, "probe_red_with_key", seed + 10)

    rewards = train_policy(policy, w_red, work, n_train, seed)
    policy_hash1 = policy.weight_hash()

    # Red: commit from W, unmount, empty ρ.
    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    red_a = make(s_red, w_red, policy, epsilon=0.0)
    red_a.reset_rho()
    probe_red_first = probe(red_a, "probe_red_with_key", seed + 10)
    shutil.copytree(s_red, s_reload, dirs_exist_ok=True)
    red_only = make(s_reload, None, policy, epsilon=0.0)
    red_only.reset_rho()
    probe_red_unmount = probe(red_only, "probe_red_with_key", seed + 10)

    # Peek: do not persist. Force peek by... policy may commit. Separate agent with empty S
    # after a peek-only step: if policy commits we still copy; peek test uses a fresh S
    # and then unmounts without copying files if peek left S empty.
    shutil.rmtree(s_peek, ignore_errors=True)
    s_peek.mkdir()
    peek_a = make(s_peek, w_red, policy, epsilon=0.0)
    peek_a.reset_rho()
    probe_peek_first = probe(peek_a, "probe_red_with_key", seed + 10)
    peek_files = peek_a.store.list_files() if hasattr(peek_a.store, "list_files") else []
    peek_u = make(s_peek, None, policy, epsilon=0.0)
    peek_u.reset_rho()
    probe_peek_unmount = probe(peek_u, "probe_red_with_key", seed + 10)

    # disable-S
    off = make(s_off, None, policy, enabled=False, epsilon=0.0)
    off.reset_rho()
    probe_off = probe(off, "probe_red_with_key", seed + 10)

    # Held-out green: never in training W.
    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    green_a = make(s_green, w_green, policy, epsilon=0.0)
    green_a.reset_rho()
    probe_green_first = probe(green_a, "probe_green", seed + 20)
    s_green_reload = run_dir / "S_green_reload"
    shutil.copytree(s_green, s_green_reload, dirs_exist_ok=True)
    green_only = make(s_green_reload, None, policy, epsilon=0.0)
    green_only.reset_rho()
    probe_green_unmount = probe(green_only, "probe_green", seed + 20)

    cortex1 = red_only.weight_hash()
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
        "untrained_red": probe_untrained,
        "red_first": probe_red_first,
        "red_commit_unmount": probe_red_unmount,
        "red_peek_first": probe_peek_first,
        "red_peek_unmount": probe_peek_unmount,
        "peek_S_files": peek_files,
        "disable_S_after_reset": probe_off,
        "green_first": probe_green_first,
        "green_heldout_unmount": probe_green_unmount,
        "green_S_files": green_a.store.list_files() if hasattr(green_a.store, "list_files") else [],
        "red_S_files": red_a.store.list_files() if hasattr(red_a.store, "list_files") else [],
    }
    # Peek-unmount: if the trained policy *commits*, S has the file and unmount still works.
    # That is commit, not peek. Control: files after one step if only peek → empty S.
    # If policy learned commit, peek_unmount may be correct because files were written.
    # Predeclared peek control: a committed file is memory; empty S after unmount must fail.
    # If policy commits, peek_S_files nonempty and peek_unmount may be True — then
    # the peek test is "S empty ⇒ fail". Use disable-S and a forced empty-S probe.
    if peek_files:
        # Policy committed (wanted). Empty-S unmount: delete files then probe.
        peek_u.reset_store()
        probe_peek_unmount = probe(peek_u, "probe_red_with_key", seed + 10)
        metrics["red_peek_unmount"] = probe_peek_unmount
        metrics["peek_note"] = "policy committed; empty-S after delete used as peek control"

    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v8 boxed use-policy

Classification: **{label}**

{rationale}

| Check | Result |
|-------|--------|
| Cortex unchanged | {metrics['cortex_unchanged']} |
| Policy changed | {metrics['policy_changed']} |
| Untrained red | {probe_untrained['correct']} ({probe_untrained['action_name']}) |
| Red commit, unmount W, ρ empty | {probe_red_unmount['correct']} ({probe_red_unmount['action_name']}) |
| Empty S (peek/delete) | {probe_peek_unmount['correct']} ({probe_peek_unmount['action_name']}) |
| disable-S | {probe_off['correct']} ({probe_off['action_name']}) |
| Held-out green, unmount W | {probe_green_unmount['correct']} ({probe_green_unmount['action_name']}) |
| Train return last 50 | {metrics['train_return_mean_last50']:.2f} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v8 boxed policy, frozen cortex")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    args = p.parse_args()
    m = run_v8(seed=args.seed, n_train=args.n_train)
    print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir", "policy_changed", "cortex_unchanged")}, indent=2))
    print("untrained", m["untrained_red"]["action_name"], m["untrained_red"]["correct"])
    print("red unmount", m["red_commit_unmount"]["action_name"], m["red_commit_unmount"]["correct"])
    print("disable-S", m["disable_S_after_reset"]["action_name"], m["disable_S_after_reset"]["correct"])
    print("green unmount", m["green_heldout_unmount"]["action_name"], m["green_heldout_unmount"]["correct"])
    print("empty S", m["red_peek_unmount"]["action_name"], m["red_peek_unmount"]["correct"])


if __name__ == "__main__":
    main()
