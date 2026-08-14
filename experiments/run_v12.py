"""v12: boxed policy learns select vs dump. Held-out blue note was not in retrieve training."""

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

from experiments.run_v10 import live_free, train_policy
from experiments.run_v11 import two_lives
from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import BLUE_FACT_ID, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, all_tag_notes, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v12"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clutter_w_no_answers() -> list[tuple[str, dict[str, Any]]]:
    return [n for n in all_tag_notes(include_red=False, include_green=False) if n[0] != f"{BLUE_FACT_ID}.tag"]


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    world = KeyDoorWorld(seed=seed)
    obs = world.reset(scenario)
    action, meta = agent.act(obs, update_rho=False, explore=False)
    result = world.step(action, scenario)
    if scenario == "probe_red_with_key":
        correct = action == Action.USE_KEY and bool(result.info.get("opened"))
    elif scenario == "probe_green":
        correct = action == Action.WAIT and bool(result.info.get("opened"))
    elif scenario == "probe_blue":
        correct = action == Action.OPEN and bool(result.info.get("opened"))
    else:
        correct = False
    pol = meta.get("policy") or {}
    return {
        "scenario": scenario,
        "action": int(action),
        "action_name": Action(action).name.lower(),
        "correct": correct,
        "opened": bool(result.info.get("opened")),
        "explored": bool(meta.get("explored")),
        "retrieve_mode": pol.get("retrieve_mode"),
        "store_len": len(agent.store),
        "policy": pol,
        "files": agent.store.list_files() if hasattr(agent.store, "list_files") else [],
    }


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    explore_epsilon: float = 0.0,
    retrieve_policy: str = "policy",
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


def train_retrieve(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed + 77)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ret_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(
            s_dir,
            w_clutter,
            policy,
            epsilon=0.0,
            explore_epsilon=0.5,
            retrieve_policy="policy",
            rng=rng,
        )
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.reset_rho()
        live_free(ag, "experience_green", seed + 20, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        ag.policy_traces = []
        ag.policy_epsilon = eps
        p = probe(ag, "probe_red_with_key", seed + 10)
        r = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r
        policy.update(ag.policy_traces, r - baseline)
        rewards.append(r)
    return rewards


def classify(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if any(lv["n_forced"] for lv in (m["red_live"], m["green_live"], m["blue_live"])):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["policy_red"], m["policy_green"], m["policy_blue"])):
        return "Confound", "Probe used exploration."
    if not m["retrieve_changed"]:
        return "Fail", "Retrieve head did not change."
    if m["untrained_retrieve_red"]["correct"]:
        return "Fail", "Untrained retrieve already solved red (did not start from dump)."
    if m["policy_red"].get("retrieve_mode") == "dump":
        return "Fail", "Trained head still dumped on red."
    if not m["policy_red"]["correct"] or not m["policy_green"]["correct"]:
        return "Fail", "Trained select missed red or green."
    if not m["blue_authored"] or not m["policy_blue"]["correct"]:
        return "Fail", "Held-out blue note missing or unused."
    if m["dump_red"]["correct"] and m["dump_green"]["correct"] and m["dump_blue"]["correct"]:
        return "Fail", "Dump-all matched select on all probes; N too small."
    if not m["dump_red"]["correct"] and m["policy_red"]["correct"] and m["policy_blue"]["correct"]:
        return (
            "Store-works",
            "Retrieve head learned to select; dump-all still mixes lives; held-out blue worked.",
        )
    return "Fail", "Dump control or other check failed."


def run_v12(seed: int = 12345, n_train: int = 400, n_retrieve: int = 200, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    work = run_dir / "train"
    s_pre = run_dir / "S_pre"
    s_both = run_dir / "S_both"
    s_dump = run_dir / "S_dump"
    s_blue = run_dir / "S_blue"
    s_off = run_dir / "S_off"
    s_empty = run_dir / "S_empty"
    write_tag_notes(w_clutter, clutter_w_no_answers())
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))
    for d in (s_pre, s_both, s_dump, s_blue, s_off, s_empty, work):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    dummy = make(run_dir / "empty_hash", None, policy)
    cortex0 = dummy.weight_hash()

    rewards_w = train_policy(policy, w_clutter, work, n_train, seed, max_steps)
    retrieve_hash0 = policy.weight_hash()

    shutil.rmtree(s_pre, ignore_errors=True)
    s_pre.mkdir()
    pre_a, pre_red, pre_green = two_lives(
        s_pre, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 3), seed=seed, max_steps=max_steps
    )
    pre_a.retrieve_policy = "policy"
    pre_a.world = None
    pre_a.reset_rho()
    untrained_ret_red = probe(pre_a, "probe_red_with_key", seed + 10)

    rewards_r = train_retrieve(policy, w_clutter, work, n_retrieve, seed, max_steps)
    retrieve_hash1 = policy.weight_hash()

    shutil.rmtree(s_both, ignore_errors=True)
    s_both.mkdir()
    both_a, live_red, live_green = two_lives(
        s_both, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 1), seed=seed, max_steps=max_steps
    )
    shutil.copytree(s_both, s_dump, dirs_exist_ok=True)
    shutil.copytree(s_both, s_blue, dirs_exist_ok=True)

    pol_a = make(s_both, None, policy, explore_epsilon=0.0, retrieve_policy="policy", epsilon=0.0)
    pol_a.reset_rho()
    policy_red = probe(pol_a, "probe_red_with_key", seed + 10)
    policy_green = probe(pol_a, "probe_green", seed + 20)

    dump_a = make(s_dump, None, policy, explore_epsilon=0.0, retrieve_policy="dump")
    dump_a.reset_rho()
    dump_red = probe(dump_a, "probe_red_with_key", seed + 10)
    dump_green = probe(dump_a, "probe_green", seed + 20)

    blue_a = make(s_blue, w_clutter, policy, explore_epsilon=0.5, retrieve_policy="policy", rng=np.random.default_rng(seed + 8))
    blue_a.reset_rho()
    live_blue = live_free(blue_a, "experience_foil", seed + 30, max_steps=max_steps)
    blue_a.world = None
    blue_a.explore_epsilon = 0.0
    blue_a.reset_rho()
    policy_blue = probe(blue_a, "probe_blue", seed + 30)
    dump_blue_dir = run_dir / "S_blue_dump"
    shutil.copytree(s_blue, dump_blue_dir, dirs_exist_ok=True)
    dump_b = make(dump_blue_dir, None, policy, retrieve_policy="dump")
    dump_b.reset_rho()
    dump_blue = probe(dump_b, "probe_blue", seed + 30)

    empty = make(s_empty, None, policy, retrieve_policy="policy")
    empty.reset_rho()
    empty_red = probe(empty, "probe_red_with_key", seed + 10)

    off_a, _, _ = two_lives(
        s_off, None, policy, enabled=False, explore_epsilon=0.5, rng=np.random.default_rng(seed + 2), seed=seed, max_steps=max_steps
    )
    off_a.reset_rho()
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    cortex1 = pol_a.weight_hash()
    s_files = sorted(pol_a.store.list_files()) if hasattr(pol_a.store, "list_files") else []
    blue_files = sorted(blue_a.store.list_files()) if hasattr(blue_a.store, "list_files") else []

    def _slim(lv: dict[str, Any]) -> dict[str, Any]:
        return {k: lv[k] for k in ("wrote", "opened", "files", "n_forced", "n_steps", "actions")}

    metrics: dict[str, Any] = {
        "seed": seed,
        "n_train": n_train,
        "n_retrieve": n_retrieve,
        "train_write_last50": float(np.mean(rewards_w[-50:])) if rewards_w else 0.0,
        "train_retrieve_last50": float(np.mean(rewards_r[-50:])) if rewards_r else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == cortex1,
        "retrieve_changed": retrieve_hash0 != retrieve_hash1,
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
        "untrained_retrieve_red": untrained_ret_red,
        "red_live": _slim(live_red),
        "green_live": _slim(live_green),
        "blue_live": _slim(live_blue),
        "s_files": s_files,
        "blue_files": blue_files,
        "blue_authored": f"{BLUE_FACT_ID}.tag" in blue_files,
        "policy_red": policy_red,
        "policy_green": policy_green,
        "policy_blue": policy_blue,
        "dump_red": dump_red,
        "dump_green": dump_green,
        "dump_blue": dump_blue,
        "empty_S_red": empty_red,
        "disable_S_red": disable_red,
        "pre_lives": {"red": _slim(pre_red), "green": _slim(pre_green)},
    }
    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v12 select vs dump head

Classification: **{label}**

{rationale}

| Check | Result |
|-------|--------|
| Cortex unchanged | {metrics['cortex_unchanged']} |
| Retrieve head changed | {metrics['retrieve_changed']} |
| Untrained retrieve red | {untrained_ret_red['correct']} ({untrained_ret_red['action_name']}, {untrained_ret_red.get('retrieve_mode')}) |
| Policy red | {policy_red['correct']} ({policy_red['action_name']}, {policy_red.get('retrieve_mode')}) |
| Policy green | {policy_green['correct']} ({policy_green['action_name']}, {policy_green.get('retrieve_mode')}) |
| Dump-all red | {dump_red['correct']} ({dump_red['action_name']}) |
| Held-out blue authored | {metrics['blue_authored']} ({blue_files}) |
| Policy blue | {policy_blue['correct']} ({policy_blue['action_name']}, {policy_blue.get('retrieve_mode')}) |
| Dump-all blue | {dump_blue['correct']} ({dump_blue['action_name']}) |
| Empty S / disable-S | {empty_red['correct']} / {disable_red['correct']} |
| Retrieve train last 50 | {metrics['train_retrieve_last50']:.2f} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v12 boxed select vs dump")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-retrieve", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v12(seed=args.seed, n_train=args.n_train, n_retrieve=args.n_retrieve, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                k: m[k]
                for k in (
                    "classification",
                    "rationale",
                    "run_dir",
                    "retrieve_changed",
                    "cortex_unchanged",
                    "blue_authored",
                )
            },
            indent=2,
        )
    )
    print("untrained retrieve red", m["untrained_retrieve_red"]["action_name"], m["untrained_retrieve_red"].get("retrieve_mode"))
    print("policy red", m["policy_red"]["action_name"], m["policy_red"].get("retrieve_mode"), m["policy_red"]["correct"])
    print("policy green", m["policy_green"]["action_name"], m["policy_green"]["correct"])
    print("dump red", m["dump_red"]["action_name"], m["dump_red"]["correct"])
    print("policy blue", m["policy_blue"]["action_name"], m["policy_blue"].get("retrieve_mode"), m["policy_blue"]["correct"])
    print("dump blue", m["dump_blue"]["action_name"], m["dump_blue"]["correct"])


if __name__ == "__main__":
    main()
