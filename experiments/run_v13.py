"""v13: learn to copy action= from the file. No USE_KEY/WAIT table. Train red only."""

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

from experiments.run_v10 import live_free
from experiments.run_v12 import clutter_w_no_answers, probe as probe_v12
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.symbols import BLUE_FACT_ID, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v13"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    out = probe_v12(agent, scenario, seed)
    return out


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
        use_read=True,
    )


def train_write_and_use(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    baseline_w = 0.0
    baseline_u = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(
            s_dir,
            w_clutter,
            policy,
            epsilon=eps,
            explore_epsilon=explore_eps,
            rng=rng,
        )
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        wrote = any(t.get("kind") == "write" and t.get("write") for t in ag.policy_traces)
        r_write = 1.0 if wrote else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        baseline_w = 0.9 * baseline_w + 0.1 * r_write
        baseline_u = 0.9 * baseline_u + 0.1 * r_use
        policy.update([t for t in ag.policy_traces if t.get("kind") == "write"], r_write - baseline_w)
        policy.update([t for t in ag.policy_traces if t.get("kind") == "use"], r_use - baseline_u)
        rewards.append(r_use)
    return rewards


def one_life(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    scenario: str,
    seed: int,
    *,
    explore_epsilon: float = 0.5,
    retrieve_policy: str = "select",
    rng: np.random.Generator | None = None,
    max_steps: int = 32,
    enabled: bool = True,
) -> tuple[ThreeMemoryAgent, dict[str, Any]]:
    ag = make(
        s_dir,
        w_dir,
        policy,
        enabled=enabled,
        epsilon=0.0,
        explore_epsilon=explore_epsilon,
        retrieve_policy=retrieve_policy,
        rng=rng,
    )
    ag.reset_rho()
    live = live_free(ag, scenario, seed, max_steps=max_steps)
    ag.world = None
    ag.explore_epsilon = 0.0
    ag.reset_rho()
    return ag, live


def classify(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if any(lv["n_forced"] for lv in (m["red_live"], m["green_live"], m["blue_live"])):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["red_probe"], m["green_probe"], m["blue_probe"])):
        return "Confound", "Probe used exploration."
    if not m["use_changed"]:
        return "Fail", "Use head did not change."
    if m["untrained_red"]["correct"] or m["untrained_planted"]["correct"]:
        return "Fail", "Untrained already used the key (table still wired, or fact leaked)."
    if not m["red_probe"]["correct"]:
        return "Fail", "Trained use-head did not copy red action=2."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; use-head learned that door, not read-the-integer."
    if not m["blue_probe"]["correct"]:
        return "Fail", "Held-out blue failed; use-head learned that door, not read-the-integer."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["dump_red"]["correct"]:
        return "Fail", "Dump-all still correct on red; mix should hurt."
    if m["red_probe"].get("use") is False:
        return "Fail", "Probe did not gate use=True."
    return (
        "Store-works",
        "Use-head learned to copy action=; cortex frozen; held-out green/blue worked; dump-all still mixes.",
    )


def run_v13(seed: int = 12345, n_train: int = 400, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    work = run_dir / "train"
    s_un = run_dir / "S_untrained"
    s_red = run_dir / "S_red"
    s_green = run_dir / "S_green"
    s_blue = run_dir / "S_blue"
    s_dump = run_dir / "S_dump"
    s_off = run_dir / "S_off"
    s_empty = run_dir / "S_empty"
    s_plant = run_dir / "S_plant"
    write_tag_notes(w_clutter, clutter_w_no_answers())
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))
    for d in (s_un, s_red, s_green, s_blue, s_dump, s_off, s_empty, s_plant, work):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    use_hash0 = str(policy.w_use.tobytes().hex()) + str(float(policy.b_use))
    dummy = make(run_dir / "empty_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a, un_live = one_life(
        s_un, w_clutter, policy, "experience_teach", seed + 10, rng=np.random.default_rng(seed + 9), max_steps=max_steps
    )
    untrained_red = probe(un_a, "probe_red_with_key", seed + 10)

    write_tag_notes(s_plant, [(f"{RED_FACT_ID}.tag", {"door": 0, "action": 2})])
    plant_a = make(s_plant, None, policy)
    plant_a.reset_rho()
    untrained_planted = probe(plant_a, "probe_red_with_key", seed + 10)

    rewards = train_write_and_use(policy, w_clutter, work, n_train, seed, max_steps)
    use_hash1 = str(policy.w_use.tobytes().hex()) + str(float(policy.b_use))

    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    red_a, red_live = one_life(
        s_red, w_clutter, policy, "experience_teach", seed + 10, rng=np.random.default_rng(seed + 1), max_steps=max_steps
    )
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)
    red_probe["use"] = (red_a.last_policy or {}).get("use")

    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    green_a, green_live = one_life(
        s_green, w_clutter, policy, "experience_green", seed + 20, rng=np.random.default_rng(seed + 3), max_steps=max_steps
    )
    green_probe = probe(green_a, "probe_green", seed + 20)

    shutil.rmtree(s_blue, ignore_errors=True)
    s_blue.mkdir()
    blue_a, blue_live = one_life(
        s_blue, w_clutter, policy, "experience_foil", seed + 30, rng=np.random.default_rng(seed + 8), max_steps=max_steps
    )
    blue_probe = probe(blue_a, "probe_blue", seed + 30)

    shutil.rmtree(s_dump, ignore_errors=True)
    s_dump.mkdir()
    dump_src = run_dir / "S_two"
    dump_src.mkdir(exist_ok=True)
    two_a, _ = one_life(
        dump_src, w_clutter, policy, "experience_teach", seed + 10, rng=np.random.default_rng(seed + 4), max_steps=max_steps
    )
    two_a.explore_epsilon = 0.5
    two_a.world = TagLibrary(w_clutter)
    two_a.reset_rho()
    live_free(two_a, "experience_green", seed + 20, max_steps=max_steps)
    shutil.copytree(dump_src, s_dump, dirs_exist_ok=True)
    dump_a = make(s_dump, None, policy, retrieve_policy="dump", explore_epsilon=0.0)
    dump_a.reset_rho()
    dump_red = probe(dump_a, "probe_red_with_key", seed + 10)

    empty = make(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)

    off_a, _ = one_life(
        s_off, None, policy, "experience_teach", seed + 10, enabled=False, rng=np.random.default_rng(seed + 2), max_steps=max_steps
    )
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    def _slim(lv: dict[str, Any]) -> dict[str, Any]:
        return {k: lv[k] for k in ("wrote", "opened", "files", "n_forced", "n_steps", "actions")}

    cortex1 = red_a.weight_hash()
    metrics: dict[str, Any] = {
        "seed": seed,
        "n_train": n_train,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == cortex1,
        "use_changed": use_hash0 != use_hash1,
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
        "untrained_live": _slim(un_live),
        "untrained_red": untrained_red,
        "untrained_planted": untrained_planted,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "green_live": _slim(green_live),
        "green_probe": green_probe,
        "blue_live": _slim(blue_live),
        "blue_probe": blue_probe,
        "dump_red": dump_red,
        "empty_S": empty_p,
        "disable_S_red": disable_red,
        "red_tag": (s_red / f"{RED_FACT_ID}.tag").read_text(encoding="utf-8") if (s_red / f"{RED_FACT_ID}.tag").is_file() else "",
        "green_tag": (s_green / f"{GREEN_FACT_ID}.tag").read_text(encoding="utf-8") if (s_green / f"{GREEN_FACT_ID}.tag").is_file() else "",
        "blue_tag": (s_blue / f"{BLUE_FACT_ID}.tag").read_text(encoding="utf-8") if (s_blue / f"{BLUE_FACT_ID}.tag").is_file() else "",
    }
    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v13 read action=

Classification: **{label}**

{rationale}

| Check | Result |
|-------|--------|
| Cortex unchanged | {metrics['cortex_unchanged']} |
| Use head changed | {metrics['use_changed']} |
| Untrained red | {untrained_red['correct']} ({untrained_red['action_name']}) |
| Untrained planted d0.tag | {untrained_planted['correct']} ({untrained_planted['action_name']}, use={untrained_planted.get('policy', {}).get('use')}) |
| Red after ρ reset | {red_probe['correct']} ({red_probe['action_name']}, use={red_probe.get('use')}) |
| Held-out green | {green_probe['correct']} ({green_probe['action_name']}) |
| Held-out blue | {blue_probe['correct']} ({blue_probe['action_name']}) |
| Dump-all red | {dump_red['correct']} ({dump_red['action_name']}) |
| Empty S / disable-S | {empty_p['correct']} / {disable_red['correct']} |
| Train last 50 | {metrics['train_return_last50']:.2f} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v13 learn to copy action= from the file")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v13(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {k: m[k] for k in ("classification", "rationale", "run_dir", "use_changed", "cortex_unchanged")},
            indent=2,
        )
    )
    print("untrained", m["untrained_red"]["action_name"], m["untrained_red"]["correct"])
    print("planted", m["untrained_planted"]["action_name"], m["untrained_planted"]["correct"])
    print("red", m["red_probe"]["action_name"], m["red_probe"]["correct"], m["red_tag"].strip())
    print("green", m["green_probe"]["action_name"], m["green_probe"]["correct"], m["green_tag"].strip())
    print("blue", m["blue_probe"]["action_name"], m["blue_probe"]["correct"], m["blue_tag"].strip())
    print("dump red", m["dump_red"]["action_name"], m["dump_red"]["correct"])


if __name__ == "__main__":
    main()
