"""v18: A learn to write do=. B learn to write here=. Read/match frozen."""

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
from experiments.run_v15 import _slim
from three_memory.agent import ThreeMemoryAgent
from three_memory.env import KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import BLUE_FACT_ID, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v18"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    return probe_v12(agent, scenario, seed)


def _head_fp(policy: UsePolicy, name: str) -> str:
    w = getattr(policy, f"w_{name}")
    b = getattr(policy, f"b_{name}")
    return str(w.tobytes().hex()) + str(float(b))


def _w_flags(w_files: list[str]) -> dict[str, Any]:
    return {
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
    }


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def make_a(
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
        explore_epsilon=explore_epsilon,
        use_read=True,
        value_key="do",
        use_wkey_head=True,
        force_use=force_use,
        force_write=force_write,
    )


def make_b(
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
        explore_epsilon=explore_epsilon,
        use_read=True,
        place_key="here",
        use_wplace_head=True,
        force_use=force_use,
        force_write=force_write,
    )


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if m["red_live"]["n_forced"] or m["green_live"]["n_forced"]:
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_write"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if m["trained_force_use"] or m["trained_force_write"]:
        return "Fail", "Use/write clamped to rescue the plot."
    if m.get("value_key_frozen_do_on_write"):
        return "Fail", "Writer frozen to do=; write-key was not required."
    if not m["wkey_changed"]:
        return "Fail", "Write-key head did not change."
    if m["untrained_write"]["correct"] or m["untrained_write"]["action_name"] == "use_key":
        return "Fail", "Untrained already wrote a usable do= (or table still wired)."
    if "do=" in m.get("untrained_tag", ""):
        return "Fail", "Untrained already wrote do=."
    if "action=" not in m.get("untrained_tag", ""):
        return "Fail", "Untrained did not write action=."
    if not m["planted_do"]["correct"]:
        return "Fail", "Frozen read do= did not copy a planted note."
    if not m["red_probe"]["correct"]:
        return "Fail", "Trained write-key did not author red do=2."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; head learned that door, not write do=."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["action_control"]["correct"]:
        return "Fail", "Writing action= still solved red; do= was not required."
    if "do=" not in m.get("red_tag", ""):
        return "Fail", "Trained red note missing do=."
    if "action=" in m.get("red_tag", ""):
        return "Fail", "Trained red note still has action=."
    if "do=" in m.get("ctrl_tag", ""):
        return "Fail", "Control still wrote do= with write-key untrained."
    return (
        "Store-works",
        "Write-key learned do=; cortex frozen; held-out green wait; action= control fails.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if m["red_live"]["n_forced"] or m["green_live"]["n_forced"]:
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_write"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if m["trained_force_use"] or m["trained_force_write"]:
        return "Fail", "Use/write clamped to rescue the plot."
    if m.get("place_key_frozen_here_on_write"):
        return "Fail", "Writer frozen to here=; write-place was not required."
    if not m["wplace_changed"]:
        return "Fail", "Write-place head did not change."
    if m["untrained_write"]["correct"] or m["untrained_write"]["action_name"] == "use_key":
        return "Fail", "Untrained already wrote a usable here= (or table still wired)."
    if "here=" in m.get("untrained_tag", ""):
        return "Fail", "Untrained already wrote here=."
    if "door=" not in m.get("untrained_tag", ""):
        return "Fail", "Untrained did not write door=."
    if not m["planted_here"]["correct"]:
        return "Fail", "Frozen match here= did not use a planted note."
    if not m["red_probe"]["correct"]:
        return "Fail", "Trained write-place did not author red here=."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; head learned that door, not write here=."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["door_control"]["correct"]:
        return "Fail", "Writing door= still solved red; here= was not required."
    if "here=" not in m.get("red_tag", ""):
        return "Fail", "Trained red note missing here=."
    if "door=" in m.get("red_tag", ""):
        return "Fail", "Trained red note still has door=."
    if "here=" in m.get("ctrl_tag", ""):
        return "Fail", "Control still wrote here= with write-place untrained."
    return (
        "Store-works",
        "Write-place learned here=; cortex frozen; held-out green wait; door= control fails.",
    )


def _train(
    make_fn,
    kind: str,
    policy: UsePolicy,
    w_clutter: Path,
    work: Path,
    n: int,
    seed: int,
    max_steps: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_w = b_u = b_h = 0.0
    alt_key = "wkey_alt" if kind == "wkey" else "wplace_alt"
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"{kind}_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make_fn(s_dir, w_clutter, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        wrote = any(t.get("kind") == "write" and t.get("write") for t in ag.policy_traces)
        alt = any(t.get("kind") == kind and t.get(alt_key) for t in ag.policy_traces)
        r_write = 1.0 if wrote else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        r_h = 1.0 if alt else 0.0
        b_w = 0.9 * b_w + 0.1 * r_write
        b_u = 0.9 * b_u + 0.1 * r_use
        b_h = 0.9 * b_h + 0.1 * r_h
        tr = ag.policy_traces
        policy.update([t for t in tr if t.get("kind") == "write"], r_write - b_w)
        policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_u)
        policy.update([t for t in tr if t.get("kind") == kind], r_h - b_h)
        rewards.append(r_use)
    return rewards


def _life(make_fn, s_dir, w_dir, policy, scenario, seed, rng, max_steps, enabled=True):
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make_fn(
        s_dir,
        w_dir,
        policy,
        enabled=enabled,
        epsilon=0.0,
        explore_epsilon=0.5,
        rng=rng,
    )
    ag.reset_rho()
    live = live_free(ag, scenario, seed, max_steps=max_steps)
    ag.world = None
    ag.explore_epsilon = 0.0
    ag.reset_rho()
    return ag, live


def _untrained_write(make_fn, s_dir: Path, seed: int) -> tuple[dict[str, Any], str]:
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    pol = UsePolicy(seed=7, lr=0.2)
    ag = make_fn(s_dir, None, pol, force_use=True, force_write=True)
    obs = KeyDoorWorld(0).reset("probe_red_with_key")
    ag.observe_outcome(obs, True, {"opened": True, "action": "use_key"})
    ag.reset_rho()
    return probe(ag, "probe_red_with_key", seed + 10), _tags(s_dir)


def _copy_write_use(src: UsePolicy) -> UsePolicy:
    ctrl = UsePolicy(seed=7, lr=0.2)
    ctrl.w_write = src.w_write.copy()
    ctrl.b_write = np.array(float(src.b_write))
    ctrl.w_use = src.w_use.copy()
    ctrl.b_use = np.array(float(src.b_use))
    return ctrl


def run_arm_a(run_dir: Path, w_clutter: Path, w_files: list[str], seed: int, n_train: int, max_steps: int) -> dict[str, Any]:
    work = run_dir / "A_train"
    s_un = run_dir / "A_untrained"
    s_plant = run_dir / "A_plant"
    s_red = run_dir / "A_red"
    s_green = run_dir / "A_green"
    s_ctrl = run_dir / "A_action"
    s_off = run_dir / "A_off"
    s_empty = run_dir / "A_empty"
    for d in (work, s_un, s_plant, s_red, s_green, s_ctrl, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    wkey0 = _head_fp(policy, "wkey")
    dummy = make_a(run_dir / "A_hash", None, policy)
    cortex0 = dummy.weight_hash()

    untrained_write, untrained_tag = _untrained_write(make_a, s_un, seed)

    write_tag_notes(s_plant, [("d0.tag", {"door": 0, "do": 2})])
    plant_a = make_a(s_plant, None, UsePolicy(seed=7), force_use=True)
    plant_a.reset_rho()
    planted_do = probe(plant_a, "probe_red_with_key", seed + 10)

    rewards = _train(make_a, "wkey", policy, w_clutter, work, n_train, seed, max_steps)
    wkey1 = _head_fp(policy, "wkey")

    red_a, red_live = _life(make_a, s_red, w_clutter, policy, "experience_teach", seed + 10, np.random.default_rng(seed + 1), max_steps)
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)
    green_a, green_live = _life(make_a, s_green, w_clutter, policy, "experience_green", seed + 20, np.random.default_rng(seed + 3), max_steps)
    green_probe = probe(green_a, "probe_green", seed + 20)

    ctrl = _copy_write_use(policy)
    ctrl_a, _ = _life(make_a, s_ctrl, w_clutter, ctrl, "experience_teach", seed + 10, np.random.default_rng(seed + 4), max_steps)
    action_control = probe(ctrl_a, "probe_red_with_key", seed + 10)

    empty = make_a(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    off_a, _ = _life(make_a, s_off, None, policy, "experience_teach", seed + 10, np.random.default_rng(seed + 2), max_steps, enabled=False)
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": "A",
        "trained_force_use": False,
        "trained_force_write": False,
        "value_key_frozen_do_on_write": False,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "wkey_changed": wkey0 != wkey1,
        **_w_flags(w_files),
        "untrained_write": untrained_write,
        "untrained_tag": untrained_tag,
        "planted_do": planted_do,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "red_tag": _tags(s_red),
        "green_live": _slim(green_live),
        "green_probe": green_probe,
        "green_tag": _tags(s_green),
        "action_control": action_control,
        "ctrl_tag": _tags(s_ctrl),
        "empty_S": empty_p,
        "disable_S_red": disable_red,
    }
    label, rationale = classify_a(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_arm_b(run_dir: Path, w_clutter: Path, w_files: list[str], seed: int, n_train: int, max_steps: int) -> dict[str, Any]:
    work = run_dir / "B_train"
    s_un = run_dir / "B_untrained"
    s_plant = run_dir / "B_plant"
    s_red = run_dir / "B_red"
    s_green = run_dir / "B_green"
    s_ctrl = run_dir / "B_door"
    s_off = run_dir / "B_off"
    s_empty = run_dir / "B_empty"
    for d in (work, s_un, s_plant, s_red, s_green, s_ctrl, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    wplace0 = _head_fp(policy, "wplace")
    dummy = make_b(run_dir / "B_hash", None, policy)
    cortex0 = dummy.weight_hash()

    untrained_write, untrained_tag = _untrained_write(make_b, s_un, seed)

    write_tag_notes(s_plant, [("d0.tag", {"here": 0, "action": 2})])
    plant_a = make_b(s_plant, None, UsePolicy(seed=7), force_use=True)
    plant_a.reset_rho()
    planted_here = probe(plant_a, "probe_red_with_key", seed + 10)

    rewards = _train(make_b, "wplace", policy, w_clutter, work, n_train, seed + 5, max_steps)
    wplace1 = _head_fp(policy, "wplace")

    red_a, red_live = _life(make_b, s_red, w_clutter, policy, "experience_teach", seed + 10, np.random.default_rng(seed + 1), max_steps)
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)
    green_a, green_live = _life(make_b, s_green, w_clutter, policy, "experience_green", seed + 20, np.random.default_rng(seed + 3), max_steps)
    green_probe = probe(green_a, "probe_green", seed + 20)

    ctrl = _copy_write_use(policy)
    ctrl_a, _ = _life(make_b, s_ctrl, w_clutter, ctrl, "experience_teach", seed + 10, np.random.default_rng(seed + 4), max_steps)
    door_control = probe(ctrl_a, "probe_red_with_key", seed + 10)

    empty = make_b(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    off_a, _ = _life(make_b, s_off, None, policy, "experience_teach", seed + 10, np.random.default_rng(seed + 2), max_steps, enabled=False)
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": "B",
        "trained_force_use": False,
        "trained_force_write": False,
        "place_key_frozen_here_on_write": False,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "wplace_changed": wplace0 != wplace1,
        **_w_flags(w_files),
        "untrained_write": untrained_write,
        "untrained_tag": untrained_tag,
        "planted_here": planted_here,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "red_tag": _tags(s_red),
        "green_live": _slim(green_live),
        "green_probe": green_probe,
        "green_tag": _tags(s_green),
        "door_control": door_control,
        "ctrl_tag": _tags(s_ctrl),
        "empty_S": empty_p,
        "disable_S_red": disable_red,
    }
    label, rationale = classify_b(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v18(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    write_tag_notes(w_clutter, clutter_w_no_answers())
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))
    a = run_arm_a(run_dir, w_clutter, w_files, seed, n_train, max_steps)
    b = run_arm_b(run_dir, w_clutter, w_files, seed, n_train, max_steps)
    out = {"seed": seed, "n_train": n_train, "run_dir": str(run_dir), "A": a, "B": b, "same_cortex": a["cortex_hash"] == b["cortex_hash"]}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v18 A vs B

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A write do= | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B write here= | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained write | {a['untrained_write']['action_name']} ({a['untrained_write']['correct']}) | {b['untrained_write']['action_name']} ({b['untrained_write']['correct']}) |
| Trained red | {a['red_probe']['action_name']} ({a['red_probe']['correct']}) | {b['red_probe']['action_name']} ({b['red_probe']['correct']}) |
| Held-out green | {a['green_probe']['action_name']} ({a['green_probe']['correct']}) | {b['green_probe']['action_name']} ({b['green_probe']['correct']}) |
| Control | action= {a['action_control']['action_name']} ({a['action_control']['correct']}) | door= {b['door_control']['action_name']} ({b['door_control']['correct']}) |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v18 write do= vs write here=")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v18(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print("A", m["A"]["untrained_write"]["action_name"], m["A"]["red_probe"]["action_name"], m["A"]["green_probe"]["action_name"], m["A"]["red_tag"].strip())
    print("B", m["B"]["untrained_write"]["action_name"], m["B"]["red_probe"]["action_name"], m["B"]["green_probe"]["action_name"], m["B"]["red_tag"].strip())


if __name__ == "__main__":
    main()
