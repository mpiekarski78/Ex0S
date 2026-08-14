"""v14: A pick-one among matches vs B write complete schema. Same frozen cortex."""

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
    d = REPO_ROOT / "runs" / f"{stamp}_v14"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    return probe_v12(agent, scenario, seed)


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
    unique_writes: bool = False,
    use_pick: bool = False,
    write_schema: bool = False,
    force_use: bool = False,
    force_write: bool = False,
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
        unique_writes=unique_writes,
        use_pick=use_pick,
        write_schema=write_schema,
        force_use=force_use,
        force_write=force_write,
    )


def _slim(lv: dict[str, Any]) -> dict[str, Any]:
    return {k: lv[k] for k in ("wrote", "opened", "files", "n_forced", "n_steps", "actions")}


def _w_flags(w_files: list[str]) -> dict[str, Any]:
    return {
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
    }


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if m["red_live"]["n_forced"] or m.get("green_live", {}).get("n_forced"):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_conflict"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if not m["pick_changed"]:
        return "Fail", "Pick head did not change."
    if m["untrained_conflict"]["correct"]:
        return "Fail", "Untrained already used the key (apply-all did not mix)."
    if not m["red_probe"]["correct"]:
        return "Fail", "Trained pick did not take newest red action=2."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; pick head learned that door, not newest-wins."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["apply_all_red"]["correct"]:
        return "Fail", "Apply-all still correct on red; mix should hurt."
    if m["red_probe"].get("policy", {}).get("one") is False:
        return "Fail", "Probe did not gate one=True."
    return (
        "Store-works",
        "Pick-head learned one vs all; newest file’s integer copied; cortex frozen; held-out green worked.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if any(lv["n_forced"] for lv in (m["untrained_live"], m["red_live"], m["green_live"])):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_red"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if not m["schema_changed"]:
        return "Fail", "Schema head did not change."
    if m["untrained_red"]["correct"]:
        return "Fail", "Untrained already used the key (schema already complete, or table wired)."
    if m["incomplete_planted"]["correct"]:
        return "Fail", "Door-only note still used the key."
    if not m["red_probe"]["correct"]:
        return "Fail", "Trained schema did not include red action=."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; schema head learned that door, not include-the-integer."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if "action=" in m.get("untrained_tag", ""):
        return "Fail", "Untrained write already included action=."
    if "action=" not in m.get("red_tag", ""):
        return "Fail", "Trained red note missing action=."
    return (
        "Store-works",
        "Schema-head learned to include action=; integer still from the event; cortex frozen; held-out green worked.",
    )


def train_pick(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"pick_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        write_tag_notes(s_dir, [("d0_stale.tag", {"door": 0, "action": 0, "when": 0})])
        ag = make(
            s_dir,
            w_clutter,
            policy,
            epsilon=eps,
            explore_epsilon=explore_eps,
            rng=rng,
            unique_writes=True,
            use_pick=True,
            force_use=True,
            force_write=True,
        )
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r
        policy.update([t for t in ag.policy_traces if t.get("kind") == "pick"], r - baseline)
        rewards.append(r)
    return rewards


def train_schema(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed + 3)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"sch_{ep}"
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
            write_schema=True,
            force_use=True,
            force_write=True,
        )
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r
        policy.update([t for t in ag.policy_traces if t.get("kind") == "schema"], r - baseline)
        rewards.append(r)
    return rewards


def _conflict_notes(door: int, stale_act: int, new_act: int) -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"d{door}_t1.tag", {"door": door, "action": stale_act, "when": 1}),
        (f"d{door}_t10.tag", {"door": door, "action": new_act, "when": 10}),
    ]


def run_arm_a(run_dir: Path, w_clutter: Path, w_files: list[str], seed: int, n_train: int, max_steps: int) -> dict[str, Any]:
    work = run_dir / "A_train"
    s_un = run_dir / "A_untrained"
    s_red = run_dir / "A_red"
    s_green = run_dir / "A_green"
    s_all = run_dir / "A_apply_all"
    s_off = run_dir / "A_off"
    s_empty = run_dir / "A_empty"
    for d in (work, s_un, s_red, s_green, s_all, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    pick0 = str(policy.w_pick.tobytes().hex()) + str(float(policy.b_pick))
    dummy = make(run_dir / "A_hash", None, policy, use_pick=True, force_use=True)
    cortex0 = dummy.weight_hash()

    write_tag_notes(s_un, _conflict_notes(0, 0, 2))
    un_a = make(s_un, None, policy, use_pick=True, force_use=True)
    un_a.reset_rho()
    untrained_conflict = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = train_pick(policy, w_clutter, work, n_train, seed, max_steps)
    pick1 = str(policy.w_pick.tobytes().hex()) + str(float(policy.b_pick))

    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    write_tag_notes(s_red, [("d0_stale.tag", {"door": 0, "action": 0, "when": 0})])
    red_a = make(
        s_red,
        w_clutter,
        policy,
        unique_writes=True,
        use_pick=True,
        force_use=True,
        force_write=True,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
    )
    red_a.reset_rho()
    red_live = live_free(red_a, "experience_teach", seed + 10, max_steps=max_steps)
    red_a.world = None
    red_a.explore_epsilon = 0.0
    red_a.reset_rho()
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)

    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    write_tag_notes(s_green, _conflict_notes(2, 1, 0))
    green_a = make(s_green, None, policy, use_pick=True, force_use=True)
    green_a.reset_rho()
    green_probe = probe(green_a, "probe_green", seed + 20)
    green_live = {"wrote": False, "opened": False, "files": green_a.store.list_files(), "n_forced": 0, "n_steps": 0, "actions": []}

    shutil.rmtree(s_all, ignore_errors=True)
    s_all.mkdir()
    write_tag_notes(s_all, _conflict_notes(0, 0, 2))
    all_a = make(s_all, None, policy, use_pick=False, force_use=True)
    all_a.reset_rho()
    apply_all_red = probe(all_a, "probe_red_with_key", seed + 10)

    empty = make(s_empty, None, policy, use_pick=True, force_use=True)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)

    off_a = make(
        s_off,
        None,
        policy,
        enabled=False,
        unique_writes=True,
        use_pick=True,
        force_use=True,
        force_write=True,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 2),
    )
    off_a.reset_rho()
    live_free(off_a, "experience_teach", seed + 10, max_steps=max_steps)
    off_a.reset_rho()
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    cortex1 = red_a.weight_hash()
    metrics: dict[str, Any] = {
        "arm": "A",
        "seed": seed,
        "n_train": n_train,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == cortex1,
        "pick_changed": pick0 != pick1,
        **_w_flags(w_files),
        "untrained_conflict": untrained_conflict,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "green_live": green_live,
        "green_probe": green_probe,
        "apply_all_red": apply_all_red,
        "empty_S": empty_p,
        "disable_S_red": disable_red,
        "red_files": sorted(p.name for p in s_red.glob("*.tag")),
        "green_files": sorted(p.name for p in s_green.glob("*.tag")),
    }
    label, rationale = classify_a(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_arm_b(run_dir: Path, w_clutter: Path, w_files: list[str], seed: int, n_train: int, max_steps: int) -> dict[str, Any]:
    work = run_dir / "B_train"
    s_un = run_dir / "B_untrained"
    s_red = run_dir / "B_red"
    s_green = run_dir / "B_green"
    s_inc = run_dir / "B_incomplete"
    s_off = run_dir / "B_off"
    s_empty = run_dir / "B_empty"
    for d in (work, s_un, s_red, s_green, s_inc, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    sch0 = str(policy.w_schema.tobytes().hex()) + str(float(policy.b_schema))
    dummy = make(run_dir / "B_hash", None, policy, write_schema=True, force_use=True, force_write=True)
    cortex0 = dummy.weight_hash()

    un_a = make(
        s_un,
        w_clutter,
        policy,
        write_schema=True,
        force_use=True,
        force_write=True,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 9),
    )
    un_a.reset_rho()
    un_live = live_free(un_a, "experience_teach", seed + 10, max_steps=max_steps)
    un_a.world = None
    un_a.explore_epsilon = 0.0
    un_a.reset_rho()
    untrained_red = probe(un_a, "probe_red_with_key", seed + 10)
    untrained_tag = ""
    for p in sorted(s_un.glob("*.tag")):
        untrained_tag += p.read_text(encoding="utf-8")

    rewards = train_schema(policy, w_clutter, work, n_train, seed, max_steps)
    sch1 = str(policy.w_schema.tobytes().hex()) + str(float(policy.b_schema))

    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    red_a = make(
        s_red,
        w_clutter,
        policy,
        write_schema=True,
        force_use=True,
        force_write=True,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
    )
    red_a.reset_rho()
    red_live = live_free(red_a, "experience_teach", seed + 10, max_steps=max_steps)
    red_a.world = None
    red_a.explore_epsilon = 0.0
    red_a.reset_rho()
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)
    red_tag = ""
    for p in sorted(s_red.glob("*.tag")):
        red_tag += p.read_text(encoding="utf-8")

    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    green_a = make(
        s_green,
        w_clutter,
        policy,
        write_schema=True,
        force_use=True,
        force_write=True,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 3),
    )
    green_a.reset_rho()
    green_live = live_free(green_a, "experience_green", seed + 20, max_steps=max_steps)
    green_a.world = None
    green_a.explore_epsilon = 0.0
    green_a.reset_rho()
    green_probe = probe(green_a, "probe_green", seed + 20)
    green_tag = ""
    for p in sorted(s_green.glob("*.tag")):
        green_tag += p.read_text(encoding="utf-8")

    write_tag_notes(s_inc, [("d0.tag", {"door": 0})])
    inc_a = make(s_inc, None, policy, write_schema=True, force_use=True)
    inc_a.reset_rho()
    incomplete_planted = probe(inc_a, "probe_red_with_key", seed + 10)

    empty = make(s_empty, None, policy, write_schema=True, force_use=True)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)

    off_a = make(
        s_off,
        None,
        policy,
        enabled=False,
        write_schema=True,
        force_use=True,
        force_write=True,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 2),
    )
    off_a.reset_rho()
    live_free(off_a, "experience_teach", seed + 10, max_steps=max_steps)
    off_a.reset_rho()
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    cortex1 = red_a.weight_hash()
    metrics: dict[str, Any] = {
        "arm": "B",
        "seed": seed,
        "n_train": n_train,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == cortex1,
        "schema_changed": sch0 != sch1,
        **_w_flags(w_files),
        "untrained_live": _slim(un_live),
        "untrained_red": untrained_red,
        "untrained_tag": untrained_tag,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "red_tag": red_tag,
        "green_live": _slim(green_live),
        "green_probe": green_probe,
        "green_tag": green_tag,
        "incomplete_planted": incomplete_planted,
        "empty_S": empty_p,
        "disable_S_red": disable_red,
    }
    label, rationale = classify_b(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v14(seed: int = 12345, n_train: int = 400, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    write_tag_notes(w_clutter, clutter_w_no_answers())
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))

    a = run_arm_a(run_dir, w_clutter, w_files, seed, n_train, max_steps)
    b = run_arm_b(run_dir, w_clutter, w_files, seed, n_train, max_steps)
    out = {
        "seed": seed,
        "n_train": n_train,
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v14 A vs B

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A pick-one | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B schema | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

Same cortex hash: {out['same_cortex']}

| Check | A | B |
|-------|---|---|
| Untrained red | {a['untrained_conflict']['action_name']} ({a['untrained_conflict']['correct']}) | {b['untrained_red']['action_name']} ({b['untrained_red']['correct']}) |
| Trained red | {a['red_probe']['action_name']} ({a['red_probe']['correct']}) | {b['red_probe']['action_name']} ({b['red_probe']['correct']}) |
| Held-out green | {a['green_probe']['action_name']} ({a['green_probe']['correct']}) | {b['green_probe']['action_name']} ({b['green_probe']['correct']}) |
| Empty S / disable-S | {a['empty_S']['correct']} / {a['disable_S_red']['correct']} | {b['empty_S']['correct']} / {b['disable_S_red']['correct']} |
| Head changed | pick={a['pick_changed']} | schema={b['schema_changed']} |
| Cortex unchanged | {a['cortex_unchanged']} | {b['cortex_unchanged']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v14 pick-one vs write schema")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v14(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print("A untrained", m["A"]["untrained_conflict"]["action_name"], "red", m["A"]["red_probe"]["action_name"], "green", m["A"]["green_probe"]["action_name"])
    print("B untrained", m["B"]["untrained_red"]["action_name"], "red", m["B"]["red_probe"]["action_name"], "green", m["B"]["green_probe"]["action_name"])
    print("B untrained tag", m["B"]["untrained_tag"].strip())
    print("B red tag", m["B"]["red_tag"].strip())
    print("B green tag", m["B"]["green_tag"].strip())


if __name__ == "__main__":
    main()
