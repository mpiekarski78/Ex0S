"""v15: joint write WHEN + schema + use-gate + pick-one. No clamps."""

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
    d = REPO_ROOT / "runs" / f"{stamp}_v15"
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
    use_pick: bool = True,
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
        unique_writes=True,
        use_pick=use_pick,
        write_schema=True,
        force_use=False,
        force_write=False,
    )


def _slim(lv: dict[str, Any]) -> dict[str, Any]:
    return {k: lv[k] for k in ("wrote", "opened", "files", "n_forced", "n_steps", "actions")}


def _head_fp(policy: UsePolicy, name: str) -> str:
    w = getattr(policy, f"w_{name}")
    b = getattr(policy, f"b_{name}")
    return str(w.tobytes().hex()) + str(float(b))


def classify(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer file was in W."
    if m["red_live"]["n_forced"] or m["green_live"]["n_forced"]:
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_conflict"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if m["force_use"] or m["force_write"]:
        return "Fail", "A gate was clamped on to rescue the plot."
    if not all(m[k] for k in ("write_changed", "schema_changed", "use_changed", "pick_changed")):
        return "Fail", "A skill head did not change."
    if m["untrained_conflict"]["correct"]:
        return "Fail", "Untrained already used the key."
    if not m["red_probe"]["correct"]:
        return "Fail", "Joint trained red did not use_key."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; a head memorized red, not the file."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["apply_all_red"]["correct"]:
        return "Fail", "Apply-all still correct on red; mix should hurt."
    pol = m["red_probe"].get("policy") or {}
    if pol.get("use") is False:
        return "Fail", "Probe did not gate use=True."
    if pol.get("one") is False:
        return "Fail", "Probe did not gate pick-one."
    return (
        "Store-works",
        "Write, schema, use, and pick learned together; cortex frozen; held-out green worked; apply-all still mixes.",
    )


def train_joint(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_w = b_s = b_u = b_p = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"ep_{ep}"
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
        )
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        wrote = any(t.get("kind") == "write" and t.get("write") for t in ag.policy_traces)
        complete = any(t.get("kind") == "schema" and t.get("complete") for t in ag.policy_traces)
        r_write = 1.0 if wrote else 0.0
        r_schema = 1.0 if complete else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        r_pick = 1.0 if p["correct"] else 0.0
        b_w = 0.9 * b_w + 0.1 * r_write
        b_s = 0.9 * b_s + 0.1 * r_schema
        b_u = 0.9 * b_u + 0.1 * r_use
        b_p = 0.9 * b_p + 0.1 * r_pick
        traces = ag.policy_traces
        policy.update([t for t in traces if t.get("kind") == "write"], r_write - b_w)
        policy.update([t for t in traces if t.get("kind") == "schema"], r_schema - b_s)
        policy.update([t for t in traces if t.get("kind") == "use"], r_use - b_u)
        policy.update([t for t in traces if t.get("kind") == "pick"], r_pick - b_p)
        rewards.append(r_use)
    return rewards


def one_life(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    scenario: str,
    seed: int,
    *,
    stale: tuple[str, dict[str, Any]] | None,
    rng: np.random.Generator,
    max_steps: int,
    enabled: bool = True,
    use_pick: bool = True,
) -> tuple[ThreeMemoryAgent, dict[str, Any]]:
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    if stale is not None:
        write_tag_notes(s_dir, [stale])
    ag = make(
        s_dir,
        w_dir,
        policy,
        enabled=enabled,
        epsilon=0.0,
        explore_epsilon=0.5,
        rng=rng,
        use_pick=use_pick,
    )
    ag.reset_rho()
    live = live_free(ag, scenario, seed, max_steps=max_steps)
    ag.world = None
    ag.explore_epsilon = 0.0
    ag.reset_rho()
    return ag, live


def run_v15(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_clutter = run_dir / "W"
    work = run_dir / "train"
    s_un = run_dir / "S_untrained"
    s_red = run_dir / "S_red"
    s_green = run_dir / "S_green"
    s_all = run_dir / "S_apply_all"
    s_off = run_dir / "S_off"
    s_empty = run_dir / "S_empty"
    write_tag_notes(w_clutter, clutter_w_no_answers())
    w_files = sorted(p.name for p in w_clutter.glob("*.tag"))
    for d in (work, s_un, s_red, s_green, s_all, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    fp0 = {k: _head_fp(policy, k) for k in ("write", "schema", "use", "pick")}
    dummy = make(run_dir / "empty_hash", None, policy)
    cortex0 = dummy.weight_hash()

    write_tag_notes(
        s_un,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    un_a = make(s_un, None, policy)
    un_a.reset_rho()
    untrained_conflict = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = train_joint(policy, w_clutter, work, n_train, seed, max_steps)
    fp1 = {k: _head_fp(policy, k) for k in ("write", "schema", "use", "pick")}

    red_a, red_live = one_life(
        s_red,
        w_clutter,
        policy,
        "experience_teach",
        seed + 10,
        stale=("d0_stale.tag", {"door": 0, "action": 0, "when": 0}),
        rng=np.random.default_rng(seed + 1),
        max_steps=max_steps,
    )
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)

    green_a, green_live = one_life(
        s_green,
        w_clutter,
        policy,
        "experience_green",
        seed + 20,
        stale=("d2_stale.tag", {"door": 2, "action": 1, "when": 0}),
        rng=np.random.default_rng(seed + 3),
        max_steps=max_steps,
    )
    green_probe = probe(green_a, "probe_green", seed + 20)

    shutil.rmtree(s_all, ignore_errors=True)
    shutil.copytree(s_red, s_all)
    all_a = make(s_all, None, policy, use_pick=False)
    all_a.reset_rho()
    apply_all_red = probe(all_a, "probe_red_with_key", seed + 10)

    empty = make(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)

    off_a, _ = one_life(
        s_off,
        None,
        policy,
        "experience_teach",
        seed + 10,
        stale=None,
        rng=np.random.default_rng(seed + 2),
        max_steps=max_steps,
        enabled=False,
    )
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    def _tags(folder: Path) -> str:
        return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))

    metrics: dict[str, Any] = {
        "seed": seed,
        "n_train": n_train,
        "force_use": False,
        "force_write": False,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "write_changed": fp0["write"] != fp1["write"],
        "schema_changed": fp0["schema"] != fp1["schema"],
        "use_changed": fp0["use"] != fp1["use"],
        "pick_changed": fp0["pick"] != fp1["pick"],
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
        "untrained_conflict": untrained_conflict,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "red_tags": _tags(s_red),
        "green_live": _slim(green_live),
        "green_probe": green_probe,
        "green_tags": _tags(s_green),
        "apply_all_red": apply_all_red,
        "empty_S": empty_p,
        "disable_S_red": disable_red,
    }
    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    pol = red_probe.get("policy") or {}
    (run_dir / "summary.md").write_text(
        f"""# v15 joint

Classification: **{label}**

{rationale}

| Check | Result |
|-------|--------|
| Cortex unchanged | {metrics['cortex_unchanged']} |
| Heads changed | write={metrics['write_changed']} schema={metrics['schema_changed']} use={metrics['use_changed']} pick={metrics['pick_changed']} |
| Untrained conflict | {untrained_conflict['correct']} ({untrained_conflict['action_name']}) |
| Red after ρ reset | {red_probe['correct']} ({red_probe['action_name']}, use={pol.get('use')}, one={pol.get('one')}) |
| Held-out green | {green_probe['correct']} ({green_probe['action_name']}) |
| Apply-all red | {apply_all_red['correct']} ({apply_all_red['action_name']}) |
| Empty S / disable-S | {empty_p['correct']} / {disable_red['correct']} |
| Train last 50 | {metrics['train_return_last50']:.2f} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v15 joint write/schema/use/pick")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v15(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "classification": m["classification"],
                "rationale": m["rationale"],
                "run_dir": m["run_dir"],
                "train_return_last50": m["train_return_last50"],
            },
            indent=2,
        )
    )
    print("untrained", m["untrained_conflict"]["action_name"], m["untrained_conflict"]["correct"])
    print("red", m["red_probe"]["action_name"], m["red_probe"]["correct"], m["red_tags"].strip())
    print("green", m["green_probe"]["action_name"], m["green_probe"]["correct"], m["green_tags"].strip())
    print("apply-all", m["apply_all_red"]["action_name"], m["apply_all_red"]["correct"])


if __name__ == "__main__":
    main()
