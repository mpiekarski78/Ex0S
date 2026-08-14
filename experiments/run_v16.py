"""v16: A rank ok= vs newest. B shared return vs v15 split credit."""

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
from three_memory.policy import UsePolicy
from three_memory.symbols import BLUE_FACT_ID, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v16"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    return probe_v12(agent, scenario, seed)


def _head_fp(policy: UsePolicy, name: str) -> str:
    if name == "rank":
        return str(policy.w_rank.tobytes().hex())
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


def plant_junk(s_dir: Path, door: int, action: int, when: int = 999) -> None:
    write_tag_notes(s_dir, [(f"d{door}_junk.tag", {"door": door, "action": action, "when": when})])


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
        unique_writes=True,
        use_pick=False,
        use_rank=True,
        write_schema=True,
        mark_ok=True,
        force_use=force_use,
        force_write=False,
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
    use_pick: bool = True,
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
        unique_writes=True,
        use_pick=use_pick,
        write_schema=True,
        force_use=False,
        force_write=False,
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
    if any(p.get("explored") for p in (m["untrained_newest"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if m["force_write"] or m["trained_force_use"]:
        return "Fail", "A gate was clamped on to rescue the trained plot."
    if not m["rank_changed"]:
        return "Fail", "Rank head did not change."
    if m["untrained_newest"]["correct"] or m["untrained_newest"]["action_name"] == "use_key":
        return "Fail", "Untrained recency prior did not follow newest (wrong)."
    if not m["red_probe"]["correct"]:
        return "Fail", "Trained rank did not prefer ok=1 on red."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed; rank learned that door, not ok=."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["newest_control"]["correct"]:
        return "Fail", "Newest-wins still solves red; junk should hurt."
    pol = m["red_probe"].get("policy") or {}
    if pol.get("has_ok") is False:
        return "Fail", "Probe did not pick the ok=1 note."
    return (
        "Store-works",
        "Rank learned to prefer ok=1 over newest; cortex frozen; held-out green worked; newest-wins still fails.",
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
    if any(p.get("explored") for p in (m["untrained_conflict"], m["red_probe"], m["green_probe"])):
        return "Confound", "Probe used exploration."
    if not all(m[k] for k in ("write_changed", "schema_changed", "use_changed", "pick_changed")):
        return "Fail", "A skill head did not change under shared return."
    if m["untrained_conflict"]["correct"]:
        return "Fail", "Untrained already used the key."
    if not m["red_probe"]["correct"]:
        return "Fail", "Shared return did not solve red (write head likely starved)."
    if not m["green_probe"]["correct"]:
        return "Fail", "Held-out green failed under shared return."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["apply_all_red"]["correct"]:
        return "Fail", "Apply-all still correct on red."
    return (
        "Store-works",
        "Shared return still solved red and held-out green; split credit was not required.",
    )


def train_a(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_w = b_s = b_u = b_r = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"a_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make_a(
            s_dir, w_clutter, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng
        )
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        plant_junk(s_dir, 0, 0)
        ag.store.reload()
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        wrote = any(t.get("kind") == "write" and t.get("write") for t in ag.policy_traces)
        complete = any(t.get("kind") == "schema" and t.get("complete") for t in ag.policy_traces)
        r_write = 1.0 if wrote else 0.0
        r_schema = 1.0 if complete else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        r_rank = 1.0 if p["correct"] else 0.0
        b_w = 0.9 * b_w + 0.1 * r_write
        b_s = 0.9 * b_s + 0.1 * r_schema
        b_u = 0.9 * b_u + 0.1 * r_use
        b_r = 0.9 * b_r + 0.1 * r_rank
        tr = ag.policy_traces
        policy.update([t for t in tr if t.get("kind") == "write"], r_write - b_w)
        policy.update([t for t in tr if t.get("kind") == "schema"], r_schema - b_s)
        policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_u)
        policy.update([t for t in tr if t.get("kind") == "rank"], r_rank - b_r)
        rewards.append(r_use)
    return rewards


def train_b_shared(
    policy: UsePolicy, w_clutter: Path, work: Path, n: int, seed: int, max_steps: int
) -> list[float]:
    rng = np.random.default_rng(seed + 11)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        s_dir = work / f"b_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        write_tag_notes(s_dir, [("d0_stale.tag", {"door": 0, "action": 0, "when": 0})])
        ag = make_b(
            s_dir, w_clutter, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng
        )
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r
        policy.update(ag.policy_traces, r - baseline)
        rewards.append(r)
    return rewards


def run_arm_a(run_dir: Path, w_clutter: Path, w_files: list[str], seed: int, n_train: int, max_steps: int) -> dict[str, Any]:
    work = run_dir / "A_train"
    s_un = run_dir / "A_untrained"
    s_red = run_dir / "A_red"
    s_green = run_dir / "A_green"
    s_ctrl = run_dir / "A_newest"
    s_off = run_dir / "A_off"
    s_empty = run_dir / "A_empty"
    for d in (work, s_un, s_red, s_green, s_ctrl, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    rank0 = _head_fp(policy, "rank")
    dummy = make_a(run_dir / "A_hash", None, policy)
    cortex0 = dummy.weight_hash()

    write_tag_notes(
        s_un,
        [
            ("d0_ok.tag", {"door": 0, "action": 2, "when": 1, "ok": 1}),
            ("d0_new.tag", {"door": 0, "action": 0, "when": 10}),
        ],
    )
    un_a = make_a(s_un, None, policy, force_use=True)
    un_a.reset_rho()
    untrained_newest = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = train_a(policy, w_clutter, work, n_train, seed, max_steps)
    rank1 = _head_fp(policy, "rank")

    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    red_a = make_a(
        s_red, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 1)
    )
    red_a.reset_rho()
    red_live = live_free(red_a, "experience_teach", seed + 10, max_steps=max_steps)
    plant_junk(s_red, 0, 0)
    red_a.store.reload()
    red_a.world = None
    red_a.explore_epsilon = 0.0
    red_a.reset_rho()
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)

    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    green_a = make_a(
        s_green, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 3)
    )
    green_a.reset_rho()
    green_live = live_free(green_a, "experience_green", seed + 20, max_steps=max_steps)
    plant_junk(s_green, 2, 1)
    green_a.store.reload()
    green_a.world = None
    green_a.explore_epsilon = 0.0
    green_a.reset_rho()
    green_probe = probe(green_a, "probe_green", seed + 20)

    shutil.rmtree(s_ctrl, ignore_errors=True)
    shutil.copytree(s_red, s_ctrl)
    ctrl = UsePolicy(seed=7, lr=0.2)
    ctrl.w_write = policy.w_write.copy()
    ctrl.b_write = np.array(float(policy.b_write))
    ctrl.w_schema = policy.w_schema.copy()
    ctrl.b_schema = np.array(float(policy.b_schema))
    ctrl.w_use = policy.w_use.copy()
    ctrl.b_use = np.array(float(policy.b_use))
    ctrl.w_rank = np.array([1.2, 0.0], dtype=np.float64)
    ctrl_a = make_a(s_ctrl, None, ctrl)
    ctrl_a.reset_rho()
    newest_control = probe(ctrl_a, "probe_red_with_key", seed + 10)

    empty = make_a(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)

    off_a = make_a(
        s_off,
        None,
        policy,
        enabled=False,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 2),
    )
    off_a.reset_rho()
    live_free(off_a, "experience_teach", seed + 10, max_steps=max_steps)
    off_a.reset_rho()
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": "A",
        "seed": seed,
        "n_train": n_train,
        "force_write": False,
        "trained_force_use": False,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "rank_changed": rank0 != rank1,
        **_w_flags(w_files),
        "untrained_newest": untrained_newest,
        "red_live": _slim(red_live),
        "red_probe": red_probe,
        "red_tags": _tags(s_red),
        "green_live": _slim(green_live),
        "green_probe": green_probe,
        "green_tags": _tags(s_green),
        "newest_control": newest_control,
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
    s_red = run_dir / "B_red"
    s_green = run_dir / "B_green"
    s_all = run_dir / "B_apply_all"
    s_off = run_dir / "B_off"
    s_empty = run_dir / "B_empty"
    for d in (work, s_un, s_red, s_green, s_all, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    fp0 = {k: _head_fp(policy, k) for k in ("write", "schema", "use", "pick")}
    dummy = make_b(run_dir / "B_hash", None, policy)
    cortex0 = dummy.weight_hash()

    write_tag_notes(
        s_un,
        [
            ("d0_t1.tag", {"door": 0, "action": 0, "when": 1}),
            ("d0_t10.tag", {"door": 0, "action": 2, "when": 10}),
        ],
    )
    un_a = make_b(s_un, None, policy)
    un_a.reset_rho()
    untrained_conflict = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = train_b_shared(policy, w_clutter, work, n_train, seed, max_steps)
    fp1 = {k: _head_fp(policy, k) for k in ("write", "schema", "use", "pick")}

    shutil.rmtree(s_red, ignore_errors=True)
    s_red.mkdir()
    write_tag_notes(s_red, [("d0_stale.tag", {"door": 0, "action": 0, "when": 0})])
    red_a = make_b(
        s_red, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 1)
    )
    red_a.reset_rho()
    red_live = live_free(red_a, "experience_teach", seed + 10, max_steps=max_steps)
    red_a.world = None
    red_a.explore_epsilon = 0.0
    red_a.reset_rho()
    red_probe = probe(red_a, "probe_red_with_key", seed + 10)

    shutil.rmtree(s_green, ignore_errors=True)
    s_green.mkdir()
    write_tag_notes(s_green, [("d2_stale.tag", {"door": 2, "action": 1, "when": 0})])
    green_a = make_b(
        s_green, w_clutter, policy, explore_epsilon=0.5, rng=np.random.default_rng(seed + 3)
    )
    green_a.reset_rho()
    green_live = live_free(green_a, "experience_green", seed + 20, max_steps=max_steps)
    green_a.world = None
    green_a.explore_epsilon = 0.0
    green_a.reset_rho()
    green_probe = probe(green_a, "probe_green", seed + 20)

    shutil.rmtree(s_all, ignore_errors=True)
    shutil.copytree(s_red, s_all)
    all_a = make_b(s_all, None, policy, use_pick=False)
    all_a.reset_rho()
    apply_all_red = probe(all_a, "probe_red_with_key", seed + 10)

    empty = make_b(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)

    off_a = make_b(
        s_off,
        None,
        policy,
        enabled=False,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 2),
    )
    off_a.reset_rho()
    live_free(off_a, "experience_teach", seed + 10, max_steps=max_steps)
    off_a.reset_rho()
    disable_red = probe(off_a, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": "B",
        "seed": seed,
        "n_train": n_train,
        "shared_return": True,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "write_changed": fp0["write"] != fp1["write"],
        "schema_changed": fp0["schema"] != fp1["schema"],
        "use_changed": fp0["use"] != fp1["use"],
        "pick_changed": fp0["pick"] != fp1["pick"],
        **_w_flags(w_files),
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
    label, rationale = classify_b(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v16(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        f"""# v16 A vs B

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A ok= vs newest | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

Same cortex hash: {out['same_cortex']}

| Check | A | B |
|-------|---|---|
| Untrained red | {a['untrained_newest']['action_name']} ({a['untrained_newest']['correct']}) | {b['untrained_conflict']['action_name']} ({b['untrained_conflict']['correct']}) |
| Trained red | {a['red_probe']['action_name']} ({a['red_probe']['correct']}) | {b['red_probe']['action_name']} ({b['red_probe']['correct']}) |
| Held-out green | {a['green_probe']['action_name']} ({a['green_probe']['correct']}) | {b['green_probe']['action_name']} ({b['green_probe']['correct']}) |
| Control | newest {a['newest_control']['action_name']} ({a['newest_control']['correct']}) | apply-all {b['apply_all_red']['action_name']} ({b['apply_all_red']['correct']}) |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v16 rank ok= vs newest; shared return")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_v16(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print("A untrained", m["A"]["untrained_newest"]["action_name"], "red", m["A"]["red_probe"]["action_name"], "green", m["A"]["green_probe"]["action_name"], "newest-ctrl", m["A"]["newest_control"]["action_name"])
    print("B untrained", m["B"]["untrained_conflict"]["action_name"], "red", m["B"]["red_probe"]["action_name"], "green", m["B"]["green_probe"]["action_name"])


if __name__ == "__main__":
    main()
