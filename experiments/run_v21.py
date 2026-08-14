"""v21: A first-file W hit vs B dump-all W hits. Newest when= is the frozen pick."""

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

from experiments.run_v12 import clutter_w_no_answers, probe as probe_v12
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.symbols import ACT_OPEN, ACT_USE_KEY, ACT_WAIT, BLUE_FACT_ID, DOOR_GREEN, DOOR_RED, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes

# Same here= index, several pages. Filename-first is junk. Newest when= is useful.
# Not d0.tag / d2.tag.
RED_JUNK = ("aaa.tag", {"here": DOOR_RED, "action": ACT_WAIT, "when": 1})
RED_USE = ("p99.tag", {"here": DOOR_RED, "action": ACT_USE_KEY, "when": 9})
GREEN_JUNK = ("aag.tag", {"here": DOOR_GREEN, "action": ACT_OPEN, "when": 1})
GREEN_WAIT = ("p98.tag", {"here": DOOR_GREEN, "action": ACT_WAIT, "when": 9})
RED_SWAP = (
    ("aaa.tag", {"here": DOOR_RED, "action": ACT_WAIT, "when": 9}),
    ("p99.tag", {"here": DOOR_RED, "action": ACT_USE_KEY, "when": 1}),
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v21"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    return probe_v12(agent, scenario, seed)


def wiki_notes(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = list(clutter_w_no_answers())
    if include_red:
        notes.extend([RED_JUNK, RED_USE])
    if include_green:
        notes.extend([GREEN_JUNK, GREEN_WAIT])
    return notes


def _head_fp(policy: UsePolicy, name: str) -> str:
    w = getattr(policy, f"w_{name}")
    b = getattr(policy, f"b_{name}")
    return str(w.tobytes().hex()) + str(float(b) if np.ndim(b) == 0 or b.size == 1 else b.tobytes().hex())


def _tags(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.tag")))


def _w_flags(w_files: list[str]) -> dict[str, Any]:
    return {
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
        "w_has_aaa": RED_JUNK[0] in w_files,
        "w_has_p99": RED_USE[0] in w_files,
    }


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    dump: bool,
    enabled: bool = True,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    force_use: bool = True,
) -> ThreeMemoryAgent:
    world = TagLibrary(w_dir) if w_dir is not None else None
    return ThreeMemoryAgent(
        store_enabled=enabled,
        cortex_seed=1337,
        native=True,
        retrieve_policy="select",
        collect_mode="commit",
        store=TagStore(s_dir, enabled=enabled),
        world=world,
        use_policy=policy,
        write_from_events=False,
        policy_epsilon=epsilon,
        policy_rng=rng,
        use_read=True,
        place_key="here",
        use_wsel_head=True,
        wsel_dump=dump,
        force_use=force_use,
    )


def _commit_unmount(
    s_dir: Path,
    w_dir: Path,
    policy: UsePolicy,
    dump: bool,
    scenario: str,
    seed: int,
    enabled: bool = True,
) -> tuple[ThreeMemoryAgent, dict[str, Any], dict[str, Any], str]:
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make(s_dir, w_dir, policy, dump=dump, enabled=enabled, epsilon=0.0)
    ag.reset_rho()
    first = probe(ag, scenario, seed)
    tag = _tags(s_dir)
    reload = s_dir.parent / f"{s_dir.name}_reload"
    shutil.rmtree(reload, ignore_errors=True)
    shutil.copytree(s_dir, reload)
    only = make(reload, None, policy, dump=dump, enabled=enabled, epsilon=0.0)
    only.reset_rho()
    unmount = probe(only, scenario, seed)
    return ag, first, unmount, tag


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if not m["w_has_aaa"] or not m["w_has_p99"]:
        return "Confound", "A train W must have both filename-first junk and newest useful."
    if m["wsel_dump"]:
        return "Confound", "A used dump-all as the untrained prior."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if not m["trained_force_use"]:
        return "Fail", "Use-gate was unclamped; v21 isolates which W page to keep."
    if m["place_key"] != "here":
        return "Fail", "Query name was not frozen to here=."
    if not m["use_wsel_head"]:
        return "Fail", "W-select head was frozen off to rescue the plot."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key (filename-first was not junk)."
    if (m["untrained"].get("policy") or {}).get("wsel_alt") is True:
        return "Fail", "Untrained already took newest; pick was frozen to rescue."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Trained policy did not keep the newest red wiki page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; policy learned that door, not newest-among-W."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["first_control"]["correct"] or m["first_control"]["action_name"] == "use_key":
        return "Fail", "Untrained first-file prior still solved red after training the rest."
    if m["recency_swap"]["correct"] or m["recency_swap"]["action_name"] == "use_key":
        return "Fail", "Newest-is-junk still use_key; memorized p99.tag not when=."
    if not m["wsel_changed"]:
        return "Fail", "W-select head did not move."
    if RED_USE[0] not in (m["red_unmount"].get("files") or []) and "action=2" not in m.get("red_tag", ""):
        return "Fail", "Red S does not contain the useful wiki page."
    if "aaa.tag" in (m["red_unmount"].get("files") or []) and "p99.tag" not in (m["red_unmount"].get("files") or []):
        return "Fail", "Red S kept filename-first junk only."
    alt = (m["red_unmount"].get("policy") or {}).get("wsel_alt")
    if alt is False:
        return "Fail", "Probe still took filename-first after training."
    return (
        "Store-works",
        "Kept newest here= wiki page over filename-first junk; cortex frozen; unmount W; held-out green wait.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if not m["w_has_aaa"] or not m["w_has_p99"]:
        return "Confound", "B train W must have both junk and useful here= pages."
    if not m["wsel_dump"]:
        return "Confound", "B did not use dump-all as the untrained prior."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if not m["trained_force_use"]:
        return "Fail", "Use-gate was unclamped; v21 isolates which W page to keep."
    if m["place_key"] != "here":
        return "Fail", "Query name was not frozen to here=."
    if not m["use_wsel_head"]:
        return "Fail", "W-select head was frozen off to rescue the plot."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained dump-all already used the key."
    if (m["untrained"].get("policy") or {}).get("wsel_alt") is True:
        return "Fail", "Untrained already took newest; pick was frozen to rescue."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Trained policy did not keep only the newest red wiki page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; dump still mixed or door was memorized."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["first_control"]["correct"] or m["first_control"]["action_name"] == "use_key":
        return "Fail", "Dump-all prior still solved red after training the rest."
    if m["recency_swap"]["correct"] or m["recency_swap"]["action_name"] == "use_key":
        return "Fail", "Newest-is-junk still use_key; memorized p99.tag not when=."
    if not m["wsel_changed"]:
        return "Fail", "W-select head did not move."
    files = m["red_unmount"].get("files") or []
    if RED_JUNK[0] in files and RED_USE[0] in files:
        return "Fail", "Red S still has both W hits; dump was not turned off."
    if "action=2" not in m.get("red_tag", ""):
        return "Fail", "Red S does not contain the useful wiki page."
    alt = (m["red_unmount"].get("policy") or {}).get("wsel_alt")
    if alt is False:
        return "Fail", "Probe still dumped or took first after training."
    return (
        "Store-works",
        "Kept newest here= wiki page instead of dumping every match; cortex frozen; unmount W; held-out green wait.",
    )


def _train(policy: UsePolicy, w_dir: Path, work: Path, n: int, seed: int, dump: bool) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_dir, policy, dump=dump, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        probe(ag, "probe_red_with_key", seed + 10)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r_use
        policy.update([t for t in ag.policy_traces if t.get("kind") == "wsel"], r_use - baseline)
        rewards.append(r_use)
    return rewards


def run_arm(
    *,
    arm: str,
    dump: bool,
    run_dir: Path,
    w_red: Path,
    w_green: Path,
    w_swap: Path,
    w_files: list[str],
    seed: int,
    n_train: int,
    train_seed: int,
) -> dict[str, Any]:
    work = run_dir / f"{arm}_train"
    s_un = run_dir / f"{arm}_untrained"
    s_red = run_dir / f"{arm}_red"
    s_green = run_dir / f"{arm}_green"
    s_ctrl = run_dir / f"{arm}_firstctrl"
    s_swap = run_dir / f"{arm}_swap"
    s_off = run_dir / f"{arm}_off"
    s_empty = run_dir / f"{arm}_empty"
    for d in (work, s_un, s_red, s_green, s_ctrl, s_swap, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    wsel0 = _head_fp(policy, "wsel")
    dummy = make(run_dir / f"{arm}_hash", None, policy, dump=dump)
    cortex0 = dummy.weight_hash()

    un_a = make(s_un, w_red, policy, dump=dump, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train(policy, w_red, work, n_train, train_seed, dump)
    wsel1 = _head_fp(policy, "wsel")

    red_a, red_first, red_unmount, red_tag = _commit_unmount(
        s_red, w_red, policy, dump, "probe_red_with_key", seed + 10
    )
    _, _, green_unmount, green_tag = _commit_unmount(
        s_green, w_green, policy, dump, "probe_green", seed + 20
    )

    ctrl = UsePolicy(seed=7, lr=0.2)
    _, _, first_control, ctrl_tag = _commit_unmount(
        s_ctrl, w_red, ctrl, dump, "probe_red_with_key", seed + 10
    )
    _, _, recency_swap, swap_tag = _commit_unmount(
        s_swap, w_swap, policy, dump, "probe_red_with_key", seed + 10
    )

    empty = make(s_empty, None, policy, dump=dump)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    empty_g = probe(empty, "probe_green", seed + 20)

    off = make(s_off, None, policy, dump=dump, enabled=False)
    off.reset_rho()
    disable_red = probe(off, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": arm,
        "wsel_dump": dummy.wsel_dump,
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_wsel_head": dummy.use_wsel_head,
        "place_key": dummy.place_key,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "wsel_changed": wsel0 != wsel1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files),
        "untrained": untrained,
        "red_first": red_first,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "first_control": first_control,
        "ctrl_tag": ctrl_tag,
        "recency_swap": recency_swap,
        "swap_tag": swap_tag,
        "empty_S": empty_p,
        "empty_S_green": empty_g,
        "disable_S_red": disable_red,
    }
    label, rationale = (classify_a if arm == "A" else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v21(seed: int = 12345, n_train: int = 400) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    w_swap = run_dir / "W_swap"
    write_tag_notes(w_red, wiki_notes(include_red=True))
    write_tag_notes(w_green, wiki_notes(include_green=True))
    write_tag_notes(w_swap, list(clutter_w_no_answers()) + list(RED_SWAP))
    w_files = sorted(p.name for p in w_red.glob("*.tag"))
    a = run_arm(
        arm="A",
        dump=False,
        run_dir=run_dir,
        w_red=w_red,
        w_green=w_green,
        w_swap=w_swap,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed,
    )
    b = run_arm(
        arm="B",
        dump=True,
        run_dir=run_dir,
        w_red=w_red,
        w_green=w_green,
        w_swap=w_swap,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
    )
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
        f"""# v21 A first-file vs B dump-all

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A filename-first vs newest | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B dump-all vs newest | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained | {a['untrained']['action_name']} ({a['untrained']['correct']}) | {b['untrained']['action_name']} ({b['untrained']['correct']}) |
| Trained red, unmount W | {a['red_unmount']['action_name']} ({a['red_unmount']['correct']}) | {b['red_unmount']['action_name']} ({b['red_unmount']['correct']}) |
| Held-out green, unmount W | {a['green_unmount']['action_name']} ({a['green_unmount']['correct']}) | {b['green_unmount']['action_name']} ({b['green_unmount']['correct']}) |
| First/dump control | {a['first_control']['action_name']} ({a['first_control']['correct']}) | {b['first_control']['action_name']} ({b['first_control']['correct']}) |
| Recency swap | {a['recency_swap']['action_name']} ({a['recency_swap']['correct']}) | {b['recency_swap']['action_name']} ({b['recency_swap']['correct']}) |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v21 select among unread W hits")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=400)
    args = p.parse_args()
    m = run_v21(seed=args.seed, n_train=args.n_train)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print(
        "A",
        m["A"]["untrained"]["action_name"],
        m["A"]["red_unmount"]["action_name"],
        m["A"]["green_unmount"]["action_name"],
        m["A"]["red_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["untrained"]["action_name"],
        m["B"]["red_unmount"]["action_name"],
        m["B"]["green_unmount"]["action_name"],
        m["B"]["red_tag"].strip().replace("\n", " | "),
    )


if __name__ == "__main__":
    main()
