"""v20: A find unread W. B find vs junk on the default index. Cortex frozen."""

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

# Unread wiki pages. Not d0.tag / d2.tag — filename is not the genome answer.
WIKI_RED = ("p99.tag", {"here": DOOR_RED, "action": ACT_USE_KEY})
WIKI_GREEN = ("p98.tag", {"here": DOOR_GREEN, "action": ACT_WAIT})
JUNK_RED = ("junk.tag", {"door": DOOR_RED, "action": ACT_WAIT})
JUNK_GREEN = ("junkg.tag", {"door": DOOR_GREEN, "action": ACT_OPEN})


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v20"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    return probe_v12(agent, scenario, seed)


def wiki_notes(*, include_red: bool = False, include_green: bool = False, junk: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = list(clutter_w_no_answers())
    if include_red:
        notes.append(WIKI_RED)
    if include_green:
        notes.append(WIKI_GREEN)
    if junk and include_red:
        notes.append(JUNK_RED)
    if junk and include_green:
        notes.append(JUNK_GREEN)
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
        "w_has_wiki_red": WIKI_RED[0] in w_files,
        "w_has_junk_red": JUNK_RED[0] in w_files,
    }


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    force_use: bool = False,
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
        use_match_head=True,
        force_use=force_use,
    )


def _copy_use(src: UsePolicy) -> UsePolicy:
    """Trained use-gate, untrained match (still door=)."""
    ctrl = UsePolicy(seed=7, lr=0.2)
    ctrl.w_use = src.w_use.copy()
    ctrl.b_use = np.array(float(src.b_use))
    return ctrl


def _s_has_here(folder: Path) -> bool:
    return "here=" in _tags(folder)


def _commit_unmount(
    s_dir: Path,
    w_dir: Path,
    policy: UsePolicy,
    scenario: str,
    seed: int,
    rng: np.random.Generator | None = None,
    enabled: bool = True,
) -> tuple[ThreeMemoryAgent, dict[str, Any], dict[str, Any], str]:
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make(s_dir, w_dir, policy, enabled=enabled, epsilon=0.0, rng=rng)
    ag.reset_rho()
    first = probe(ag, scenario, seed)
    tag = _tags(s_dir)
    reload = s_dir.parent / f"{s_dir.name}_reload"
    shutil.rmtree(reload, ignore_errors=True)
    shutil.copytree(s_dir, reload)
    only = make(reload, None, policy, enabled=enabled, epsilon=0.0, rng=rng)
    only.reset_rho()
    unmount = probe(only, scenario, seed)
    return ag, first, unmount, tag


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if not m["w_has_wiki_red"]:
        return "Confound", "A train W had no unread red wiki page."
    if m["w_has_junk_red"]:
        return "Confound", "A train W included the door= junk page."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on; find is not isolated."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue the plot."
    if not m["use_match_head"]:
        return "Fail", "Match head was frozen off to rescue the plot."
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key (or table still wired)."
    if (m["untrained"].get("policy") or {}).get("match_alt") is True:
        return "Fail", "Untrained already matched here=; index was frozen to rescue."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Trained policy did not use the red wiki page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green wiki page failed; policy learned that door, not find."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["match_control"]["correct"] or m["match_control"]["action_name"] == "use_key":
        return "Fail", "Trained use-gate with untrained door= match still solved red."
    if not m["match_changed"]:
        return "Fail", "Match head did not move."
    if not m["use_changed"]:
        return "Fail", "Use-gate did not move."
    if "here=" not in m.get("red_tag", ""):
        return "Fail", "Red S has no here= page; commit did not keep the wiki note."
    if "d0.tag" in (m["red_unmount"].get("files") or []):
        return "Fail", "Answer landed in d0.tag; filename genome."
    alt = (m["red_unmount"].get("policy") or {}).get("match_alt")
    if alt is False:
        return "Fail", "Probe still matched door= after training."
    return (
        "Store-works",
        "Found unread here= wiki page; cortex frozen; unmount W; held-out green wait; door= control fails.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if not m["w_has_wiki_red"] or not m["w_has_junk_red"]:
        return "Confound", "B train W must have both the here= wiki page and door= junk."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on; find is not isolated."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue the plot."
    if not m["use_match_head"]:
        return "Fail", "Match head was frozen off to rescue the plot."
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key (or table still wired)."
    if (m["untrained"].get("policy") or {}).get("match_alt") is True:
        return "Fail", "Untrained already matched here=; index was frozen to rescue."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Trained policy did not prefer the here= wiki page over door= junk."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green wiki page failed; policy learned that door, not find."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["match_control"]["correct"] or m["match_control"]["action_name"] == "use_key":
        return "Fail", "Trained use-gate with untrained door= match still solved red (junk ignored)."
    if m["junk_only"]["correct"] or m["junk_only"]["action_name"] == "use_key":
        return "Fail", "Junk-only W still use_key; fact leaked into the policy."
    if not m["match_changed"]:
        return "Fail", "Match head did not move."
    if not m["use_changed"]:
        return "Fail", "Use-gate did not move."
    if "here=" not in m.get("red_tag", ""):
        return "Fail", "Red S has no here= page; committed junk instead of the wiki note."
    if "d0.tag" in (m["red_unmount"].get("files") or []):
        return "Fail", "Answer landed in d0.tag; filename genome."
    alt = (m["red_unmount"].get("policy") or {}).get("match_alt")
    if alt is False:
        return "Fail", "Probe still matched door= after training."
    return (
        "Store-works",
        "Found here= wiki page over door= junk; cortex frozen; unmount W; held-out green wait; junk/door= controls fail.",
    )


def _train(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    scenario: str,
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_c = b_m = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_dir, policy, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        probe(ag, scenario, seed + 10)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, scenario, seed + 10)
        tr = ag.policy_traces
        r_found = 1.0 if _s_has_here(s_dir) else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        b_c = 0.9 * b_c + 0.1 * r_found
        b_m = 0.9 * b_m + 0.1 * r_use
        policy.update([t for t in tr if t.get("kind") == "match"], r_found - b_c)
        policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_m)
        rewards.append(r_use)
    return rewards


def run_arm(
    *,
    arm: str,
    run_dir: Path,
    w_red: Path,
    w_green: Path,
    w_junk: Path | None,
    w_files: list[str],
    seed: int,
    n_train: int,
    train_seed: int,
) -> dict[str, Any]:
    work = run_dir / f"{arm}_train"
    s_un = run_dir / f"{arm}_untrained"
    s_red = run_dir / f"{arm}_red"
    s_green = run_dir / f"{arm}_green"
    s_ctrl = run_dir / f"{arm}_matchctrl"
    s_off = run_dir / f"{arm}_off"
    s_empty = run_dir / f"{arm}_empty"
    s_junk = run_dir / f"{arm}_junkonly"
    for d in (work, s_un, s_red, s_green, s_ctrl, s_off, s_empty, s_junk):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    match0 = _head_fp(policy, "match")
    use0 = _head_fp(policy, "use")
    dummy = make(run_dir / f"{arm}_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a = make(s_un, w_red, policy, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train(policy, w_red, work, n_train, train_seed, "probe_red_with_key")
    match1 = _head_fp(policy, "match")
    use1 = _head_fp(policy, "use")

    red_a, red_first, red_unmount, red_tag = _commit_unmount(
        s_red, w_red, policy, "probe_red_with_key", seed + 10
    )
    green_a, green_first, green_unmount, green_tag = _commit_unmount(
        s_green, w_green, policy, "probe_green", seed + 20
    )

    ctrl = _copy_use(policy)
    _, _, match_control, ctrl_tag = _commit_unmount(
        s_ctrl, w_red, ctrl, "probe_red_with_key", seed + 10
    )

    empty = make(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    empty_g = probe(empty, "probe_green", seed + 20)

    off = make(s_off, None, policy, enabled=False)
    off.reset_rho()
    disable_red = probe(off, "probe_red_with_key", seed + 10)

    junk_only = {"correct": False, "action_name": "n/a", "explored": False}
    junk_tag = ""
    if w_junk is not None:
        _, _, junk_only, junk_tag = _commit_unmount(
            s_junk, w_junk, policy, "probe_red_with_key", seed + 10
        )

    metrics: dict[str, Any] = {
        "arm": arm,
        "trained_force_use": False,
        "write_from_events": dummy.write_from_events,
        "use_match_head": dummy.use_match_head,
        "use_read": dummy.use_read,
        "collect_mode": dummy.collect_mode,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "match_changed": match0 != match1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        "red_s_dir": str(s_red),
        **_w_flags(w_files),
        "untrained": untrained,
        "red_first": red_first,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_first": green_first,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "match_control": match_control,
        "ctrl_tag": ctrl_tag,
        "empty_S": empty_p,
        "empty_S_green": empty_g,
        "disable_S_red": disable_red,
        "junk_only": junk_only,
        "junk_tag": junk_tag,
    }
    label, rationale = (classify_a if arm == "A" else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v20(seed: int = 12345, n_train: int = 500) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_A"
    w_b = run_dir / "W_B"
    w_ag = run_dir / "W_A_green"
    w_bg = run_dir / "W_B_green"
    w_junk = run_dir / "W_junk"
    write_tag_notes(w_a, wiki_notes(include_red=True, junk=False))
    write_tag_notes(w_b, wiki_notes(include_red=True, junk=True))
    write_tag_notes(w_ag, wiki_notes(include_green=True, junk=False))
    write_tag_notes(w_bg, wiki_notes(include_green=True, junk=True))
    write_tag_notes(w_junk, wiki_notes(include_red=False, junk=False) + [JUNK_RED])
    a_files = sorted(p.name for p in w_a.glob("*.tag"))
    b_files = sorted(p.name for p in w_b.glob("*.tag"))
    a = run_arm(
        arm="A",
        run_dir=run_dir,
        w_red=w_a,
        w_green=w_ag,
        w_junk=None,
        w_files=a_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed,
    )
    b = run_arm(
        arm="B",
        run_dir=run_dir,
        w_red=w_b,
        w_green=w_bg,
        w_junk=w_junk,
        w_files=b_files,
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
        f"""# v20 A find vs B find-vs-junk

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A find unread `here=` page | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B find vs `door=` junk | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained | {a['untrained']['action_name']} ({a['untrained']['correct']}) | {b['untrained']['action_name']} ({b['untrained']['correct']}) |
| Trained red, unmount W | {a['red_unmount']['action_name']} ({a['red_unmount']['correct']}) | {b['red_unmount']['action_name']} ({b['red_unmount']['correct']}) |
| Held-out green, unmount W | {a['green_unmount']['action_name']} ({a['green_unmount']['correct']}) | {b['green_unmount']['action_name']} ({b['green_unmount']['correct']}) |
| Match control (door=) | {a['match_control']['action_name']} ({a['match_control']['correct']}) | {b['match_control']['action_name']} ({b['match_control']['correct']}) |
| Junk-only W | n/a | {b['junk_only']['action_name']} ({b['junk_only']['correct']}) |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v20 find in W vs find vs junk")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    args = p.parse_args()
    m = run_v20(seed=args.seed, n_train=args.n_train)
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
        "junk",
        m["B"]["junk_only"]["action_name"],
    )


if __name__ == "__main__":
    main()
