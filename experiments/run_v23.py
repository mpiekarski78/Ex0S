"""v23: joint match+complete+use, no when=. A split credit. B shared return."""

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

from experiments.run_v12 import clutter_w_no_answers
from experiments.run_v20 import JUNK_GREEN, JUNK_RED
from experiments.run_v22 import (
    RED_STUB,
    RED_SWAP,
    _commit_unmount,
    _copy_heads,
    _head_fp,
    _tags,
    _w_flags,
    probe,
    stub_notes,
)
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v23"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wiki_notes(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = stub_notes(include_red=include_red, include_green=include_green)
    if include_red:
        notes.append(JUNK_RED)
    if include_green:
        notes.append(JUNK_GREEN)
    return notes


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
        use_wcomp_head=True,
        force_use=force_use,
    )


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if m["w_has_when"]:
        return "Confound", "Planted when=; recency cheat restored."
    if not (m["w_has_aaa"] and m["w_has_p99"] and m["w_has_junk"]):
        return "Confound", "W must have stub, complete here= page, and door= junk."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue the joint."
    if m["place_key"] != "door":
        return "Fail", "Match was frozen to here= to rescue the joint."
    if not (m["use_match_head"] and m["use_wcomp_head"] and m["use_read"]):
        return "Fail", "A joint head was frozen off to rescue the plot."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Split joint did not use the complete here= page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; a head learned that door."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["match_control"]["correct"] or m["match_control"]["action_name"] == "use_key":
        return "Fail", "Trained complete+use with untrained door= match still solved red."
    if m["stub_control"]["correct"] or m["stub_control"]["action_name"] == "use_key":
        return "Fail", "Trained match+use with filename-first stub still solved red."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Trained match+complete with use-gate off still solved red."
    if m["stub_only"]["correct"] or m["stub_only"]["action_name"] == "use_key":
        return "Fail", "Stub-only W still use_key; fact leaked into the policy."
    if m["complete_swap"]["correct"] or m["complete_swap"]["action_name"] == "use_key":
        return "Fail", "Complete-is-junk still use_key; memorized p99.tag not payload."
    if not (m["match_changed"] and m["wcomp_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    if "action=2" not in m.get("red_tag", "") or "here=" not in m.get("red_tag", ""):
        return "Fail", "Red S is not the complete here= wiki page."
    if "when=" in m.get("red_tag", ""):
        return "Fail", "Red S has when=; recency sneak."
    return (
        "Store-works",
        "Split find+complete+use, no when=; cortex frozen; unmount W; held-out green wait; controls fail.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key."
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Shared return did not solve red (last-50 starved or red stays open)."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed under shared return."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["match_control"]["correct"] or m["match_control"]["action_name"] == "use_key":
        return "Fail", "door= control still use_key."
    if m["stub_control"]["correct"] or m["stub_control"]["action_name"] == "use_key":
        return "Fail", "Stub-first control still use_key."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Use-off control still use_key."
    if not (m["match_changed"] and m["wcomp_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return solved the same joint as A; cortex frozen; unmount W; held-out green wait.",
    )


def _train(policy: UsePolicy, w_dir: Path, work: Path, n: int, seed: int, *, split: bool) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_c = b_u = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_dir, policy, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        probe(ag, "probe_red_with_key", seed + 10)
        tag = _tags(s_dir)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r_found = 1.0 if "here=" in tag else 0.0
        r_comp = 1.0 if "action=2" in tag else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        if split:
            b_f = 0.9 * b_f + 0.1 * r_found
            b_c = 0.9 * b_c + 0.1 * r_comp
            b_u = 0.9 * b_u + 0.1 * r_use
            policy.update([t for t in tr if t.get("kind") == "match"], r_found - b_f)
            policy.update([t for t in tr if t.get("kind") == "wcomp"], r_comp - b_c)
            policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_u)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") == "match"], adv)
            policy.update([t for t in tr if t.get("kind") == "wcomp"], adv)
            policy.update([t for t in tr if t.get("kind") == "use"], adv)
        rewards.append(r_use)
    return rewards


def run_arm(
    *,
    arm: str,
    split: bool,
    run_dir: Path,
    w_red: Path,
    w_green: Path,
    w_swap: Path,
    w_stub: Path,
    w_files: list[str],
    seed: int,
    n_train: int,
    train_seed: int,
) -> dict[str, Any]:
    work = run_dir / f"{arm}_train"
    s_un = run_dir / f"{arm}_untrained"
    s_red = run_dir / f"{arm}_red"
    s_green = run_dir / f"{arm}_green"
    s_m = run_dir / f"{arm}_matchctrl"
    s_s = run_dir / f"{arm}_stubctrl"
    s_u = run_dir / f"{arm}_usectrl"
    s_swap = run_dir / f"{arm}_swap"
    s_stub = run_dir / f"{arm}_stubonly"
    s_off = run_dir / f"{arm}_off"
    s_empty = run_dir / f"{arm}_empty"
    for d in (work, s_un, s_red, s_green, s_m, s_s, s_u, s_swap, s_stub, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    match0 = _head_fp(policy, "match")
    wcomp0 = _head_fp(policy, "wcomp")
    use0 = _head_fp(policy, "use")
    dummy = make(run_dir / f"{arm}_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a = make(s_un, w_red, policy, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train(policy, w_red, work, n_train, train_seed, split=split)
    match1 = _head_fp(policy, "match")
    wcomp1 = _head_fp(policy, "wcomp")
    use1 = _head_fp(policy, "use")

    red_a, _, red_unmount, red_tag = _commit_unmount(make, s_red, w_red, policy, "probe_red_with_key", seed + 10)
    _, _, green_unmount, green_tag = _commit_unmount(make, s_green, w_green, policy, "probe_green", seed + 20)
    _, _, match_control, _ = _commit_unmount(make, s_m, w_red, _copy_heads(policy, "wcomp", "use"), "probe_red_with_key", seed + 10)
    _, _, stub_control, _ = _commit_unmount(make, s_s, w_red, _copy_heads(policy, "match", "use"), "probe_red_with_key", seed + 10)
    _, _, use_control, _ = _commit_unmount(make, s_u, w_red, _copy_heads(policy, "match", "wcomp"), "probe_red_with_key", seed + 10)
    _, _, complete_swap, swap_tag = _commit_unmount(make, s_swap, w_swap, policy, "probe_red_with_key", seed + 10)
    _, _, stub_only, stub_tag = _commit_unmount(make, s_stub, w_stub, policy, "probe_red_with_key", seed + 10)

    empty = make(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    empty_g = probe(empty, "probe_green", seed + 20)
    off = make(s_off, None, policy, enabled=False)
    off.reset_rho()
    disable_red = probe(off, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": arm,
        "trained_split": split,
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_match_head": dummy.use_match_head,
        "use_wcomp_head": dummy.use_wcomp_head,
        "use_read": dummy.use_read,
        "place_key": dummy.place_key,
        "w_has_when": any("when=" in p.read_text(encoding="utf-8") for p in w_red.glob("*.tag")),
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "match_changed": match0 != match1,
        "wcomp_changed": wcomp0 != wcomp1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files),
        "untrained": untrained,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "match_control": match_control,
        "stub_control": stub_control,
        "use_control": use_control,
        "complete_swap": complete_swap,
        "swap_tag": swap_tag,
        "stub_only": stub_only,
        "stub_tag": stub_tag,
        "empty_S": empty_p,
        "empty_S_green": empty_g,
        "disable_S_red": disable_red,
    }
    label, rationale = (classify_a if split else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v23(seed: int = 12345, n_train: int = 500) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    w_swap = run_dir / "W_swap"
    w_stub = run_dir / "W_stub"
    write_tag_notes(w_red, wiki_notes(include_red=True))
    write_tag_notes(w_green, wiki_notes(include_green=True))
    write_tag_notes(w_swap, list(clutter_w_no_answers()) + list(RED_SWAP) + [JUNK_RED])
    write_tag_notes(w_stub, list(clutter_w_no_answers()) + [RED_STUB, JUNK_RED])
    w_files = sorted(p.name for p in w_red.glob("*.tag"))
    a = run_arm(
        arm="A",
        split=True,
        run_dir=run_dir,
        w_red=w_red,
        w_green=w_green,
        w_swap=w_swap,
        w_stub=w_stub,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed,
    )
    b = run_arm(
        arm="B",
        split=False,
        run_dir=run_dir,
        w_red=w_red,
        w_green=w_green,
        w_swap=w_swap,
        w_stub=w_stub,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
    )
    out = {"seed": seed, "n_train": n_train, "run_dir": str(run_dir), "A": a, "B": b, "same_cortex": a["cortex_hash"] == b["cortex_hash"]}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v23 A split vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split find+complete+use | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained | {a['untrained']['action_name']} ({a['untrained']['correct']}) | {b['untrained']['action_name']} ({b['untrained']['correct']}) |
| Trained red, unmount W | {a['red_unmount']['action_name']} ({a['red_unmount']['correct']}) | {b['red_unmount']['action_name']} ({b['red_unmount']['correct']}) |
| Held-out green, unmount W | {a['green_unmount']['action_name']} ({a['green_unmount']['correct']}) | {b['green_unmount']['action_name']} ({b['green_unmount']['correct']}) |
| Match / stub / use-off | {a['match_control']['action_name']} / {a['stub_control']['action_name']} / {a['use_control']['action_name']} | {b['match_control']['action_name']} / {b['stub_control']['action_name']} / {b['use_control']['action_name']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v23 joint find+complete+use; split vs shared return")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    args = p.parse_args()
    m = run_v23(seed=args.seed, n_train=args.n_train)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print("A", m["A"]["untrained"]["action_name"], m["A"]["red_unmount"]["action_name"], m["A"]["green_unmount"]["action_name"], m["A"]["red_tag"].strip().replace("\n", " | "))
    print("B", m["B"]["untrained"]["action_name"], m["B"]["red_unmount"]["action_name"], m["B"]["green_unmount"]["action_name"], m["B"]["red_tag"].strip().replace("\n", " | "), "last50", m["B"]["train_return_last50"])


if __name__ == "__main__":
    main()
