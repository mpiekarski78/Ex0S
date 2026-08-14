"""v22: A complete vs stub (no when=). B joint match+wsel+use, no clamps."""

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
from experiments.run_v20 import JUNK_GREEN, JUNK_RED
from experiments.run_v21 import GREEN_JUNK as GREEN_OLD, GREEN_WAIT as GREEN_NEW, RED_JUNK as RED_OLD, RED_USE as RED_NEW
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.symbols import ACT_USE_KEY, ACT_WAIT, BLUE_FACT_ID, DOOR_GREEN, DOOR_RED, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes

# A: no when=. Filename-first is a stub. Useful page has a payload.
RED_STUB = ("aaa.tag", {"here": DOOR_RED})
RED_COMPLETE = ("p99.tag", {"here": DOOR_RED, "action": ACT_USE_KEY})
GREEN_STUB = ("aag.tag", {"here": DOOR_GREEN})
GREEN_COMPLETE = ("p98.tag", {"here": DOOR_GREEN, "action": ACT_WAIT})
RED_SWAP = (
    ("aaa.tag", {"here": DOOR_RED, "action": ACT_WAIT}),
    ("p99.tag", {"here": DOOR_RED}),
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v22"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(agent: ThreeMemoryAgent, scenario: str, seed: int) -> dict[str, Any]:
    return probe_v12(agent, scenario, seed)


def stub_notes(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = list(clutter_w_no_answers())
    if include_red:
        notes.extend([RED_STUB, RED_COMPLETE])
    if include_green:
        notes.extend([GREEN_STUB, GREEN_COMPLETE])
    return notes


def joint_notes(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = list(clutter_w_no_answers())
    if include_red:
        notes.extend([RED_OLD, RED_NEW, JUNK_RED])
    if include_green:
        notes.extend([GREEN_OLD, GREEN_NEW, JUNK_GREEN])
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
        "w_has_aaa": "aaa.tag" in w_files,
        "w_has_p99": "p99.tag" in w_files,
        "w_has_junk": JUNK_RED[0] in w_files,
    }


def make_a(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
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
        use_wcomp_head=True,
        force_use=force_use,
    )


def make_b(
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
        use_wsel_head=True,
        wsel_dump=False,
        force_use=force_use,
    )


def _copy_heads(src: UsePolicy, *names: str) -> UsePolicy:
    ctrl = UsePolicy(seed=7, lr=0.2)
    for name in names:
        setattr(ctrl, f"w_{name}", getattr(src, f"w_{name}").copy())
        setattr(ctrl, f"b_{name}", np.array(float(getattr(src, f"b_{name}"))))
    return ctrl


def _commit_unmount(make_fn, s_dir, w_dir, policy, scenario, seed, **kwargs):
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make_fn(s_dir, w_dir, policy, epsilon=0.0, **kwargs)
    ag.reset_rho()
    first = probe(ag, scenario, seed)
    tag = _tags(s_dir)
    reload = s_dir.parent / f"{s_dir.name}_reload"
    shutil.rmtree(reload, ignore_errors=True)
    shutil.copytree(s_dir, reload)
    only = make_fn(reload, None, policy, epsilon=0.0, **kwargs)
    only.reset_rho()
    unmount = probe(only, scenario, seed)
    return ag, first, unmount, tag


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if m["w_has_when"]:
        return "Confound", "Planted when= on A W; recency cheat restored."
    if not m["w_has_aaa"] or not m["w_has_p99"]:
        return "Confound", "A train W must have stub aaa.tag and complete p99.tag."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if not m["trained_force_use"]:
        return "Fail", "Use-gate was unclamped; A isolates complete vs stub."
    if m["place_key"] != "here":
        return "Fail", "Query name was not frozen to here=."
    if not m["use_wcomp_head"]:
        return "Fail", "Complete-page head was frozen off to rescue the plot."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key (stub was not first)."
    if (m["untrained"].get("policy") or {}).get("wcomp_alt") is True:
        return "Fail", "Untrained already took the complete page; pick was frozen to rescue."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Trained policy did not keep the complete red wiki page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; policy learned that door, not complete-vs-stub."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["stub_only"]["correct"] or m["stub_only"]["action_name"] == "use_key":
        return "Fail", "Stub-only W still use_key; fact leaked into the policy."
    if m["complete_swap"]["correct"] or m["complete_swap"]["action_name"] == "use_key":
        return "Fail", "Complete-is-junk still use_key; memorized p99.tag not payload."
    if not m["wcomp_changed"]:
        return "Fail", "Complete-page head did not move."
    if "action=2" not in m.get("red_tag", ""):
        return "Fail", "Red S does not contain the complete wiki page."
    if "when=" in m.get("red_tag", ""):
        return "Fail", "Red S has when=; recency sneak."
    return (
        "Store-works",
        "Kept complete here= page over a stub; no when=; cortex frozen; unmount W; held-out green wait.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if not (m["w_has_aaa"] and m["w_has_p99"] and m["w_has_junk"]):
        return "Confound", "B train W must have here= first/newest and door= junk."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue the joint."
    if m["place_key"] != "door":
        return "Fail", "Match was frozen to here= to rescue the joint."
    if not (m["use_match_head"] and m["use_wsel_head"] and m["use_read"]):
        return "Fail", "A joint head was frozen off to rescue the plot."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained"], m["red_unmount"], m["green_unmount"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    if m["untrained"]["correct"] or m["untrained"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key."
    if not m["red_unmount"]["correct"]:
        return "Fail", "Joint did not use the newest here= page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; a head learned that door."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["match_control"]["correct"] or m["match_control"]["action_name"] == "use_key":
        return "Fail", "Trained wsel+use with untrained door= match still solved red."
    if m["first_control"]["correct"] or m["first_control"]["action_name"] == "use_key":
        return "Fail", "Trained match+use with untrained first-file still solved red."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Trained match+wsel with use-gate off still solved red."
    if not (m["match_changed"] and m["wsel_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    if "action=2" not in m.get("red_tag", "") or "here=" not in m.get("red_tag", ""):
        return "Fail", "Red S is not the newest here= wiki page."
    return (
        "Store-works",
        "Joint find+pick+use without clamps; cortex frozen; unmount W; held-out green wait; door=/first/use-off fail.",
    )


def _train_a(policy: UsePolicy, w_dir: Path, work: Path, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make_a(s_dir, w_dir, policy, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        probe(ag, "probe_red_with_key", seed + 10)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        baseline = 0.9 * baseline + 0.1 * r_use
        policy.update([t for t in ag.policy_traces if t.get("kind") == "wcomp"], r_use - baseline)
        rewards.append(r_use)
    return rewards


def _train_b(policy: UsePolicy, w_dir: Path, work: Path, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_s = b_u = 0.0
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        s_dir = work / f"ep_{ep}"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make_b(s_dir, w_dir, policy, epsilon=eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        probe(ag, "probe_red_with_key", seed + 10)
        tag = _tags(s_dir)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r_found = 1.0 if "here=" in tag else 0.0
        r_sel = 1.0 if "action=2" in tag else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        b_f = 0.9 * b_f + 0.1 * r_found
        b_s = 0.9 * b_s + 0.1 * r_sel
        b_u = 0.9 * b_u + 0.1 * r_use
        tr = ag.policy_traces
        policy.update([t for t in tr if t.get("kind") == "match"], r_found - b_f)
        policy.update([t for t in tr if t.get("kind") == "wsel"], r_sel - b_s)
        policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_u)
        rewards.append(r_use)
    return rewards


def run_arm_a(run_dir: Path, w_red: Path, w_green: Path, w_swap: Path, w_stub: Path, w_files: list[str], seed: int, n_train: int) -> dict[str, Any]:
    work = run_dir / "A_train"
    s_un = run_dir / "A_untrained"
    s_red = run_dir / "A_red"
    s_green = run_dir / "A_green"
    s_swap = run_dir / "A_swap"
    s_stub = run_dir / "A_stubonly"
    s_off = run_dir / "A_off"
    s_empty = run_dir / "A_empty"
    for d in (work, s_un, s_red, s_green, s_swap, s_stub, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    wcomp0 = _head_fp(policy, "wcomp")
    dummy = make_a(run_dir / "A_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a = make_a(s_un, w_red, policy, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train_a(policy, w_red, work, n_train, seed)
    wcomp1 = _head_fp(policy, "wcomp")

    red_a, _, red_unmount, red_tag = _commit_unmount(make_a, s_red, w_red, policy, "probe_red_with_key", seed + 10)
    _, _, green_unmount, green_tag = _commit_unmount(make_a, s_green, w_green, policy, "probe_green", seed + 20)
    _, _, complete_swap, swap_tag = _commit_unmount(make_a, s_swap, w_swap, policy, "probe_red_with_key", seed + 10)
    _, _, stub_only, stub_tag = _commit_unmount(make_a, s_stub, w_stub, policy, "probe_red_with_key", seed + 10)

    empty = make_a(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    empty_g = probe(empty, "probe_green", seed + 20)
    off = make_a(s_off, None, policy, enabled=False)
    off.reset_rho()
    disable_red = probe(off, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": "A",
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_wcomp_head": dummy.use_wcomp_head,
        "place_key": dummy.place_key,
        "w_has_when": any("when=" in p.read_text(encoding="utf-8") for p in w_red.glob("*.tag")),
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "wcomp_changed": wcomp0 != wcomp1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files),
        "untrained": untrained,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "complete_swap": complete_swap,
        "swap_tag": swap_tag,
        "stub_only": stub_only,
        "stub_tag": stub_tag,
        "empty_S": empty_p,
        "empty_S_green": empty_g,
        "disable_S_red": disable_red,
    }
    label, rationale = classify_a(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_arm_b(run_dir: Path, w_red: Path, w_green: Path, w_files: list[str], seed: int, n_train: int) -> dict[str, Any]:
    work = run_dir / "B_train"
    s_un = run_dir / "B_untrained"
    s_red = run_dir / "B_red"
    s_green = run_dir / "B_green"
    s_m = run_dir / "B_matchctrl"
    s_f = run_dir / "B_firstctrl"
    s_u = run_dir / "B_usectrl"
    s_off = run_dir / "B_off"
    s_empty = run_dir / "B_empty"
    for d in (work, s_un, s_red, s_green, s_m, s_f, s_u, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    match0 = _head_fp(policy, "match")
    wsel0 = _head_fp(policy, "wsel")
    use0 = _head_fp(policy, "use")
    dummy = make_b(run_dir / "B_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a = make_b(s_un, w_red, policy, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train_b(policy, w_red, work, n_train, seed + 5)
    match1 = _head_fp(policy, "match")
    wsel1 = _head_fp(policy, "wsel")
    use1 = _head_fp(policy, "use")

    red_a, _, red_unmount, red_tag = _commit_unmount(make_b, s_red, w_red, policy, "probe_red_with_key", seed + 10)
    _, _, green_unmount, green_tag = _commit_unmount(make_b, s_green, w_green, policy, "probe_green", seed + 20)

    _, _, match_control, _ = _commit_unmount(make_b, s_m, w_red, _copy_heads(policy, "wsel", "use"), "probe_red_with_key", seed + 10)
    _, _, first_control, _ = _commit_unmount(make_b, s_f, w_red, _copy_heads(policy, "match", "use"), "probe_red_with_key", seed + 10)
    _, _, use_control, _ = _commit_unmount(make_b, s_u, w_red, _copy_heads(policy, "match", "wsel"), "probe_red_with_key", seed + 10)

    empty = make_b(s_empty, None, policy)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    empty_g = probe(empty, "probe_green", seed + 20)
    off = make_b(s_off, None, policy, enabled=False)
    off.reset_rho()
    disable_red = probe(off, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": "B",
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_match_head": dummy.use_match_head,
        "use_wsel_head": dummy.use_wsel_head,
        "use_read": dummy.use_read,
        "place_key": dummy.place_key,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "match_changed": match0 != match1,
        "wsel_changed": wsel0 != wsel1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files),
        "untrained": untrained,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "match_control": match_control,
        "first_control": first_control,
        "use_control": use_control,
        "empty_S": empty_p,
        "empty_S_green": empty_g,
        "disable_S_red": disable_red,
    }
    label, rationale = classify_b(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_v22(seed: int = 12345, n_train: int = 500) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_A"
    w_ag = run_dir / "W_A_green"
    w_swap = run_dir / "W_A_swap"
    w_stub = run_dir / "W_A_stub"
    w_b = run_dir / "W_B"
    w_bg = run_dir / "W_B_green"
    write_tag_notes(w_a, stub_notes(include_red=True))
    write_tag_notes(w_ag, stub_notes(include_green=True))
    write_tag_notes(w_swap, list(clutter_w_no_answers()) + list(RED_SWAP))
    write_tag_notes(w_stub, list(clutter_w_no_answers()) + [RED_STUB])
    write_tag_notes(w_b, joint_notes(include_red=True))
    write_tag_notes(w_bg, joint_notes(include_green=True))
    a_files = sorted(p.name for p in w_a.glob("*.tag"))
    b_files = sorted(p.name for p in w_b.glob("*.tag"))
    a = run_arm_a(run_dir, w_a, w_ag, w_swap, w_stub, a_files, seed, n_train)
    b = run_arm_b(run_dir, w_b, w_bg, b_files, seed, n_train)
    out = {"seed": seed, "n_train": n_train, "run_dir": str(run_dir), "A": a, "B": b, "same_cortex": a["cortex_hash"] == b["cortex_hash"]}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v22 A complete vs stub / B joint no clamps

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A complete vs stub (no when=) | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B joint match+wsel+use | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained | {a['untrained']['action_name']} ({a['untrained']['correct']}) | {b['untrained']['action_name']} ({b['untrained']['correct']}) |
| Trained red, unmount W | {a['red_unmount']['action_name']} ({a['red_unmount']['correct']}) | {b['red_unmount']['action_name']} ({b['red_unmount']['correct']}) |
| Held-out green, unmount W | {a['green_unmount']['action_name']} ({a['green_unmount']['correct']}) | {b['green_unmount']['action_name']} ({b['green_unmount']['correct']}) |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v22 complete vs stub; joint find+pick+use")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    args = p.parse_args()
    m = run_v22(seed=args.seed, n_train=args.n_train)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print("A", m["A"]["untrained"]["action_name"], m["A"]["red_unmount"]["action_name"], m["A"]["green_unmount"]["action_name"], m["A"]["red_tag"].strip().replace("\n", " | "))
    print("B", m["B"]["untrained"]["action_name"], m["B"]["red_unmount"]["action_name"], m["B"]["green_unmount"]["action_name"], m["B"]["red_tag"].strip().replace("\n", " | "))


if __name__ == "__main__":
    main()
