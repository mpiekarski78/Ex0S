"""TM.0.1.0: open query names from files, not a {door, here} menu.

A split credit. B shared return.
"""

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
from experiments.run_v22 import _commit_unmount, _tags, probe
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.symbols import ACT_USE_KEY, ACT_WAIT, BLUE_FACT_ID, DOOR_GREEN, DOOR_RED, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes

# Useful index is not door= or here=. The string lives in the experiment, not the agent.
WIKI_RED = ("p99.tag", {"loc": DOOR_RED, "action": ACT_USE_KEY})
WIKI_GREEN = ("p98.tag", {"loc": DOOR_GREEN, "action": ACT_WAIT})
SWAP_RED = ("p99.tag", {"loc": DOOR_RED, "action": ACT_WAIT})


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm010"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wiki_notes(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = list(clutter_w_no_answers())
    if include_red:
        notes.append(WIKI_RED)
    if include_green:
        notes.append(WIKI_GREEN)
    return notes


def _head_fp(policy: UsePolicy, name: str) -> str:
    w = getattr(policy, f"w_{name}")
    b = getattr(policy, f"b_{name}", None)
    if b is None:
        return w.tobytes().hex()
    return str(w.tobytes().hex()) + str(float(b) if np.ndim(b) == 0 or b.size == 1 else b.tobytes().hex())


def _copy_heads(src: UsePolicy, *names: str) -> UsePolicy:
    ctrl = UsePolicy(seed=7, lr=0.2)
    for name in names:
        setattr(ctrl, f"w_{name}", getattr(src, f"w_{name}").copy())
        if hasattr(src, f"b_{name}"):
            setattr(ctrl, f"b_{name}", np.array(float(getattr(src, f"b_{name}"))))
    return ctrl


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    texts = "".join(p.read_text(encoding="utf-8") for p in w_dir.glob("*.tag"))
    return {
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
        "w_has_p99": WIKI_RED[0] in w_files,
        "w_has_when": "when=" in texts,
        "w_has_loc": "loc=" in texts,
        "w_has_here": "here=" in texts,
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
    use_qname_head: bool = True,
    use_match_head: bool = False,
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
        use_qname_head=use_qname_head,
        use_match_head=use_match_head,
        force_use=force_use,
    )


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if m["w_has_when"]:
        return "Confound", "Planted when=; recency cheat restored."
    if not m["w_has_p99"] or not m["w_has_loc"]:
        return "Confound", "W must have an unread page on a name that is not door=/here=."
    if m["w_has_here"]:
        return "Confound", "here= was planted; old match menu could cheat."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue open names."
    if m["use_match_head"]:
        return "Fail", "{door, here} match menu was restored."
    if not (m["use_qname_head"] and m["use_read"]):
        return "Fail", "Open-name head was frozen off to rescue the plot."
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
        return "Fail", "Open names did not find the unread loc= page after unmount W."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; a head learned that door."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["menu_control"]["correct"] or m["menu_control"]["action_name"] == "use_key":
        return "Fail", "Restored {door, here} menu still solved red."
    if m["qname_control"]["correct"] or m["qname_control"]["action_name"] == "use_key":
        return "Fail", "Untrained open-name + trained use still solved red."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Trained open-name with use-gate off still solved red."
    if m["name_swap"]["correct"] or m["name_swap"]["action_name"] == "use_key":
        return "Fail", "loc= page is wait and still use_key; fact leaked."
    if not (m["qname_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    if "action=2" not in m.get("red_tag", ""):
        return "Fail", "Red S is not the useful unread page."
    if "door=" in m.get("red_tag", "") or "here=" in m.get("red_tag", ""):
        return "Fail", "Red S used the old place-name menu."
    return (
        "Store-works",
        "Open query name from files, no {door, here} menu; cortex frozen; unmount W; held-out green wait; controls fail.",
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
    if not (m["qname_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return solved open names; cortex frozen; unmount W; held-out green wait.",
    )


def _train(policy: UsePolicy, w_dir: Path, work: Path, n: int, seed: int, *, split: bool) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_u = 0.0
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
        r_found = 1.0 if "action=2" in tag else 0.0
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        if split:
            b_f = 0.9 * b_f + 0.1 * r_found
            b_u = 0.9 * b_u + 0.1 * r_use
            policy.update([t for t in tr if t.get("kind") == "qname"], r_found - b_f)
            policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_u)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") == "qname"], adv)
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
    w_files: list[str],
    seed: int,
    n_train: int,
    train_seed: int,
) -> dict[str, Any]:
    work = run_dir / f"{arm}_train"
    s_un = run_dir / f"{arm}_untrained"
    s_red = run_dir / f"{arm}_red"
    s_green = run_dir / f"{arm}_green"
    s_menu = run_dir / f"{arm}_menuctrl"
    s_q = run_dir / f"{arm}_qnamectrl"
    s_u = run_dir / f"{arm}_usectrl"
    s_swap = run_dir / f"{arm}_swap"
    s_off = run_dir / f"{arm}_off"
    s_empty = run_dir / f"{arm}_empty"
    for d in (work, s_un, s_red, s_green, s_menu, s_q, s_u, s_swap, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    qname0 = _head_fp(policy, "qname")
    use0 = _head_fp(policy, "use")
    dummy = make(run_dir / f"{arm}_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a = make(s_un, w_red, policy, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train(policy, w_red, work, n_train, train_seed, split=split)
    qname1 = _head_fp(policy, "qname")
    use1 = _head_fp(policy, "use")

    red_a, _, red_unmount, red_tag = _commit_unmount(make, s_red, w_red, policy, "probe_red_with_key", seed + 10)
    _, _, green_unmount, green_tag = _commit_unmount(make, s_green, w_green, policy, "probe_green", seed + 20)
    _, _, menu_control, _ = _commit_unmount(
        lambda s, w, p, **kw: make(s, w, p, use_qname_head=False, use_match_head=True, **kw),
        s_menu,
        w_red,
        policy,
        "probe_red_with_key",
        seed + 10,
    )
    _, _, qname_control, _ = _commit_unmount(make, s_q, w_red, _copy_heads(policy, "use"), "probe_red_with_key", seed + 10)
    _, _, use_control, _ = _commit_unmount(make, s_u, w_red, _copy_heads(policy, "qname"), "probe_red_with_key", seed + 10)
    _, _, name_swap, swap_tag = _commit_unmount(make, s_swap, w_swap, policy, "probe_red_with_key", seed + 10)

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
        "use_qname_head": dummy.use_qname_head,
        "use_match_head": dummy.use_match_head,
        "use_read": dummy.use_read,
        "place_key": dummy.place_key,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "qname_changed": qname0 != qname1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_red),
        "untrained": untrained,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "menu_control": menu_control,
        "qname_control": qname_control,
        "use_control": use_control,
        "name_swap": name_swap,
        "swap_tag": swap_tag,
        "empty_S": empty_p,
        "empty_S_green": empty_g,
        "disable_S_red": disable_red,
    }
    label, rationale = (classify_a if split else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_tm010(seed: int = 12345, n_train: int = 500) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    w_swap = run_dir / "W_swap"
    write_tag_notes(w_red, wiki_notes(include_red=True))
    write_tag_notes(w_green, wiki_notes(include_green=True))
    write_tag_notes(w_swap, list(clutter_w_no_answers()) + [SWAP_RED])
    w_files = sorted(p.name for p in w_red.glob("*.tag"))
    a = run_arm(
        arm="A",
        split=True,
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
        split=False,
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
        "version": "TM.0.1.0",
        "seed": seed,
        "n_train": n_train,
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.1.0 A open names vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split open query name | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained | {a['untrained']['action_name']} ({a['untrained']['correct']}) | {b['untrained']['action_name']} ({b['untrained']['correct']}) |
| Trained red, unmount W | {a['red_unmount']['action_name']} ({a['red_unmount']['correct']}) | {b['red_unmount']['action_name']} ({b['red_unmount']['correct']}) |
| Held-out green, unmount W | {a['green_unmount']['action_name']} ({a['green_unmount']['correct']}) | {b['green_unmount']['action_name']} ({b['green_unmount']['correct']}) |
| Menu / qname-off / use-off | {a['menu_control']['action_name']} / {a['qname_control']['action_name']} / {a['use_control']['action_name']} | {b['menu_control']['action_name']} / {b['qname_control']['action_name']} / {b['use_control']['action_name']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.1.0 open query names; split vs shared return")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    args = p.parse_args()
    m = run_tm010(seed=args.seed, n_train=args.n_train)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print("A", m["A"]["untrained"]["action_name"], m["A"]["red_unmount"]["action_name"], m["A"]["green_unmount"]["action_name"], m["A"]["red_tag"].strip().replace("\n", " | "))
    print("B", m["B"]["untrained"]["action_name"], m["B"]["red_unmount"]["action_name"], m["B"]["green_unmount"]["action_name"], m["B"]["red_tag"].strip().replace("\n", " | "), "last50", m["B"]["train_return_last50"])


if __name__ == "__main__":
    main()
