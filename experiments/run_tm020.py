"""TM.0.2.0: scale of W. Same messy search; hundreds of unread .tag files.

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

from experiments.run_tm010 import _copy_heads, _head_fp
from experiments.run_v22 import _commit_unmount, _tags, probe
from three_memory.agent import ThreeMemoryAgent
from three_memory.policy import UsePolicy
from three_memory.symbols import ACT_OPEN, ACT_USE_KEY, ACT_WAIT, BLUE_FACT_ID, DOOR_GREEN, DOOR_RED, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, write_tag_notes

N_W = 256
# Place value is on an unknown key, plus extra pad=. Exact loc=/door=/here= misses.
WIKI_RED = ("p99.tag", {"where": DOOR_RED, "action": ACT_USE_KEY, "pad": 7})
WIKI_GREEN = ("p98.tag", {"where": DOOR_GREEN, "action": ACT_WAIT, "pad": 7})
SWAP_RED = ("p99.tag", {"where": DOOR_RED, "action": ACT_WAIT, "pad": 7})
# Rare distractor keys only on place=1 so they never share has_code with red/green.
_RARE_KEYS = ("extra", "misc", "note", "aux", "tmp")


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm020"
    d.mkdir(parents=True, exist_ok=True)
    return d


def scale_clutter(n_clutter: int = N_W - 1) -> list[tuple[str, dict[str, Any]]]:
    """Hundreds of messy pages. action never 2. Filenames sort before p99/p98."""
    notes: list[tuple[str, dict[str, Any]]] = []
    place1: list[int] = []
    for i in range(n_clutter):
        place = i % 3
        # OPEN only: WAIT(0) equals door-red code; USE_KEY(2) would cheat found-reward.
        tags: dict[str, Any] = {"place": place, "action": ACT_OPEN}
        if i % 2 == 0:
            tags["pad"] = 3
        if place == 1:
            place1.append(i)
        notes.append((f"c{i:03d}.tag", tags))
    for k, idx in zip(_RARE_KEYS, place1):
        notes[idx][1][k] = 9
    return notes


def wiki_notes(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, dict[str, Any]]]:
    notes = scale_clutter(N_W - 1)
    if include_red:
        notes.append(WIKI_RED)
    if include_green:
        notes.append(WIKI_GREEN)
    return notes


def _has_field(text: str, name: str) -> bool:
    return f"\n{name}=" in text or text.startswith(f"{name}=")


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    texts = "".join(p.read_text(encoding="utf-8") for p in w_dir.glob("*.tag"))
    return {
        "w_n": len(w_files),
        "w_files": w_files,
        "w_has_red": f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.tag" in w_files,
        "w_has_p99": WIKI_RED[0] in w_files,
        "w_has_when": _has_field(texts, "when"),
        "w_has_where": _has_field(texts, "where"),
        "w_has_loc": _has_field(texts, "loc"),
        "w_has_here": _has_field(texts, "here"),
        "w_has_pad": _has_field(texts, "pad"),
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
    use_search_head: bool = True,
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
        use_search_head=use_search_head,
        use_match_head=use_match_head,
        force_use=force_use,
    )


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_n"] < 200:
        return "Confound", "W was shrunk below 200; scale jump abandoned."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if m["w_has_when"]:
        return "Confound", "Planted when=; recency cheat restored."
    if m["w_has_loc"] or m["w_has_here"]:
        return "Confound", "loc=/here= planted; exact query could cheat."
    if not m["w_has_p99"] or not m["w_has_where"] or not m["w_has_pad"]:
        return "Confound", "W must have a messy unread page (unknown place key + extra field)."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue search."
    if m["use_match_head"] or m["use_qname_head"]:
        return "Fail", "Exact query match was restored."
    if not (m["use_search_head"] and m["use_read"]):
        return "Fail", "Search head was frozen off to rescue the plot."
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
        return "Fail", "Search did not find the messy unread page in scaled W after unmount."
    if not m["green_unmount"]["correct"]:
        return "Fail", "Held-out green failed; a head learned that door."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["menu_control"]["correct"] or m["menu_control"]["action_name"] == "use_key":
        return "Fail", "Exact {door, here} match still solved red."
    if m["search_control"]["correct"] or m["search_control"]["action_name"] == "use_key":
        return "Fail", "Untrained search + trained use still solved red."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Trained search with use-gate off still solved red."
    if m["name_swap"]["correct"] or m["name_swap"]["action_name"] == "use_key":
        return "Fail", "Messy page is wait and still use_key; fact leaked."
    if not (m["search_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    if "action=2" not in m.get("red_tag", "") or not _has_field(m.get("red_tag", ""), "where"):
        return "Fail", "Red S is not the messy unread page."
    tag = m.get("red_tag", "")
    if _has_field(tag, "loc") or _has_field(tag, "door") or _has_field(tag, "here"):
        return "Fail", "Red S used an exact place-name query."
    return (
        "Store-works",
        "Scale-W messy retrieve, no exact loc=/door=; cortex frozen; unmount W; held-out green wait; controls fail.",
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
    if not (m["search_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return solved scale-W messy retrieve; cortex frozen; unmount W; held-out green wait.",
    )


def _train(policy: UsePolicy, w_dir: Path, work: Path, n: int, seed: int, *, split: bool) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_u = 0.0
    s_dir = work / "ep"
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
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
            policy.update([t for t in tr if t.get("kind") == "search"], r_found - b_f)
            policy.update([t for t in tr if t.get("kind") == "use"], r_use - b_u)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") == "search"], adv)
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
    s_s = run_dir / f"{arm}_searchctrl"
    s_u = run_dir / f"{arm}_usectrl"
    s_swap = run_dir / f"{arm}_swap"
    s_off = run_dir / f"{arm}_off"
    s_empty = run_dir / f"{arm}_empty"
    for d in (work, s_un, s_red, s_green, s_menu, s_s, s_u, s_swap, s_off, s_empty):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    search0 = _head_fp(policy, "search")
    use0 = _head_fp(policy, "use")
    dummy = make(run_dir / f"{arm}_hash", None, policy)
    cortex0 = dummy.weight_hash()

    un_a = make(s_un, w_red, policy, epsilon=0.0)
    un_a.reset_rho()
    untrained = probe(un_a, "probe_red_with_key", seed + 10)

    rewards = _train(policy, w_red, work, n_train, train_seed, split=split)
    search1 = _head_fp(policy, "search")
    use1 = _head_fp(policy, "use")

    red_a, _, red_unmount, red_tag = _commit_unmount(make, s_red, w_red, policy, "probe_red_with_key", seed + 10)
    _, _, green_unmount, green_tag = _commit_unmount(make, s_green, w_green, policy, "probe_green", seed + 20)
    _, _, menu_control, _ = _commit_unmount(
        lambda s, w, p, **kw: make(s, w, p, use_search_head=False, use_match_head=True, **kw),
        s_menu,
        w_red,
        policy,
        "probe_red_with_key",
        seed + 10,
    )
    _, _, search_control, _ = _commit_unmount(make, s_s, w_red, _copy_heads(policy, "use"), "probe_red_with_key", seed + 10)
    _, _, use_control, _ = _commit_unmount(make, s_u, w_red, _copy_heads(policy, "search"), "probe_red_with_key", seed + 10)
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
        "use_search_head": dummy.use_search_head,
        "use_match_head": dummy.use_match_head,
        "use_qname_head": dummy.use_qname_head,
        "use_read": dummy.use_read,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "search_changed": search0 != search1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_red),
        "untrained": untrained,
        "red_unmount": red_unmount,
        "red_tag": red_tag,
        "green_unmount": green_unmount,
        "green_tag": green_tag,
        "menu_control": menu_control,
        "search_control": search_control,
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


def run_tm020(seed: int = 12345, n_train: int = 10000) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    w_swap = run_dir / "W_swap"
    write_tag_notes(w_red, wiki_notes(include_red=True))
    write_tag_notes(w_green, wiki_notes(include_green=True))
    write_tag_notes(w_swap, scale_clutter(N_W - 1) + [SWAP_RED])
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
        "version": "TM.0.2.0",
        "seed": seed,
        "n_train": n_train,
        "w_n": len(w_files),
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.2.0 A scale-W messy retrieve vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split scale W | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

W size: {len(w_files)} files.

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained | {a['untrained']['action_name']} ({a['untrained']['correct']}) | {b['untrained']['action_name']} ({b['untrained']['correct']}) |
| Trained red, unmount W | {a['red_unmount']['action_name']} ({a['red_unmount']['correct']}) | {b['red_unmount']['action_name']} ({b['red_unmount']['correct']}) |
| Held-out green, unmount W | {a['green_unmount']['action_name']} ({a['green_unmount']['correct']}) | {b['green_unmount']['action_name']} ({b['green_unmount']['correct']}) |
| Exact-match / search-off / use-off | {a['menu_control']['action_name']} / {a['search_control']['action_name']} / {a['use_control']['action_name']} | {b['menu_control']['action_name']} / {b['search_control']['action_name']} / {b['use_control']['action_name']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.2.0 scale of W; split vs shared return")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=10000)
    args = p.parse_args()
    m = run_tm020(seed=args.seed, n_train=args.n_train)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "w_n": m["w_n"], "run_dir": m["run_dir"]}, indent=2))
    print("A", m["A"]["untrained"]["action_name"], m["A"]["red_unmount"]["action_name"], m["A"]["green_unmount"]["action_name"], m["A"]["red_tag"].strip().replace("\n", " | "))
    print("B", m["B"]["untrained"]["action_name"], m["B"]["red_unmount"]["action_name"], m["B"]["green_unmount"]["action_name"], m["B"]["red_tag"].strip().replace("\n", " | "), "last50", m["B"]["train_return_last50"])


if __name__ == "__main__":
    main()
