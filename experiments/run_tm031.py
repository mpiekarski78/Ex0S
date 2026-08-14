"""TM.0.3.1: documents. Free life over unread .md pages, not tidy .tag W.

A split credit. B shared return.
Same free-life procedure as TM.0.3.0; W is markdown documents with embedded k=v lines.
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
from experiments.run_tm012 import _has_field
from experiments.run_v12 import clutter_w_no_answers
from experiments.run_v22 import _tags, probe
from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import ACT_OPEN, ACT_USE_KEY, ACT_WAIT, BLUE_FACT_ID, DOOR_GREEN, DOOR_RED, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import DocLibrary, TagStore, write_doc_notes

# Useful page is a document, not a .tag file. Place value on unknown key + pad=.
WIKI_RED = (
    "p99.md",
    "Staff scrap from the red room. Numbers below are filed fields, not a story.",
    {"where": DOOR_RED, "action": ACT_USE_KEY, "pad": 7},
)
WIKI_GREEN = (
    "p98.md",
    "Staff scrap from the green room. Numbers below are filed fields, not a story.",
    {"where": DOOR_GREEN, "action": ACT_WAIT, "pad": 7},
)
SWAP_RED = (
    "p99.md",
    "Staff scrap from the red room. Numbers below are filed fields, not a story.",
    {"where": DOOR_RED, "action": ACT_WAIT, "pad": 7},
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm031"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clutter_docs() -> list[tuple[str, str, dict[str, Any]]]:
    """Same integer clutter as tag W, but as .md documents with prose."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for name, tags in clutter_w_no_answers():
        stem = Path(name).stem
        # OPEN-only would also work; keep original clutter ints except never plant USE_KEY.
        safe = dict(tags)
        if safe.get("action") == ACT_USE_KEY:
            safe["action"] = ACT_OPEN
        out.append((f"{stem}.md", f"Clutter page {stem}. Unrelated hallway notes.", safe))
    return out


def wiki_docs(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, str, dict[str, Any]]]:
    notes = clutter_docs()
    if include_red:
        notes.append(WIKI_RED)
    if include_green:
        notes.append(WIKI_GREEN)
    return notes


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    texts = "".join(p.read_text(encoding="utf-8") for p in w_dir.glob("*.md"))
    tag_files = sorted(p.name for p in w_dir.glob("*.tag"))
    return {
        "w_files": w_files,
        "w_n": len(w_files),
        "w_tag_files": tag_files,
        "w_has_tag_files": bool(tag_files),
        "w_all_md": bool(w_files) and all(n.endswith(".md") for n in w_files),
        "w_has_red": f"{RED_FACT_ID}.md" in w_files or f"{RED_FACT_ID}.tag" in w_files,
        "w_has_green": f"{GREEN_FACT_ID}.md" in w_files or f"{GREEN_FACT_ID}.tag" in w_files,
        "w_has_blue": f"{BLUE_FACT_ID}.md" in w_files or f"{BLUE_FACT_ID}.tag" in w_files,
        "w_has_p99": WIKI_RED[0] in w_files,
        "w_has_when": _has_field(texts, "when"),
        "w_has_where": _has_field(texts, "where"),
        "w_has_loc": _has_field(texts, "loc"),
        "w_has_here": _has_field(texts, "here"),
        "w_has_pad": _has_field(texts, "pad"),
        "w_has_prose": any(
            ln.strip() and not ln.strip().startswith("#") and "=" not in ln
            for ln in texts.splitlines()
        ),
    }


def make(
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    explore_epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
    force_use: bool = False,
    use_search_head: bool = True,
    use_match_head: bool = False,
    record_search_on_explore: bool = True,
) -> ThreeMemoryAgent:
    world = DocLibrary(w_dir) if w_dir is not None else None
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
        explore_epsilon=explore_epsilon,
        use_read=True,
        use_search_head=use_search_head,
        use_match_head=use_match_head,
        force_use=force_use,
        record_search_on_explore=record_search_on_explore,
    )


def live_free(
    agent: ThreeMemoryAgent,
    scenario: str,
    seed: int,
    *,
    max_steps: int = 32,
) -> dict[str, Any]:
    """Agent acts. No scripted OPEN→PICK→USE. n_forced is always 0."""
    world = KeyDoorWorld(seed=seed)
    obs = world.reset(scenario)
    log: list[dict[str, Any]] = []
    opened = False
    n_explored = 0
    for _ in range(max_steps):
        action, meta = agent.act(obs, update_rho=True, explore=True)
        result = world.step(int(action), scenario)
        agent.observe_outcome(result.obs, result.success, result.info)
        n_explored += int(bool(meta.get("explored")))
        log.append(
            {
                "action": int(action),
                "action_name": Action(action).name.lower(),
                "explored": bool(meta.get("explored")),
                "success": result.success,
                "opened": bool(result.info.get("opened")),
                "forced": False,
                "store_files": agent.store.list_files() if hasattr(agent.store, "list_files") else [],
            }
        )
        opened = opened or bool(result.info.get("opened"))
        obs = result.obs
        if result.done:
            break
    files = agent.store.list_files() if hasattr(agent.store, "list_files") else []
    tag = _tags(Path(agent.store.root)) if hasattr(agent.store, "root") else ""
    return {
        "opened": opened,
        "files": files,
        "tag": tag,
        "n_forced": sum(1 for s in log if s.get("forced")),
        "n_steps": len(log),
        "n_explored": n_explored,
        "actions": [s["action_name"] for s in log],
        "found_action2": "action=2" in tag and _has_field(tag, "where"),
        "steps": log,
    }


def _life_then_probe(
    make_fn,
    s_dir: Path,
    w_dir: Path | None,
    policy: UsePolicy,
    life_scenario: str,
    probe_scenario: str,
    seed: int,
    *,
    max_steps: int,
    explore_epsilon: float,
    rng: np.random.Generator | None = None,
    **make_kw: Any,
) -> tuple[ThreeMemoryAgent, dict[str, Any], dict[str, Any], str]:
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    ag = make_fn(s_dir, w_dir, policy, explore_epsilon=explore_epsilon, rng=rng, **make_kw)
    ag.reset_rho()
    ag.policy_traces = []
    live = live_free(ag, life_scenario, seed, max_steps=max_steps)
    tag = live["tag"]
    ag.world = None
    ag.reset_rho()
    p = probe(ag, probe_scenario, seed)
    return ag, live, p, tag


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    if not m["cortex_unchanged"]:
        return "Confound", "Cortex (genome) weights moved."
    if m["w_has_tag_files"] or not m["w_all_md"]:
        return "Confound", "W still used .tag files; documents jump abandoned."
    if not m["w_has_prose"]:
        return "Confound", "W .md files have no prose; only bare k=v lines."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if m["w_has_when"]:
        return "Confound", "Planted when=; recency cheat restored."
    if m["w_has_loc"] or m["w_has_here"]:
        return "Confound", "loc=/here= planted; exact query could cheat."
    if not m["w_has_p99"] or not m["w_has_where"] or not m["w_has_pad"]:
        return "Confound", "W must have a messy unread document (unknown place key + extra field)."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on; this must find W, not author d0."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue documents."
    if m["use_match_head"] or m["use_qname_head"]:
        return "Fail", "Exact query match was restored."
    if not (m["use_search_head"] and m["use_read"]):
        return "Fail", "Search head was frozen off to rescue the plot."
    if not m["trained_life"]:
        return "Fail", "Training was not a free life (scripted unmount curriculum restored)."
    if any(lv.get("n_forced", 0) for lv in (m["untrained_live"], m["red_live"], m["green_live"])):
        return "Confound", "A forced curriculum ran; this is not a free life."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_probe"], m["red_after_reset"], m["green_after_reset"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green; held-out is not a transfer test."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key after a free life on documents."
    if not m["red_after_reset"]["correct"]:
        return "Fail", "Free red life on .md W did not leave a usable S after ρ reset / W gone."
    if not m["green_after_reset"]["correct"]:
        return "Fail", "Held-out green free life on documents failed after reset."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["menu_control"]["correct"] or m["menu_control"]["action_name"] == "use_key":
        return "Fail", "Exact {door, here} match still solved red."
    if m["search_control"]["correct"] or m["search_control"]["action_name"] == "use_key":
        return "Fail", "Untrained search + trained use still solved red."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Trained search with use-gate off still solved red."
    if m["name_swap"]["correct"] or m["name_swap"]["action_name"] == "use_key":
        return "Fail", "Messy document is wait and still use_key; fact leaked."
    if not (m["search_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    if not m["red_live"]["found_action2"]:
        return "Fail", "Free red life never committed the messy unread document."
    tag = m.get("red_tag", "")
    if _has_field(tag, "loc") or _has_field(tag, "door") or _has_field(tag, "here"):
        return "Fail", "Red S used an exact place-name query."
    return (
        "Store-works",
        "Free life over .md documents; after ρ reset and W gone, probe uses S; cortex frozen; held-out green wait.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key after a free life on documents."
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["red_after_reset"]["correct"]:
        return "Fail", "Shared return did not solve red after a free life on documents."
    if not m["green_after_reset"]["correct"]:
        return "Fail", "Held-out green failed under shared return."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if not (m["search_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return on a free life over documents; cortex frozen; W gone after reset; held-out green wait.",
    )


def _train(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    *,
    split: bool,
    max_steps: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_u = 0.0
    s_dir = work / "ep"
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(
            s_dir,
            w_dir,
            policy,
            epsilon=eps,
            explore_epsilon=explore_eps,
            rng=rng,
            record_search_on_explore=True,
        )
        ag.policy_traces = []
        ag.reset_rho()
        live = live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        r_found = 1.0 if live["found_action2"] else 0.0
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
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
    max_steps: int,
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
    dummy = make(run_dir / f"{arm}_hash", None, policy, explore_epsilon=0.0)
    cortex0 = dummy.weight_hash()

    _, untrained_live, untrained_probe, _ = _life_then_probe(
        make,
        s_un,
        w_red,
        policy,
        "experience_teach",
        "probe_red_with_key",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 99),
    )

    rewards = _train(policy, w_red, work, n_train, train_seed, split=split, max_steps=max_steps)
    search1 = _head_fp(policy, "search")
    use1 = _head_fp(policy, "use")

    red_a, red_live, red_after, red_tag = _life_then_probe(
        make,
        s_red,
        w_red,
        policy,
        "experience_teach",
        "probe_red_with_key",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 1),
    )
    _, green_live, green_after, green_tag = _life_then_probe(
        make,
        s_green,
        w_green,
        policy,
        "experience_green",
        "probe_green",
        seed + 20,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 3),
    )
    _, _, menu_control, _ = _life_then_probe(
        lambda s, w, p, **kw: make(s, w, p, use_search_head=False, use_match_head=True, **kw),
        s_menu,
        w_red,
        policy,
        "experience_teach",
        "probe_red_with_key",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 4),
    )
    _, _, search_control, _ = _life_then_probe(
        make,
        s_s,
        w_red,
        _copy_heads(policy, "use"),
        "experience_teach",
        "probe_red_with_key",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 5),
    )
    _, _, use_control, _ = _life_then_probe(
        make,
        s_u,
        w_red,
        _copy_heads(policy, "search"),
        "experience_teach",
        "probe_red_with_key",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 6),
    )
    _, _, name_swap, swap_tag = _life_then_probe(
        make,
        s_swap,
        w_swap,
        policy,
        "experience_teach",
        "probe_red_with_key",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=0.5,
        rng=np.random.default_rng(seed + 7),
    )

    empty = make(s_empty, None, policy, explore_epsilon=0.0)
    empty.reset_rho()
    empty_p = probe(empty, "probe_red_with_key", seed + 10)
    empty_g = probe(empty, "probe_green", seed + 20)
    off = make(s_off, None, policy, enabled=False, explore_epsilon=0.5, rng=np.random.default_rng(seed + 8))
    off.reset_rho()
    live_free(off, "experience_teach", seed + 10, max_steps=max_steps)
    off.reset_rho()
    disable_red = probe(off, "probe_red_with_key", seed + 10)

    metrics: dict[str, Any] = {
        "arm": arm,
        "trained_split": split,
        "trained_life": True,
        "trained_force_use": dummy.force_use,
        "write_from_events": dummy.write_from_events,
        "use_search_head": dummy.use_search_head,
        "use_match_head": dummy.use_match_head,
        "use_qname_head": dummy.use_qname_head,
        "use_read": dummy.use_read,
        "record_search_on_explore": dummy.record_search_on_explore,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "search_changed": search0 != search1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_red),
        "untrained_live": {k: untrained_live[k] for k in ("opened", "files", "n_forced", "n_steps", "n_explored", "found_action2", "actions")},
        "untrained_probe": untrained_probe,
        "red_live": {k: red_live[k] for k in ("opened", "files", "n_forced", "n_steps", "n_explored", "found_action2", "actions")},
        "red_after_reset": red_after,
        "red_tag": red_tag,
        "green_live": {k: green_live[k] for k in ("opened", "files", "n_forced", "n_steps", "n_explored", "found_action2", "actions")},
        "green_after_reset": green_after,
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


def run_tm031(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    w_swap = run_dir / "W_swap"
    write_doc_notes(w_red, wiki_docs(include_red=True))
    write_doc_notes(w_green, wiki_docs(include_green=True))
    write_doc_notes(w_swap, clutter_docs() + [SWAP_RED])
    w_files = sorted(p.name for p in w_red.glob("*.md"))
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
        max_steps=max_steps,
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
        max_steps=max_steps,
    )
    out = {
        "version": "TM.0.3.1",
        "seed": seed,
        "n_train": n_train,
        "max_steps": max_steps,
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.3.1 A documents free life vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split documents | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained probe after life | {a['untrained_probe']['action_name']} ({a['untrained_probe']['correct']}) | {b['untrained_probe']['action_name']} ({b['untrained_probe']['correct']}) |
| Trained red after ρ reset, W gone | {a['red_after_reset']['action_name']} ({a['red_after_reset']['correct']}) | {b['red_after_reset']['action_name']} ({b['red_after_reset']['correct']}) |
| Held-out green after reset | {a['green_after_reset']['action_name']} ({a['green_after_reset']['correct']}) | {b['green_after_reset']['action_name']} ({b['green_after_reset']['correct']}) |
| Red life found messy doc | {a['red_live']['found_action2']} | {b['red_live']['found_action2']} |
| W all .md / has prose | {a['w_all_md']} / {a['w_has_prose']} | {b['w_all_md']} / {b['w_has_prose']} |
| Exact-match / search-off / use-off | {a['menu_control']['action_name']} / {a['search_control']['action_name']} / {a['use_control']['action_name']} | {b['menu_control']['action_name']} / {b['search_control']['action_name']} / {b['use_control']['action_name']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.3.1 documents free life; split vs shared return")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm031(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print(
        "A",
        m["A"]["untrained_probe"]["action_name"],
        m["A"]["red_after_reset"]["action_name"],
        m["A"]["green_after_reset"]["action_name"],
        "found",
        m["A"]["red_live"]["found_action2"],
        "md",
        m["A"]["w_all_md"],
        m["A"]["red_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["untrained_probe"]["action_name"],
        m["B"]["red_after_reset"]["action_name"],
        m["B"]["green_after_reset"]["action_name"],
        "found",
        m["B"]["red_live"]["found_action2"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
