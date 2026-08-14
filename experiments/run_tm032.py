"""TM.0.3.2: prose retrieve. Free life over .md pages with no filed where=/action=.

A split credit. B shared return.
Digits are scanned into anonymous n* tags; vname picks which int to copy.
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
from experiments.run_v22 import _tags, probe
from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld
from three_memory.policy import UsePolicy
from three_memory.symbols import ACT_OPEN, ACT_USE_KEY, ACT_WAIT, BLUE_FACT_ID, DOOR_GREEN, DOOR_RED, GREEN_FACT_ID, RED_FACT_ID
from three_memory.tag_store import ProseLibrary, TagStore, extract_prose_ints, prose_to_record, write_prose_notes

# Pure prose. No where=/action=/loc=/door=/here= lines.
WIKI_RED = (
    "p99.md",
    f"Staff scrap from the red room. At place {DOOR_RED} the working motor was {ACT_USE_KEY}.\n",
)
WIKI_GREEN = (
    "p98.md",
    f"Staff scrap from the green room. At place {DOOR_GREEN} the working motor was {ACT_WAIT}.\n",
)
SWAP_RED = (
    "p99.md",
    f"Staff scrap from the red room. At place {DOOR_RED} the working motor was {ACT_WAIT}.\n",
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm032"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clutter_prose() -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    for i in range(11):
        place = i % 3
        # Filenames without digits in the body; never plant USE_KEY.
        body = f"Clutter hallway notes. Place {place} is busy with open tries {ACT_OPEN}."
        notes.append((f"c{i:02d}.md", body + "\n"))
    return notes


def wiki_prose(*, include_red: bool = False, include_green: bool = False) -> list[tuple[str, str]]:
    notes = clutter_prose()
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
        "w_has_action": _has_field(texts, "action"),
        "w_has_door": _has_field(texts, "door"),
        "w_has_act": _has_field(texts, "act"),
        "w_has_do": _has_field(texts, "do"),
        "w_has_pad": _has_field(texts, "pad"),
        "w_has_place": _has_field(texts, "place"),
        "w_has_prose": any(
            ln.strip() and not ln.strip().startswith("#") and "=" not in ln
            for ln in texts.splitlines()
        ),
        "w_red_ints": (
            [int(v) for k, v in sorted(prose_to_record(w_dir / WIKI_RED[0]).tags.items()) if k.startswith("n")]
            if (w_dir / WIKI_RED[0]).exists() and prose_to_record(w_dir / WIKI_RED[0]) is not None
            else []
        ),
    }


def _s_has_red_pair(tag: str) -> bool:
    """Committed anonymous ints include door-red code and USE_KEY."""
    vals = set()
    for ln in tag.splitlines():
        if "=" in ln and not ln.startswith("#"):
            k, _, v = ln.partition("=")
            if k.strip().startswith("n"):
                try:
                    vals.add(int(v.strip()))
                except ValueError:
                    pass
    return DOOR_RED in vals and ACT_USE_KEY in vals


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
    use_vname_head: bool = True,
    record_search_on_explore: bool = True,
    use_prose_ints: bool = True,
) -> ThreeMemoryAgent:
    world = ProseLibrary(w_dir) if w_dir is not None else None
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
        use_vname_head=use_vname_head,
        force_use=force_use,
        record_search_on_explore=record_search_on_explore,
        use_prose_ints=use_prose_ints,
    )


def live_free(
    agent: ThreeMemoryAgent,
    scenario: str,
    seed: int,
    *,
    max_steps: int = 32,
) -> dict[str, Any]:
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
        "found_red_pair": _s_has_red_pair(tag),
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
        return "Confound", "W still used .tag files; prose jump abandoned."
    if not m["w_has_prose"]:
        return "Confound", "W .md files have no prose."
    if (
        m["w_has_where"]
        or m["w_has_action"]
        or m["w_has_loc"]
        or m["w_has_here"]
        or m["w_has_door"]
        or m["w_has_act"]
        or m["w_has_do"]
        or m["w_has_pad"]
        or m["w_has_place"]
        or m["w_has_when"]
    ):
        return "Confound", "Filed k=v motor/place tags were planted; not prose."
    if m["w_has_red"] or m["w_has_green"] or m["w_has_blue"]:
        return "Confound", "Answer filename d0/d1/d2 was in W."
    if not m["w_has_p99"]:
        return "Confound", "W must have the useful prose page p99.md."
    if DOOR_RED not in m.get("w_red_ints", []) or ACT_USE_KEY not in m.get("w_red_ints", []):
        return "Confound", "Useful prose must contain door code and motor ints as digits."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue prose."
    if m["use_match_head"] or m["use_qname_head"]:
        return "Fail", "Exact query match was restored."
    if not (m["use_search_head"] and m["use_vname_head"] and m["use_read"] and m["use_prose_ints"]):
        return "Fail", "Prose search/vname path was frozen off."
    if not m["trained_life"]:
        return "Fail", "Training was not a free life."
    if any(lv.get("n_forced", 0) for lv in (m["untrained_live"], m["red_live"], m["green_live"])):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_red"]["correct"]:
        return "Confound", "disable-S still used the key; fact leaked."
    if any(p.get("explored") for p in (m["untrained_probe"], m["red_after_reset"], m["green_after_reset"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_green"]["correct"] or m["empty_S_green"]["action_name"] == "wait":
        return "Confound", "Empty S already wait on green."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key after a free life on prose."
    if not m["red_after_reset"]["correct"]:
        return "Fail", "Free red life on prose did not leave a usable S after ρ reset / W gone."
    if not m["green_after_reset"]["correct"]:
        return "Fail", "Held-out green free life on prose failed after reset."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if m["menu_control"]["correct"] or m["menu_control"]["action_name"] == "use_key":
        return "Fail", "Exact {door, here} match still solved red."
    if m["search_control"]["correct"] or m["search_control"]["action_name"] == "use_key":
        return "Fail", "Untrained search + trained use/vname still solved red."
    if m["use_control"]["correct"] or m["use_control"]["action_name"] == "use_key":
        return "Fail", "Trained search with use/vname frozen off still solved red."
    if m["name_swap"]["correct"] or m["name_swap"]["action_name"] == "use_key":
        return "Fail", "Prose swap is wait and still use_key; fact leaked."
    if not (m["search_changed"] and m["vname_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    if not m["red_live"]["found_red_pair"]:
        return "Fail", "Free red life never committed prose ints for door+motor."
    tag = m.get("red_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "where") or _has_field(tag, "loc") or _has_field(tag, "door"):
        return "Fail", "Red S restored filed place/motor tag names."
    return (
        "Store-works",
        "Free life over prose .md (no filed action=); after ρ reset W gone, probe uses S; cortex frozen; held-out green wait.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "use_key":
        return "Fail", "Untrained already used the key after a free life on prose."
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["red_after_reset"]["correct"]:
        return "Fail", "Shared return did not solve red after a free life on prose."
    if not m["green_after_reset"]["correct"]:
        return "Fail", "Held-out green failed under shared return."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still use_key."
    if not (m["search_changed"] and m["vname_changed"] and m["use_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return on a free life over prose; cortex frozen; W gone after reset; held-out green wait.",
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
        )
        ag.policy_traces = []
        ag.reset_rho()
        live = live_free(ag, "experience_teach", seed + 10, max_steps=max_steps)
        r_found = 1.0 if live["found_red_pair"] else 0.0
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_red_with_key", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        if split:
            b_f = 0.9 * b_f + 0.1 * r_found
            b_u = 0.9 * b_u + 0.1 * r_use
            policy.update([t for t in tr if t.get("kind") == "search"], r_found - b_f)
            policy.update([t for t in tr if t.get("kind") in ("vname", "use")], r_use - b_u)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") == "search"], adv)
            policy.update([t for t in tr if t.get("kind") in ("vname", "use")], adv)
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
    vname0 = _head_fp(policy, "vname")
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
    vname1 = _head_fp(policy, "vname")
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
        lambda s, w, p, **kw: make(
            s, w, p, use_search_head=False, use_match_head=True, use_vname_head=False, use_prose_ints=False, **kw
        ),
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
        _copy_heads(policy, "use", "vname"),
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
        "use_vname_head": dummy.use_vname_head,
        "use_read": dummy.use_read,
        "use_prose_ints": dummy.use_prose_ints,
        "record_search_on_explore": dummy.record_search_on_explore,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == red_a.weight_hash(),
        "search_changed": search0 != search1,
        "vname_changed": vname0 != vname1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_red),
        "untrained_live": {
            k: untrained_live[k]
            for k in ("opened", "files", "n_forced", "n_steps", "n_explored", "found_red_pair", "actions")
        },
        "untrained_probe": untrained_probe,
        "red_live": {
            k: red_live[k] for k in ("opened", "files", "n_forced", "n_steps", "n_explored", "found_red_pair", "actions")
        },
        "red_after_reset": red_after,
        "red_tag": red_tag,
        "green_live": {
            k: green_live[k]
            for k in ("opened", "files", "n_forced", "n_steps", "n_explored", "found_red_pair", "actions")
        },
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


def run_tm032(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_red = run_dir / "W_red"
    w_green = run_dir / "W_green"
    w_swap = run_dir / "W_swap"
    write_prose_notes(w_red, wiki_prose(include_red=True))
    write_prose_notes(w_green, wiki_prose(include_green=True))
    write_prose_notes(w_swap, clutter_prose() + [SWAP_RED])
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
        "version": "TM.0.3.2",
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
        f"""# TM.0.3.2 A prose free life vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split prose | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained probe after life | {a['untrained_probe']['action_name']} ({a['untrained_probe']['correct']}) | {b['untrained_probe']['action_name']} ({b['untrained_probe']['correct']}) |
| Trained red after ρ reset, W gone | {a['red_after_reset']['action_name']} ({a['red_after_reset']['correct']}) | {b['red_after_reset']['action_name']} ({b['red_after_reset']['correct']}) |
| Held-out green after reset | {a['green_after_reset']['action_name']} ({a['green_after_reset']['correct']}) | {b['green_after_reset']['action_name']} ({b['green_after_reset']['correct']}) |
| Red life found door+motor ints | {a['red_live']['found_red_pair']} | {b['red_live']['found_red_pair']} |
| Filed where=/action= in W | {a['w_has_where']} / {a['w_has_action']} | {b['w_has_where']} / {b['w_has_action']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.3.2 prose retrieve free life; split vs shared return")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm032(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(json.dumps({"A": m["A"]["classification"], "B": m["B"]["classification"], "run_dir": m["run_dir"]}, indent=2))
    print(
        "A",
        m["A"]["untrained_probe"]["action_name"],
        m["A"]["red_after_reset"]["action_name"],
        m["A"]["green_after_reset"]["action_name"],
        "found",
        m["A"]["red_live"]["found_red_pair"],
        m["A"]["red_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["untrained_probe"]["action_name"],
        m["B"]["red_after_reset"]["action_name"],
        m["B"]["green_after_reset"]["action_name"],
        "found",
        m["B"]["red_live"]["found_red_pair"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
