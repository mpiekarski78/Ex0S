"""TM.0.5.1: correct a wrong commit. Drop junk S, retry, keep after ρ reset.

A split credit. B shared return.
Search stays untrained (first remaining file) so correction is load-bearing.
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
from experiments.run_tm040 import probe
from experiments.run_tm050 import (
    SWAP_A,
    _s_has_token,
    _w_flags,
    clutter_prose,
    wiki_prose,
)
from experiments.run_v22 import _tags
from three_memory.agent import ThreeMemoryAgent
from three_memory.dial_env import DialAction
from three_memory.policy import UsePolicy
from three_memory.tag_store import ProseLibrary, TagStore, write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm051"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    use_prose_tokens: bool = True,
    use_prose_ints: bool = False,
    use_revise_head: bool = True,
) -> ThreeMemoryAgent:
    world = ProseLibrary(w_dir) if w_dir is not None else None
    return ThreeMemoryAgent(
        store_enabled=enabled,
        cortex_seed=1337,
        native=True,
        n_actions=5,
        domain="dial",
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
        use_prose_tokens=use_prose_tokens,
        use_revise_head=use_revise_head,
    )


def live_free(
    agent: ThreeMemoryAgent,
    scenario: str,
    seed: int,
    *,
    max_steps: int = 40,
) -> dict[str, Any]:
    from three_memory.dial_env import ChannelDialWorld, DIAL_ACTION_NAMES

    world = ChannelDialWorld(seed=seed)
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
                "action_name": DIAL_ACTION_NAMES.get(DialAction(action), str(action)),
                "explored": bool(meta.get("explored")),
                "success": result.success,
                "opened": bool(result.info.get("opened")),
                "forced": False,
            }
        )
        opened = opened or bool(result.info.get("opened"))
        obs = result.obs
        # Lucky PRESS must not abort correction; probe is still a one-shot greedy act.
        if result.done and not str(scenario).startswith("experience"):
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
        "n_revised": int(agent.n_revised),
        "actions": [s["action_name"] for s in log],
        "found_press": _s_has_token(tag, "press"),
        "found_tune": _s_has_token(tag, "tune"),
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
    if m.get("door_world"):
        return "Confound", "Door world was restored."
    if m.get("world") != "channel_dial":
        return "Confound", "Not the channel dial world."
    if m["w_has_tag_files"] or not m["w_all_md"]:
        return "Confound", "W still used .tag files."
    if m.get("w_body_ints") or m.get("w_a_ints"):
        return "Confound", "Answer integers were restored in W."
    if m.get("use_prose_ints"):
        return "Confound", "Digit-copy was restored."
    if m["w_has_where"] or m["w_has_action"] or m["w_has_loc"] or m["w_has_here"] or m["w_has_door"]:
        return "Confound", "Filed k=v tags were planted."
    if not m["w_has_p99"] or "press" not in m.get("w_a_tokens", []):
        return "Confound", "Useful prose p99.md / press token missing."
    if m["write_from_events"]:
        return "Confound", "Writes from life were on."
    if m["search_changed"]:
        return "Confound", "Search was trained to skip correction."
    if m["trained_force_use"]:
        return "Fail", "Use clamped to rescue the dial."
    if not (m["use_search_head"] and m["use_vname_head"] and m["use_read"] and m["use_prose_tokens"] and m["use_revise_head"]):
        return "Fail", "Token + revise path was frozen off."
    if not m["trained_life"]:
        return "Fail", "Training was not a free life."
    if any(lv.get("n_forced", 0) for lv in (m["untrained_live"], m["a_live"], m["c_live"])):
        return "Confound", "A forced curriculum ran."
    if m["disable_S_a"]["correct"]:
        return "Confound", "disable-S still solved A; fact leaked outside S."
    if any(p.get("explored") for p in (m["untrained_probe"], m["a_after_reset"], m["c_after_reset"])):
        return "Confound", "Probe used exploration."
    if m["empty_S_c"]["correct"] or m["empty_S_c"]["action_name"] == "tune":
        return "Confound", "Empty S already TUNE on held-out C."
    if m.get("domain") != "dial" or m.get("n_actions") != 5:
        return "Confound", "Agent was not dial domain / 5 actions."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "press":
        return "Fail", "Untrained already PRESS-correct on A."
    if m["untrained_live"].get("found_press"):
        return "Fail", "Untrained already found press without learning to correct."
    if (m["untrained_live"].get("n_revised") or 0) > 0:
        return "Fail", "Untrained already revised S."
    if not m["a_after_reset"]["correct"] or m["a_after_reset"]["action_name"] != "press":
        return "Fail", "After ρ reset W gone, A was not PRESS from corrected S."
    if not m["c_after_reset"]["correct"] or m["c_after_reset"]["action_name"] != "tune":
        return "Fail", "Held-out C did not TUNE after correction."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if m["revise_control"]["correct"]:
        return "Fail", "Revise-off still solved A; search/copy alone rescued."
    if m["use_control"]["correct"]:
        return "Fail", "Use/vname off still solved A."
    if m["name_swap"]["correct"]:
        return "Fail", "Swap idle still solved A."
    if not m["revise_changed"]:
        return "Fail", "Revise head did not move."
    if not (m["vname_changed"] and m["use_changed"]):
        return "Fail", "Vname/use did not move."
    if (m["a_live"].get("n_revised") or 0) < 1:
        return "Fail", "Eval A life never revised S."
    if (m["c_live"].get("n_revised") or 0) < 1:
        return "Fail", "Eval C life never revised S."
    if not m["a_live"]["found_press"]:
        return "Fail", "Corrected S never held press."
    if not m["c_live"]["found_tune"]:
        return "Fail", "Corrected S never held tune."
    tag = m.get("a_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "door") or _has_field(tag, "n0"):
        return "Fail", "S restored filed or digit tags."
    return (
        "Store-works",
        "Wrong commit dropped; after ρ reset W gone, A PRESS from corrected S; held-out C TUNE; search frozen; cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["a_after_reset"]["correct"]:
        return "Fail", "Shared return did not solve A after a free life."
    if m["a_after_reset"]["action_name"] != "press":
        return "Fail", "Shared return A probe was not PRESS."
    if not m["c_after_reset"]["correct"] or m["c_after_reset"]["action_name"] != "tune":
        return "Fail", "Held-out C failed under shared return."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if not m["revise_changed"]:
        return "Fail", "Revise head did not move."
    return (
        "Store-works",
        "Shared return on correction; cortex frozen; held-out C TUNE.",
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
        explore_eps = 0.20 * (1.0 - 0.4 * ep / max(n, 1))
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        ag = make(s_dir, w_dir, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live = live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps)
        r_found = 1.0 if live["found_press"] else 0.0
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_channel_a", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        if split:
            b_f = 0.9 * b_f + 0.1 * r_found
            b_u = 0.9 * b_u + 0.1 * r_use
            policy.update([t for t in tr if t.get("kind") == "revise"], r_found - b_f)
            policy.update([t for t in tr if t.get("kind") in ("vname", "use")], r_use - b_u)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") == "revise"], adv)
            policy.update([t for t in tr if t.get("kind") in ("vname", "use")], adv)
        rewards.append(r_use)
    return rewards


def run_arm(
    *,
    arm: str,
    split: bool,
    run_dir: Path,
    w_a: Path,
    w_c: Path,
    w_swap: Path,
    w_files: list[str],
    seed: int,
    n_train: int,
    train_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    work = run_dir / f"{arm}_train"
    dirs = {
        k: run_dir / f"{arm}_{k}"
        for k in ("untrained", "a", "c", "revisectrl", "usectrl", "swap", "off", "empty")
    }
    for d in (work, *dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    policy = UsePolicy(seed=7, lr=0.2)
    search0 = _head_fp(policy, "search")
    revise0, vname0, use0 = _head_fp(policy, "revise"), _head_fp(policy, "vname"), _head_fp(policy, "use")
    dummy = make(run_dir / f"{arm}_hash", None, policy, explore_epsilon=0.0)
    cortex0 = dummy.weight_hash()

    eval_explore = 0.2
    _, untrained_live, untrained_probe, _ = _life_then_probe(
        make,
        dirs["untrained"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=eval_explore,
        rng=np.random.default_rng(seed + 99),
    )

    rewards = _train(policy, w_a, work, n_train, train_seed, split=split, max_steps=max_steps)
    search1 = _head_fp(policy, "search")
    revise1, vname1, use1 = _head_fp(policy, "revise"), _head_fp(policy, "vname"), _head_fp(policy, "use")

    a_ag, a_live, a_after, a_tag = _life_then_probe(
        make,
        dirs["a"],
        w_a,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=eval_explore,
        rng=np.random.default_rng(seed + 1),
    )
    _, c_live, c_after, c_tag = _life_then_probe(
        make,
        dirs["c"],
        w_c,
        policy,
        "experience_channel_c",
        "probe_channel_c",
        seed + 20,
        max_steps=max_steps,
        explore_epsilon=eval_explore,
        rng=np.random.default_rng(seed + 3),
    )
    _, _, revise_control, _ = _life_then_probe(
        lambda s, w, p, **kw: make(s, w, p, use_revise_head=False, **kw),
        dirs["revisectrl"],
        w_a,
        _copy_heads(policy, "use", "vname"),
        "experience_channel_a",
        "probe_channel_a",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=eval_explore,
        rng=np.random.default_rng(seed + 5),
    )
    _, _, use_control, _ = _life_then_probe(
        make,
        dirs["usectrl"],
        w_a,
        _copy_heads(policy, "revise"),
        "experience_channel_a",
        "probe_channel_a",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=eval_explore,
        rng=np.random.default_rng(seed + 6),
    )
    _, _, name_swap, swap_tag = _life_then_probe(
        make,
        dirs["swap"],
        w_swap,
        policy,
        "experience_channel_a",
        "probe_channel_a",
        seed + 10,
        max_steps=max_steps,
        explore_epsilon=eval_explore,
        rng=np.random.default_rng(seed + 7),
    )

    empty = make(dirs["empty"], None, policy, explore_epsilon=0.0)
    empty.reset_rho()
    empty_p = probe(empty, "probe_channel_a", seed + 10)
    empty_c = probe(empty, "probe_channel_c", seed + 20)
    off = make(
        dirs["off"], None, policy, enabled=False, explore_epsilon=eval_explore, rng=np.random.default_rng(seed + 8)
    )
    off.reset_rho()
    live_free(off, "experience_channel_a", seed + 10, max_steps=max_steps)
    off.reset_rho()
    disable_a = probe(off, "probe_channel_a", seed + 10)

    live_keys = (
        "opened",
        "files",
        "n_forced",
        "n_steps",
        "n_explored",
        "n_revised",
        "found_press",
        "found_tune",
        "actions",
    )
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
        "use_prose_tokens": dummy.use_prose_tokens,
        "use_revise_head": dummy.use_revise_head,
        "domain": dummy.domain,
        "n_actions": dummy.n_actions,
        "train_return_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "cortex_hash": cortex0,
        "cortex_unchanged": cortex0 == a_ag.weight_hash(),
        "search_changed": search0 != search1,
        "revise_changed": revise0 != revise1,
        "vname_changed": vname0 != vname1,
        "use_changed": use0 != use1,
        "policy_n_updates": policy.n_updates,
        **_w_flags(w_files, w_a),
        "untrained_live": {k: untrained_live[k] for k in live_keys},
        "untrained_probe": untrained_probe,
        "a_live": {k: a_live[k] for k in live_keys},
        "a_after_reset": a_after,
        "a_tag": a_tag,
        "c_live": {k: c_live[k] for k in live_keys},
        "c_after_reset": c_after,
        "c_tag": c_tag,
        "revise_control": revise_control,
        "use_control": use_control,
        "name_swap": name_swap,
        "swap_tag": swap_tag,
        "empty_S": empty_p,
        "empty_S_c": empty_c,
        "disable_S_a": disable_a,
    }
    label, rationale = (classify_a if split else classify_b)(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    return metrics


def run_tm051(seed: int = 12345, n_train: int = 500, max_steps: int = 40) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_c = run_dir / "W_c"
    w_swap = run_dir / "W_swap"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_c, wiki_prose(include_c=True))
    write_prose_notes(w_swap, clutter_prose() + [SWAP_A])
    w_files = sorted(p.name for p in w_a.glob("*.md"))
    a = run_arm(
        arm="A",
        split=True,
        run_dir=run_dir,
        w_a=w_a,
        w_c=w_c,
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
        w_a=w_a,
        w_c=w_c,
        w_swap=w_swap,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
        max_steps=max_steps,
    )
    out = {
        "version": "TM.0.5.1",
        "seed": seed,
        "n_train": n_train,
        "max_steps": max_steps,
        "world": "channel_dial",
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.5.1 A correct vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split correct | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained probe A | {a['untrained_probe']['action_name']} ({a['untrained_probe']['correct']}) | {b['untrained_probe']['action_name']} ({b['untrained_probe']['correct']}) |
| n_revised A life | {a['a_live']['n_revised']} | {b['a_live']['n_revised']} |
| Trained A after ρ reset | {a['a_after_reset']['action_name']} ({a['a_after_reset']['correct']}) | {b['a_after_reset']['action_name']} ({b['a_after_reset']['correct']}) |
| Held-out C after reset | {a['c_after_reset']['action_name']} ({a['c_after_reset']['correct']}) | {b['c_after_reset']['action_name']} ({b['c_after_reset']['correct']}) |
| Revise-off control | {a['revise_control']['action_name']} ({a['revise_control']['correct']}) | {b['revise_control']['action_name']} ({b['revise_control']['correct']}) |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.1 correct a wrong commit")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=40)
    args = p.parse_args()
    m = run_tm051(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {"A": m["A"]["classification"], "B": m["B"]["classification"], "world": m["world"], "run_dir": m["run_dir"]},
            indent=2,
        )
    )
    print(
        "A",
        m["A"]["untrained_probe"]["action_name"],
        m["A"]["a_after_reset"]["action_name"],
        m["A"]["c_after_reset"]["action_name"],
        "n_rev",
        m["A"]["a_live"]["n_revised"],
        m["A"]["a_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["untrained_probe"]["action_name"],
        m["B"]["a_after_reset"]["action_name"],
        m["B"]["c_after_reset"]["action_name"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
