"""v0: key/door fact survives ρ reset iff inspectable store S is on."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.agent import ThreeMemoryAgent
from three_memory.env import Action, KeyDoorWorld


def _run_dir(prefix: str = "v0") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_{prefix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(exist_ok=True)
    return d


def teach_red_door(agent: ThreeMemoryAgent, seed: int = 0, max_steps: int = 12) -> dict[str, Any]:
    """Experience that writes the red-door fact into S (when enabled) and stains ρ."""
    world = KeyDoorWorld(seed=seed)
    obs = world.reset("experience_teach")
    log = []
    for _ in range(max_steps):
        action, meta = agent.act(obs)
        result = world.step(action, "experience_teach")
        write_meta = agent.observe_outcome(result.obs, result.success, result.info)
        log.append(
            {
                "action": int(action),
                "success": result.success,
                "info": result.info,
                "meta": meta,
                "write": write_meta,
            }
        )
        obs = result.obs
        if result.done:
            break

    # Ensure the contingency was experienced at least once (OPEN fail + USE_KEY success).
    # Fallback curriculum if free policy never opens the door — still event-driven writes, not labeled facts.
    saw_key_success = any(
        (step.get("info") or {}).get("opened") and step.get("action") == int(Action.USE_KEY)
        for step in log
    )
    if not saw_key_success:
        world = KeyDoorWorld(seed=seed + 1)
        obs = world.reset("experience_teach")
        for forced in (Action.OPEN, Action.PICK_KEY, Action.USE_KEY):
            _, meta = agent.act(obs, update_rho=True)
            result = world.step(int(forced), "experience_teach")
            write_meta = agent.observe_outcome(result.obs, result.success, result.info)
            log.append(
                {
                    "action": int(forced),
                    "forced": True,
                    "success": result.success,
                    "info": result.info,
                    "meta": meta,
                    "write": write_meta,
                }
            )
            obs = result.obs
            if result.done:
                break

    return {
        "steps": log,
        "store": agent.store.to_jsonable(),
        "has_fact": agent.store.has_fact(KeyDoorWorld.FACT_ID),
        "rho_l2": float(np.linalg.norm(agent.rho.rho)),
        "weights_unchanged": agent.weights_unchanged(),
        "n_forced_steps": sum(1 for s in log if s.get("forced")),
    }


def foil_blue_door(agent: ThreeMemoryAgent, seed: int = 0, max_steps: int = 8) -> dict[str, Any]:
    world = KeyDoorWorld(seed=seed)
    obs = world.reset("experience_foil")
    log = []
    for _ in range(max_steps):
        action, meta = agent.act(obs)
        # Foil curriculum: open blue door.
        if action != Action.OPEN:
            action = Action.OPEN
        result = world.step(int(action), "experience_foil")
        write_meta = agent.observe_outcome(result.obs, result.success, result.info)
        log.append({"action": int(action), "success": result.success, "write": write_meta, "meta": meta})
        obs = result.obs
        if result.done:
            break
    return {
        "steps": log,
        "store": agent.store.to_jsonable(),
        "has_fact": agent.store.has_fact(KeyDoorWorld.FACT_ID),
        "rho_l2": float(np.linalg.norm(agent.rho.rho)),
        "weights_unchanged": agent.weights_unchanged(),
    }


def probe(agent: ThreeMemoryAgent, scenario: str = "probe_red_with_key", seed: int = 99) -> dict[str, Any]:
    world = KeyDoorWorld(seed=seed)
    obs = world.reset(scenario)
    action, meta = agent.act(obs, update_rho=False)
    result = world.step(action, scenario)
    correct = False
    if scenario == "probe_red_with_key":
        correct = action == Action.USE_KEY and bool(result.info.get("opened"))
    elif scenario == "probe_blue":
        correct = action == Action.OPEN and bool(result.info.get("opened"))
    return {
        "scenario": scenario,
        "action": int(action),
        "action_name": Action(action).name.lower(),
        "correct": correct,
        "opened": bool(result.info.get("opened")),
        "meta": meta,
        "store_len": len(agent.store),
        "has_red_fact": agent.store.has_fact(KeyDoorWorld.FACT_ID),
        "rho_l2": float(np.linalg.norm(agent.rho.rho)),
    }


def classify(metrics: dict[str, Any]) -> tuple[str, str]:
    """Predeclared categories from the protocol."""
    if not metrics.get("weights_unchanged_all", True):
        return "Confound", "Slow weights changed during the life (illegal in v0)."

    a_after = metrics["A_probe_after_rho_reset"]["correct"]
    a_store = metrics["A_has_inspectable_fact"]
    b_after = metrics["B_probe_after_rho_reset"]["correct"]
    disable = metrics["disable_S_probe_after_rho_reset"]["correct"]
    a_before_reset = metrics["A_probe_before_rho_reset"]["correct"]

    if a_after and a_store and not b_after and not disable:
        return (
            "Store-works",
            "Fact survives ρ reset via inspectable S; B and disable-S fail (BDH-like Category B).",
        )
    if (a_before_reset or metrics["A_rho_l2_after_exp"] > 1e-6) and not a_after and not disable:
        return (
            "Trace-only",
            "Session residue moved behavior, but ρ reset wiped it (same letter as BDH Category B).",
        )
    return "Fail", "Store did not produce the intended A/B + reset pattern."


def run_v0(seed: int = 12345) -> dict[str, Any]:
    run_dir = _run_dir("v0")

    # --- A: store on, experience red-door fact ---
    A = ThreeMemoryAgent(store_enabled=True, cortex_seed=1337)
    h0 = A.weight_hash()
    teach_A = teach_red_door(A, seed=seed)
    rho_A = A.rho.snapshot()
    probe_A_before = probe(A, "probe_red_with_key", seed=seed + 10)
    A.reset_rho()
    probe_A_after = probe(A, "probe_red_with_key", seed=seed + 10)
    A.rho.load(rho_A)
    probe_A_restored = probe(A, "probe_red_with_key", seed=seed + 10)
    A.reset_rho()
    # Keep S; already reset ρ for after-reset probe.

    # --- B: same cortex, foil experience (no red-door fact) ---
    B = ThreeMemoryAgent(store_enabled=True, cortex_seed=1337)
    teach_B = foil_blue_door(B, seed=seed + 1)
    B.reset_rho()
    probe_B_after = probe(B, "probe_red_with_key", seed=seed + 10)

    # --- disable-S: same experience as A but S writes blocked ---
    C = ThreeMemoryAgent(store_enabled=False, cortex_seed=1337)
    teach_C = teach_red_door(C, seed=seed)
    probe_C_before = probe(C, "probe_red_with_key", seed=seed + 10)
    C.reset_rho()
    probe_C_after = probe(C, "probe_red_with_key", seed=seed + 10)

    # --- identical-experience twins ---
    T1 = ThreeMemoryAgent(store_enabled=True, cortex_seed=1337)
    T2 = ThreeMemoryAgent(store_enabled=True, cortex_seed=1337)
    teach_red_door(T1, seed=seed)
    teach_red_door(T2, seed=seed)
    twin_dist = T1.rho.distance(T2.rho)

    # --- reset S after A learned ---
    A2 = ThreeMemoryAgent(store_enabled=True, cortex_seed=1337)
    teach_red_door(A2, seed=seed)
    A2.reset_rho()
    A2.reset_store()
    probe_reset_S = probe(A2, "probe_red_with_key", seed=seed + 10)

    weights_ok = (
        A.weight_hash() == h0
        and B.weight_hash() == h0
        and C.weight_hash() == h0
        and teach_A["weights_unchanged"]
        and teach_B["weights_unchanged"]
        and teach_C["weights_unchanged"]
    )

    metrics: dict[str, Any] = {
        "seed": seed,
        "weight_hash": h0,
        "weights_unchanged_all": weights_ok,
        "A_teach": {
            "has_fact": teach_A["has_fact"],
            "store": teach_A["store"],
            "rho_l2": teach_A["rho_l2"],
            "n_forced_steps": teach_A["n_forced_steps"],
        },
        "B_teach": {
            "has_fact": teach_B["has_fact"],
            "store": teach_B["store"],
            "rho_l2": teach_B["rho_l2"],
        },
        "C_teach_store_disabled": {
            "has_fact": teach_C["has_fact"],
            "store": teach_C["store"],
            "rho_l2": teach_C["rho_l2"],
            "writes_blocked": C.store._writes_blocked,
            "n_forced_steps": teach_C["n_forced_steps"],
        },
        "A_has_inspectable_fact": teach_A["has_fact"],
        "A_rho_l2_after_exp": teach_A["rho_l2"],
        "A_probe_before_rho_reset": probe_A_before,
        "A_probe_after_rho_reset": probe_A_after,
        "A_probe_after_rho_restore": probe_A_restored,
        "B_probe_after_rho_reset": probe_B_after,
        "disable_S_probe_before_rho_reset": probe_C_before,
        "disable_S_probe_after_rho_reset": probe_C_after,
        "reset_S_probe_after_rho_reset": probe_reset_S,
        "twin_rho_distance": twin_dist,
        "rho_restore_action_match": probe_A_before["action"] == probe_A_restored["action"],
    }

    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)

    # Persist
    A.store.dump(run_dir / "store_A.json")
    B.store.dump(run_dir / "store_B.json")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    summary = f"""# v0 key/door three-memory

Classification: **{label}**

{rationale}

## Checklist

| Check | Result |
|-------|--------|
| Weights unchanged | {weights_ok} |
| A has inspectable fact in S | {teach_A["has_fact"]} |
| A correct before ρ reset | {probe_A_before["correct"]} ({probe_A_before["action_name"]}) |
| A correct after ρ reset (S kept) | {probe_A_after["correct"]} ({probe_A_after["action_name"]}) |
| A after ρ restore matches pre-reset action | {metrics["rho_restore_action_match"]} |
| B correct after ρ reset | {probe_B_after["correct"]} ({probe_B_after["action_name"]}) |
| disable-S correct after ρ reset | {probe_C_after["correct"]} ({probe_C_after["action_name"]}) |
| reset S then probe | {probe_reset_S["correct"]} |
| Twin ρ L2 | {twin_dist["l2"]} |

## Store A (plain text)

```json
{json.dumps(teach_A["store"], indent=2)}
```

## Store B

```json
{json.dumps(teach_B["store"], indent=2)}
```
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v0 three-memory key/door experiment")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    metrics = run_v0(seed=args.seed)
    print(json.dumps({k: metrics[k] for k in ("classification", "rationale", "run_dir")}, indent=2))
    print("A after ρ reset correct:", metrics["A_probe_after_rho_reset"]["correct"])
    print("disable-S after ρ reset correct:", metrics["disable_S_probe_after_rho_reset"]["correct"])
    print("B after ρ reset correct:", metrics["B_probe_after_rho_reset"]["correct"])


if __name__ == "__main__":
    main()
