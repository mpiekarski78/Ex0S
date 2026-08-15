"""TM.0.9.2: antecedent MATCH.

A stored X→action may steer only when X is in the current observation.
Genome exposes bind_present_in_current_stream (bool). No token ids.
Historical BOX is not rewritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm040 import probe
from experiments.run_tm052 import live_free
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm091 import (
    classify_b as _classify_b091,
    make as _make091,
    run_arm as _run_arm091,
    _w_flags as _w_flags091,
)
from three_memory.policy import UsePolicy
from three_memory.symbols import record_to_tagfile
from three_memory.tag_store import write_prose_notes

_NONCES = ("flim", "zorg", "blen", "nork", "quop", "daff", "mib", "vex")
_MOTORS = ("press", "tune")


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm092"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_bind_match: bool = True, **kwargs):
    if kwargs.get("use_here_match") is False or kwargs.get("use_event_annotate") is False:
        use_bind_match = False
    if kwargs.get("use_alias_bind") is False:
        use_bind_match = False
    return _make091(*args, use_bind_match=use_bind_match, **kwargs)


def permute_pair(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x, y = (str(t) for t in rng.choice(_NONCES, size=2, replace=False))
    m1, m2 = (str(t) for t in rng.permutation(_MOTORS))
    names = [f"n{int(i):02d}" for i in rng.choice(40, size=2, replace=False)]
    return {"x": x, "y": y, "m1": m1, "m2": m2, "fx": names[0], "fy": names[1], "seed": seed}


def write_relation_s(dest: Path, rows: list[tuple[str, str, str]]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for fact_id, bind, did in rows:
        tags = {"bind": bind, "did": did, "here": "chb", "w0": bind}
        (dest / f"{fact_id}.tag").write_text(record_to_tagfile(fact_id, tags), encoding="utf-8")


def probe_match(
    policy: UsePolicy,
    s_dir: Path | None,
    seed: int,
    *,
    cue: str | None,
    force_use: bool = False,
) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm092_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make(store, None, policy, explore_epsilon=0.0, force_use=force_use)
        ag.reset_rho()
        out = probe(ag, "probe_channel_b", seed, tokens=tokens)
        out["cue"] = cue
        out["bind_present"] = bool((out.get("policy") or {}).get("bind_present_in_current_stream"))
        return out


def _motor(name: str) -> str:
    return str(name or "hold").lower()


def classify_match_battery(cells: dict[str, dict[str, Any]], spec: dict[str, Any]) -> tuple[str, str]:
    x, y, m1, m2 = spec["x"], spec["y"], spec["m1"], spec["m2"]
    same_x = _motor(cells["same_x"]["action_name"])
    same_y = _motor(cells["same_y"]["action_name"])
    cross_x = _motor(cells["cross_x"]["action_name"])
    cross_y = _motor(cells["cross_y"]["action_name"])
    empty = _motor(cells["empty"]["action_name"])
    if empty not in ("hold",):
        return "Fail", f"Empty S with cue {x} was {empty}, not HOLD."
    if same_x != m1:
        return "Fail", f"Cue {x} + {x}→{m1}/{y}→{m2} was {same_x}, not {m1}."
    if same_y != m2:
        return "Fail", f"Cue {y} + same S was {same_y}, not {m2}."
    if cross_x != "hold":
        return "Fail", f"Cue {x} + only {y}→{m2} was {cross_x}, not HOLD."
    if cross_y != "hold":
        return "Fail", f"Cue {y} + only {x}→{m1} was {cross_y}, not HOLD."
    return "Store-works", "Same-S cue switch fires the matching motor; crossed single-relation cues HOLD."


def run_match_battery(
    policy: UsePolicy,
    spec: dict[str, Any],
    dest: Path,
    *,
    force_use: bool = False,
) -> dict[str, Any]:
    x, y, m1, m2 = spec["x"], spec["y"], spec["m1"], spec["m2"]
    fx, fy = spec["fx"], spec["fy"]
    both = dest / "both"
    only_y = dest / "only_y"
    only_x = dest / "only_x"
    write_relation_s(both, [(fx, x, m1), (fy, y, m2)])
    write_relation_s(only_y, [(fy, y, m2)])
    write_relation_s(only_x, [(fx, x, m1)])
    seed = int(spec["seed"])
    cells = {
        "same_x": probe_match(policy, both, seed + 1, cue=x, force_use=force_use),
        "same_y": probe_match(policy, both, seed + 2, cue=y, force_use=force_use),
        "cross_x": probe_match(policy, only_y, seed + 3, cue=x, force_use=force_use),
        "cross_y": probe_match(policy, only_x, seed + 4, cue=y, force_use=force_use),
        "empty": probe_match(policy, None, seed + 5, cue=x, force_use=force_use),
    }
    label, why = classify_match_battery(cells, spec)
    return {"spec": spec, "cells": cells, "classification": label, "rationale": why}


def _useful_body(token: str, distractor: str) -> str:
    return (
        f"# page\n\n"
        f"{token.capitalize()} the fixture on the shelf. {distractor.capitalize()} in the bin.\n"
    )


def _train_cue(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    *,
    token: str,
    max_steps: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_u = 0.0
    s_dir = work / "ep"
    s_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        ag = make(s_dir, w_dir, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps)
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_channel_a", seed + 10, tokens=frozenset({token}))
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        b_u = 0.9 * b_u + 0.1 * r_use
        adv = r_use - b_u
        policy.update([t for t in tr if t.get("kind") in ("search", "write")], adv)
        policy.update([t for t in tr if t.get("kind") == "vname"], adv)
        rewards.append(r_use)
    return rewards


def train_match_policy(
    *,
    seed: int,
    run_dir: Path,
    n_train: int,
    max_steps: int,
) -> tuple[UsePolicy, dict[str, Any]]:
    spec = permute_pair(seed)
    w = run_dir / f"W_train_{seed}"
    work = run_dir / f"train_{seed}"
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)
    write_prose_notes(
        w,
        [
            ("p99.md", _useful_body(spec["x"], spec["y"])),
            ("p98.md", _useful_body(spec["y"], spec["x"])),
        ],
    )
    policy = UsePolicy(seed=7, lr=0.2)
    rewards = _train_cue(
        policy, w, work, n_train, seed, token=spec["x"], max_steps=max_steps
    )
    return policy, {"spec": spec, "rewards": rewards, "last50": float(np.mean(rewards[-50:]))}


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if m.get("use_bind_match") is False:
        return "Fail", "Bind-match was frozen off."
    if not m.get("bind_match"):
        return "Fail", "Bind-match was frozen off."
    label = m.get("match_classification") or "Fail"
    why = m.get("match_rationale") or "MATCH battery missing."
    return str(label), str(why)


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    if m.get("use_bind_match") is False:
        return "Fail", "Bind-match was frozen off."
    saved = m.get("use_bind_match")
    m["use_bind_match"] = False
    label, why = _classify_b091(m)
    m["use_bind_match"] = saved
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    return label, why


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags091(w_files, w_dir)
    flags["use_bind_match"] = True
    flags["bind_match"] = True
    return flags


def run_arm_b(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm091 as tm091

    saved = (tm091.make, tm091.classify_b, tm091._w_flags)
    tm091.make = make
    tm091.classify_b = classify_b
    tm091._w_flags = _w_flags
    try:
        m = _run_arm091(**kwargs)
    finally:
        tm091.make, tm091.classify_b, tm091._w_flags = saved
    m["use_bind_match"] = True
    m["bind_match"] = True
    label, why = classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm092(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    policy, train_info = train_match_policy(
        seed=seed, run_dir=run_dir, n_train=n_train, max_steps=max_steps
    )
    match_seeds = [seed, seed + 1, seed + 2]
    batteries = []
    for s in match_seeds:
        spec = permute_pair(s)
        bat = run_match_battery(policy, spec, run_dir / f"match_{s}", force_use=False)
        batteries.append(bat)
    gate = run_match_battery(
        UsePolicy(seed=7, lr=0.2), permute_pair(seed + 9), run_dir / "match_force", force_use=True
    )
    a_ok = all(b["classification"] == "Store-works" for b in batteries) and gate["classification"] == "Store-works"
    a = {
        "use_bind_match": True,
        "bind_match": True,
        "match_classification": "Store-works" if a_ok else "Fail",
        "match_rationale": (
            "MATCH battery Pass on trained policy (3 permuted seeds) and force-use gate."
            if a_ok
            else "MATCH battery miss: " + "; ".join(b["rationale"] for b in batteries + [gate])
        ),
        "train": train_info,
        "batteries": batteries,
        "force_use_gate": gate,
        "cortex_hash": make(run_dir / "hash", None, policy, enabled=False).weight_hash(),
    }
    a["classification"], a["rationale"] = classify_a(a)

    from experiments.run_tm054 import clutter_prose as clutter_closed
    from experiments.run_tm080 import wiki_prose

    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_closed())
    w_files = sorted(p.name for p in w_both.glob("*.md"))
    b = run_arm_b(
        arm="B",
        split=False,
        run_dir=run_dir,
        w_a=w_a,
        w_both=w_both,
        w_clutter=w_clutter,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
        max_steps=max_steps,
    )
    out = {
        "version": "TM.0.9.2",
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
        f"""# TM.0.9.2 A antecedent MATCH vs B motor bar

| Arm | Classification |
|-----|----------------|
| A MATCH | **{a['classification']}** |
| B motor bar | **{b['classification']}** |

A: {a['rationale']}

B: {b['rationale']}
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.9.2 antecedent MATCH")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm092(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "n": m["B"]["train_s"]["n"],
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
