"""TM.0.5.9: correct the dirty store — stop appending, drop junk.

Same scaled multi-rare never-wipe W as TM.0.5.8. Once S names here, do not
commit more W pages. On fail, drop S files that do not name an act.
Not English, not shared-return rescue, not domain= drop, not dropping has_code.
"""

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

from experiments.run_tm010 import _head_fp
from experiments.run_tm040 import probe
from experiments.run_tm052 import live_free
from experiments.run_tm054 import make as _make054
from experiments.run_tm055 import _live_extra
from experiments.run_tm057 import _s_snapshot
from experiments.run_tm056 import _c_life_on_s as _c_life_on_s056
from experiments.run_tm056 import _copy_s, _probe_s
from experiments.run_tm058 import (
    N_CLUTTER,
    classify_a as _classify_a058,
    classify_b as _classify_b058,
    classify_common as _classify_c058,
    clutter_prose,
    run_arm as _run_arm058,
    wiki_prose,
)
from experiments.run_tm058 import _w_flags as _w_flags058
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes

# Predeclared: dirty 0.5.8 train S had 19 files. Corrected store must be smaller.
MAX_TRAIN_S_FILES = 8


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm059"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(
    *args,
    use_commit_rare_only: bool = True,
    use_revise_head: bool = True,
    use_commit_here_only: bool = True,
    **kwargs,
):
    return _make054(
        *args,
        use_commit_rare_only=use_commit_rare_only,
        use_revise_head=use_revise_head,
        use_commit_here_only=use_commit_here_only,
        **kwargs,
    )


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags058(w_files, w_dir)
    flags["correct_dirty_s"] = True
    flags["use_commit_here_only"] = True
    flags["use_revise_head"] = True
    flags["max_train_s_files"] = MAX_TRAIN_S_FILES
    return flags


def _train_keep(
    policy: UsePolicy,
    w_dir: Path,
    work: Path,
    n: int,
    seed: int,
    *,
    split: bool,
    max_steps: int,
) -> tuple[list[float], Path]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    b_f = b_m = b_u = b_r = 0.0
    n_revised = 0
    revise0 = _head_fp(policy, "revise")
    s_dir = work / "ep"
    s_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(n):
        eps = 0.45 * (1.0 - ep / max(n, 1)) + 0.05
        explore_eps = 0.55 * (1.0 - 0.4 * ep / max(n, 1))
        snap = _s_snapshot(s_dir)
        ag = make(s_dir, w_dir, policy, epsilon=eps, explore_epsilon=explore_eps, rng=rng)
        ag.policy_traces = []
        ag.reset_rho()
        live = _live_extra(live_free(ag, "experience_channel_a", seed + 10, max_steps=max_steps))
        n_revised += int(ag.n_revised)
        tr_life = list(ag.policy_traces)
        r_find = 1.0 if any(t.get("kind") == "search" and t.get("has_rare") for t in tr_life) else 0.0
        wrote = any(t.get("kind") == "write" and t.get("write") for t in tr_life)
        r_mark = (
            1.0
            if live["n_annotated"] > 0
            or (snap["found_press"] and snap["found_cha"] and wrote)
            else 0.0
        )
        ag.world = None
        ag.reset_rho()
        p = probe(ag, "probe_channel_a", seed + 10)
        r_use = 1.0 if p["correct"] else 0.0
        tr = ag.policy_traces
        if split:
            b_f = 0.9 * b_f + 0.1 * r_find
            b_m = 0.9 * b_m + 0.1 * r_mark
            b_u = 0.9 * b_u + 0.1 * r_use
            b_r = 0.9 * b_r + 0.1 * r_use
            policy.update([t for t in tr if t.get("kind") == "search"], r_find - b_f)
            policy.update([t for t in tr if t.get("kind") == "write"], r_mark - b_m)
            policy.update([t for t in tr if t.get("kind") == "vname"], r_use - b_u)
            policy.update([t for t in tr if t.get("kind") == "revise"], r_use - b_r)
        else:
            b_u = 0.9 * b_u + 0.1 * r_use
            adv = r_use - b_u
            policy.update([t for t in tr if t.get("kind") in ("search", "write", "revise")], adv)
            policy.update([t for t in tr if t.get("kind") == "vname"], adv)
        rewards.append(r_use)
    (work / "tm059_train.json").write_text(
        json.dumps(
            {
                "n_revised": n_revised,
                "revise_changed": revise0 != _head_fp(policy, "revise"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rewards, s_dir


def _c_life_on_s(*args, **kwargs):
    import experiments.run_tm056 as tm056

    saved = tm056.make
    tm056.make = make
    try:
        return _c_life_on_s056(*args, **kwargs)
    finally:
        tm056.make = saved


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    early = _classify_c058(m)
    if early:
        return early
    if not m.get("correct_dirty_s"):
        return "Fail", "Correct-dirty-S was frozen off."
    if not m.get("use_revise_head"):
        return "Fail", "Revise head was frozen off."
    if not m.get("use_commit_here_only"):
        return "Fail", "Commit-here-only was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    label, why = _classify_a058(m)
    if label != "Store-works":
        return label, why
    if (m.get("n_revised_train") or 0) < 1:
        return "Fail", "Never-wipe train never revised S."
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Train S is still stamp-collecting clutter."
    if not m.get("revise_changed"):
        return "Fail", "Revise head did not move."
    return (
        "Store-works",
        "Corrected dirty S: never-wipe train still PRESS with a small store; C life A PRESS, C TUNE. Cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    return _classify_b058(m)


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm056 as tm056
    import experiments.run_tm057 as tm057
    import experiments.run_tm058 as tm058

    saved = (
        tm056.make,
        tm056._train_keep,
        tm057.classify_a,
        tm057.classify_b,
        tm057._w_flags,
        tm058.classify_a,
        tm058.classify_b,
        tm058._w_flags,
    )
    tm056.make = make
    tm056._train_keep = _train_keep
    tm057.classify_a = classify_a
    tm057.classify_b = classify_b
    tm057._w_flags = _w_flags
    tm058.classify_a = classify_a
    tm058.classify_b = classify_b
    tm058._w_flags = _w_flags
    try:
        m = _run_arm058(**kwargs)
    finally:
        (
            tm056.make,
            tm056._train_keep,
            tm057.classify_a,
            tm057.classify_b,
            tm057._w_flags,
            tm058.classify_a,
            tm058.classify_b,
            tm058._w_flags,
        ) = saved
    side = Path(kwargs["run_dir"]) / f"{kwargs['arm']}_train" / "tm059_train.json"
    extra = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    m["n_revised_train"] = int(extra.get("n_revised") or 0)
    m["revise_changed"] = bool(extra.get("revise_changed"))
    m["correct_dirty_s"] = True
    m["use_revise_head"] = True
    m["use_commit_here_only"] = True
    label, why = (classify_a if kwargs.get("split") else classify_b)(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm059(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_prose(hapax=False))
    w_files = sorted(p.name for p in w_both.glob("*.md"))
    a = run_arm(
        arm="A",
        split=True,
        run_dir=run_dir,
        w_a=w_a,
        w_both=w_both,
        w_clutter=w_clutter,
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
        w_both=w_both,
        w_clutter=w_clutter,
        w_files=w_files,
        seed=seed,
        n_train=n_train,
        train_seed=seed + 5,
        max_steps=max_steps,
    )
    out = {
        "version": "TM.0.5.9",
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
        f"""# TM.0.5.9 A correct dirty S vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split corrected S | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Wipe-between: A / C | {a['wipe_ctrl_a']['action_name']} / {a['wipe_ctrl_c']['action_name']} | {b['wipe_ctrl_a']['action_name']} / {b['wipe_ctrl_c']['action_name']} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
| n_revised train | {a.get('n_revised_train')} | {b.get('n_revised_train')} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.9 correct the dirty store")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm059(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {"A": m["A"]["classification"], "B": m["B"]["classification"], "world": m["world"], "run_dir": m["run_dir"]},
            indent=2,
        )
    )
    print(
        "A",
        "trainS",
        m["A"]["train_s_probe"]["action_name"],
        m["A"]["train_s_foil"]["action_name"],
        "n",
        m["A"]["train_s"]["n"],
        "rev",
        m["A"].get("n_revised_train"),
        "dirtyC",
        m["A"]["both_after_a"]["action_name"],
        m["A"]["both_after_c"]["action_name"],
    )
    print(
        "B",
        m["B"]["train_s_probe"]["action_name"],
        m["B"]["both_after_c"]["action_name"],
        "n",
        m["B"]["train_s"]["n"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
