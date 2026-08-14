"""TM.0.6.8: find the unread page that adds the most rare tokens S lacks.

Same in-hand new-here English store as TM.0.6.7. Search keeps pages whose rare
words are most novel vs committed S. Not a p98 ranker, not unique-rare restored,
not leftover W walk, not math, not dropping has_code or domain="dial", not solving B.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import clutter_prose as clutter_closed
from experiments.run_tm064 import wiki_prose
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm067 import (
    classify_a as _classify_a067,
    classify_b as _classify_b067,
    classify_common as _classify_common067,
    make as _make067,
    run_arm as _run_arm067,
    _w_flags as _w_flags067,
)
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm068"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_find_novel: bool = True, **kwargs):
    if kwargs.get("use_search_head") is False:
        use_find_novel = False
    return _make067(*args, use_find_novel=use_find_novel, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags067(w_files, w_dir)
    flags["use_find_novel"] = True
    flags["find_novel"] = True
    flags["use_in_hand_new_here"] = True
    flags["in_hand_new_here"] = True
    flags["use_revise_head"] = True
    flags["use_commit_here_only"] = True
    flags["correct_dirty_s"] = True
    flags["use_block_here"] = True
    flags["use_stamp_new_here"] = True
    flags["find_without_unique_rare"] = True
    flags["english_life"] = True
    return flags


def _hide_find(m: dict[str, Any]) -> Any:
    saved = m.get("use_find_novel")
    m["use_find_novel"] = False
    return saved


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide_find(m)
    early = _classify_common067(m)
    m["use_find_novel"] = saved
    if early:
        return early
    if not saved:
        return "Fail", "Find-novel was frozen off."
    if not m.get("find_novel"):
        return "Fail", "Find-novel was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide_find(m)
    label, why = _classify_a067(m)
    m["use_find_novel"] = saved
    if not saved:
        return "Fail", "Find-novel was frozen off."
    if not m.get("find_novel"):
        return "Fail", "Find-novel was frozen off."
    if label == "Store-works":
        return (
            "Store-works",
            "Find-novel: search keeps the unread page with the most rare tokens S lacks; train S PRESS from push; C life A PRESS, C TUNE from adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide_find(m)
    label, why = _classify_b067(m)
    m["use_find_novel"] = saved
    if not saved:
        return "Fail", "Find-novel was frozen off."
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return find-novel; small store two facts; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm067 as tm067

    saved = (tm067.make, tm067.classify_a, tm067.classify_b, tm067._w_flags)
    tm067.make = make
    tm067.classify_a = classify_a
    tm067.classify_b = classify_b
    tm067._w_flags = _w_flags
    try:
        m = _run_arm067(**kwargs)
    finally:
        tm067.make, tm067.classify_a, tm067.classify_b, tm067._w_flags = saved
    m["use_find_novel"] = True
    m["find_novel"] = True
    label, why = (classify_a if kwargs.get("split") else classify_b)(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm068(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_closed())
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
        "version": "TM.0.6.8",
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
        f"""# TM.0.6.8 A find-novel vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split find-novel | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Wipe-between: A / C | {a['wipe_ctrl_a']['action_name']} / {a['wipe_ctrl_c']['action_name']} | {b['wipe_ctrl_a']['action_name']} / {b['wipe_ctrl_c']['action_name']} |
| Bind-off A | {a['bind_control']['action_name']} | {b['bind_control']['action_name']} |
| Nonce-only A | {a['nonce_control']['action_name']} | {b['nonce_control']['action_name']} |
| Bind-all nonce A | {a['bindall_nonce']['action_name']} | {b['bindall_nonce']['action_name']} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
| n_revised train | {a.get('n_revised_train', 0)} | {b.get('n_revised_train', 0)} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.6.8 find unread pages with the most novel rares vs S")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm068(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
        "dirtyC",
        m["A"]["both_after_a"]["action_name"],
        m["A"]["both_after_c"]["action_name"],
        "n",
        m["A"]["train_s"]["n"],
        "rev",
        m["A"].get("n_revised_train"),
        m["A"]["a_tag"].strip().replace("\n", " | "),
        "both",
        m["A"]["both_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["train_s_probe"]["action_name"],
        m["B"]["both_after_c"]["action_name"],
        "n",
        m["B"]["train_s"]["n"],
        "last50",
        m["B"]["train_return_last50"],
        m["B"]["a_tag"].strip().replace("\n", " | "),
        "both",
        m["B"]["both_tag"].strip().replace("\n", " | "),
    )


if __name__ == "__main__":
    main()
