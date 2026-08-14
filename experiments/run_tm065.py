"""TM.0.6.5: concurrent bind — stamp the page in play, block extra hapax here.

Same English multi-rare W as TM.0.6.4. Genome: bind the attended page when the
body succeeds, and do not add a second CS at the same station. Not a ranker,
not unique-rare restored, not math, not dropping has_code or domain="dial".
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
from experiments.run_tm062 import run_arm as _run_arm062
from experiments.run_tm064 import (
    classify_a as _classify_a064,
    classify_b as _classify_b064,
    classify_common as _classify_common064,
    make as _make064,
    wiki_prose,
    _w_flags as _w_flags064,
)
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm065"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_block_here: bool = True, **kwargs):
    if kwargs.get("use_here_match") is False or kwargs.get("use_event_annotate") is False:
        use_block_here = False
    return _make064(*args, use_block_here=use_block_here, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags064(w_files, w_dir)
    flags["use_block_here"] = True
    flags["find_without_unique_rare"] = True
    flags["english_life"] = True
    flags["use_stamp_new_here"] = True
    return flags


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = m.get("use_block_here")
    m["use_block_here"] = False
    early = _classify_common064(m)
    m["use_block_here"] = saved
    if early:
        return early
    if not saved:
        return "Fail", "Concurrent bind/block was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = m.get("use_block_here")
    m["use_block_here"] = False
    label, why = _classify_a064(m)
    m["use_block_here"] = saved
    if not saved:
        return "Fail", "Concurrent bind/block was frozen off."
    if label == "Store-works":
        return (
            "Store-works",
            "Concurrent bind: multi-rare English W, one CS here from the page in play; train S PRESS from push; C life A PRESS, C TUNE. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = m.get("use_block_here")
    m["use_block_here"] = False
    label, why = _classify_b064(m)
    m["use_block_here"] = saved
    if not saved:
        return "Fail", "Concurrent bind/block was frozen off."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return concurrent bind; dirty English S two facts; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm062 as tm062

    saved = (tm062.make, tm062.classify_a, tm062.classify_b, tm062._w_flags)
    tm062.make = make
    tm062.classify_a = classify_a
    tm062.classify_b = classify_b
    tm062._w_flags = _w_flags
    try:
        return _run_arm062(**kwargs)
    finally:
        tm062.make, tm062.classify_a, tm062.classify_b, tm062._w_flags = saved


def run_tm065(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.6.5",
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
        f"""# TM.0.6.5 A concurrent bind vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split concurrent bind | **{a['classification']}** | {a['train_return_last50']:.2f} |
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
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.6.5 concurrent bind / block extra hapax here")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm065(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
        m["A"]["a_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["train_s_probe"]["action_name"],
        m["B"]["both_after_c"]["action_name"],
        "last50",
        m["B"]["train_return_last50"],
    )


if __name__ == "__main__":
    main()
