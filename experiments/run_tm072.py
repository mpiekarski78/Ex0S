"""TM.0.7.2: keep the note that steered; drop other same-here stamps.

Same local-alias English store as TM.0.7.1. After a successful act, S keeps the
retrieved/in-hand note at this station and drops the rest. Not a p98 ranker,
not unique-pair, not leftover W walk, not math, not dropping has_code or
domain="dial", not solving B.
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
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm069 import wiki_prose
from experiments.run_tm071 import (
    classify_a as _classify_a071,
    classify_b as _classify_b071,
    classify_common as _classify_common071,
    make as _make071,
    run_arm as _run_arm071,
    _w_flags as _w_flags071,
)
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm072"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_keep_steerer: bool = True, **kwargs):
    if kwargs.get("use_here_match") is False:
        use_keep_steerer = False
    return _make071(*args, use_keep_steerer=use_keep_steerer, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags071(w_files, w_dir)
    flags["use_keep_steerer"] = True
    flags["keep_steerer"] = True
    flags["use_local_alias"] = True
    flags["local_alias"] = True
    flags["english_life"] = True
    return flags


def _hide(m: dict[str, Any]) -> Any:
    saved = m.get("use_keep_steerer")
    m["use_keep_steerer"] = False
    return saved


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide(m)
    early = _classify_common071(m)
    m["use_keep_steerer"] = saved
    if early:
        return early
    if not saved:
        return "Fail", "Keep-steerer was frozen off."
    if not m.get("keep_steerer"):
        return "Fail", "Keep-steerer was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_a071(m)
    m["use_keep_steerer"] = saved
    if not saved:
        return "Fail", "Keep-steerer was frozen off."
    if not m.get("keep_steerer"):
        return "Fail", "Keep-steerer was frozen off."
    if label == "Store-works":
        return (
            "Store-works",
            "Keep-steerer: after success, other same-here notes are dropped; retrieve uses push then adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_b071(m)
    m["use_keep_steerer"] = saved
    if not saved:
        return "Fail", "Keep-steerer was frozen off."
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return keep-steerer; small store; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm071 as tm071

    saved = (tm071.make, tm071.classify_a, tm071.classify_b, tm071._w_flags)
    tm071.make = make
    tm071.classify_a = classify_a
    tm071.classify_b = classify_b
    tm071._w_flags = _w_flags
    try:
        m = _run_arm071(**kwargs)
    finally:
        tm071.make, tm071.classify_a, tm071.classify_b, tm071._w_flags = saved
    m["use_keep_steerer"] = True
    m["keep_steerer"] = True
    label, why = (classify_a if kwargs.get("split") else classify_b)(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm072(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_closed())
    w_files = sorted(p.name for p in w_both.glob("*.md"))
    a = run_arm(
        arm="A", split=True, run_dir=run_dir, w_a=w_a, w_both=w_both, w_clutter=w_clutter,
        w_files=w_files, seed=seed, n_train=n_train, train_seed=seed, max_steps=max_steps,
    )
    b = run_arm(
        arm="B", split=False, run_dir=run_dir, w_a=w_a, w_both=w_both, w_clutter=w_clutter,
        w_files=w_files, seed=seed, n_train=n_train, train_seed=seed + 5, max_steps=max_steps,
    )
    out = {
        "version": "TM.0.7.2",
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
        f"""# TM.0.7.2 A keep-steerer vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split keep-steerer | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
| n_revised train | {a.get('n_revised_train', 0)} | {b.get('n_revised_train', 0)} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.7.2 keep the note that steered")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm072(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "n": m["A"]["train_s"]["n"],
                "used_a": m["A"].get("used_bind_a"),
                "used_c": m["A"].get("used_bind_c"),
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
