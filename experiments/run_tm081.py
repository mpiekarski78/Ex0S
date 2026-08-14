"""TM.0.8.1: one shared return is the A recipe.

Same 64-page English keep-steerer store as TM.0.8.0. A no longer gets split
find/mark/use credit. One signal: the probe worked after the life. Child
connects look with it worked. Not a p98 ranker, not unique-pair, not math,
not dropping has_code or domain="dial", not raising n_train.
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
from experiments.run_tm080 import (
    classify_a as _classify_a080,
    classify_b as _classify_b080,
    classify_common as _classify_common080,
    make as _make080,
    run_arm as _run_arm080,
    wiki_prose,
    _w_flags as _w_flags080,
)
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm081"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, **kwargs):
    return _make080(*args, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags080(w_files, w_dir)
    flags["one_return_recipe"] = True
    flags["w_scale"] = True
    flags["scale_english_w"] = True
    flags["english_life"] = True
    return flags


def _hide(m: dict[str, Any], *, as_split_a: bool = False) -> dict[str, Any]:
    saved = {"one_return_recipe": m.get("one_return_recipe"), "trained_split": m.get("trained_split")}
    m["one_return_recipe"] = False
    if as_split_a:
        # 080 A's chain expects split; 062 B Fail if split is on.
        m["trained_split"] = True
    return saved


def _restore(m: dict[str, Any], saved: dict[str, Any]) -> None:
    m.update(saved)


def _require_one(m: dict[str, Any]) -> tuple[str, str] | None:
    if m.get("trained_split"):
        return "Confound", "Split credit was restored; one return is the recipe."
    if not m.get("one_return_recipe"):
        return "Fail", "One-return recipe was frozen off."
    return None


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide(m)
    early = _classify_common080(m)
    _restore(m, saved)
    if early:
        return early
    return _require_one(m)


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m, as_split_a=True)
    label, why = _classify_a080(m)
    _restore(m, saved)
    req = _require_one(m)
    if req:
        return req
    if label == "Store-works":
        return (
            "Store-works",
            "One return: look and it-worked share one signal; retrieve uses push then adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_b080(m)
    _restore(m, saved)
    req = _require_one(m)
    if req:
        return req
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "One-return motor bar; small store; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm080 as tm080

    saved = (tm080.make, tm080.classify_a, tm080.classify_b, tm080._w_flags)
    tm080.make = make
    tm080.classify_a = classify_a
    tm080.classify_b = classify_b
    tm080._w_flags = _w_flags
    try:
        m = _run_arm080(**kwargs)
    finally:
        tm080.make, tm080.classify_a, tm080.classify_b, tm080._w_flags = saved
    m["one_return_recipe"] = True
    # A is the jump even when split=False.
    label, why = classify_a(m) if kwargs.get("arm") == "A" else classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm081(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_closed())
    w_files = sorted(p.name for p in w_both.glob("*.md"))
    a = run_arm(
        arm="A", split=False, run_dir=run_dir, w_a=w_a, w_both=w_both, w_clutter=w_clutter,
        w_files=w_files, seed=seed, n_train=n_train, train_seed=seed, max_steps=max_steps,
    )
    b = run_arm(
        arm="B", split=False, run_dir=run_dir, w_a=w_a, w_both=w_both, w_clutter=w_clutter,
        w_files=w_files, seed=seed, n_train=n_train, train_seed=seed + 5, max_steps=max_steps,
    )
    out = {
        "version": "TM.0.8.1",
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
        f"""# TM.0.8.1 A one return vs B one return motor bar

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A one-return recipe | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B one-return motor bar | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| trained_split | {a.get('trained_split')} | {b.get('trained_split')} |
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.8.1 one shared return is the recipe")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm081(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "split_a": m["A"].get("trained_split"),
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
