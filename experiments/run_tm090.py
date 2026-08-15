"""TM.0.9.0: first math life — count unread rares, not + in cortex.

Same one-machine one-return English store as TM.0.8.2. Search may use the
cardinality of tokens S still lacks. Genome may count a stream. It may not
add. Not a p98 ranker, not unique-pair, not restoring domain=, not raising
n_train.
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
from experiments.run_tm080 import wiki_prose
from experiments.run_tm082 import (
    classify_a as _classify_a082,
    classify_b as _classify_b082,
    classify_common as _classify_common082,
    make as _make082,
    run_arm as _run_arm082,
    _w_flags as _w_flags082,
)
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm090"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_count_search: bool = True, **kwargs):
    if kwargs.get("use_search_head") is False:
        use_count_search = False
    return _make082(*args, use_count_search=use_count_search, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags082(w_files, w_dir)
    flags["use_count_search"] = True
    flags["count_search"] = True
    flags["math_life"] = True
    flags["no_domain_switch"] = True
    flags["one_return_recipe"] = True
    flags["english_life"] = True
    return flags


def _hide(m: dict[str, Any]) -> Any:
    saved = m.get("use_count_search")
    m["use_count_search"] = False
    return saved


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide(m)
    early = _classify_common082(m)
    m["use_count_search"] = saved
    if early:
        return early
    if m.get("use_hyp_survive"):
        return "Confound", "Hyp-survive was smuggled onto this slice."
    if not saved or not m.get("count_search"):
        return "Fail", "Count-search was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_a082(m)
    m["use_count_search"] = saved
    if m.get("use_hyp_survive"):
        return "Confound", "Hyp-survive was smuggled onto this slice."
    if not saved or not m.get("count_search"):
        return "Fail", "Count-search was frozen off."
    if label == "Store-works":
        return (
            "Store-works",
            "Math life: search uses the count of unread rares, not +; retrieve uses push then adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_b082(m)
    m["use_count_search"] = saved
    if m.get("use_hyp_survive"):
        return "Confound", "Hyp-survive was smuggled onto this slice."
    if not saved:
        return "Fail", "Count-search was frozen off."
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Count-search motor bar; small store; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm082 as tm082

    saved = (tm082.make, tm082.classify_a, tm082.classify_b, tm082._w_flags)
    tm082.make = make
    tm082.classify_a = classify_a
    tm082.classify_b = classify_b
    tm082._w_flags = _w_flags
    try:
        m = _run_arm082(**kwargs)
    finally:
        tm082.make, tm082.classify_a, tm082.classify_b, tm082._w_flags = saved
    m["use_count_search"] = True
    m["count_search"] = True
    m["math_life"] = True
    label, why = classify_a(m) if kwargs.get("arm") == "A" else classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm090(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.9.0",
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
        f"""# TM.0.9.0 A count-search math life vs B motor bar

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A count-search | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B motor bar | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.9.0 first math life: count unread rares")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm090(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
