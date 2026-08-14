"""TM.0.7.1: file-local bind→did. No global hapax lexicon.

Same retry-novel English store as TM.0.7.0. Alias a page word to a motor only
from the retrieved note, not from every stamp in S. Dirty S may keep clutter.
Retrieve must use bind=push / bind=adjust. Not a p98 ranker, not unique-pair,
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
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm069 import wiki_prose
from experiments.run_tm070 import (
    classify_a as _classify_a070,
    classify_b as _classify_b070,
    classify_common as _classify_common070,
    make as _make070,
    run_arm as _run_arm070,
    _w_flags as _w_flags070,
)
from three_memory.tag_store import write_prose_notes

_HAPAX = (
    "xenon",
    "neon",
    "krypton",
    "radon",
    "lithium",
    "cesium",
    "nickel",
    "cobalt",
    "quartz",
    "argon",
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm071"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_local_alias: bool = True, **kwargs):
    if kwargs.get("use_event_annotate") is False:
        use_local_alias = False
    return _make070(*args, use_local_alias=use_local_alias, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags070(w_files, w_dir)
    flags["use_local_alias"] = True
    flags["local_alias"] = True
    flags["use_retry_novel"] = True
    flags["retry_novel"] = True
    flags["english_life"] = True
    return flags


def _hide_local(m: dict[str, Any]) -> Any:
    saved = m.get("use_local_alias")
    m["use_local_alias"] = False
    return saved


def _used_bind(m: dict[str, Any], key: str) -> str:
    pol = (m.get(key) or {}).get("policy") or {}
    return str(pol.get("used_bind") or "").lower()


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide_local(m)
    early = _classify_common070(m)
    m["use_local_alias"] = saved
    if early:
        return early
    if not saved:
        return "Fail", "Local-alias was frozen off."
    if not m.get("local_alias"):
        return "Fail", "Local-alias was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide_local(m)
    label, why = _classify_a070(m)
    m["use_local_alias"] = saved
    if not saved:
        return "Fail", "Local-alias was frozen off."
    if not m.get("local_alias"):
        return "Fail", "Local-alias was frozen off."
    ub_a = _used_bind(m, "train_s_probe")
    ub_c = _used_bind(m, "both_after_c")
    m["used_bind_a"] = ub_a
    m["used_bind_c"] = ub_c
    if ub_a in _HAPAX:
        return "Fail", f"Retrieve used clutter hapax {ub_a}."
    if ub_c in _HAPAX:
        return "Fail", f"C retrieve used clutter hapax {ub_c}."
    if ub_a != "push":
        return "Fail", "Retrieve did not use bind=push."
    if ub_c and ub_c != "adjust":
        return "Fail", "C retrieve did not use bind=adjust."
    if label == "Fail" and "Clutter hapax" in why:
        return (
            "Store-works",
            "Local-alias: dirty S may keep clutter; retrieve uses bind=push then adjust. Cortex frozen.",
        )
    if label == "Store-works":
        return (
            "Store-works",
            "Local-alias: retrieve uses the retrieved note's bind only; train PRESS from push; C TUNE from adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide_local(m)
    label, why = _classify_b070(m)
    m["use_local_alias"] = saved
    if not saved:
        return "Fail", "Local-alias was frozen off."
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return local-alias; dirty store motors; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm070 as tm070

    saved = (tm070.make, tm070.classify_a, tm070.classify_b, tm070._w_flags)
    tm070.make = make
    tm070.classify_a = classify_a
    tm070.classify_b = classify_b
    tm070._w_flags = _w_flags
    try:
        m = _run_arm070(**kwargs)
    finally:
        tm070.make, tm070.classify_a, tm070.classify_b, tm070._w_flags = saved
    m["use_local_alias"] = True
    m["local_alias"] = True
    label, why = (classify_a if kwargs.get("split") else classify_b)(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm071(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.7.1",
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
        f"""# TM.0.7.1 A local-alias vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split local-alias | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| Untrained A / foil C | {a['untrained_probe']['action_name']} / {a['untrained_foil_c']['action_name']} | {b['untrained_probe']['action_name']} / {b['untrained_foil_c']['action_name']} |
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.7.1 file-local bind to did")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm071(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "used_a": m["A"].get("used_bind_a"),
                "used_c": m["A"].get("used_bind_c"),
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
