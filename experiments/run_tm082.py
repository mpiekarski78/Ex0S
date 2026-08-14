"""TM.0.8.2: one machine — body from n_actions / percepts, not domain=.

Same one-return 64-page English keep-steerer store as TM.0.8.1. Motors,
affordances, and station names come from body size and the current percept.
Not a p98 ranker, not unique-pair, not math, not dropping has_code, not
raising n_train.
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
from experiments.run_tm081 import (
    classify_a as _classify_a081,
    classify_b as _classify_b081,
    classify_common as _classify_common081,
    make as _make081,
    run_arm as _run_arm081,
    _w_flags as _w_flags081,
)
from three_memory.tag_store import write_prose_notes

_HIDE = ("no_domain_switch", "domain_switch")


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm082"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, **kwargs):
    return _make081(*args, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags081(w_files, w_dir)
    flags["no_domain_switch"] = True
    flags["domain_switch"] = False
    flags["one_return_recipe"] = True
    flags["english_life"] = True
    return flags


def _hide(m: dict[str, Any]) -> dict[str, Any]:
    saved = {k: m.get(k) for k in _HIDE}
    m["no_domain_switch"] = False
    m["domain_switch"] = True
    return saved


def _restore(m: dict[str, Any], saved: dict[str, Any]) -> None:
    m.update(saved)


def _require_one_machine(m: dict[str, Any]) -> tuple[str, str] | None:
    if m.get("domain_switch") or not m.get("no_domain_switch"):
        return "Fail", "domain= switch is still the body."
    return None


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide(m)
    early = _classify_common081(m)
    _restore(m, saved)
    if early:
        return early
    return _require_one_machine(m)


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_a081(m)
    _restore(m, saved)
    req = _require_one_machine(m)
    if req:
        return req
    if label == "Store-works":
        return (
            "Store-works",
            "One machine: body from n_actions / percepts; retrieve uses push then adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_b081(m)
    _restore(m, saved)
    req = _require_one_machine(m)
    if req:
        return req
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "One-machine motor bar; small store; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm081 as tm081

    saved = (tm081.make, tm081.classify_a, tm081.classify_b, tm081._w_flags)
    tm081.make = make
    tm081.classify_a = classify_a
    tm081.classify_b = classify_b
    tm081._w_flags = _w_flags
    try:
        m = _run_arm081(**kwargs)
    finally:
        tm081.make, tm081.classify_a, tm081.classify_b, tm081._w_flags = saved
    m["no_domain_switch"] = True
    m["domain_switch"] = False
    label, why = classify_a(m) if kwargs.get("arm") == "A" else classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm082(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.8.2",
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
        f"""# TM.0.8.2 A one machine vs B motor bar

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A one machine | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B motor bar | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
| n_actions | {a.get('n_actions')} | {b.get('n_actions')} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.8.2 one machine, no domain= switch")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm082(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "n": m["A"]["train_s"]["n"],
                "n_actions": m["A"].get("n_actions"),
                "used_a": m["A"].get("used_bind_a"),
                "used_c": m["A"].get("used_bind_c"),
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
