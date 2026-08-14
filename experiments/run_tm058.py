"""TM.0.5.8: scale of Open W — a pile of documents, not a dozen logs.

Same multi-rare never-wipe recipe as TM.0.5.7. Clutter grows to 64 distinct
pages. Search still has has_code. Not English, not shared-return rescue,
not domain= drop, not dropping has_code.
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

from experiments.run_tm054 import (
    WIKI_A,
    WIKI_C,
    _MOTOR_WORDS,
    _STATION_WORDS,
    _n_paragraphs,
    _rare_words,
)
from experiments.run_tm057 import (
    _c_life_on_s,
    _copy_s,
    _probe_s,
    _s_snapshot,
    _train_keep,
    classify_a as _classify_a057,
    classify_b as _classify_b057,
    classify_common as _classify_c057,
    make,
    run_arm as _run_arm057,
)
from experiments.run_tm057 import _w_flags as _w_flags057
from three_memory.tag_store import extract_prose_ints, prose_tokens, write_prose_notes

N_CLUTTER = 64
_HAPAX = {61: "xenon", 62: "argon", 63: "neon"}
_SHELF = ("dusty", "cable", "noisy", "grit", "cloth", "clamp", "spool", "hook")
_BIN = ("fixture", "rust", "metal", "mesh", "plate", "strap", "rack", "bolt")


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm058"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _closed_bodies() -> list[str]:
    bodies = []
    for a in _SHELF:
        for b in _BIN:
            bodies.append(
                "Staff bench log.\n\n"
                f"{a.capitalize()} fixture on the shelf. {b.capitalize()} in the bin.\n"
                "Notes from the lab follow. The tray was quiet.\n"
            )
    assert len(bodies) == N_CLUTTER
    assert len(set(bodies)) == len(bodies)
    return bodies


def clutter_prose(*, hapax: bool = True) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    for i, body in enumerate(_closed_bodies()):
        if hapax and i in _HAPAX:
            word = _HAPAX[i]
            body = body.replace(
                "The tray was quiet.",
                f"{word.capitalize()} in the tray. The tray was quiet.",
                1,
            )
            assert word in prose_tokens(body)
        assert _n_paragraphs(body) >= 2
        assert not extract_prose_ints(body)
        toks = prose_tokens(body)
        assert not (toks & _MOTOR_WORDS)
        assert not (toks & _STATION_WORDS)
        notes.append((f"c{i:02d}.md", body))
    assert len(notes) == N_CLUTTER
    return notes


def wiki_prose(*, include_a: bool = False, include_c: bool = False) -> list[tuple[str, str]]:
    notes = clutter_prose(hapax=True)
    if include_a:
        assert _n_paragraphs(WIKI_A[1]) >= 2
        assert not extract_prose_ints(WIKI_A[1])
        assert not (prose_tokens(WIKI_A[1]) & _MOTOR_WORDS)
        assert not (prose_tokens(WIKI_A[1]) & _STATION_WORDS)
        notes.append(WIKI_A)
    if include_c:
        assert _n_paragraphs(WIKI_C[1]) >= 2
        assert not extract_prose_ints(WIKI_C[1])
        assert not (prose_tokens(WIKI_C[1]) & _MOTOR_WORDS)
        assert not (prose_tokens(WIKI_C[1]) & _STATION_WORDS)
        notes.append(WIKI_C)
    return notes


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags057(w_files, w_dir)
    both = wiki_prose(include_a=True, include_c=True)
    flags["w_min_paragraphs"] = min((_n_paragraphs(b) for _, b in both), default=0)
    rare = _rare_words(both)
    clutter_rare = [n for n, ws in rare.items() if n.startswith("c") and ws]
    flags["w_clutter_has_rare"] = bool(clutter_rare)
    flags["w_n_rare_clutter"] = len(clutter_rare)
    flags["w_useful_has_rare"] = bool(rare.get("p99.md")) and bool(rare.get("p98.md"))
    flags["w_useful_only_rare"] = bool(flags["w_useful_has_rare"] and not flags["w_clutter_has_rare"])
    flags["w_n_distinct_clutter"] = len(clutter_prose(hapax=True))
    flags["w_n"] = len(w_files)
    flags["w_scale"] = True
    flags["find_without_unique_rare"] = True
    flags["open_w"] = True
    flags["english_life"] = False
    flags["accumulate_s"] = True
    flags["train_wipe_s"] = False
    flags["has_code_in_search"] = True
    return flags


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    if (m.get("w_n_distinct_clutter") or 0) < N_CLUTTER:
        return "Confound", "Open W was not scaled; still a dozen logs."
    if (m.get("w_n") or 0) < N_CLUTTER:
        return "Confound", "W pile is still tiny."
    if not m.get("w_scale"):
        return "Fail", "Scale of Open W was frozen off."
    return _classify_c057(m)


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    label, why = _classify_a057(m)
    if label != "Store-works":
        return label, why
    return (
        "Store-works",
        "Scaled Open W: never-wipe train S still PRESS; C life on that dirty S: A PRESS, C TUNE. Cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    return _classify_b057(m)


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm057 as tm057

    saved = (tm057.classify_a, tm057.classify_b, tm057._w_flags)
    tm057.classify_a = classify_a
    tm057.classify_b = classify_b
    tm057._w_flags = _w_flags
    try:
        return _run_arm057(**kwargs)
    finally:
        tm057.classify_a, tm057.classify_b, tm057._w_flags = saved


def run_tm058(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.5.8",
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
        f"""# TM.0.5.8 A scale of Open W vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split scaled W | **{a['classification']}** | {a['train_return_last50']:.2f} |
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
| Distinct clutter | {a.get('w_n_distinct_clutter')} | {b.get('w_n_distinct_clutter')} |
| Rare clutter pages | {a.get('w_n_rare_clutter')} | {b.get('w_n_rare_clutter')} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.8 scale of Open W — a pile of documents")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm058(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
        "nW",
        m["A"].get("w_n_distinct_clutter"),
        "rareC",
        m["A"].get("w_n_rare_clutter"),
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
