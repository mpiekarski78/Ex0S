"""TM.0.6.4: English find without a unique rare token.

Same never-wipe one-bind + new-here recipe as TM.0.6.3, but Open W has several
English hapax clutter pages. has_rare is no longer a unique pointer at p99.
Search still has has_code. Not math, not shared-return rescue, not dropping
has_code or domain="dial".
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
    _n_paragraphs,
    _rare_words,
    clutter_prose as clutter_closed,
)
from experiments.run_tm061 import (
    WIKI_A,
    WIKI_C,
    _MOTOR_WORDS,
    _STATION_WORDS,
    _s_has_bind,
)
from experiments.run_tm062 import run_arm as _run_arm062
from experiments.run_tm063 import (
    classify_a as _classify_a063,
    classify_b as _classify_b063,
    classify_common as _classify_common063,
    make,
    _w_flags as _w_flags063,
)
from three_memory.tag_store import extract_prose_ints, prose_token_stream, prose_tokens, write_prose_notes

# Hapax on late clutter files so untrained first-file (c00) is still common.
# Not argon: that is the distractor on the useful A page.
_HAPAX = {8: "xenon", 9: "neon", 10: "krypton"}
_CLUTTER_HAPAX = ("xenon", "neon", "krypton")


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm064"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clutter_prose() -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    bodies = []
    for name, body in clutter_closed():
        idx = int(Path(name).stem[1:])
        if idx in _HAPAX:
            word = _HAPAX[idx]
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
        bodies.append(body)
        notes.append((name, body))
    assert len(set(bodies)) == len(bodies)
    return notes


def wiki_prose(*, include_a: bool = False, include_c: bool = False) -> list[tuple[str, str]]:
    notes = clutter_prose()
    if include_a:
        assert _n_paragraphs(WIKI_A[1]) >= 2
        assert not extract_prose_ints(WIKI_A[1])
        toks = prose_tokens(WIKI_A[1])
        assert not (toks & _MOTOR_WORDS)
        assert not (toks & _STATION_WORDS)
        stream = prose_token_stream(WIKI_A[1])
        assert stream.index("push") < stream.index("argon")
        notes.append(WIKI_A)
    if include_c:
        assert _n_paragraphs(WIKI_C[1]) >= 2
        assert not extract_prose_ints(WIKI_C[1])
        toks = prose_tokens(WIKI_C[1])
        assert not (toks & _MOTOR_WORDS)
        assert not (toks & _STATION_WORDS)
        stream = prose_token_stream(WIKI_C[1])
        assert stream.index("adjust") < stream.index("alpha")
        notes.append(WIKI_C)
    return notes


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags063(w_files, w_dir)
    both = wiki_prose(include_a=True, include_c=True)
    flags["w_min_paragraphs"] = min((_n_paragraphs(b) for _, b in both), default=0)
    rare = _rare_words(both)
    clutter_rare = [n for n, ws in rare.items() if n.startswith("c") and ws]
    flags["w_clutter_has_rare"] = bool(clutter_rare)
    flags["w_n_rare_clutter"] = len(clutter_rare)
    flags["w_useful_has_rare"] = bool(rare.get("p99.md")) and bool(rare.get("p98.md"))
    flags["w_useful_n_rare"] = len(rare.get("p99.md") or [])
    flags["w_useful_only_rare"] = bool(flags["w_useful_has_rare"] and not flags["w_clutter_has_rare"])
    flags["find_without_unique_rare"] = True
    flags["english_life"] = True
    flags["use_stamp_new_here"] = True
    return flags


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved_clutter = m.get("w_clutter_has_rare")
    m["w_clutter_has_rare"] = False
    early = _classify_common063(m)
    m["w_clutter_has_rare"] = saved_clutter
    if early:
        return early
    if not saved_clutter or m.get("w_useful_only_rare"):
        return "Confound", "Useful page is still the only rare token."
    if (m.get("w_n_rare_clutter") or 0) < 3:
        return "Confound", "Need several rare clutter pages so uniqueness is gone."
    if not m.get("find_without_unique_rare"):
        return "Fail", "Find-without-unique-rare was frozen off."
    if m.get("use_block_here"):
        return "Confound", "Concurrent bind/block was smuggled onto this slice."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved_clutter = m.get("w_clutter_has_rare")
    m["w_clutter_has_rare"] = False
    label, why = _classify_a063(m)
    m["w_clutter_has_rare"] = saved_clutter
    if not saved_clutter or m.get("w_useful_only_rare"):
        return "Confound", "Useful page is still the only rare token."
    if (m.get("w_n_rare_clutter") or 0) < 3:
        return "Confound", "Need several rare clutter pages so uniqueness is gone."
    if not m.get("find_without_unique_rare"):
        return "Fail", "Find-without-unique-rare was frozen off."
    if m.get("use_block_here"):
        return "Confound", "Concurrent bind/block was smuggled onto this slice."
    tag = (m.get("train_s") or {}).get("tag") or m.get("a_tag") or ""
    for w in _CLUTTER_HAPAX:
        if _s_has_bind(tag, w) or _s_has_bind(m.get("both_tag") or "", w):
            return "Fail", f"Clutter hapax {w} was bound as the act."
    if label == "Store-works":
        return (
            "Store-works",
            "English multi-rare W: never-wipe train S still PRESS from push; C life on that S: A PRESS, C TUNE. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved_clutter = m.get("w_clutter_has_rare")
    m["w_clutter_has_rare"] = False
    label, why = _classify_b063(m)
    m["w_clutter_has_rare"] = saved_clutter
    if not saved_clutter or m.get("w_useful_only_rare"):
        return "Confound", "Useful page is still the only rare token."
    if m.get("use_block_here"):
        return "Confound", "Concurrent bind/block was smuggled onto this slice."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return English multi-rare; dirty S two facts; cortex frozen.",
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


def run_tm064(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.6.4",
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
        f"""# TM.0.6.4 A English multi-rare vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split English multi-rare | **{a['classification']}** | {a['train_return_last50']:.2f} |
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
    p = argparse.ArgumentParser(description="TM.0.6.4 English find without a unique rare token")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm064(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
