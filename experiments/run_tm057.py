"""TM.0.5.7: find without a unique rare token.

Same never-wipe train as TM.0.5.6, but Open W has several distinctive clutter
pages. has_rare is no longer a unique pointer at p99. Search still has has_code.
Not English, not shared-return rescue, not domain= drop, not dropping has_code.
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

from experiments.run_tm012 import _has_field
from experiments.run_tm052 import _s_has_token
from experiments.run_tm054 import (
    WIKI_A,
    WIKI_C,
    _MOTOR_WORDS,
    _STATION_WORDS,
    _n_paragraphs,
    _rare_words,
    clutter_prose as clutter_closed,
)
from experiments.run_tm055 import classify_common as _classify055
from experiments.run_tm056 import (
    _c_life_on_s,
    _copy_s,
    _probe_s,
    _s_snapshot as _s_snapshot056,
    _train_keep,
    make,
    run_arm as _run_arm056,
)
from experiments.run_tm056 import _w_flags as _w_flags056
from three_memory.tag_store import extract_prose_ints, prose_tokens, write_prose_notes

# Hapax on late clutter files so untrained first-file (c00) is still common.
# Trained has_rare then lands on a distinctive page that is not uniquely p99.
_HAPAX = {8: "xenon", 9: "argon", 10: "neon"}
_DISTINCTIVE = ("xenon", "argon", "neon", "krypton", "helium")


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm057"
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
    flags = _w_flags056(w_files, w_dir)
    both = wiki_prose(include_a=True, include_c=True)
    flags["w_min_paragraphs"] = min((_n_paragraphs(b) for _, b in both), default=0)
    rare = _rare_words(both)
    clutter_rare = [n for n, ws in rare.items() if n.startswith("c") and ws]
    flags["w_clutter_has_rare"] = bool(clutter_rare)
    flags["w_n_rare_clutter"] = len(clutter_rare)
    flags["w_useful_has_rare"] = bool(rare.get("p99.md")) and bool(rare.get("p98.md"))
    flags["w_useful_only_rare"] = bool(flags["w_useful_has_rare"] and not flags["w_clutter_has_rare"])
    flags["find_without_unique_rare"] = True
    flags["open_w"] = True
    flags["english_life"] = False
    flags["accumulate_s"] = True
    flags["train_wipe_s"] = False
    return flags


def _s_snapshot(s_dir: Path) -> dict[str, Any]:
    snap = _s_snapshot056(s_dir)
    tag = snap.get("tag") or ""
    for w in _DISTINCTIVE:
        snap[f"found_{w}"] = _s_has_token(tag, w)
    snap["found_distinctive"] = any(snap[f"found_{w}"] for w in _DISTINCTIVE)
    return snap


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved_clutter = m.get("w_clutter_has_rare")
    saved_wipe = m.get("train_wipe_s")
    # 0.5.5 treats rare clutter as a confound; this jump requires it.
    m["w_clutter_has_rare"] = False
    m["train_wipe_s"] = True
    early = _classify055(m)
    m["w_clutter_has_rare"] = saved_clutter
    m["train_wipe_s"] = saved_wipe
    if early:
        return early
    if not saved_clutter or m.get("w_useful_only_rare"):
        return "Confound", "Useful page is still the only rare token."
    if (m.get("w_n_rare_clutter") or 0) < 3:
        return "Confound", "Need several rare clutter pages so uniqueness is gone."
    if not m.get("w_useful_has_rare"):
        return "Confound", "Useful page has no rare word."
    if m.get("train_wipe_s"):
        return "Fail", "Train still wiped S every episode."
    if not m.get("find_without_unique_rare"):
        return "Fail", "Find-without-unique-rare was frozen off."
    if not m.get("has_code_in_search"):
        return "Confound", "has_code was dropped from search (not this jump)."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["untrained_probe"]["correct"] or m["untrained_probe"]["action_name"] == "press":
        return "Fail", "Untrained already PRESS-correct on A."
    if m["untrained_foil_c"]["action_name"] == "press":
        return "Fail", "Untrained already PRESS on foil C."
    if m["untrained_live"].get("found_press") or m["untrained_live"].get("n_annotated", 0):
        return "Fail", "Untrained already annotated a motor onto S."
    if not m["train_s_probe"]["correct"] or m["train_s_probe"]["action_name"] != "press":
        return "Fail", "After never-wipe train, dirty S was not PRESS."
    if m["train_s_foil"]["action_name"] == "press" or m["train_s_foil"]["correct"]:
        return "Fail", "Train S still fired PRESS on channel C."
    if m["train_s_foil"]["action_name"] != "hold":
        return "Fail", "Train S on channel C was not HOLD."
    if not m["both_after_a"]["correct"] or m["both_after_a"]["action_name"] != "press":
        return "Fail", "C life on dirty S lost A's PRESS."
    if not m["both_after_c"]["correct"] or m["both_after_c"]["action_name"] != "tune":
        return "Fail", "C life on dirty S was not TUNE."
    if m["wipe_ctrl_a"]["action_name"] == "press" or m["wipe_ctrl_a"]["correct"]:
        return "Fail", "Wiping dirty S still PRESS on A; keep-S was not load-bearing."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if m["menu_control"]["correct"]:
        return "Fail", "Exact match still solved A."
    if m["search_control"]["correct"]:
        return "Fail", "Untrained search still solved A."
    if m["write_control"]["correct"]:
        return "Fail", "Annotate-off still solved A."
    if m["clutter_control"]["correct"]:
        return "Fail", "Closed-lexicon clutter-only W still solved A."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    if not m["train_s"]["found_press"] or not m["train_s"]["found_cha"]:
        return "Fail", "Never-wipe train never stamped press+cha."
    if not m["c_live"]["found_tune"] or not m["c_live"].get("found_chc"):
        return "Fail", "C life on dirty S never stamped tune+chc."
    if not m["c_live"]["found_press"] or not m["c_live"].get("found_cha"):
        return "Fail", "C life clobbered train's press+cha."
    if not m["train_s"].get("found_distinctive"):
        return "Fail", "Train S never committed a distinctive page."
    tag = m.get("both_tag", "")
    if _has_field(tag, "action") or _has_field(tag, "door") or _has_field(tag, "where"):
        return "Fail", "S restored filed tag names."
    if _has_field(tag, "n0") or _has_field(tag, "n1"):
        return "Fail", "S still has n* digit tags."
    return (
        "Store-works",
        "Multi-rare W: never-wipe train S still PRESS; C life on that dirty S: A PRESS, C TUNE. Cortex frozen.",
    )


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    early = classify_common(m)
    if early:
        return early
    if m["trained_split"]:
        return "Fail", "Split credit was restored to rescue shared return."
    if not m["train_s_probe"]["correct"] or m["train_s_probe"]["action_name"] != "press":
        return "Fail", "Shared return never-wipe train S was not PRESS."
    if not m["both_after_c"]["correct"] or m["both_after_c"]["action_name"] != "tune":
        return "Fail", "Shared return C on dirty S was not TUNE."
    if m["empty_S"]["correct"]:
        return "Fail", "Empty S still solved A."
    if not (m["search_changed"] and m["vname_changed"] and m["write_changed"]):
        return "Fail", "A joint head did not move."
    return (
        "Store-works",
        "Shared return multi-rare never-wipe train; dirty S two facts; cortex frozen.",
    )


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm056 as tm056

    saved = (
        tm056.classify_a,
        tm056.classify_b,
        tm056._w_flags,
        tm056._s_snapshot,
    )
    tm056.classify_a = classify_a
    tm056.classify_b = classify_b
    tm056._w_flags = _w_flags
    tm056._s_snapshot = _s_snapshot
    try:
        return _run_arm056(**kwargs)
    finally:
        (
            tm056.classify_a,
            tm056.classify_b,
            tm056._w_flags,
            tm056._s_snapshot,
        ) = saved


def run_tm057(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    # Closed-lexicon clutter: no hapax, so stamp-on-rare cannot rescue this control.
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
        "version": "TM.0.5.7",
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
        f"""# TM.0.5.7 A find without unique rare vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split multi-rare | **{a['classification']}** | {a['train_return_last50']:.2f} |
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
| Rare clutter pages | {a.get('w_n_rare_clutter')} | {b.get('w_n_rare_clutter')} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.5.7 find without a unique rare token")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm057(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
