"""TM.0.8.0: scale English Open W — a pile, not a dozen logs.

Same keep-steerer local-alias recipe as TM.0.7.2. Clutter grows to 64 distinct
English pages; many also have three hapax so p99/p98 are not a unique novel
pair. Not a p98 ranker, not unique-pair restored, not leftover W walk, not
math, not dropping has_code or domain="dial", not solving B.
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

from experiments.run_tm054 import _n_paragraphs, _rare_words, clutter_prose as clutter_closed
from experiments.run_tm058 import N_CLUTTER, _closed_bodies
from experiments.run_tm061 import WIKI_A, WIKI_C, _MOTOR_WORDS, _STATION_WORDS
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm072 import (
    classify_a as _classify_a072,
    classify_b as _classify_b072,
    classify_common as _classify_common072,
    make as _make072,
    run_arm as _run_arm072,
    _w_flags as _w_flags072,
)
from three_memory.tag_store import extract_prose_ints, prose_token_stream, prose_tokens, write_prose_notes

# Late pages so untrained first-file (c00) stays common. Three body hapax so
# heading-rare "p" on p99/p98 is not a unique novel count. Not argon/push/
# adjust/alpha. Not motor or station names. Not helium (054 closed wiki).
_TRIPLES: dict[int, tuple[str, str, str]] = {
    8: ("xenon", "radon", "lithium"),
    9: ("neon", "cesium", "nickel"),
    10: ("krypton", "cobalt", "quartz"),
    11: ("sodium", "boron", "carbon"),
    12: ("fluorine", "silicon", "sulfur"),
    13: ("chlorine", "titanium", "vanadium"),
    14: ("chromium", "manganese", "copper"),
    15: ("zinc", "gallium", "germanium"),
    16: ("arsenic", "selenium", "bromine"),
    17: ("rubidium", "strontium", "yttrium"),
    18: ("zirconium", "niobium", "molybdenum"),
    19: ("palladium", "silver", "cadmium"),
    20: ("indium", "antimony", "tellurium"),
    21: ("iodine", "barium", "lanthanum"),
    22: ("tungsten", "platinum", "mercury"),
    23: ("thallium", "bismuth", "uranium"),
}
_SCALE_KEYS = ("w_scale", "w_n_distinct_clutter", "scale_english_w")
MIN_TWO_RARE = 16


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm080"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, **kwargs):
    return _make072(*args, **kwargs)


def _n_ok(body: str) -> bool:
    assert _n_paragraphs(body) >= 2
    assert not extract_prose_ints(body)
    toks = prose_tokens(body)
    assert not (toks & _MOTOR_WORDS)
    assert not (toks & _STATION_WORDS)
    return True


def clutter_prose() -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    bodies = []
    for i, body in enumerate(_closed_bodies()):
        if i in _TRIPLES:
            w1, w2, w3 = _TRIPLES[i]
            body = body.replace(
                "The tray was quiet.",
                f"{w1.capitalize()} in the tray. {w2.capitalize()} in the bin. "
                f"{w3.capitalize()} on the shelf. The tray was quiet.",
                1,
            )
            assert {w1, w2, w3} <= prose_tokens(body)
        assert _n_ok(body)
        bodies.append(body)
        notes.append((f"c{i:02d}.md", body))
    assert len(notes) == N_CLUTTER
    assert len(set(bodies)) == len(bodies)
    return notes


def wiki_prose(*, include_a: bool = False, include_c: bool = False) -> list[tuple[str, str]]:
    notes = clutter_prose()
    if include_a:
        assert _n_ok(WIKI_A[1])
        stream = prose_token_stream(WIKI_A[1])
        assert stream.index("push") < stream.index("argon")
        notes.append(WIKI_A)
    if include_c:
        assert _n_ok(WIKI_C[1])
        stream = prose_token_stream(WIKI_C[1])
        assert stream.index("adjust") < stream.index("alpha")
        notes.append(WIKI_C)
    return notes


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags072(w_files, w_dir)
    both = wiki_prose(include_a=True, include_c=True)
    rare = _rare_words(both)
    two_clutter = [n for n, ws in rare.items() if n.startswith("c") and len(ws) >= 2]
    flags["w_clutter_has_two_rare"] = bool(two_clutter)
    flags["w_n_two_rare_clutter"] = len(two_clutter)
    flags["w_useful_has_two_rare"] = len(rare.get("p99.md") or []) >= 2 and len(rare.get("p98.md") or []) >= 2
    flags["w_useful_only_two_rare"] = bool(flags["w_useful_has_two_rare"] and not flags["w_clutter_has_two_rare"])
    flags["w_n_distinct_clutter"] = len(clutter_prose())
    flags["w_n"] = len(w_files)
    flags["w_scale"] = True
    flags["scale_english_w"] = True
    flags["find_without_unique_pair"] = True
    flags["use_keep_steerer"] = True
    flags["keep_steerer"] = True
    flags["english_life"] = True
    return flags


def _hide(m: dict[str, Any]) -> dict[str, Any]:
    saved = {k: m.get(k) for k in _SCALE_KEYS}
    saved["w_n_two_rare_clutter"] = m.get("w_n_two_rare_clutter")
    m["w_scale"] = False
    m["scale_english_w"] = False
    if (m.get("w_n_distinct_clutter") or 0) >= N_CLUTTER:
        m["w_n_distinct_clutter"] = 11
    return saved


def _restore(m: dict[str, Any], saved: dict[str, Any]) -> None:
    m.update(saved)


def _require_scale(m: dict[str, Any]) -> tuple[str, str] | None:
    if (m.get("w_n_distinct_clutter") or 0) < N_CLUTTER:
        return "Confound", "English Open W was not scaled; still a dozen logs."
    if (m.get("w_n") or 0) < N_CLUTTER:
        return "Confound", "W pile is still tiny."
    if not m.get("w_scale") or not m.get("scale_english_w"):
        return "Fail", "Scale of English Open W was frozen off."
    if not m.get("w_clutter_has_two_rare") or m.get("w_useful_only_two_rare"):
        return "Confound", "Useful pages are still the only two-rare pages."
    if (m.get("w_n_two_rare_clutter") or 0) < MIN_TWO_RARE:
        return "Confound", "Need a pile of two-rare clutter, not three late logs."
    return None


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide(m)
    early = _classify_common072(m)
    _restore(m, saved)
    if early:
        return early
    return _require_scale(m)


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_a072(m)
    _restore(m, saved)
    req = _require_scale(m)
    if req:
        return req
    if label == "Store-works":
        return (
            "Store-works",
            "Scale English Open W: 64-page pile, more two-rare clutter; retrieve uses push then adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_b072(m)
    _restore(m, saved)
    req = _require_scale(m)
    if req:
        return req
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return scale English Open W; small store; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm072 as tm072

    saved = (tm072.make, tm072.classify_a, tm072.classify_b, tm072._w_flags)
    tm072.make = make
    tm072.classify_a = classify_a
    tm072.classify_b = classify_b
    tm072._w_flags = _w_flags
    try:
        m = _run_arm072(**kwargs)
    finally:
        tm072.make, tm072.classify_a, tm072.classify_b, tm072._w_flags = saved
    m["w_scale"] = True
    m["scale_english_w"] = True
    label, why = (classify_a if kwargs.get("split") else classify_b)(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm080(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.8.0",
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
        f"""# TM.0.8.0 A scale English Open W vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split scale English W | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B shared return | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Distinct clutter | {a.get('w_n_distinct_clutter')} | {b.get('w_n_distinct_clutter')} |
| Two-rare clutter | {a.get('w_n_two_rare_clutter')} | {b.get('w_n_two_rare_clutter')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.8.0 scale English Open W")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm080(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "n": m["A"]["train_s"]["n"],
                "w": m["A"].get("w_n_distinct_clutter"),
                "two_rare": m["A"].get("w_n_two_rare_clutter"),
                "used_a": m["A"].get("used_bind_a"),
                "used_c": m["A"].get("used_bind_c"),
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
