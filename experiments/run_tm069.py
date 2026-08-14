"""TM.0.6.9: find-novel without a unique two-rare pair.

Same find-novel + in-hand English store as TM.0.6.8, but several clutter
pages also have two hapax. p99/p98 are no longer the only 2-rare pages.
Not a p98 ranker, not unique-rare restored, not leftover W walk, not math,
not dropping has_code or domain="dial", not solving B.
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
from experiments.run_tm061 import (
    WIKI_A,
    WIKI_C,
    _MOTOR_WORDS,
    _STATION_WORDS,
    _s_has_bind,
)
from experiments.run_tm064 import _CLUTTER_HAPAX, clutter_prose as clutter_one_rare
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm068 import (
    classify_a as _classify_a068,
    classify_b as _classify_b068,
    classify_common as _classify_common068,
    make as _make068,
    run_arm as _run_arm068,
    _w_flags as _w_flags068,
)
from three_memory.tag_store import extract_prose_ints, prose_token_stream, prose_tokens, write_prose_notes

# Extra hapax on the same late clutter files as TM.0.6.4. Useful pages score 3
# novel tokens because "# p99"/"# p98" tokenize to rare "p"; two body hapax
# still lose. Not argon/push/adjust/alpha. Not motor or station names.
# Not helium (054 closed wiki).
_EXTRA = {8: ("radon", "lithium"), 9: ("cesium", "nickel"), 10: ("cobalt", "quartz")}
_CLUTTER_HAPAX2 = ("radon", "lithium", "cesium", "nickel", "cobalt", "quartz")
_PAIR_KEYS = (
    "w_clutter_has_two_rare",
    "w_n_two_rare_clutter",
    "find_without_unique_pair",
    "w_useful_only_two_rare",
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm069"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, **kwargs):
    return _make068(*args, **kwargs)


def clutter_prose() -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    bodies = []
    for name, body in clutter_one_rare():
        idx = int(Path(name).stem[1:])
        if idx in _EXTRA:
            w2, w3 = _EXTRA[idx]
            body = body.replace(
                "The tray was quiet.",
                f"{w2.capitalize()} in the bin. {w3.capitalize()} on the shelf. The tray was quiet.",
                1,
            )
            assert {w2, w3} <= prose_tokens(body)
        assert _n_ok(body)
        bodies.append(body)
        notes.append((name, body))
    assert len(set(bodies)) == len(bodies)
    return notes


def _n_ok(body: str) -> bool:
    assert _n_paragraphs(body) >= 2
    assert not extract_prose_ints(body)
    toks = prose_tokens(body)
    assert not (toks & _MOTOR_WORDS)
    assert not (toks & _STATION_WORDS)
    return True


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
    flags = _w_flags068(w_files, w_dir)
    both = wiki_prose(include_a=True, include_c=True)
    rare = _rare_words(both)
    two_clutter = [n for n, ws in rare.items() if n.startswith("c") and len(ws) >= 2]
    flags["w_clutter_has_two_rare"] = bool(two_clutter)
    flags["w_n_two_rare_clutter"] = len(two_clutter)
    flags["w_useful_has_two_rare"] = len(rare.get("p99.md") or []) >= 2 and len(rare.get("p98.md") or []) >= 2
    flags["w_useful_only_two_rare"] = bool(flags["w_useful_has_two_rare"] and not flags["w_clutter_has_two_rare"])
    flags["find_without_unique_pair"] = True
    flags["use_find_novel"] = True
    flags["find_novel"] = True
    flags["use_in_hand_new_here"] = True
    flags["in_hand_new_here"] = True
    flags["use_revise_head"] = True
    flags["use_commit_here_only"] = True
    flags["correct_dirty_s"] = True
    flags["use_block_here"] = True
    flags["use_stamp_new_here"] = True
    flags["find_without_unique_rare"] = True
    flags["english_life"] = True
    return flags


def _hide_pair(m: dict[str, Any]) -> dict[str, Any]:
    saved = {k: m.get(k) for k in _PAIR_KEYS}
    m["w_clutter_has_two_rare"] = False
    m["w_n_two_rare_clutter"] = 0
    m["find_without_unique_pair"] = False
    m["w_useful_only_two_rare"] = False
    return saved


def _restore_pair(m: dict[str, Any], saved: dict[str, Any]) -> None:
    m.update(saved)


def _require_pair(m: dict[str, Any]) -> tuple[str, str] | None:
    if not m.get("w_clutter_has_two_rare") or m.get("w_useful_only_two_rare"):
        return "Confound", "Useful pages are still the only two-rare pages."
    if (m.get("w_n_two_rare_clutter") or 0) < 3:
        return "Confound", "Need several two-rare clutter pages so the pair is not unique."
    if not m.get("find_without_unique_pair"):
        return "Fail", "Find-without-unique-pair was frozen off."
    if m.get("use_retry_novel"):
        return "Confound", "Retry-novel was smuggled onto this slice."
    return None


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide_pair(m)
    early = _classify_common068(m)
    _restore_pair(m, saved)
    if early:
        return early
    return _require_pair(m)


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide_pair(m)
    label, why = _classify_a068(m)
    _restore_pair(m, saved)
    req = _require_pair(m)
    if req:
        return req
    tag = (m.get("train_s") or {}).get("tag") or m.get("a_tag") or ""
    for w in _CLUTTER_HAPAX + _CLUTTER_HAPAX2:
        if _s_has_bind(tag, w) or _s_has_bind(m.get("both_tag") or "", w):
            return "Fail", f"Clutter hapax {w} was bound as the act."
    if label == "Store-works":
        return (
            "Store-works",
            "Find-novel without a unique two-rare pair: several clutter pages also have two hapax; train S PRESS from push; C life A PRESS, C TUNE from adjust. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide_pair(m)
    label, why = _classify_b068(m)
    _restore_pair(m, saved)
    req = _require_pair(m)
    if req:
        return req
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Shared return find-novel without a unique two-rare pair; small store two facts; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm068 as tm068

    saved = (tm068.make, tm068.classify_a, tm068.classify_b, tm068._w_flags)
    tm068.make = make
    tm068.classify_a = classify_a
    tm068.classify_b = classify_b
    tm068._w_flags = _w_flags
    try:
        m = _run_arm068(**kwargs)
    finally:
        tm068.make, tm068.classify_a, tm068.classify_b, tm068._w_flags = saved
    m["use_find_novel"] = True
    m["find_novel"] = True
    m["find_without_unique_pair"] = True
    label, why = (classify_a if kwargs.get("split") else classify_b)(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm069(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.6.9",
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
        f"""# TM.0.6.9 A find-novel without unique two-rare vs B shared return

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A split find-novel, two-rare clutter | **{a['classification']}** | {a['train_return_last50']:.2f} |
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
| n_revised train | {a.get('n_revised_train', 0)} | {b.get('n_revised_train', 0)} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.6.9 find-novel without a unique two-rare pair")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm069(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
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
        "rev",
        m["A"].get("n_revised_train"),
        m["A"]["a_tag"].strip().replace("\n", " | "),
        "both",
        m["A"]["both_tag"].strip().replace("\n", " | "),
    )
    print(
        "B",
        m["B"]["train_s_probe"]["action_name"],
        m["B"]["both_after_c"]["action_name"],
        "n",
        m["B"]["train_s"]["n"],
        "last50",
        m["B"]["train_return_last50"],
        m["B"]["a_tag"].strip().replace("\n", " | "),
        "both",
        m["B"]["both_tag"].strip().replace("\n", " | "),
    )


if __name__ == "__main__":
    main()
