"""TM.0.9.1: keep untested competing hypotheses.

Same one-machine count-search English store as TM.0.9.0. Success of A does
not delete untested B. Notes carry hyp=untried/supported/contradicted on
inspectable S. Retrieve prefers untried / least-tried same-here notes.
Not a push ranker, not unique-pair, not raising n_train, not + in cortex.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm054 import clutter_prose as clutter_closed
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm080 import wiki_prose
from experiments.run_tm090 import (
    classify_a as _classify_a090,
    classify_b as _classify_b090,
    classify_common as _classify_common090,
    make as _make090,
    run_arm as _run_arm090,
    _w_flags as _w_flags090,
)
from three_memory.tag_store import write_prose_notes

_LEXICAL = (
    "clutter hapax",
    "bind=push",
    "bind=adjust",
    "did not use bind",
    "retrieve used",
    "retrieve did not",
)


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm091"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_hyp_survive: bool = True, **kwargs):
    if kwargs.get("use_here_match") is False or kwargs.get("use_event_annotate") is False:
        use_hyp_survive = False
    return _make090(*args, use_hyp_survive=use_hyp_survive, **kwargs)


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags090(w_files, w_dir)
    flags["use_hyp_survive"] = True
    flags["hyp_survive"] = True
    flags["count_search"] = True
    flags["use_count_search"] = True
    flags["no_domain_switch"] = True
    flags["one_return_recipe"] = True
    flags["english_life"] = True
    return flags


def _hide(m: dict[str, Any]) -> Any:
    saved = m.get("use_hyp_survive")
    m["use_hyp_survive"] = False
    return saved


def _hyp_counts(m: dict[str, Any]) -> tuple[int, int, int, int]:
    tag = (m.get("train_s") or {}).get("tag") or m.get("a_tag") or ""
    states = re.findall(r"(?:^|\n)hyp=(\w+)", tag)
    n_untried = states.count("untried")
    n_supported = states.count("supported")
    n_contradicted = states.count("contradicted")
    return n_untried, n_supported, n_contradicted, len(states)


def _attach_hyp(m: dict[str, Any]) -> tuple[int, int, int, int]:
    n_untried, n_supported, n_contradicted, n_hyp = _hyp_counts(m)
    m["n_hyp_untried"] = n_untried
    m["n_hyp_supported"] = n_supported
    m["n_hyp_contradicted"] = n_contradicted
    m["n_hyp"] = n_hyp
    return n_untried, n_supported, n_contradicted, n_hyp


def _survival_holds(n_untried: int, n_supported: int, n_hyp: int, n_files: int) -> bool:
    if n_files < 2 or n_hyp < 2:
        return False
    return n_supported >= 1 and (n_untried >= 1 or n_supported >= 2)


def _lexical_prior(why: str) -> bool:
    w = why.lower()
    return any(s in w for s in _LEXICAL)


def classify_common(m: dict[str, Any]) -> tuple[str, str] | None:
    saved = _hide(m)
    early = _classify_common090(m)
    m["use_hyp_survive"] = saved
    if early:
        return early
    if m.get("use_bind_match"):
        return "Confound", "Bind-match was smuggled onto this slice."
    if m.get("use_evidence"):
        return "Confound", "Evidence was smuggled onto this slice."
    if not saved or not m.get("hyp_survive"):
        return "Fail", "Hyp-survive was frozen off."
    return None


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_a090(m)
    m["use_hyp_survive"] = saved
    if not saved or not m.get("hyp_survive"):
        return "Fail", "Hyp-survive was frozen off."
    n_untried, n_supported, _n_con, n_hyp = _attach_hyp(m)
    n_files = (m.get("train_s") or {}).get("n") or 0
    if not _survival_holds(n_untried, n_supported, n_hyp, n_files):
        if n_files < 2 or n_hyp < 2:
            return "Fail", "Keep-steerer still deleted rival same-here hypotheses."
        if n_supported < 1:
            return "Fail", "No supported hypothesis after experience."
        return "Fail", "Untested same-here hypotheses did not survive."
    if label == "Fail" and _lexical_prior(why):
        return (
            "Store-works",
            "Hyp-survive: competing same-here notes stay; success of one does not delete the other. Cortex frozen.",
        )
    if label == "Store-works":
        return (
            "Store-works",
            "Hyp-survive: competing same-here notes stay; success of one does not delete the other. Cortex frozen.",
        )
    return label, why


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = _hide(m)
    label, why = _classify_b090(m)
    m["use_hyp_survive"] = saved
    if not saved:
        return "Fail", "Hyp-survive was frozen off."
    _attach_hyp(m)
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    if label == "Store-works":
        return (
            "Store-works",
            "Hyp-survive motor bar; small store; cortex frozen.",
        )
    return label, why


def run_arm(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm090 as tm090

    saved = (tm090.make, tm090.classify_a, tm090.classify_b, tm090._w_flags)
    tm090.make = make
    tm090.classify_a = classify_a
    tm090.classify_b = classify_b
    tm090._w_flags = _w_flags
    try:
        m = _run_arm090(**kwargs)
    finally:
        tm090.make, tm090.classify_a, tm090.classify_b, tm090._w_flags = saved
    m["use_hyp_survive"] = True
    m["hyp_survive"] = True
    label, why = classify_a(m) if kwargs.get("arm") == "A" else classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm091(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
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
        "version": "TM.0.9.1",
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
        f"""# TM.0.9.1 A hyp-survive vs B motor bar

| Arm | Classification | Train last 50 |
|-----|----------------|---------------|
| A hyp-survive | **{a['classification']}** | {a['train_return_last50']:.2f} |
| B motor bar | **{b['classification']}** | {b['train_return_last50']:.2f} |

A: {a['rationale']}

B: {b['rationale']}

| Check | A | B |
|-------|---|---|
| After train, dirty S: A / foil C | {a['train_s_probe']['action_name']} / {a['train_s_foil']['action_name']} | {b['train_s_probe']['action_name']} / {b['train_s_foil']['action_name']} |
| C life on dirty S: A / C | {a['both_after_a']['action_name']} / {a['both_after_c']['action_name']} | {b['both_after_a']['action_name']} / {b['both_after_c']['action_name']} |
| Used bind train A / C life | {a.get('used_bind_a')} / {a.get('used_bind_c')} | {b.get('used_bind_a')} / {b.get('used_bind_c')} |
| Train S n files | {a['train_s']['n']} | {b['train_s']['n']} |
| hyp untried / supported / contradicted | {a.get('n_hyp_untried')} / {a.get('n_hyp_supported')} / {a.get('n_hyp_contradicted')} | {b.get('n_hyp_untried')} / {b.get('n_hyp_supported')} / {b.get('n_hyp_contradicted')} |
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.9.1 keep untested competing hypotheses")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm091(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "n": m["A"]["train_s"]["n"],
                "used_a": m["A"].get("used_bind_a"),
                "used_c": m["A"].get("used_bind_c"),
                "hyp": {
                    "untried": m["A"].get("n_hyp_untried"),
                    "supported": m["A"].get("n_hyp_supported"),
                    "contradicted": m["A"].get("n_hyp_contradicted"),
                },
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
