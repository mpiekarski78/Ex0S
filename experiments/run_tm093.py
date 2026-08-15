"""TM.0.9.3: EVIDENCE among MATCH-eligible relations.

Prefer the better-supported applicable relation. Equal evidence stays
unresolved. Counts are earned in S, not stored in P.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm040 import probe
from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm091 import classify_b as _classify_b091, run_arm as _run_arm091, _w_flags as _w_flags091
from experiments.run_tm092 import (
    classify_match_battery,
    make as _make092,
    permute_pair,
    run_match_battery,
    write_relation_s,
)
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm093"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_evidence: bool = True, **kwargs):
    if kwargs.get("use_bind_match") is False or kwargs.get("use_hyp_survive") is False:
        use_evidence = False
    return _make092(*args, use_evidence=use_evidence, use_keep_steerer=False, **kwargs)


def write_triple_s(
    dest: Path,
    spec: dict[str, Any],
    *,
    y_support: int = 100,
    counts: dict[str, tuple[int, int]] | None = None,
) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    rows = (
        (spec["fx"], spec["x"], spec["m1"]),
        (spec["fy"], spec["x"], spec["m2"]),
        (spec.get("fy2") or "ny", spec["y"], "hold"),
    )
    counts = counts or {}
    for fact_id, bind, did in rows:
        tags: dict[str, Any] = {
            "bind": bind,
            "did": did,
            "here": "chb",
            "w0": bind,
            "hyp": "untried",
            "trials": 0,
            "wins": 0,
            "losses": 0,
            "support": 0,
            "contradiction": 0,
        }
        if bind == spec["y"]:
            tags.update(
                {
                    "hyp": "supported",
                    "wins": y_support,
                    "support": y_support,
                    "trials": y_support,
                }
            )
        if fact_id in counts:
            w, n = counts[fact_id]
            tags["wins"] = w
            tags["losses"] = n
            tags["support"] = w
            tags["contradiction"] = n
            tags["trials"] = w + n
            tags["hyp"] = "contradicted" if n else ("supported" if w else "untried")
        (dest / f"{fact_id}.tag").write_text(record_to_tagfile(fact_id, tags), encoding="utf-8")


def permute_evidence(seed: int) -> dict[str, Any]:
    spec = permute_pair(seed)
    rng = np.random.default_rng(seed + 99)
    spec["fy2"] = f"n{int(rng.integers(40, 80)):02d}"
    return spec


def _motor(name: str) -> str:
    return str(name or "hold").lower()


def probe_cue(policy: UsePolicy, s_dir: Path | None, seed: int, cue: str | None) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm093_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make(store, None, policy, explore_epsilon=0.0)
        ag.reset_rho()
        out = probe(ag, "probe_channel_b", seed, tokens=tokens)
        out["cue"] = cue
        out["evidence_resolved"] = bool((out.get("policy") or {}).get("evidence_resolved"))
        out["evidence_tie"] = bool((out.get("policy") or {}).get("evidence_tie"))
        return out


def _find_ids(s_dir: Path, spec: dict[str, Any]) -> dict[str, str]:
    out = {}
    for path in s_dir.glob("*.tag"):
        fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        b, d = tags.get("bind"), tags.get("did")
        if b == spec["x"] and d == spec["m1"]:
            out["m1"] = fid
        elif b == spec["x"] and d == spec["m2"]:
            out["m2"] = fid
        elif b == spec["y"]:
            out["y"] = fid
    return out


def _counts(s_dir: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for path in s_dir.glob("*.tag"):
        fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        out[fid] = {
            "support": int(tags.get("support") or tags.get("wins") or 0),
            "contradiction": int(tags.get("contradiction") or tags.get("losses") or 0),
        }
    return out


def earn_unequal(s_dir: Path, spec: dict[str, Any], policy: UsePolicy) -> dict[str, Any]:
    """M1 succeeds, M2 fails, M1 succeeds — through the organism update path."""
    ag = make(s_dir, None, policy, explore_epsilon=0.0)
    ids = _find_ids(s_dir, spec)
    from three_memory.dial_env import ChannelDialWorld

    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    for fid, ok in ((ids["m1"], True), (ids["m2"], False), (ids["m1"], True)):
        ag._last_chosen_ids = [fid]
        ag.observe_outcome(obs, ok, {"opened": ok})
    return {"ids": ids, "counts": _counts(s_dir)}


def classify_evidence(cell: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str]:
    m1, m2 = spec["m1"], spec["m2"]
    if _motor(cell["unequal"]["action_name"]) != m1:
        return "Fail", f"Unequal evidence steered {_motor(cell['unequal']['action_name'])}, not {m1}."
    if _motor(cell["equal"]["action_name"]) != "hold":
        return "Fail", "Equal evidence claimed a winner from ordering."
    if cell["equal"].get("evidence_resolved"):
        return "Fail", "Equal evidence set evidence_resolved."
    if _motor(cell["swap_a"]["action_name"]) != m1:
        return "Fail", "S_a did not follow stronger M1."
    if _motor(cell["swap_b"]["action_name"]) != m2:
        return "Fail", "S_b did not follow stronger M2."
    if _motor(cell["wiped"]["action_name"]) != "hold":
        return "Fail", "Wiped S still steered."
    if _motor(cell["reset"]["action_name"]) != m1:
        return "Fail", "ρ reset lost the S preference."
    earned = cell.get("earned") or {}
    ids = earned.get("ids") or {}
    counts = earned.get("counts") or {}
    c1 = counts.get(ids.get("m1"), {})
    c2 = counts.get(ids.get("m2"), {})
    if c1.get("support") != 2 or c1.get("contradiction") != 0:
        return "Fail", f"M1 counts not earned: {c1}"
    if c2.get("support") != 0 or c2.get("contradiction") != 1:
        return "Fail", f"M2 counts not earned: {c2}"
    if ids.get("m1") not in counts or ids.get("m2") not in counts:
        return "Fail", "A rival was erased."
    if ids.get("y") not in counts:
        return "Fail", "Y note was erased."
    return (
        "Store-works",
        "Unequal follows stronger X; equal unresolved; swap follows S; Y dropped by MATCH.",
    )


def run_evidence_battery(policy: UsePolicy, spec: dict[str, Any], dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    unequal = dest / "unequal"
    equal = dest / "equal"
    swap_a = dest / "swap_a"
    swap_b = dest / "swap_b"
    write_triple_s(unequal, spec)
    earned = earn_unequal(unequal, spec, policy)
    write_triple_s(
        equal,
        spec,
        counts={spec["fx"]: (1, 0), spec["fy"]: (1, 0)},
    )
    write_triple_s(swap_a, spec, counts={spec["fx"]: (2, 0), spec["fy"]: (0, 1)})
    write_triple_s(swap_b, spec, counts={spec["fx"]: (0, 1), spec["fy"]: (2, 0)})
    seed = int(spec["seed"])
    cells = {
        "unequal": probe_cue(policy, unequal, seed + 1, spec["x"]),
        "equal": probe_cue(policy, equal, seed + 2, spec["x"]),
        "swap_a": probe_cue(policy, swap_a, seed + 3, spec["x"]),
        "swap_b": probe_cue(policy, swap_b, seed + 4, spec["x"]),
        "reset": probe_cue(policy, unequal, seed + 5, spec["x"]),
        "wiped": probe_cue(policy, None, seed + 6, spec["x"]),
        "earned": earned,
    }
    label, why = classify_evidence(cells, spec)
    cells["classification"] = label
    cells["rationale"] = why
    cells["spec"] = spec
    return cells


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if m.get("use_compose"):
        return "Confound", "Compose was smuggled onto this slice."
    if m.get("use_evidence") is False:
        return "Fail", "Evidence was frozen off."
    return str(m.get("evidence_classification") or "Fail"), str(m.get("evidence_rationale") or "")


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = m.get("use_evidence")
    m["use_evidence"] = False
    m["use_bind_match"] = False
    label, why = _classify_b091(m)
    m["use_evidence"] = saved
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    return label, why


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags091(w_files, w_dir)
    flags["use_bind_match"] = True
    flags["use_evidence"] = True
    flags["bind_match"] = True
    flags["evidence"] = True
    return flags


def run_arm_b(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm091 as tm091

    saved = (tm091.make, tm091.classify_b, tm091._w_flags)
    tm091.make = make
    tm091.classify_b = classify_b
    tm091._w_flags = _w_flags
    try:
        m = _run_arm091(**kwargs)
    finally:
        tm091.make, tm091.classify_b, tm091._w_flags = saved
    m["use_evidence"] = True
    label, why = classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm093(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    policy = UsePolicy(seed=7, lr=0.2)
    match_seeds = [seed, seed + 1, seed + 2]
    batteries = [run_evidence_battery(policy, permute_evidence(s), run_dir / f"ev_{s}") for s in match_seeds]
    box_match = run_match_battery(policy, permute_pair(seed + 17), run_dir / "box_match", force_use=True)
    # BOX-MATCH must stay Store-works on the 0.9.2 make (evidence off).
    from experiments.run_tm092 import make as make092
    import experiments.run_tm092 as tm092

    saved_make = tm092.make
    tm092.make = make092
    try:
        box_match_092 = run_match_battery(
            UsePolicy(seed=7, lr=0.2), permute_pair(seed + 18), run_dir / "box_match_092", force_use=True
        )
    finally:
        tm092.make = saved_make
    a_ok = all(b["classification"] == "Store-works" for b in batteries)
    match_ok = box_match_092["classification"] == "Store-works"
    a = {
        "use_evidence": True,
        "evidence_classification": "Store-works" if a_ok and match_ok else "Fail",
        "evidence_rationale": (
            "EVIDENCE A/B/C Pass on 3 permuted seeds; BOX-MATCH still Store-works."
            if a_ok and match_ok
            else "EVIDENCE miss: "
            + "; ".join(b["rationale"] for b in batteries)
            + f"; BOX-MATCH {box_match_092['classification']}"
        ),
        "batteries": batteries,
        "box_match_092": box_match_092,
        "box_match_093": box_match,
        "cortex_hash": make(run_dir / "hash", None, policy, enabled=False).weight_hash(),
    }
    a["classification"], a["rationale"] = classify_a(a)

    from experiments.run_tm054 import clutter_prose as clutter_closed
    from experiments.run_tm080 import wiki_prose

    w_a = run_dir / "W_a"
    w_both = run_dir / "W_both"
    w_clutter = run_dir / "W_clutter"
    write_prose_notes(w_a, wiki_prose(include_a=True))
    write_prose_notes(w_both, wiki_prose(include_a=True, include_c=True))
    write_prose_notes(w_clutter, clutter_closed())
    w_files = sorted(p.name for p in w_both.glob("*.md"))
    b = run_arm_b(
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
        "version": "TM.0.9.3",
        "seed": seed,
        "n_train": n_train,
        "max_steps": max_steps,
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.9.3 A EVIDENCE vs B motor bar

| Arm | Classification |
|-----|----------------|
| A EVIDENCE | **{a['classification']}** |
| B motor bar | **{b['classification']}** |

A: {a['rationale']}

B: {b['rationale']}
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.9.3 EVIDENCE")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm093(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "n": m["B"].get("train_s", {}).get("n"),
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
