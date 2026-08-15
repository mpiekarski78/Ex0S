"""TM.0.9.4: REVISION — beliefs remain editable after evidence changes.

No new genome. Same 0.9.3 comparison. Preference can withdraw and reverse.
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

from experiments.run_tm066 import MAX_TRAIN_S_FILES
from experiments.run_tm091 import classify_b as _classify_b091, run_arm as _run_arm091, _w_flags as _w_flags091
from experiments.run_tm093 import (
    _counts,
    _find_ids,
    _motor,
    make as _make093,
    permute_evidence,
    probe_cue,
    write_triple_s,
)
from three_memory.dial_env import ChannelDialWorld
from three_memory.policy import UsePolicy
from three_memory.tag_store import write_prose_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm094"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, **kwargs):
    return _make093(*args, **kwargs)


def earn_seq(
    s_dir: Path,
    spec: dict[str, Any],
    policy: UsePolicy,
    steps: list[tuple[str, bool]],
    *,
    reset_after: int | None = None,
) -> dict[str, Any]:
    ag = make(s_dir, None, policy, explore_epsilon=0.0)
    ids = _find_ids(s_dir, spec)
    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    for i, (which, ok) in enumerate(steps):
        if reset_after is not None and i == reset_after:
            ag.reset_rho()
        ag._last_chosen_ids = [ids[which]]
        ag.observe_outcome(obs, ok, {"opened": ok})
    return {"ids": ids, "counts": _counts(s_dir)}


def _pair_counts(counts: dict[str, dict[str, int]], ids: dict[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    return counts.get(ids.get("m1"), {}), counts.get(ids.get("m2"), {})


def classify_revision(cell: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str]:
    m1, m2 = spec["m1"], spec["m2"]
    walk = [_motor(p["action_name"]) for p in cell["walk"]]
    if walk != ["hold", m1, "hold", m2]:
        return "Fail", f"Walk was {walk}, not HOLD/{m1}/HOLD/{m2}."
    if _motor(cell["early"]["action_name"]) != m1:
        return "Fail", "Early evidence did not prefer M1."
    if _motor(cell["revised"]["action_name"]) != m2:
        return "Fail", "Later evidence did not reverse to M2."
    early_c = cell.get("early_counts") or {}
    e1, e2 = _pair_counts(early_c, cell.get("revise_ids") or {})
    if e1.get("support") != 2 or e1.get("contradiction") != 0:
        return "Fail", f"Early M1 counts not earned: {e1}"
    if e2.get("support") != 0 or e2.get("contradiction") != 1:
        return "Fail", f"Early M2 counts not earned: {e2}"
    later_c = cell.get("revised_counts") or {}
    r1, r2 = _pair_counts(later_c, cell.get("revise_ids") or {})
    if r1.get("support") != 2 or r1.get("contradiction") != 2:
        return "Fail", f"Revised M1 counts not earned: {r1}"
    if r2.get("support") != 3 or r2.get("contradiction") != 1:
        return "Fail", f"Revised M2 counts not earned: {r2}"
    a1, a2 = _pair_counts(cell.get("order_a_counts") or {}, cell.get("order_a_ids") or {})
    b1, b2 = _pair_counts(cell.get("order_b_counts") or {}, cell.get("order_b_ids") or {})
    if (a1, a2) != (b1, b2):
        return "Fail", f"Order histories did not land on the same S evidence: {(a1, a2)} vs {(b1, b2)}."
    if _motor(cell["order_a"]["action_name"]) != _motor(cell["order_b"]["action_name"]):
        return "Fail", "Same final evidence, different order, different motor."
    if _motor(cell["order_a"]["action_name"]) == "hold":
        return "Fail", "Order-invariant histories produced HOLD, not a shared winner."
    return (
        "Store-works",
        "HOLD→M1→HOLD→M2; mid-reset then reverse; same final S evidence → same motor.",
    )


def run_revision_battery(policy: UsePolicy, spec: dict[str, Any], dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    walk_s = dest / "walk"
    rev_s = dest / "revise"
    oa = dest / "order_a"
    ob = dest / "order_b"
    write_triple_s(walk_s, spec)
    seed = int(spec["seed"])
    walk_probes = [probe_cue(policy, walk_s, seed + 1, spec["x"])]
    earn_seq(walk_s, spec, policy, [("m1", True)])
    walk_probes.append(probe_cue(policy, walk_s, seed + 2, spec["x"]))
    earn_seq(walk_s, spec, policy, [("m2", True)])
    walk_probes.append(probe_cue(policy, walk_s, seed + 3, spec["x"]))
    earn_seq(walk_s, spec, policy, [("m2", True)])
    walk_probes.append(probe_cue(policy, walk_s, seed + 4, spec["x"]))

    write_triple_s(rev_s, spec)
    life = make(rev_s, None, policy, explore_epsilon=0.0)
    revise_ids = _find_ids(rev_s, spec)
    obs = ChannelDialWorld(seed=1).reset("probe_channel_b")
    for which, ok in (("m1", True), ("m1", True), ("m2", False)):
        life._last_chosen_ids = [revise_ids[which]]
        life.observe_outcome(obs, ok, {"opened": ok})
    early = probe_cue(policy, rev_s, seed + 5, spec["x"])
    early_counts = _counts(rev_s)
    life.reset_rho()
    for which, ok in (("m1", False), ("m1", False), ("m2", True), ("m2", True), ("m2", True)):
        life._last_chosen_ids = [revise_ids[which]]
        life.observe_outcome(obs, ok, {"opened": ok})
    revised = probe_cue(policy, rev_s, seed + 6, spec["x"])

    hist_a = [("m1", True), ("m2", False), ("m1", True), ("m1", False), ("m2", True)]
    hist_b = [("m2", True), ("m1", False), ("m1", True), ("m2", False), ("m1", True)]
    write_triple_s(oa, spec)
    write_triple_s(ob, spec)
    a_earn = earn_seq(oa, spec, policy, hist_a)
    b_earn = earn_seq(ob, spec, policy, hist_b)
    cell = {
        "walk": walk_probes,
        "early": early,
        "revised": revised,
        "revise_ids": revise_ids,
        "early_counts": early_counts,
        "revised_counts": _counts(rev_s),
        "order_a": probe_cue(policy, oa, seed + 7, spec["x"]),
        "order_b": probe_cue(policy, ob, seed + 8, spec["x"]),
        "order_a_ids": a_earn["ids"],
        "order_b_ids": b_earn["ids"],
        "order_a_counts": a_earn["counts"],
        "order_b_counts": b_earn["counts"],
        "spec": spec,
    }
    label, why = classify_revision(cell, spec)
    cell["classification"] = label
    cell["rationale"] = why
    return cell


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    return str(m.get("revision_classification") or "Fail"), str(m.get("revision_rationale") or "")


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
    flags["use_evidence"] = True
    flags["revision"] = True
    flags["genome_delta"] = 0
    return flags


def run_arm_b(**kwargs: Any) -> dict[str, Any]:
    import experiments.run_tm091 as tm091
    from experiments.run_tm093 import make as make093

    saved = (tm091.make, tm091.classify_b, tm091._w_flags)
    tm091.make = make093
    tm091.classify_b = classify_b
    tm091._w_flags = _w_flags
    try:
        m = _run_arm091(**kwargs)
    finally:
        tm091.make, tm091.classify_b, tm091._w_flags = saved
    label, why = classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm094(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    policy = UsePolicy(seed=7, lr=0.2)
    batteries = [
        run_revision_battery(policy, permute_evidence(s), run_dir / f"rev_{s}")
        for s in (seed, seed + 1, seed + 2)
    ]
    a_ok = all(b["classification"] == "Store-works" for b in batteries)
    a = {
        "revision_classification": "Store-works" if a_ok else "Fail",
        "revision_rationale": (
            "REVISION Pass on 3 permuted seeds. Genome is 0.9.3."
            if a_ok
            else "REVISION miss: " + "; ".join(b["rationale"] for b in batteries)
        ),
        "batteries": batteries,
        "genome_delta": 0,
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
        "version": "TM.0.9.4",
        "seed": seed,
        "n_train": n_train,
        "genome_delta": 0,
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.9.4 A REVISION vs B motor bar

| Arm | Classification |
|-----|----------------|
| A REVISION | **{a['classification']}** |
| B motor bar | **{b['classification']}** |

A: {a['rationale']}

B: {b['rationale']}

Genome delta vs 0.9.3: 0
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.9.4 REVISION")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm094(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "genome_delta": 0,
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
