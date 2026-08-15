"""TM.0.11: COMPOSE — independently acquired relations jointly steer.

One primitive: a chosen non-motor consequent becomes the next MATCH frontier.
Visited fact ids are excluded. No hop cap. No S shortcut. Act-local derived state.
"""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.run_tm092 import permute_pair
from experiments.run_tm094 import make as _make094
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile, record_to_tagfile
from three_memory.tag_store import write_prose_notes

_MOTORS = frozenset({"press", "tune", "flip", "hold", "idle"})


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_tm011compose"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(*args, use_compose: bool = True, use_context_kappa: bool = False, **kwargs):
    if kwargs.get("use_evidence") is False or kwargs.get("use_bind_match") is False:
        use_compose = False
    if not use_compose:
        use_context_kappa = False
    # Explicit: 0.0.003 compose path never enables CONTEXT κ by default.
    return _make094(
        *args, use_compose=use_compose, use_context_kappa=use_context_kappa, **kwargs
    )


def permute_compose(seed: int) -> dict[str, Any]:
    spec = permute_pair(seed)
    rng = np.random.default_rng(seed + 77)
    taken = {spec["x"], spec["y"]}

    def _nonce() -> str:
        while True:
            w = "".join(
                str(rng.choice(list("bcdfghjklmnpqrstvwxz"))) + str(rng.choice(list("aeiou")))
                for _ in range(2)
            )
            if w not in taken and w not in _MOTORS:
                taken.add(w)
                return w

    mid = _nonce()
    z = _nonce()
    names = [f"n{int(i):02d}" for i in rng.choice(80, size=5, replace=False)]
    return {
        **spec,
        "mid": mid,
        "z": z,
        "fx": names[0],
        "fy": names[1],
        "fz": names[2],
        "firr": names[3],
        "fspare": names[4],
        "seed": seed,
    }


def write_rels(dest: Path, rows: list[tuple[str, str, str, tuple[int, int] | None]]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for fact_id, bind, did, counts in rows:
        w, n = counts or (0, 0)
        tags: dict[str, Any] = {
            "bind": bind,
            "did": did,
            "here": "chb",
            "w0": bind,
            "hyp": "contradicted" if n else ("supported" if w else "untried"),
            "trials": w + n,
            "wins": w,
            "losses": n,
            "support": w,
            "contradiction": n,
        }
        (dest / f"{fact_id}.tag").write_text(record_to_tagfile(fact_id, tags), encoding="utf-8")


def s_hash(s_dir: Path) -> str:
    parts: list[bytes] = []
    if not s_dir.exists():
        return hashlib.sha256(b"").hexdigest()
    for p in sorted(s_dir.glob("*.tag")):
        parts.append(p.name.encode())
        parts.append(p.read_bytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def has_direct_motor(s_dir: Path, bind: str) -> bool:
    for path in s_dir.glob("*.tag"):
        _fid, tags = parse_tagfile(path.read_text(encoding="utf-8"))
        if tags.get("bind") == bind and str(tags.get("did") or "").lower() in _MOTORS - {"hold", "idle"}:
            return True
    return False


def _motor(name: str) -> str:
    return str(name or "hold").lower()


def probe_cue(policy: UsePolicy, s_dir: Path | None, seed: int, cue: str | None) -> dict[str, Any]:
    tokens = frozenset({cue.lower()}) if cue else frozenset()
    with tempfile.TemporaryDirectory(prefix="tm011compose_empty_") as tmp:
        store = s_dir if s_dir is not None else Path(tmp)
        ag = make(store, None, policy, explore_epsilon=0.0)
        ag.reset_rho()
        out = probe(ag, "probe_channel_b", seed, tokens=tokens)
        out["cue"] = cue
        out["compose_hops"] = (out.get("policy") or {}).get("compose_hops")
        out["evidence_resolved"] = bool((out.get("policy") or {}).get("evidence_resolved"))
        out["evidence_tie"] = bool((out.get("policy") or {}).get("evidence_tie"))
        return out


def probe_same_agent(ag: Any, seed: int, cue: str) -> dict[str, Any]:
    tokens = frozenset({cue.lower()})
    out = probe(ag, "probe_channel_b", seed, tokens=tokens)
    out["cue"] = cue
    out["compose_hops"] = (out.get("policy") or {}).get("compose_hops")
    return out


def classify_compose(cell: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str]:
    m1, m2 = spec["m1"], spec["m2"]
    if _motor(cell["main"]["action_name"]) != m1:
        return "Fail", f"Main chain was {_motor(cell['main']['action_name'])}, not {m1}."
    if cell.get("direct_before"):
        return "Fail", "Direct X→motor existed before the compose probe."
    if cell.get("direct_after"):
        return "Fail", "Direct X→motor appeared after the compose probe."
    if cell.get("s_hash_before") != cell.get("s_hash_after"):
        return "Fail", "S mutated during composition use."
    if _motor(cell["broken"]["action_name"]) != "hold":
        return "Fail", "Broken chain did not HOLD."
    if _motor(cell["wrong_second"]["action_name"]) != m2:
        return "Fail", f"Wrong second edge was {_motor(cell['wrong_second']['action_name'])}, not {m2}."
    if _motor(cell["wrong_first"]["action_name"]) != "hold":
        return "Fail", "Wrong first edge did not HOLD."
    if _motor(cell["irr"]["action_name"]) != m1:
        return "Fail", "Irrelevant high-support edge stole the chain."
    if _motor(cell["donor_press"]["action_name"]) != m1:
        return "Fail", "Downstream donor S1 did not follow PRESS edge."
    if _motor(cell["donor_tune"]["action_name"]) != m2:
        return "Fail", "Downstream donor S2 did not follow TUNE edge."
    if _motor(cell["upstream"]["action_name"]) != m2:
        return "Fail", "Upstream donor S3 did not follow the first-edge swap."
    if _motor(cell["residue"]["action_name"]) != "hold":
        return "Fail", "Derived frontier leaked across acts."
    if _motor(cell["wiped"]["action_name"]) != "hold":
        return "Fail", "Wiped S still steered."
    if _motor(cell["reset"]["action_name"]) != m1:
        return "Fail", "ρ reset lost composition from S."
    hops = cell["main"].get("compose_hops")
    if hops != 2:
        return "Fail", f"Main chain compose_hops was {hops}, not 2."
    return (
        "Store-works",
        "Two independently acquired relations composed at use; no shortcut in S.",
    )


def run_compose_battery(policy: UsePolicy, spec: dict[str, Any], dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    x, y, z = spec["x"], spec["mid"], spec["z"]
    m1, m2 = spec["m1"], spec["m2"]
    fx, fy, fz, firr = spec["fx"], spec["fy"], spec["fz"], spec["firr"]
    seed = int(spec["seed"])

    main = dest / "main"
    write_rels(
        main,
        [
            (fx, x, y, (1, 0)),
            (fy, y, m1, (1, 0)),
        ],
    )
    direct_before = has_direct_motor(main, x)
    h_before = s_hash(main)
    main_probe = probe_cue(policy, main, seed + 1, x)
    h_after = s_hash(main)
    direct_after = has_direct_motor(main, x)

    broken = dest / "broken"
    write_rels(broken, [(fx, x, y, (1, 0))])
    wrong_second = dest / "wrong_second"
    write_rels(
        wrong_second,
        [
            (fx, x, y, (1, 0)),
            (fy, y, m2, (1, 0)),
        ],
    )
    wrong_first = dest / "wrong_first"
    write_rels(
        wrong_first,
        [
            (fx, x, z, (1, 0)),
            (fy, y, m1, (1, 0)),
        ],
    )
    irr = dest / "irr"
    write_rels(
        irr,
        [
            (fx, x, y, (1, 0)),
            (fy, y, m1, (1, 0)),
            (firr, z, m1, (1000, 0)),
        ],
    )
    donor_press = dest / "donor_press"
    donor_tune = dest / "donor_tune"
    write_rels(
        donor_press,
        [
            (fx, x, y, (1, 0)),
            (fy, y, m1, (1, 0)),
        ],
    )
    write_rels(
        donor_tune,
        [
            (fx, x, y, (1, 0)),
            (fy, y, m2, (1, 0)),
        ],
    )
    upstream = dest / "upstream"
    write_rels(
        upstream,
        [
            (fx, x, z, (1, 0)),
            (fz, z, m2, (1, 0)),
            (fy, y, m1, (1, 0)),
        ],
    )

    # No-residue: same agent, then remount S to only Y→motor and cue an unseen token.
    residue_s = dest / "residue"
    write_rels(
        residue_s,
        [
            (fx, x, y, (1, 0)),
            (fy, y, m1, (1, 0)),
        ],
    )
    ag = make(residue_s, None, policy, explore_epsilon=0.0)
    ag.reset_rho()
    first = probe_same_agent(ag, seed + 40, x)
    write_rels(residue_s, [(fy, y, m1, (1, 0))])
    ag.store.reload()
    residue = probe_same_agent(ag, seed + 41, z)

    cell = {
        "main": main_probe,
        "direct_before": direct_before,
        "direct_after": direct_after,
        "s_hash_before": h_before,
        "s_hash_after": h_after,
        "broken": probe_cue(policy, broken, seed + 2, x),
        "wrong_second": probe_cue(policy, wrong_second, seed + 3, x),
        "wrong_first": probe_cue(policy, wrong_first, seed + 4, x),
        "irr": probe_cue(policy, irr, seed + 5, x),
        "donor_press": probe_cue(policy, donor_press, seed + 6, x),
        "donor_tune": probe_cue(policy, donor_tune, seed + 7, x),
        "upstream": probe_cue(policy, upstream, seed + 8, x),
        "residue_first": first,
        "residue": residue,
        "wiped": probe_cue(policy, None, seed + 9, x),
        "reset": probe_cue(policy, main, seed + 10, x),
        "spec": spec,
    }
    label, why = classify_compose(cell, spec)
    cell["classification"] = label
    cell["rationale"] = why
    return cell


def classify_a(m: dict[str, Any]) -> tuple[str, str]:
    if m.get("use_compose") is False:
        return "Fail", "Compose was frozen off."
    return str(m.get("compose_classification") or "Fail"), str(m.get("compose_rationale") or "")


def classify_b(m: dict[str, Any]) -> tuple[str, str]:
    saved = m.get("use_compose")
    m["use_compose"] = False
    m["use_evidence"] = False
    m["use_bind_match"] = False
    label, why = _classify_b091(m)
    m["use_compose"] = saved
    if (m.get("train_s", {}).get("n") or 0) > MAX_TRAIN_S_FILES:
        return "Fail", "Shared return train S is still stamp-collecting clutter."
    return label, why


def _w_flags(w_files: list[str], w_dir: Path) -> dict[str, Any]:
    flags = _w_flags091(w_files, w_dir)
    flags["use_evidence"] = True
    flags["use_compose"] = True
    flags["compose"] = True
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
    label, why = classify_b(m)
    m["classification"] = label
    m["rationale"] = why
    return m


def run_tm011compose(seed: int = 12345, n_train: int = 500, max_steps: int = 32) -> dict[str, Any]:
    run_dir = _run_dir()
    policy = UsePolicy(seed=7, lr=0.2)
    batteries = [
        run_compose_battery(policy, permute_compose(s), run_dir / f"cmp_{s}")
        for s in (seed, seed + 1, seed + 2)
    ]
    a_ok = all(b["classification"] == "Store-works" for b in batteries)
    a = {
        "compose_classification": "Store-works" if a_ok else "Fail",
        "compose_rationale": (
            "COMPOSE Pass on 3 permuted seeds. No shortcut in S."
            if a_ok
            else "COMPOSE miss: " + "; ".join(b["rationale"] for b in batteries)
        ),
        "batteries": batteries,
        "use_compose": True,
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
        "version": "TM.0.11",
        "ex0s": "0.0.002",
        "seed": seed,
        "n_train": n_train,
        "run_dir": str(run_dir),
        "A": a,
        "B": b,
        "same_cortex": a["cortex_hash"] == b["cortex_hash"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# TM.0.11 A COMPOSE vs B motor bar · Ex0S 0.0.002

| Arm | Classification |
|-----|----------------|
| A COMPOSE | **{a['classification']}** |
| B motor bar | **{b['classification']}** |

A: {a['rationale']}

B: {b['rationale']}

Claim: two independently acquired relations composed at use time without materializing a shortcut in S.
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TM.0.11 COMPOSE")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=32)
    args = p.parse_args()
    m = run_tm011compose(seed=args.seed, n_train=args.n_train, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "A": m["A"]["classification"],
                "B": m["B"]["classification"],
                "run_dir": m["run_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
