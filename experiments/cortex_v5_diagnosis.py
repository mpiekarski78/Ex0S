"""TM.0.23.CORTEX.V5.DIAG — observational diagnosis of C6 asymmetry and C4 swap timing.

Does not edit neural_cortex.py. Does not rescore revealed v5 gate worlds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cortex_develop_life import (
    BODY0,
    bind_life_actuators,
    curriculum_tokens,
    motor_latent,
    pair_seeds,
    teach_loop,
)
from experiments.cortex_develop_scorers import _act_token_counts
from experiments.run_tm023cortex import build_observe, make_cortex
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_LOCK = REPO_ROOT / "docs" / "cortex_diagnosis.v5.lock"
DIAG_MD = REPO_ROOT / "docs" / "tm023cortex_v5_diagnosis.md"
CANDIDATE_V5 = REPO_ROOT / "docs" / "cortex.candidate.v5.lock"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_credit(ag: NeuralCortex) -> None:
    def _noop(s_t, body_t):  # noqa: ANN001
        return {"adv": 0.0, "pred_err": 0.0}

    ag._apply_credit = _noop  # type: ignore[method-assign]


def _observe_only(
    ag: NeuralCortex,
    *,
    ix: str,
    symbols: list[str],
    body: list[float],
    state: list[str],
) -> dict[str, Any]:
    return ag.observe(
        build_observe(
            interaction_token=ix,
            source_token="src_diag",
            ordered_symbols=symbols,
            observable_state=state,
            body_state=body,
        )
    )


def _birth_cosines(ag: NeuralCortex, handles: list[str], cue: list[str]) -> dict[str, Any]:
    """Force one sensory pass then read ACT-query cosine per bound handle (no credit)."""
    _freeze_credit(ag)
    _observe_only(ag, ix="cos0", symbols=cue, body=list(BODY0), state=["st_idle"])
    q = ag._from_t(ag.W_act_query @ ag.rho)
    qn = float(np.linalg.norm(q) + 1e-12)
    rows = []
    for i, hid in enumerate(handles):
        v = ag.motor_vocab[hid]
        vn = float(np.linalg.norm(v) + 1e-12)
        cos = float(np.dot(q, v) / (qn * vn))
        rows.append(
            {
                "slot": i,
                "handle": hid,
                "cos": cos,
                "vec_norm": vn,
                "unit_cos": cos,
            }
        )
    rows_sorted = sorted(rows, key=lambda r: -r["cos"])
    return {
        "by_slot": rows,
        "argmax_slot": rows_sorted[0]["slot"] if rows_sorted else None,
        "argmax_handle": rows_sorted[0]["handle"] if rows_sorted else None,
        "margin": (rows_sorted[0]["cos"] - rows_sorted[1]["cos"]) if len(rows_sorted) > 1 else None,
    }


def diagnose_asymmetry() -> dict[str, Any]:
    """Why one opaque handle wins without useful consequences."""
    slot0_argmax = 0
    n_org = 0
    norms: list[float] = []
    motor_unique = True
    vec_pairs: list[tuple[str, bytes]] = []
    bind_order_follows: list[bool] = []
    physics_swap_bind: list[dict[str, Any]] = []
    c6_replay: dict[str, Any] | None = None

    # Per-organism uniqueness: 16 pairs × main/twin + extra motor seeds
    seen_motor_seeds: set[int] = set()
    for pid in range(16):
        for seeds in pair_seeds(pid):
            n_org += 1
            if seeds.seed_motor in seen_motor_seeds:
                motor_unique = False
            seen_motor_seeds.add(seeds.seed_motor)
            toks = curriculum_tokens(seeds)
            handles = [toks[r] for r in ("press", "harm", "get", "drop")]
            with tempfile.TemporaryDirectory(prefix="v5d_as_") as tmp:
                ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
                ag.bind_actuators(handles)
                for hid in handles:
                    vec_pairs.append((hid, ag.motor_vocab[hid].tobytes()))
                    norms.append(float(np.linalg.norm(ag.motor_vocab[hid])))
                cos = _birth_cosines(ag, handles, [toks["a"]])
                if cos["argmax_slot"] == 0:
                    slot0_argmax += 1

    # Same handle ID across organisms should not share vectors (IDs differ; check vector collisions)
    vec_counts = Counter(v for _, v in vec_pairs)
    vector_collisions = sum(1 for c in vec_counts.values() if c > 1)

    # Bind-order vs physics: pair 3 (C6 organism), two-handle, permute bind order
    seeds, _ = pair_seeds(3)
    toks = curriculum_tokens(seeds)
    a_h, b_h = toks["press"], toks["harm"]
    neutral = {
        "act_effects": {
            a_h: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
            b_h: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
        }
    }
    for order in ([a_h, b_h], [b_h, a_h]):
        with tempfile.TemporaryDirectory(prefix="v5d_bo_") as tmp:
            ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
            ag.bind_actuators(order)
            teach_loop(
                ag,
                seeds,
                n=80,
                symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
                latent=neutral,
            )
            counts = _act_token_counts(ag, toks, 40, [toks["a"]], latent=neutral)
            first = order[0]
            bind_order_follows.append(counts.get(first, 0) >= counts.get(order[1], 0))
            physics_swap_bind.append(
                {
                    "bind_order": order,
                    "counts": counts,
                    "first_slot_preferred": counts.get(first, 0) > counts.get(order[1], 0),
                    "press_count": counts.get(a_h, 0),
                    "harm_count": counts.get(b_h, 0),
                }
            )

    # Counterbalance: assign beneficial physics to slot 0 vs slot 1 (two-handle)
    counter: list[dict[str, Any]] = []
    for beneficial_first in (True, False):
        lat = {
            "act_effects": {
                a_h: {"state": ["st_pressed"], "delta": [0.25, -0.1, 0.15, 0.0]},
                b_h: {"state": ["st_hurt"], "delta": [-0.35, 0.45, -0.15, 0.0]},
            }
        }
        order = [a_h, b_h] if beneficial_first else [b_h, a_h]
        with tempfile.TemporaryDirectory(prefix="v5d_cb_") as tmp:
            ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
            ag.bind_actuators(order)
            teach_loop(
                ag,
                seeds,
                n=80,
                symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
                latent=lat,
            )
            counts = _act_token_counts(ag, toks, 40, [toks["a"]], latent=lat)
            counter.append(
                {
                    "beneficial_in_slot0": beneficial_first,
                    "bind_order": order,
                    "counts": counts,
                    "press": counts.get(a_h, 0),
                    "harm": counts.get(b_h, 0),
                }
            )

    # Replay C6 four-handle no-consequence on pair 3
    with tempfile.TemporaryDirectory(prefix="v5d_c6_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        handles = bind_life_actuators(ag, toks)
        birth = _birth_cosines(ag, handles, [toks["a"]])
        # new organism for teach (birth cosines froze credit on the first)
        ag2 = make_cortex(Path(tmp) / "s2", genome=seeds.genome(), device="cpu")
        bind_life_actuators(ag2, toks)
        lat4 = motor_latent(toks)
        for h in handles:
            lat4["act_effects"][h] = {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]}
        teach_loop(
            ag2,
            seeds,
            n=80,
            symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
            latent=lat4,
        )
        counts = _act_token_counts(ag2, toks, 40, [toks["a"]], latent=lat4)
        c6_replay = {
            "handles_in_bind_order": handles,
            "birth_argmax_slot": birth["argmax_slot"],
            "birth_argmax_handle": birth["argmax_handle"],
            "birth_margin": birth["margin"],
            "birth_by_slot": birth["by_slot"],
            "post_teach_counts": counts,
            "press": counts.get(toks["press"], 0),
            "harm": counts.get(toks["harm"], 0),
        }

    return {
        "n_organisms": n_org,
        "motor_seeds_unique": motor_unique,
        "n_motor_seeds": len(seen_motor_seeds),
        "vector_collisions": vector_collisions,
        "slot0_birth_argmax": slot0_argmax,
        "slot0_birth_argmax_rate": slot0_argmax / n_org,
        "vec_norm_mean": float(np.mean(norms)),
        "vec_norm_std": float(np.std(norms)),
        "vectors_unnormalized": abs(float(np.mean(norms)) - 1.0) > 0.5,
        "bind_order_preferred_after_neutral_teach": bind_order_follows,
        "bind_order_runs": physics_swap_bind,
        "beneficial_slot_counterbalance": counter,
        "c6_pair3_replay": c6_replay,
    }


def diagnose_swap() -> dict[str, Any]:
    """Separate learned-A / swap / immediate-no-consequence / teach / later-B."""
    seeds, _ = pair_seeds(1)
    toks = curriculum_tokens(seeds)
    a_h, b_h = toks["press"], toks["harm"]
    lat = {
        "act_effects": {
            a_h: {"state": ["st_pressed"], "delta": [0.25, -0.1, 0.15, 0.0]},
            b_h: {"state": ["st_hurt"], "delta": [-0.35, 0.45, -0.15, 0.0]},
        }
    }
    with tempfile.TemporaryDirectory(prefix="v5d_sw_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        ag.bind_actuators([a_h, b_h])
        teach_loop(
            ag,
            seeds,
            n=160,
            symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
            latent=lat,
        )
        learned = _act_token_counts(ag, toks, 40, [toks["a"]], latent=lat)
        pref_a = learned.get(a_h, 0) >= 3 and learned.get(a_h, 0) > learned.get(b_h, 0)
        ckpt = ag.checkpoint()

        lat_swap = copy.deepcopy(lat)
        lat_swap["act_effects"][a_h], lat_swap["act_effects"][b_h] = (
            lat_swap["act_effects"][b_h],
            lat_swap["act_effects"][a_h],
        )

        # Immediate probe 1: frozen credit, no physics — first action only
        ag.load_checkpoint(ckpt)
        _freeze_credit(ag)
        first = _observe_only(
            ag, ix="imm0", symbols=[toks["a"]], body=list(BODY0), state=["st_idle"]
        )
        first_tok = (first.get("action") or {}).get("token")
        first_op = (first.get("action") or {}).get("op")

        # Immediate probe 2: 8 frozen-credit observes, body held at BODY0 (no physics)
        ag.load_checkpoint(ckpt)
        _freeze_credit(ag)
        frozen_counts: dict[str, int] = {}
        for i in range(8):
            out = _observe_only(
                ag,
                ix=f"immf_{i}",
                symbols=[toks["a"]],
                body=list(BODY0),
                state=["st_idle"],
            )
            act = out.get("action") or {}
            if act.get("op") == "ACT" and act.get("token"):
                frozen_counts[act["token"]] = frozen_counts.get(act["token"], 0) + 1

        # Immediate probe 3: contaminated 20-probe apply_event (old C4 method)
        ag.load_checkpoint(ckpt)
        contaminated = _act_token_counts(ag, toks, 20, [toks["a"]], latent=lat_swap)

        # Post-swap teaching then later probe
        ag.load_checkpoint(ckpt)
        teach_loop(
            ag,
            seeds,
            n=40,
            symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
            latent=lat_swap,
        )
        later = _act_token_counts(ag, toks, 40, [toks["a"]], latent=lat_swap)
        pref_b = later.get(b_h, 0) >= 3 and later.get(b_h, 0) > later.get(a_h, 0)

        # Restore
        ag.load_checkpoint(ckpt)
        restored = _act_token_counts(ag, toks, 40, [toks["a"]], latent=lat)

    first_is_a = first_tok == a_h
    first_is_b = first_tok == b_h
    frozen_pref_a = frozen_counts.get(a_h, 0) > frozen_counts.get(b_h, 0)
    frozen_pref_b = frozen_counts.get(b_h, 0) > frozen_counts.get(a_h, 0)
    contaminated_pref_b = contaminated.get(b_h, 0) > contaminated.get(a_h, 0)
    leak = first_is_b
    probe_contamination = (not frozen_pref_b) and contaminated_pref_b
    credit_fail = frozen_pref_a and (not pref_b)

    return {
        "pref_a_before_swap": pref_a,
        "counts_learned": learned,
        "immediate_first_op": first_op,
        "immediate_first_token": first_tok,
        "immediate_first_is_A": first_is_a,
        "immediate_first_is_B": first_is_b,
        "immediate_frozen_counts": frozen_counts,
        "immediate_frozen_pref_A": frozen_pref_a,
        "immediate_contaminated_counts": contaminated,
        "immediate_contaminated_pref_B": contaminated_pref_b,
        "later_counts": later,
        "pref_b_after_teach": pref_b,
        "restore_counts": restored,
        "physics_or_future_leak_into_first_selection": leak,
        "stale_probe_contaminated_by_inprobe_consequences": probe_contamination,
        "credit_assignment_fails_to_revise": credit_fail,
    }


def run_diagnosis(*, write_lock: bool = False) -> dict[str, Any]:
    if not CANDIDATE_V5.exists():
        raise RuntimeError("missing candidate v5")
    cand = json.loads(CANDIDATE_V5.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural != candidate v5 — refuse diagnosis of drifted tree")
    asym = diagnose_asymmetry()
    swap = diagnose_swap()
    causes: list[dict[str, Any]] = []
    if asym["slot0_birth_argmax_rate"] > 0.4:
        causes.append(
            {
                "id": "slot_index_birth_cosine_bias",
                "claim": "Bind-order slot 0 is over-represented as birth ACT-query argmax; slots are not exchangeable.",
                "evidence": {
                    "slot0_birth_argmax_rate": asym["slot0_birth_argmax_rate"],
                    "n": asym["n_organisms"],
                },
            }
        )
    if any(asym["bind_order_preferred_after_neutral_teach"]):
        causes.append(
            {
                "id": "bind_order_neutral_preference",
                "claim": "After no-consequence teaching, the first-bound handle is preferred — slot/order bias, not physics.",
                "evidence": {"runs": asym["bind_order_runs"]},
            }
        )
    if asym["vectors_unnormalized"]:
        causes.append(
            {
                "id": "unnormalized_motor_vectors",
                "claim": "Motor-registry vectors are N(0,1) not unit-normalized (cosine is scale-invariant; norms still vary).",
                "evidence": {
                    "norm_mean": asym["vec_norm_mean"],
                    "norm_std": asym["vec_norm_std"],
                },
            }
        )
    if swap["stale_probe_contaminated_by_inprobe_consequences"]:
        causes.append(
            {
                "id": "stale_probe_applies_new_consequences",
                "claim": "Old C4 immediate 20-probe used apply_event under swapped physics, so credit+body leaked into the 'stale' window.",
                "evidence": {
                    "first_frozen_is_A": swap["immediate_first_is_A"],
                    "contaminated_pref_B": swap["immediate_contaminated_pref_B"],
                },
            }
        )
    if swap["physics_or_future_leak_into_first_selection"]:
        causes.append(
            {
                "id": "selection_sees_swapped_physics",
                "claim": "First frozen no-consequence probe after swap already selects B — selection is not a function of learned weights alone.",
                "evidence": {"first_token": swap["immediate_first_token"]},
            }
        )
    if swap["credit_assignment_fails_to_revise"]:
        causes.append(
            {
                "id": "credit_fails_post_swap_revision",
                "claim": "Immediate frozen probe stays A but 40 post-swap episodes do not move preference to B.",
                "evidence": {"later": swap["later_counts"]},
            }
        )
    if not swap["pref_b_after_teach"] and swap["immediate_first_is_A"]:
        causes.append(
            {
                "id": "revision_incomplete_or_credit_weak",
                "claim": "Post-swap teaching did not produce B preference; credit to the selected motor vector is insufficient or mis-linked.",
                "evidence": {"later": swap["later_counts"], "learned": swap["counts_learned"]},
            }
        )

    authorized = [
        "exchangeable_motor_slots",
        "unit_motor_vectors",
        "tiebreak_independent_of_bind_order",
        "exact_credit_to_selected_motor_vector_snapshot",
        "no_consequence_neutrality",
        "frozen_immediate_swap_probe_no_new_consequence",
        "evidence_driven_post_swap_revision",
        "per_organism_motor_rng",
    ]
    summary = {
        "version": "TM.0.23.CORTEX.V5.DIAGNOSIS",
        "lab": "TM.0.23.CORTEX.V5.DIAG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_mechanism_changed": False,
        "candidate_v5_sha": _sha_file(CANDIDATE_V5),
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "asymmetry": asym,
        "swap_timing": swap,
        "ranked_root_causes": causes,
        "v6_authorized_only_if_this_lock": True,
        "v6_boundary_must_require": authorized,
        "refuse": [
            "edit-and-rescore v5 on revealed gate worlds",
            "open DEVELOP.v5",
            "open DEVELOP.v6 before v6 D1–D2 ≥13/16",
            "soften D1/D2 thresholds",
        ],
        "note": "Observational. Authorizes isolated v6 apparatus; does not implement v6.",
    }
    if write_lock:
        if DIAG_LOCK.exists():
            raise RuntimeError("cortex_diagnosis.v5.lock exists — refuse rewrite")
        DIAG_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        lines = [
            "# TM.0.23.CORTEX v5 diagnosis",
            "",
            "Observational on frozen candidate v5. Does not rescore revealed gate worlds.",
            "",
            "## No-consequence asymmetry",
            "",
            f"- motor seeds unique: `{asym['motor_seeds_unique']}` ({asym['n_motor_seeds']} seeds / {asym['n_organisms']} organisms)",
            f"- vector collisions: `{asym['vector_collisions']}`",
            f"- slot-0 birth ACT-query argmax: `{asym['slot0_birth_argmax']}/{asym['n_organisms']}` "
            f"({asym['slot0_birth_argmax_rate']:.3f})",
            f"- motor vector ‖v‖ mean/std: `{asym['vec_norm_mean']:.3f}` / `{asym['vec_norm_std']:.3f}`",
            "",
            "## Swap timing",
            "",
            f"- learned A: `{swap['pref_a_before_swap']}` counts `{swap['counts_learned']}`",
            f"- immediate frozen first token: `{swap['immediate_first_token']}` (is A: `{swap['immediate_first_is_A']}`)",
            f"- immediate frozen 8-probe: `{swap['immediate_frozen_counts']}`",
            f"- contaminated 20-probe (old C4): `{swap['immediate_contaminated_counts']}`",
            f"- later after 40 swap episodes: `{swap['later_counts']}` pref B `{swap['pref_b_after_teach']}`",
            f"- leak into first selection: `{swap['physics_or_future_leak_into_first_selection']}`",
            f"- stale window contaminated: `{swap['stale_probe_contaminated_by_inprobe_consequences']}`",
            f"- credit fails to revise: `{swap['credit_assignment_fails_to_revise']}`",
            "",
            "## Ranked causes",
            "",
        ]
        for i, c in enumerate(causes, 1):
            lines.append(f"{i}. **{c['id']}** — {c['claim']}")
        lines += [
            "",
            "V6 authorized only by this lock. Full D0–D12 stays closed until a fresh v6 D1–D2 gate ≥13/16.",
            "",
        ]
        DIAG_MD.write_text("\n".join(lines), encoding="utf-8")
        summary["locks_written"] = True
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_diagnosis(write_lock=args.write_lock), indent=2, default=str))
