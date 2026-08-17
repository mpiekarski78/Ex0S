"""TM.0.24.LINEAGE Phase A — canonical θ slice layout. No evolution engine.

Writes docs/lineage_genome_layout.json. Topology is fixed n=64; genes do not
include tokens, answers, S, actuator handles, world IDs or stage IDs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "lineage_genome_layout.json"

N = 64
D_SYM = 32
K_S = 8
D_BODY = 4
D_X = D_SYM + K_S * D_SYM + D_BODY + 1  # 293
N_OP = 6
OPS = ["RETRIEVE", "WRITE", "EMIT", "ACT", "STOP", "HOLD"]
AGE_STAGES = [
    "birth",
    "high_plasticity",
    "experience_replay",
    "pruning_stabilization",
    "mature_plasticity",
    "novelty_reopen",
]

PROHIBITED = [
    "token_strings",
    "english",
    "actuator_handles",
    "world_ids",
    "domain_ids",
    "stage_ids",
    "expected_answers",
    "target_output_lengths",
    "response_sequences",
    "scorer_thresholds",
    "S_rows",
    "phrase_buffers",
    "capability_switches",
    "life_rng_seeds",
]


def _scalar(name: str, default: float, lo: float, hi: float, unit: str = "linear") -> dict[str, Any]:
    return {
        "name": name,
        "size": 1,
        "kind": "scalar",
        "unit": unit,
        "default": float(default),
        "lo": float(lo),
        "hi": float(hi),
    }


def arm_d_scalars() -> list[dict[str, Any]]:
    log = math.log
    items: list[dict[str, Any]] = []
    rec_logstd = log(1.0 / math.sqrt(N))
    in_logstd = log(1.0 / math.sqrt(D_X))
    n_logstd = log(1.0 / math.sqrt(N))
    body_logstd = log(1.0 / math.sqrt(D_BODY))

    for mat, mu, ls in (
        ("W_rec", 0.0, rec_logstd),
        ("W_in", 0.0, in_logstd),
        ("W_pred", 0.0, n_logstd),
        ("W_op", 0.0, n_logstd),
        ("W_emit_query", 0.0, n_logstd),
        ("W_write", 0.0, n_logstd),
        ("W_att", 0.0, n_logstd),
        ("W_body", 0.0, body_logstd),
        ("b", 0.0, -20.0),  # v27 zeros → near-zero scale
        ("v_start", 0.0, 0.0),  # std 1
        ("v_end", 0.0, 0.0),
    ):
        items.append(_scalar(f"init.{mat}.mu", mu, -2.0, 2.0))
        items.append(_scalar(f"init.{mat}.log_std", ls, -20.0, 2.0, unit="log_std"))
    # v27 W_act_query is zeros; scale 0 keeps that default
    items.append(_scalar("init.W_act_query.mu", 0.0, -2.0, 2.0))
    items.append(_scalar("init.W_act_query.log_std", -20.0, -20.0, 2.0, unit="log_std"))
    items.append(_scalar("init.b_op.ACT", 0.85, -2.0, 2.0))
    for op in OPS:
        if op != "ACT":
            items.append(_scalar(f"init.b_op.{op}", 0.0, -2.0, 2.0))

    items.append(_scalar("connect.p_connect", 0.10, 0.01, 0.50))
    items.append(_scalar("connect.growth_rate", 0.0, 0.0, 0.20))
    items.append(_scalar("connect.prune_rate", 0.0, 0.0, 0.20))
    items.append(_scalar("connect.prune_threshold", 0.0, 0.0, 2.0))
    for r in range(4):
        items.append(_scalar(f"connect.region_{r}_bias", 0.0, -2.0, 2.0))

    items.extend(
        [
            _scalar("dyn.eta_pred", 0.05, 1e-4, 0.5),
            _scalar("dyn.eta_act", 0.15, 1e-4, 0.5),
            _scalar("dyn.beta", 0.01, 0.0, 0.2),
            _scalar("dyn.clip", 2.0, 0.1, 8.0),
            _scalar("dyn.tau", 1.0, 0.1, 8.0),
            _scalar("dyn.t_max", 8.0, 1.0, 16.0),
            _scalar("dyn.cos_thresh", 0.15, 0.01, 0.9),
            _scalar("dyn.familiarity_ratio", 0.5, 0.0, 2.0),
            _scalar("dyn.familiarity_decay", 0.98, 0.5, 0.999),
            _scalar("dyn.familiarity_abs", 16.0, 1.0, 64.0),
            _scalar("dyn.echoic_max", 8.0, 1.0, 16.0),
            _scalar("dyn.echoic_bias", 0.08, 0.0, 1.0),
            _scalar("dyn.vocal_refractory", 1.5, 0.0, 8.0),
            _scalar("dyn.utterance_persist", 1.5, 0.0, 8.0),
            _scalar("dyn.conflict_hold_bias", 2.0, 0.0, 8.0),
            _scalar("dyn.adv_baseline_alpha", 0.05, 0.0, 1.0),
            _scalar("dyn.equal_evidence_min_symbols", 3.0, 1.0, 8.0),
            _scalar("dyn.explore_T0", 1.0, 0.05, 8.0),
            _scalar("dyn.explore_decay", 1.0, 0.5, 1.0),
            _scalar("dyn.eligibility_decay", 1.0, 0.5, 1.0),
            _scalar("dyn.retrieval_thresh", 0.0, -2.0, 2.0),
            _scalar("dyn.write_thresh", 0.0, -2.0, 2.0),
        ]
    )
    costs = {"RETRIEVE": 1.0, "WRITE": 1.0, "EMIT": 1.0, "ACT": 0.05, "STOP": 0.0, "HOLD": 0.0}
    for op, c in costs.items():
        items.append(_scalar(f"dyn.op_cost.{op}", c, 0.0, 4.0))
    for i, v in enumerate([1.0, 0.0, 1.0, 0.0]):
        items.append(_scalar(f"dyn.body_setpoint.{i}", v, 0.0, 1.0))

    for name in ("recency", "similarity", "surprise", "random"):
        items.append(_scalar(f"replay.mix.{name}", 0.25, 0.0, 1.0))
    for name in ("homeostasis", "surprise", "conflict", "novelty", "controllability", "social"):
        items.append(_scalar(f"neuromod.gain.{name}", 1.0, 0.0, 4.0))

    for st in AGE_STAGES:
        for p, d, lo, hi in (
            ("eta_pred_scale", 1.0, 0.0, 4.0),
            ("eta_act_scale", 1.0, 0.0, 4.0),
            ("beta_scale", 1.0, 0.0, 4.0),
            ("explore_T", 1.0, 0.05, 8.0),
            ("conflict_hold_scale", 1.0, 0.0, 4.0),
            ("growth_scale", 0.0, 0.0, 1.0),
            ("prune_scale", 0.0, 0.0, 1.0),
            ("wm_persist", 1.5, 0.0, 8.0),
            ("refractory", 1.5, 0.0, 8.0),
        ):
            items.append(_scalar(f"age.{st}.{p}", d, lo, hi))
    return items


def assign_offsets(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    off = 0
    out = []
    for it in items:
        row = dict(it)
        row["offset"] = off
        out.append(row)
        off += int(row["size"])
    return out, off


def arm_c_tensors() -> list[dict[str, Any]]:
    mats = [
        ("W_rec", N * N, [N, N]),
        ("M", N * N, [N, N]),
        ("W_in", N * D_X, [N, D_X]),
        ("W_pred", D_SYM * N, [D_SYM, N]),
        ("W_op", N_OP * N, [N_OP, N]),
        ("W_emit_query", D_SYM * N, [D_SYM, N]),
        ("W_act_query", D_SYM * N, [D_SYM, N]),
        ("W_write", D_SYM * N, [D_SYM, N]),
        ("W_att", D_SYM * N, [D_SYM, N]),
        ("W_body", N * D_BODY, [N, D_BODY]),
        ("b", N, [N]),
        ("b_op", N_OP, [N_OP]),
        ("v_start", D_SYM, [D_SYM]),
        ("v_end", D_SYM, [D_SYM]),
    ]
    items = []
    for name, size, shape in mats:
        items.append(
            {
                "name": f"tensor.{name}",
                "size": size,
                "kind": "tensor",
                "shape": shape,
                "unit": "linear",
                "default": "v27_birth_sample",
                "lo": -8.0,
                "hi": 8.0,
            }
        )
    return items


def build() -> dict[str, Any]:
    d_items, d_dim = assign_offsets(arm_d_scalars())
    c_tens, c_t_dim = assign_offsets(arm_c_tensors())
    # Arm C also carries the same developmental scalars after tensors
    c_dyn, c_dyn_dim = assign_offsets(arm_d_scalars())
    for row in c_dyn:
        row["offset"] = int(row["offset"]) + c_t_dim
    c_dim = c_t_dim + c_dyn_dim
    return {
        "version": "TM.0.24.LINEAGE.GENOME.LAYOUT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "topology_fixed": {
            "n": N,
            "d_sym": D_SYM,
            "k_s": K_S,
            "d_body": D_BODY,
            "d_x": D_X,
            "n_op": N_OP,
            "ops": OPS,
            "dtype": "float64",
            "note": "Neural topology is fixed. World 'topology' means relation graph only.",
        },
        "prohibited_genome_contents": PROHIBITED,
        "arms": {
            "D": {
                "name": "developmental",
                "role": "main_biological_lineage",
                "dim": d_dim,
                "slices": d_items,
                "birth_sampling": "life_specific_developmental_RNG",
                "siblings_not_clones": True,
            },
            "C": {
                "name": "dense_theta",
                "role": "engineering_control",
                "dim": c_dim,
                "tensor_dim": c_t_dim,
                "dynamics_dim": c_dyn_dim,
                "slices": c_tens + c_dyn,
                "note": "Success here does not support the strongest developmental claim and cannot set eligible_for_000005.",
            },
        },
        "refuse_audit": [
            "decoded_theta_contains_no_utf8_tokens",
            "no_S_rows",
            "no_stage_or_world_ids",
            "all_scalars_within_preregistered_bounds",
        ],
    }


def main() -> None:
    layout = build()
    OUT.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print(f"Arm D dim={layout['arms']['D']['dim']} Arm C dim={layout['arms']['C']['dim']} -> {OUT}")


if __name__ == "__main__":
    main()
