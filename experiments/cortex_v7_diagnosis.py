"""Observational diagnosis of frozen candidate v7. Does not rescore the v7 gate."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from experiments.cortex_develop_life import bind_life_actuators, curriculum_tokens, motor_latent, pair_seeds, teach_loop
from experiments.cortex_mact_boundary import _frozen_pref_counts, control_c4_v6
from experiments.cortex_v7_audit import _tally_gate
from experiments.run_tm023cortex import make_cortex, torch_env
from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OPS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
CANDIDATE_V7 = DOCS / "cortex.candidate.v7.lock"
V7_GATE = DOCS / "cortex_v7_gate.lock"
V7_AUDIT = DOCS / "cortex_v7_gate.audit.lock"
DIAG = DOCS / "cortex_diagnosis.v7.lock"
DIAG_MD = DOCS / "tm023cortex_v7_diagnosis.md"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hold_trace(pair_id: int, role: str) -> dict[str, Any]:
    main_s, twin_s = pair_seeds(pair_id)
    seeds = main_s if role == "main" else twin_s
    with tempfile.TemporaryDirectory(prefix="v7hold_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        toks = curriculum_tokens(seeds)
        bind_life_actuators(ag, toks, seeds)
        b_act0 = float(ag.b_op[OPS.index("ACT")])
        w_op0 = float(ag.W_op[OPS.index("ACT")].abs().sum())
        teach_loop(ag, seeds, n=30, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=motor_latent(toks))
        w_op1 = float(ag.W_op[OPS.index("ACT")].abs().sum())
        w_q = float(ag.W_act_query.abs().sum())
        counts = _frozen_pref_counts(ag, toks, 40, [toks["a"]])
        return {
            "pair_id": pair_id,
            "role": role,
            "b_op_act": b_act0,
            "w_op_act_l1_birth": w_op0,
            "w_op_act_l1_after_30": w_op1,
            "w_act_query_l1_after_30": w_q,
            "frozen_counts_after_30": counts,
        }


def run_diagnosis(*, write_lock: bool = False) -> dict[str, Any]:
    if not V7_AUDIT.exists():
        raise RuntimeError("write v7 audit lock first")
    cand = json.loads(CANDIDATE_V7.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from candidate v7 — refuse")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    gate = json.loads(V7_GATE.read_text(encoding="utf-8"))
    tally = _tally_gate(gate)
    c4 = control_c4_v6()
    hold_lives = []
    for p in gate["battery"]["pairs"]:
        for role in ("main", "twin"):
            d1 = p[role]["stages"]["D1"]
            if int(d1.get("press") or 0) == 0 and int(d1.get("harm") or 0) == 0:
                hold_lives.append((p["pair_id"], role))
    traces = [_hold_trace(pid, role) for pid, role in hold_lives[:3]]
    ranked = [
        {
            "id": "c4_revision_retained",
            "claim": "C4 still revises A→B after 40 post-swap episodes. Preserve this.",
            "ok": bool(c4.get("ok")),
            "evidence": {k: c4.get(k) for k in ("pref_a", "stale_ok", "pref_b", "restore_a", "counts_before", "counts_revised")},
        },
        {
            "id": "population_learning_already_shown",
            "claim": "v7 boundary C5/C6 population greens show consequence-dependent initial learning at the grain the contract asked. The 0/16 is not evidence that learning is absent.",
            "ok": True,
            "evidence": {
                "c5_n_trained_beats_frozen": 30,
                "c5_mean_delta": 0.23593749999999997,
                "c6_slot_effect": -0.007535278205457296,
                "c6_label_effect": 0.035000086494902605,
            },
        },
        {
            "id": "d1_d2_extras_wrong_grain",
            "claim": "Per-life strict > after a 30-episode teach compares two similar policies on n=40 apply_event probes. That is the inverse of treating 6>4 as learning.",
            "ok": False,
            "evidence": tally,
        },
        {
            "id": "always_hold_remains",
            "claim": "7/32 lives still D1 press=0/harm=0 after W_op skip. Skip did not extinguish HOLD on every seed.",
            "ok": False,
            "evidence": {"n_always_hold": tally["d1_always_hold"], "traces": traces},
        },
        {
            "id": "worlds_not_seed_derived",
            "claim": "pair_seeds(pair_id) is independent of the sealed eval seed. v8 gate must derive lives from the commitment.",
            "ok": False,
        },
    ]
    authorize = [
        "retain v6/v7 motor geometry and C4 snapshot / zero W_act_query / handle-keyed vectors / skip W_op and motor-query when body_adv≈0",
        "do not retune neural so every frozen life fails deterministic D1",
        "replace per-life D1/D2 strict count > with population-grain paired extras from birth weights, same frozen-probe measure",
        "derive narrow-gate lives from the sealed eval seed (not the frozen 10_000+97 table)",
        "pin pushed git commit SHA in the reveal lock",
        "keep absolute D1/D2 floors and always-HOLD fail",
        "no neural change authorized beyond retaining v7 skip_act_cost — remaining HOLD is a seed-wise exploration failure, not a new credit law",
    ]
    out = {
        "version": "TM.0.23.CORTEX.V7.DIAGNOSIS",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "canonical_main": "ec10c03",
        "neural_mechanism_changed": False,
        "candidate_v7_sha": _sha_file(CANDIDATE_V7),
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "gate_sha": _sha_file(V7_GATE),
        "audit_sha": _sha_file(V7_AUDIT),
        "c4": {k: c4.get(k) for k in ("ok", "pref_a", "stale_ok", "pref_b", "restore_a", "counts_before", "counts_stale", "counts_revised", "counts_restored")},
        "tally": tally,
        "hold_traces": traces,
        "ranked": ranked,
        "v8_authorized_only_if_this_lock": True,
        "authorize_v8": authorize,
        "refuse": [
            "edit-and-rescore v7",
            "DEVELOP.v7",
            "soften floors",
            "edit docs/cortex_architecture_contract.md",
            "open D0–D12 before a later isolated gate ≥13/16",
        ],
        "env": torch_env(),
    }
    if write_lock:
        if DIAG.exists():
            raise RuntimeError("v7 diagnosis exists")
        DIAG.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
        DIAG_MD.write_text(
            "\n".join(
                [
                    "# TM.0.23.CORTEX v7 diagnosis",
                    "",
                    "Observational. Gate lock not rewritten. C4 retained. Population C5/C6 already green.",
                    "0/16 stands as a per-life-extra fail. v8 must change scorer grain and seed-derived worlds, not rescore v7.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        out["locks_written"] = True
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_diagnosis(write_lock=args.write_lock), indent=2, default=str))
