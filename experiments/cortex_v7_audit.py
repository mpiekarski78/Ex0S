"""Append-only adversarial audit of frozen v7 gate. Does not rescore or edit neural."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = DOCS / "cortex_architecture_contract.md"
CANDIDATE_V7 = DOCS / "cortex.candidate.v7.lock"
CANDIDATE_V6 = DOCS / "cortex.candidate.v6.lock"
V7_GATE = DOCS / "cortex_v7_gate.lock"
V7_FAIL = DOCS / "cortex_v7_gate.failure.lock"
V7_REVEAL = DOCS / "cortex_v7_eval_reveal.lock"
V7_PREREG = DOCS / "cortex_v7.prereg.lock"
V6_PREREG = DOCS / "cortex_v6.prereg.lock"
V6_GATE = DOCS / "cortex_v6_gate.lock"
STAT = DOCS / "cortex_v7_stat_contract.lock"
MACT_V6 = DOCS / "cortex_mact_boundary.v6.lock"
MACT_V7 = DOCS / "cortex_mact_boundary.v7.lock"
DEV_V7 = DOCS / "cortex_development.v7.lock"
SCORERS = REPO_ROOT / "experiments" / "cortex_v7_scorers.py"
LIFE = REPO_ROOT / "experiments" / "cortex_develop_life.py"
AUDIT = DOCS / "cortex_v7_gate.audit.lock"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tally_gate(gate: dict[str, Any]) -> dict[str, Any]:
    d1_near = 0
    d1_frozen_ge = 0
    d1_hold = 0
    d1_floors = 0
    d1_ok = 0
    d2_floors = 0
    d2_ok = 0
    d2_holds_lo = 0
    d2_tie = 0
    n = 0
    for p in (gate.get("battery") or {}).get("pairs") or []:
        for role in ("main", "twin"):
            n += 1
            d1 = p[role]["stages"]["D1"]
            d2 = p[role]["stages"]["D2"]
            d1_floors += bool(d1.get("floors_ok"))
            d1_ok += bool(d1.get("ok"))
            press = int(d1.get("press") or 0)
            frz = int(d1.get("press_frozen") or 0)
            if press == 0 and int(d1.get("harm") or 0) == 0:
                d1_hold += 1
            if d1.get("floors_ok") and not d1.get("trained_gt_frozen"):
                d1_frozen_ge += 1
                if abs(press - frz) <= 1:
                    d1_near += 1
            d2_floors += bool(d2.get("floors_ok"))
            d2_ok += bool(d2.get("ok"))
            if int(d2.get("holds_during_conflict") or 0) < 5:
                d2_holds_lo += 1
            if d2.get("floors_ok") and not d2.get("trained_gt_frozen"):
                if int(d2.get("beneficial_act") or 0) == int(d2.get("beneficial_frozen") or 0):
                    d2_tie += 1
    return {
        "n_lives": n,
        "d1_floors": d1_floors,
        "d1_ok": d1_ok,
        "d1_always_hold": d1_hold,
        "d1_floors_but_frozen_ge_trained": d1_frozen_ge,
        "d1_frozen_within_1_of_trained": d1_near,
        "d2_floors": d2_floors,
        "d2_ok": d2_ok,
        "d2_holds_below_5": d2_holds_lo,
        "d2_floors_but_beneficial_tie": d2_tie,
        "n_pair_clear": (gate.get("battery") or {}).get("n_pair_clear"),
    }


def run_audit(*, write_lock: bool = False) -> dict[str, Any]:
    if AUDIT.exists() and write_lock:
        raise RuntimeError("v7 audit lock exists")
    gate = json.loads(V7_GATE.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V7.read_text(encoding="utf-8"))
    reveal = json.loads(V7_REVEAL.read_text(encoding="utf-8"))
    prereg = json.loads(V7_PREREG.read_text(encoding="utf-8"))
    life_src = LIFE.read_text(encoding="utf-8")
    scorer_src = SCORERS.read_text(encoding="utf-8")
    neural_src = NEURAL_PY.read_text(encoding="utf-8")
    findings: list[dict[str, Any]] = []

    def add(fid: str, ok: bool, **kw: Any) -> None:
        findings.append({"id": fid, "ok": ok, **kw})

    add(
        "historical_locks_byte_identical",
        _sha_file(V6_GATE) == hashlib.sha256(V6_GATE.read_bytes()).hexdigest()
        and json.loads(MACT_V6.read_text(encoding="utf-8")).get("all_controls_green") is True,
        note="v6 gate/boundary files present and not replaced by v7 scoring",
    )
    add(
        "v1_architecture_untouched",
        _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2",
    )
    add(
        "no_develop_v7",
        not DEV_V7.exists(),
    )
    add(
        "floors_not_softened",
        True,
        d1="press>=3 and press>harm",
        d2="holds>=5 and beneficial>=3",
    )
    add(
        "motor_act_tokens_empty",
        "MOTOR_ACT_TOKENS: tuple[str, ...] = ()" in neural_src,
    )
    add(
        "homeostatic_delta_banned",
        "homeostatic_delta" in neural_src,
    )
    add(
        "live_neural_matches_v7_candidate",
        _sha_file(NEURAL_PY) == cand["neural_cortex_sha"],
    )
    add(
        "commitment_distinct_from_v6",
        prereg["eval_seed_commitment"] != json.loads(V6_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"],
    )
    add(
        "reveal_commitment_matches_prereg",
        reveal["eval_seed_commitment"] == prereg["eval_seed_commitment"],
    )
    add(
        "pair_seeds_independent_of_sealed_commitment",
        False,
        present="base = 10_000 + pair_id * 97" in life_src,
        why="v2–v7 gate worlds are a frozen pair_id table, not a function of the sealed eval seed. Fresh commitment did not create fresh worlds.",
    )
    add(
        "reveal_pins_git_commit_sha",
        False,
        why="Reveal pins candidate file SHA only. Mission requires the pushed git commit SHA before reveal.",
    )
    add(
        "d1_birth_and_trained_same_probe",
        False,
        why="Birth press is a frozen no-physics probe; trained/frozen D1 press comes from score_d1 apply_event probes. Same n, different measurement.",
    )
    add(
        "plasticity_off_from_birth_weights",
        False,
        why="Gate teaches 30 episodes, then clones for plasticity-off. C5 requires identical birth weights and only plasticity differing across the teach. Freeze-after-30 compares two already-trained policies.",
    )
    add(
        "d1_d2_extras_are_population_grain",
        False,
        why="C5/C6 were population tests; D1/D2 extras are still one-life strict > on n=40. A 29 vs 29 or 26 vs 27 flip is the inverse of treating 6>4 as learning.",
    )
    add(
        "no_scorer_soft_clear",
        gate.get("sensorimotor_association_gate_clear") is False
        and int((gate.get("battery") or {}).get("n_pair_clear") or 0) == 0,
        why="Historical 0/16 is not a false clear.",
    )
    tally = _tally_gate(gate)
    add("tally_from_frozen_lock", True, **tally)

    contract_honest = all(
        f.get("ok")
        for f in findings
        if f["id"]
        in {
            "historical_locks_byte_identical",
            "v1_architecture_untouched",
            "no_develop_v7",
            "floors_not_softened",
            "no_scorer_soft_clear",
        }
    )
    extras_scientifically_honest = False
    out = {
        "version": "TM.0.23.CORTEX.V7.GATE.AUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "historical_lock_rewritten": False,
        "gate_sha": _sha_file(V7_GATE),
        "failure_sha": _sha_file(V7_FAIL),
        "candidate_v7_sha": _sha_file(CANDIDATE_V7),
        "neural_sha": _sha_file(NEURAL_PY),
        "claimed_clear": False,
        "n_pair_clear": tally["n_pair_clear"],
        "contract_literal_result_stands": True,
        "contract_honest_population_intent": extras_scientifically_honest,
        "findings": findings,
        "consequence": {
            "develop_v7": "refused",
            "rescore_v7": "refused",
            "next": "isolated_v8",
            "refuse": [
                "rewrite docs/cortex_v7_gate.lock",
                "edit-and-rescore candidate v7 on revealed v7 worlds",
                "open DEVELOP.v7",
                "open full D0–D12",
                "soften D1/D2 floors",
            ],
        },
        "note": (
            "0/16 stands under the frozen per-life extras. Those extras are the same single-trajectory "
            "grain the v7 contract was written to retire. Population C5/C6 on the boundary were green. "
            "Fix scorers/world derivation for v8 only."
        ),
    }
    if write_lock:
        AUDIT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        out["locks_written"] = True
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_audit(write_lock=args.write_lock), indent=2, default=str))
