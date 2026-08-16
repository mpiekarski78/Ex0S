"""Observational diagnosis of frozen candidate v6. Does not rescore the v6 gate lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.cortex_mact_boundary import control_c4_v6
from experiments.cortex_v7_stats import run_c5_population, run_c6_population, summarize_v6_gate_failures
from experiments.run_tm023cortex import torch_env
from three_memory.neural_cortex import MOTOR_ACT_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
CANDIDATE_V6 = DOCS / "cortex.candidate.v6.lock"
V6_GATE = DOCS / "cortex_v6_gate.lock"
STAT_LOCK = DOCS / "cortex_v7_stat_contract.lock"
DIAG_LOCK = DOCS / "cortex_diagnosis.v6.lock"
DIAG_MD = DOCS / "tm023cortex_v6_population_diagnosis.md"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_diagnosis(*, write_lock: bool = False) -> dict[str, Any]:
    if not STAT_LOCK.exists():
        raise RuntimeError("freeze v7 stat contract first")
    cand = json.loads(CANDIDATE_V6.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from candidate v6 — refuse")
    if MOTOR_ACT_TOKENS:
        raise RuntimeError("MOTOR_ACT_TOKENS must stay empty")
    gate = json.loads(V6_GATE.read_text(encoding="utf-8"))
    c4 = control_c4_v6()
    c5 = run_c5_population()
    c6 = run_c6_population()
    fail_modes = summarize_v6_gate_failures(gate)
    ranked = []
    if not c5["ok"]:
        ranked.append(
            {
                "id": "single_trajectory_d1_is_not_learning",
                "claim": "A frozen or chance ACT stream can look D1-shaped; C5 must be trained≫frozen at population level.",
                "evidence": {
                    "n_trained_beats_frozen": c5["n_trained_beats_frozen"],
                    "mean_delta": c5["mean_delta"],
                    "frozen_mean_p_press": c5["frozen_mean_p_press"],
                },
            }
        )
    if not c6["ok"]:
        ranked.append(
            {
                "id": "no_consequence_label_or_slot_effect",
                "claim": "Under neutral physics, slot or post-hoc beneficial labels still move aggregate preference.",
                "evidence": {
                    "slot_effect": c6["slot_effect"],
                    "label_effect": c6["label_effect"],
                    "slot_perm_p": c6["slot_perm_p"],
                    "label_perm_p": c6["label_perm_p"],
                },
            }
        )
    ranked.append(
        {
            "id": "gate_failures_are_hold_or_conflict_not_8_0_bias",
            "claim": "Revealed v6 pair reds are mostly D1 press=0/harm=0 or D2 holds<5, not a transferred 8–0 bind-order win.",
            "evidence": fail_modes,
        }
    )
    if c4.get("ok"):
        ranked.append(
            {
                "id": "c4_revision_retained",
                "claim": "Frozen swap probe stays A then revises to B after 40 episodes. Preserve this.",
                "evidence": {
                    "stale_ok": c4.get("stale_ok"),
                    "pref_b": c4.get("pref_b"),
                    "restore_a": c4.get("restore_a"),
                },
            }
        )
    out = {
        "version": "TM.0.23.CORTEX.V6.POPULATION.DIAGNOSIS",
        "lab": "TM.0.23.CORTEX.V6.DIAG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "canonical_baseline": "97691cd",
        "neural_mechanism_changed": False,
        "candidate_v6_sha": _sha_file(CANDIDATE_V6),
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "stat_contract_sha": _sha_file(STAT_LOCK),
        "c4": {k: c4.get(k) for k in ("ok", "pref_a", "stale_ok", "pref_b", "restore_a", "counts_before", "counts_stale", "counts_revised", "counts_restored")},
        "c5_population": {k: c5[k] for k in c5 if k != "rows"},
        "c6_population": {k: c6[k] for k in c6 if k != "rows"},
        "c5_rows": c5["rows"],
        "c6_rows": c6["rows"],
        "v6_gate_failure_modes": fail_modes,
        "ranked_root_causes": ranked,
        "v7_authorized_only_if_this_lock": True,
        "v7_must": [
            "preserve_c4_revision",
            "population_c5_plasticity_necessity",
            "population_c6_neutrality",
            "d1_d2_trained_gt_birth_and_frozen",
            "d2_consequence_association",
            "do_not_force_every_frozen_life_to_fail_deterministic_d1",
        ],
        "refuse": [
            "edit-and-rescore v6 on revealed gate worlds",
            "open DEVELOP.v6",
            "soften D1/D2 floors",
            "open DEVELOP.v7 before v7 D1–D2 ≥13/16",
        ],
        "env": torch_env(),
        "note": "Observational on frozen v6. Authorizes isolated v7 apparatus; does not implement v7.",
    }
    if write_lock:
        if DIAG_LOCK.exists():
            raise RuntimeError("diagnosis.v6 exists")
        DIAG_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
        DIAG_MD.write_text(
            "\n".join(
                [
                    "# TM.0.23.CORTEX v6 population diagnosis",
                    "",
                    "Observational on frozen candidate v6. Does not rewrite the v6 gate lock.",
                    "",
                    f"**C4 revision:** `{c4.get('ok')}` stale_ok=`{c4.get('stale_ok')}` pref_b=`{c4.get('pref_b')}`",
                    f"**C5 population:** beats_frozen `{c5['n_trained_beats_frozen']}/{c5['n_pairs']}` mean_delta=`{c5['mean_delta']:.4f}` frozen_p_press=`{c5['frozen_mean_p_press']:.4f}`",
                    f"**C6 population:** slot_effect=`{c6['slot_effect']:.4f}` label_effect=`{c6['label_effect']:.4f}`",
                    f"**Gate failure modes:** `{fail_modes}`",
                    "",
                    "V7 authorized only by this lock plus the frozen stat contract.",
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
