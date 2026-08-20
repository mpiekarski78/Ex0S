"""
R2.1 historical control reconstruction assessment.

Non-blocking. Does NOT rerun behavioral scoring on consumed R2.1 worlds.
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET_SCAFFOLD_HASH = "4ee5a698cdd07850eb4daf36446c466a6e9bbb07c4e9eee6da994dd968f72fb8"
RUN_LEDGER = "runs/exos_dev1/stage_a_r2_1/exos_dev1_r2_1b_scored_20260819/candidate_life_records.jsonl"
SEARCH_SUMMARY = "runs/exos_dev1/stage_a_r2_1/exos_dev1_r2_1b_scored_20260819/search_summary.json"


def assess_control_recovery() -> dict:
    """
    Verify whether exact scaffold recovery is possible without re-evaluation.

    The R2.1b scored ledger records per-life metrics and generation-level
    aggregates but does NOT log per-child scaffold phenotypes or selection
    indices. Reconstructing the winning scaffold therefore requires
    re-evaluating candidates on consumed training worlds — forbidden.
    """
    summary_exists = Path(SEARCH_SUMMARY).exists()
    ledger_exists = Path(RUN_LEDGER).exists()

    has_per_child_scaffold_log = False
    if ledger_exists:
        with open(RUN_LEDGER) as f:
            for line in f:
                rec = json.loads(line)
                if "continuous_scaffold" in rec or "child_index" in rec:
                    has_per_child_scaffold_log = True
                    break

    if has_per_child_scaffold_log:
        status = "replay_from_logged_decisions_possible"
        executable = True
        reason = "logged_scaffold_decisions_present"
    else:
        status = "non_executable"
        executable = False
        reason = (
            "R2.1b ledger lacks per-child scaffold phenotypes and selection "
            "indices. Recovery would require re-evaluating candidates on "
            "consumed training worlds (exos_dev1_r2_1b_scored_world_001/002). "
            "Use numerical record in dev_decision.lock; do not manufacture "
            "approximate control."
        )

    return {
        "version": "EX0S-DEV1.STAGE_A_R2_1.CONTROL.RECOVERY.ASSESSMENT",
        "status": status,
        "executable": executable,
        "reason": reason,
        "expected_scaffold_hash": TARGET_SCAFFOLD_HASH,
        "run_ledger": RUN_LEDGER if ledger_exists else None,
        "search_summary": SEARCH_SUMMARY if summary_exists else None,
        "forbidden": [
            "rerun_behavioral_scoring_on_consumed_r2_1_worlds",
            "approximate_scaffold_without_exact_hash",
            "manufacture_replacement_control",
        ],
        "fallback": "docs/exos_dev1.stage_a_r2_1.dev_decision.lock best_arm numerical record",
    }


def write_control_recovery_assessment(
    output_path: str = "docs/exos_dev1.stage_a_r2_1.control.recovery.lock",
) -> dict:
    result = assess_control_recovery()
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(write_control_recovery_assessment(), indent=2))
