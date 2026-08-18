# TM.0.43.MECHCORR contract

**Lab:** TM.0.43.MECHCORR · **Not a v40 candidate.**

TM042 first-match `postinstall_mech_install_fail` and the 12-cell wall remain immutable. This wall repairs only the targeting mistake: reconstruct the exact TM039 `reg1` pair (seed `1584000025`, both orders), not regs 2/3.

## Setup preconditions (before continuity scoring)

1. `violations_after_v37 > 0`
2. `fallback_invoked = true`
3. `solver_installed = true`

Failure is `setup_precondition_fail`, not organism failure. Reconstruct is further pinned by TM039 cell IDs and W hashes.

## Continue gates

Identical to TM042 `_continue_after_install`. Do not rerun natural cells.

## Candidate review

Even on pass, do not write `cortex.candidate.v40.lock`. Open only a separate candidate review of whether pinned global SOCP consolidation is accepted as part of the Ex0S organism. Frozen TM042 `candidate_discussion_open` stays false.
