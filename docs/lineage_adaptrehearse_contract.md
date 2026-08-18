# TM.0.33.ADAPTREHEARSE contract

**Lab:** TM.0.33.ADAPTREHEARSE · **Organism:** v38 (neural after this freeze)

Fresh `TM033.ADAPTREHEARSE.DEV./TWIN.` worlds. Do **not** edit TM032 locks or rerun TM032. Do **not** add `early_raw_half_spacing` to `ACT_RECALL_MODES`. Pre-final SHA `30800288…` is superseded by this freeze.

## Limitation (explicit)

TM032 routed v38 from **two informative splits, both `reg1`**. Six other splits were already post-awake converged. 44 row-updates is an observation, **not** a fitted law. Failure at 16 passes stays a failure.

## Plateau (deterministic)

Signature `(n_violations, sorted violating slot indices)` from `_episode_rehearsal_violation`. Improvement: `n_violations` drops by ≥1 (tolerance 0). **One** non-improving pass stops adaptive, including `n_updates==0` with remaining violations.

## Debt

Record **passes** and **actual rehearsal-update calls**. REST debit uses passes: `max(0, 16 - rehearsal_pass_debt)`. Debt accumulates across credits, resets after REST (including zero-budget REST), and is checkpointed.

## `fixed_extra_replay`

Same violation-row targeting as adaptive. Stop at zero or 16 passes; **no plateau**. **No REST debit**. Targeting is matched, so only-adaptive supports **adaptive control** (plateau + debit), not prioritization versus all-rows. Targeting mismatch → **controller bundle**.

## Work metric

Compare arms on rehearsal-update calls. Still cap the controller at 16 passes.

## Frozen outcomes

| Code | When |
|------|------|
| `adaptrehearse_no_v37_acquire_fail` | zero informative triples |
| `adaptrehearse_debt_integrity_fail` | adaptive debt remains after scheduled REST |
| `adaptrehearse_efficient_scheduling` | both extra-replay arms pass; adaptive fewer updates |
| `adaptrehearse_adaptive_control` | only adaptive passes; targeting matched |
| `adaptrehearse_controller_bundle` | targeting mismatch, or several mechanisms confounded |
| `adaptrehearse_extra_compute_not_scheduler` | only fixed extra passes |
| `adaptrehearse_both_pass_compute_tied` | both pass; adaptive not strictly cheaper |
| `adaptrehearse_core_acquire_fail` | neither extra-replay arm passes (includes 16-pass exhaustion) |
| `adaptrehearse_mixed_routes` | diagnostic triples disagree |

Product **0.0.004**.
