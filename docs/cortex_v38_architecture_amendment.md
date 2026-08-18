# CORTEX v38 architecture amendment

Authorized by [`lineage_restsplit.decision.lock`](lineage_restsplit.decision.lock). TM032 / TM031 historical locks are **not** edited. TM032 is **not** rerun.

TM032 showed: extra gated replay updates suffice; REST idle dynamics are unnecessary; slow mixing alone is insufficient; full REST succeeds because it includes the same effective replay. Routing evidence is **two informative splits, both `reg1`**. Do **not** install the observed 44 row-updates as a fitted constant. A failure at the pre-TM032 16-pass REST allowance remains a failure.

## Controller (treatment `adaptive_violation`)

After credit (the one-shot ranking-error ACT update is **not** rehearsal debt):

1. Evaluate stored-row violations.
2. Rehearse violating rows (`_gated_rehearsal_pass`: only rows with `_episode_rehearsal_violation`).
3. Re-evaluate.
4. Stop at zero, plateau, or hard budget.

### Plateau (deterministic)

- **Signature:** `(n_violations, violating_slots)` where `violating_slots` is the sorted tuple of valid slot indices with `_episode_rehearsal_violation(p1, handle, adv) == True` (ranking error or geometric margin `< ACT_MARGIN_FLOOR`).
- **Improvement measure:** `n_violations_before - n_violations_after` (integer count). Slot identity is recorded; it does not define improvement.
- **Tolerance:** 0 (strict integer; no epsilon).
- **Unchanged passes to stop:** 1. A pass that does not decrease `n_violations` by at least 1 is a plateau, including `n_updates == 0` while `n_violations > 0`.

### Hard budget

`EPISODE_REPLAY_EPOCHS` = **16 passes**. Record pass count **and** actual rehearsal-update calls (`_apply_act_query_update` from gated rehearsal). Work comparisons use **updates**.

### Replay debt

- **Units recorded:** `rehearsal_pass_debt` (attempted gated passes) and `rehearsal_update_debt` (actual rehearsal-update calls).
- **REST debit uses passes:** next `_replay_episodes` pass budget is `max(0, 16 - rehearsal_pass_debt)`.
- **Carry:** both debts accumulate across credits until REST.
- **Reset:** both set to 0 after that REST replay finishes (including a zero-budget REST). `reset_cortex` also zeros them.
- **Checkpoint:** serialize both debts and the active rehearse arm. Missing keys load as 0 / `v37_awake_cap`.

## `fixed_extra_replay` (exact)

Same **targeting** as adaptive: violating stored rows only. Same 16-pass cap. **Stops at zero or budget; no plateau stop.** **No REST debit.** Because targeting is matched, “only adaptive passes” supports the **adaptive controller** (plateau stop + debit), not row-prioritization versus all-rows. If targeting ever diverges, that triple is **controller bundle**, not adaptive control.

`v37_awake_cap` is the frozen skip-just-written burst; no debt.

## Interpretation

| Result | Claim |
|--------|--------|
| Both extra-replay arms pass; adaptive uses strictly fewer **updates** | efficient scheduling |
| Only adaptive passes; targeting matched | adaptive control |
| Targeting/stopping/compute all differ | controller bundle, not prioritization alone |
| Neither extra-replay arm passes | 16-pass REST allowance insufficient or update-rule wall |

Zero v37 acquire fails → no informative triples. Mixed seeds stay mixed. Budget exhaustion is recorded; it does not promote 44. Leftover debt after the scheduled REST is an integrity fail. Product **0.0.004**.
