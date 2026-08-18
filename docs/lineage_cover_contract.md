# TM.0.53.COVER contract

**Lab:** TM.0.53.COVER · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. Do not install W\*. Do not extend v40 SOCP. Do not select a fitted N as an organism constant.

TM052 first-match `shared_W_star_satisfies` stays frozen. The addendum reading stays frozen: sampled shared-decoder capacity is available; that does not license installing SOCP for grounding. Train-only W\* still scored held-out wraps **1/4**. Full-oracle W\* contains future test states as constraints and remains diagnostic only.

## Question

Does broader generic experience produce generalization, or does the feedback representation lack context-invariant action structure?

Collect balanced wrapped feedback examples from N ∈ {1,2,4,8,16,32} independent development contexts per action. Always include the four reference-action constraints. Solve W\*_N from those development examples only. Evaluate untouched feedback contexts and ordinary references. Discard every W\*_N. Use multiple registry seeds and fresh context families.

## Load-bearing rules

- Held-out contexts are never SOCP constraints.
- Parent `W_act_query` is the reference-only decoder. Plasticity is frozen before wrap collection.
- `n_dev_repeats` stays 4. N is a diagnostic coverage index, not an organism constant.
- Existing joint-SOCP geometry and organism predicate only.

## Record

Training accuracy; untouched-context accuracy; reference retention; within-action vs between-action geometry; movement from parent W; geometric margins.

## Ladder (setup excluded)

`setup_precondition_fail` → `coverage_infeasible` → `reference_interference` → `no_transfer` → `seed_dependent` → `coverage_generalizes`

| Result | Conclusion |
| --- | --- |
| Held-out becomes consistently perfect as coverage grows | Generic consolidation from experienced grounding examples is plausible |
| Training passes while held-out stays collapsed | Representation lacks context-invariant action structure; an alignment/representation mechanism is needed |
| Reference behavior breaks | Shared-decoder interference remains despite feasibility |
| Strong seed dependence | No architecture claim yet |

## Refuse

Install W\*, extend v40 SOCP, increase `n_dev_repeats`, fit N as an organism constant, second decoder, K/Q/V, rewrite TM052 DEV, v41, product-earn.
