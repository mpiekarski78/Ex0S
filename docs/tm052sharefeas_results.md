# TM.0.52.SHAREFEAS results

## Decision: `shared_W_star_satisfies`

Product **0.0.004**. No v41 lock. W* not installed. Existing joint SOCP only. TM051 first-match `heldout_feedback_decode_fail` is unchanged.

A shared `W_act_query` exists for the four reference development P1s, the four wrapped grounding-training P1s, and the four held-out wrapped feedback P1s. Do not add a decoder. Investigate generic consolidation for grounding. Local TM051 plasticity did not reach this nearby solution.

DEV ran once on clean `f9f267ae6a54889a3fadae25004476dc2f495b0e`. Frozen runner SHA `36c119262be5a7b2e186b22d3a5e37ffc4e27c4706249156562905f7d025abeb`. 8 cells (2 setup + 6 scored). TM051/TM050/TM049/TM046 locks were not edited. Parent `W_act_query` hashes did not move.

## TM051 training-wrap decode

Grounded last-P1 states written during TM051 feedback-grounded development, scored with grounded W (no SOCP).

| World | In-sample last-P1 | Held-out wrap (TM051 lock) | Reading |
| --- | --- | --- | --- |
| `w0` | **1/4** | 1/4 | never fitted training writes — optimization |
| `w1` | **4/4** | 1/4 | fitted training writes, held-out stayed 1/4 — generalization |

The wall is mixed at the existing decoder. It is not a clean “never-fit” or a clean “train-only” story. First-match is still the joint oracle: a shared W* exists for all three sets.

## Ceilings (side-effect-free)

W0 is the grounded checkpoint. Constraints are the existing rival-difference rows. W* hashed and discarded. Organism predicate 12/12 on the full set.

| Cell | Code | Feasible | ΔF | Hold on W*_train |
| --- | --- | --- | --- | --- |
| `wrapped_train\|w0` | `train_feasible` | yes | 0.015 | **1/4** |
| `wrapped_train\|w1` | `train_feasible` | yes | 0.0015 | **1/4** |
| `train_ref\|w0` | `train_ref_feasible` | yes | 0.15 | — |
| `train_ref\|w1` | `train_ref_feasible` | yes | 0.15 | — |
| `full_oracle\|w0` | `full_feasible` | yes | 0.26 | — |
| `full_oracle\|w1` | `full_feasible` | yes | 0.19 | — |

W*_train does not read held-out wraps (1/4 both worlds). That is telemetry, not first-match: the joint oracle that includes those held-out states is feasible.

Reconstruction matched locked TM051 reference P1 hashes and held-out wrap hashes.

## Ladder reading

| Result | Here |
| --- | --- |
| Training jointly infeasible | no |
| Reference and feedback conflict | no (`train_ref` feasible) |
| Context-entangled (full joint infeasible) | no |
| Shared W* satisfies everything | **yes** |

Full oracle feasible while TM051 local grounding failed: existing plasticity/optimizer cannot reach a nearby shared solution. Do not install W*. Do not extend v40 SOCP. Do not increase `n_dev_repeats`. Do not add a second decoder.

## What this is not

Not an installed oracle, not a new decoder earn, not a v41 candidate, not product 0.0.005. TM051 `heldout_feedback_decode_fail` stays frozen.
