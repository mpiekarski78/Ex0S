# TM.0.48.CREDITINFO results

## Decision: `credit_action_information_absent`

ρ changed with credited action: **false**. Unique post-credit ρ: **1**. Earned action-feedback *next*: **true**. Earned K/Q/V: **false**. Earned value decoder: **false**.

Product **0.0.004**. No `cortex.candidate.v41.lock`. K/Q/V not tuned. Decoder not retrained. Action-feedback channel **not implemented**.

From one identical pre-credit TM046 checkpoint, four clones credited the same cue with four different actions. Pre-credit ρ was identical. Post-credit ρ, unit P1, \(W_v\rho\), and the episode value written to S were each a **single shared vector** (pairwise cosine 1, L2 0). Body outcomes were identical (shared positive delta). The development ceiling still decoded every action 4/4. Credit ρ decoded 1/4 — the default action — because that is the only ranking the shared cue state can produce.

The four distinct stored values on TM046/TM047 identified **four cues**. Holding the cue fixed, storage collapses to one value with no decodable answer. The runner did not copy motor vectors into S.

DEV ran once on clean `cc6bf089e325bf503c0f440b5fefe6675fadd4b2`. Frozen runner SHA `57b8d4c6908908e25bd6fedcd561bf60c5a1a3b8a7e5e11e085be5596350a7c9`. 4 cells. TM046/TM047 locks were not edited.

## Split (both worlds)

| Record | Unique hashes | Motor n_ok | Notes |
| --- | --- | --- | --- |
| development ceiling | 4 | 4/4 | Positive control |
| ρ before credit | 1 | — | Same cue, same checkpoint |
| ρ after credit | 1 | 1/4 | Default-action collapse |
| unit P1 after credit | 1 | 1/4 | Scoring address for credit ρ |
| \(W_v\rho\) | 1 | 1/4 | Projection of the same missing information |
| stored episode P1 | 1 | 1/4 | Equals pre-credit cue P1 `49305bf2…`, not \(W_v\rho\) |

Cross-world geometry is the same (`rho_after=51fa982a…`, `p1_after=023b1715…`, stored=`49305bf2…`). Worlds only relabel the default winner (`h_810668987` on w0, `h_394767965` on w1).

## What this earns next — not on this wall

K/Q/V training, a reinstatement channel, and a storage optimizer cannot reconstruct an action that never entered ρ. Scalar reward cannot identify which unperformed action was correct. The earned next mechanism is a **generic action-feedback / efference-copy channel**:

cue experience → action performed or demonstrated → generic action vector enters recurrent cortex → reward/credit gates learning → post-feedback ρ → opaque value write

That edit is **not** this wall. After it, TM047 boundaries must be repeated. Required then: credit ρ decodes all actions, stored value preserves them, exact reinstatement passes, no-memory still fails. Only then is value projection or learned K/Q/V earned.

This is not a v41 candidate review.
