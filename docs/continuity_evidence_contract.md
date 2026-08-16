# Continuity-evidence contract — prereg only

**Lab:** none assigned

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`continuity_evidence.prereg.lock`](continuity_evidence.prereg.lock)

GAPWALL measured the frozen continuity boundary (empty-skip ≠ object continuity; episode gap cuts; two equals HOLD). This contract asks what observable evidence could justify provisional post-gap identity; it does not implement or test a persistence mechanism. Alias fingerprints remain a separate track.

## Claim

> An opaque post-gap token may be linked provisionally to a pre-gap token only when an observable pre-gap intervention placed a specified mark or state on the latter and exactly one post-gap candidate produces the corresponding observable readout. Token spelling, route position, empty-event skip, episode structure, or sole candidacy contribute no identity evidence. The resulting continuity hypothesis is defeasible and must be withdrawn when later observations contradict it.

## Frozen evidence rule

- Empty-event skip, episode structure, sole candidacy, token spelling, and route position are **not** identity evidence.
- A strong witness is a **causal apply** before the gap and a matching **readout** after on **exactly one** candidate.
- Observing the same property twice without an apply intervention is insufficient.
- A bare recurring `mark_id` without the apply→read sequence is insufficient.
- Both verify / neither verifies / mutually conflicting readouts → refuse a unique link.
- Later contradiction **withdraws** the prior permission; no stale merge may remain effective.
- **Admissibility** (“hypothesis permitted”) is not a claimed pass. A future earn requires an identity-dependent probe that cannot be solved by skip, sole candidacy, or spelling (always-HOLD fails earn).

## Channel contract

| Allowed field | Meaning |
|---|---|
| `token` | Opaque symbol under observation |
| `mark_id` | Independent mark identity (not an object ID); fresh/counterbalanced; swappable, removable, collision-capable; no permanent token→mark map |
| `phase` | `pre_gap` = observe **apply**; `post_gap` = observe **readout** |
| `operation` | `apply` or `read`, locked to phase |
| `observed_state` | Observable mark/state at that phase |

Token spellings are randomized independently of marks. Forbidden: object ID, same-as, continuity class, canonical identity, latent map, oracle “same object”, empty-skip-as-continuity. Scorer-only ground truth. A future organism authors raw continuity evidence into **S only**.

## Future battery contract

| Cell | Evidence | Honest outcome |
|---|---|---|
| C0 gapwall | Frozen GAPWALL lessons | Skip ≠ continuity; episode no bridge; two peers HOLD |
| C1 weak | Reappear only / incomplete apply→read / bare mark_id | HOLD — no permission |
| C2 mark | Apply + exactly one verifying readout | Hypothesis **permitted** (admissibility); earn needs identity-dependent probe |
| C3a both | Both candidates verify | Refuse unique |
| C3b neither | Neither verifies | Refuse unique |
| C3c conflict | Mutually conflicting readouts | Refuse unique |
| C4 swap | Mark evidence swapped; tokens/marks independent | Follows mark evidence (attacks mark-as-ID) |
| C5 contradiction | Later conflicting observation | **Withdraw**; HOLD/separate; no stale merge |
| C6 causality | Reset ρ / strip continuity rows / donor swap | Evidence follows S only |

C2–C6 are not claimed green. No organism, runner, test, or post-run freeze lock exists for this contract.

## Later

After this contract is frozen: build an opt-in persistence candidate that authors continuity evidence into S, attack it with C0–C6 (including the identity-dependent earn probe), and freeze only if earned. That candidate is TM.0.16.PERSIST ([`tm016persist_results.md`](tm016persist_results.md); `earned_next=false`). Alias fingerprints stay separate; anonymous features and encoders remain later still.
