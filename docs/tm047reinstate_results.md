# TM.0.47.REINSTATE results

## Decision: `credit_rho_fail`

First failing boundary: `credit_full_rho`. Earned interface: **false**. Earned K/Q/V: **false**.

Product **0.0.004**. No `cortex.candidate.v41.lock`. K/Q/V not tuned. Decoder not retrained.

The reconstructed TM046 checkpoint still expresses every development action (4/4 both worlds). One-shot credit ρ then ranks the intended action on **1/4** facts. The value written to S, the post-reinstatement address, and the canonical organism path keep that same 1/4. The 1/4 hit is the default action, not a fact whose credit state sat on the decoder manifold.

Oracle retrieval of the **intended fact record** holds on every probe: slot *i* for cue *i*, matching handle, retrieved P1 hash equals the stored episode P1. Post-reinstatement address hash equals stored P1 (cosine 1, L2 0). Injection did not mix or ignore a valid value. Exact replay of stored/credit state does **not** pass while no-memory fails, so this is not an earned reinstatement-interface repair.

DEV ran once on clean `acac6de2a932fce4ac3da6f23beb7e590ab3f3ca`. Frozen runner SHA `c5d5a0be88e8704039c8c2e0d8e3fb86de1fc85ec69863129c5f11c26eccc6c4`. 6 cells. TM046 runner/DEV/decision/addendum were not edited.

## Boundary split (oracle, both worlds)

| Boundary | w0 | w1 | Interpretation |
| --- | --- | --- | --- |
| development_reference | 4/4 | 4/4 | Decoder precondition holds |
| credit_full_rho | 1/4 | 1/4 | One-shot credit state is off the motor manifold |
| stored_value_direct | 1/4 | 1/4 | Stored P1 is not credit ρ; motor ranking stays collapsed |
| post_reinstatement | 1/4 | 1/4 | Scoring address is an exact copy of stored P1 |
| canonical_path | 1/4 | 1/4 | Organism winner matches stored-value ranking |

## Intended-record audit

Oracle `split|symbolic_oracle` retrieved the cue-matched episode, not a shared leftover:

- w0/w1 slots `[0, 1, 2, 3]`, `memory_path=episodic_completed`, `scoring_address_source=reinstated_value`
- `intended_record=true` on all four facts both worlds (retrieved handle and P1 hash match the written episode)
- Four distinct stored/address hashes both worlds: `49305bf204ed68eb…`, `b8c8d1be29185c15…`, `38a8da62b15fab7e…`, `091b9460669a2f2b…`
- `d_w_q=0` after freeze

No-memory cells have `intended_record=false` (no retrieval); their credit/stored 1/4 matches oracle, so live ρ never carried the mapping either.

## Per-fact motor collapse (oracle w0)

Default winner on stored/post/canonical: `h_810668987` (the last handle; the TM046 1/4 hit).

- fact0 `h_182767705`: credit fails (winner `h_810668987`). Stored hash `49305bf2…` equals TM046 p0 replay address. Cosine credit–dev 0.910, credit–stored 0.959, not same hash.
- fact1 `h_901663069`: **credit passes**; stored then ranks `h_810668987`. Secondary stored-value loss on a fact whose credit ρ was on-manifold.
- fact2 `h_806684696`: credit fails (winner `h_901663069`); stored/canonical default.
- fact3 `h_810668987`: credit fails (winner `h_901663069`); stored/post/canonical **pass** because that default action is what S already ranks.

Oracle w1 is the same geometry with default `h_394767965` (fact1). Credit/stored/post/canonical all pass only on that default fact.

Development P1 hashes that pass the decoder (`5336bfd9…`, `1da72187…`, `bc7a87d4…`, `f5e89169…`) are **not** the one-shot stored hashes. Credit P1 is unit-norm, cosine ~0.89–0.92 to development and ~0.96 to stored, but the decoder winner is already wrong at credit for 3/4 facts.

## What the split rules out

- Setup/decoder precondition: development reference 4/4.
- Addressing: intended records retrieved.
- Injection/mixing: post-reinstatement is an exact stored-P1 copy; several cues have identical before/after scores.
- Canonical-only measurement inconsistency: canonical winners match stored-value winners.
- Earned reinstatement interface: exact stored/credit replay does not pass.

Loss is at **value formation** (one-shot credit ρ off the learned motor manifold), with a secondary **storage** miss on the one fact whose credit was correct. Do not repair learned addressing. Do not train a new decoder or tune K/Q/V on this wall. A later dedicated memory-input channel or learned value decoder remains a possible architecture, but only after treating credit/storage representation—not injection—as the failure site.

This is not a v41 candidate review.
