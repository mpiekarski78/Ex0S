# Interpretation-evidence contract — TM.0.22.INTERPRET Phase B

**Lab:** TM.0.22.INTERPRET.MECH

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`interpret_mech.prereg.lock`](interpret_mech.prereg.lock)

## Claim

> From independently grounded source-specific actions, exposure, and opaque consequence histories, Ex0S reconstructed first-order interpretations of symbolic messages, distinguished insufficient reconstruction from behavioral conflict, adapted communication to the recipient under organism-resolved goal cues, and remained causally dependent on interpretation evidence in S—without claiming subjective comprehension, world-truth from another’s map, honesty, stability, or intent.

## Epistemic scope

**Behaviorally evidenced interpretation** — not subjective comprehension, belief, honesty, stability, or intent. Dual use-time outputs: reconstruction ∈ `{UNIQUE, AMBIGUOUS, INSUFFICIENT}`; fit ∈ `{SUPPORTED, CONFLICT, UNKNOWN}`. Never `understood` / `cause` ABI.

## Independent behavioral anchor

Interpretation evidence counts **only** when action/selection/resulting state is independently grounded through **non-INTERPRET** evidence (`experience_grounding` with provenance ≠ `testimony_derived`). Interpretation rows cannot recursively ground their own anchors.

## Observation ABI

```text
observe_source_consequence({
  source_token, interaction_token,
  exposure_event_token, consequence_event_token,
  context_symbols, message_symbols, action_symbols,
  state_before, state_after
})
```

All values opaque. No `result`. Interaction/event tokens are pre-outcome observable identifiers—not scorer causal oracles.

## Dual outputs

```text
interpret_message({source_token, context_symbols, ordered_symbols})
  → UNIQUE | AMBIGUOUS | INSUFFICIENT

interpretation_fit({source_token, context_symbols, message_symbols,
                    action_symbols, state_before, state_after})
  → SUPPORTED | CONFLICT | UNKNOWN
```

`interpretation_fit` internally calls `interpret_message`. Never inject candidate. No derived statuses in S.

## World separation

Unique world evidence answers world questions. INTERPRET answers **source-relative** questions only. Never promote UNIQUE+SUPPORTED into world ANSWER.

## Factorized reconstruct

Symbol ↔ grounded-atom, ordered-role, relation, context-conditioned maps. No Jaccard. Backoff never relaxes `source_token`. Same-context conflict → AMBIGUOUS (no newest-wins).

## Repair

```text
plan_recipient_message({recipient_token, context_symbols, goal_cue_symbols})
```

Resolve unique goal from pre-existing S. Atomic HOLD. Harness never supplies row IDs or desired structures.

## Refuse

Subjective comprehension; honesty_score; cause ABI; `result` field; injected candidates; Jaccard; derived statuses in S; silently changing PERSPECTIVE/RELIABILITY/INQUIRE/SEQUENCE locks; product stamp.
