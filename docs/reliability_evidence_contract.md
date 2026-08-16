# Reliability-evidence contract — TM.0.20.RELIABILITY Phase B

**Lab:** TM.0.20.RELIABILITY.MECH

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`reliability_mech.prereg.lock`](reliability_mech.prereg.lock)

## Claim

> An opt-in recipe may record opaque-channel testimony, compare claim_atoms to independently observed world atoms (provenance ∈ {direct, experiment, state_read}), append experience_reliability rows, recompute bounded source_evidence_margin (λ=4, n_min=2), weight non-duplicated later conflicts, and keep inquiry on minimax value/cost — without host confirm|contradict, trust_score, or collapsing honesty/access/independence/intent into one score.

## Epistemic scope

**Predictive accuracy only** (`source_evidence_margin`). Does **not** implement honesty, access, independence, intent, or a collapsed `trust_score`. Wall preregisters those distinctions diagnostically.

## ABIs

```text
observe_testimony({speaker_token, context_atoms, claim_atoms, event_token})
  → experience_testimony only
  → event_token opaque correlation id (no correctness encoding)

observe_symbol_ground(..., provenance∈{direct,experiment,state_read})
  → organism compares claim_atoms vs observed atoms for matching event_token
  → appends experience_reliability derived row (append-only)
```

Forbidden: host `confirm|contradict`; post-hoc claim↔verify pairing after seeing the answer; majority-as-verification; teacher-verifies-teacher as calibration.

## Provenance

Verification-eligible: `direct` | `experiment` | `state_read`.

Not eligible: `testimony_derived`, testimony itself, conclusions derived from testimony.

## Arithmetic

```text
λ = 4
n_min = 2
quality = (S - K) / n
confidence = n / (n + λ)
source_evidence_margin = max(0, quality × confidence)
```

Compare at 1e-9; unique maximizer required else HOLD.

## Live-claim dedup

At most one live claim per `speaker_token × cue × hypothesis × context_projection`. Replacement supersedes live bit without deleting history.

## Context projection

Include fixture factor atoms (`fac_*`, `feat_*`, `ctxf_*`, `factor_*`). Exclude speaker/event/answer/domain/trial tokens. Jaccard threshold **0.5**; zero overlap never transfers.

## Inquiry

When weighted testimony is non-unique, ask-speaker and experiment candidates enter existing one-step partition value then locked cost. Reliability may change predicted value of asking a speaker; **no** fixed priority list.

## R10 liveness

Opt-in only when `use_source_reliability` is on (`make_reliability`). Default `make_inquire` unchanged.

## Dual memory / isolation

| Fork | Expected |
|------|----------|
| Strip verification rows | Calibration disappears |
| Keep testimony, remove calibration | Conflict → HOLD |
| Keep calibration, remove current testimony | Calibration alone cannot answer |
| Donor calibration into compatible context | Weighting follows where overlap allows |

## Pin order

Unscored smoke → `reliability.candidate.lock` → cells → R-life → capacity → wall (wall prereg already frozen). Preserve `reliability.candidate.v1.lock` if audit rewrites the agent after score.

## Refuse

`trust_score`; honesty/intent ABI fields; host confirm; silently changing `make_inquire` / INQUIRE locks; product stamp / Ex0S 1.0.
