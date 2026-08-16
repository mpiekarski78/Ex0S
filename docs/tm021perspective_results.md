# TM.0.21.PERSPECTIVE results: source exposure & report alignment

**Ex0S under test:** **0.0.004** (not a new stamp)
**Lab:** TM.0.21.PERSPECTIVE
**Date:** 16 August 2026
**ok:** `True`
**life_last_stage_clear:** `P12`
**first_fail (life):** `None`
**Wall first_fail:** `W_misunderstood` / `diagnostic_fail` → next-primitive hint **comprehension**

Locks: [`perspective_baseline.lock`](perspective_baseline.lock) · [`perspective.candidate.lock`](perspective.candidate.lock) · [`perspective.candidate.v1.lock`](perspective.candidate.v1.lock) · [`perspective_mech.lock`](perspective_mech.lock) · [`perspective.lock`](perspective.lock) · [`perspective_wall.lock`](perspective_wall.lock)

`earned_next`: **false** — no Ex0S 0.0.005 / 1.0. Product stamp remains **0.0.004**.

## Bounded claim

> Ex0S reconstructed a source's last uniquely supported evidence perspective from observable exposure events.

Expanded: Given a closed symbolic information-flow topology and observable exposure events, Ex0S reconstructed first-order source-specific evidenced perspectives, distinguished reports aligned with those perspectives from reports inconsistent with them, and returned UNKNOWN when exposure or perspective was insufficient—without claiming knowledge, honesty or intent.

## What cleared

| Phase | Result |
|-------|--------|
| A baseline (`make_reliability`, perspective off) | no ALIGNED/MISALIGNED |
| B unit cells | **9/9** |
| C life P0–P12 + twin | clear through **P12** |
| Capacity lanes | ok |
| Wall scored scripts | pass (`W_attention_gap`, `W_indistinguishable`); `W_misunderstood` = executed **diagnostic_fail** |
| Wall diagnostic first fail | `W_misunderstood` → **comprehension** (`not_run` probes are not diagnostics) |

## Explicit non-claims

- Not belief / knowledge / honesty_score / liar / intent
- Presence never attaches world facts
- Jaccard never attaches perspective across events
- Nested ToM / genuine intent remain open (wall)
- Default `make_reliability` / RELIABILITY locks unchanged

## Essential breakthrough

> Given a closed symbolic information-flow topology and observable exposure events, Ex0S reconstructed first-order source-specific evidenced perspectives, distinguished reports aligned with those perspectives from reports inconsistent with them, and returned UNKNOWN when exposure or perspective was insufficient—without claiming knowledge, honesty or intent.

## Next

Comprehension / nested ToM / intent remain open — named by wall `first_fail_wall`, not earned here.

## Audit notes (apparatus)

Post-freeze audit fixes (scientific claim unchanged):

1. **World-unique first** — with perspective on, unique direct grounding answers before `source_evidence_margin` (frozen influence #1).
2. **Donor-exposure forks** — U5/P12 strip+donor causality; donor-swapped exposure revises ALIGNED→MISALIGNED.
3. **Repetition dedup** — margin key is speaker×cue×hyp×evidenced perspective (not per-report event_token).
4. **Wall first_fail** — `W_misunderstood` is executed `diagnostic_fail`; `not_run` probes are not diagnostics.

