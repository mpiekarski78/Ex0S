# Perspective-evidence contract — TM.0.21.PERSPECTIVE Phase B

**Lab:** TM.0.21.PERSPECTIVE.MECH

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`perspective_mech.prereg.lock`](perspective_mech.prereg.lock)

## Claim

> Given a closed symbolic information-flow topology and observable exposure events, Ex0S reconstructed first-order source-specific evidenced perspectives, distinguished reports aligned with those perspectives from reports inconsistent with them, and returned UNKNOWN when exposure or perspective was insufficient—without claiming knowledge, honesty or intent.

## Epistemic scope

**Source exposure, evidenced perspective, report alignment** — not belief, knowledge, honesty, or intent. Internal: `report_alignment_margin` / use-time `{ALIGNED, MISALIGNED, UNKNOWN}`. Never `honesty_score`.

## Closed-world scaffold

TM.0.21 life/unit fixtures use a **closed, fully observable information-flow topology** with fixture-scaffolded strong exposure atoms. Open-world absence of exposure → **UNKNOWN**. Presence alone never attaches a world fact.

## Exposure levels

| Event | Implication |
|-------|-------------|
| present | opportunity only; insufficient |
| absent | no observed opportunity; not proof of ignorance |
| delivered / ack_read / receipt | strong exposure (may attach via exact event link) |
| sensor_connected | possible channel only |

## ABIs

```text
observe_exposure({speaker_token, context_atoms, exposure_atoms, event_token})
observe_testimony(...)   # frozen RELIABILITY
observe_symbol_ground    # reuse reliability-gated provenance only
```

## Use-time recompute

Raw exposure + world + reports → ALIGNED/MISALIGNED/UNKNOWN at use. No derived alignment as ordinary factual evidence. Jaccard transfers `report_alignment_margin` across similar contexts only — never attaches observation identity.

## Frozen influence

1. Unique direct world grounding → ANSWER  
2. Unique predictive winner and status ≠ MISALIGNED → ANSWER  
3. MISALIGNED → withhold that channel (weight 0), never invert  
4. UNKNOWN → unpenalized predictive weight  
5. Else HOLD / inquire  

When `use_source_perspective` is on, step 1 is evaluated **before** predictive testimony weighting (reliability-only order unchanged when perspective is off).

## Refuse

`honesty_score`; knows/believes/has_access; false-belief claims; Jaccard-as-event-identity; silently changing RELIABILITY locks; product stamp.
