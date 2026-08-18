# TM.0.41.LIVEADDR results

Live-address classification of TM040 failed acquire cues. Product **0.0.004**. No `cortex.candidate.v40.lock`.

## Decision: `liveaddr_canonical_path_inconsistency`

Locked TM040 telemetry could not name the scoring address. Reconstructing the four failed acquire stems (`TM040.CAUSAL.DEV.`, seed `404000039`) shows:

| Failed live path | Result |
|---|---|
| `cortical_fallback` because unfamiliar | **no** — all failed cues are `episodic_completed`, familiar, \(d_1 < R\) |
| Episodic completion to wrong slot | **no** — retrieved slot equals teach index 2 |
| Correct slot and stored P1, but live still fails | **yes** — organism scores stored P1 and ranks correctly; TM040 scored a different live P1 |
| Forced stored P1 passes while live scoring address fails | **no** — scoring-address hash equals stored P1; that address passes |
| Forced stored P1 also fails despite zero violations | **no** — zero store violations; stored P1 \(\gamma \approx 0.015\) |

always-joint did **not** change retrieval. Path, familiarity, \(d_1\), \(R\), slot, scoring-address hash, stored-P1 hash, and live-P1 hash match fallback. \(W\) hashes differ. always-joint only moved \(W\) so the TM040 live P1 also ranked. SOCP cannot change keys; this is that case.

Later learning: **not_exercised**. Contradict: **jointly_feasible_atomic_apply**. Overall TM040 decision remains acquire fail.

## Failed cue (example `acquire|c8|A_then_B|w0|fallback_joint`)

Cue `s_294555646` want `h_812030613` (TM040: win `h_679764572`, \(\gamma \approx -0.005\)).

- recall path `episodic_completed`, familiar, \(d_1 \approx 0.224\), \(R \approx 0.314\), slot 2 = teach index
- scoring-address hash = stored P1 hash `d96ff800…`; live P1 hash `d22c5e88…`
- store ranking ok, \(\gamma \approx 0.015\); TM040 live ranking fail, \(\gamma \approx -0.005\)
- counterfactual stored P1 through `actuator_scores`: pass
- counterfactual live scoring address through `actuator_scores`: pass (same vector as stored P1)
- v37 and fallback share \(W\) `1861e346…`; always-joint `54bfeeb7…`

The organism ACT path would rank this cue correctly. TM040's live 7/8 used `actuator_scores(live P1)`, which is not the canonical scoring address.

## What not to do

Do not invoke SOCP unconditionally, add live constraints, change the fallback trigger, or modify half-spacing. The measurement boundary (probe live P1 vs organism scoring address) is the next repair, not another architecture.
