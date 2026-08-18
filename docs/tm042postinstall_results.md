# TM.0.42.POSTINSTALL results

Post-install continuity wall. Product **0.0.004**. No `cortex.candidate.v40.lock`.

## Lineage stop

R2 first-match **`canonical_r2_later_learning_not_exercised`** stands. This lab does not replace it.

## Frozen first-match: `postinstall_mech_install_fail`

Preregistered mechanistic cells were TM039 `reg2`/`reg3`. Those TM039 worlds are **v37-already-converged**, so fallback was not invoked (`violations_after_v37=0`). That is a reconstruct targeting miss, not an organism SOCP rejection.

## Natural handoff (fresh `TM042.POSTINSTALL.*`)

| Scale | Install | Continuity |
|---|---|---|
| c8h2 (8×2) | 0/4 `postinstall_not_exercised` | n/a |
| c8h4 (8×4) | **4/4** atomic `optimal` | **4/4 pass** |

On every 8×4 install: previously learned mappings held; a new mapping was acquired; reversal replaced the old constraint; subsequent fallback stayed atomic; novelty stayed unfamiliar; solver handles did not enter ACT routing.

Example handoff `natural|c8h4|A_then_B|w0`: pre-install `9255ceb3…` → solver `optimal` applied → installed `b70b654f…` → later credit → post-credit `1508d113…`.

## Candidate discussion

The user pair (natural fallback on untouched worlds, and later learning after that install) is observed on 8×4. The frozen ladder still requires the mechanistic reconstruct to install, so `candidate_discussion_open` remains false and no candidate lock is written.
