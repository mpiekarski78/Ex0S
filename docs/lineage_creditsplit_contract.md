# TM.0.35.CREDITSPLIT contract

**Lab:** TM.0.35.CREDITSPLIT · **Not v39.** No neural edit. v38 closed unchanged.

Fresh `TM035.CREDITSPLIT.DEV./TWIN.` worlds. Do **not** edit TM031–TM034 locks. Do **not** change rehearsal, write L2, `R`, or `ACT_RECALL_MODES`.

## Why this split

TM034 locked a credit-prefix rebreak (store at zero after credits 0–6; credit 7’s burst starts at 4 violations) but did not snapshot between episode write and the ranking-error one-shot. v39 safe scaling of that one-shot is authorized only if the one-shot, not write, not burst, not their interaction, is the locked cause.

## What is measured

Parent teaches credits **0–6** with default v37. Checkpoint. Every credit-7 arm clones from that identical checkpoint. Assert matching weights, episodes, RNG, and pending credit before the component.

Four arms on credit 7:

- `write_only` — `_episode_write` only
- `oneshot_only` — write + ranking-error one-shot, no burst
- `burst_only` — write + v37 awake burst, no one-shot
- `complete` — native `_credit_act_p1_episode` (not a reimplemented write+shot+burst)

When asking whether previously correct rows broke, **exclude the newly written slot**. Previously correct = valid stored rows that were non-violating on the pre-credit-7 checkpoint.

Live 8-probes run on **another clone** of the post-component snapshot.

Imported TM032 helpers are pure (`clone_plastic`, `stored_rows`, `live_probes`). No historical module-global patching.

## Routes (first-match on diagnostic worlds)

Diagnostic = `complete` rebreaks previously correct rows (new slot excluded).

1. `creditsplit_write_causal` — `write_only` already breaks them
2. `creditsplit_oneshot_causal` — write leaves them correct; one-shot breaks them; burst-only does not
3. `creditsplit_burst_causal` — prefix leaves them correct; burst breaks them
4. `creditsplit_interaction_only` — only `complete` breaks
5. `creditsplit_mixed_oneshot_and_burst` — diagnostic worlds disagree, or both one-shot and burst independently break
6. `creditsplit_no_diagnostic` — no world where complete rebreaks previously correct rows

**v39 gate:** proceed only if every diagnostic world is `creditsplit_oneshot_causal`. Mixed, write, burst, or interaction-only must **not** open one-shot-only v39.

Product **0.0.004**.
