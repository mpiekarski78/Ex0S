# Changelog

## 2026-08-14

- Created public repo [mpiekarski78/three-memory](https://github.com/mpiekarski78/three-memory) (not a BDH fork).
- Scaffold: frozen cortex, session ρ, inspectable store S, innate novelty/integrity drives.
- v0 key/door world + CLI `experiments.run_v0`.
- Reported classification: **Store-works**. disable-S recovers Trace-only (BDH Category B analogue).
- Audit: removed env-injected lesson strings; writes are event-driven. Documented tag→action retrieve limit. Metrics include `n_forced_steps`.
- v1 tiny byte LM + NOTE retrieve; same `my lo` probes as BDH. Classification **Store-works**. Published BDH numbers are the trace-only column ([`docs/v1_results.md`](v1_results.md)).
- v2: plain prior (no NOTE-copy, follow-acc 0.025) + raw snippet retrieve. Classification **Trace-only**. S is inspectable and prepended as `my love\nmy lo`; after ρ reset P(v) stays at prior. v1 Store-works needed the taught copy protocol.
- v3: S is a folder of `.md` files (no embeddings). New agent + empty ρ + reload from disk. note arm **Store-works** (P(v)=0.988, JS vs in-process = 0). raw arm **Trace-only** (P(v)=0.093). Files persist; the LM still only *uses* them with the taught NOTE protocol.
- v4: 13 notes in S, select longest heading vs dump-all. note+select **Store-works** (P(v)=0.988, 12 rejected). Dump-all **collapses** P(v) to 0.007. raw+select **Fail** (file unused). [`docs/v4_results.md`](v4_results.md).
- v5: unread library W. commit copies one file into S then unmounts W: note **Store-works** (P(v)=0.988, S=`my-lo.md` only). Peek then unmount → prior. Collect off ignores W. raw commit **Fail**. [`docs/v5_results.md`](v5_results.md).
- v6: plain prior only. **tool** grammar (heading→byte, bias +3.0) **Store-works** P(v)=0.649; LM context is `my lo` (no `love`). fewshot and untaught NOTE **Fail** (~0.053). Use-skill belongs in machinery, not in cortex. [`docs/v6_results.md`](v6_results.md).
- v7: native integer tags, no English prior. Genome = cortex seed 1337 (not ACGT). Experience + reload `.tag` files **Store-works** (`use_key`). Collect commit unmount works; peek unmount and dump-all return `open`. [`docs/v7_results.md`](v7_results.md).
- v8: boxed use-policy may learn; cortex frozen. Features `{s_hit, w_hit}` only. Two-step train (commit then apply after unmount). Red unmount `use_key`; held-out green `wait`; disable-S / empty S `open`. **Store-works**. [`docs/v8_results.md`](v8_results.md).
- v9: write from a life, not from W. Clutter library has no `d0.tag`/`d2.tag`. Policy learns when to author `{door, action}` from a door-opening. Red → `d0.tag` `use_key`; held-out green → `d2.tag` `wait`. **Store-works**. [`docs/v9_results.md`](v9_results.md).
- v10: free life, no forced curriculum. ε-greedy over percept affordances; probe greedy. `n_forced=0`. Red sequence was not OPEN→PICK→USE; green found WAIT. **Store-works**. [`docs/v10_results.md`](v10_results.md).
- v11: two free lives, one S. Authored `d0.tag`+`d2.tag`. Select red `use_key` / green `wait`. Dump-all red **`wait`**. **Store-works**. [`docs/v11_results.md`](v11_results.md).
- v12: retrieve head learns select vs dump (`{n_store, n_hits}`). Untrained dumps red `wait`; trained selects. Held-out blue `open`. Dump-all still fails. **Store-works**. [`docs/v12_results.md`](v12_results.md).
- v13: use-gate + generic `logits[int(action)] += 3.0`. Untrained ignores a planted tag (`open`). Trained red `use_key`; held-out green `wait`. **Store-works**. [`docs/v13_results.md`](v13_results.md).
- v14: A pick-one among same-door matches vs B write `{door}` vs `{door, action}`. Both **Store-works**. [`docs/v14_results.md`](v14_results.md).
- v15: joint write/schema/use/pick with no clamps. **Store-works**. [`docs/v15_results.md`](v15_results.md).
- v16: A prefer `ok=1` over newest junk (**Store-works**); B shared return on v15 joint (**Fail**). [`docs/v16_results.md`](v16_results.md).
- v17: A read `do=` vs `action=`; B match `here=` vs `door=`. Both **Store-works**. [`docs/v17_results.md`](v17_results.md).
- v18: A write `do=` vs `action=`; B write `here=` vs `door=`. Read/match frozen. Both **Store-works**. [`docs/v18_results.md`](v18_results.md).
