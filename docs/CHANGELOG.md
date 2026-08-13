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
