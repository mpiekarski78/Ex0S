# Changelog

## 2026-08-14

- Created public repo [mpiekarski78/three-memory](https://github.com/mpiekarski78/three-memory) (not a BDH fork).
- Scaffold: frozen cortex, session ρ, inspectable store S, innate novelty/integrity drives.
- v0 key/door world + CLI `experiments.run_v0`.
- Reported classification: **Store-works**. disable-S recovers Trace-only (BDH Category B analogue).
- Audit: removed env-injected lesson strings; writes are event-driven. Documented tag→action retrieve limit. Metrics include `n_forced_steps`.
- v1 tiny byte LM + NOTE retrieve; same `my lo` probes as BDH. Classification **Store-works**. Published BDH numbers are the trace-only column ([`docs/v1_results.md`](v1_results.md)).
- v2: plain prior (no NOTE-copy, follow-acc 0.025) + raw snippet retrieve. Classification **Trace-only**. S is inspectable and prepended as `my love\nmy lo`; after ρ reset P(v) stays at prior. v1 Store-works needed the taught copy protocol.
