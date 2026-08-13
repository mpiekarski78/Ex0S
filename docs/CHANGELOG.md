# Changelog

## 2026-08-14

- Created public repo [mpiekarski78/three-memory](https://github.com/mpiekarski78/three-memory) (not a BDH fork).
- Scaffold: frozen cortex, session ρ, inspectable store S, innate novelty/integrity drives.
- v0 key/door world + CLI `experiments.run_v0`.
- Reported classification: **Store-works**. disable-S recovers Trace-only (BDH Category B analogue).
- Audit: removed env-injected lesson strings; writes are event-driven. Documented tag→action retrieve limit. Metrics include `n_forced_steps`.
- v1 tiny byte LM + NOTE retrieve; same `my lo` probes as BDH. Classification **Store-works**. Published BDH numbers are the trace-only column ([`docs/v1_results.md`](v1_results.md)).
- Audit (v1): probe-time S-off effect is a prefix→byte logit bias; hidden EMA is for write novelty only. Classification now requires reset-S to return to prior. NOTE-copy is the S-on mechanism after ρ reset (stated in v1_results).
