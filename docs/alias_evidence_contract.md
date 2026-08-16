# Alias-evidence contract — prereg only

**Lab:** none assigned

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`alias_evidence.prereg.lock`](alias_evidence.prereg.lock)

ALIASWALL established that frozen RELATE depends on externally supplied symbol equivalence. This contract asks what observable evidence could justify equivalence; it does not implement or test an alias mechanism.

## Claim

> Two opaque aliases may be treated as equivalent only when observable behavioral evidence shows the same consequences under multiple independent probe contexts, without supplying a hidden role, canonical identity, equivalence class, mapping, or route position as identity.

## Frozen evidence rule

- One matching consequence is insufficient: **HOLD**.
- A fingerprint witness is the tuple (`probe_context`, `action`, `observed_outcome`); both aliases must match the action and outcome under that context.
- Matching fingerprint witnesses under at least two independent probe contexts permit equivalence.
- The witnesses must be distinct probes, not duplicate, relabelled, or role-coded copies.
- Route position is not identity evidence.
- “Permitted” is a future candidate gate, not a result from this pass.

## Channel contract

| Allowed field | Meaning |
|---|---|
| `alias` | Opaque token under probe |
| `probe_context` | Independent probe identity; not a role or route position |
| `action` | Motor or act taken |
| `observed_outcome` | Success/failure or named observable outcome enum |

Forbidden: role, equivalence class, canonical token or ID, role↔alias mapping, latent map, or route-position-as-identity. The scorer alone may know the latent map. A future organism would have to author fingerprint evidence into S.

## Future battery contract

| Cell | Evidence | Honest outcome |
|---|---|---|
| A0 wall | Same latent route only; frozen ALIASWALL Kill | HOLD |
| A1 weak | One matching consequence | HOLD — ambiguous |
| A2 convergent | Two independent matching consequences | Equivalence permitted |
| A3 collision | Same first consequence, different second | Keep separate |
| A4 swap | Fingerprints exchanged between aliases | Behavior follows evidence |
| A5 contradiction | Later evidence conflicts | Split or HOLD |
| A6 causality | Reset ρ / wipe S / donor swap | Equivalence follows S only |

A2–A6 are not claimed green. No organism, runner, test, or post-run freeze lock exists for this contract.

## Later

After this contract is frozen: build a small candidate that authors fingerprint evidence into S, attack it with A0–A6, and freeze only if the future battery earns it. Gap persistence / object continuity remains a separate question; anonymous features and encoders remain later still.
