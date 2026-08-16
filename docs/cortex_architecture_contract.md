# Cortex architecture contract — TM.0.23.CORTEX

**Lab:** TM.0.23.CORTEX  
**Product under test:** Ex0S **0.0.004** (stamp 0.0.005 only after a later scored earn gate)  
**Flags:** `earned_next=false`, `ex0s=null`  
**Prereg:** [`cortex.prereg.lock`](cortex.prereg.lock)

## Claim (later earn only)

> Starting from a pinned birth genome, one recurrent plastic artificial cortex matured during continuous symbolic lives and acquired relation learning, revision, persistent recall, symbolic grounding and variable-length sequence construction through generic event, memory and action interfaces, including transfer to held-out renamed worlds.

This pass freezes architecture, worlds, birth substrate, and unscored learning-law sanity. **No D scoring. Neurons running is not a stamp.**

## Stance

- New organism. Does **not** wrap `make_interpret` / `ThreeMemoryAgent` capability methods.
- Frozen 0.0.004 locks are phenotype references / evaluation specifications, never callable tools.
- No LLM / Transformer LM / pretrained language weights / next-token foundation / external corpus / language oracle.
- Consciousness is outside the operational contract.
- HONESTY is a future wall, not a hand-written mechanism.
- Existing [`three_memory/cortex.py`](../three_memory/cortex.py) (`FrozenCortex`) stays frozen. Neural code lives in `neural_cortex.py` / `cortex_memory.py`.

## Memory division

| Store | Role |
|-------|------|
| ρ | Fast recurrent activation; resettable |
| Cortex | Weights, connectivity, plasticity, skills, attention, retrieval policy, values |
| S | External episodic/declarative records; opaque symbols; provenance |

Individual = genome + developed cortical state + plasticity state + homeostatic/body last + developmental age + external S.

## Observe ABI (exact)

```text
observe({
  interaction_token,
  source_token,
  ordered_symbols,
  observable_state,
  body_state
})
```

- Exact key set. Extra or missing keys → reject.
- Banned keys: `homeostatic_delta`, `correct`, `reward`, `result`, `stage`, `lab`, `capability`, `answer`, `intended`, `expected`.
- `ordered_symbols`: ordered; sequential microticks (never mean-pooled).
- `observable_state`: **set**; JSON list order irrelevant; encoded as sum of vocab vectors.
- `body_state`: exactly 4 floats from frozen world physics (energy, damage, resource, analog). Not a correctness label. Not from `scorer_only`.
- Runner never chooses a reward scalar from evaluator truth.

Operations only: **RETRIEVE, WRITE, EMIT, ACT, STOP, HOLD** (fixed order indices 0..5).

Motor surfaces (not capability heads): `W_emit_query`, `W_act_query`.

## Genome dimensions

| Symbol | Value |
|--------|-------|
| `n` | 64 (default scored birth) |
| `d_sym` | 32 |
| `k_s` | 8 |
| `d_body` | 4 |
| `d_x` | `32 + 8*32 + 4 + 1 = 293` |
| `n_op` | 6 |
| `p_connect` | 0.10 |
| `T_max` | 8 |
| `τ` | 1.0 |
| cosine threshold | 0.15 |
| `η_pred` | 0.05 |
| `η_act` | 0.05 |
| `β` | 0.01 |
| clip | [-2, 2] |
| `body*` setpoint | `[1.0, 0.0, 1.0, 0.0]` |
| dtype (CPU gold) | float64 |

Cost table: RETRIEVE=1, WRITE=1, EMIT=1, ACT=1, STOP=0, HOLD=0.

## Sensory microticks

One host `observe` expands to:

1. `EVENT_START` — inject `v_start` + source vector `v_src`.
2. For each `u` in `ordered_symbols` (order preserved): inject vocab `v_u`; update ρ.
3. `EVENT_END` — inject `v_end`.
4. `STATE` — inject `sum_{u in observable_state} v_u` (empty → 0).

`body_t` and `same_ix ∈ {0,1}` are concatenated on every tick via genetically wired channels (`W_body` frozen; `same_ix` channel).

`interaction_token` is provenance + equality only (not a unique scene embedding).

### Tick update

```text
x_tick = concat(injected d_sym, flat(retrieval_buffer), body_t, [same_ix])
pre = (W_rec ⊙ M) ρ + W_in x_tick + b
ρ = tanh(pre)
```

Mean pooling of messages is forbidden. `[A,B]` ≠ `[B,A]` trajectories.

## Internal motor loop

After sensory ticks:

```text
for k in 1..T_max:
  select op ~ Categorical(softmax(W_op ρ / τ)) via rng_action
  HOLD  → discard uncommitted emit buffer; stop
  STOP  → commit emit buffer; stop
  EMIT  → append token from W_emit_query; continue (no new observe)
  ACT   → choose operand from W_act_query; stop; WAIT next observe
  RETRIEVE → populate retrieval_buffer for NEXT tick; continue
  WRITE → stage write_t; visible after tick commits; continue
```

During internal ticks: no host observe; sensory injection zeros except retrieval_buffer, last body, same_ix. Plasticity creates/updates eligibility only; outcome-driven weight updates run on the **next** observe.

## Credit-assignment order

On observe at time t:

1. Read `body_t`, encode set `s_t`.
2. If prior eligibility exists:
   - `ε_t = s_t - ŝ_{t-1}`
   - `adv_t = ||body_{t-1}-body*||₂ - ||body_t-body*||₂ - cost_{t-1}`
   - `W_pred += η_pred · outer(ε_t, ρ_elig_{t-1})`
   - three-factor on used motor matrices: `W += η_act · adv_t · outer(e_vec, ρ_elig_{t-1})`
   - clip; consolidate
3. Sensory microticks.
4. Motor loop; select action.
5. Store eligibility_t from action tick; `ŝ_t = W_pred ρ`.

Consolidation: `W^slow ← (1-β)W^slow + β W`; `W ← W^slow + 0.5(W - W^slow)`. `W_rec` updates × `M`. `W_body` frozen.

## Retrieval / WRITE

- `retrieval_buffer ∈ R^{k_s×d_sym}`, init zeros.
- RETRIEVE: top-`k_s` by cosine; ties by `fact_id` lex order; fills **next** tick.
- Non-RETRIEVE: buffer persists. ρ reset clears buffer. Checkpoint preserves it.
- WRITE: `{fact_id, content, when, interaction_token, source_token, source:"cortex_write"}`. Visible after commit. No `experience_*` capability sources.

## RNG streams (checkpointed)

| Stream | Role |
|--------|------|
| `rng_birth` | mask, weight init, `W_body`, `v_start`/`v_end` |
| `rng_registry` | vocab vectors (spelling/hash independent) |
| `rng_source` | source identity vectors |
| `rng_action` | exploration |
| `rng_permute` | world presentation permutation |

Main vs twin: independent registry/source seeds.

## Resets

- **ρ reset:** ρ=0; eligibility=0; retrieval_buffer=0; emit buffer cleared; ŝ=0. Keeps W, W^slow, M, S, registries, last body, age, RNG streams.
- **Full cortical reset:** restore birth W/W^slow/M/W_body; clear fast state; age=0. Does not copy adult S into genome.

## Capacity lanes

- **Architecture lane:** independent births at n ∈ {32, 64, 128, 256}. No transplant from a 64-unit adult.
- **Mature-state lanes** (fork 64-unit mature checkpoint only): vocab 32→128→512→2048; S 1k→10k→100k; age 1k→10k→100k; length 1→2→4→8→16; domains 1→3→6→12; alternatives 2→4→8→16.

## Statistics (exact)

- Earn battery: **16** main/twin pairs. Optional extra 16 separately preregistered, never required.
- Earn ≥13/16. Maturation ≥14/16. No seed replacement.

## Held-out

`eval_seed_commitment = SHA256(seed_bytes || salt_bytes)` with independent 256-bit secrets. Reveal only after candidate. Post-reveal change ⇒ new candidate + new commitment.

## Scorer split

Each world: `organism_events` vs `scorer_only`. Tests prove `scorer_only` never reaches observe, S, registries, cortex init, body path, or action updates.

## GPU

CPU float64 is the equation reference. Spark GB10 used for device/batched sanity. Candidate requires learning-law sanity, not mere numerical similarity. Later D scoring evaluates statistically on the pinned GPU environment.

## Refuse

Subjective comprehension; honesty_score; LLM; wrapping `make_interpret`; host `homeostatic_delta`; mean-pool messages; undirected Hebb on `W_pred`; random W_h as homeostasis; neuron transplant; `earned_next=true` / non-null `ex0s` this pass; D scoring this pass; product stamp 0.0.005 without earn gate; silently editing 0.0.004 locks.
