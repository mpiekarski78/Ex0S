"""TM.0.23.CORTEX.DEVELOP — tightened D0–D12 scorers (no import from cortex_develop_life).

Duplicated constants/helpers avoid circular import when life imports this module.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from experiments.run_tm023cortex import build_observe, make_cortex, physics
from three_memory.cortex_memory import CortexRecord
from three_memory.neural_cortex import GenomeConfig, NeuralCortex

# --- frozen D0 chance model (must match cortex_development_contract.md) ---
D0_N_PROBES = 64
D0_P0 = 0.5
D0_ALPHA = 0.01

STAGES = [f"D{i}" for i in range(13)]

DEFAULT_LATENT = {
    "act_effects": {
        "press": {"state": ["st_pressed"], "delta": [0.25, -0.1, 0.15, 0.0]},
        "harm": {"state": ["st_hurt"], "delta": [-0.35, 0.45, -0.15, 0.0]},
        "idle": {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
        "get": {"state": ["st_got"], "delta": [0.1, 0.0, 0.2, 0.0]},
        "drop": {"state": ["st_drop"], "delta": [-0.05, 0.05, -0.2, 0.0]},
    }
}

BODY0 = [0.5, 0.25, 0.5, 0.0]


class SeedLike(Protocol):
    """Minimal seed surface used by scorers (LifeSeeds-compatible)."""

    pair_id: int
    role: str
    seed_permute: int
    seed_registry: int

    def genome(self) -> GenomeConfig: ...


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X~Binomial(n,p), exact sum (n<=64)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return float(total)


def d0_chance_spec() -> dict[str, Any]:
    return {
        "n_probes": D0_N_PROBES,
        "p0": D0_P0,
        "alpha": D0_ALPHA,
        "test": "one_sided_binomial_target_preference",
        "pass_rule": "fail_to_reject_H0_at_alpha_and_empty_S_and_no_preloaded_targets",
    }


def _tok(prefix: str, seeds: SeedLike, name: str) -> str:
    if seeds.role == "main":
        return f"{prefix}_{name}"
    h = hashlib.sha256(f"{seeds.seed_registry}:{name}".encode()).hexdigest()[:8]
    return f"tw_{h}_{name}"


def apply_event(
    ag: NeuralCortex,
    *,
    ix: str,
    source: str,
    symbols: list[str],
    state: list[str],
    body: list[float],
    latent: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[float]]:
    out = ag.observe(
        build_observe(
            interaction_token=ix,
            source_token=source,
            ordered_symbols=symbols,
            observable_state=state,
            body_state=body,
        )
    )
    action = out.get("action") or {}
    next_state, next_body = list(state), list(body)
    if action.get("op") == "ACT":
        next_state, next_body = physics(body, action.get("token"), latent)
    return out, next_state, next_body


def teach_loop(
    ag: NeuralCortex,
    seeds: SeedLike,
    *,
    n: int,
    symbols_fn,
    source: str = "src_teach",
    latent: dict[str, Any] | None = None,
    body0: list[float] | None = None,
) -> list[float]:
    latent = latent or DEFAULT_LATENT
    body = list(body0 or BODY0)
    state = ["st_idle"]
    rng = np.random.default_rng(seeds.seed_permute)
    for i in range(n):
        syms = symbols_fn(i, rng)
        if seeds.role == "twin" and len(syms) > 1 and (i % 2 == 0):
            syms = list(reversed(syms))
        _, state, body = apply_event(
            ag,
            ix=f"{seeds.role}_t{i}",
            source=source,
            symbols=syms,
            state=state,
            body=body,
            latent=latent,
        )
    return body


def _act_token_counts(ag: NeuralCortex, toks: dict[str, str], n: int, cue: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    body = list(BODY0)
    state = ["st_idle"]
    for i in range(n):
        out, state, body = apply_event(
            ag,
            ix=f"probe_{i}",
            source="src_probe",
            symbols=cue,
            state=state,
            body=body,
            latent=DEFAULT_LATENT,
        )
        act = out.get("action") or {}
        if act.get("op") == "ACT" and act.get("token"):
            counts[act["token"]] = counts.get(act["token"], 0) + 1
    return counts


def _op_histogram(ag: NeuralCortex, n: int, cue: list[str], *, prefix: str = "oph") -> dict[str, int]:
    hist: dict[str, int] = {}
    body = list(BODY0)
    state = ["st_idle"]
    for i in range(n):
        out, state, body = apply_event(
            ag,
            ix=f"{prefix}_{i}",
            source="src_oph",
            symbols=cue,
            state=state,
            body=body,
            latent=DEFAULT_LATENT,
        )
        op = (out.get("action") or {}).get("op") or "NONE"
        hist[op] = hist.get(op, 0) + 1
    return hist


def _nonhold_rate(ag: NeuralCortex, n: int, cue: list[str], *, prefix: str = "nh") -> float:
    if n <= 0:
        return 0.0
    hist = _op_histogram(ag, n, cue, prefix=prefix)
    holds = hist.get("HOLD", 0)
    return float(n - holds) / float(n)


def _retrieval_buf_norm(ag: NeuralCortex) -> float:
    ag._do_retrieve()
    ag._commit_pending_retrieve()
    return float(torch.linalg.vector_norm(ag.retrieval_buffer).item())


def _nonhold_count(ag: NeuralCortex, n: int, cue: list[str], *, prefix: str = "nhc") -> int:
    hist = _op_histogram(ag, n, cue, prefix=prefix)
    return int(n - hist.get("HOLD", 0))


def score_d0(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    """Population-level absence at birth (before teaching)."""
    s_empty = len(ag.memory.records()) == 0
    preloaded = False
    rng = np.random.default_rng(seeds.seed_permute ^ 0xD0)
    target_hits = 0
    foil_hits = 0
    hold_n = 0
    for i in range(D0_N_PROBES):
        order = [toks["a"], toks["foil"]]
        if rng.integers(0, 2) == 1:
            order = list(reversed(order))
        out = ag.observe(
            build_observe(
                interaction_token=f"d0_{i}",
                source_token="src_d0",
                ordered_symbols=order,
                observable_state=["st_idle"],
                body_state=BODY0,
            )
        )
        act = out.get("action") or {}
        tok = act.get("token")
        op = act.get("op")
        emit = act.get("emit_sequence") or []
        if op == "HOLD" or (op == "STOP" and not emit):
            hold_n += 1
        chosen = tok
        if emit:
            chosen = emit[0]
        if chosen == toks["a"]:
            target_hits += 1
        elif chosen == toks["foil"]:
            foil_hits += 1
        if op == "STOP" and emit == [toks["emit1"], toks["emit2"]]:
            preloaded = True

    n = D0_N_PROBES
    decisive = target_hits + foil_hits
    rate = target_hits / n
    foil_rate = foil_hits / n
    p_value = _binom_sf(target_hits, n, D0_P0)
    foil_p_value = _binom_sf(foil_hits, n, D0_P0)
    above_chance = p_value < D0_ALPHA
    foil_above_chance = foil_p_value < D0_ALPHA
    # pass only when neither target nor foil is significantly preferred
    ok = (not above_chance) and (not foil_above_chance) and s_empty and (not preloaded)
    return {
        "stage": "D0",
        "ok": ok,
        "target_hits": target_hits,
        "foil_hits": foil_hits,
        "hold_n": hold_n,
        "decisive": decisive,
        "rate": rate,
        "foil_rate": foil_rate,
        "p_value": p_value,
        "foil_p_value": foil_p_value,
        "above_chance": above_chance,
        "foil_above_chance": foil_above_chance,
        "s_empty": s_empty,
        "preloaded": preloaded,
        "chance": d0_chance_spec(),
    }


def score_d1(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    teach_loop(ag, seeds, n=80, symbols_fn=lambda i, rng: [toks["a"], toks["b"]])
    counts = _act_token_counts(ag, toks, 40, [toks["a"]])
    press = counts.get("press", 0)
    harm = counts.get("harm", 0)
    s1, b1 = physics(BODY0, "press", DEFAULT_LATENT)
    s2, b2 = physics(BODY0, "harm", DEFAULT_LATENT)
    cf_differs = (s1 != s2) or (b1 != b2)
    ok = press >= 3 and press > harm and cf_differs
    return {"stage": "D1", "ok": ok, "press": press, "harm": harm, "cf_differs": cf_differs}


def score_d2(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    teach_loop(ag, seeds, n=60, symbols_fn=lambda i, rng: [toks["c"]])
    body = list(BODY0)
    state = ["st_idle"]
    holds = 0
    for i in range(30):
        out, state, body = apply_event(
            ag,
            ix=f"d2c_{i}",
            source="src_d2",
            symbols=[toks["c"]],
            state=state,
            body=body,
            latent=DEFAULT_LATENT,
        )
        if (out.get("action") or {}).get("op") == "HOLD":
            holds += 1
    w_before = ag.weight_hash()
    ag.reset_rho()
    w_after = ag.weight_hash()
    teach_loop(ag, seeds, n=40, symbols_fn=lambda i, rng: [toks["c"]])
    counts = _act_token_counts(ag, toks, 30, [toks["c"]])
    beneficial = counts.get("press", 0) + counts.get("get", 0)
    rho_ok = w_before == w_after
    ok = rho_ok and beneficial >= 3 and holds >= 5
    return {
        "stage": "D2",
        "ok": ok,
        "holds_during_conflict": holds,
        "rho_reset_preserves_weights": rho_ok,
        "beneficial_act": beneficial,
        "counts": counts,
    }


def score_d3(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    teach_loop(
        ag,
        seeds,
        n=80,
        symbols_fn=lambda i, rng: [toks["rel_l"], toks["rel_r"]]
        if i % 3
        else [toks["distr"], toks["rel_l"], toks["rel_r"]],
    )
    equal_holds = 0
    for i in range(20):
        out = ag.observe(
            build_observe(
                interaction_token=f"d3eq_{i}",
                source_token="src_d3",
                ordered_symbols=[toks["rel_l"], toks["rel_r"]],
                observable_state=["st_idle"],
                body_state=BODY0,
            )
        )
        if (out.get("action") or {}).get("op") == "HOLD":
            equal_holds += 1
    clear_hist = _op_histogram(ag, 20, [toks["rel_l"]], prefix="d3clear")
    clear_nonhold = 20 - clear_hist.get("HOLD", 0)
    clear_hold = clear_hist.get("HOLD", 0)
    distr_hist = _op_histogram(ag, 20, [toks["distr"]], prefix="d3distr")
    distr_hold = distr_hist.get("HOLD", 0)
    # Relation cue must be more actionable than distractor-only; equal-HOLD everywhere is not a pass.
    distr_nonhold = 20 - distr_hold
    ok = (
        equal_holds >= 8
        and clear_nonhold >= 5
        and clear_nonhold > distr_nonhold
        and distr_hold > clear_hold
    )
    return {
        "stage": "D3",
        "ok": ok,
        "holds": equal_holds,
        "equal_holds": equal_holds,
        "clear_nonhold": clear_nonhold,
        "clear_hold": clear_hold,
        "distractor_hold": distr_hold,
        "distractor_nonhold": distr_nonhold,
    }


def score_d4(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    teach_loop(ag, seeds, n=40, symbols_fn=lambda i, rng: [toks["fact"]])
    wrote = [
        r
        for r in ag.memory.records()
        if r.source == "cortex_write" and float(np.linalg.norm(np.asarray(r.content, dtype=np.float64))) > 0.0
    ]
    if not wrote:
        return {"stage": "D4", "ok": False, "why": "no_organism_write"}
    fact_id = wrote[-1].fact_id
    content = list(wrote[-1].content)
    content_norm = float(np.linalg.norm(content))
    ag.reset_rho()
    teach_loop(ag, seeds, n=20, symbols_fn=lambda i, rng: [toks["distr"]])
    ag.reset_rho()
    persisted = fact_id in {r.fact_id for r in ag.memory.records()}
    ag.memory.delete(fact_id)
    stripped = fact_id not in {r.fact_id for r in ag.memory.records()}
    stripped_norm = _retrieval_buf_norm(ag)
    donor_vec = list(np.random.default_rng(seeds.seed_registry).normal(0, 1, size=ag.genome.d_sym))
    ag.memory.write(
        CortexRecord(
            fact_id="donor_fact",
            content=donor_vec,
            when=ag.age,
            interaction_token="donor",
            source_token="src_donor",
            source="cortex_write",
        )
    )
    donor_ok = any(r.fact_id == "donor_fact" for r in ag.memory.records())
    donor_norm = _retrieval_buf_norm(ag)
    donor_changes = donor_norm > stripped_norm + 1e-6
    ok = (
        content_norm > 0.0
        and persisted
        and stripped
        and donor_ok
        and donor_changes
    )
    return {
        "stage": "D4",
        "ok": ok,
        "persisted": persisted,
        "stripped": stripped,
        "donor_ok": donor_ok,
        "donor_changes_retrieval": donor_changes,
        "stripped_buf_norm": stripped_norm,
        "donor_buf_norm": donor_norm,
        "n_writes": len(wrote),
        "content_norm": content_norm,
    }


def score_d5(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    teach_loop(ag, seeds, n=50, symbols_fn=lambda i, rng: [toks["ground"]])
    unknown = _tok("sym", seeds, "unknown_x")
    unknown_holds = 0
    n_u = 20
    for i in range(n_u):
        out = ag.observe(
            build_observe(
                interaction_token=f"d5u_{i}",
                source_token="src_d5",
                ordered_symbols=[unknown],
                observable_state=["st_idle"],
                body_state=BODY0,
            )
        )
        if (out.get("action") or {}).get("op") == "HOLD":
            unknown_holds += 1
    unknown_nonhold_rate = float(n_u - unknown_holds) / float(n_u)
    known_nonhold_rate = _nonhold_rate(ag, 20, [toks["ground"]], prefix="d5k")
    ok = (
        unknown_holds >= 12
        and known_nonhold_rate >= 0.30
        and known_nonhold_rate >= unknown_nonhold_rate + 0.15
    )
    return {
        "stage": "D5",
        "ok": ok,
        "unknown_holds": unknown_holds,
        "unknown_nonhold_rate": unknown_nonhold_rate,
        "known_nonhold_rate": known_nonhold_rate,
    }


def score_d6(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    taught = ([toks["emit1"]], [toks["emit1"], toks["emit2"]])
    teach_loop(
        ag,
        seeds,
        n=60,
        symbols_fn=lambda i, rng: [toks["emit1"]] if i % 2 == 0 else [toks["emit1"], toks["emit2"]],
    )
    emits = matched = holds = 0
    for i in range(30):
        out = ag.observe(
            build_observe(
                interaction_token=f"d6_{i}",
                source_token="src_d6",
                ordered_symbols=[toks["emit1"]],
                observable_state=["st_idle"],
                body_state=BODY0,
            )
        )
        act = out.get("action") or {}
        op = act.get("op")
        seq = list(act.get("emit_sequence") or [])
        if seq:
            emits += 1
            if seq in taught:
                matched += 1
        if op == "HOLD":
            holds += 1
    ok = emits >= 2 and matched >= 1 and holds >= 3
    return {"stage": "D6", "ok": ok, "emits": emits, "matched": matched, "holds": holds}


def score_d7(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    for L in (1, 2, 4, 8):
        teach_loop(
            ag,
            seeds,
            n=15,
            symbols_fn=lambda i, rng, L=L: [toks["emit1"]] * min(L, 2)
            + ([toks["emit2"]] if L >= 2 else []),
        )
    lengths: set[int] = set()
    terminals = 0
    max_len = 0
    curriculum_hits = 0
    n_probe = 20
    for i in range(n_probe):
        out = ag.observe(
            build_observe(
                interaction_token=f"d7_{i}",
                source_token="src_d7",
                ordered_symbols=[toks["emit1"], toks["emit2"]],
                observable_state=["st_idle"],
                body_state=BODY0,
            )
        )
        act = out.get("action") or {}
        seq = act.get("emit_sequence") or []
        op = act.get("op")
        L = len(seq)
        if L:
            lengths.add(L)
            max_len = max(max_len, L)
            if all(t in {toks["emit1"], toks["emit2"]} for t in seq):
                curriculum_hits += 1
        if op in {"HOLD", "STOP"}:
            terminals += 1
    # Variable-length: must reach length ≥4 (taught 1→2→4→8), not merely random 1–2 EMITs.
    ok = (
        len(lengths) >= 2
        and max_len >= 4
        and max_len <= 8
        and terminals >= 15
        and curriculum_hits >= 2
    )
    return {
        "stage": "D7",
        "ok": ok,
        "distinct_lengths": sorted(lengths),
        "terminals": terminals,
        "max_length": max_len,
        "curriculum_hits": curriculum_hits,
    }


def score_d8(
    ag: NeuralCortex,
    seeds: SeedLike,
    toks: dict[str, str],
    *,
    withheld: bool = True,
) -> dict[str, Any]:
    teach_loop(ag, seeds, n=40, symbols_fn=lambda i, rng: [toks["comp_x"], toks["comp_y"]])
    teach_loop(ag, seeds, n=40, symbols_fn=lambda i, rng: [toks["comp_y"], toks["comp_z"]])
    hits = 0
    for i in range(25):
        out = ag.observe(
            build_observe(
                interaction_token=f"d8_{i}",
                source_token="src_d8",
                ordered_symbols=[toks["comp_x"]],
                observable_state=["st_idle"],
                body_state=BODY0,
            )
        )
        seq = (out.get("action") or {}).get("emit_sequence") or []
        if seq == [toks["comp_x"], toks["comp_z"]]:
            hits += 1
    ok = hits >= 1 if withheld else True
    return {"stage": "D8", "ok": ok, "target_hits": hits, "withheld": withheld}


def score_d9(ag: NeuralCortex, seeds: SeedLike, toks: dict[str, str]) -> dict[str, Any]:
    old_before = _nonhold_count(ag, 20, [toks["a"]], prefix="d9ob")
    teach_loop(ag, seeds, n=40, symbols_fn=lambda i, rng: [toks["domain2"]])
    old_nonhold = _nonhold_count(ag, 20, [toks["a"]], prefix="d9old")
    new_nonhold = _nonhold_count(ag, 20, [toks["domain2"]], prefix="d9new")
    # Retention: both domains active above exploration floor; old domain not collapsed after new teaching.
    retained = old_nonhold >= max(8, int(0.5 * old_before)) if old_before > 0 else old_nonhold >= 8
    ok = retained and new_nonhold >= 8
    return {
        "stage": "D9",
        "ok": ok,
        "old_nonhold_before": old_before,
        "old_nonhold": old_nonhold,
        "new_nonhold": new_nonhold,
        "retained": retained,
    }


def score_d10(
    ag: NeuralCortex,
    seeds: SeedLike,
    toks: dict[str, str],
    child_ckpt: dict[str, Any],
    mature_ckpt: dict[str, Any],
) -> dict[str, Any]:
    """Adult vs child probe score on same cue."""

    def probe_score(ckpt: dict[str, Any]) -> float:
        with tempfile.TemporaryDirectory(prefix="d10_") as tmp:
            ag2 = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device=ag.device)
            ag2.load_checkpoint(ckpt)
            counts = _act_token_counts(ag2, toks, 30, [toks["a"]])
            return float(counts.get("press", 0) + counts.get("get", 0))

    child_s = probe_score(child_ckpt)
    adult_s = probe_score(mature_ckpt)
    w0 = ag.weight_hash()
    ag.reset_rho()
    w1 = ag.weight_hash()
    rho_ok = w0 == w1
    ok = adult_s > child_s and adult_s >= 3 and rho_ok
    return {
        "stage": "D10",
        "ok": ok,
        "child_score": child_s,
        "adult_score": adult_s,
        "rho_reset_preserves_weights": rho_ok,
    }


def score_d11_twin_follows(
    main_result: dict[str, Any],
    twin_result: dict[str, Any],
    *,
    cross_lexicon: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Twin must follow twin lexicon, not main spellings.

    `both_fail_d8` alone is insufficient (D8 is currently never earned).
    Require renamed tokens plus a cross-lexicon behavioral probe:
    twin nonhold on twin cue > twin nonhold on main's spelling of the same role.
    """
    main_toks = main_result.get("tokens") or {}
    twin_toks = twin_result.get("tokens") or {}
    # Prefer flat role keys; ignore nested "all" map if present.
    main_a = main_toks.get("a")
    twin_a = twin_toks.get("a")
    twin_flat = {k: v for k, v in twin_toks.items() if isinstance(v, str)}
    renamed = (
        isinstance(main_a, str)
        and isinstance(twin_a, str)
        and main_a != twin_a
        and main_a not in twin_flat.values()
    )

    def _ok_vec(life: dict[str, Any]) -> tuple[bool | None, ...]:
        stages = life.get("stages") or {}
        return tuple((stages.get(s) or {}).get("ok") for s in STAGES if s != "D11")

    vectors_differ = _ok_vec(main_result) != _ok_vec(twin_result)
    cross = cross_lexicon or {}
    twin_on_own = float(cross.get("twin_nonhold_own_cue", 0.0))
    twin_on_main = float(cross.get("twin_nonhold_main_spelling", 0.0))
    follows_own = twin_on_own > twin_on_main + 0.05
    ok = (
        main_result.get("role") == "main"
        and twin_result.get("role") == "twin"
        and renamed
        and follows_own
    )
    return {
        "stage": "D11",
        "ok": bool(ok),
        "renamed": renamed,
        "vectors_differ": vectors_differ,
        "follows_own_lexicon": follows_own,
        "twin_nonhold_own_cue": twin_on_own,
        "twin_nonhold_main_spelling": twin_on_main,
        "twin_token_a": twin_toks.get("a"),
    }


def score_d12_forks(
    mature: NeuralCortex,
    seeds: SeedLike,
    toks: dict[str, str],
    birth_genome: GenomeConfig,
    device: str,
) -> dict[str, Any]:
    """Required cortex/S separation probes (behavioral)."""
    mature_ckpt = mature.checkpoint()
    mature_s = mature.memory.snapshot()
    mature_wh = mature.weight_hash()
    mature_n_s = len(mature_s)
    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="d12_") as tmp:
        root = Path(tmp)

        mprobe = make_cortex(root / "s0", genome=seeds.genome(), device=device)
        mprobe.load_checkpoint(mature_ckpt)
        mc = _act_token_counts(mprobe, toks, 20, [toks["a"]])
        mature_skill = float(mc.get("press", 0) + mc.get("get", 0))
        # Vacuous 0==0 skill comparisons are not developmental evidence.
        has_skill = mature_skill >= 3.0

        # 1) mature cortex + stripped facts → facts gone, acquisition skill retained
        a = make_cortex(root / "s1", genome=seeds.genome(), device=device)
        a.load_checkpoint(mature_ckpt)
        for fid in [r.fact_id for r in a.memory.records()]:
            a.memory.delete(fid)
        buf1 = _retrieval_buf_norm(a)
        a_probe = make_cortex(root / "s1p", genome=seeds.genome(), device=device)
        a_probe.load_checkpoint(a.checkpoint())
        ac = _act_token_counts(a_probe, toks, 20, [toks["a"]])
        strip_skill = float(ac.get("press", 0) + ac.get("get", 0))
        results["strip_facts"] = {
            "ok": has_skill
            and len(a.memory.records()) == 0
            and a.weight_hash() == mature_wh
            and buf1 == 0.0
            and strip_skill >= max(3.0, 0.5 * mature_skill),
            "skill_retained": strip_skill,
            "mature_skill": mature_skill,
            "retrieve_buf": buf1,
            "s_empty": len(a.memory.records()) == 0,
        }

        # 2) birth cortex + adult S — mature skill must beat birth skill; no vacuous 0>=0
        b = make_cortex(root / "s2", genome=birth_genome, device=device)
        b.memory.restore(mature_s)
        s_ok_b = len(b.memory.records()) == mature_n_s
        b_probe = make_cortex(root / "s2p", genome=birth_genome, device=device)
        b_probe.memory.restore(mature_s)
        # copy vocab from mature checkpoint onto birth body for fair cue probes
        b_probe.vocab = {
            k: np.asarray(v, dtype="float64")
            for k, v in (mature_ckpt.get("vocab") or {}).items()
        }
        bc = _act_token_counts(b_probe, toks, 20, [toks["a"]])
        birth_skill = float(bc.get("press", 0) + bc.get("get", 0))
        results["birth_cortex_adult_s"] = {
            "ok": has_skill
            and b.weight_hash() != mature_wh
            and s_ok_b
            and mature_skill > birth_skill,
            "n_s": len(b.memory.records()),
            "mature_skill": mature_skill,
            "birth_skill": birth_skill,
            "s_count_ok": s_ok_b,
        }

        # 3) donor S
        c = make_cortex(root / "s3", genome=seeds.genome(), device=device)
        c.load_checkpoint(mature_ckpt)
        c.memory.clear()
        donor_vec = list(np.random.default_rng(seeds.seed_registry ^ 0xD12).normal(0, 1, size=c.genome.d_sym))
        c.memory.restore(
            [
                {
                    "fact_id": "donor_only",
                    "content": donor_vec,
                    "when": 0,
                    "interaction_token": "d",
                    "source_token": "src",
                    "source": "cortex_write",
                    "tags": {},
                }
            ]
        )
        buf3 = _retrieval_buf_norm(c)
        results["donor_s"] = {
            "ok": [r.fact_id for r in c.memory.records()] == ["donor_only"] and buf3 > 0.0,
            "retrieve_buf": buf3,
        }

        # 4) donor mature cortex + fresh S — cortical identity retained without S
        d = make_cortex(root / "s4", genome=seeds.genome(), device=device)
        d.load_checkpoint(mature_ckpt)
        d.memory.clear()
        results["donor_cortex_fresh_s"] = {
            "ok": has_skill and len(d.memory.records()) == 0 and d.weight_hash() == mature_wh,
            "mature_skill": mature_skill,
        }

        # 5) reset cortical weights, keep S — competence must strictly decrease
        e = make_cortex(root / "s5", genome=seeds.genome(), device=device)
        e.load_checkpoint(mature_ckpt)
        s_keep = e.memory.snapshot()
        # Probe skill on a disposable clone so S is not polluted before the fork check.
        e_probe = make_cortex(root / "s5p", genome=seeds.genome(), device=device)
        e_probe.load_checkpoint(mature_ckpt)
        before_c = _act_token_counts(e_probe, toks, 20, [toks["a"]])
        skill_before = float(before_c.get("press", 0) + before_c.get("get", 0))
        e.reset_cortex()
        e.memory.restore(s_keep)
        s_ok = len(e.memory.records()) == len(s_keep)
        weights_changed = e.weight_hash() != mature_wh
        e_after = make_cortex(root / "s5a", genome=seeds.genome(), device=device)
        e_after.load_checkpoint(e.checkpoint())
        after_c = _act_token_counts(e_after, toks, 20, [toks["a"]])
        skill_after = float(after_c.get("press", 0) + after_c.get("get", 0))
        results["reset_w_keep_s"] = {
            "ok": has_skill
            and weights_changed
            and s_ok
            and skill_after < skill_before,
            "skill_before": skill_before,
            "skill_after": skill_after,
            "weights_changed": weights_changed,
            "s_count_ok": s_ok,
            "n_s": len(e.memory.records()),
            "n_keep": len(s_keep),
        }

        # 6) reset rho only — mature competence remains
        f = make_cortex(root / "s6", genome=seeds.genome(), device=device)
        f.load_checkpoint(mature_ckpt)
        wh = f.weight_hash()
        f.reset_rho()
        f_probe = make_cortex(root / "s6p", genome=seeds.genome(), device=device)
        f_probe.load_checkpoint(f.checkpoint())
        fc = _act_token_counts(f_probe, toks, 20, [toks["a"]])
        rho_skill = float(fc.get("press", 0) + fc.get("get", 0))
        results["reset_rho_only"] = {
            "ok": has_skill
            and f.weight_hash() == wh
            and rho_skill >= max(3.0, 0.5 * mature_skill),
            "rho_skill": rho_skill,
            "mature_skill": mature_skill,
        }

    ok = all(v.get("ok") for v in results.values())
    return {"stage": "D12", "ok": ok, "forks": results}
