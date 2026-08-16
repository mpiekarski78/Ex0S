"""TM.0.23.CORTEX.DEVELOP — scored D0–D12 apparatus (no neural mechanism edits).

Teaching comes only from fixtures + frozen physics. scorer_only never credits actions.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import (
    build_observe,
    make_cortex,
    physics,
)
from three_memory.cortex_memory import CortexRecord
from three_memory.neural_cortex import GenomeConfig, NeuralCortex

# --- frozen D0 chance model (must match cortex_development_contract.md) ---
D0_N_PROBES = 64
D0_P0 = 0.5
D0_ALPHA = 0.01

STAGES = [f"D{i}" for i in range(13)]

# Role → physical effect templates (physics only; cortex never sees role names as innate lexicon)
ROLE_ACT_EFFECTS = {
    "press": {"state": ["st_pressed"], "delta": [0.25, -0.1, 0.15, 0.0]},
    "harm": {"state": ["st_hurt"], "delta": [-0.35, 0.45, -0.15, 0.0]},
    "idle": {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
    "get": {"state": ["st_got"], "delta": [0.1, 0.0, 0.2, 0.0]},
    "drop": {"state": ["st_drop"], "delta": [-0.05, 0.05, -0.2, 0.0]},
}

BODY0 = [0.5, 0.25, 0.5, 0.0]

MOTOR_ROLES = ("press", "harm", "get", "drop")


def motor_latent(toks: dict[str, str]) -> dict[str, Any]:
    """Map opaque handle IDs → frozen physics effects (environment-side only)."""
    effects = {
        toks[role]: dict(ROLE_ACT_EFFECTS[role])
        for role in MOTOR_ROLES
        if role in toks
    }
    idle_id = toks.get("idle", "idle")
    effects[idle_id] = dict(ROLE_ACT_EFFECTS["idle"])
    return {"act_effects": effects}


# Backward-compatible alias for fixtures that still use literal press/harm keys
DEFAULT_LATENT = {
    "act_effects": {k: dict(v) for k, v in ROLE_ACT_EFFECTS.items()}
}


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X~Binomial(n,p), exact sum (n<=64)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # sum_{i=k..n} C(n,i) p^i (1-p)^{n-i}
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


@dataclass
class LifeSeeds:
    pair_id: int
    role: str  # main | twin
    seed_birth: int
    seed_registry: int
    seed_source: int
    seed_action: int
    seed_permute: int
    seed_motor: int = 0

    def genome(self) -> GenomeConfig:
        return GenomeConfig(
            seed_birth=self.seed_birth,
            seed_registry=self.seed_registry,
            seed_source=self.seed_source,
            seed_action=self.seed_action,
            seed_permute=self.seed_permute,
            seed_motor=self.seed_motor or (self.seed_registry ^ 0x4D07),
        )


def pair_seeds(pair_id: int) -> tuple[LifeSeeds, LifeSeeds]:
    """Frozen 16-pair table derivation (independent of eval reveal seed)."""
    base = 10_000 + pair_id * 97
    main = LifeSeeds(
        pair_id=pair_id,
        role="main",
        seed_birth=base + 1,
        seed_registry=base + 2,
        seed_source=base + 3,
        seed_action=base + 4,
        seed_permute=base + 5,
        seed_motor=base + 6,
    )
    twin = LifeSeeds(
        pair_id=pair_id,
        role="twin",
        seed_birth=base + 11,
        seed_registry=base + 12,
        seed_source=base + 13,
        seed_action=base + 14,
        seed_permute=base + 15,
        seed_motor=base + 16,
    )
    return main, twin


def development_seed_table(n_pairs: int = 16) -> list[dict[str, Any]]:
    rows = []
    for i in range(n_pairs):
        m, t = pair_seeds(i)
        rows.append(
            {
                "pair_id": i,
                "main": m.__dict__,
                "twin": t.__dict__,
            }
        )
    return rows


def _tok(prefix: str, seeds: LifeSeeds, name: str) -> str:
    """Token spelling for main; twin gets renamed spelling with same role."""
    if seeds.role == "main":
        return f"{prefix}_{name}"
    # twin renamed
    h = hashlib.sha256(f"{seeds.seed_registry}:{name}".encode()).hexdigest()[:8]
    return f"tw_{h}_{name}"


def _motor_handle(seeds: LifeSeeds, role: str) -> str:
    """Opaque environment handle ID (not English meaning; not neural input)."""
    h = hashlib.sha256(f"motor:{seeds.seed_registry}:{role}".encode()).hexdigest()[:12]
    return f"h_{h}"


def curriculum_tokens(seeds: LifeSeeds) -> dict[str, str]:
    return {
        "a": _tok("sym", seeds, "a"),
        "b": _tok("sym", seeds, "b"),
        "c": _tok("sym", seeds, "c"),
        "rel_l": _tok("sym", seeds, "rel_l"),
        "rel_r": _tok("sym", seeds, "rel_r"),
        "distr": _tok("sym", seeds, "distr"),
        "fact": _tok("sym", seeds, "fact"),
        "ground": _tok("sym", seeds, "ground"),
        "emit1": _tok("sym", seeds, "emit1"),
        "emit2": _tok("sym", seeds, "emit2"),
        "comp_x": _tok("sym", seeds, "comp_x"),
        "comp_y": _tok("sym", seeds, "comp_y"),
        "comp_z": _tok("sym", seeds, "comp_z"),
        "domain2": _tok("sym", seeds, "domain2"),
        "foil": _tok("sym", seeds, "foil"),
        "press": _motor_handle(seeds, "press"),
        "harm": _motor_handle(seeds, "harm"),
        "get": _motor_handle(seeds, "get"),
        "drop": _motor_handle(seeds, "drop"),
        "idle": _motor_handle(seeds, "idle"),
    }


def bind_life_actuators(ag: NeuralCortex, toks: dict[str, str]) -> list[str]:
    """Attach opaque motor handles via cortex-internal motor-registry RNG."""
    handles = [toks[r] for r in MOTOR_ROLES]
    if not hasattr(ag, "bind_actuators"):
        raise RuntimeError("candidate lacks bind_actuators — refuse planted lexicon path")
    ag.bind_actuators(handles)
    return handles


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
    """Observe then, if ACT, advance world via frozen physics for next body/state."""
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
    seeds: LifeSeeds,
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
        # permute presentation order for twin schedules
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



from experiments.cortex_develop_scorers import (
    score_d0,
    score_d1,
    score_d2,
    score_d3,
    score_d4,
    score_d5,
    score_d6,
    score_d7,
    score_d8,
    score_d9,
    score_d10,
    score_d11_twin_follows,
    score_d12_forks,
)



def run_one_life(
    seeds: LifeSeeds,
    *,
    device: str = "cpu",
    tmp: Path | None = None,
) -> dict[str, Any]:
    own_tmp = tmp is None
    if own_tmp:
        td = tempfile.TemporaryDirectory(prefix=f"life_{seeds.role}_")
        tmp = Path(td.name)
    assert tmp is not None
    ag = make_cortex(tmp / "s", genome=seeds.genome(), device=device)
    toks = curriculum_tokens(seeds)
    stages: dict[str, Any] = {}

    # D0 at birth (before actuator binding)
    stages["D0"] = score_d0(ag, seeds, toks)
    bind_life_actuators(ag, toks)

    # early child checkpoint after light teaching
    teach_loop(
        ag,
        seeds,
        n=30,
        symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
        latent=motor_latent(toks),
    )
    child_ckpt = ag.checkpoint()

    stages["D1"] = score_d1(ag, seeds, toks)
    stages["D2"] = score_d2(ag, seeds, toks)
    stages["D3"] = score_d3(ag, seeds, toks)
    stages["D4"] = score_d4(ag, seeds, toks)
    stages["D5"] = score_d5(ag, seeds, toks)
    stages["D6"] = score_d6(ag, seeds, toks)
    stages["D7"] = score_d7(ag, seeds, toks)
    stages["D8"] = score_d8(ag, seeds, toks, withheld=True)
    stages["D9"] = score_d9(ag, seeds, toks)

    mature_ckpt = ag.checkpoint()
    stages["D10"] = score_d10(ag, seeds, toks, child_ckpt, mature_ckpt)

    birth_g = GenomeConfig(
        seed_birth=seeds.seed_birth,
        seed_registry=seeds.seed_registry,
        seed_source=seeds.seed_source,
        seed_action=seeds.seed_action,
        seed_permute=seeds.seed_permute,
        seed_motor=seeds.seed_motor,
    )
    stages["D12"] = score_d12_forks(ag, seeds, toks, birth_g, device)

    last_clear = None
    first_fail = None
    for sid in STAGES:
        if sid == "D11":
            continue  # pair-level
        st = stages.get(sid)
        if st is None:
            continue
        if st.get("ok"):
            if first_fail is None:
                last_clear = sid
        elif first_fail is None:
            first_fail = sid

    out = {
        "pair_id": seeds.pair_id,
        "role": seeds.role,
        "tokens": {
            "a": toks["a"],
            "foil": toks["foil"],
            "emit1": toks["emit1"],
            "all": toks,
        },
        "stages": stages,
        "last_stage_clear": last_clear,
        "first_fail": first_fail,
        "d10_adult_gt_child": bool(stages["D10"].get("ok")),
        "all_required_clear": all(
            stages[s].get("ok") for s in STAGES if s != "D11"
        ),
        "device": device,
        "weight_hash": ag.weight_hash(),
        "age": ag.age,
        "mature_checkpoint": mature_ckpt,
        "seeds": seeds.__dict__,
    }
    if own_tmp:
        td.cleanup()  # type: ignore[name-defined]
    return out


def _cross_lexicon_probe(twin: dict[str, Any], main: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Twin agent: nonhold on own cue vs main's spelling of the same role."""
    from experiments.cortex_develop_scorers import _nonhold_rate

    twin_toks = (twin.get("tokens") or {}).get("all") or twin.get("tokens") or {}
    main_toks = (main.get("tokens") or {}).get("all") or main.get("tokens") or {}
    ckpt = twin.get("mature_checkpoint")
    seeds = twin.get("seeds") or {}
    g = GenomeConfig(
        seed_birth=int(seeds["seed_birth"]),
        seed_registry=int(seeds["seed_registry"]),
        seed_source=int(seeds["seed_source"]),
        seed_action=int(seeds["seed_action"]),
        seed_permute=int(seeds["seed_permute"]),
    )
    with tempfile.TemporaryDirectory(prefix="d11x_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=g, device=device)
        ag.load_checkpoint(ckpt)
        own = _nonhold_rate(ag, 20, [twin_toks["a"]], prefix="d11own")
        ag.reset_rho()
        foreign = _nonhold_rate(ag, 20, [main_toks["a"]], prefix="d11main")
    return {
        "twin_nonhold_own_cue": own,
        "twin_nonhold_main_spelling": foreign,
    }


def run_pair(pair_id: int, *, device: str = "cpu") -> dict[str, Any]:
    main_s, twin_s = pair_seeds(pair_id)
    main = run_one_life(main_s, device=device)
    twin = run_one_life(twin_s, device=device)
    cross = _cross_lexicon_probe(twin, main, device=device)
    d11 = score_d11_twin_follows(main, twin, cross_lexicon=cross)
    main["stages"]["D11"] = d11
    twin["stages"]["D11"] = d11
    # Drop bulky checkpoints from recorded results (SHA/age remain).
    for life in (main, twin):
        life.pop("mature_checkpoint", None)
        if isinstance(life.get("tokens"), dict):
            life["tokens"].pop("all", None)

    def full_clear(life: dict[str, Any]) -> bool:
        st = life["stages"]
        return all(st[s].get("ok") for s in STAGES)

    main_clear = full_clear(main)
    twin_clear = full_clear(twin)
    pair_clear = main_clear and twin_clear
    return {
        "pair_id": pair_id,
        "main": main,
        "twin": twin,
        "pair_clear": pair_clear,
        "maturation_main": main.get("d10_adult_gt_child"),
        "maturation_twin": twin.get("d10_adult_gt_child"),
        "maturation_pair": bool(main.get("d10_adult_gt_child") and twin.get("d10_adult_gt_child")),
        "d11_cross": cross,
    }


def run_battery(n_pairs: int = 16, *, device: str = "cpu", pair_ids: list[int] | None = None) -> dict[str, Any]:
    ids = pair_ids if pair_ids is not None else list(range(n_pairs))
    pairs = []
    for pid in ids:
        pairs.append(run_pair(pid, device=device))
    n = len(pairs)
    n_clear = sum(1 for p in pairs if p["pair_clear"])
    n_mat = sum(1 for p in pairs if p["maturation_pair"])
    development_gate_clear = n_clear >= 13 if n >= 16 else n_clear == n and n_clear >= max(1, int(math.ceil(0.8125 * n)))
    # For full 16: >=13/16. For smoke subsets: require all run pairs clear AND scaled threshold only when n==16.
    if n == 16:
        development_gate_clear = n_clear >= 13
        maturation_ok = n_mat >= 14
    else:
        development_gate_clear = False  # smoke never claims gate
        maturation_ok = False
    eligible = bool(development_gate_clear and maturation_ok)
    # per-stage distribution
    dist: dict[str, int] = {s: 0 for s in STAGES}
    for p in pairs:
        for role in ("main", "twin"):
            for s in STAGES:
                if p[role]["stages"].get(s, {}).get("ok"):
                    dist[s] += 1
    return {
        "n_pairs": n,
        "n_pair_clear": n_clear,
        "n_maturation": n_mat,
        "development_gate_clear": development_gate_clear,
        "eligible_for_000005": eligible,
        "earned_next": False,
        "ex0s": None,
        "product": "0.0.004",
        "stage_pass_counts_main_and_twin": dist,
        "pairs": pairs,
        "device": device,
    }


def generate_eval_fixture(seed_hex: str, salt_hex: str) -> dict[str, Any]:
    """Materialize held-out eval worlds from revealed 256-bit secrets."""
    seed = bytes.fromhex(seed_hex)
    salt = bytes.fromhex(salt_hex)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed + salt).digest()[:8], "big"))
    worlds = []
    for i in range(8):
        syms = [f"ev_{i}_{j}_{rng.integers(0, 1_000_000)}" for j in range(3)]
        worlds.append(
            {
                "world_id": f"eval_{i}",
                "split": "eval",
                "organism_events": [
                    {
                        "op": "observe",
                        "event": build_observe(
                            interaction_token=f"eval_{i}_ix0",
                            source_token=f"src_eval_{i}",
                            ordered_symbols=syms[:2],
                            observable_state=["st_idle"],
                            body_state=BODY0,
                        ),
                    }
                ],
                "scorer_only": {
                    "latent_structure": DEFAULT_LATENT,
                    "held_out": True,
                    "target_compose": syms,
                },
            }
        )
    return {
        "version": "TM.0.23.CORTEX.WORLDS.EVAL",
        "body_setpoint": [1.0, 0.0, 1.0, 0.0],
        "worlds": worlds,
        "note": "Materialized after candidate reveal. scorer_only must not enter observe.",
    }


def hygiene_eval(fixture: dict[str, Any]) -> dict[str, Any]:
    issues = []
    for w in fixture.get("worlds", []):
        if "organism_events" not in w or "scorer_only" not in w:
            issues.append(f"{w.get('world_id')}: missing split keys")
        for ev in w.get("organism_events", []):
            if ev.get("op") != "observe":
                continue
            event = ev.get("event") or {}
            if "homeostatic_delta" in event or "correct" in event:
                issues.append(f"{w.get('world_id')}: banned key in organism event")
            if set(event) - {
                "interaction_token",
                "source_token",
                "ordered_symbols",
                "observable_state",
                "body_state",
            }:
                issues.append(f"{w.get('world_id')}: extra observe keys")
            # ensure scorer_only not nested in event
            if "scorer_only" in event:
                issues.append(f"{w.get('world_id')}: scorer_only inside event")
    return {"ok": not issues, "issues": issues}


def diagnostic_capacity_smoke(device: str = "cpu") -> dict[str, Any]:
    """Diagnostic only — not eligibility."""
    lanes = []
    for n in (32, 64):
        with tempfile.TemporaryDirectory(prefix="cap_") as tmp:
            g = GenomeConfig(n=n, seed_birth=1, seed_registry=2, seed_source=3, seed_action=4, seed_permute=5)
            ag = make_cortex(Path(tmp) / "s", genome=g, device=device)
            out = ag.observe(
                build_observe(
                    interaction_token="c0",
                    source_token="src",
                    ordered_symbols=["x"],
                    observable_state=["st_idle"],
                    body_state=BODY0,
                )
            )
            lanes.append({"n": n, "ok": bool(out.get("ok")), "action": out.get("action")})
    return {"diagnostic": True, "architecture_birth_smoke": lanes, "eligibility_relevant": False}


def diagnostic_wall(device: str = "cpu") -> dict[str, Any]:
    """ABI-only neural parity wall — diagnostic; need not pass."""
    probes = ["W_persist", "W_inquire", "W_reliability", "W_perspective", "W_interpret", "W_honesty"]
    results = []
    first_fail = None
    with tempfile.TemporaryDirectory(prefix="wall_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device=device)
        for pid in probes:
            # simplified: present opaque probe symbols; do not call 0.0.004 mechanisms
            out = ag.observe(
                build_observe(
                    interaction_token=pid,
                    source_token="wall_src",
                    ordered_symbols=[pid.lower(), "probe"],
                    observable_state=["st_idle"],
                    body_state=BODY0,
                )
            )
            # wall "pass" would require phenotype-specific behavior we do not claim
            ok = False
            results.append({"id": pid, "ok": ok, "action": out.get("action"), "why": "diagnostic_expected_fail"})
            if first_fail is None:
                first_fail = pid
    return {
        "diagnostic": True,
        "need_not_fully_pass": True,
        "first_fail_neural_wall": first_fail,
        "results": results,
        "eligibility_relevant": False,
        "cannot_negate_development_gate": True,
    }
