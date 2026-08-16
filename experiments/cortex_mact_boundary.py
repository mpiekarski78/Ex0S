"""TM.0.23.CORTEX.MACT.BOUNDARY — v1 (planted v4) + v2 (bind_actuators) controls."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cortex_develop_life import (
    BODY0,
    DEFAULT_LATENT,
    apply_event,
    bind_life_actuators,
    curriculum_tokens,
    motor_latent,
    pair_seeds,
    teach_loop,
)
from experiments.cortex_develop_scorers import score_d1, _act_token_counts
from experiments.run_tm023cortex import make_cortex, torch_env
from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OPS, OP_COST, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "docs" / "cortex_mact_boundary_contract.md"
CANDIDATE_V4 = REPO_ROOT / "docs" / "cortex.candidate.v4.lock"
CANDIDATE_V5 = REPO_ROOT / "docs" / "cortex.candidate.v5.lock"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
BOUNDARY_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.lock"
BOUNDARY_MD = REPO_ROOT / "docs" / "tm023cortex_mact_boundary_results.md"
BOUNDARY_V5_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.v5.lock"
BOUNDARY_V5_MD = REPO_ROOT / "docs" / "tm023cortex_mact_boundary_v5_results.md"
RUNNER_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.runner.lock"
RUNNER_V2_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.runner.v2.lock"

SWAP_REVISE_EPISODES = 40


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_plasticity(ag: NeuralCortex) -> None:
    ag._plasticity_off = True  # type: ignore[attr-defined]

    def _noop_credit(s_t, body_t):  # noqa: ANN001
        ag._pending = None
        return {"adv": 0.0, "pred_err": 0.0}

    ag._apply_credit = _noop_credit  # type: ignore[method-assign]


def _pref_counts(ag: NeuralCortex, toks: dict[str, str], n: int, cue: list[str], latent) -> dict[str, int]:
    return _act_token_counts(ag, toks, n, cue, latent=latent)


# --- v1 controls (planted lexicon inspection; historical) ---


def control_c1_env_exposed_v4() -> dict[str, Any]:
    planted = set(MOTOR_ACT_TOKENS) & {"press", "harm"}
    with tempfile.TemporaryDirectory(prefix="mact_c1_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        birth_keys = set(ag.motor_vocab.keys())
    ok = len(planted) == 0 and not ({"press", "harm"} <= birth_keys)
    return {
        "id": "C1_env_exposed_handles",
        "ok": ok,
        "planted_tokens": sorted(planted),
        "birth_motor_keys": sorted(birth_keys),
        "why": None if ok else "planted_motor_dictionary_in_genome",
    }


def control_c2_spellings_v4() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mact_c2_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        in_vocab = [t for t in ("press", "harm") if t in ag.vocab]
        ok = len(in_vocab) == 0
    return {
        "id": "C2_spellings_not_neural_input",
        "ok": ok,
        "in_vocab_at_birth": in_vocab,
        "why": None if ok else "handle_spellings_in_sensory_vocab",
    }


def control_c3_twin_v4() -> dict[str, Any]:
    main_s, twin_s = pair_seeds(0)
    main_toks = curriculum_tokens(main_s)
    twin_toks = curriculum_tokens(twin_s)
    ids_differ = main_toks["press"] != twin_toks["press"]
    with tempfile.TemporaryDirectory(prefix="mact_c3_") as tmp:
        m = make_cortex(Path(tmp) / "m", genome=main_s.genome(), device="cpu")
        t = make_cortex(Path(tmp) / "t", genome=twin_s.genome(), device="cpu")
        has_motor_rng = hasattr(m, "rng_motor")
        vec_equal = False
        if "press" in m.motor_vocab and "press" in t.motor_vocab:
            vec_equal = bool(np.allclose(m.motor_vocab["press"], t.motor_vocab["press"]))
    ok = bool(ids_differ and has_motor_rng and not vec_equal)
    return {
        "id": "C3_twin_independent_rename_vectorize",
        "ok": ok,
        "handle_ids_differ": ids_differ,
        "has_motor_rng": has_motor_rng,
        "press_vectors_equal": vec_equal,
        "why": None if ok else "shared_literals_or_no_motor_rng",
    }


CONTROLS_V4 = [
    control_c1_env_exposed_v4,
    control_c2_spellings_v4,
    control_c3_twin_v4,
]


def freeze_boundary_runner_v1() -> dict[str, Any]:
    if not CONTRACT.exists():
        raise RuntimeError("missing cortex_mact_boundary_contract.md")
    if not CANDIDATE_V4.exists():
        raise RuntimeError("missing cortex.candidate.v4.lock")
    if RUNNER_LOCK.exists():
        return {"ok": True, "path": str(RUNNER_LOCK), "sha": _sha_file(RUNNER_LOCK), "note": "already frozen"}
    cand = json.loads(CANDIDATE_V4.read_text(encoding="utf-8"))
    lock = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RUNNER.V1",
        "lab": "TM.0.23.CORTEX.MACT.BOUNDARY",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "contract": "docs/cortex_mact_boundary_contract.md",
        "contract_sha": _sha_file(CONTRACT),
        "boundary_module": "experiments.cortex_mact_boundary",
        "boundary_module_sha": _sha_file(Path(__file__)),
        "swap_revise_episodes": SWAP_REVISE_EPISODES,
        "candidate_under_test": "docs/cortex.candidate.v4.lock",
        "candidate_v4_sha": _sha_file(CANDIDATE_V4),
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "cortex_memory_sha": cand["cortex_memory_sha"],
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "class": "NeuralCortex",
            "note": "v1 runner targets planted MOTOR_ACT_TOKENS API",
        },
        "refuse": [
            "rewrite this lock after freeze",
            "soften D1/D2 scorers",
            "neural edits before documenting v4 reds",
        ],
        "note": "Frozen before v5 apparatus. Do not mutate after scoring v4.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(RUNNER_LOCK), "sha": _sha_file(RUNNER_LOCK)}


def run_boundary_v4(*, write_lock: bool = False) -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("freeze boundary runner v1 first")
    # Historical result may already exist
    if BOUNDARY_LOCK.exists() and write_lock:
        raise RuntimeError("cortex_mact_boundary.lock exists — refuse rewrite")
    results = [fn() for fn in CONTROLS_V4]
    # Keep historical behavioral notes if re-run without write
    n_ok = sum(1 for r in results if r.get("ok"))
    summary = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V4",
        "lab": "TM.0.23.CORTEX.MACT.BOUNDARY",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "candidate": "docs/cortex.candidate.v4.lock",
        "all_controls_green": n_ok == len(results),
        "n_ok": n_ok,
        "n_controls": len(results),
        "controls": results,
        "note": "Structural re-check only if re-run after v5 neural; prefer immutable lock.",
    }
    if write_lock:
        BOUNDARY_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


# --- v5 controls (bind_actuators) ---


def control_c1_v5() -> dict[str, Any]:
    planted = set(MOTOR_ACT_TOKENS) & {"press", "harm"}
    with tempfile.TemporaryDirectory(prefix="mact5_c1_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        birth_keys = set(ag.motor_vocab.keys())
        english = {"press", "harm"} & (birth_keys | set(ag.vocab.keys()))
    ok = len(planted) == 0 and len(birth_keys) == 0 and len(english) == 0
    return {
        "id": "C1_env_exposed_handles",
        "ok": ok,
        "planted_tokens": sorted(planted),
        "birth_motor_keys": sorted(birth_keys),
        "why": None if ok else "planted_or_birth_motor_lexicon",
    }


def control_c2_v5() -> dict[str, Any]:
    main_s, _ = pair_seeds(0)
    toks = curriculum_tokens(main_s)
    with tempfile.TemporaryDirectory(prefix="mact5_c2_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
        bind_life_actuators(ag, toks)
        leaked = [h for h in ag.motor_vocab if h in ag.vocab]
        ok = len(leaked) == 0 and all(not h.startswith(("press", "harm")) for h in ag.motor_vocab)
    return {
        "id": "C2_spellings_not_neural_input",
        "ok": ok,
        "leaked_into_vocab": leaked,
        "handles": sorted(ag.motor_vocab.keys()),
        "why": None if ok else "handles_entered_sensory_vocab",
    }


def control_c3_v5() -> dict[str, Any]:
    main_s, twin_s = pair_seeds(0)
    mt, tt = curriculum_tokens(main_s), curriculum_tokens(twin_s)
    ids_differ = mt["press"] != tt["press"] and mt["harm"] != tt["harm"]
    with tempfile.TemporaryDirectory(prefix="mact5_c3_") as tmp:
        m = make_cortex(Path(tmp) / "m", genome=main_s.genome(), device="cpu")
        t = make_cortex(Path(tmp) / "t", genome=twin_s.genome(), device="cpu")
        bind_life_actuators(m, mt)
        bind_life_actuators(t, tt)
        has_motor_rng = hasattr(m, "rng_motor") and m.genome.seed_motor != t.genome.seed_motor
        v_m = m.motor_vocab[mt["press"]]
        v_t = t.motor_vocab[tt["press"]]
        vec_equal = bool(np.allclose(v_m, v_t))
        # rebind restores
        v_before = v_m.copy()
        m.bind_actuators([mt["harm"], mt["press"]])
        restore_ok = bool(np.allclose(m.motor_vocab[mt["press"]], v_before))
    ok = bool(ids_differ and has_motor_rng and not vec_equal and restore_ok)
    return {
        "id": "C3_twin_independent_rename_vectorize",
        "ok": ok,
        "handle_ids_differ": ids_differ,
        "has_motor_rng": has_motor_rng,
        "press_vectors_equal": vec_equal,
        "rebind_restores": restore_ok,
        "why": None if ok else "twin_not_independent_or_rebind_fail",
    }


def control_c4_v5() -> dict[str, Any]:
    main_s, _ = pair_seeds(1)
    toks = curriculum_tokens(main_s)
    a_h, b_h = toks["press"], toks["harm"]
    with tempfile.TemporaryDirectory(prefix="mact5_c4_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
        # Two-handle world: revision target must be B, not distractor get/drop
        ag.bind_actuators([a_h, b_h])
        lat = {
            "act_effects": {
                a_h: {"state": ["st_pressed"], "delta": [0.25, -0.1, 0.15, 0.0]},
                b_h: {"state": ["st_hurt"], "delta": [-0.35, 0.45, -0.15, 0.0]},
            }
        }
        teach_loop(ag, main_s, n=160, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=lat)
        c0 = _pref_counts(ag, toks, 40, [toks["a"]], lat)
        pref_a = c0.get(a_h, 0) >= 3 and c0.get(a_h, 0) > c0.get(b_h, 0)
        if not pref_a:
            return {
                "id": "C4_consequence_swap_timed",
                "ok": False,
                "why": "failed_to_learn_A_before_swap",
                "counts_before": c0,
            }
        ckpt = ag.checkpoint()
        lat_swap = copy.deepcopy(lat)
        lat_swap["act_effects"][a_h], lat_swap["act_effects"][b_h] = (
            lat_swap["act_effects"][b_h],
            lat_swap["act_effects"][a_h],
        )
        c_stale = _pref_counts(ag, toks, 20, [toks["a"]], lat_swap)
        # Contract: stale_ok = still prefer A on immediate 20 probes (must not flip from physics alone).
        stale_ok = c_stale.get(a_h, 0) >= 3 and c_stale.get(a_h, 0) > c_stale.get(b_h, 0)
        teach_loop(
            ag,
            main_s,
            n=SWAP_REVISE_EPISODES,
            symbols_fn=lambda i, rng: [toks["a"], toks["b"]],
            latent=lat_swap,
        )
        c_rev = _pref_counts(ag, toks, 40, [toks["a"]], lat_swap)
        pref_b = c_rev.get(b_h, 0) >= 3 and c_rev.get(b_h, 0) > c_rev.get(a_h, 0)
        ag.load_checkpoint(ckpt)
        c_rest = _pref_counts(ag, toks, 40, [toks["a"]], lat)
        restore_a = c_rest.get(a_h, 0) >= 3 and c_rest.get(a_h, 0) > c_rest.get(b_h, 0)
        ok = bool(pref_a and stale_ok and pref_b and restore_a)
        return {
            "id": "C4_consequence_swap_timed",
            "ok": ok,
            "swap_revise_episodes": SWAP_REVISE_EPISODES,
            "pref_a": pref_a,
            "stale_ok": stale_ok,
            "pref_b": pref_b,
            "restore_a": restore_a,
            "counts_before": c0,
            "counts_stale": c_stale,
            "counts_revised": c_rev,
            "counts_restored": c_rest,
            "why": None if ok else "swap_revision_or_restore_failed",
        }


def control_c5_v5() -> dict[str, Any]:
    main_s, _ = pair_seeds(2)
    toks = curriculum_tokens(main_s)
    with tempfile.TemporaryDirectory(prefix="mact5_c5_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
        bind_life_actuators(ag, toks)
        b_op = float(ag.b_op[OPS.index("ACT")])
        _freeze_plasticity(ag)
        d1 = score_d1(ag, main_s, toks)
        learn_fail = not bool(d1.get("ok"))
        ok = learn_fail and abs(b_op - 0.85) < 1e-9 and OP_COST["ACT"] == 0.05
        return {
            "id": "C5_plasticity_off",
            "ok": ok,
            "d1_ok": d1.get("ok"),
            "press": d1.get("press"),
            "b_op_act": b_op,
            "op_cost_act": OP_COST["ACT"],
            "why": None if ok else "learning_still_passes_or_bias_missing",
        }


def control_c6_v5() -> dict[str, Any]:
    main_s, _ = pair_seeds(3)
    toks = curriculum_tokens(main_s)
    a_h, b_h = toks["press"], toks["harm"]
    neutral = {
        "act_effects": {
            a_h: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
            b_h: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
            toks["get"]: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
            toks["drop"]: {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
            toks.get("idle", "idle"): {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
        }
    }
    with tempfile.TemporaryDirectory(prefix="mact5_c6_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
        bind_life_actuators(ag, toks)
        teach_loop(ag, main_s, n=80, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=neutral)
        counts = _pref_counts(ag, toks, 40, [toks["a"]], neutral)
        press, harm = counts.get(a_h, 0), counts.get(b_h, 0)
        # Same D1-shaped preference bar as the frozen contract (do not raise after seeing reds).
        systematic = press >= 3 and press > harm
        ok = not systematic
        return {
            "id": "C6_no_consequence",
            "ok": ok,
            "counts": counts,
            "why": None if ok else "systematic_preference_without_consequences",
        }


def control_c7_v5() -> dict[str, Any]:
    main_s, _ = pair_seeds(5)
    toks = curriculum_tokens(main_s)
    distractor = f"h_distr_{main_s.seed_registry:x}"
    a_h = toks["press"]
    lat = motor_latent(toks)
    lat["act_effects"][distractor] = {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]}
    with tempfile.TemporaryDirectory(prefix="mact5_c7_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
        handles = [toks["press"], toks["harm"], toks["get"], toks["drop"], distractor]
        ag.bind_actuators(handles)
        teach_loop(ag, main_s, n=80, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=lat)
        counts = _pref_counts(ag, toks, 40, [toks["a"]], lat)
        d_n = counts.get(distractor, 0)
        a_n = counts.get(a_h, 0)
        # distractor must not become preferred over beneficial
        ok = not (d_n >= 3 and d_n > a_n)
        return {
            "id": "C7_distractor_motors",
            "ok": ok,
            "counts": counts,
            "distractor": distractor,
            "why": None if ok else "distractor_preferred_without_benefit",
        }


def control_c8_v5() -> dict[str, Any]:
    main_s, _ = pair_seeds(4)
    toks = curriculum_tokens(main_s)
    a_h = toks["press"]
    lat = motor_latent(toks)
    with tempfile.TemporaryDirectory(prefix="mact5_c8_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=main_s.genome(), device="cpu")
        bind_life_actuators(ag, toks)
        body = list(BODY0)
        state = ["st_idle"]
        for i in range(60):
            out, state, body = apply_event(
                ag,
                ix=f"act_{i}",
                source="src",
                symbols=[toks["a"]],
                state=state,
                body=body,
                latent=lat,
            )
            if (out.get("action") or {}).get("op") == "ACT":
                bad_body = list(BODY0)
                ag.observe(
                    {
                        "interaction_token": f"unrelated_{i}",
                        "source_token": "src_other",
                        "ordered_symbols": [toks["b"]],
                        "observable_state": ["st_idle"],
                        "body_state": bad_body,
                    }
                )
                body = bad_body
                state = ["st_idle"]
        counts = _pref_counts(ag, toks, 40, [toks["a"]], lat)
        press, harm = counts.get(a_h, 0), counts.get(toks["harm"], 0)
        reinforced = press >= 3 and press > harm
        ok = not reinforced
        return {
            "id": "C8_shuffled_credit",
            "ok": ok,
            "counts": counts,
            "why": None if ok else "actuator_reinforced_under_shuffled_credit",
        }


CONTROLS_V5 = [
    control_c1_v5,
    control_c2_v5,
    control_c3_v5,
    control_c4_v5,
    control_c5_v5,
    control_c6_v5,
    control_c7_v5,
    control_c8_v5,
]


def run_boundary_v5(*, write_lock: bool = False) -> dict[str, Any]:
    if not RUNNER_V2_LOCK.exists():
        raise RuntimeError("freeze boundary runner v2 first")
    if not CANDIDATE_V5.exists():
        raise RuntimeError("missing cortex.candidate.v5.lock")
    cand = json.loads(CANDIDATE_V5.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted from candidate v5 — refuse")
    results = [fn() for fn in CONTROLS_V5]
    n_ok = sum(1 for r in results if r.get("ok"))
    summary = {
        "version": "TM.0.23.CORTEX.MACT.BOUNDARY.RESULT.V5",
        "lab": "TM.0.23.CORTEX.MACT.BOUNDARY",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "candidate": "docs/cortex.candidate.v5.lock",
        "candidate_sha": _sha_file(CANDIDATE_V5),
        "runner_v2_sha": _sha_file(RUNNER_V2_LOCK),
        "contract_sha": _sha_file(CONTRACT),
        "all_controls_green": n_ok == len(results),
        "n_ok": n_ok,
        "n_controls": len(results),
        "controls": results,
        "env": torch_env(),
        "note": "Boundary hygiene on generic motor ABI. Does not prove D1–D2 competence.",
    }
    if write_lock:
        if BOUNDARY_V5_LOCK.exists():
            raise RuntimeError("cortex_mact_boundary.v5.lock exists — refuse rewrite")
        BOUNDARY_V5_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        lines = [
            "# TM.0.23.CORTEX M_act boundary (v5)",
            "",
            f"**all_controls_green:** `{summary['all_controls_green']}` ({n_ok}/{len(results)})",
            "",
        ]
        for r in results:
            lines.append(f"- `{r['id']}`: **{'PASS' if r.get('ok') else 'FAIL'}** — {r.get('why')}")
        lines += ["", summary["note"], ""]
        BOUNDARY_V5_MD.write_text("\n".join(lines), encoding="utf-8")
        summary["locks_written"] = True
    return summary
