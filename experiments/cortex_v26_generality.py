"""TM.0.23.CORTEX.GENERALITY.v26 — frozen audit of immutable candidate v26."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cortex_develop_life import (
    BODY0,
    bind_life_actuators,
    curriculum_tokens,
    motor_latent,
    pair_seeds,
)
from experiments.cortex_develop_scorers import _act_token_counts, _retrieval_buf_norm, teach_loop
from experiments.cortex_mact_boundary import _freeze_plasticity
from experiments.run_tm023cortex import build_observe, make_cortex, physics
from three_memory.cortex_memory import CortexRecord
from three_memory.neural_cortex import BODY_SETPOINT, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
CONTRACT = DOCS / "cortex_v26_generality_contract.md"
RUNNER_LOCK = DOCS / "cortex_v26_generality.runner.lock"
RESULT_LOCK = DOCS / "cortex_v26_generality.lock"
RESULT_MD = DOCS / "tm023cortex_v26_generality_results.md"
CAND_V26 = DOCS / "cortex.candidate.v26.lock"
CAND_LIVE = DOCS / "cortex.candidate.lock"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"

FORBIDDEN_IDENTIFIERS = (
    "phrase_program",
    "known_chunks",
    "phrase_target",
    "expected_length",
    "expected_len",
    "stored_length",
    "target_length",
)
FORBIDDEN_REGEX = (
    r"\b_phrase\b",
    r"\bstage\s*==",
    r"\bdomain\s*==",
)

BETTER = [float(x) for x in BODY_SETPOINT]
WORSE = [0.0, 1.0, 0.0, 1.0]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rng_fp(ag: NeuralCortex) -> dict[str, str]:
    snap = ag.checkpoint()
    rs = snap.get("rng") or {}
    out = {}
    for k in ("birth", "registry", "source", "action", "permute", "motor"):
        out[k] = hashlib.sha256(json.dumps(rs.get(k), sort_keys=True, default=str).encode()).hexdigest()
    return out


def _act_tuple(out: dict[str, Any]) -> tuple[Any, ...]:
    act = out.get("action") or {}
    return (act.get("op"), act.get("token"), tuple(act.get("emit_sequence") or []))


def _observe(
    ag: NeuralCortex,
    *,
    ix: str,
    symbols: list[str],
    body: list[float] | None = None,
    source: str = "src_gen",
    state: list[str] | None = None,
) -> dict[str, Any]:
    return ag.observe(
        build_observe(
            interaction_token=ix,
            source_token=source,
            ordered_symbols=symbols,
            observable_state=list(state or ["st_idle"]),
            body_state=list(body or BODY0),
        )
    )


def freeze_generality_runner() -> dict[str, Any]:
    if not CONTRACT.exists():
        raise RuntimeError("missing cortex_v26_generality_contract.md")
    if not CAND_V26.exists():
        raise RuntimeError("missing cortex.candidate.v26.lock")
    if RUNNER_LOCK.exists():
        return {"ok": True, "path": str(RUNNER_LOCK), "sha": _sha_file(RUNNER_LOCK), "note": "already frozen"}
    cand = json.loads(CAND_V26.read_text(encoding="utf-8"))
    if cand.get("version") != "TM.0.23.CORTEX.CANDIDATE.V26":
        raise RuntimeError("live candidate is not v26")
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — refuse freeze on non-v26 weights")
    if _sha_file(CAND_LIVE) != _sha_file(CAND_V26):
        raise RuntimeError("live cortex.candidate.lock is not v26")
    lock = {
        "version": "TM.0.23.CORTEX.GENERALITY.V26.RUNNER",
        "lab": "TM.0.23.CORTEX.GENERALITY.v26",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "contract": "docs/cortex_v26_generality_contract.md",
        "contract_sha": _sha_file(CONTRACT),
        "generality_module": "experiments.cortex_v26_generality",
        "generality_module_sha": _sha_file(Path(__file__)),
        "candidate_under_test": "docs/cortex.candidate.v26.lock",
        "candidate_v26_sha": _sha_file(CAND_V26),
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "cortex_memory_sha": cand["cortex_memory_sha"],
        "controls": [
            "G1_no_scripted_phrase_machinery",
            "G2_cross_modal_transfer",
            "G3_non_echo_response",
            "G4_order_length_counterfactuals",
            "G5_stop_evidence",
            "G6_neophobia_provenance",
            "G7_ablations",
            "G8_trace_purity",
        ],
        "refuse": [
            "rewrite this lock after freeze",
            "edit neural_cortex.py or cortex_memory.py",
            "edit cortex_develop_scorers.py",
            "reveal FULLDEV.R7 unless all_controls_green",
            "stamp 0.0.005",
        ],
        "note": "Frozen before scoring v26. Candidate remains immutable.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(RUNNER_LOCK), "sha": _sha_file(RUNNER_LOCK)}


def control_g1() -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for path in (NEURAL_PY, MEMORY_PY):
        text = path.read_text(encoding="utf-8")
        for ident in FORBIDDEN_IDENTIFIERS:
            n = text.count(ident)
            if n:
                hits.append({"file": path.name, "kind": "identifier", "pattern": ident, "n": n})
        for pat in FORBIDDEN_REGEX:
            found = re.findall(pat, text)
            if found:
                hits.append({"file": path.name, "kind": "regex", "pattern": pat, "n": len(found)})
    ok = len(hits) == 0
    return {
        "id": "G1_no_scripted_phrase_machinery",
        "ok": ok,
        "hits": hits,
        "why": None if ok else "scripted_phrase_or_stage_branch",
    }


def control_g2() -> dict[str, Any]:
    seeds, _ = pair_seeds(0)
    toks = curriculum_tokens(seeds)
    lat = motor_latent(toks)
    with tempfile.TemporaryDirectory(prefix="g2_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        bind_life_actuators(ag, toks, seeds)
        teach_loop(ag, seeds, n=80, symbols_fn=lambda i, rng: [toks["a"]], latent=lat, toks=toks)
        n_emit = n_act = after_emit_hold = after_act_hold = 0
        prev = None
        body = list(BODY0)
        state = ["st_idle"]
        for i in range(80):
            out = _observe(ag, ix=f"g2_{i}", symbols=[toks["a"]], body=body, state=state)
            op = (out.get("action") or {}).get("op")
            if prev == "EMIT":
                n_emit += 1
                if op == "HOLD":
                    after_emit_hold += 1
            if prev == "ACT":
                n_act += 1
                if op == "HOLD":
                    after_act_hold += 1
            prev = op
            action = out.get("action") or {}
            if action.get("op") == "ACT":
                state, body = physics(body, action.get("token"), lat)
            else:
                body = list(BODY0)
                state = ["st_idle"]
    r_e = (after_emit_hold / n_emit) if n_emit else 0.0
    r_a = (after_act_hold / n_act) if n_act else 0.0
    ok = (
        n_emit >= 4
        and n_act >= 4
        and r_e >= 0.30
        and r_a >= 0.30
        and abs(r_e - r_a) <= 0.25
    )
    return {
        "id": "G2_cross_modal_transfer",
        "ok": ok,
        "n_emit": n_emit,
        "n_act": n_act,
        "after_emit_hold_rate": r_e,
        "after_act_hold_rate": r_a,
        "why": None if ok else "continuation_not_cross_modal",
    }


def control_g3() -> dict[str, Any]:
    seeds, _ = pair_seeds(0)
    toks = curriculum_tokens(seeds)
    a, b = toks["a"], toks["b"]
    with tempfile.TemporaryDirectory(prefix="g3_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        bind_life_actuators(ag, toks, seeds)
        for i in range(40):
            _observe(ag, ix=f"g3fam_a_{i}", symbols=[a], body=BODY0)
        for i in range(40):
            _observe(ag, ix=f"g3fam_b_{i}", symbols=[b], body=BODY0)
        body = list(BODY0)
        for i in range(80):
            out = _observe(ag, ix=f"g3teach_{i}", symbols=[a], body=body)
            seq = list((out.get("action") or {}).get("emit_sequence") or [])
            if seq and seq[0] == b:
                body = list(BETTER)
            elif seq and seq[0] == a:
                body = list(WORSE)
            else:
                body = list(BODY0)
        echo_a = emit_b = other = 0
        for i in range(20):
            out = _observe(ag, ix=f"g3probe_{i}", symbols=[a], body=BODY0)
            seq = list((out.get("action") or {}).get("emit_sequence") or [])
            if seq and seq[0] == b:
                emit_b += 1
            elif seq and seq[0] == a:
                echo_a += 1
            else:
                other += 1
    ok = emit_b > echo_a and emit_b >= 3
    return {
        "id": "G3_non_echo_response",
        "ok": ok,
        "emit_b": emit_b,
        "echo_a": echo_a,
        "other": other,
        "why": None if ok else "echo_is_the_answer_function",
    }


def _copy_count(ag: NeuralCortex, cue: list[str], prefix: str, n: int = 20) -> int:
    hits = 0
    for i in range(n):
        out = _observe(ag, ix=f"{prefix}_{i}", symbols=cue, body=BODY0)
        seq = list((out.get("action") or {}).get("emit_sequence") or [])
        if seq == cue:
            hits += 1
    return hits


def control_g4() -> dict[str, Any]:
    main_s, twin_s = pair_seeds(0)
    main_t = curriculum_tokens(main_s)
    twin_t = curriculum_tokens(twin_s)
    p, q = main_t["emit1"], main_t["emit2"]
    d = main_t["distr"]
    p2, q2 = twin_t["emit1"], twin_t["emit2"]
    with tempfile.TemporaryDirectory(prefix="g4_") as tmp:
        main = make_cortex(Path(tmp) / "m", genome=main_s.genome(), device="cpu")
        twin = make_cortex(Path(tmp) / "t", genome=twin_s.genome(), device="cpu")
        bind_life_actuators(main, main_t, main_s)
        bind_life_actuators(twin, twin_t, twin_s)
        teach_loop(main, main_s, n=60, symbols_fn=lambda i, rng: [p, q])
        teach_loop(twin, twin_s, n=60, symbols_fn=lambda i, rng: [p2, q2])
        renamed_leak = _copy_count(twin, [p, q], "g4leak")
        reverse_copy = _copy_count(main, [q, p], "g4rev")
        distr_copy = _copy_count(main, [p, d, q], "g4d")
        unseen = 0
        five = [p, q, d, main_t["a"], main_t["c"]]
        for i in range(20):
            out = _observe(main, ix=f"g4len_{i}", symbols=five, body=BODY0)
            seq = list((out.get("action") or {}).get("emit_sequence") or [])
            if len(seq) == 5:
                unseen += 1
        spellings_differ = {p, q} != {p2, q2}
    renamed_ok = spellings_differ and renamed_leak < 8
    reverse_ok = reverse_copy < 8
    distr_ok = distr_copy < 8
    length_ok = unseen < 8
    ok = renamed_ok and reverse_ok and distr_ok and length_ok
    return {
        "id": "G4_order_length_counterfactuals",
        "ok": ok,
        "renamed_ok": renamed_ok,
        "reverse_ok": reverse_ok,
        "distractor_ok": distr_ok,
        "unseen_length_ok": length_ok,
        "renamed_leak": renamed_leak,
        "reverse_copy": reverse_copy,
        "distractor_copy": distr_copy,
        "unseen_len5": unseen,
        "spellings_differ": spellings_differ,
        "why": None if ok else "copies_probe_instead_of_learned_evidence",
    }


def _teach_stop_boundary(ag: NeuralCortex, cue: list[str], target_len: int, prefix: str) -> None:
    body = list(BODY0)
    for i in range(80):
        out = _observe(ag, ix=f"{prefix}_t{i}", symbols=cue, body=body)
        act = out.get("action") or {}
        seq = list(act.get("emit_sequence") or [])
        if act.get("op") == "STOP" and len(seq) == target_len:
            body = list(BETTER)
        elif act.get("op") == "STOP":
            body = list(WORSE)
        else:
            body = list(BODY0)


def _mean_stop_len(ag: NeuralCortex, cue: list[str], prefix: str, n: int = 20) -> float:
    lens: list[int] = []
    for i in range(n):
        out = _observe(ag, ix=f"{prefix}_{i}", symbols=cue, body=BODY0)
        act = out.get("action") or {}
        seq = list(act.get("emit_sequence") or [])
        if act.get("op") == "STOP":
            lens.append(len(seq))
        elif seq:
            lens.append(len(seq))
    return float(sum(lens) / len(lens)) if lens else 0.0


def control_g5() -> dict[str, Any]:
    s1, _ = pair_seeds(0)
    s2, _ = pair_seeds(1)
    t1 = curriculum_tokens(s1)
    t2 = curriculum_tokens(s2)
    cue1 = [t1["a"], t1["b"], t1["c"], t1["foil"]]
    cue2 = [t2["a"], t2["b"], t2["c"], t2["foil"]]
    with tempfile.TemporaryDirectory(prefix="g5_") as tmp:
        a = make_cortex(Path(tmp) / "a", genome=s1.genome(), device="cpu")
        b = make_cortex(Path(tmp) / "b", genome=s2.genome(), device="cpu")
        bind_life_actuators(a, t1, s1)
        bind_life_actuators(b, t2, s2)
        for i in range(40):
            _observe(a, ix=f"g5fam_a_{i}", symbols=cue1, body=BODY0)
            _observe(b, ix=f"g5fam_b_{i}", symbols=cue2, body=BODY0)
        _teach_stop_boundary(a, cue1, 2, "g5a")
        _teach_stop_boundary(b, cue2, 4, "g5b")
        mean2 = _mean_stop_len(a, cue1, "g5p2")
        mean4 = _mean_stop_len(b, cue2, "g5p4")
    closer2 = abs(mean2 - 2.0) < abs(mean2 - 4.0)
    closer4 = abs(mean4 - 4.0) < abs(mean4 - 2.0)
    ok = abs(mean4 - mean2) >= 1.0 and closer2 and closer4
    return {
        "id": "G5_stop_evidence",
        "ok": ok,
        "mean_len_boundary2": mean2,
        "mean_len_boundary4": mean4,
        "closer_to_2": closer2,
        "closer_to_4": closer4,
        "why": None if ok else "stop_follows_copied_length_not_boundary_evidence",
    }


def control_g6() -> dict[str, Any]:
    g1 = control_g1()
    has_chunks = any(h.get("pattern") == "known_chunks" for h in g1.get("hits") or [])
    seeds, _ = pair_seeds(0)
    toks = curriculum_tokens(seeds)
    known = toks["ground"]
    unknown = toks["foil"]
    with tempfile.TemporaryDirectory(prefix="g6_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        bind_life_actuators(ag, toks, seeds)
        teach_loop(ag, seeds, n=80, symbols_fn=lambda i, rng: [known])
        taught_ckpt = ag.checkpoint()
        known_holds = 0
        unk_holds = 0
        for i in range(20):
            o1 = _observe(ag, ix=f"g6k_{i}", symbols=[known], body=BODY0)
            o2 = _observe(ag, ix=f"g6u_{i}", symbols=[unknown], body=BODY0)
            if (o1.get("action") or {}).get("op") == "HOLD":
                known_holds += 1
            if (o2.get("action") or {}).get("op") == "HOLD":
                unk_holds += 1
        known_nonhold = (20 - known_holds) / 20.0
        unk_nonhold = (20 - unk_holds) / 20.0
        ag.reset_cortex()
        bind_life_actuators(ag, toks, seeds)
        after_reset_holds = 0
        for i in range(20):
            o = _observe(ag, ix=f"g6r_{i}", symbols=[known], body=BODY0)
            if (o.get("action") or {}).get("op") == "HOLD":
                after_reset_holds += 1
        ag.load_checkpoint(taught_ckpt)
        donor_holds = 0
        for i in range(20):
            o = _observe(ag, ix=f"g6d_{i}", symbols=[known], body=BODY0)
            if (o.get("action") or {}).get("op") == "HOLD":
                donor_holds += 1
    taught_gap = known_nonhold >= unk_nonhold + 0.15
    reset_unfamiliar = after_reset_holds >= 12
    donor_familiar = (20 - donor_holds) / 20.0 >= known_nonhold - 0.15
    ok = taught_gap and reset_unfamiliar and donor_familiar and not has_chunks
    return {
        "id": "G6_neophobia_provenance",
        "ok": ok,
        "known_nonhold": known_nonhold,
        "unknown_nonhold": unk_nonhold,
        "after_reset_holds": after_reset_holds,
        "donor_holds": donor_holds,
        "known_chunks_present": has_chunks,
        "why": None if ok else "familiarity_not_from_this_life_experience",
    }


def control_g7() -> dict[str, Any]:
    seeds, twin_s = pair_seeds(0)
    toks = curriculum_tokens(seeds)
    twin_t = curriculum_tokens(twin_s)
    lat = motor_latent(toks)
    with tempfile.TemporaryDirectory(prefix="g7_") as tmp:
        birth = make_cortex(Path(tmp) / "b", genome=seeds.genome(), device="cpu")
        bind_life_actuators(birth, toks, seeds)
        birth_counts = _act_token_counts(birth, toks, 40, [toks["a"]], latent=lat)
        birth_press = birth_counts.get(toks["press"], 0)
        birth_harm = birth_counts.get(toks["harm"], 0)
        birth_ok = not (birth_press >= 3 and birth_press > birth_harm)

        frozen = make_cortex(Path(tmp) / "f", genome=seeds.genome(), device="cpu")
        bind_life_actuators(frozen, toks, seeds)
        _freeze_plasticity(frozen)
        teach_loop(frozen, seeds, n=80, symbols_fn=lambda i, rng: [toks["a"], toks["b"]], latent=lat, toks=toks)
        fz_counts = _act_token_counts(frozen, toks, 40, [toks["a"]], latent=lat)
        fz_press = fz_counts.get(toks["press"], 0)
        fz_harm = fz_counts.get(toks["harm"], 0)
        plasticity_ok = not (fz_press >= 3 and fz_press > fz_harm)

        mem = make_cortex(Path(tmp) / "m", genome=seeds.genome(), device="cpu")
        bind_life_actuators(mem, toks, seeds)
        teach_loop(mem, seeds, n=40, symbols_fn=lambda i, rng: [toks["fact"]])
        wrote = [
            r
            for r in mem.memory.records()
            if r.source == "cortex_write" and float(np.linalg.norm(np.asarray(r.content, dtype=np.float64))) > 0.0
        ]
        if not wrote:
            strip_ok = False
            stripped_norm = donor_norm = 0.0
        else:
            fid = wrote[-1].fact_id
            mem.memory.delete(fid)
            stripped_norm = _retrieval_buf_norm(mem)
            donor_vec = list(np.random.default_rng(seeds.seed_registry).normal(0, 1, size=mem.genome.d_sym))
            mem.memory.write(
                CortexRecord(
                    fact_id="donor_fact",
                    content=donor_vec,
                    when=mem.age,
                    interaction_token="donor",
                    source_token="src_donor",
                    source="cortex_write",
                )
            )
            donor_norm = _retrieval_buf_norm(mem)
            strip_ok = donor_norm > stripped_norm + 1e-6

        rho = make_cortex(Path(tmp) / "r", genome=seeds.genome(), device="cpu")
        bind_life_actuators(rho, toks, seeds)
        teach_loop(rho, seeds, n=20, symbols_fn=lambda i, rng: [toks["a"]])
        w0 = rho.weight_hash()
        rho.reset_rho()
        rho_ok = w0 == rho.weight_hash()

        twin = make_cortex(Path(tmp) / "t", genome=twin_s.genome(), device="cpu")
        bind_life_actuators(twin, twin_t, twin_s)
        teach_loop(twin, twin_s, n=40, symbols_fn=lambda i, rng: [twin_t["emit1"], twin_t["emit2"]])
        leak = 0
        for i in range(20):
            out = _observe(twin, ix=f"g7tw_{i}", symbols=[toks["emit1"], toks["emit2"]], body=BODY0)
            seq = list((out.get("action") or {}).get("emit_sequence") or [])
            if toks["emit1"] in seq:
                leak += 1
        twin_ok = toks["emit1"] != twin_t["emit1"] and leak < 8
    ok = birth_ok and plasticity_ok and strip_ok and rho_ok and twin_ok
    return {
        "id": "G7_ablations",
        "ok": ok,
        "birth_ok": birth_ok,
        "plasticity_off_ok": plasticity_ok,
        "stripped_s_ok": strip_ok,
        "rho_reset_ok": rho_ok,
        "renamed_twin_ok": twin_ok,
        "birth_press": birth_press,
        "birth_harm": birth_harm,
        "frozen_press": fz_press,
        "frozen_harm": fz_harm,
        "twin_main_token_leak": leak,
        "why": None if ok else "ablation_fork_missing_expected_difference",
    }


def control_g8() -> dict[str, Any]:
    seeds, _ = pair_seeds(0)
    toks = curriculum_tokens(seeds)
    with tempfile.TemporaryDirectory(prefix="g8_") as tmp:
        ag = make_cortex(Path(tmp) / "s", genome=seeds.genome(), device="cpu")
        bind_life_actuators(ag, toks, seeds)
        teach_loop(ag, seeds, n=20, symbols_fn=lambda i, rng: [toks["a"]])
        ckpt = ag.checkpoint()
        traj = []
        for i in range(8):
            out = _observe(ag, ix=f"g8a_{i}", symbols=[toks["a"]], body=BODY0)
            traj.append(_act_tuple(out))
        rng_a = _rng_fp(ag)
        ag.load_checkpoint(ckpt)
        traj_b = []
        for i in range(8):
            out = _observe(ag, ix=f"g8a_{i}", symbols=[toks["a"]], body=BODY0)
            _ = (ag.age, list(ag.emit_buffer), ag.last_action, ag.weight_hash())
            traj_b.append(_act_tuple(out))
        rng_b = _rng_fp(ag)
    ok = traj == traj_b and rng_a == rng_b
    return {
        "id": "G8_trace_purity",
        "ok": ok,
        "traj_match": traj == traj_b,
        "rng_match": rng_a == rng_b,
        "why": None if ok else "audit_instrumentation_perturbed_trajectory",
    }


CONTROLS = (
    control_g1,
    control_g2,
    control_g3,
    control_g4,
    control_g5,
    control_g6,
    control_g7,
    control_g8,
)


def run_generality_v26(*, write_lock: bool = False) -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("freeze generality runner first")
    runner = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CAND_V26.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural drifted — refuse score")
    if _sha_file(Path(__file__)) != runner["generality_module_sha"]:
        raise RuntimeError("runner module drifted after freeze — refuse score")
    if _sha_file(CONTRACT) != runner["contract_sha"]:
        raise RuntimeError("contract drifted after freeze — refuse score")
    if RESULT_LOCK.exists() and write_lock:
        raise RuntimeError("cortex_v26_generality.lock exists — refuse rewrite")
    results = [fn() for fn in CONTROLS]
    n_ok = sum(1 for r in results if r.get("ok"))
    all_green = n_ok == len(results)
    summary = {
        "version": "TM.0.23.CORTEX.GENERALITY.V26.RESULT",
        "lab": "TM.0.23.CORTEX.GENERALITY.v26",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "candidate": "docs/cortex.candidate.v26.lock",
        "candidate_sha": _sha_file(CAND_V26),
        "runner_sha": _sha_file(RUNNER_LOCK),
        "contract_sha": _sha_file(CONTRACT),
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "all_controls_green": all_green,
        "n_ok": n_ok,
        "n_controls": len(results),
        "controls": results,
        "refuse_fulldev_r7": not all_green,
        "note": "v26 immutable. Green required before FULLDEV.R7. Any red → diagnose isolated v27.",
    }
    if write_lock:
        RESULT_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        RESULT_MD.write_text(_results_md(summary), encoding="utf-8")
    return summary


def _results_md(summary: dict[str, Any]) -> str:
    lines = [
        "# TM.0.23.CORTEX.GENERALITY.v26",
        "",
        f"**all_controls_green:** `{summary['all_controls_green']}` ({summary['n_ok']}/{summary['n_controls']})",
        "",
        "Candidate v26 remains immutable. Product **0.0.004**. `earned_next=false`. `ex0s=null`.",
        "",
    ]
    for c in summary["controls"]:
        flag = "PASS" if c.get("ok") else "FAIL"
        why = c.get("why") or "None"
        lines.append(f"- `{c['id']}`: **{flag}** — {why}")
    lines.append("")
    if summary["all_controls_green"]:
        lines.append("All controls green. FULLDEV.R7 on a fresh sealed domain is authorized. Not a product stamp.")
    else:
        lines.append("At least one control is red. Do not reveal FULLDEV.R7. Diagnose and isolate v27.")
    lines.append("")
    return "\n".join(lines) + "\n"


def verify_generality_v26() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        return {"ok": False, "why": "missing runner lock"}
    runner = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if runner.get("earned_next") is not False or runner.get("ex0s") is not None:
        return {"ok": False, "why": "runner stamp leak"}
    if not RESULT_LOCK.exists():
        return {"ok": True, "pending": True, "refuse_fulldev_r7": True}
    result = json.loads(RESULT_LOCK.read_text(encoding="utf-8"))
    if result.get("runner_sha") != _sha_file(RUNNER_LOCK):
        return {"ok": False, "why": "result runner pin mismatch"}
    if result.get("candidate_sha") != _sha_file(CAND_V26):
        return {"ok": False, "why": "result candidate pin mismatch"}
    if result.get("earned_next") is not False or result.get("ex0s") is not None:
        return {"ok": False, "why": "result stamp leak"}
    return {
        "ok": True,
        "pending": False,
        "all_controls_green": bool(result.get("all_controls_green")),
        "n_ok": result.get("n_ok"),
        "refuse_fulldev_r7": bool(result.get("refuse_fulldev_r7")),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("freeze", "score", "verify"))
    args = p.parse_args()
    if args.cmd == "freeze":
        print(json.dumps(freeze_generality_runner(), indent=2))
    elif args.cmd == "score":
        print(json.dumps(run_generality_v26(write_lock=True), indent=2, default=str))
    else:
        print(json.dumps(verify_generality_v26(), indent=2))
