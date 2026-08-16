"""TM.0.23.CORTEX.DIAG — snapshot-only probes on frozen v1 candidate (no neural edits).

Trace must not perturb execution: disabled == enabled trajectories.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.cortex_develop_life import (
    BODY0,
    DEFAULT_LATENT,
    LifeSeeds,
    apply_event,
    curriculum_tokens,
    pair_seeds,
)
from experiments.run_tm023cortex import make_cortex, torch_env
from three_memory.neural_cortex import BODY_SETPOINT, OPS, GenomeConfig, NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "cortex_diag"
DIAG_LOCK = REPO_ROOT / "docs" / "cortex_diag.lock"
DIAG_RESULTS_MD = REPO_ROOT / "docs" / "tm023cortex_diag_results.md"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "cortex.candidate.lock"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _rng_states(ag: NeuralCortex) -> dict[str, Any]:
    return {
        "birth": copy.deepcopy(ag.rng_birth.bit_generator.state),
        "registry": copy.deepcopy(ag.rng_registry.bit_generator.state),
        "source": copy.deepcopy(ag.rng_source.bit_generator.state),
        "action": copy.deepcopy(ag.rng_action.bit_generator.state),
        "permute": copy.deepcopy(ag.rng_permute.bit_generator.state),
    }


def _tensor_digest(ag: NeuralCortex) -> str:
    h = hashlib.sha256()
    for name in ag._plastic_names:
        arr = ag._from_t(getattr(ag, name))
        h.update(arr.tobytes())
    h.update(ag._from_t(ag.rho).tobytes())
    return h.hexdigest()


def _best_act_cos(ag: NeuralCortex) -> tuple[float, str | None]:
    """Snapshot-only: max cosine of W_act_query @ rho vs vocab (no sampling)."""
    if not ag.vocab:
        return -1.0, None
    q = ag._from_t(ag.W_act_query @ ag.rho)
    qn = np.linalg.norm(q) + 1e-12
    best_tok = None
    best = -1.0
    for tok, v in ag.vocab.items():
        cos = float(np.dot(q, v) / (qn * (np.linalg.norm(v) + 1e-12)))
        if cos > best:
            best = cos
            best_tok = tok
    return best, best_tok


def _op_probs(ag: NeuralCortex) -> dict[str, float]:
    """Snapshot-only softmax over W_op @ rho (+ b_op if present); does not advance RNG."""
    logits = ag._from_t(ag.W_op @ ag.rho)
    b_op = getattr(ag, "b_op", None)
    if b_op is not None:
        logits = logits + ag._from_t(b_op)
    logits = logits / float(ag.genome.tau)
    z = logits - np.max(logits)
    e = np.exp(z)
    p = e / np.sum(e)
    return {OPS[i]: float(p[i]) for i in range(len(OPS))}


def _clip_frac(ag: NeuralCortex) -> float:
    c = float(ag.genome.clip)
    hits = 0
    total = 0
    for name in ag._plastic_names:
        w = ag._from_t(getattr(ag, name))
        hits += int(np.sum(np.abs(w) >= c - 1e-12))
        total += w.size
    return float(hits) / float(total) if total else 0.0


def _consol_gap(ag: NeuralCortex) -> float:
    gaps = []
    for name in ag._plastic_names:
        w = ag._from_t(getattr(ag, name))
        slow = ag._from_t(ag.W_slow[name])
        gaps.append(float(np.linalg.norm(w - slow)))
    return float(np.mean(gaps)) if gaps else 0.0


def _rho_sat(ag: NeuralCortex) -> dict[str, float]:
    r = ag._from_t(ag.rho)
    return {
        "rho_norm": float(np.linalg.norm(r)),
        "frac_near_pm1": float(np.mean(np.abs(r) > 0.95)),
        "mean_abs": float(np.mean(np.abs(r))),
    }


def _discrete_traj(steps: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            s["op"],
            s.get("token"),
            tuple(s.get("body") or []),
            tuple(r.get("fact_id") for r in (s.get("s_ids") or [])),
        )
        for s in steps
    ]


def run_traced_teach(
    *,
    seeds: LifeSeeds,
    device: str,
    n_teach: int,
    n_probe: int,
    enable_trace: bool,
    tmp: Path,
) -> dict[str, Any]:
    """D1-like teach then probe. Snapshot reads only when enable_trace."""
    ag = make_cortex(tmp / ("t" if enable_trace else "b"), genome=seeds.genome(), device=device)
    toks = curriculum_tokens(seeds)
    latent = DEFAULT_LATENT
    body = list(BODY0)
    state = ["st_idle"]
    steps: list[dict[str, Any]] = []
    credit_log: list[dict[str, Any]] = []
    birth_probs = _op_probs(ag) if enable_trace else {}

    pending_meta: dict[str, Any] | None = None
    action_seq = 0

    def one_event(ix: str, symbols: list[str], phase: str) -> None:
        nonlocal body, state, pending_meta, action_seq
        # Snapshot pending BEFORE observe (eligibility for credit applied this step)
        pend = ag._pending
        elig_id = None
        prior_action_id = None
        prior_op = None
        prior_token = None
        if pend is not None:
            prior_op = pend.get("op")
            prior_token = pend.get("token")
            prior_action_id = pend.get("_action_id")
            elig_id = pend.get("_eligibility_id")

        s_before = [r.fact_id for r in ag.memory.records()]
        out, state, body = apply_event(
            ag,
            ix=ix,
            source="src_diag",
            symbols=symbols,
            state=state,
            body=body,
            latent=latent,
        )
        act = out.get("action") or {}
        metrics = out.get("metrics") or {}
        action_seq += 1
        action_id = f"a{action_seq}"
        eligibility_id = f"e{action_seq}"
        # Annotate pending only when tracing (read-path metadata; credit math ignores these keys)
        if enable_trace and ag._pending is not None:
            ag._pending["_action_id"] = action_id
            ag._pending["_eligibility_id"] = eligibility_id
            ag._pending["_interaction_token"] = ix

        credited_at = ag._t
        # body_consequence_step: step index when ACT emitted (prior action's consequence lands next observe)
        body_consequence_step = None
        if prior_op == "ACT":
            body_consequence_step = credited_at

        if enable_trace and pend is not None:
            credit_log.append(
                {
                    "action_id": prior_action_id,
                    "interaction_token": ix,
                    "eligibility_id": elig_id,
                    "credited_at_step": credited_at,
                    "body_consequence_step": body_consequence_step,
                    "credited_op": prior_op,
                    "credited_token": prior_token,
                    "adv": metrics.get("adv"),
                    "pred_err": metrics.get("pred_err"),
                    "phase": phase,
                }
            )

        row = {
            "phase": phase,
            "ix": ix,
            "op": act.get("op"),
            "token": act.get("token"),
            "body": list(body),
            "s_ids": [{"fact_id": fid} for fid in [r.fact_id for r in ag.memory.records()]],
            "adv": metrics.get("adv"),
            "pred_err": metrics.get("pred_err"),
            "rho_norm": out.get("rho_norm"),
        }
        if enable_trace:
            best_cos, best_tok = _best_act_cos(ag)
            probs = _op_probs(ag)
            sat = _rho_sat(ag)
            row.update(
                {
                    "action_id": action_id,
                    "eligibility_id": eligibility_id,
                    "best_act_cos": best_cos,
                    "best_act_tok": best_tok,
                    "forced_hold_suspect": act.get("op") == "HOLD"
                    and best_cos < float(ag.genome.cos_thresh),
                    "vocab_has_press": "press" in ag.vocab,
                    "vocab_has_harm": "harm" in ag.vocab,
                    "vocab_size": len(ag.vocab),
                    "op_probs": probs,
                    "clip_frac": _clip_frac(ag),
                    "consol_gap": _consol_gap(ag),
                    **sat,
                    "n_writes": len(ag.memory.records()) - len(s_before),
                    "buf_occ": int(
                        np.sum(
                            np.linalg.norm(ag._from_t(ag.retrieval_buffer), axis=1) > 1e-9
                        )
                    ),
                }
            )
        steps.append(row)

    rng = np.random.default_rng(seeds.seed_permute)
    for i in range(n_teach):
        syms = [toks["a"], toks["b"]]
        if seeds.role == "twin" and i % 2 == 0:
            syms = list(reversed(syms))
        one_event(f"{seeds.role}_t{i}", syms, "teach")

    mid_probs = _op_probs(ag) if enable_trace else {}

    for i in range(n_probe):
        one_event(f"{seeds.role}_p{i}", [toks["a"]], "probe")

    end_probs = _op_probs(ag) if enable_trace else {}
    traj = {
        "ops_tokens": [(s["op"], s.get("token")) for s in steps],
        "bodies": [s["body"] for s in steps],
        "s_writes": [tuple(x["fact_id"] for x in s["s_ids"]) for s in steps],
        "rng": _rng_states(ag),
        "tensor_digest": _tensor_digest(ag),
        "discrete": _discrete_traj(steps),
    }
    return {
        "ag_device": device,
        "steps": steps if enable_trace else [{"op": s["op"], "token": s.get("token"), "body": s["body"], "s_ids": s["s_ids"]} for s in steps],
        "credit_log": credit_log if enable_trace else [],
        "traj": traj,
        "birth_probs": birth_probs,
        "mid_probs": mid_probs,
        "end_probs": end_probs,
        "toks": toks,
        "enable_trace": enable_trace,
    }


def compare_purity(off: dict[str, Any], on: dict[str, Any], *, device: str) -> dict[str, Any]:
    t0, t1 = off["traj"], on["traj"]
    ops_eq = t0["ops_tokens"] == t1["ops_tokens"]
    body_eq = t0["bodies"] == t1["bodies"]
    s_eq = t0["s_writes"] == t1["s_writes"]
    rng_eq = t0["rng"] == t1["rng"]
    if device == "cpu":
        tensor_eq = t0["tensor_digest"] == t1["tensor_digest"]
        discrete_eq = True
    else:
        # GPU: discrete trajectory under existing tolerance (ops parity)
        tensor_eq = True
        discrete_eq = [a[0] for a in t0["discrete"]] == [a[0] for a in t1["discrete"]]
    ok = ops_eq and body_eq and s_eq and rng_eq and tensor_eq and discrete_eq
    return {
        "ok": ok,
        "ops_tokens_eq": ops_eq,
        "bodies_eq": body_eq,
        "s_writes_eq": s_eq,
        "rng_eq": rng_eq,
        "tensor_digest_eq": tensor_eq if device == "cpu" else None,
        "gpu_discrete_ops_eq": discrete_eq if device != "cpu" else None,
        "device": device,
    }


def summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    ops = Counter(s["op"] for s in steps)
    n = len(steps) or 1
    act_tokens = Counter(s.get("token") for s in steps if s.get("op") == "ACT" and s.get("token"))
    hold = ops.get("HOLD", 0)
    forced = sum(1 for s in steps if s.get("forced_hold_suspect"))
    teach = [s for s in steps if s.get("phase") == "teach"]
    probe = [s for s in steps if s.get("phase") == "probe"]
    pred = [float(s["pred_err"]) for s in steps if s.get("pred_err") is not None]
    advs = [float(s["adv"]) for s in steps if s.get("adv") is not None]
    by_op_adv: dict[str, list[float]] = {}
    # pair credit with credited_op from credit_log handled separately
    vocab_press = any(s.get("vocab_has_press") for s in steps)
    vocab_harm = any(s.get("vocab_has_harm") for s in steps)
    early_pred = float(np.mean(pred[: max(1, len(pred) // 4)])) if pred else None
    late_pred = float(np.mean(pred[-max(1, len(pred) // 4) :])) if pred else None
    return {
        "n_steps": len(steps),
        "op_counts": dict(ops),
        "op_rates": {k: v / n for k, v in ops.items()},
        "unique_ops": sorted(ops.keys()),
        "act_rate": ops.get("ACT", 0) / n,
        "hold_rate": hold / n,
        "act_token_counts": {str(k): v for k, v in act_tokens.items()},
        "forced_hold_suspect_count": forced,
        "forced_hold_suspect_rate": forced / n,
        "vocab_ever_has_press": vocab_press,
        "vocab_ever_has_harm": vocab_harm,
        "mean_adv": float(np.mean(advs)) if advs else None,
        "adv_quantiles": {
            "p10": float(np.quantile(advs, 0.1)) if advs else None,
            "p50": float(np.quantile(advs, 0.5)) if advs else None,
            "p90": float(np.quantile(advs, 0.9)) if advs else None,
        },
        "frac_adv_pos": float(np.mean([a > 0 for a in advs])) if advs else None,
        "pred_err_early": early_pred,
        "pred_err_late": late_pred,
        "mean_rho_norm": float(np.mean([s.get("rho_norm") or 0 for s in steps])),
        "mean_frac_near_pm1": float(np.mean([s.get("frac_near_pm1") or 0 for s in steps])),
        "mean_clip_frac": float(np.mean([s.get("clip_frac") or 0 for s in steps])),
        "mean_consol_gap": float(np.mean([s.get("consol_gap") or 0 for s in steps])),
        "write_events": sum(1 for s in steps if (s.get("n_writes") or 0) > 0),
        "retrieve_events": ops.get("RETRIEVE", 0),
        "mean_buf_occ": float(np.mean([s.get("buf_occ") or 0 for s in steps])),
        "teach_hold_rate": (sum(1 for s in teach if s["op"] == "HOLD") / len(teach)) if teach else None,
        "probe_act_rate": (sum(1 for s in probe if s["op"] == "ACT") / len(probe)) if probe else None,
    }


def summarize_credit(credit_log: list[dict[str, Any]]) -> dict[str, Any]:
    if not credit_log:
        return {"n": 0}
    by_op: dict[str, list[float]] = {}
    act_credited = 0
    act_with_body = 0
    for c in credit_log:
        op = str(c.get("credited_op"))
        by_op.setdefault(op, []).append(float(c.get("adv") or 0))
        if op == "ACT":
            act_credited += 1
            if c.get("body_consequence_step") is not None:
                act_with_body += 1
    return {
        "n": len(credit_log),
        "adv_by_credited_op": {
            k: {
                "mean": float(np.mean(v)),
                "frac_pos": float(np.mean([x > 0 for x in v])),
                "n": len(v),
            }
            for k, v in by_op.items()
        },
        "act_credit_events": act_credited,
        "note": "body_consequence_step is set on the ACT-emitting observe, not the credit observe",
        "sample_ids": credit_log[:5],
    }


def beneficial_prob_gain(birth: dict[str, float], end: dict[str, float]) -> dict[str, float]:
    return {
        "delta_p_act": float(end.get("ACT", 0) - birth.get("ACT", 0)),
        "p_act_birth": float(birth.get("ACT", 0)),
        "p_act_end": float(end.get("ACT", 0)),
        "p_hold_birth": float(birth.get("HOLD", 0)),
        "p_hold_end": float(end.get("HOLD", 0)),
    }


def run_diag(*, write_lock: bool = False, candidate_path: Path | None = None) -> dict[str, Any]:
    cand_path = candidate_path or CANDIDATE_LOCK
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    neural_sha = _sha_file(NEURAL_PY)
    memory_sha = _sha_file(MEMORY_PY)
    if neural_sha != cand["neural_cortex_sha"] or memory_sha != cand["cortex_memory_sha"]:
        raise RuntimeError(
            f"DIAG requires neural SHAs matching {cand_path.name} "
            f"(got neural={neural_sha[:12]}… cand={cand['neural_cortex_sha'][:12]}…)"
        )

    # Never clobber the frozen v1 DIAG tip when diagnosing a later candidate.
    v1_neural = None
    if CANDIDATE_LOCK.parent.joinpath("cortex.candidate.v1.lock").exists():
        v1_neural = json.loads(
            (CANDIDATE_LOCK.parent / "cortex.candidate.v1.lock").read_text(encoding="utf-8")
        ).get("neural_cortex_sha")
    write_path = DIAG_LOCK
    if write_lock and v1_neural and neural_sha != v1_neural:
        ver = str(cand.get("version") or "current").replace(".", "_")
        write_path = REPO_ROOT / "docs" / f"cortex_diag.{ver}.lock"

    main, _twin = pair_seeds(0)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    purity: dict[str, Any] = {}
    device_summaries: dict[str, Any] = {}
    raw_paths: dict[str, str] = {}

    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        with tempfile.TemporaryDirectory(prefix=f"diag_{device}_") as tmp:
            root = Path(tmp)
            off = run_traced_teach(
                seeds=main, device=device, n_teach=80, n_probe=40, enable_trace=False, tmp=root
            )
            on = run_traced_teach(
                seeds=main, device=device, n_teach=80, n_probe=40, enable_trace=True, tmp=root
            )
            purity[device] = compare_purity(off, on, device=device)
            summ = summarize_steps(on["steps"])
            credit = summarize_credit(on["credit_log"])
            gain = beneficial_prob_gain(on["birth_probs"], on["end_probs"])
            device_summaries[device] = {
                "summary": summ,
                "credit": credit,
                "prob_gain": gain,
                "birth_probs": on["birth_probs"],
                "end_probs": on["end_probs"],
            }
            raw = {
                "device": device,
                "steps": on["steps"],
                "credit_log": on["credit_log"],
                "birth_probs": on["birth_probs"],
                "end_probs": on["end_probs"],
            }
            raw_path = RUNS_DIR / f"diag_pair0_main_{device}.json"
            raw_bytes = json.dumps(raw, indent=2, default=str).encode()
            raw_path.write_bytes(raw_bytes)
            raw_paths[device] = str(raw_path.relative_to(REPO_ROOT))
            device_summaries[device]["raw_sha"] = _sha_bytes(raw_bytes)
            device_summaries[device]["raw_path"] = raw_paths[device]

    # CPU vs GPU discrete divergence on traced run summaries
    cpu_gpu = {"ok": None, "why": "no_cuda"}
    if "cuda" in device_summaries and "cpu" in device_summaries:
        c_ops = device_summaries["cpu"]["summary"]["op_counts"]
        g_ops = device_summaries["cuda"]["summary"]["op_counts"]
        # Compare hold/act rates rather than exact counts if lengths match
        cpu_gpu = {
            "ok": c_ops.get("HOLD") == g_ops.get("HOLD") and c_ops.get("ACT") == g_ops.get("ACT"),
            "cpu_op_counts": c_ops,
            "gpu_op_counts": g_ops,
            "policy": "HOLD/ACT count parity on identical seed D1-like trajectory",
        }

    purity_ok = all(p.get("ok") for p in purity.values())
    lock = {
        "version": "TM.0.23.CORTEX.DIAG",
        "lab": "TM.0.23.CORTEX.DIAG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_mechanism_changed": False,
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "neural_cortex_sha": neural_sha,
        "cortex_memory_sha": memory_sha,
        "trace_purity": purity,
        "trace_purity_ok": purity_ok,
        "devices": device_summaries,
        "cpu_gpu_divergence": cpu_gpu,
        "env": torch_env(),
        "note": "Bounded summaries. Raw traces under gitignored runs/; SHAs pinned per device.",
    }
    if not purity_ok:
        lock["ok"] = False
        lock["why"] = "trace_purity_failed"
    else:
        lock["ok"] = True

    if write_lock:
        write_path.write_text(json.dumps(lock, indent=2, default=str) + "\n", encoding="utf-8")
        lock["lock_path"] = str(write_path.relative_to(REPO_ROOT))
        _write_diag_md(lock)
    return lock


def _write_diag_md(lock: dict[str, Any]) -> None:
    cpu = (lock.get("devices") or {}).get("cpu") or {}
    s = cpu.get("summary") or {}
    lines = [
        "# TM.0.23.CORTEX.DIAG results",
        "",
        f"**product:** `{lock.get('product')}`",
        f"**earned_next:** `{lock.get('earned_next')}`",
        f"**ex0s:** `{lock.get('ex0s')}`",
        f"**trace_purity_ok:** `{lock.get('trace_purity_ok')}`",
        f"**neural_mechanism_changed:** `{lock.get('neural_mechanism_changed')}`",
        "",
        "## CPU D1-like summary (pair 0 main)",
        "",
        f"- op_counts: `{s.get('op_counts')}`",
        f"- act_rate: `{s.get('act_rate')}` hold_rate: `{s.get('hold_rate')}`",
        f"- forced_hold_suspect_rate: `{s.get('forced_hold_suspect_rate')}`",
        f"- vocab_ever_has_press: `{s.get('vocab_ever_has_press')}` harm: `{s.get('vocab_ever_has_harm')}`",
        f"- act_token_counts: `{s.get('act_token_counts')}`",
        f"- pred_err early→late: `{s.get('pred_err_early')}` → `{s.get('pred_err_late')}`",
        f"- mean_adv: `{s.get('mean_adv')}` frac_adv_pos: `{s.get('frac_adv_pos')}`",
        f"- prob_gain: `{(cpu.get('prob_gain'))}`",
        f"- credit: `{(cpu.get('credit') or {}).get('adv_by_credited_op')}`",
        "",
        f"cpu_gpu_divergence: `{lock.get('cpu_gpu_divergence')}`",
        "",
        "Raw traces: gitignored `runs/cortex_diag/` (SHAs in lock).",
        "",
    ]
    DIAG_RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
