"""v1: language three-memory vs published BDH Category B probes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import load_lm
from three_memory.bytes_util import CLEAN_FILLER, LINE_LORD, LINE_LOVE, PROBE
from three_memory.lm_agent import LanguageAgent, probe_js

# Published BDH numbers (mpiekarski78/bdh docs/hardware_and_metrics.md). Do not re-run BDH.
BDH = {
    "source": "https://github.com/mpiekarski78/bdh/blob/main/docs/hardware_and_metrics.md",
    "classification": "B",
    "empty_prior_p_r": 0.636,
    "empty_prior_p_v": 0.181,
    "k8_delta_p_v_love": 0.43,
    "k8_delta_p_r_lord": -0.59,
    "dedicated_js": 0.571,
    "p_v_given_B_8x": 0.61,
    "js_after_rho_reset": 0.0,
    "one_byte_filler_can_wipe_p_v": True,
}


def _run_dir(prefix: str = "v1") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_{prefix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_agent(model, device, store: bool, retrieve_mode: str = "note") -> LanguageAgent:
    return LanguageAgent(
        model, device, store_enabled=store, prefix_len=5, retrieve_mode=retrieve_mode
    )


def snapshot(agent: LanguageAgent) -> dict:
    return {"rho": agent.rho.snapshot(), "session_next": dict(agent.session_next)}


def restore(agent: LanguageAgent, snap: dict) -> None:
    agent.rho.load(snap["rho"])
    agent.session_next = dict(snap["session_next"])


def has_love_fact(agent: LanguageAgent) -> bool:
    for r in agent.store.records():
        if r.tags.get("prefix") == PROBE and r.tags.get("next") == "v":
            return True
        snip = str(r.tags.get("snippet") or r.what or "")
        if snip.startswith("my lov"):
            return True
    return False


def has_lord_fact(agent: LanguageAgent) -> bool:
    for r in agent.store.records():
        if r.tags.get("prefix") == PROBE and r.tags.get("next") == "r":
            return True
        snip = str(r.tags.get("snippet") or r.what or "")
        if snip.startswith("my lor"):
            return True
    return False


def classify(m: dict[str, Any]) -> tuple[str, str]:
    if not m["weights_unchanged_all"]:
        return "Confound", "Slow weights changed during the life (illegal in v1)."
    love_after = m["love_S_on_after_rho_reset"]["p_v"]
    prior_v = m["empty_prior"]["p_v"]
    off_after = m["love_S_off_after_rho_reset"]["p_v"]
    off_before = m["love_S_off_before_rho_reset"]["p_v"]
    reset_s = m["love_reset_S"]["p_v"]
    fact = m["love_has_inspectable_fact"]
    delta_on = love_after - prior_v
    delta_off_after = off_after - prior_v
    reset_s_near_prior = abs(reset_s - prior_v) < 0.10
    # Predeclared: S-on after ρ reset must raise P(v); S-off after reset must not;
    # clearing S must remove the effect.
    if fact and delta_on >= 0.10 and abs(delta_off_after) < 0.10 and reset_s_near_prior:
        return (
            "Store-works",
            "Language fact in S survives ρ reset; disable-S after reset returns near prior (BDH-like B).",
        )
    if off_before - prior_v >= 0.05 and abs(delta_off_after) < 0.10 and not (fact and delta_on >= 0.10):
        return "Trace-only", "Session residue moved P(v); ρ reset wiped it (BDH Category B analogue)."
    return "Fail", "S did not produce a durable, inspectable language association after ρ reset."


def run_v1(
    ckpt: Path,
    exposures: int,
    seed: int,
    *,
    retrieve_mode: str = "note",
    run_prefix: str = "v1",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(ckpt, device)
    h0 = make_agent(model, device, True, retrieve_mode).weight_hash()
    run_dir = _run_dir(run_prefix)

    prior_agent = make_agent(model, device, True, retrieve_mode)
    empty = prior_agent.probe(PROBE, use_store=True, apply_rho=True)

    # A: lord experience, S on
    A = make_agent(model, device, True, retrieve_mode)
    teach_A = A.experience(LINE_LORD * exposures)
    A_before = A.probe(PROBE)
    snap_A = snapshot(A)
    A.reset_rho()
    A_after = A.probe(PROBE)
    restore(A, snap_A)
    A_restored = A.probe(PROBE)
    A.reset_rho()

    # B: love experience, S on
    B = make_agent(model, device, True, retrieve_mode)
    teach_B = B.experience(LINE_LOVE * exposures)
    B_before = B.probe(PROBE)
    snap_B = snapshot(B)
    B.reset_rho()
    B_after = B.probe(PROBE)
    restore(B, snap_B)
    B_restored = B.probe(PROBE)

    # disable-S: same love experience
    C = make_agent(model, device, False, retrieve_mode)
    teach_C = C.experience(LINE_LOVE * exposures)
    C_before = C.probe(PROBE)
    C.reset_rho()
    C_after = C.probe(PROBE)

    # reset S after love
    D = make_agent(model, device, True, retrieve_mode)
    D.experience(LINE_LOVE * exposures)
    D.reset_rho()
    D.reset_store()
    D_after = D.probe(PROBE)

    # 1-byte filler after love, S on and S off (BDH comparison)
    E = make_agent(model, device, True, retrieve_mode)
    E.experience(LINE_LOVE * exposures)
    E.reset_rho()
    E_pre_fill = E.probe(PROBE)
    E.experience(CLEAN_FILLER[:1])
    E_post_fill = E.probe(PROBE)

    F = make_agent(model, device, False, retrieve_mode)
    F.experience(LINE_LOVE * exposures)
    F_pre_fill = F.probe(PROBE)
    F.experience(CLEAN_FILLER[:1])
    F_post_fill = F.probe(PROBE)

    # twins
    T1 = make_agent(model, device, True, retrieve_mode)
    T2 = make_agent(model, device, True, retrieve_mode)
    T1.experience(LINE_LOVE * exposures)
    T2.experience(LINE_LOVE * exposures)
    twin = T1.rho.distance(T2.rho)

    weights_ok = all(
        ag.weight_hash() == h0
        for ag in (A, B, C, D, E, F, T1, T2)
    ) and teach_A["weights_unchanged"] and teach_B["weights_unchanged"] and teach_C["weights_unchanged"]

    slim = lambda d: {
        "p_r": d["p_r"],
        "p_v": d["p_v"],
        "argmax": d["argmax"],
        "argmax_ch": bytes([d["argmax"]]).decode("latin-1", errors="replace"),
        "context": d["context"],
        "n_store": d["n_store"],
        "rho_l2": d["rho_l2"],
        "session_next": d["session_next"],
    }

    metrics: dict[str, Any] = {
        "retrieve_mode": retrieve_mode,
        "seed": seed,
        "exposures": exposures,
        "checkpoint": str(ckpt),
        "device": str(device),
        "weight_hash": h0,
        "weights_unchanged_all": weights_ok,
        "empty_prior": slim(empty),
        "lord_S_on_before_rho_reset": slim(A_before),
        "lord_S_on_after_rho_reset": slim(A_after),
        "lord_S_on_restored": slim(A_restored),
        "love_S_on_before_rho_reset": slim(B_before),
        "love_S_on_after_rho_reset": slim(B_after),
        "love_S_on_restored": slim(B_restored),
        "love_S_off_before_rho_reset": slim(C_before),
        "love_S_off_after_rho_reset": slim(C_after),
        "love_reset_S": slim(D_after),
        "love_S_on_1byte_filler_before": slim(E_pre_fill),
        "love_S_on_1byte_filler_after": slim(E_post_fill),
        "love_S_off_1byte_filler_before": slim(F_pre_fill),
        "love_S_off_1byte_filler_after": slim(F_post_fill),
        "love_has_inspectable_fact": has_love_fact(B),
        "lord_has_inspectable_fact": has_lord_fact(A),
        "store_B": B.store.to_jsonable(),
        "store_A": A.store.to_jsonable(),
        "js_love_vs_lord_after_rho_reset": probe_js(A_after, B_after),
        "js_love_vs_prior_after_rho_reset": probe_js(B_after, empty),
        "js_love_S_off_after_reset_vs_prior": probe_js(C_after, empty),
        "delta_p_v_love_S_on_after_reset": B_after["p_v"] - empty["p_v"],
        "delta_p_r_lord_S_on_after_reset": A_after["p_r"] - empty["p_r"],
        "delta_p_v_love_S_off_before_reset": C_before["p_v"] - empty["p_v"],
        "delta_p_v_love_S_off_after_reset": C_after["p_v"] - empty["p_v"],
        "rho_restore_p_v_match": abs(B_before["p_v"] - B_restored["p_v"]) < 1e-9,
        "twin_rho_distance": twin,
        "teach": {
            "A_writes": teach_A["writes"],
            "B_writes": teach_B["writes"],
            "C_writes": teach_C["writes"],
            "C_blocked": C.store._writes_blocked,
        },
        "bdh_published": BDH,
    }
    label, rationale = classify(metrics)
    metrics["classification"] = label
    metrics["rationale"] = rationale
    metrics["run_dir"] = str(run_dir)

    B.store.dump(run_dir / "store_B.json")
    A.store.dump(run_dir / "store_A.json")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    summary = f"""# v1 language three-memory

Classification: **{label}**

{rationale}

Probe `{PROBE}` → r vs v. Exposures={exposures}. Empty prior P(r)={empty['p_r']:.4f} P(v)={empty['p_v']:.4f}.

## Side by side with published BDH (not a re-run)

| Check | BDH (published) | three-memory S off | three-memory S on |
|-------|-----------------|--------------------|-------------------|
| Weights unchanged | yes | {weights_ok} | {weights_ok} |
| Empty prior P(v) | 0.181 | {empty['p_v']:.4f} | {empty['p_v']:.4f} |
| Empty prior P(r) | 0.636 | {empty['p_r']:.4f} | {empty['p_r']:.4f} |
| After {exposures}× love, P(v) before ρ reset | ~0.61 (dedicated B) | {C_before['p_v']:.4f} | {B_before['p_v']:.4f} |
| After {exposures}× love, P(v) after ρ reset | **0 (effect gone)** | {C_after['p_v']:.4f} | **{B_after['p_v']:.4f}** |
| ΔP(v) vs prior after ρ reset | n/a (reset wipes) | {metrics['delta_p_v_love_S_off_after_reset']:+.4f} | {metrics['delta_p_v_love_S_on_after_reset']:+.4f} |
| 1 extra filler byte, P(v) | can collapse | {F_pre_fill['p_v']:.4f} → {F_post_fill['p_v']:.4f} | {E_pre_fill['p_v']:.4f} → {E_post_fill['p_v']:.4f} |
| Inspectable fact | no | no | {has_love_fact(B)} |
| JS(love, lord) after ρ reset | 0 after reset | {metrics['js_love_S_off_after_reset_vs_prior']:.4f} vs prior | {metrics['js_love_vs_lord_after_rho_reset']:.4f} |

## Store B (love)

```json
{json.dumps(B.store.to_jsonable(), indent=2)}
```
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v1 language three-memory vs BDH probes")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "prior.pt"))
    p.add_argument("--exposures", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}; run: python -m experiments.train_prior")
    m = run_v1(ckpt, exposures=args.exposures, seed=args.seed)
    print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir")}, indent=2))
    print("empty P(v)", m["empty_prior"]["p_v"], "P(r)", m["empty_prior"]["p_r"])
    print("S-on after reset P(v)", m["love_S_on_after_rho_reset"]["p_v"])
    print("S-off after reset P(v)", m["love_S_off_after_rho_reset"]["p_v"])
    print("love fact", m["love_has_inspectable_fact"])


if __name__ == "__main__":
    main()
