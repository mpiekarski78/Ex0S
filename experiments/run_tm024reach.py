"""TM.0.24.REACH — developmental reachability after v28 credit-path repair.

Not a lineage version. Not a capability earn. Product 0.0.004.
Scoring requires docs/lineage_reach.runner.lock on clean origin/main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cortex_v28_pipeline import probe_zero_elig_no_motion
from experiments.run_tm023cortex import make_cortex, torch_env
from experiments.run_tm024lineage import live_once, probe_beneficial
from experiments.run_tm024wallmap import evaluate_arm_d_causal, make_diag_world
from three_memory.cortex_lineage import (
    AdamState,
    adam_step,
    antithetic_children,
    defaults_theta,
    refuse_audit,
    sha_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_reach.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_reach_contract.md"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_reach.runner.lock"
RESULT_LOCK = REPO_ROOT / "docs" / "lineage_reach.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024reach_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v28.lock"
WALLMAP_Q2 = REPO_ROOT / "docs" / "lineage_wallmap_q2.lock"

N_WAKE = 40
N_REPLAY = 8
N_PROBE = 20
TAU = 0.60
DELTA_B = 0.05
DELTA_P = 0.05


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def reach_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "candidate_v28": CANDIDATE,
        "wallmap_evaluate": REPO_ROOT / "experiments" / "run_tm024wallmap.py",
        "cortex_lineage": REPO_ROOT / "three_memory" / "cortex_lineage.py",
        "v28_pipeline": REPO_ROOT / "experiments" / "cortex_v28_pipeline.py",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no reach runner.lock — refuse diagnostic scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    live = reach_shas()
    if live != lock.get("shas"):
        raise RuntimeError("reach implementation drifted after runner.lock — versioned lock required")
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if sha_file(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from v28 candidate")
    if cand.get("genome", {}).get("n") != 64:
        raise RuntimeError("n must stay 64")
    return lock


def assert_clean_to_begin() -> None:
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    if not porcelain.strip():
        return
    allowed = ("docs/lineage_reach.lock", "docs/tm024reach_results.md")
    for line in porcelain.splitlines():
        path = line[3:].strip()
        if path.startswith("docs/lineage_reach.lock") or path.startswith("docs/tm024reach_results.md"):
            continue
        if path in allowed:
            continue
        raise RuntimeError(f"dirty tree blocks reach scoring: {path}")


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    assert prereg["fit_domain"] != "TM024.WALLMAP.Q2.DIAG.FIT."
    assert prereg["check_domain"] != "TM024.WALLMAP.Q2.DIAG.CHECK."
    assert prereg["n"] == 64
    credit = probe_zero_elig_no_motion()
    w = make_diag_world(prereg["fit_domain"], 0)
    with tempfile.TemporaryDirectory(prefix="reach_smoke_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(w["handles"]))
        live_once(ag, w, n_wake=6, n_replay=2, teacher_seed=1)
        pb = float(probe_beneficial(ag, w, n_probe=8))
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "credit_precondition_ok": bool(credit.get("ok")),
        "probe": pb,
        "n": 64,
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if not _git_clean():
        raise RuntimeError("write runner.lock only on a clean tree")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.REACH.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "shas": reach_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "candidate_v28_sha": sha_file(CANDIDATE),
        "n_wake": N_WAKE,
        "n_replay": N_REPLAY,
        "n_probe": N_PROBE,
        "tau": TAU,
        "delta_B": DELTA_B,
        "delta_P": DELTA_P,
        "n": 64,
        "es": prereg["es"],
        "fit_domain": prereg["fit_domain"],
        "check_domain": prereg["check_domain"],
        "git_head_at_freeze": _git_head(),
        "note": "SHA pin is the integrity gate. Clean tree required once at suite start; result locks may appear mid-run.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_reach(*, write_lock: bool = False) -> dict[str, Any]:
    assert_clean_to_begin()
    assert_runner_frozen()
    prereg = load_prereg()
    credit = probe_zero_elig_no_motion()
    fit_worlds = [make_diag_world(prereg["fit_domain"], i) for i in range(int(prereg["n_fit_worlds"]))]
    check_worlds = [make_diag_world(prereg["check_domain"], i) for i in range(int(prereg["n_check_worlds"]))]
    layout_theta = defaults_theta("D")
    assert refuse_audit(layout_theta, "D")["ok"]
    adam = AdamState(m=np.zeros_like(layout_theta), v=np.zeros_like(layout_theta), lr=float(prereg["es"]["adam_lr"]))
    theta = layout_theta.copy()
    sigma = float(prereg["es"]["sigma"])
    gens = int(prereg["es"]["gens"])
    pairs = int(prereg["es"]["pairs"])
    for gen in range(gens):
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(f"TM024.REACH.es:g{gen}".encode()).digest()[:8], "big") % (2**31)
        )
        fits = []
        noises = []
        for _i in range(pairs):
            eps = rng.normal(0.0, 1.0, size=theta.size)
            plus, minus = antithetic_children(theta, eps, sigma)
            n_fit = int(prereg["es"]["fit_worlds_per_child"])
            births = list(range(int(prereg["es"]["fit_births_per_child"])))
            fp = evaluate_arm_d_causal(plus, fit_worlds[:n_fit], births)["adult_mean"]
            fm = evaluate_arm_d_causal(minus, fit_worlds[:n_fit], births)["adult_mean"]
            fits.extend([fp, fm])
            noises.extend([eps, -eps])
        ranks = np.argsort(np.argsort(fits)).astype(np.float64)
        ranks = (ranks / max(len(ranks) - 1, 1)) - 0.5
        grad = np.zeros_like(theta)
        for r, e in zip(ranks, noises, strict=True):
            grad += r * e
        grad /= max(len(noises), 1) * sigma
        theta = adam_step(theta, grad, adam)
        theta = np.clip(theta, -5.0, 5.0)
    check_births = list(range(int(prereg["es"]["check_births"])))
    check = evaluate_arm_d_causal(theta, check_worlds, check_births)
    gates_ok = bool(check["adult_ok"] and check["birth_below"] and check["off_below"] and check["G_k"])
    credit_ok = bool(credit.get("ok"))
    passed = bool(credit_ok and gates_ok)
    note = "Same genotype on CHECK. Favorable birth is not a pass."
    if not credit_ok:
        note = "Credit path still broken; CHECK fail is not a maturation/replay diagnosis."
    out = {
        "version": "TM.0.24.REACH",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "passed": passed,
        "credit_precondition": credit,
        "credit_precondition_ok": credit_ok,
        "check": check,
        "one_genotype": True,
        "n": 64,
        "fit_domain": prereg["fit_domain"],
        "check_domain": prereg["check_domain"],
        "wallmap_q2_historical": True,
        "wallmap_q2_sha": sha_file(WALLMAP_Q2) if WALLMAP_Q2.exists() else None,
        "candidate": "docs/cortex.candidate.v28.lock",
        "note": note,
    }
    if write_lock:
        if RESULT_LOCK.exists():
            raise RuntimeError("reach result lock exists")
        RESULT_LOCK.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        RESULT_MD.write_text(
            "# TM.0.24.REACH results\n\n"
            f"Product remains **0.0.004**. `earned_next=false`. `ex0s=null`.\n\n"
            f"**passed:** `{passed}`\n\n"
            f"- credit precondition (zero elig): **{credit_ok}**\n"
            f"- CHECK adult mean: `{check['adult_mean']}`\n"
            f"- CHECK birth mean: `{check['birth_mean']}`\n"
            f"- CHECK plasticity-off mean: `{check['plasticity_off_mean']}`\n"
            f"- CI lower: `{check['ci_lower']}`\n"
            f"- G_k: `{check['G_k']}`\n\n"
            f"n stays 64. LINEAGE/WALLMAP historical. QUAL/EVAL sealed.\n",
            encoding="utf-8",
        )
        out["locks_written"] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--write-runner-lock", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args()
    ran = False
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
        ran = True
    if args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
        ran = True
    if args.score:
        print(json.dumps(run_reach(write_lock=args.write_lock), indent=2, default=str))
        ran = True
    if not ran:
        p.print_help()


if __name__ == "__main__":
    main()
