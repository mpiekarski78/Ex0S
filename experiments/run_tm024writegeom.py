"""TM.0.24.WRITEGEOM — actuator-local plastic-write geometry.

v31 amendment candidate. Not a product earn. Product 0.0.004.
DEV on unused TM024.WRITEGEOM.DEV. worlds.
Scoring requires docs/lineage_writegeom.runner.lock on clean origin/main
after W1 is frozen. Neural edit only after this freeze is on origin/main.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import make_cortex, physics, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, motor_scores, observe_cue
from experiments.run_tm024lineage import make_synthetic_world, opaque_spelling
from experiments.run_tm024motorpersist import POS_DELTA, TEACH_ORDERS
from experiments.run_tm024statemap import teach_one
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_writegeom.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_writegeom_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_writegeom.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_writegeom.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_writegeom.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024writegeom_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_LIVE = REPO_ROOT / "docs" / "cortex.candidate.lock"
V30_NEURAL = "cc22cf381839049246776d2c223683078f8c13abf00cbd8e99ab2554206538b5"

DEV_DOMAIN = "TM024.WRITEGEOM.DEV."
SCORE_DOMAIN = "TM024.WRITEGEOM.SCORE."
TWIN_DOMAIN = "TM024.WRITEGEOM.TWIN."
REGRESS_DOMAIN = "TM024.WRITEGEOM.REGRESS."


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def thr() -> dict[str, Any]:
    return load_prereg()["margin"]


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def writegeom_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "v31_prereg": REPO_ROOT / "docs" / "cortex_v31.prereg.lock",
        "v31_isolation": REPO_ROOT / "docs" / "cortex_v31.isolation.lock",
        "v31_amendment": REPO_ROOT / "docs" / "cortex_v31_architecture_amendment.lock",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def neural_has_proto() -> bool:
    src = NEURAL.read_text(encoding="utf-8")
    return sha_file(NEURAL) != V30_NEURAL and "actuator_scores" in src and "ACT_SCORE_PROTO" in src


def make_cell_world(index: int, domain: str) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def capacity_world(
    index: int,
    domain: str,
    *,
    n_cues: int,
    n_handles: int,
) -> dict[str, Any]:
    """Fixed-handle overlay: extra cues/handles, identical positive deltas, balanced map."""
    prereg = load_prereg()
    if n_handles > int(prereg["H_max"]):
        raise RuntimeError(f"n_handles {n_handles} exceeds H_max {prereg['H_max']}")
    w = make_cell_world(index, domain)
    rng = np.random.default_rng(domain_seed(domain, f"cap_{index}_{n_cues}_{n_handles}"))
    handles = list(w["handles"])
    symbols = list(w["symbols"])
    while len(handles) < n_handles:
        handles.append(opaque_spelling(rng, "h"))
    while len(symbols) < n_cues:
        symbols.append(opaque_spelling(rng, "s"))
    handles = handles[:n_handles]
    cues = symbols[:n_cues]
    effects = {}
    mapping = []
    for i, cue in enumerate(cues):
        h = handles[i % n_handles]
        effects[h] = {"state": [f"st_p{i % n_handles}"], "delta": list(POS_DELTA)}
        mapping.append({"cue": cue, "handle": h})
    w["handles"] = handles
    w["symbols"] = symbols
    w["cues"] = cues
    w["capacity"] = {"n_cues": n_cues, "n_handles": n_handles}
    w["cue_handle"] = mapping
    w["latent"] = {"act_effects": effects}
    w["beneficial"] = handles[0]
    return w


def enable_w1(ag: NeuralCortex) -> None:
    if not hasattr(ag.genome, "act_score_mode"):
        raise RuntimeError("act_score_mode missing — implement W1 after this freeze is on origin/main")
    ag.genome.act_score_mode = "proto"


def _fresh(tmp: str, tag: str, world: dict[str, Any], *, proto: bool = False) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    if proto:
        enable_w1(ag)
    return ag


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ranking_margin(scores: dict[str, float], winner: str) -> float:
    if winner not in scores:
        return 0.0
    win = float(scores[winner])
    others = [float(v) for k, v in scores.items() if k != winner]
    if not others:
        return win
    return float(win - max(others))


def perturb_stable(
    ag: NeuralCortex,
    rho: np.ndarray,
    winner: str,
    *,
    domain: str,
    key: str,
) -> dict[str, Any]:
    m = thr()
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = np.asarray(rho, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r0)) + 1e-12
    r_hat = r0 / nrm
    n_ok = 0
    for i in range(n):
        noise = rng.normal(0.0, sigma, size=r_hat.shape)
        rp = r_hat + noise
        pn = float(np.linalg.norm(rp)) + 1e-12
        scores = ag.actuator_scores(rp / pn) if hasattr(ag, "actuator_scores") else motor_scores(ag)
        ranked = max(scores, key=lambda h: scores[h])
        if ranked == winner:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_cell_world(0, DEV_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="wg_smk_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(w["handles"]))
        t = teach_one(ag, w, w["beneficial"], tag="smk")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "H_max": prereg["H_max"],
        "state_budget": prereg["state_budget"],
        "cosine_margin_min": prereg["margin"]["cosine_margin_min"],
        "d_w_op": t["d_w_op"],
        "neural_has_proto": neural_has_proto(),
        "env": torch_env(),
    }


def refuse_dev() -> None:
    if not neural_has_proto():
        raise RuntimeError("WRITEGEOM DEV requires W1 neural law after this freeze is on origin/main")


def refuse_score() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no writegeom runner.lock — refuse cell scoring")
    if SCORE_DOMAIN != load_prereg()["domains"]["SCORE"]:
        raise RuntimeError("SCORE domain drifted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.verify_prereg:
        p = load_prereg()
        assert p["n"] == 64
        assert p["H_max"] == 8
        assert p["state_budget"] == 2 * 8 * 64
        assert p["margin"]["cosine_margin_min"] == 0.01
        assert p["arms"]["W2"]["lambda"] == 0.01
        assert p["reversal"]["ecological"]["required_w1_pass"] is True
        assert p["reversal"]["positive_only_reassignment"]["required_w1_pass"] is False
        print(json.dumps({"ok": True, "product": p["product"], "H_max": p["H_max"]}, indent=2))
    elif args.dev:
        refuse_dev()
        raise RuntimeError("DEV runner lands with W1 neural law")
    elif args.score:
        refuse_score()
        raise RuntimeError("SCORE opens only after runner.lock and candidate hash on origin/main")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
