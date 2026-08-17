"""TM.0.24.AFFINEMAP.R2 — common two-actuator geometric margin on unused R2 worlds.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
AFFINEMAP V1 on 4a5183e is immutable. Intercept conclusion remains valid.
DEV on unused TM024.AFFINEMAP.R2.DEV. after this freeze is on origin/main.
SCORE reserved. A3 diagnostic only and not authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024affinemap import (
    ARMS,
    BATCH_ARMS,
    EXPECTED_N_ACQUIRE,
    EXPECTED_N_CELLS,
    EXPECTED_N_STABLE,
    EXPECTED_N_TWIN,
    GMIN,
    PassiveAggressiveBias,
    decorate,
    expected_cell_ids,
    expected_ids_sha,
    make_learner,
    make_store,
    train_store,
)
from experiments.run_tm024convergencemap import PassiveAggressive, unique_winner
from experiments.run_tm024eligmap import _fresh, record_rest, unit_or_zero
from experiments.run_tm024lifecyclemarginmap import HardMarginOracle
from experiments.run_tm024memorylifecyclemap_r2 import ingest
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024tracebridge import probe_address, require_query, teach_bridged
from experiments.run_tm024writegeom import capacity_world, domain_seed, mapping_pairs
from three_memory.cortex_lineage import sha_file

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_affinemap.r2.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_affinemap.r2.contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_affinemap.r2.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_affinemap.r2.runner.lock"
MANIFEST = REPO_ROOT / "docs" / "lineage_affinemap.r2.manifest.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_affinemap.r2.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024affinemap_r2_results.md"
V1_PREREG = REPO_ROOT / "docs" / "lineage_affinemap.prereg.lock"
V1_CONTRACT = REPO_ROOT / "docs" / "lineage_affinemap.contract.md"
V1_ISOLATION = REPO_ROOT / "docs" / "lineage_affinemap.isolation.lock"
V1_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_affinemap.runner.lock"
V1_MANIFEST = REPO_ROOT / "docs" / "lineage_affinemap.manifest.lock"
V1_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024affinemap.py"
LMM_DEC = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.lock"
LMM_DEV = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock"
LMM_ADD = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.addendum.lock"
LMM_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.runner.lock"
LMM_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024lifecyclemarginmap.py"
LMM_PREREG = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.prereg.lock"
CVG_RUNNER = REPO_ROOT / "experiments" / "run_tm024convergencemap.py"
D1_RUNNER = REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.AFFINEMAP.R2.DEV."
TWIN_DOMAIN = "TM024.AFFINEMAP.R2.TWIN."
SCORE_DOMAIN = "TM024.AFFINEMAP.R2.SCORE."
SCORE_MARKERS = ("TM024.AFFINEMAP.R2.SCORE.", "TM024.AFFINEMAP.SCORE.")
V1_GIT = "4a5183eedc45417fd29cd639c3f2fcb4c4a87ad3"
V1_RUNNER_LOCK_SHA = "5cd319ecc1872dadcc1e193f05a1794dd850c2d37d8500fa11e2bf96b3577669"
V1_RUNNER_PY_SHA = "be7a360ef8b635085b7cd22490812c311ad2335c463054a7471e443dec664eea"
V1_PREREG_SHA = "746faa55139c010d265142a055903f898b3c26730e529c93d607a904061017d5"
V1_ISOLATION_SHA = "386fe75eece7e8d48c7088de6d4a70f2446e737d7c3aceb670ab795733fc294a"
V1_CONTRACT_SHA = "025ffaa181434548d1e29bad4901266f2c0f346e7d8be48671e5f96ce002d8ba"
V1_MANIFEST_SHA = "e8acd76acf61c62d487487d94a3ea8acdf8a032a61982a1d7d17629d3c600e59"
LMM_DEC_SHA = "851d4a9312a7a8164600f53b857f65d3f50fc22fba136e52f00d3266422ddff0"
LMM_DEV_SHA = "57015fef334b533a77173bb06323e3f28e8d9bc5ad41e3453bfe126ee4a34bf8"
LMM_ADD_SHA = "d4dd4ca797d4c6c0aff6725fa79723abd870491a59f4cae41f73ca03fd75f794"
LMM_RUNNER_LOCK_SHA = "d0e5eee16752a7ae89bdfc16f3e0294ce14cfd4726aa8e71f7a8eba1c7c848dd"
LMM_RUNNER_PY_SHA = "5f7dc1a79e49c42edc45ccd7d12d4c4a8d2989a067071becc433b46c6234ddce"
LMM_PREREG_SHA = "4a753d811a14b428321f88767ce5d018a42dc047eccaa959ab83ed9f4c1ee8e2"
D1_RUNNER_PY_SHA = "06f5f2c6edc0dffef570e75295708ea2816ea737cd0af9dab157cd94f4c26b41"
CVG_RUNNER_PY_SHA = "232cffa23619de1fcdbde7b8c82fc3de8e1c2fbe84a014a40bc27f3723cbbcf6"
EPS = 1e-12
UNIT_EPS = 1e-6


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def affinemap_r2_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "manifest": MANIFEST,
        "candidate_v30": CANDIDATE_V30,
        "v1_prereg": V1_PREREG,
        "v1_contract": V1_CONTRACT,
        "v1_isolation": V1_ISOLATION,
        "v1_runner_lock": V1_RUNNER_LOCK,
        "v1_manifest": V1_MANIFEST,
        "v1_runner": V1_RUNNER_PY,
        "lmm_decision": LMM_DEC,
        "lmm_dev": LMM_DEV,
        "lmm_addendum": LMM_ADD,
        "lmm_runner_lock": LMM_RUNNER_LOCK,
        "lmm_runner": LMM_RUNNER_PY,
        "lmm_prereg": LMM_PREREG,
        "convergencemap_runner": CVG_RUNNER,
        "discrimmap_r2_runner": D1_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def assert_historical_frozen() -> None:
    if sha_file(V1_RUNNER_LOCK) != V1_RUNNER_LOCK_SHA:
        raise RuntimeError("AFFINEMAP V1 runner.lock must remain 4a5183e")
    if sha_file(V1_RUNNER_PY) != V1_RUNNER_PY_SHA:
        raise RuntimeError("AFFINEMAP V1 runner.py must remain 4a5183e")
    if sha_file(V1_PREREG) != V1_PREREG_SHA:
        raise RuntimeError("AFFINEMAP V1 prereg.lock must remain 4a5183e")
    if sha_file(V1_ISOLATION) != V1_ISOLATION_SHA:
        raise RuntimeError("AFFINEMAP V1 isolation.lock must remain 4a5183e")
    if sha_file(V1_CONTRACT) != V1_CONTRACT_SHA:
        raise RuntimeError("AFFINEMAP V1 contract must remain 4a5183e")
    if sha_file(V1_MANIFEST) != V1_MANIFEST_SHA:
        raise RuntimeError("AFFINEMAP V1 manifest.lock must remain 4a5183e")
    if sha_file(LMM_DEC) != LMM_DEC_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP decision.lock must remain the published freeze")
    if sha_file(LMM_DEV) != LMM_DEV_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP DEV lock must remain the published freeze")
    if sha_file(LMM_ADD) != LMM_ADD_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP addendum must remain the published freeze")
    if sha_file(LMM_RUNNER_LOCK) != LMM_RUNNER_LOCK_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP runner.lock must remain the published freeze")
    if sha_file(LMM_RUNNER_PY) != LMM_RUNNER_PY_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP runner.py must remain the published freeze")
    if sha_file(LMM_PREREG) != LMM_PREREG_SHA:
        raise RuntimeError("LIFECYCLEMARGINMAP prereg.lock must remain the published freeze")
    if sha_file(D1_RUNNER) != D1_RUNNER_PY_SHA:
        raise RuntimeError("DISCRIMMAP R2 runner.py must remain the published freeze")
    if sha_file(CVG_RUNNER) != CVG_RUNNER_PY_SHA:
        raise RuntimeError("CONVERGENCEMAP runner.py must remain the published freeze")
    if (REPO_ROOT / "docs" / "lineage_affinemap.dev.lock").exists():
        raise RuntimeError("AFFINEMAP V1 DEV must stay unopened")


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no affinemap r2 runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if affinemap_r2_shas() != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    assert_historical_frozen()
    return lock


def refuse_rerun() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("same frozen DEV execution requested again")


def refuse_score_markers(payload: str) -> None:
    for mark in SCORE_MARKERS:
        if mark in payload:
            raise RuntimeError("SCORE identifier appeared in DEV payload")


def refuse_v31() -> None:
    if CANDIDATE_V31.exists() or CANDIDATE_V32.exists():
        raise RuntimeError("v31/v32 candidate must not exist")


def rival(handles: list[str], want: str) -> str:
    others = [h for h in handles if h != want]
    if len(others) != 1:
        raise RuntimeError("common score requires exactly two handles")
    return others[0]


def two_row_params(learner: Any, handles: list[str]) -> dict[str, tuple[np.ndarray, float]]:
    h0, h1 = handles
    if isinstance(learner, HardMarginOracle):
        w = np.asarray(learner.w, dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(w))
        b_unit = float(learner.b) / nrm if nrm > EPS else 0.0
        return {h0: (w.copy(), b_unit), h1: ((-w).copy(), -b_unit)}
    if isinstance(learner, PassiveAggressiveBias):
        return {h: (np.asarray(learner.pa.rows[h], dtype=np.float64).copy(), float(learner.bias[h])) for h in handles}
    if isinstance(learner, PassiveAggressive):
        return {h: (np.asarray(learner.rows[h], dtype=np.float64).copy(), 0.0) for h in handles}
    raise RuntimeError(f"unknown learner class {type(learner)}")


class CommonTwoRow:
    """Shared two-actuator scores and geometric margin. Not an installed law."""

    def __init__(self, learner: Any, handles: list[str]):
        self.learner = learner
        self.handles = list(handles)
        self.refresh()

    def refresh(self) -> None:
        self.params = two_row_params(self.learner, self.handles)

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        x = unit_or_zero(addr)
        out: dict[str, float] = {}
        for h, (w, b) in self.params.items():
            out[h] = float(np.dot(unit_or_zero(w), x) + float(b))
        return out

    def pairwise_score_gap(self, addr: np.ndarray, want: str) -> float:
        return float(self.margin_state(addr, want)["pairwise_score_gap"])

    def geometric_margin(self, addr: np.ndarray, want: str) -> float:
        return float(self.margin_state(addr, want)["normalized_geometric_margin"])

    def margin_state(self, addr: np.ndarray, want: str) -> dict[str, Any]:
        x = unit_or_zero(addr)
        ot = rival(self.handles, want)
        w_w, b_w = self.params[want]
        w_o, b_o = self.params[ot]
        u_w = unit_or_zero(w_w)
        u_o = unit_or_zero(w_o)
        v = u_w - u_o
        c = float(b_w) - float(b_o)
        v_norm = float(np.linalg.norm(v))
        x_norm = float(np.linalg.norm(x))
        u_w_norm = float(np.linalg.norm(u_w))
        u_o_norm = float(np.linalg.norm(u_o))
        sc = self.scores(addr)
        gap = float(sc[want] - sc[ot])
        if v_norm <= EPS:
            gamma = 0.0
        else:
            gamma = float((float(np.dot(v, x)) + c) / v_norm)
        features_unit = _norm_is_unit_or_zero(x_norm) and _norm_is_unit_or_zero(u_w_norm) and _norm_is_unit_or_zero(u_o_norm)
        return {
            "pairwise_score_gap": gap,
            "normalized_geometric_margin": gamma,
            "c": c,
            "v_norm": v_norm,
            "x_norm": x_norm,
            "want_row_norm": u_w_norm,
            "rival_row_norm": u_o_norm,
            "features_unit": bool(features_unit),
            "zero_separator": bool(v_norm <= EPS and abs(c) <= EPS),
        }


def _norm_is_unit_or_zero(nrm: float) -> bool:
    return bool(abs(float(nrm)) <= EPS or abs(float(nrm) - 1.0) <= UNIT_EPS)


def expected_stability_pass(*, ranking_ok: bool, gamma: float, perturbation_ok: bool) -> bool:
    return bool(ranking_ok and perturbation_ok and float(gamma) >= GMIN)


def perturb_rank(
    score_fn: Callable[[np.ndarray], dict[str, float]],
    addr: np.ndarray,
    winner: str,
    *,
    domain: str,
    key: str,
    p: dict[str, Any],
) -> dict[str, Any]:
    m = p["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = unit_or_zero(addr)
    n_ok = 0
    for _i in range(n):
        unit = unit_or_zero(r0 + rng.normal(0.0, sigma, size=r0.shape))
        scores = score_fn(unit)
        if scores and unique_winner(scores) == winner:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def probe_addr(ag: Any, world: dict[str, Any], cue: str, learner: Any, *, tag: str) -> np.ndarray:
    n0 = int(getattr(learner, "n_updates", 0))
    w0 = ag.W_act_query.detach().clone()
    probe = clone_frozen(ag)
    require_query(probe)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    addr = probe_address("B3", probe, None)
    if int(getattr(learner, "n_updates", 0)) != n0:
        raise RuntimeError("probe updated learner weights")
    if float((ag.W_act_query - w0).abs().max().item()) > 1e-12:
        raise RuntimeError("probe updated organism state")
    return np.asarray(addr, dtype=np.float64)


def probe_pairs_r2(
    ag: Any,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    view: CommonTwoRow,
    *,
    tag: str,
    p: dict[str, Any],
) -> dict[str, Any]:
    gmin = float(p["margin"]["geometric_margin_min"])
    if gmin != GMIN:
        raise RuntimeError("geometric margin min must stay 0.01")
    if p["margin"].get("pairwise_score_gap_is_not_pass_statistic") is not True:
        raise RuntimeError("pairwise score gap must not be the pass statistic")
    probes: list[dict[str, Any]] = []
    ranking_ok = True
    perturbation_ok = True
    geometric_ok = True
    view.refresh()
    for i, (cue, handle) in enumerate(pairs):
        addr = probe_addr(ag, world, cue, view.learner, tag=f"{tag}_p{i}")
        sc = view.scores(addr)
        win = unique_winner(sc)
        rank = bool(win == handle)
        st = view.margin_state(addr, handle)
        gap = float(st["pairwise_score_gap"])
        gamma = float(st["normalized_geometric_margin"])
        stab = perturb_rank(view.scores, addr, win or "", domain=world["domain"], key=f"{tag}_{cue}", p=p)
        pert = bool(stab["stable"])
        ranking_ok = ranking_ok and rank
        perturbation_ok = perturbation_ok and pert
        geometric_ok = geometric_ok and bool(gamma >= gmin)
        probes.append(
            {
                "want": handle,
                "winner": win,
                "ranking_ok": rank,
                "pairwise_score_gap": gap,
                "normalized_geometric_margin": gamma,
                "perturbation_ok": pert,
                "c": float(st["c"]),
                "v_norm": float(st["v_norm"]),
                "x_norm": float(st["x_norm"]),
                "features_unit": bool(st["features_unit"]),
                "zero_separator": bool(st["zero_separator"]),
            }
        )
    gaps = [float(q["pairwise_score_gap"]) for q in probes]
    gammas = [float(q["normalized_geometric_margin"]) for q in probes]
    return {
        "ranking_ok": bool(ranking_ok and probes),
        "perturbation_ok": bool(perturbation_ok and probes),
        "pairwise_score_gap": float(min(gaps) if gaps else 0.0),
        "normalized_geometric_margin": float(min(gammas) if gammas else 0.0),
        "min_pairwise_score_gap": float(min(gaps) if gaps else 0.0),
        "min_normalized_geometric_margin": float(min(gammas) if gammas else 0.0),
        "geometric_ok": bool(geometric_ok and probes),
        "c": float(probes[0]["c"]) if probes else 0.0,
        "v_norm": float(probes[0]["v_norm"]) if probes else 0.0,
        "features_unit": bool(probes) and all(bool(q["features_unit"]) for q in probes),
        "zero_separator": bool(probes) and all(bool(q["zero_separator"]) for q in probes),
        "probes": probes,
        "n_probe": len(probes),
        "pass_statistic": "normalized_geometric_margin",
    }


def attach_probe_traj(traj: dict[str, Any], acquire: dict[str, Any], stable: dict[str, Any] | None) -> dict[str, Any]:
    traj["pass_statistic"] = "normalized_geometric_margin"
    traj["pairwise_score_gap_is_not_pass_statistic"] = True
    traj["stored_min_ranking_margin_is_not_pass_statistic"] = True
    traj["pre_rest_min_normalized_geometric_margin"] = float(acquire["min_normalized_geometric_margin"])
    traj["pre_rest_min_pairwise_score_gap"] = float(acquire["min_pairwise_score_gap"])
    if stable is None:
        traj["post_rest_min_normalized_geometric_margin"] = None
        traj["post_rest_min_pairwise_score_gap"] = None
        return traj
    traj["post_rest_min_normalized_geometric_margin"] = float(stable["min_normalized_geometric_margin"])
    traj["post_rest_min_pairwise_score_gap"] = float(stable["min_pairwise_score_gap"])
    return traj


def eval_phased_map(
    *,
    arm: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = load_prereg()
    gmin = float(p["margin"]["geometric_margin_min"])
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    learner = make_learner(arm, handles, p)
    store = make_store(p)
    captured: list[dict[str, Any]] = []
    n_live = 0
    with tempfile.TemporaryDirectory(prefix="affr2_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        for i, (cue, handle) in enumerate(seq):
            rec = teach_bridged(ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}")
            n_live += 1
            ingest(store, captured, rec["addr"], rec["handle"], rec["adv"], world, cue=cue, tag=f"{tag}_cap{i}")
        if world.get("purpose") == "rename_twin":
            store.score_twin()
        store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
        traj = train_store(arm, learner, store, handles, int(p["arms"][arm]["replay_epochs"]), gmin, p)
        view = CommonTwoRow(learner, handles)
        acquire = probe_pairs_r2(ag, world, pairs, view, tag=f"{tag}_acq", p=p)
        record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
        view.refresh()
        stable = probe_pairs_r2(ag, world, pairs, view, tag=f"{tag}_st", p=p)
    attach_probe_traj(traj, acquire, stable)
    if arm == "A3":
        traj["n_bias_updates"] = int(getattr(learner, "n_bias_updates", 0))
        traj["bias"] = {h: float(v) for h, v in getattr(learner, "bias", {}).items()}
    shared = {
        "n_live_teaches": int(n_live),
        "n_replay_updates": int(traj["n_replay_calls"]),
        "n_updates": int(getattr(learner, "n_updates", traj["n_actual_updates"])),
        "store": store.stats(),
        "matcher": store.ledger.summary(),
        "ceiling_only": arm in BATCH_ARMS,
        "a3_diagnostic_only": arm == "A3",
        "n_cues": len(pairs),
        "n_live_reversal_updates": 0,
        "matched_live_reversal": False,
        "stability_gate": "normalized_geometric_margin_0.01_and_perturbation",
        "bounded_match_sanity_used_for_pass": False,
        "margin_trajectory": traj,
        "replay_learner": str(p["arms"][arm]["replay_learner"]),
        "store_policy": str(p["arms"][arm]["policy"]),
        "pass_statistic": "normalized_geometric_margin",
        "affine_b": traj.get("affine_b"),
        "d1_status": traj.get("d1_status"),
    }
    acq_out = {
        **shared,
        **acquire,
        "passed": bool(acquire["ranking_ok"]),
        "phase": "acquisition",
    }
    stab_out = {
        **shared,
        **stable,
        "passed": expected_stability_pass(
            ranking_ok=bool(stable["ranking_ok"]),
            gamma=float(stable["min_normalized_geometric_margin"]),
            perturbation_ok=bool(stable["perturbation_ok"]),
        ),
        "phase": "stability",
    }
    return acq_out, stab_out


def acquire_all_ok(cells: list[dict[str, Any]], arm: str, n_cues: int) -> bool:
    rows = [c for c in cells if c["arm"] == arm and c["kind"] == "acquire" and int(c["n_cues"]) == int(n_cues)]
    return bool(rows) and all(bool(c["passed"]) and bool(c.get("ranking_ok")) for c in rows)


def robust_all_ok(cells: list[dict[str, Any]], arm: str, n_cues: int) -> bool:
    rows = [c for c in cells if c["arm"] == arm and c["kind"] == "stable" and int(c["n_cues"]) == int(n_cues)]
    return bool(rows) and all(
        expected_stability_pass(
            ranking_ok=bool(c.get("ranking_ok")),
            gamma=float(c.get("min_normalized_geometric_margin") or c.get("normalized_geometric_margin") or 0.0),
            perturbation_ok=bool(c.get("perturbation_ok")),
        )
        for c in rows
    )


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    flags = {
        "A0_acquire_4": acquire_all_ok(cells, "A0", 4),
        "A0_acquire_8": acquire_all_ok(cells, "A0", 8),
        "A1_acquire_4": acquire_all_ok(cells, "A1", 4),
        "A1_acquire_8": acquire_all_ok(cells, "A1", 8),
        "A2_acquire_4": acquire_all_ok(cells, "A2", 4),
        "A2_acquire_8": acquire_all_ok(cells, "A2", 8),
        "A3_acquire_4": acquire_all_ok(cells, "A3", 4),
        "A3_acquire_8": acquire_all_ok(cells, "A3", 8),
        "A2_robust_4": robust_all_ok(cells, "A2", 4),
        "A2_robust_8": robust_all_ok(cells, "A2", 8),
        "A3_robust_4": robust_all_ok(cells, "A3", 4),
        "A3_robust_8": robust_all_ok(cells, "A3", 8),
        "m1_previously_failed_four_cue_acquire": True,
        "a3_implementation_authorized": bool(p.get("a3_implementation_authorized")),
        "a3_raw_gap_increase_is_not_bias_support": True,
    }
    a0_48 = flags["A0_acquire_4"] and flags["A0_acquire_8"]
    a1_48 = flags["A1_acquire_4"] and flags["A1_acquire_8"]
    a2_48 = flags["A2_acquire_4"] and flags["A2_acquire_8"]
    a1_fail = (not flags["A1_acquire_4"]) or (not flags["A1_acquire_8"])
    a2_fail = (not flags["A2_acquire_4"]) or (not flags["A2_acquire_8"])
    a3_robust_48 = flags["A3_robust_4"] and flags["A3_robust_8"]
    a2_robust_fail = (not flags["A2_robust_4"]) or (not flags["A2_robust_8"])
    both_fail_4 = (not flags["A0_acquire_4"]) and (not flags["A1_acquire_4"])
    both_fail_8 = (not flags["A0_acquire_8"]) and (not flags["A1_acquire_8"])
    ladder = [(r["id"], r["then"]) for r in p["decision_ladder"]]
    if a0_48 and a1_fail:
        code, then = ladder[0]
    elif a1_48 and a2_fail:
        code, then = ladder[1]
    elif a2_48:
        code, then = ladder[2]
    elif a3_robust_48 and a2_robust_fail:
        code, then = ladder[3]
    elif both_fail_4 or both_fail_8:
        code, then = ladder[4]
    else:
        code, then = ladder[5]
    return code, then, flags


def cell_manifest_hash(cells: list[dict[str, Any]]) -> str:
    rows = []
    for c in cells:
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "passed": c["passed"],
                "ranking_ok": c.get("ranking_ok"),
                "pairwise_score_gap": c.get("pairwise_score_gap"),
                "normalized_geometric_margin": c.get("normalized_geometric_margin"),
                "perturbation_ok": c.get("perturbation_ok"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_cell_metrics(c: dict[str, Any]) -> None:
    cid = str(c.get("id") or "")
    if c.get("kind") in ("eco", "spec"):
        raise RuntimeError("eco/spec cells are refused")
    for key in ("ranking_ok", "pairwise_score_gap", "normalized_geometric_margin", "perturbation_ok"):
        if key not in c:
            raise RuntimeError(f"missing recorded field {key} on {cid}")
    if c.get("pass_statistic") != "normalized_geometric_margin":
        raise RuntimeError(f"pass statistic must be geometric {cid}")
    gamma = float(c["normalized_geometric_margin"])
    gap = float(c["pairwise_score_gap"])
    if "min_normalized_geometric_margin" in c:
        if abs(float(c["min_normalized_geometric_margin"]) - gamma) > 1e-12:
            raise RuntimeError(f"min geometric field must match recorded gamma {cid}")
    if "geometric_ok" in c:
        if bool(c["geometric_ok"]) != bool(gamma >= GMIN):
            raise RuntimeError(f"geometric_ok must follow normalized geometric margin {cid}")
    if c.get("kind") in ("acquire", "twin"):
        if bool(c.get("passed")) != bool(c.get("ranking_ok")):
            raise RuntimeError(f"acquire/twin pass must be ranking_ok {cid}")
    if c.get("kind") == "stable":
        expected = expected_stability_pass(
            ranking_ok=bool(c.get("ranking_ok")),
            gamma=gamma,
            perturbation_ok=bool(c.get("perturbation_ok")),
        )
        if bool(c.get("passed")) != expected:
            if bool(c.get("passed")) and gamma < GMIN and gap >= GMIN:
                raise RuntimeError(f"pairwise_score_gap_used_as_geometric_gate {cid}")
            raise RuntimeError(f"scoring inconsistency {cid}")
    if c.get("arm") in ("A1", "A2"):
        v_norm = c.get("v_norm")
        c_off = c.get("c")
        if v_norm is not None and c_off is not None:
            v_n = float(v_norm)
            c_v = float(c_off)
            if abs(c_v) <= EPS and v_n <= EPS and bool(c.get("ranking_ok")):
                raise RuntimeError(f"zero effective separator cannot rank {cid}")
            features_unit = c.get("features_unit")
            if features_unit is True and abs(c_v) <= EPS and v_n > EPS and abs(gamma) > 1.0 + UNIT_EPS:
                raise RuntimeError(f"homogeneous geometric margin exceeded 1 {cid}")
    if SCORE_DOMAIN in str(c.get("domain") or ""):
        raise RuntimeError("SCORE identifier appeared in DEV payload")
    if c.get("bounded_match_sanity_used_for_pass"):
        raise RuntimeError(f"bounded match used as stability gate {cid}")


def assert_cell_coverage(cells: list[dict[str, Any]]) -> str:
    ids = [c["id"] for c in cells]
    expected = expected_cell_ids()
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    if set(ids) != set(expected):
        raise RuntimeError("cell IDs do not match frozen AFFINEMAP grid")
    kinds = Counter(c["kind"] for c in cells)
    if dict(kinds) != {"acquire": EXPECTED_N_ACQUIRE, "stable": EXPECTED_N_STABLE, "twin": EXPECTED_N_TWIN}:
        raise RuntimeError(f"kind counts {dict(kinds)}")
    for c in cells:
        assert_cell_metrics(c)
    return cell_manifest_hash(cells)


def _append_arm(cells: list[dict[str, Any]], arm: str) -> None:
    p = load_prereg()
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                acq, stab = eval_phased_map(
                    arm=arm, world=world, pairs=pairs, order=order, tag=f"affr2_{arm}_{wi}_{n_cues}_{order}"
                )
                cells.append(
                    decorate(acq, kind="acquire", arm=arm, n_cues=n_cues, order=order, world=wi, domain=world["domain"])
                )
                cells.append(
                    decorate(stab, kind="stable", arm=arm, n_cues=n_cues, order=order, world=wi, domain=world["domain"])
                )
    world_t = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
    world_t["purpose"] = "rename_twin"
    pairs_t = mapping_pairs(world_t, flip=False)
    for order in TEACH_ORDERS:
        acq, _stab = eval_phased_map(arm=arm, world=world_t, pairs=pairs_t, order=order, tag=f"affr2_{arm}_twin_{order}")
        cells.append(decorate(acq, kind="twin", arm=arm, n_cues=2, order=order, world=1, domain=world_t["domain"]))


def run_dev() -> dict[str, Any]:
    refuse_v31()
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    if p.get("open_v1_dev") is True:
        raise RuntimeError("AFFINEMAP V1 DEV is refused")
    if p.get("a3_implementation_authorized") is True:
        raise RuntimeError("A3 must remain unauthorized")
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        _append_arm(cells, arm)
    for c in cells:
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
    manifest = assert_cell_coverage(cells)
    code, then, extra = _decision(cells, p)
    out = {
        "version": "TM.0.24.AFFINEMAP.R2.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain_opened": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "a3_implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "pass_statistic": "normalized_geometric_margin",
        "phase_flags": extra,
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells),
        "manifest_sha": manifest,
        "cells": cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": affinemap_r2_shas(),
        "note": "AFFINEMAP R2 DEV only. V1 4a5183e preserved. A3 diagnostic-only. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def smoke() -> dict[str, Any]:
    p = load_prereg()
    world = capacity_world(0, "TM024.AFFINEMAP.R2.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    a0, a0s = eval_phased_map(arm="A0", world=world, pairs=pairs, order="A_then_B", tag="affr2smk0")
    a1, _a1s = eval_phased_map(arm="A1", world=world, pairs=pairs, order="A_then_B", tag="affr2smk1")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "a0_fields": [
            k
            for k in ("ranking_ok", "pairwise_score_gap", "normalized_geometric_margin", "perturbation_ok")
            if k in a0
        ],
        "a0_stable_passed_uses_geometric": bool(a0s["passed"] == (a0s["ranking_ok"] and a0s["geometric_ok"] and a0s["perturbation_ok"])),
        "a1_gamma": float(a1["min_normalized_geometric_margin"]),
        "a1_gap": float(a1["min_pairwise_score_gap"]),
        "a1_gamma_le_1": bool(float(a1["min_normalized_geometric_margin"]) <= 1.0 + 1e-6),
        "pass_statistic": a0.get("pass_statistic"),
        "expected_id_count": len(expected_cell_ids()),
        "skip_eco_spec": True,
        "replay_epochs": int(p["arms"]["A2"]["replay_epochs"]),
    }


def write_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        raise RuntimeError("affinemap r2 manifest already exists")
    refuse_v31()
    ids = expected_cell_ids()
    shas = {k: v for k, v in affinemap_r2_shas().items() if k != "manifest"}
    out = {
        "version": "TM.0.24.AFFINEMAP.R2.MANIFEST",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "lab": "TM.0.24.AFFINEMAP.R2",
        "expected_n_cells": EXPECTED_N_CELLS,
        "expected_kind_counts": {"acquire": EXPECTED_N_ACQUIRE, "stable": EXPECTED_N_STABLE, "twin": EXPECTED_N_TWIN},
        "id_format": "{kind}|{arm}|c{n_cues}|{order}|w{world}",
        "expected_cell_ids": ids,
        "expected_ids_sha": expected_ids_sha(ids),
        "domains": {"DEV": DEV_DOMAIN, "TWIN": TWIN_DOMAIN, "SCORE": SCORE_DOMAIN},
        "recorded_fields": ["ranking_ok", "pairwise_score_gap", "normalized_geometric_margin", "perturbation_ok"],
        "pass_statistic": "normalized_geometric_margin",
        "historical_affinemap_v1_git_head": V1_GIT,
        "shas": shas,
        "n": 64,
        "note": "AFFINEMAP R2 cell-ID manifest. V1 4a5183e preserved. Product remains 0.0.004.",
    }
    MANIFEST.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("affinemap r2 runner.lock already exists")
    refuse_v31()
    assert_historical_frozen()
    if not MANIFEST.exists():
        write_manifest()
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.AFFINEMAP.R2.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "a3_implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "shas": affinemap_r2_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "arms": list(ARMS),
        "skip_eco_spec": True,
        "pass_statistic": "normalized_geometric_margin",
        "pairwise_score_gap_is_not_pass_statistic": True,
        "recorded_fields": ["ranking_ok", "pairwise_score_gap", "normalized_geometric_margin", "perturbation_ok"],
        "match_radius": float(prereg["match"]["radius"]),
        "replay_epochs": 16,
        "expected_n_cells": EXPECTED_N_CELLS,
        "historical_affinemap_v1_git_head": V1_GIT,
        "historical_affinemap_v1_runner_lock_sha": V1_RUNNER_LOCK_SHA,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen AFFINEMAP R2 runner. V1 4a5183e preserved. DEV lock only after this file is on origin/main.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    refuse_rerun()
    manifest = assert_cell_coverage(out["cells"])
    if out.get("manifest_sha") != manifest:
        raise RuntimeError("DEV manifest hash must be asserted before write")
    refuse_score_markers(json.dumps(out, default=str))
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("affinemap r2 decision lock already exists")
    out = {
        "version": "TM.0.24.AFFINEMAP.R2.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "neural_edit": False,
        "implementation_authorized": False,
        "a3_implementation_authorized": False,
        "pass_statistic": "normalized_geometric_margin",
        "decision": {"code": dev["decision_code"], "then": dev["decision_then"], "phase_flags": dev.get("phase_flags")},
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "git_head": _git_head(),
        "note": "AFFINEMAP R2 common geometric margin. V1 preserved. A3 unauthorized. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.AFFINEMAP.R2 DEV\n\n"
        f"Decision: **{out['decision']['code']}**.\n\n"
        "V1 4a5183e preserved. Pass statistic is normalized geometric margin. "
        "Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze installs a sufficient write rule")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("AFFINEMAP R2 DEV lock requires runner.lock on origin/main after this freeze")
    assert_runner_frozen()
    refuse_rerun()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--write-dev-lock", action="store_true")
    ap.add_argument("--write-decision", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.verify_prereg:
        p = load_prereg()
        assert p["n"] == 64
        assert p["open_v1_dev"] is False
        assert p["margin"]["pass_statistic"] == "normalized_geometric_margin"
        assert p["common_score"]["only_normalized_geometric_margin_satisfies_0.01_gate"] is True
        assert p["a3_implementation_authorized"] is False
        assert p["domains"]["DEV"] == DEV_DOMAIN
        assert_historical_frozen()
        print(json.dumps({"ok": True, "product": p["product"], "expected_n_cells": EXPECTED_N_CELLS}, indent=2))
    elif args.write_manifest:
        print(json.dumps(write_manifest(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2, default=str))
    elif args.dev:
        out = run_dev()
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_dev_lock:
        refuse_dev_lock()
    elif args.write_decision:
        if not DEV_LOCK.exists():
            raise RuntimeError("DEV lock required before decision")
        dev = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        print(json.dumps(write_decision(dev), indent=2, default=str))
    elif args.score:
        refuse_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
