"""TM.0.24.LIFECYCLEMARGINMAP — margin-conditioned replay on the frozen L2 lifecycle.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
MEMORYLIFECYCLEMAP R2/V1 locks are immutable. Write-geometry closed. W1 not resurrected.
DEV on unused TM024.LIFECYCLEMARGINMAP.DEV. after this freeze is on origin/main.
SCORE reserved and unopened. No trace or neural candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024convergencemap import ErrorOnlyBank, PassiveAggressive, unique_winner
from experiments.run_tm024discrimmap_r2 import hard_margin_linear
from experiments.run_tm024eligmap import _fresh, record_rest, unit_or_zero
from experiments.run_tm024memorylifecyclemap_r2 import (
    EpisodeStore,
    checkpoint_error,
    desired_winner,
    ingest,
    live_probe,
    probe_pairs,
    perturb_rank,
    snapshot_rows,
)
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024tracebridge import require_query, teach_bridged
from experiments.run_tm024writegeom import NEG_DELTA, capacity_world, mapping_pairs, ranking_margin, set_handle_delta
from three_memory.cortex_lineage import sha_file

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.runner.lock"
MANIFEST = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.manifest.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_lifecyclemarginmap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024lifecyclemarginmap_results.md"
R2_PREREG = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.prereg.lock"
R2_CONTRACT = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.contract.md"
R2_ISOLATION = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.isolation.lock"
R2_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.runner.lock"
R2_DEV = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.dev.lock"
R2_DEC = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.lock"
R2_ADD = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.addendum.lock"
R2_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap_r2.py"
V1_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock"
V1_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py"
CVG_DEC = REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock"
CVG_ADD = REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock"
CVG_RUNNER = REPO_ROOT / "experiments" / "run_tm024convergencemap.py"
TB_RUNNER = REPO_ROOT / "experiments" / "run_tm024tracebridge.py"
WG_RUNNER = REPO_ROOT / "experiments" / "run_tm024writegeom.py"
D1_PREREG = REPO_ROOT / "docs" / "lineage_discrimmap.r2.prereg.lock"
D1_RUNNER = REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.LIFECYCLEMARGINMAP.DEV."
TWIN_DOMAIN = "TM024.LIFECYCLEMARGINMAP.TWIN."
SCORE_DOMAIN = "TM024.LIFECYCLEMARGINMAP.SCORE."
SCORE_MARKERS = (
    "TM024.LIFECYCLEMARGINMAP.SCORE.",
    "TM024.MEMORYLIFECYCLEMAP.R2.SCORE.",
    "TM024.MEMORYLIFECYCLEMAP.SCORE.",
)
ARMS = ("M0", "M1", "M2", "M3")
KINDS = ("acquire", "stable", "twin", "eco", "spec")
MATCHED_LIVE_ARMS = ("M0", "M1", "M2")
R2_DEC_SHA = "484c38d90582b650633e76a9a92481022a5d3c97308c72e8d51d30d6c9b266dd"
R2_DEV_SHA = "9321e57bb4f3bd1f4fe108c8fcb7751eca4fdb9da3d23401da5e5e2abd09eaed"
R2_ADD_SHA = "92321043267e95092863e1d6e0ac08256d36cdf313c11078f352471fb25c7228"
R2_RUNNER_LOCK_SHA = "c05c9254e9e1b1d6b6039d7cee43b487f83a53b2d1c0b46b60feb039dd6a1077"
R2_RUNNER_PY_SHA = "30f3c4ee67fb4e6524088cc545232682ea7c189758c06d11db0a80428af825a2"
R2_PREREG_SHA = "d1063d354cbe4162355c377d8e7a42cf6508035c9f7c6273291096477c8a2924"
V1_RUNNER_LOCK_SHA = "28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593"
V1_RUNNER_PY_SHA = "edec8809938f3f1ab77948feb3661bea4fc3e6bb1abf573a81089ae628dfc974"
EPS = 1e-12
EXPECTED_N_ACQUIRE = 48
EXPECTED_N_STABLE = 48
EXPECTED_N_TWIN = 8
EXPECTED_N_ECO = 4
EXPECTED_N_SPEC = 4
EXPECTED_N_CELLS = EXPECTED_N_ACQUIRE + EXPECTED_N_STABLE + EXPECTED_N_TWIN + EXPECTED_N_ECO + EXPECTED_N_SPEC
N_SLOTS = 8
ROW_DIM = 64
MAX_STATE_SCALARS = N_SLOTS * ROW_DIM
GMIN = 0.01


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def lifecyclemarginmap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "manifest": MANIFEST,
        "candidate_v30": CANDIDATE_V30,
        "r2_decision": R2_DEC,
        "r2_dev": R2_DEV,
        "r2_addendum": R2_ADD,
        "r2_runner_lock": R2_RUNNER_LOCK,
        "r2_runner": R2_RUNNER_PY,
        "r2_prereg": R2_PREREG,
        "v1_runner_lock": V1_RUNNER_LOCK,
        "v1_runner": V1_RUNNER_PY,
        "convergencemap_decision": CVG_DEC,
        "convergencemap_addendum": CVG_ADD,
        "convergencemap_runner": CVG_RUNNER,
        "tracebridge_runner": TB_RUNNER,
        "writegeom_runner": WG_RUNNER,
        "discrimmap_r2_prereg": D1_PREREG,
        "discrimmap_r2_runner": D1_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, arm: str, n_cues: int, order: str, world: int) -> str:
    if kind not in KINDS:
        raise RuntimeError(f"kind must be one of {KINDS}, got {kind}")
    if arm not in ARMS:
        raise RuntimeError(f"unknown arm {arm}")
    return f"{kind}|{arm}|c{n_cues}|{order}|w{world}"


def assert_historical_frozen() -> None:
    if sha_file(R2_DEC) != R2_DEC_SHA:
        raise RuntimeError("R2 decision.lock must remain the published freeze")
    if sha_file(R2_DEV) != R2_DEV_SHA:
        raise RuntimeError("R2 DEV lock must remain the published freeze")
    if sha_file(R2_ADD) != R2_ADD_SHA:
        raise RuntimeError("R2 decision addendum must remain the published freeze")
    if sha_file(R2_RUNNER_LOCK) != R2_RUNNER_LOCK_SHA:
        raise RuntimeError("R2 runner.lock must remain the published freeze")
    if sha_file(R2_RUNNER_PY) != R2_RUNNER_PY_SHA:
        raise RuntimeError("R2 runner.py must remain the published freeze")
    if sha_file(R2_PREREG) != R2_PREREG_SHA:
        raise RuntimeError("R2 prereg.lock must remain the published freeze")
    if sha_file(V1_RUNNER_LOCK) != V1_RUNNER_LOCK_SHA:
        raise RuntimeError("V1 runner.lock must remain the published freeze")
    if sha_file(V1_RUNNER_PY) != V1_RUNNER_PY_SHA:
        raise RuntimeError("V1 runner.py must remain the published freeze")


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no lifecyclemarginmap runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    shas = lifecyclemarginmap_shas()
    if shas != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
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


def reversal_live_learner(arm: str, learner: Any) -> Any | None:
    if arm in MATCHED_LIVE_ARMS:
        return learner
    return None


class HardMarginOracle:
    """Runner-only D1 ceiling on stored P1 rows. Not installed."""

    def __init__(self, handles: list[str], spec: dict[str, Any]):
        if len(handles) != 2:
            raise RuntimeError("D1 oracle requires exactly two handles")
        if spec.get("soft_margin") or spec.get("soft_margin_C") is not None:
            raise RuntimeError("soft-margin is refused")
        self.handles = list(handles)
        self.spec = spec
        self.w = np.zeros(64, dtype=np.float64)
        self.b = 0.0
        self.status = "infeasible"
        self.n_updates = 0
        self.n_sv = 0

    def update(self, _addr: np.ndarray, _chosen: str, _adv: float) -> None:
        return

    def fit(self, rows: list[dict[str, Any]]) -> None:
        h0, h1 = self.handles
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for r in rows:
            want = desired_winner(self.handles, str(r["handle"]), float(r["adv"]))
            if want not in (h0, h1):
                continue
            xs.append(unit_or_zero(r["p1"]))
            ys.append(1.0 if want == h0 else -1.0)
        if len(xs) < 2 or len(set(ys)) < 2:
            self.w = np.zeros(64, dtype=np.float64)
            self.b = 0.0
            self.status = "infeasible"
            self.n_sv = 0
            self.n_updates += 1
            return
        fit = hard_margin_linear(np.stack(xs, axis=0), np.asarray(ys, dtype=np.float64), self.spec)
        self.w = np.asarray(fit["w"], dtype=np.float64).reshape(-1)
        self.b = float(fit["b"])
        self.status = str(fit["status"])
        self.n_sv = int(fit.get("n_sv") or 0)
        self.n_updates += 1

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        x = unit_or_zero(addr)
        s = float(np.dot(self.w, x) + self.b)
        h0, h1 = self.handles
        return {h0: s, h1: -s}


def make_learner(arm: str, handles: list[str], p: dict[str, Any]) -> Any:
    if p.get("no_new_eta_grid") is not True:
        raise RuntimeError("new eta grid is refused")
    if arm == "M0":
        a = p["learner"]["error_only"]
        if float(a["eta"]) != 0.15 or float(a["c_max"]) != 1.0:
            raise RuntimeError("error-only eta/c_max must stay frozen")
        return ErrorOnlyBank(handles, eta=float(a["eta"]), c_max=float(a["c_max"]))
    if arm in ("M1", "M2"):
        a = p["learner"]["c3_passive_aggressive"]
        if a.get("learning_rate_grid"):
            raise RuntimeError("passive-aggressive learning-rate grid is refused")
        if a.get("clipnorm"):
            raise RuntimeError("passive-aggressive clipnorm is refused")
        if float(a["geometric_margin_target"]) != GMIN:
            raise RuntimeError("C3 geometric margin target must stay 0.01")
        return PassiveAggressive(handles, gamma=float(a["geometric_margin_target"]))
    if arm == "M3":
        return HardMarginOracle(handles, p["learner"]["d1_hard_margin"])
    raise RuntimeError(f"unknown arm {arm}")


def make_store(arm: str, p: dict[str, Any]) -> EpisodeStore:
    spec = p["arms"][arm]
    radius = float(p["match"]["radius"])
    if radius != 0.05:
        raise RuntimeError("match radius must stay 0.05")
    return EpisodeStore(policy=str(spec["policy"]), n_slots=N_SLOTS, match_l2=radius)


def stored_state(learner: Any, handles: list[str], rows: list[dict[str, Any]], gmin: float) -> dict[str, Any]:
    n_wrong = 0
    n_below = 0
    margins: list[float] = []
    for r in rows:
        want = desired_winner(handles, str(r["handle"]), float(r["adv"]))
        sc = learner.scores(r["p1"])
        win = unique_winner(sc)
        if win != want:
            n_wrong += 1
        rm = ranking_margin(sc, str(want)) if want else 0.0
        margins.append(float(rm))
        if float(rm) < float(gmin):
            n_below += 1
    return {
        "n_wrong": int(n_wrong),
        "n_below_0.01": int(n_below),
        "min_ranking_margin": float(min(margins) if margins else 0.0),
        "all_correct": bool(rows) and n_wrong == 0,
    }


def empty_traj() -> dict[str, Any]:
    return {
        "n_replay_calls": 0,
        "n_actual_updates": 0,
        "first_all_correct_call": None,
        "n_replay_after_first_all_correct": None,
        "n_below_0.01_at_first_all_correct": None,
        "min_ranking_margin_at_first_all_correct": None,
        "after_replay_min_stored_margin": 0.0,
        "after_replay_n_below_0.01": 0,
        "n_checkpoint_errors": 0,
        "pre_rest_min_probe_margin": None,
        "post_rest_min_probe_margin": None,
        "rest_delta": None,
        "rest_reduced_margin": None,
        "batch_fit": False,
        "d1_status": None,
    }


def replay_traced(
    learner: Any,
    handles: list[str],
    rows: list[dict[str, Any]],
    epochs: int,
    gmin: float,
) -> dict[str, Any]:
    frozen = snapshot_rows(rows)
    traj = empty_traj()
    if not frozen:
        return traj
    n0 = int(getattr(learner, "n_updates", 0))
    n_calls = 0
    n_ck_err = 0
    first: dict[str, Any] | None = None
    for _cy in range(int(epochs)):
        for r in frozen:
            learner.update(r["p1"], r["handle"], r["adv"])
            n_calls += 1
            n_ck_err += checkpoint_error(learner, handles, frozen)
            st = stored_state(learner, handles, frozen, gmin)
            if first is None and st["all_correct"]:
                first = {
                    "call": n_calls,
                    "n_below_0.01": st["n_below_0.01"],
                    "min_ranking_margin": st["min_ranking_margin"],
                }
    end = stored_state(learner, handles, frozen, gmin)
    traj["n_replay_calls"] = int(n_calls)
    traj["n_actual_updates"] = int(getattr(learner, "n_updates", n0) - n0)
    traj["n_checkpoint_errors"] = int(n_ck_err)
    if first is not None:
        traj["first_all_correct_call"] = int(first["call"])
        traj["n_replay_after_first_all_correct"] = int(n_calls - int(first["call"]))
        traj["n_below_0.01_at_first_all_correct"] = int(first["n_below_0.01"])
        traj["min_ranking_margin_at_first_all_correct"] = float(first["min_ranking_margin"])
    traj["after_replay_min_stored_margin"] = float(end["min_ranking_margin"])
    traj["after_replay_n_below_0.01"] = int(end["n_below_0.01"])
    return traj


def fit_oracle_traced(learner: HardMarginOracle, handles: list[str], rows: list[dict[str, Any]], gmin: float) -> dict[str, Any]:
    frozen = snapshot_rows(rows)
    traj = empty_traj()
    traj["batch_fit"] = True
    learner.fit(frozen)
    st = stored_state(learner, handles, frozen, gmin)
    traj["n_replay_calls"] = 0
    traj["n_actual_updates"] = int(learner.n_updates)
    traj["d1_status"] = str(learner.status)
    if st["all_correct"]:
        traj["first_all_correct_call"] = 0
        traj["n_replay_after_first_all_correct"] = 0
        traj["n_below_0.01_at_first_all_correct"] = int(st["n_below_0.01"])
        traj["min_ranking_margin_at_first_all_correct"] = float(st["min_ranking_margin"])
    traj["after_replay_min_stored_margin"] = float(st["min_ranking_margin"])
    traj["after_replay_n_below_0.01"] = int(st["n_below_0.01"])
    return traj


def attach_probe_traj(traj: dict[str, Any], acquire: dict[str, Any], stable: dict[str, Any] | None) -> dict[str, Any]:
    traj["pre_rest_min_probe_margin"] = float(acquire["min_probe_margin"])
    if stable is None:
        traj["post_rest_min_probe_margin"] = None
        traj["rest_delta"] = None
        traj["rest_reduced_margin"] = None
        return traj
    traj["post_rest_min_probe_margin"] = float(stable["min_probe_margin"])
    delta = float(stable["min_probe_margin"]) - float(acquire["min_probe_margin"])
    traj["rest_delta"] = float(delta)
    traj["rest_reduced_margin"] = bool(delta < -1e-12)
    return traj


def train_store(
    arm: str,
    learner: Any,
    store: EpisodeStore,
    captured: list[dict[str, Any]],
    handles: list[str],
    rows_epochs: int,
    gmin: float,
    p: dict[str, Any],
) -> dict[str, Any]:
    if int(rows_epochs) != 16:
        raise RuntimeError("replay budget must stay 16")
    if float(p["match"]["radius"]) != 0.05:
        raise RuntimeError("match radius must stay 0.05")
    rows = store.valid_rows()
    if arm == "M3":
        return fit_oracle_traced(learner, handles, rows, gmin)
    return replay_traced(learner, handles, rows, rows_epochs, gmin)


def eval_phased_map(
    *,
    arm: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    learner = make_learner(arm, handles, p)
    store = make_store(arm, p)
    captured: list[dict[str, Any]] = []
    n_live = 0
    with tempfile.TemporaryDirectory(prefix="lmm_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        for i, (cue, handle) in enumerate(seq):
            rec = teach_bridged(ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}")
            n_live += 1
            ingest(store, captured, rec["addr"], rec["handle"], rec["adv"], world, cue=cue, tag=f"{tag}_cap{i}")
        if world.get("purpose") == "rename_twin":
            store.score_twin()
        store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
        traj = train_store(arm, learner, store, captured, handles, int(p["arms"][arm]["replay_epochs"]), gmin, p)
        acquire = probe_pairs(ag, world, pairs, learner, tag=f"{tag}_acq", gmin=gmin)
        record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
        stable = probe_pairs(ag, world, pairs, learner, tag=f"{tag}_st", gmin=gmin)
    attach_probe_traj(traj, acquire, stable)
    shared = {
        "n_live_teaches": int(n_live),
        "n_replay_updates": int(traj["n_replay_calls"]),
        "n_updates": int(getattr(learner, "n_updates", traj["n_actual_updates"])),
        "store": store.stats(),
        "matcher": store.ledger.summary(),
        "m3_ceiling_only": arm == "M3",
        "n_cues": len(pairs),
        "n_live_reversal_updates": 0,
        "matched_live_reversal": arm in MATCHED_LIVE_ARMS,
        "stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_match_sanity_used_for_pass": False,
        "margin_trajectory": traj,
        "replay_learner": str(p["arms"][arm]["replay_learner"]),
        "store_policy": str(p["arms"][arm]["policy"]),
    }
    acq_out = {**shared, **acquire, "passed": bool(acquire["ranking_ok"]), "phase": "acquisition"}
    stab_out = {**shared, **stable, "passed": bool(stable["robust_ok"]), "phase": "stability"}
    return acq_out, stab_out


def eval_ecological(arm: str, world: dict[str, Any], *, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1 = world["handles"][0]
    h2 = world["handles"][1]
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    handles = list(world["handles"])
    learner = make_learner(arm, handles, p)
    store = make_store(arm, p)
    captured: list[dict[str, Any]] = []
    advs: list[float] = []
    n_live = 0
    n_live_reversal = 0
    with tempfile.TemporaryDirectory(prefix="lmm_eco_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)

        def teach(w: dict[str, Any], handle: str, suffix: str) -> dict[str, Any]:
            return teach_bridged(ag, w, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_{suffix}")

        t1 = teach(world, h1, "p")
        wneg = set_handle_delta(world, h1, NEG_DELTA)
        t2 = teach(wneg, h1, "n")
        t3 = teach(world, h2, "r")
        advs = [float(t1["adv"]), float(t2["adv"]), float(t3["adv"])]
        n_live += 1
        ingest(store, captured, t1["addr"], t1["handle"], t1["adv"], world, cue=cue, tag=f"{tag}_p")
        n_live += 1
        i_neg = ingest(
            store,
            captured,
            t2["addr"],
            t2["handle"],
            t2["adv"],
            world,
            cue=cue,
            reversal=True,
            tag=f"{tag}_n",
            live_learner=reversal_live_learner(arm, learner),
        )
        n_live_reversal += int(bool(i_neg.get("live_trained")))
        n_live += 1
        i_rev = ingest(
            store,
            captured,
            t3["addr"],
            t3["handle"],
            t3["adv"],
            world,
            cue=cue,
            reversal=True,
            tag=f"{tag}_r",
            live_learner=reversal_live_learner(arm, learner),
        )
        n_live_reversal += int(bool(i_rev.get("live_trained")))
        store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
        traj = train_store(arm, learner, store, captured, handles, int(p["arms"][arm]["replay_epochs"]), gmin, p)
        pr = live_probe(ag, world, cue, learner, tag=f"{tag}_q")
        stab = perturb_rank(
            learner.scores,
            pr["addr"],
            pr["winner"] or "",
            domain=world["domain"],
            key=f"{tag}_eco",
        )
        win = pr["winner"]
        margin = float(pr["margin"])
        ranking_ok = bool(win == h2)
        attach_probe_traj(traj, {"min_probe_margin": margin}, None)
        passed = bool(
            len(advs) == 3
            and advs[0] > 0.0
            and advs[1] < 0.0
            and advs[2] > 0.0
            and ranking_ok
            and margin >= gmin
            and stab["stable"]
        )
    return {
        "passed": passed,
        "ranking_ok": ranking_ok,
        "robust_ok": passed,
        "required": True,
        "phase": "plasticity",
        "adv": advs,
        "winner": win,
        "want": h2,
        "margin": margin,
        "min_probe_margin": margin,
        "perturb_stable": bool(stab["stable"]),
        "stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_match_sanity_used_for_pass": False,
        "n_live_teaches": int(n_live),
        "n_live_reversal_updates": int(n_live_reversal),
        "matched_live_reversal": arm in MATCHED_LIVE_ARMS,
        "n_replay_updates": int(traj["n_replay_calls"]),
        "n_updates": int(getattr(learner, "n_updates", traj["n_actual_updates"])),
        "store": store.stats(),
        "matcher": store.ledger.summary(),
        "m3_ceiling_only": arm == "M3",
        "n_cues": 2,
        "n_probe": 1,
        "margin_trajectory": traj,
        "replay_learner": str(p["arms"][arm]["replay_learner"]),
        "store_policy": str(p["arms"][arm]["policy"]),
    }


def eval_specificity(arm: str, world: dict[str, Any], pairs: list[tuple[str, str]], *, tag: str) -> dict[str, Any]:
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    handles = list(world["handles"])
    learner = make_learner(arm, handles, p)
    store = make_store(arm, p)
    captured: list[dict[str, Any]] = []
    n_live = 0
    n_live_reversal = 0
    cue0, h_old = pairs[0]
    h_new = [h for h in handles if h != h_old][0]
    want = {c: h for c, h in pairs}
    want[cue0] = h_new
    want_pairs = [(c, want[c]) for c, _h in pairs]
    epochs = int(p["arms"][arm]["replay_epochs"])
    with tempfile.TemporaryDirectory(prefix="lmm_spec_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        for i, (cue, handle) in enumerate(pairs):
            rec = teach_bridged(ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}")
            n_live += 1
            ingest(store, captured, rec["addr"], rec["handle"], rec["adv"], world, cue=cue, tag=f"{tag}_cap{i}")
        train_store(arm, learner, store, captured, handles, epochs, gmin, p)
        wneg = set_handle_delta(world, h_old, NEG_DELTA)
        tneg = teach_bridged(ag, wneg, cue0, h_old, arm="B3", tracer=None, bank=None, tag=f"{tag}_revn")
        i_neg = ingest(
            store,
            captured,
            tneg["addr"],
            tneg["handle"],
            tneg["adv"],
            world,
            cue=cue0,
            reversal=True,
            tag=f"{tag}_revn",
            live_learner=reversal_live_learner(arm, learner),
        )
        n_live_reversal += int(bool(i_neg.get("live_trained")))
        n_live += 1
        tpos = teach_bridged(ag, world, cue0, h_new, arm="B3", tracer=None, bank=None, tag=f"{tag}_revp")
        i_pos = ingest(
            store,
            captured,
            tpos["addr"],
            tpos["handle"],
            tpos["adv"],
            world,
            cue=cue0,
            reversal=True,
            tag=f"{tag}_revp",
            live_learner=reversal_live_learner(arm, learner),
        )
        n_live_reversal += int(bool(i_pos.get("live_trained")))
        n_live += 1
        store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
        traj = train_store(arm, learner, store, captured, handles, epochs, gmin, p)
        probed = probe_pairs(ag, world, want_pairs, learner, tag=f"{tag}_spec", gmin=gmin)
        unrelated_ok = all(q["ranking_ok"] for q in probed["probes"][1:]) if len(probed["probes"]) == 4 else False
        reversed_ok = bool(probed["probes"] and probed["probes"][0]["ranking_ok"])
        passed = bool(probed["robust_ok"] and unrelated_ok and reversed_ok)
        attach_probe_traj(traj, probed, None)
    return {
        **probed,
        "passed": passed,
        "phase": "specificity",
        "reversed_ok": reversed_ok,
        "unrelated_ok": unrelated_ok,
        "n_live_teaches": int(n_live),
        "n_live_reversal_updates": int(n_live_reversal),
        "matched_live_reversal": arm in MATCHED_LIVE_ARMS,
        "stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_match_sanity_used_for_pass": False,
        "n_replay_updates": int(traj["n_replay_calls"]),
        "n_updates": int(getattr(learner, "n_updates", traj["n_actual_updates"])),
        "store": store.stats(),
        "matcher": store.ledger.summary(),
        "m3_ceiling_only": arm == "M3",
        "want_reversed": h_new,
        "want_kept": [h for _c, h in pairs[1:]],
        "margin_trajectory": traj,
        "replay_learner": str(p["arms"][arm]["replay_learner"]),
        "store_policy": str(p["arms"][arm]["policy"]),
    }


def decorate(
    out: dict[str, Any],
    *,
    kind: str,
    arm: str,
    n_cues: int,
    order: str,
    world: int,
    domain: str,
) -> dict[str, Any]:
    out.update(
        {
            "id": cell_id(kind, arm, n_cues, order, world),
            "kind": kind,
            "arm": arm,
            "n_cues": n_cues,
            "order": order,
            "world": world,
            "domain": domain,
            "required": True,
            "m3_ceiling_only": arm == "M3",
        }
    )
    return out


def _phase_flags(cells: list[dict[str, Any]], arm: str) -> dict[str, bool]:
    def rows(kind: str, *, n_cues: int | None = None) -> list[dict[str, Any]]:
        out = [c for c in cells if c["arm"] == arm and c["kind"] == kind]
        if n_cues is not None:
            out = [c for c in out if int(c["n_cues"]) == int(n_cues)]
        return out

    def ok(xs: list[dict[str, Any]]) -> bool:
        return bool(xs) and all(bool(c["passed"]) for c in xs)

    return {
        "acquire_all": ok(rows("acquire")),
        "acquire8": ok(rows("acquire", n_cues=8)),
        "twin": ok(rows("twin")),
        "stable": ok(rows("stable")),
        "plasticity": ok(rows("eco")),
        "specificity": ok(rows("spec")),
    }


def _four(flags: dict[str, bool]) -> bool:
    return bool(
        flags["acquire_all"]
        and flags["twin"]
        and flags["stable"]
        and flags["plasticity"]
        and flags["specificity"]
    )


def _m1_eight(cells: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [c for c in cells if c["arm"] == "M1" and c["kind"] == kind and int(c["n_cues"]) == 8]


def _m1_pre_rest_reaches(cells: list[dict[str, Any]]) -> bool:
    rows = _m1_eight(cells, "acquire")
    return bool(rows) and all(float(c.get("min_probe_margin") or 0.0) >= GMIN for c in rows)


def _m1_post_rest_below(cells: list[dict[str, Any]]) -> bool:
    rows = _m1_eight(cells, "stable")
    return bool(rows) and any(float(c.get("min_probe_margin") or 0.0) < GMIN for c in rows)


def _m1_never_reaches(cells: list[dict[str, Any]]) -> bool:
    rows = _m1_eight(cells, "acquire") + _m1_eight(cells, "stable")
    return bool(rows) and all(float(c.get("min_probe_margin") or 0.0) < GMIN for c in rows)


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    flags = {arm: _phase_flags(cells, arm) for arm in ARMS}
    extra = {
        **flags,
        "m1_pre_rest_reaches_0.01": _m1_pre_rest_reaches(cells),
        "m1_post_rest_below_0.01": _m1_post_rest_below(cells),
        "m1_never_reaches_0.01": _m1_never_reaches(cells),
        "ecological_match_not_first_failure": True,
    }
    ladder = p["decision_ladder"]
    if _four(flags["M1"]) and (not flags["M0"]["stable"]) and flags["M2"]["plasticity"]:
        return ladder[0]["id"], ladder[0]["then"], extra
    if flags["M1"]["stable"] and flags["M1"]["plasticity"] and (not flags["M2"]["plasticity"]):
        return ladder[1]["id"], ladder[1]["then"], extra
    if extra["m1_pre_rest_reaches_0.01"] and extra["m1_post_rest_below_0.01"]:
        return ladder[2]["id"], ladder[2]["then"], extra
    if extra["m1_never_reaches_0.01"] and (not _four(flags["M3"])):
        return ladder[3]["id"], ladder[3]["then"], extra
    if _four(flags["M3"]) and (not any(_four(flags[a]) for a in ("M0", "M1", "M2"))):
        return ladder[4]["id"], ladder[4]["then"], extra
    return ladder[5]["id"], ladder[5]["then"], extra


def expected_cell_ids() -> list[str]:
    ids: list[str] = []
    for arm in ARMS:
        for n in (2, 4, 8):
            for wi in range(2):
                for order in TEACH_ORDERS:
                    ids.append(cell_id("acquire", arm, n, order, wi))
                    ids.append(cell_id("stable", arm, n, order, wi))
        for order in TEACH_ORDERS:
            ids.append(cell_id("twin", arm, 2, order, 1))
        ids.append(cell_id("eco", arm, 2, "A_then_B", 0))
        ids.append(cell_id("spec", arm, 4, "A_then_B", 0))
    return ids


def cell_manifest_hash(cells: list[dict[str, Any]]) -> str:
    rows = []
    for c in cells:
        traj = c.get("margin_trajectory") or {}
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "passed": c["passed"],
                "ranking_ok": c.get("ranking_ok"),
                "min_probe_margin": c.get("min_probe_margin"),
                "first_all_correct_call": traj.get("first_all_correct_call"),
                "pre_rest_min_probe_margin": traj.get("pre_rest_min_probe_margin"),
                "post_rest_min_probe_margin": traj.get("post_rest_min_probe_margin"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_cell_coverage(cells: list[dict[str, Any]]) -> str:
    ids = [c["id"] for c in cells]
    expected = expected_cell_ids()
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    if set(ids) != set(expected):
        raise RuntimeError("cell IDs do not match frozen LIFECYCLEMARGINMAP grid")
    kinds = Counter(c["kind"] for c in cells)
    if dict(kinds) != {
        "acquire": EXPECTED_N_ACQUIRE,
        "stable": EXPECTED_N_STABLE,
        "twin": EXPECTED_N_TWIN,
        "eco": EXPECTED_N_ECO,
        "spec": EXPECTED_N_SPEC,
    }:
        raise RuntimeError(f"kind counts {dict(kinds)}")
    for c in cells:
        if c["arm"] == "M3" and not c.get("m3_ceiling_only", False):
            raise RuntimeError("M3 must remain ceiling-only")
        if SCORE_DOMAIN in str(c.get("domain") or ""):
            raise RuntimeError("SCORE identifier appeared in DEV payload")
        store = c.get("store")
        if store is not None and int(store.get("n_p1_scalars") or 0) > MAX_STATE_SCALARS:
            raise RuntimeError("episode store exceeded 512 state scalars")
        if c.get("margin_trajectory") is None:
            raise RuntimeError(f"missing margin trajectory {c['id']}")
        if c.get("bounded_match_sanity_used_for_pass"):
            raise RuntimeError(f"bounded match used as stability gate {c['id']}")
    return cell_manifest_hash(cells)


def _append_arm(cells: list[dict[str, Any]], arm: str) -> None:
    p = load_prereg()
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
                raise RuntimeError("SCORE identifier appeared in DEV payload")
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                acq, stab = eval_phased_map(
                    arm=arm,
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"lmm_{arm}_{wi}_{n_cues}_{order}",
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
        acq, _stab = eval_phased_map(
            arm=arm, world=world_t, pairs=pairs_t, order=order, tag=f"lmm_{arm}_twin_{order}"
        )
        cells.append(decorate(acq, kind="twin", arm=arm, n_cues=2, order=order, world=1, domain=world_t["domain"]))
    world_c = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    eco = eval_ecological(arm, world_c, tag=f"lmm_{arm}_eco")
    cells.append(decorate(eco, kind="eco", arm=arm, n_cues=2, order="A_then_B", world=0, domain=world_c["domain"]))
    world_s = capacity_world(0, DEV_DOMAIN, n_cues=4, n_handles=2)
    pairs_s = mapping_pairs(world_s, flip=False)
    spec = eval_specificity(arm, world_s, pairs_s, tag=f"lmm_{arm}_spec")
    cells.append(decorate(spec, kind="spec", arm=arm, n_cues=4, order="A_then_B", world=0, domain=world_s["domain"]))


def run_dev() -> dict[str, Any]:
    refuse_v31()
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        _append_arm(cells, arm)
    for c in cells:
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        if c.get("stability_gate") != "ranking_perturb_sigma_0.01":
            raise RuntimeError(f"stability gate missing {c['id']}")
        if c["arm"] == "M3" and int(c.get("n_live_reversal_updates") or 0) != 0:
            raise RuntimeError(f"M3 live reversal {c['id']}")
        if c["arm"] in MATCHED_LIVE_ARMS and c["kind"] in ("eco", "spec"):
            if int(c.get("n_live_reversal_updates") or 0) != 2:
                raise RuntimeError(f"matched live reversal missing {c['id']}")
    live_by_kind: dict[str, set[int]] = {}
    for c in cells:
        if c["kind"] in ("eco", "spec") and c["arm"] in MATCHED_LIVE_ARMS:
            live_by_kind.setdefault(str(c["kind"]), set()).add(int(c.get("n_live_reversal_updates") or 0))
    for kind, vals in live_by_kind.items():
        if len(vals) != 1 or 0 in vals:
            raise RuntimeError(f"unmatched M0-M2 live reversal on {kind}: {sorted(vals)}")
    manifest = assert_cell_coverage(cells)
    code, then, extra = _decision(cells, p)
    out = {
        "version": "TM.0.24.LIFECYCLEMARGINMAP.DEV",
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
        "m3_ceiling_only": True,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "phase_flags": extra,
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells),
        "manifest_sha": manifest,
        "cells": cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": lifecyclemarginmap_shas(),
        "note": "LIFECYCLEMARGINMAP DEV only. R2 freeze preserved. Write-geometry closed. M3 ceiling-only. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def smoke() -> dict[str, Any]:
    p = load_prereg()
    world = capacity_world(0, "TM024.LIFECYCLEMARGINMAP.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    acq, stab = eval_phased_map(arm="M0", world=world, pairs=pairs, order="A_then_B", tag="lmmsmk")
    eco = eval_ecological("M0", world, tag="lmmsmk_eco")
    eco_traj = eco.get("margin_trajectory") or {}
    ids = expected_cell_ids()
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "w1_resurrected": False,
        "act_score_mode": "query",
        "m0_acquire_n_probe": acq["n_probe"],
        "m0_stable_n_probe": stab["n_probe"],
        "m0_traj_keys": sorted((acq.get("margin_trajectory") or {}).keys()),
        "stability_gate": acq.get("stability_gate"),
        "expected_id_count": len(ids),
        "m3_ceiling_only": True,
        "replay_epochs": int(p["arms"]["M1"]["replay_epochs"]),
        "match_radius": float(p["match"]["radius"]),
        "eco_post_rest_min_probe_margin": eco_traj.get("post_rest_min_probe_margin"),
        "eco_rest_delta": eco_traj.get("rest_delta"),
        "eco_rest_reduced_margin": eco_traj.get("rest_reduced_margin"),
    }


def expected_ids_sha(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        raise RuntimeError("lifecyclemarginmap manifest already exists")
    refuse_v31()
    ids = expected_cell_ids()
    shas = {k: v for k, v in lifecyclemarginmap_shas().items() if k != "manifest"}
    out = {
        "version": "TM.0.24.LIFECYCLEMARGINMAP.MANIFEST",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "lab": "TM.0.24.LIFECYCLEMARGINMAP",
        "expected_n_cells": EXPECTED_N_CELLS,
        "expected_kind_counts": {
            "acquire": EXPECTED_N_ACQUIRE,
            "stable": EXPECTED_N_STABLE,
            "twin": EXPECTED_N_TWIN,
            "eco": EXPECTED_N_ECO,
            "spec": EXPECTED_N_SPEC,
        },
        "id_format": "{kind}|{arm}|c{n_cues}|{order}|w{world}",
        "expected_cell_ids": ids,
        "expected_ids_sha": expected_ids_sha(ids),
        "domains": {"DEV": DEV_DOMAIN, "TWIN": TWIN_DOMAIN, "SCORE": SCORE_DOMAIN},
        "matched_live_reversal_arms": list(MATCHED_LIVE_ARMS),
        "m3_no_live_reversal": True,
        "lifecycle_stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_cannot_satisfy_stability_gate": True,
        "historical_r2_decision_sha": R2_DEC_SHA,
        "shas": shas,
        "n": 64,
        "note": "LIFECYCLEMARGINMAP cell-ID manifest. R2 freeze preserved. Product remains 0.0.004.",
    }
    MANIFEST.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("lifecyclemarginmap runner.lock already exists")
    refuse_v31()
    assert_historical_frozen()
    if not MANIFEST.exists():
        write_manifest()
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.LIFECYCLEMARGINMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "shas": lifecyclemarginmap_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "arms": list(ARMS),
        "matched_live_reversal_arms": list(MATCHED_LIVE_ARMS),
        "m3_no_live_reversal": True,
        "match_radius": float(prereg["match"]["radius"]),
        "replay_epochs": 16,
        "no_new_eta_grid": True,
        "lifecycle_stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_cannot_satisfy_stability_gate": True,
        "n_slots": N_SLOTS,
        "max_state_scalars": MAX_STATE_SCALARS,
        "m3_ceiling_only": True,
        "trace_budget_unopened": 512,
        "declared_budget_remains_closed": 1536,
        "expected_n_cells": EXPECTED_N_CELLS,
        "historical_r2_decision_sha": R2_DEC_SHA,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen LIFECYCLEMARGINMAP runner. R2 locks preserved. DEV lock only after this file is on origin/main. No neural edit.",
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
        raise RuntimeError("lifecyclemarginmap decision lock already exists")
    out = {
        "version": "TM.0.24.LIFECYCLEMARGINMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "scored_worlds": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "candidate_v31": False,
        "candidate_v32": False,
        "w1_resurrected": False,
        "lineage_reopened": False,
        "eligibility_budget_installed": False,
        "trace_rows_installed": False,
        "declared_budget_remains_closed": 1536,
        "write_geometry_branch_closed": True,
        "m3_ceiling_only": True,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "phase_flags": dev.get("phase_flags"),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": "LIFECYCLEMARGINMAP runner-only margin trigger on frozen L2 lifecycle. R2 preserved. No v31. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.LIFECYCLEMARGINMAP DEV\n\n"
        f"Decision: **{out['decision']['code']}**.\n\n"
        f"Phase flags: `{out['decision']['phase_flags']}`.\n\n"
        "Write-geometry closed. SCORE unopened. No neural candidate. "
        "512/1536 budgets stay closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze installs a sufficient write rule")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("LIFECYCLEMARGINMAP DEV lock requires runner.lock on origin/main after this freeze")
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
        assert p["neural_edit"] is False
        assert p["act_score_mode"] == "query"
        assert p["match"]["radius"] == 0.05
        assert p["no_new_eta_grid"] is True
        assert p["arms"]["M0"]["replay_learner"] == "error_only"
        assert p["arms"]["M1"]["replay_learner"] == "c3_passive_aggressive"
        assert p["arms"]["M2"]["policy"] == "retain_stale"
        assert p["arms"]["M3"]["ceiling_only"] is True
        assert p["arms"]["M1"]["replay_epochs"] == 16
        assert p["expected_n_cells"] == EXPECTED_N_CELLS
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
