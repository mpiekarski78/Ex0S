"""TM.0.24.MEMORYLIFECYCLEMAP.R2 — matched live reversal, dual match perturbation.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
V1 runner.lock on ec317ba is immutable. Write-geometry closed. W1 not resurrected.
DEV on unused TM024.MEMORYLIFECYCLEMAP.R2.DEV. after this freeze is on origin/main.
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
from typing import Any, Callable

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024convergencemap import ErrorOnlyBank, unique_winner
from experiments.run_tm024eligmap import _fresh, l2, record_rest, unit_or_zero
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024tracebridge import probe_address, require_query, teach_bridged
from experiments.run_tm024writegeom import (
    NEG_DELTA,
    SequentialRLS,
    capacity_world,
    domain_seed,
    mapping_pairs,
    ranking_margin,
    set_handle_delta,
)
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.runner.lock"
MANIFEST = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.manifest.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.r2.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024memorylifecyclemap_r2_results.md"
ADDENDUM = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.addendum.lock"
HIST_PREREG = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.prereg.lock"
HIST_CONTRACT = REPO_ROOT / "docs" / "lineage_memorylifecyclemap_contract.md"
HIST_ISOLATION = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.isolation.lock"
HIST_RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_memorylifecyclemap.runner.lock"
HIST_RUNNER_PY = REPO_ROOT / "experiments" / "run_tm024memorylifecyclemap.py"
CVG_DEC = REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock"
CVG_ADD = REPO_ROOT / "docs" / "lineage_convergencemap.decision.addendum.lock"
CVG_DEV = REPO_ROOT / "docs" / "lineage_convergencemap.dev.lock"
CVG_RUNNER = REPO_ROOT / "experiments" / "run_tm024convergencemap.py"
TB_RUNNER = REPO_ROOT / "experiments" / "run_tm024tracebridge.py"
WG_RUNNER = REPO_ROOT / "experiments" / "run_tm024writegeom.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.MEMORYLIFECYCLEMAP.R2.DEV."
TWIN_DOMAIN = "TM024.MEMORYLIFECYCLEMAP.R2.TWIN."
SCORE_DOMAIN = "TM024.MEMORYLIFECYCLEMAP.R2.SCORE."
SCORE_MARKERS = (
    "TM024.MEMORYLIFECYCLEMAP.R2.SCORE.",
    "TM024.MEMORYLIFECYCLEMAP.SCORE.",
)
MATCHED_LIVE_ARMS = ("L1", "L2", "L3")
V1_RUNNER_LOCK_SHA = "28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593"
V1_RUNNER_PY_SHA = "edec8809938f3f1ab77948feb3661bea4fc3e6bb1abf573a81089ae628dfc974"
EPS = 1e-12
ARMS = ("L0", "L1", "L2", "L3", "L4")
KINDS = ("acquire", "stable", "twin", "eco", "spec")
ALLOWED_EPISODE_FIELDS = ("p1", "handle", "adv", "age", "version", "valid")
FORBIDDEN_EPISODE_TOKENS = ("cue", "symbol", "spelling", "identity", "answer")
EXPECTED_N_ACQUIRE = 60
EXPECTED_N_STABLE = 60
EXPECTED_N_TWIN = 10
EXPECTED_N_ECO = 5
EXPECTED_N_SPEC = 5
EXPECTED_N_CELLS = EXPECTED_N_ACQUIRE + EXPECTED_N_STABLE + EXPECTED_N_TWIN + EXPECTED_N_ECO + EXPECTED_N_SPEC
N_SLOTS = 8
ROW_DIM = 64
MAX_STATE_SCALARS = N_SLOTS * ROW_DIM


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def memorylifecyclemap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "addendum": ADDENDUM,
        "manifest": MANIFEST,
        "candidate_v30": CANDIDATE_V30,
        "historical_runner_lock": HIST_RUNNER_LOCK,
        "historical_runner_py": HIST_RUNNER_PY,
        "historical_prereg": HIST_PREREG,
        "historical_contract": HIST_CONTRACT,
        "historical_isolation": HIST_ISOLATION,
        "convergencemap_decision": CVG_DEC,
        "convergencemap_addendum": CVG_ADD,
        "convergencemap_dev": CVG_DEV,
        "convergencemap_runner": CVG_RUNNER,
        "tracebridge_runner": TB_RUNNER,
        "writegeom_runner": WG_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def bounded_match_sigma(radius: float) -> float:
    return float(min(0.01, float(radius) / (2.0 * float(np.sqrt(ROW_DIM)))))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, arm: str, n_cues: int, order: str, world: int) -> str:
    if kind not in KINDS:
        raise RuntimeError(f"kind must be one of {KINDS}, got {kind}")
    if arm not in ARMS:
        raise RuntimeError(f"unknown arm {arm}")
    return f"{kind}|{arm}|c{n_cues}|{order}|w{world}"


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no memorylifecyclemap r2 runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    shas = memorylifecyclemap_shas()
    if shas != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    if sha_file(HIST_RUNNER_LOCK) != V1_RUNNER_LOCK_SHA:
        raise RuntimeError("V1 runner.lock must remain the published freeze")
    if sha_file(HIST_RUNNER_PY) != V1_RUNNER_PY_SHA:
        raise RuntimeError("V1 runner.py must remain the published freeze")
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


def desired_winner(handles: list[str], chosen: str, adv: float) -> str | None:
    if float(adv) > 0.0:
        return chosen
    others = [h for h in handles if h != chosen]
    return others[0] if len(others) == 1 else None


def reversal_live_learner(arm: str, learner: Any) -> Any | None:
    """L1–L3 share one live reversal update; L0/L4 do not."""
    if arm in MATCHED_LIVE_ARMS:
        return learner
    return None


def assert_episode_legal(ep: dict[str, Any], world: dict[str, Any] | None = None) -> None:
    extra = set(ep) - set(ALLOWED_EPISODE_FIELDS)
    if extra:
        raise RuntimeError(f"illegal episode fields {sorted(extra)}")
    for k in ep:
        lk = str(k).lower()
        for bad in FORBIDDEN_EPISODE_TOKENS:
            if bad in lk:
                raise RuntimeError(f"cue-like episode key {k}")
    if world is not None:
        cues = {str(c) for c in (world.get("cues") or [])}
        cues |= {str(m["cue"]) for m in (world.get("cue_handle") or [])}
        if str(ep.get("handle")) in cues:
            raise RuntimeError("episode handle collides with cue string")


def _rate(num: int, den: int) -> float | None:
    if int(den) <= 0:
        return None
    return float(num) / float(den)


class MatchLedger:
    """Runner-only matcher diagnostics. Origin cues never enter episodes."""

    def __init__(self) -> None:
        self.n_writes = 0
        self.n_no_match = 0
        self.n_unique_match = 0
        self.n_multiple_match = 0
        self.n_same_cue_attempts = 0
        self.n_same_cue_hits = 0
        self.n_cross_cue_false = 0
        self.n_reversal_writes = 0
        self.n_reversal_hits = 0
        self.n_twin_queries = 0
        self.n_twin_ok = 0
        self.n_bounded_perturb = 0
        self.n_bounded_ok = 0
        self.n_bounded_batches = 0
        self.n_bounded_stable_batches = 0
        self.n_eco_perturb = 0
        self.n_eco_ok = 0
        self.n_eco_batches = 0
        self.n_eco_stable_batches = 0
        self.bounded_match_sigma = None
        self.ecological_match_sigma = None

    def summary(self) -> dict[str, Any]:
        rev_miss = bool(self.n_reversal_writes > 0 and self.n_reversal_hits < self.n_reversal_writes)
        eco_rate = _rate(self.n_eco_ok, self.n_eco_perturb)
        eco_batch = _rate(self.n_eco_stable_batches, self.n_eco_batches)
        eco_fail = bool(self.n_eco_batches > 0 and self.n_eco_stable_batches < self.n_eco_batches)
        return {
            "same_cue_recall": _rate(self.n_same_cue_hits, self.n_same_cue_attempts),
            "cross_cue_false_match_rate": _rate(self.n_cross_cue_false, self.n_writes),
            "reversal_match_recall": _rate(self.n_reversal_hits, self.n_reversal_writes),
            "reversal_matcher_miss": rev_miss,
            "twin_match_stable": _rate(self.n_twin_ok, self.n_twin_queries),
            "bounded_match_sanity": _rate(self.n_bounded_ok, self.n_bounded_perturb),
            "bounded_match_batch_stable": _rate(self.n_bounded_stable_batches, self.n_bounded_batches),
            "bounded_match_sigma": self.bounded_match_sigma,
            "ecological_match_stability": eco_rate,
            "ecological_match_batch_stable": eco_batch,
            "ecological_match_sigma": self.ecological_match_sigma,
            "ecological_match_failure": eco_fail,
            "n_no_match": int(self.n_no_match),
            "n_unique_match": int(self.n_unique_match),
            "n_multiple_match": int(self.n_multiple_match),
            "n_writes": int(self.n_writes),
            "n_reversal_writes": int(self.n_reversal_writes),
            "n_reversal_hits": int(self.n_reversal_hits),
        }


class EpisodeStore:
    """Fixed eight-slot P1 store. Runner-only. No cue string."""

    def __init__(self, *, policy: str, n_slots: int = N_SLOTS, match_l2: float = 0.05):
        if policy not in ("fifo", "replace", "retain_stale"):
            raise RuntimeError(f"unknown store policy {policy}")
        self.policy = policy
        self.n_slots = int(n_slots)
        self.match_l2 = float(match_l2)
        self.slots: list[dict[str, Any]] = []
        self.origin: list[str | None] = []
        self.clock = 0
        self.n_inserts = 0
        self.n_replaced = 0
        self.n_invalidated = 0
        self.n_evicted = 0
        self.n_matches = 0
        self.n_refused = 0
        self.n_refreshed = 0
        self.ledger = MatchLedger()

    def _episode(self, p1: np.ndarray, handle: str, adv: float, *, version: int = 1) -> dict[str, Any]:
        ep = {
            "p1": np.asarray(p1, dtype=np.float64).reshape(-1).copy(),
            "handle": str(handle),
            "adv": float(adv),
            "age": int(self.clock),
            "version": int(version),
            "valid": True,
        }
        if ep["p1"].shape != (ROW_DIM,):
            raise RuntimeError(f"episode p1 must be {ROW_DIM}-d")
        return ep

    def query(self, p1: np.ndarray) -> dict[str, Any]:
        x = unit_or_zero(p1)
        ranked: list[tuple[float, int, int]] = []
        for i, ep in enumerate(self.slots):
            if not ep["valid"]:
                continue
            d = l2(ep["p1"], x)
            ranked.append((float(d), int(ep["age"]), int(i)))
        ranked.sort()
        n_within = sum(1 for d, _a, _i in ranked if d <= self.match_l2 + EPS)
        nearest_i = ranked[0][2] if ranked else None
        nearest_d = ranked[0][0] if ranked else None
        match_i = nearest_i if nearest_d is not None and nearest_d <= self.match_l2 + EPS else None
        n_ties = 0
        if match_i is not None and nearest_d is not None:
            n_ties = sum(
                1 for d, _a, _i in ranked if d <= self.match_l2 + EPS and abs(d - nearest_d) <= EPS
            )
        if n_within == 0:
            kind = "no_match"
        elif n_within == 1:
            kind = "unique_match"
        else:
            kind = "multiple_match"
        return {
            "kind": kind,
            "match_i": match_i,
            "nearest_i": nearest_i,
            "nearest_d": nearest_d,
            "n_within": int(n_within),
            "n_ties": int(n_ties),
        }

    def nearest(self, p1: np.ndarray) -> int | None:
        return self.query(p1)["match_i"]

    @staticmethod
    def contradictory(ep: dict[str, Any], handle: str, adv: float) -> bool:
        if not ep["valid"]:
            return False
        old_pos = float(ep["adv"]) > 0.0
        new_pos = float(adv) > 0.0
        if old_pos != new_pos and abs(float(ep["adv"])) > EPS and abs(float(adv)) > EPS:
            return True
        if str(handle) != str(ep["handle"]) and float(adv) > 0.0:
            return True
        return False

    def _record_query(
        self,
        q: dict[str, Any],
        *,
        cue: str | None,
        reversal: bool,
    ) -> None:
        led = self.ledger
        led.n_writes += 1
        kind = str(q["kind"])
        if kind == "no_match":
            led.n_no_match += 1
        elif kind == "unique_match":
            led.n_unique_match += 1
        else:
            led.n_multiple_match += 1
        match_i = q["match_i"]
        if cue is not None:
            same_slots = [i for i, c in enumerate(self.origin) if c == cue and self.slots[i]["valid"]]
            if same_slots:
                led.n_same_cue_attempts += 1
                if match_i is not None and self.origin[int(match_i)] == cue:
                    led.n_same_cue_hits += 1
            if match_i is not None and self.origin[int(match_i)] not in (None, cue):
                led.n_cross_cue_false += 1
        if reversal:
            led.n_reversal_writes += 1
            if match_i is not None:
                led.n_reversal_hits += 1

    def _perturb_match(self, p1: np.ndarray, slot: int, *, domain: str, tag: str, mode: str) -> None:
        p = load_prereg()
        m = p["margin"]
        n = int(m["perturb_n"])
        need = int(m["perturb_stable_min"])
        if mode == "bounded":
            sigma = bounded_match_sigma(self.match_l2)
            self.ledger.bounded_match_sigma = float(sigma)
            n_attr, ok_attr, b_attr, sb_attr = (
                "n_bounded_perturb",
                "n_bounded_ok",
                "n_bounded_batches",
                "n_bounded_stable_batches",
            )
        elif mode == "ecological":
            sigma = float(m["rho_perturb_sigma"])
            self.ledger.ecological_match_sigma = float(sigma)
            n_attr, ok_attr, b_attr, sb_attr = (
                "n_eco_perturb",
                "n_eco_ok",
                "n_eco_batches",
                "n_eco_stable_batches",
            )
        else:
            raise RuntimeError(f"unknown match perturbation mode {mode}")
        rng = np.random.default_rng(domain_seed(domain, f"{tag}_{mode}_{self.clock}_{slot}"))
        r0 = unit_or_zero(p1)
        n_ok = 0
        for _i in range(n):
            unit = unit_or_zero(r0 + rng.normal(0.0, sigma, size=r0.shape))
            q = self.query(unit)
            setattr(self.ledger, n_attr, getattr(self.ledger, n_attr) + 1)
            if q["kind"] == "unique_match" and q["match_i"] == slot:
                n_ok += 1
                setattr(self.ledger, ok_attr, getattr(self.ledger, ok_attr) + 1)
        setattr(self.ledger, b_attr, getattr(self.ledger, b_attr) + 1)
        if n_ok >= need:
            setattr(self.ledger, sb_attr, getattr(self.ledger, sb_attr) + 1)

    def score_twin(self) -> None:
        origins = [c for c in self.origin if c is not None]
        if len(set(origins)) < 2:
            return
        for i, ep in enumerate(self.slots):
            if not ep["valid"]:
                continue
            self.ledger.n_twin_queries += 1
            q = self.query(ep["p1"])
            if q["kind"] == "unique_match" and q["match_i"] == i:
                self.ledger.n_twin_ok += 1

    def score_perturbation(self, *, domain: str, tag: str) -> None:
        for i, ep in enumerate(self.slots):
            if not ep["valid"]:
                continue
            self._perturb_match(ep["p1"], i, domain=domain, tag=tag, mode="bounded")
            self._perturb_match(ep["p1"], i, domain=domain, tag=tag, mode="ecological")

    def _insert(self, ep: dict[str, Any], *, origin: str | None) -> None:
        if len(self.slots) < self.n_slots:
            self.slots.append(ep)
            self.origin.append(origin)
            self.n_inserts += 1
            return
        evict = min(range(len(self.slots)), key=lambda i: (int(self.slots[i]["age"]), i))
        self.slots[evict] = ep
        self.origin[evict] = origin
        self.n_evicted += 1
        self.n_inserts += 1

    def write(
        self,
        p1: np.ndarray,
        handle: str,
        adv: float,
        *,
        world: dict[str, Any] | None = None,
        cue: str | None = None,
        reversal: bool = False,
        domain: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        self.clock += 1
        x = unit_or_zero(p1)
        ep = self._episode(x, handle, adv)
        assert_episode_legal(ep, world)
        q = self.query(x)
        self._record_query(q, cue=cue, reversal=reversal)
        if self.policy == "fifo":
            self._insert(ep, origin=cue)
            return {**q, "action": "insert"}
        match_i = q["match_i"]
        if match_i is None:
            self._insert(ep, origin=cue)
            return {**q, "action": "insert"}
        self.n_matches += 1
        old = self.slots[int(match_i)]
        if self.contradictory(old, handle, adv):
            if self.policy == "replace":
                ep["version"] = int(old["version"]) + 1
                self.slots[int(match_i)] = ep
                self.origin[int(match_i)] = cue
                self.n_replaced += 1
                self.n_invalidated += 1
                return {**q, "action": "replace"}
            self.n_refused += 1
            return {**q, "action": "refuse"}
        old["age"] = int(self.clock)
        self.n_refreshed += 1
        return {**q, "action": "refresh"}

    def valid_rows(self) -> list[dict[str, Any]]:
        return [e for e in self.slots if e["valid"]]

    def stats(self) -> dict[str, Any]:
        n_valid = len(self.valid_rows())
        n_scalars = int(n_valid * ROW_DIM)
        if n_scalars > MAX_STATE_SCALARS:
            raise RuntimeError("episode store exceeded 512 state scalars")
        for ep in self.slots:
            assert_episode_legal(ep)
        return {
            "policy": self.policy,
            "n_slots": int(self.n_slots),
            "n_occupied": len(self.slots),
            "n_valid": n_valid,
            "n_p1_scalars": n_scalars,
            "n_inserts": int(self.n_inserts),
            "n_replaced": int(self.n_replaced),
            "n_invalidated": int(self.n_invalidated),
            "n_evicted": int(self.n_evicted),
            "n_matches": int(self.n_matches),
            "n_refused": int(self.n_refused),
            "n_refreshed": int(self.n_refreshed),
        }

    def public_rows(self) -> list[dict[str, Any]]:
        rows = []
        for e in self.slots:
            pub = {k: e[k] for k in ("handle", "adv", "age", "version", "valid")}
            assert_episode_legal({**pub, "p1": e["p1"]})
            rows.append(pub)
        return rows


def episode_core(ep: dict[str, Any]) -> tuple[Any, ...]:
    """Store payload excluding age/clock. Used to prove L3 refuse does not mutate rows."""
    return (
        tuple(np.asarray(ep["p1"], dtype=np.float64).reshape(-1).tolist()),
        str(ep["handle"]),
        float(ep["adv"]),
        int(ep["version"]),
        bool(ep["valid"]),
    )


def make_learner(arm: str, handles: list[str], p: dict[str, Any]) -> Any:
    if arm == "L4":
        return SequentialRLS(64, handles, lam=float(p["arms"]["L4"]["lambda"]))
    a = p["learner"]
    return ErrorOnlyBank(handles, eta=float(a["eta"]), c_max=float(a["c_max"]))


def make_store(arm: str, p: dict[str, Any]) -> EpisodeStore | None:
    spec = p["arms"][arm]
    if not spec.get("store"):
        return None
    return EpisodeStore(policy=str(spec["policy"]), n_slots=N_SLOTS, match_l2=float(p["match"]["radius"]))


def perturb_rank(
    score_fn: Callable[[np.ndarray], dict[str, float]],
    addr: np.ndarray,
    winner: str,
    *,
    domain: str,
    key: str,
) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = unit_or_zero(addr)
    n_ok = 0
    for _i in range(n):
        noise = rng.normal(0.0, sigma, size=r0.shape)
        unit = unit_or_zero(r0 + noise)
        scores = score_fn(unit)
        if not scores:
            continue
        ranked = unique_winner(scores)
        if ranked == winner:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def live_probe(
    ag: NeuralCortex,
    world: dict[str, Any],
    cue: str,
    learner: Any,
    *,
    tag: str,
) -> dict[str, Any]:
    n0 = int(getattr(learner, "n_updates", 0))
    w0 = ag.W_act_query.detach().clone()
    w_rls = np.asarray(learner.W).copy() if hasattr(learner, "W") else None
    probe = clone_frozen(ag)
    require_query(probe)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    addr = probe_address("B3", probe, None)
    scores = learner.scores(addr)
    if int(getattr(learner, "n_updates", 0)) != n0:
        raise RuntimeError("probe updated learner weights")
    if w_rls is not None and float(np.max(np.abs(np.asarray(learner.W) - w_rls))) > EPS:
        raise RuntimeError("probe updated learner weights")
    if float((ag.W_act_query - w0).abs().max().item()) > 1e-12:
        raise RuntimeError("probe updated organism state")
    win = unique_winner(scores)
    margin = ranking_margin(scores, win or "") if win else 0.0
    return {"winner": win, "addr": addr, "margin": float(margin)}


def checkpoint_error(learner: Any, handles: list[str], rows: list[dict[str, Any]]) -> int:
    n_err = 0
    for r in rows:
        want = desired_winner(handles, str(r["handle"]), float(r["adv"]))
        got = unique_winner(learner.scores(r["p1"]))
        if got != want:
            n_err += 1
    return n_err


def snapshot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        row = {
            "p1": np.asarray(r["p1"], dtype=np.float64).reshape(-1).copy(),
            "handle": str(r["handle"]),
            "adv": float(r["adv"]),
            "age": int(r.get("age") or 0),
            "version": int(r.get("version") or 1),
            "valid": True,
        }
        assert_episode_legal(row)
        out.append(row)
    return out


def replay_rows(learner: Any, handles: list[str], rows: list[dict[str, Any]], epochs: int) -> dict[str, int]:
    frozen = snapshot_rows(rows)
    n_replay = 0
    n_err = 0
    n_ck = 0
    for _cy in range(int(epochs)):
        for r in frozen:
            learner.update(r["p1"], r["handle"], r["adv"])
            n_replay += 1
            n_ck += 1
            n_err += checkpoint_error(learner, handles, frozen)
    return {"n_replay": n_replay, "n_checkpoint_errors": n_err, "n_checkpoints": n_ck}


def ingest(
    store: EpisodeStore | None,
    captured: list[dict[str, Any]],
    p1: np.ndarray,
    handle: str,
    adv: float,
    world: dict[str, Any],
    *,
    cue: str,
    reversal: bool = False,
    tag: str,
    live_learner: Any | None = None,
) -> dict[str, Any]:
    if live_learner is not None:
        live_learner.update(p1, handle, adv)
    live_trained = live_learner is not None
    row = {
        "p1": unit_or_zero(p1),
        "handle": str(handle),
        "adv": float(adv),
        "age": len(captured) + 1,
        "version": 1,
        "valid": True,
    }
    assert_episode_legal(row, world)
    if store is None:
        captured.append(row)
        return {"action": "capture", "kind": "no_store", "live_trained": live_trained}
    q = store.write(
        row["p1"],
        row["handle"],
        row["adv"],
        world=world,
        cue=cue,
        reversal=reversal,
        domain=world["domain"],
        tag=tag,
    )
    if q.get("action") != "refuse":
        captured.append(row)
    return {**q, "live_trained": live_trained}


def probe_pairs(
    ag: NeuralCortex,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    learner: Any,
    *,
    tag: str,
    gmin: float,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    ranking_ok = True
    robust_ok = True
    for i, (cue, handle) in enumerate(pairs):
        pr = live_probe(ag, world, cue, learner, tag=f"{tag}_p{i}")
        stab = perturb_rank(
            learner.scores,
            pr["addr"],
            pr["winner"] or "",
            domain=world["domain"],
            key=f"{tag}_{cue}",
        )
        rank = bool(pr["winner"] == handle)
        robust = bool(rank and pr["margin"] >= gmin and stab["stable"])
        ranking_ok = ranking_ok and rank
        robust_ok = robust_ok and robust
        probes.append(
            {
                "want": handle,
                "winner": pr["winner"],
                "margin": float(pr["margin"]),
                "perturb_stable": bool(stab["stable"]),
                "ranking_ok": rank,
                "robust_ok": robust,
            }
        )
    return {
        "ranking_ok": bool(ranking_ok and probes),
        "robust_ok": bool(robust_ok and probes),
        "perturb_stable": bool(all(q["perturb_stable"] for q in probes)) if probes else False,
        "min_probe_margin": float(min(q["margin"] for q in probes)) if probes else 0.0,
        "probes": probes,
        "n_probe": len(probes),
    }


def budgets(arm: str, p: dict[str, Any]) -> dict[str, int]:
    spec = p["arms"][arm]
    return {
        "live_cycles": int(spec.get("live_cycles") or 0),
        "replay_epochs": int(spec.get("replay_epochs") or 0),
        "live_capture_passes": int(spec.get("live_capture_passes") or 0),
    }


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
    bud = budgets(arm, p)
    captured: list[dict[str, Any]] = []
    n_live = 0
    n_checkpoint_errors = 0
    n_checkpoints = 0
    n_replay = 0
    with tempfile.TemporaryDirectory(prefix="mlm_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        if arm == "L0":
            last: dict[str, dict[str, Any]] = {}
            for cy in range(bud["live_cycles"]):
                for i, (cue, handle) in enumerate(seq):
                    rec = teach_bridged(
                        ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_c{cy}_t{i}"
                    )
                    n_live += 1
                    learner.update(rec["addr"], rec["handle"], rec["adv"])
                    last[cue] = {
                        "p1": unit_or_zero(rec["addr"]),
                        "handle": rec["handle"],
                        "adv": rec["adv"],
                        "age": n_live,
                        "version": 1,
                        "valid": True,
                    }
                    assert_episode_legal(last[cue], world)
                    rows = [last[c] for c, _h in seq if c in last]
                    n_checkpoints += 1
                    n_checkpoint_errors += checkpoint_error(learner, handles, rows)
        else:
            for _pass in range(max(1, bud["live_capture_passes"])):
                for i, (cue, handle) in enumerate(seq):
                    rec = teach_bridged(
                        ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}"
                    )
                    n_live += 1
                    ingest(
                        store,
                        captured,
                        rec["addr"],
                        rec["handle"],
                        rec["adv"],
                        world,
                        cue=cue,
                        tag=f"{tag}_cap{i}",
                    )
            if store is not None:
                if world.get("purpose") == "rename_twin":
                    store.score_twin()
                store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
            rows = store.valid_rows() if store is not None else captured
            rep = replay_rows(learner, handles, rows, bud["replay_epochs"])
            n_replay += int(rep["n_replay"])
            n_checkpoint_errors += int(rep["n_checkpoint_errors"])
            n_checkpoints += int(rep["n_checkpoints"])
        acquire = probe_pairs(ag, world, pairs, learner, tag=f"{tag}_acq", gmin=gmin)
        record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
        stable = probe_pairs(ag, world, pairs, learner, tag=f"{tag}_st", gmin=gmin)
    store_stats = store.stats() if store is not None else None
    shared = {
        "n_live_teaches": int(n_live),
        "n_replay_updates": int(n_replay),
        "n_checkpoint_errors": int(n_checkpoint_errors),
        "n_checkpoints": int(n_checkpoints),
        "monotonic_retention_ok": bool(n_checkpoint_errors == 0),
        "n_updates": int(getattr(learner, "n_updates", n_live + n_replay)),
        "store": store_stats,
        "matcher": store.ledger.summary() if store is not None else None,
        "l4_ceiling_only": arm == "L4",
        "n_cues": len(pairs),
        "n_live_reversal_updates": 0,
        "matched_live_reversal": arm in MATCHED_LIVE_ARMS,
        "stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_match_sanity_used_for_pass": False,
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
        "passed": bool(stable["robust_ok"]),
        "phase": "stability",
    }
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
    bud = budgets(arm, p)
    captured: list[dict[str, Any]] = []
    advs: list[float] = []
    n_live = 0
    n_replay = 0
    n_checkpoint_errors = 0
    n_live_reversal = 0
    with tempfile.TemporaryDirectory(prefix="mlm_eco_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)

        def teach(w: dict[str, Any], handle: str, suffix: str) -> dict[str, Any]:
            rec = teach_bridged(ag, w, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_{suffix}")
            return rec

        if arm == "L0":
            for cy in range(bud["live_cycles"]):
                t1 = teach(world, h1, f"c{cy}p")
                learner.update(t1["addr"], t1["handle"], t1["adv"])
                n_live += 1
                if cy == 0:
                    advs = [float(t1["adv"])]
                wneg = set_handle_delta(world, h1, NEG_DELTA)
                t2 = teach(wneg, h1, f"c{cy}n")
                learner.update(t2["addr"], t2["handle"], t2["adv"])
                n_live += 1
                if cy == 0:
                    advs.append(float(t2["adv"]))
                t3 = teach(world, h2, f"c{cy}r")
                learner.update(t3["addr"], t3["handle"], t3["adv"])
                n_live += 1
                if cy == 0:
                    advs.append(float(t3["adv"]))
        else:
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
            if store is not None:
                store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
            rows = store.valid_rows() if store is not None else captured
            rep = replay_rows(learner, handles, rows, bud["replay_epochs"])
            n_replay = int(rep["n_replay"])
            n_checkpoint_errors = int(rep["n_checkpoint_errors"])
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
        "perturb_stable": bool(stab["stable"]),
        "stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_match_sanity_used_for_pass": False,
        "n_live_teaches": int(n_live),
        "n_live_reversal_updates": int(n_live_reversal),
        "matched_live_reversal": arm in MATCHED_LIVE_ARMS,
        "n_replay_updates": int(n_replay),
        "n_checkpoint_errors": int(n_checkpoint_errors),
        "monotonic_retention_ok": bool(n_checkpoint_errors == 0),
        "n_updates": int(getattr(learner, "n_updates", n_live + n_replay)),
        "store": store.stats() if store is not None else None,
        "matcher": store.ledger.summary() if store is not None else None,
        "l4_ceiling_only": arm == "L4",
        "n_cues": 2,
        "n_probe": 1,
    }


def eval_specificity(arm: str, world: dict[str, Any], pairs: list[tuple[str, str]], *, tag: str) -> dict[str, Any]:
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    handles = list(world["handles"])
    learner = make_learner(arm, handles, p)
    store = make_store(arm, p)
    bud = budgets(arm, p)
    captured: list[dict[str, Any]] = []
    n_live = 0
    n_replay = 0
    n_checkpoint_errors = 0
    n_live_reversal = 0
    cue0, h_old = pairs[0]
    h_new = [h for h in handles if h != h_old][0]
    want = {c: h for c, h in pairs}
    want[cue0] = h_new
    want_pairs = [(c, want[c]) for c, _h in pairs]
    with tempfile.TemporaryDirectory(prefix="mlm_spec_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        if arm == "L0":
            for cy in range(bud["live_cycles"]):
                for i, (cue, handle) in enumerate(pairs):
                    rec = teach_bridged(
                        ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_c{cy}_t{i}"
                    )
                    n_live += 1
                    learner.update(rec["addr"], rec["handle"], rec["adv"])
            for cy in range(bud["live_cycles"]):
                wneg = set_handle_delta(world, h_old, NEG_DELTA)
                tneg = teach_bridged(
                    ag, wneg, cue0, h_old, arm="B3", tracer=None, bank=None, tag=f"{tag}_revn{cy}"
                )
                learner.update(tneg["addr"], tneg["handle"], tneg["adv"])
                n_live += 1
                tpos = teach_bridged(
                    ag, world, cue0, h_new, arm="B3", tracer=None, bank=None, tag=f"{tag}_revp{cy}"
                )
                learner.update(tpos["addr"], tpos["handle"], tpos["adv"])
                n_live += 1
        else:
            for i, (cue, handle) in enumerate(pairs):
                rec = teach_bridged(ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}")
                n_live += 1
                ingest(store, captured, rec["addr"], rec["handle"], rec["adv"], world, cue=cue, tag=f"{tag}_cap{i}")
            rows = store.valid_rows() if store is not None else captured
            rep = replay_rows(learner, handles, rows, bud["replay_epochs"])
            n_replay += int(rep["n_replay"])
            n_checkpoint_errors += int(rep["n_checkpoint_errors"])
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
            if store is not None:
                store.score_perturbation(domain=world["domain"], tag=f"{tag}_pert")
            rows = store.valid_rows() if store is not None else captured
            rep2 = replay_rows(learner, handles, rows, bud["replay_epochs"])
            n_replay += int(rep2["n_replay"])
            n_checkpoint_errors += int(rep2["n_checkpoint_errors"])
        probed = probe_pairs(ag, world, want_pairs, learner, tag=f"{tag}_spec", gmin=gmin)
        unrelated_ok = all(q["ranking_ok"] for q in probed["probes"][1:]) if len(probed["probes"]) == 4 else False
        reversed_ok = bool(probed["probes"] and probed["probes"][0]["ranking_ok"])
        passed = bool(probed["robust_ok"] and unrelated_ok and reversed_ok)
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
        "n_replay_updates": int(n_replay),
        "n_checkpoint_errors": int(n_checkpoint_errors),
        "monotonic_retention_ok": bool(n_checkpoint_errors == 0),
        "n_updates": int(getattr(learner, "n_updates", n_live + n_replay)),
        "store": store.stats() if store is not None else None,
        "matcher": store.ledger.summary() if store is not None else None,
        "l4_ceiling_only": arm == "L4",
        "want_reversed": h_new,
        "want_kept": [h for _c, h in pairs[1:]],
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
            "l4_ceiling_only": arm == "L4",
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


def _l2_match_failure(cells: list[dict[str, Any]]) -> bool:
    rows = [c for c in cells if c["arm"] == "L2" and c["kind"] in ("eco", "spec")]
    if not rows:
        return False
    return any(
        bool((c.get("matcher") or {}).get("reversal_matcher_miss"))
        or bool((c.get("matcher") or {}).get("ecological_match_failure"))
        for c in rows
    )


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    flags = {arm: _phase_flags(cells, arm) for arm in ARMS}
    extra = {
        **flags,
        "l2_reversal_matcher_miss": _l2_match_failure(cells),
        "l2_match_failure": _l2_match_failure(cells),
    }
    ladder = p["decision_ladder"]
    if _four(flags["L0"]):
        return ladder[0]["id"], ladder[0]["then"], extra
    if _four(flags["L1"]):
        return ladder[1]["id"], ladder[1]["then"], extra
    if (not flags["L1"]["plasticity"]) and _four(flags["L2"]) and (not flags["L3"]["plasticity"]):
        return ladder[2]["id"], ladder[2]["then"], extra
    if _four(flags["L2"]) and _four(flags["L3"]):
        return ladder[3]["id"], ladder[3]["then"], extra
    if (not _four(flags["L2"])) and extra["l2_match_failure"]:
        return ladder[4]["id"], ladder[4]["then"], extra
    if _four(flags["L4"]) and (not any(_four(flags[a]) for a in ("L0", "L1", "L2", "L3"))):
        return ladder[5]["id"], ladder[5]["then"], extra
    return ladder[6]["id"], ladder[6]["then"], extra


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
        rows.append(
            {
                "id": c["id"],
                "arm": c["arm"],
                "kind": c["kind"],
                "n_cues": c["n_cues"],
                "passed": c["passed"],
                "ranking_ok": c.get("ranking_ok"),
                "reversal_matcher_miss": (c.get("matcher") or {}).get("reversal_matcher_miss"),
                "ecological_match_failure": (c.get("matcher") or {}).get("ecological_match_failure"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_cell_coverage(cells: list[dict[str, Any]]) -> str:
    ids = [c["id"] for c in cells]
    expected = expected_cell_ids()
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    if set(ids) != set(expected):
        raise RuntimeError("cell IDs do not match frozen MEMORYLIFECYCLEMAP grid")
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
        if c["arm"] == "L4" and not c.get("l4_ceiling_only", False):
            raise RuntimeError("L4 must remain ceiling-only")
        if SCORE_DOMAIN in str(c.get("domain") or ""):
            raise RuntimeError("SCORE identifier appeared in DEV payload")
        store = c.get("store")
        if store is not None and int(store.get("n_p1_scalars") or 0) > MAX_STATE_SCALARS:
            raise RuntimeError("episode store exceeded 512 state scalars")
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
                    tag=f"mlm_{arm}_{wi}_{n_cues}_{order}",
                )
                cells.append(
                    decorate(acq, kind="acquire", arm=arm, n_cues=n_cues, order=order, world=wi, domain=world["domain"])
                )
                cells.append(
                    decorate(stab, kind="stable", arm=arm, n_cues=n_cues, order=order, world=wi, domain=world["domain"])
                )
    world_t = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
    world_t["purpose"] = "rename_twin"
    if SCORE_DOMAIN in world_t["domain"] or "SCORE." in world_t["domain"]:
        raise RuntimeError("SCORE identifier appeared in DEV payload")
    pairs_t = mapping_pairs(world_t, flip=False)
    for order in TEACH_ORDERS:
        acq, _stab = eval_phased_map(
            arm=arm, world=world_t, pairs=pairs_t, order=order, tag=f"mlm_{arm}_twin_{order}"
        )
        cells.append(
            decorate(acq, kind="twin", arm=arm, n_cues=2, order=order, world=1, domain=world_t["domain"])
        )
    world_c = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    eco = eval_ecological(arm, world_c, tag=f"mlm_{arm}_eco")
    cells.append(decorate(eco, kind="eco", arm=arm, n_cues=2, order="A_then_B", world=0, domain=world_c["domain"]))
    world_s = capacity_world(0, DEV_DOMAIN, n_cues=4, n_handles=2)
    pairs_s = mapping_pairs(world_s, flip=False)
    spec = eval_specificity(arm, world_s, pairs_s, tag=f"mlm_{arm}_spec")
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
        if c["kind"] in ("acquire", "stable", "twin", "spec") and int(c.get("n_probe") or 0) != int(c["n_cues"]):
            raise RuntimeError(f"empty probe {c['id']}")
        if c.get("bounded_match_sanity_used_for_pass"):
            raise RuntimeError(f"bounded match used as stability gate {c['id']}")
        if c.get("stability_gate") != "ranking_perturb_sigma_0.01":
            raise RuntimeError(f"stability gate missing {c['id']}")
        if c["arm"] == "L4" and int(c.get("n_live_reversal_updates") or 0) != 0:
            raise RuntimeError(f"L4 live reversal {c['id']}")
        if c["arm"] in MATCHED_LIVE_ARMS:
            if not isinstance(c.get("matcher"), dict):
                raise RuntimeError(f"missing matcher {c['id']}")
            if c.get("store") is None:
                raise RuntimeError(f"missing store stats {c['id']}")
            matcher = c["matcher"]
            if "bounded_match_sanity" not in matcher or "ecological_match_stability" not in matcher:
                raise RuntimeError(f"missing dual perturbation {c['id']}")
            if c["arm"] == "L3" and int((c.get("store") or {}).get("n_evicted") or 0) > 0:
                raise RuntimeError(f"L3 evicted {c['id']}")
    live_by_kind: dict[str, set[int]] = {}
    for c in cells:
        if c["kind"] in ("eco", "spec") and c["arm"] in MATCHED_LIVE_ARMS:
            live_by_kind.setdefault(str(c["kind"]), set()).add(int(c.get("n_live_reversal_updates") or 0))
    for kind, vals in live_by_kind.items():
        if len(vals) != 1:
            raise RuntimeError(f"unmatched L1-L3 live reversal on {kind}: {sorted(vals)}")
        if 0 in vals:
            raise RuntimeError(f"L1-L3 live reversal missing on {kind}")
    manifest = assert_cell_coverage(cells)
    code, then, extra = _decision(cells, p)
    out = {
        "version": "TM.0.24.MEMORYLIFECYCLEMAP.R2.DEV",
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
        "l4_ceiling_only": True,
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
        "shas": memorylifecyclemap_shas(),
        "note": "MEMORYLIFECYCLEMAP R2 DEV only. V1 freeze preserved. Write-geometry closed. L4 ceiling-only. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def smoke() -> dict[str, Any]:
    p = load_prereg()
    world = capacity_world(0, "TM024.MEMORYLIFECYCLEMAP.R2.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    radius = float(p["match"]["radius"])
    store = EpisodeStore(policy="replace", match_l2=radius)
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    y = unit_or_zero(np.arange(64, dtype=np.float64)[::-1] + 3.0)
    store.write(x, world["handles"][0], 1.0, world=world)
    store.write(x, world["handles"][0], -1.0, world=world)
    store.write(y, world["handles"][1], 1.0, world=world)
    fifo = EpisodeStore(policy="fifo", match_l2=radius)
    stale = EpisodeStore(policy="retain_stale", match_l2=radius)
    for _i in range(2):
        fifo.write(x, world["handles"][0], 1.0, world=world)
        stale.write(x, world["handles"][0], 1.0, world=world)
    stale.write(x, world["handles"][0], -1.0, world=world)
    q_tie = fifo.query(x)
    acq, stab = eval_phased_map(
        arm="L1", world=world, pairs=pairs, order="A_then_B", tag="mlmsmk"
    )
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
        "match_radius": radius,
        "match_threshold": radius,
        "replace_n_invalidated": store.stats()["n_invalidated"],
        "replace_n_valid": store.stats()["n_valid"],
        "fifo_n_occupied": fifo.stats()["n_occupied"],
        "stale_n_valid": stale.stats()["n_valid"],
        "stale_n_refused": stale.stats()["n_refused"],
        "stale_n_evicted": stale.stats()["n_evicted"],
        "stale_n_invalidated": stale.stats()["n_invalidated"],
        "fifo_query_kind": q_tie["kind"],
        "fifo_query_match_i": q_tie["match_i"],
        "l1_acquire_n_probe": acq["n_probe"],
        "l1_stable_n_probe": stab["n_probe"],
        "stability_gate": acq.get("stability_gate"),
        "bounded_match_sanity_used_for_pass": acq.get("bounded_match_sanity_used_for_pass"),
        "expected_id_count": len(ids),
        "l4_ceiling_only": True,
        "episode_fields": list(ALLOWED_EPISODE_FIELDS),
    }


def expected_ids_sha(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_r2_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        raise RuntimeError("memorylifecyclemap r2 manifest already exists")
    refuse_v31()
    ids = expected_cell_ids()
    shas = {k: v for k, v in memorylifecyclemap_shas().items() if k != "manifest"}
    out = {
        "version": "TM.0.24.MEMORYLIFECYCLEMAP.R2.MANIFEST",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "lab": "TM.0.24.MEMORYLIFECYCLEMAP.R2",
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
        "l4_no_live_reversal": True,
        "perturbation_modes": ["bounded_match_sanity", "ecological_match_stability"],
        "lifecycle_stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_cannot_satisfy_stability_gate": True,
        "historical_runner_lock_sha": V1_RUNNER_LOCK_SHA,
        "historical_runner_py_sha": V1_RUNNER_PY_SHA,
        "shas": shas,
        "n": 64,
        "note": "R2 cell-ID manifest. V1 freeze preserved. Product remains 0.0.004.",
    }
    MANIFEST.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("memorylifecyclemap r2 runner.lock already exists")
    refuse_v31()
    if sha_file(HIST_RUNNER_LOCK) != V1_RUNNER_LOCK_SHA:
        raise RuntimeError("V1 runner.lock must remain the published freeze")
    if sha_file(HIST_RUNNER_PY) != V1_RUNNER_PY_SHA:
        raise RuntimeError("V1 runner.py must remain the published freeze")
    if not MANIFEST.exists():
        write_r2_manifest()
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.MEMORYLIFECYCLEMAP.R2.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "shas": memorylifecyclemap_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "arms": list(ARMS),
        "matched_live_reversal_arms": list(MATCHED_LIVE_ARMS),
        "l4_no_live_reversal": True,
        "match_radius": float(prereg["match"]["radius"]),
        "match_threshold": float(prereg["match"]["threshold"]),
        "match_role": prereg["match"]["role"],
        "l3_refuse_contradictory_replacement": True,
        "perturbation_modes": ["bounded_match_sanity", "ecological_match_stability"],
        "lifecycle_stability_gate": "ranking_perturb_sigma_0.01",
        "bounded_cannot_satisfy_stability_gate": True,
        "n_slots": N_SLOTS,
        "max_state_scalars": MAX_STATE_SCALARS,
        "l4_ceiling_only": True,
        "trace_budget_unopened": 512,
        "declared_budget_remains_closed": 1536,
        "expected_n_cells": EXPECTED_N_CELLS,
        "historical_runner_lock_sha": V1_RUNNER_LOCK_SHA,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen MEMORYLIFECYCLEMAP R2 runner. V1 runner.lock preserved. DEV lock only after this file is on origin/main. No neural edit.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    refuse_rerun()
    manifest = assert_cell_coverage(out["cells"])
    if len(set(c["id"] for c in out["cells"])) != EXPECTED_N_CELLS:
        raise RuntimeError("unique IDs required before writing DEV")
    if out.get("manifest_sha") != manifest:
        raise RuntimeError("DEV manifest hash must be asserted before write")
    refuse_score_markers(json.dumps(out, default=str))
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("memorylifecyclemap decision lock already exists")
    out = {
        "version": "TM.0.24.MEMORYLIFECYCLEMAP.R2.DECISION",
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
        "q3": False,
        "eligibility_budget_installed": False,
        "trace_rows_installed": False,
        "declared_budget_remains_closed": 1536,
        "write_geometry_branch_closed": True,
        "l4_ceiling_only": True,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "phase_flags": dev.get("phase_flags"),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "R2 runner-only memory lifecycle on P1. V1 freeze preserved. Write-geometry closed. "
            "No v31. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.MEMORYLIFECYCLEMAP R2 DEV\n\n"
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
        raise RuntimeError("MEMORYLIFECYCLEMAP R2 DEV lock requires r2.runner.lock on origin/main after this freeze")
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
        assert p["match"]["threshold"] == 0.05
        assert p["match"]["role"] == "preregistered_match_radius"
        assert p["match"]["not_same_cue_ceiling"] is True
        assert p["arms"]["L3"]["refuse_contradictory_replacement"] is True
        assert p["arms"]["L1"]["matched_live_reversal"] is True
        assert p["arms"]["L2"]["matched_live_reversal"] is True
        assert p["arms"]["L3"]["matched_live_reversal"] is True
        assert p["arms"]["L4"]["live_reversal_trains_learner"] is False
        assert p["matched_live_reversal_arms"] == ["L1", "L2", "L3"]
        assert p["phased_contract"]["bounded_match_sanity_cannot_satisfy_stability_gate"] is True
        assert p["match"]["perturbation"]["ecological_match_stability"]["sigma"] == 0.01
        assert p["domains"]["DEV"] == DEV_DOMAIN
        assert p["historical_runner_lock_sha"] == V1_RUNNER_LOCK_SHA
        assert p["episode_store"]["max_state_scalars"] == 512
        assert p["expected_n_cells"] == EXPECTED_N_CELLS
        assert p["arms"]["L4"]["ceiling_only"] is True
        assert len(p["decision_ladder"]) == 7
        assert sha_file(HIST_RUNNER_LOCK) == V1_RUNNER_LOCK_SHA
        print(json.dumps({"ok": True, "product": p["product"], "expected_n_cells": EXPECTED_N_CELLS, "r2": True}, indent=2))
    elif args.write_manifest:
        print(json.dumps(write_r2_manifest(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2, default=str))
    elif args.dev:
        out = run_dev()
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_dev_lock:
        refuse_dev_lock()
        out = run_dev()
        write_dev_lock(out)
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_decision:
        if not DEV_LOCK.exists():
            raise RuntimeError("write DEV lock first")
        dev = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        print(json.dumps(write_decision(dev), indent=2, default=str))
    elif args.score:
        refuse_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
