"""TM.0.24.CONVERGENCEMAP — runner-only one-shot vs replay on exact P1.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Write-geometry closed. W1 not resurrected. Default v29 query scoring.
DEV on unused TM024.CONVERGENCEMAP.DEV. after this freeze is on origin/main.
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
from experiments.run_tm024eligmap import _fresh, capacity_world, clipnorm, mapping_pairs, record_rest, unit_or_zero
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024tracebridge import CompetitiveBank, probe_address, require_query, teach_bridged, winner_of
from experiments.run_tm024writegeom import NEG_DELTA, SequentialRLS, domain_seed, ranking_margin, set_handle_delta
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_convergencemap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_convergencemap_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_convergencemap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_convergencemap.runner.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_convergencemap.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_convergencemap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024convergencemap_results.md"
TB_DEC = REPO_ROOT / "docs" / "lineage_tracebridge.decision.lock"
TB_ADD = REPO_ROOT / "docs" / "lineage_tracebridge.decision.addendum.lock"
TB_DEV = REPO_ROOT / "docs" / "lineage_tracebridge.dev.lock"
TB_RUNNER = REPO_ROOT / "experiments" / "run_tm024tracebridge.py"
WG_RUNNER = REPO_ROOT / "experiments" / "run_tm024writegeom.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.CONVERGENCEMAP.DEV."
TWIN_DOMAIN = "TM024.CONVERGENCEMAP.TWIN."
SCORE_DOMAIN = "TM024.CONVERGENCEMAP.SCORE."
SCORE_MARKERS = ("TM024.CONVERGENCEMAP.SCORE.",)
EPS = 1e-12
ARMS = ("C0", "C1", "C2", "C3", "C4")
C1_LIVE_CYCLES = (1, 2, 4, 8, 16)
C3_LIVE_CYCLES = (2, 4, 8, 16)
REPLAY_CYCLES = 16
EXPECTED_N_RANK = 168
EXPECTED_N_TWIN = 28
EXPECTED_N_ECO = 12
EXPECTED_N_REST = 12
EXPECTED_N_CELLS = EXPECTED_N_RANK + EXPECTED_N_TWIN + EXPECTED_N_ECO + EXPECTED_N_REST
EXPECTED_N_LIVE = 176
EXPECTED_N_REPLAY = 44
EXPECTED_C1_K16_LIVE = 16
EXPECTED_C1_K16_REPLAY = 14
EXPECTED_C3_K16_LIVE = 16
EXPECTED_C3_K16_REPLAY = 14
PA_BISECT = 50
PA_GEO_ATOL = 1e-9


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def live_specs() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = [("C0", 1)]
    out.extend(("C1", k) for k in C1_LIVE_CYCLES)
    out.append(("C2", 1))
    out.extend(("C3", k) for k in C3_LIVE_CYCLES)
    return out


def replay_specs() -> list[tuple[str, int]]:
    return [("C4", REPLAY_CYCLES), ("C1", REPLAY_CYCLES), ("C3", REPLAY_CYCLES)]


def convergencemap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "tracebridge_decision": TB_DEC,
        "tracebridge_addendum": TB_ADD,
        "tracebridge_dev": TB_DEV,
        "tracebridge_runner": TB_RUNNER,
        "writegeom_runner": WG_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, arm: str, n_cues: int, order: str, world: int, cycles: int, exposure_mode: str) -> str:
    if exposure_mode not in ("live", "replay"):
        raise RuntimeError(f"exposure_mode must be live|replay, got {exposure_mode}")
    return f"{kind}|{arm}|c{n_cues}|{order}|w{world}|k{cycles}|{exposure_mode}"


def cell_exposure_mode(cell: dict[str, Any]) -> str:
    mode = cell.get("exposure_mode") or cell.get("exposure")
    if mode not in ("live", "replay"):
        raise RuntimeError(f"cell missing exposure_mode: {cell.get('id')}")
    return str(mode)


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no convergencemap runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if convergencemap_shas() != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
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


def refuse_pa_grid(p: dict[str, Any]) -> None:
    for arm in ("C2", "C3"):
        if p["arms"][arm].get("learning_rate_grid"):
            raise RuntimeError("passive-aggressive learning-rate grid is refused")


def unique_winner(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    mx = max(float(v) for v in scores.values())
    wins = [h for h, v in scores.items() if float(v) == mx]
    return wins[0] if len(wins) == 1 else None


def is_ranking_error(scores: dict[str, float], chosen: str, adv: float) -> bool:
    win = unique_winner(scores)
    if float(adv) > 0.0:
        return win != chosen
    return win == chosen or win is None


def rival_handle(handles: list[str], chosen: str) -> str:
    others = [h for h in handles if h != chosen]
    if len(others) != 1:
        raise RuntimeError("passive-aggressive requires exactly two handles")
    return others[0]


def desired_pair(handles: list[str], chosen: str, adv: float) -> tuple[str, str]:
    other = rival_handle(handles, chosen)
    if float(adv) > 0.0:
        return chosen, other
    return other, chosen


def geometric_margin(w_ch: np.ndarray, w_ot: np.ndarray, x: np.ndarray) -> float:
    d = np.asarray(w_ch, dtype=np.float64).reshape(-1) - np.asarray(w_ot, dtype=np.float64).reshape(-1)
    xn = unit_or_zero(x)
    dn = float(np.linalg.norm(d))
    if dn <= EPS:
        return 0.0
    return float(np.dot(d, xn) / dn)


def min_tau_for_margin(
    w_ch: np.ndarray,
    w_ot: np.ndarray,
    x: np.ndarray,
    gamma: float,
    *,
    n_bisect: int = PA_BISECT,
) -> float:
    """Smallest τ≥0 such that geometric margin of (w_ch+τx, w_ot-τx) is at least gamma."""
    xn = unit_or_zero(x)
    if float(np.linalg.norm(xn)) <= EPS:
        return 0.0
    d = np.asarray(w_ch, dtype=np.float64).reshape(-1) - np.asarray(w_ot, dtype=np.float64).reshape(-1)
    target = float(gamma)
    if float(np.linalg.norm(d)) <= EPS:
        return target

    def g(tau: float) -> float:
        d2 = d + (2.0 * float(tau)) * xn
        n = float(np.linalg.norm(d2))
        if n <= EPS:
            return 0.0
        return float(np.dot(d2, xn) / n)

    if g(0.0) >= target and float(np.dot(d, xn)) > 0.0:
        return 0.0
    lo = 0.0
    hi = 1.0
    while g(hi) < target and hi < 1e6:
        hi *= 2.0
    if g(hi) < target:
        return float(hi)
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        if g(mid) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def functional_margin(w_ch: np.ndarray, w_ot: np.ndarray, x: np.ndarray) -> float:
    """Unsigned-direction functional margin (w_ch-w_ot)·x. Not the pass criterion."""
    d = np.asarray(w_ch, dtype=np.float64).reshape(-1) - np.asarray(w_ot, dtype=np.float64).reshape(-1)
    return float(np.dot(d, unit_or_zero(x)))


def apply_pa_delta(
    w_ch: np.ndarray,
    w_ot: np.ndarray,
    x: np.ndarray,
    tau: float,
    *,
    row_c_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xn = unit_or_zero(x)
    ch = np.asarray(w_ch, dtype=np.float64).reshape(-1) + float(tau) * xn
    ot = np.asarray(w_ot, dtype=np.float64).reshape(-1) - float(tau) * xn
    if row_c_max is not None:
        ch = clipnorm(ch, float(row_c_max))
        ot = clipnorm(ot, float(row_c_max))
    return ch, ot


def pa_step(
    w_ch: np.ndarray,
    w_ot: np.ndarray,
    x: np.ndarray,
    gamma: float,
    *,
    row_c_max: float | None = None,
) -> dict[str, Any]:
    """Minimal-τ PA. Status is post-update *normalized* geometric margin, never functional margin."""
    xn = unit_or_zero(x)
    target = float(gamma)
    g0 = geometric_margin(w_ch, w_ot, xn)
    f0 = functional_margin(w_ch, w_ot, xn)
    if float(np.linalg.norm(xn)) <= EPS:
        return {
            "tau": 0.0,
            "status": "infeasible",
            "geometric": float(g0),
            "functional": float(f0),
            "w_ch": np.asarray(w_ch, dtype=np.float64).reshape(-1).copy(),
            "w_ot": np.asarray(w_ot, dtype=np.float64).reshape(-1).copy(),
        }
    if g0 >= target - PA_GEO_ATOL and f0 > 0.0:
        return {
            "tau": 0.0,
            "status": "skipped",
            "geometric": float(g0),
            "functional": float(f0),
            "w_ch": np.asarray(w_ch, dtype=np.float64).reshape(-1).copy(),
            "w_ot": np.asarray(w_ot, dtype=np.float64).reshape(-1).copy(),
        }
    tau = min_tau_for_margin(w_ch, w_ot, xn, target)
    ch, ot = apply_pa_delta(w_ch, w_ot, xn, tau, row_c_max=row_c_max)
    g1 = geometric_margin(ch, ot, xn)
    f1 = functional_margin(ch, ot, xn)
    status = "met" if g1 >= target - PA_GEO_ATOL else "infeasible"
    return {
        "tau": float(tau),
        "status": status,
        "geometric": float(g1),
        "functional": float(f1),
        "w_ch": ch,
        "w_ot": ot,
    }


class AlwaysBank:
    def __init__(self, handles: list[str], *, eta: float, c_max: float):
        self.inner = CompetitiveBank(handles, eta=eta, c_max=c_max)
        self.n_updates = 0

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        before = {h: v.copy() for h, v in self.inner.rows.items()}
        self.inner.update(addr, chosen, adv)
        if any(np.linalg.norm(self.inner.rows[h] - before[h]) > EPS for h in before):
            self.n_updates += 1

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        return self.inner.scores(addr)


class ErrorOnlyBank:
    def __init__(self, handles: list[str], *, eta: float, c_max: float):
        self.inner = CompetitiveBank(handles, eta=eta, c_max=c_max)
        self.n_updates = 0

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        if abs(float(adv)) <= EPS:
            return
        if not is_ranking_error(self.inner.scores(addr), chosen, adv):
            return
        self.inner.update(addr, chosen, adv)
        self.n_updates += 1

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        return self.inner.scores(addr)


class PassiveAggressive:
    def __init__(self, handles: list[str], *, gamma: float, row_c_max: float | None = None):
        self.handles = list(handles)
        self.gamma = float(gamma)
        self.row_c_max = row_c_max
        self.rows = {h: np.zeros(64, dtype=np.float64) for h in self.handles}
        self.n_updates = 0
        self.n_infeasible = 0
        self.last_tau = 0.0
        self.last_status = "skipped"
        self.last_geometric = 0.0
        self.last_functional = 0.0

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        if abs(float(adv)) <= EPS or chosen not in self.rows:
            self.last_tau = 0.0
            self.last_status = "skipped"
            return
        x = unit_or_zero(addr)
        ch, ot = desired_pair(self.handles, chosen, adv)
        step = pa_step(self.rows[ch], self.rows[ot], x, self.gamma, row_c_max=self.row_c_max)
        self.last_tau = float(step["tau"])
        self.last_status = str(step["status"])
        self.last_geometric = float(step["geometric"])
        self.last_functional = float(step["functional"])
        if step["status"] == "skipped":
            return
        self.rows[ch] = np.asarray(step["w_ch"], dtype=np.float64)
        self.rows[ot] = np.asarray(step["w_ot"], dtype=np.float64)
        self.n_updates += 1
        if step["status"] == "infeasible":
            self.n_infeasible += 1

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        x = unit_or_zero(addr)
        out: dict[str, float] = {}
        for h, row in self.rows.items():
            rn = float(np.linalg.norm(row))
            out[h] = 0.0 if rn <= EPS else float(np.dot(row, x) / rn)
        return out


class RlsBank:
    def __init__(self, handles: list[str], *, lam: float):
        self.inner = SequentialRLS(64, handles, lam=lam)
        self.n_updates = 0

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        self.inner.update(addr, chosen, adv)
        self.n_updates += 1

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        return self.inner.scores(addr)


def make_learner(arm: str, handles: list[str], p: dict[str, Any]) -> Any:
    if arm == "C0":
        a = p["arms"]["C0"]
        return AlwaysBank(handles, eta=float(a["eta"]), c_max=float(a["c_max"]))
    if arm == "C1":
        a = p["arms"]["C1"]
        return ErrorOnlyBank(handles, eta=float(a["eta"]), c_max=float(a["c_max"]))
    if arm in ("C2", "C3"):
        a = p["arms"][arm]
        if a.get("learning_rate_grid"):
            raise RuntimeError("passive-aggressive learning-rate grid is refused")
        if a.get("clipnorm"):
            raise RuntimeError("passive-aggressive clipnorm is refused")
        return PassiveAggressive(handles, gamma=float(a["geometric_margin_target"]))
    if arm == "C4":
        a = p["arms"]["C4"]
        return RlsBank(handles, lam=float(a["lambda"]))
    raise RuntimeError(f"unknown arm {arm}")


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
        ranked = winner_of(scores)
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
    probe = clone_frozen(ag)
    require_query(probe)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    addr = probe_address("B3", probe, None)
    scores = learner.scores(addr)
    if int(getattr(learner, "n_updates", 0)) != n0:
        raise RuntimeError("retention/final probe updated learner weights")
    if float((ag.W_act_query - w0).abs().max().item()) > 1e-12:
        raise RuntimeError("retention/final probe updated organism state")
    win = winner_of(scores)
    margin = ranking_margin(scores, win or "") if win else 0.0
    return {"cue": cue, "winner": win, "addr": addr, "scores": scores, "margin": float(margin)}


def eval_learner_block(
    *,
    arm: str,
    cycles: int,
    exposure_mode: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool = False,
) -> dict[str, Any]:
    if exposure_mode not in ("live", "replay"):
        raise RuntimeError(f"exposure_mode must be live|replay, got {exposure_mode}")
    p = load_prereg()
    refuse_pa_grid(p)
    gmin = float(p["margin"]["native_ranking_min"])
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    want = {c: h for c, h in pairs}
    learner = make_learner(arm, handles, p)
    retention_ok = True
    n_checkpoints = 0
    n_live = 0
    n_capture = 0
    n_replay = 0
    last_by_cue: dict[str, dict[str, Any]] = {}
    stored: list[dict[str, Any]] = []
    taught_cues: list[str] = []
    p1_source = "live_regenerated" if exposure_mode == "live" else "frozen_first_pass"

    def mark_retention(ok: bool) -> None:
        nonlocal retention_ok, n_checkpoints
        n_checkpoints += 1
        retention_ok = retention_ok and bool(ok)

    def freeze_row(rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "cue": rec["cue"],
            "handle": rec["handle"],
            "adv": rec["adv"],
            "addr": np.asarray(rec["addr"], dtype=np.float64).copy(),
        }

    with tempfile.TemporaryDirectory(prefix="cvg_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        if exposure_mode == "live":
            for cy in range(int(cycles)):
                for i, (cue, handle) in enumerate(seq):
                    rec = teach_bridged(
                        ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_c{cy}_t{i}"
                    )
                    n_live += 1
                    learner.update(rec["addr"], rec["handle"], rec["adv"])
                    last_by_cue[cue] = freeze_row(rec)
                    if cue not in taught_cues:
                        taught_cues.append(cue)
                    ok = True
                    for tc in taught_cues:
                        pr = live_probe(ag, world, tc, learner, tag=f"{tag}_r{cy}_{tc}")
                        ok = ok and pr["winner"] == want[tc]
                    mark_retention(ok)
        else:
            for i, (cue, handle) in enumerate(seq):
                rec = teach_bridged(
                    ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_cap{i}"
                )
                n_capture += 1
                stored.append(freeze_row(rec))
                last_by_cue[cue] = stored[-1]
            frozen_addrs = [np.asarray(r["addr"], dtype=np.float64).copy() for r in stored]
            if rest:
                record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
                rest = False
            seen: list[str] = []
            for cy in range(int(cycles)):
                for j, rec in enumerate(stored):
                    if not np.allclose(rec["addr"], frozen_addrs[j]):
                        raise RuntimeError("replay mutated frozen first-pass P1 rows")
                    learner.update(frozen_addrs[j], rec["handle"], rec["adv"])
                    n_replay += 1
                    if rec["cue"] not in seen:
                        seen.append(rec["cue"])
                    rows = [r for r in stored if r["cue"] in seen]
                    mark_retention(all(winner_of(learner.scores(r["addr"])) == r["handle"] for r in rows))
        if rest:
            record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
        train_rank = bool(last_by_cue) and all(
            winner_of(learner.scores(last_by_cue[c]["addr"])) == h for c, h in pairs
        )
        probes: list[dict[str, Any]] = []
        ranking_ok = True
        all_ok = True
        for i, (cue, handle) in enumerate(pairs):
            pr = live_probe(ag, world, cue, learner, tag=f"{tag}_p{i}")
            stab = perturb_rank(
                learner.scores,
                pr["addr"],
                pr["winner"] or "",
                domain=world["domain"],
                key=f"{tag}_{cue}",
            )
            ok = bool(pr["winner"] == handle and pr["margin"] >= gmin and stab["stable"])
            ranking_ok = ranking_ok and bool(pr["winner"] == handle)
            all_ok = all_ok and ok
            probes.append(
                {
                    "cue": cue,
                    "want": handle,
                    "winner": pr["winner"],
                    "margin": float(pr["margin"]),
                    "perturb_stable": bool(stab["stable"]),
                    "ok": ok,
                }
            )
    passed = bool(all_ok and retention_ok)
    n_train = int(getattr(learner, "n_updates", n_live + n_replay))
    out = {
        "passed": passed,
        "ranking_ok": ranking_ok,
        "train_ranking_ok": bool(train_rank),
        "retention_ok": bool(retention_ok),
        "n_checkpoints": int(n_checkpoints),
        "perturb_stable": bool(all(q["perturb_stable"] for q in probes)) if probes else False,
        "min_probe_margin": float(min(q["margin"] for q in probes)) if probes else 0.0,
        "taught": [{"cue": t["cue"], "handle": t["handle"], "adv": t["adv"]} for t in (stored or list(last_by_cue.values()))],
        "probes": probes,
        "n_train": n_train,
        "n_live_teaches": int(n_live),
        "n_capture": int(n_capture),
        "n_replay_updates": int(n_replay),
        "n_probe": len(pairs),
        "n_updates": int(getattr(learner, "n_updates", 0)),
        "cycles": int(cycles),
        "exposure_mode": exposure_mode,
        "p1_source": p1_source,
        "k_complete_passes": int(cycles),
    }
    if isinstance(learner, PassiveAggressive):
        out["pa_last_status"] = learner.last_status
        out["pa_last_geometric"] = float(learner.last_geometric)
        out["pa_n_infeasible"] = int(learner.n_infeasible)
    return out


def eval_ecological(
    arm: str,
    cycles: int,
    exposure_mode: str,
    world: dict[str, Any],
    *,
    tag: str,
) -> dict[str, Any]:
    if exposure_mode not in ("live", "replay"):
        raise RuntimeError(f"exposure_mode must be live|replay, got {exposure_mode}")
    cue = world["cue_handle"][0]["cue"]
    h1 = world["handles"][0]
    h2 = world["handles"][1]
    p = load_prereg()
    refuse_pa_grid(p)
    gmin = float(p["margin"]["native_ranking_min"])
    learner = make_learner(arm, list(world["handles"]), p)
    retention_ok = True
    n_checkpoints = 0
    advs: list[float] = []
    p1_source = "live_regenerated" if exposure_mode == "live" else "frozen_first_pass"
    with tempfile.TemporaryDirectory(prefix="cvg_eco_") as tmp:
        ag = _fresh(tmp, "s", world)

        def teach(w: dict[str, Any], handle: str, suffix: str) -> dict[str, Any]:
            rec = teach_bridged(ag, w, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_{suffix}")
            return {
                "cue": rec["cue"],
                "handle": rec["handle"],
                "adv": rec["adv"],
                "addr": np.asarray(rec["addr"], dtype=np.float64).copy(),
            }

        def retain_live(want: str) -> None:
            nonlocal retention_ok, n_checkpoints
            n_checkpoints += 1
            pr = live_probe(ag, world, cue, learner, tag=f"{tag}_r{n_checkpoints}")
            retention_ok = retention_ok and pr["winner"] == want

        if exposure_mode == "live":
            for cy in range(int(cycles)):
                t1 = teach(world, h1, f"c{cy}p")
                learner.update(t1["addr"], t1["handle"], t1["adv"])
                if cy == 0:
                    advs = [float(t1["adv"])]
                retain_live(h1)
                wneg = set_handle_delta(world, h1, NEG_DELTA)
                t2 = teach(wneg, h1, f"c{cy}n")
                learner.update(t2["addr"], t2["handle"], t2["adv"])
                if cy == 0:
                    advs.append(float(t2["adv"]))
                retain_live(h2)
                t3 = teach(world, h2, f"c{cy}r")
                learner.update(t3["addr"], t3["handle"], t3["adv"])
                if cy == 0:
                    advs.append(float(t3["adv"]))
                retain_live(h2)
        else:
            t1 = teach(world, h1, "p")
            wneg = set_handle_delta(world, h1, NEG_DELTA)
            t2 = teach(wneg, h1, "n")
            t3 = teach(world, h2, "r")
            advs = [float(t1["adv"]), float(t2["adv"]), float(t3["adv"])]
            stored = [t1, t2, t3]
            frozen = [np.asarray(r["addr"], dtype=np.float64).copy() for r in stored]
            wants = [h1, h2, h2]
            seen: list[int] = []
            for cy in range(int(cycles)):
                for i, rec in enumerate(stored):
                    if not np.allclose(rec["addr"], frozen[i]):
                        raise RuntimeError("replay mutated frozen first-pass P1 rows")
                    learner.update(frozen[i], rec["handle"], rec["adv"])
                    if i not in seen:
                        seen.append(i)
                    n_checkpoints += 1
                    ok = True
                    for j in seen:
                        sc = learner.scores(stored[j]["addr"])
                        ok = ok and winner_of(sc) == wants[j]
                    retention_ok = retention_ok and ok
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
        ok = bool(
            len(advs) == 3
            and advs[0] > 0.0
            and advs[1] < 0.0
            and advs[2] > 0.0
            and win == h2
            and margin >= gmin
            and stab["stable"]
            and retention_ok
        )
    out = {
        "passed": ok,
        "required": True,
        "adv": advs,
        "winner": win,
        "want": h2,
        "margin": margin,
        "perturb_stable": bool(stab["stable"]),
        "retention_ok": bool(retention_ok),
        "n_checkpoints": int(n_checkpoints),
        "n_updates": int(getattr(learner, "n_updates", 0)),
        "cycles": int(cycles),
        "exposure_mode": exposure_mode,
        "p1_source": p1_source,
        "k_complete_passes": int(cycles),
    }
    if isinstance(learner, PassiveAggressive):
        out["pa_last_status"] = learner.last_status
        out["pa_last_geometric"] = float(learner.last_geometric)
        out["pa_n_infeasible"] = int(learner.n_infeasible)
    return out


def decorate(
    out: dict[str, Any],
    *,
    kind: str,
    arm: str,
    n_cues: int,
    order: str,
    world: int,
    domain: str,
    cycles: int,
    exposure_mode: str,
) -> dict[str, Any]:
    out.update(
        {
            "id": cell_id(kind, arm, n_cues, order, world, cycles, exposure_mode),
            "kind": kind,
            "arm": arm,
            "n_cues": n_cues,
            "order": order,
            "world": world,
            "domain": domain,
            "cycles": int(cycles),
            "exposure_mode": exposure_mode,
            "required": True,
            "c4_ceiling_only": arm == "C4",
        }
    )
    return out


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    def rows(arm: str, cycles: int, exposure_mode: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        out = [
            c
            for c in cells
            if c["arm"] == arm and int(c["cycles"]) == int(cycles) and cell_exposure_mode(c) == exposure_mode
        ]
        if kind is not None:
            out = [c for c in out if c["kind"] == kind]
        return out

    def robust(arm: str, cycles: int, exposure_mode: str) -> bool:
        rank = rows(arm, cycles, exposure_mode, kind="rank")
        twin = rows(arm, cycles, exposure_mode, kind="twin")
        need = rank + twin
        if exposure_mode == "live" or arm == "C4":
            need = need + rows(arm, cycles, exposure_mode, kind="eco") + rows(arm, cycles, exposure_mode, kind="rest")
        return bool(need) and all(bool(c["passed"]) for c in need)

    live_flags = {arm: {k: robust(arm, k, "live") for k in ks} for arm, ks in (
        ("C0", (1,)),
        ("C1", C1_LIVE_CYCLES),
        ("C2", (1,)),
        ("C3", C3_LIVE_CYCLES),
    )}
    replay_flags = {
        "C1": {REPLAY_CYCLES: robust("C1", REPLAY_CYCLES, "replay")},
        "C3": {REPLAY_CYCLES: robust("C3", REPLAY_CYCLES, "replay")},
        "C4": {REPLAY_CYCLES: robust("C4", REPLAY_CYCLES, "replay")},
    }
    extra = {"live": live_flags, "replay": replay_flags}
    ladder = p["decision_ladder"]
    if live_flags["C2"][1] or live_flags["C1"][1]:
        return ladder[0]["id"], ladder[0]["then"], extra
    if any(live_flags["C1"][k] for k in (2, 4, 8, 16)) or any(live_flags["C3"][k] for k in C3_LIVE_CYCLES):
        return ladder[1]["id"], ladder[1]["then"], extra
    live_c1 = any(live_flags["C1"][k] for k in C1_LIVE_CYCLES)
    live_c3 = any(live_flags["C3"][k] for k in C3_LIVE_CYCLES)
    replay_compact = bool(replay_flags["C1"][REPLAY_CYCLES] or replay_flags["C3"][REPLAY_CYCLES])
    if replay_compact and (not live_c1) and (not live_c3):
        return ladder[2]["id"], ladder[2]["then"], extra
    if replay_flags["C4"][REPLAY_CYCLES]:
        return ladder[3]["id"], ladder[3]["then"], extra
    return ladder[4]["id"], ladder[4]["then"], extra


def _ids_for(arm: str, cycles: int, exposure_mode: str, *, include_eco_rest: bool) -> list[str]:
    ids: list[str] = []
    for n in (2, 4, 8):
        for wi in range(2):
            for order in TEACH_ORDERS:
                ids.append(cell_id("rank", arm, n, order, wi, cycles, exposure_mode))
    for order in TEACH_ORDERS:
        ids.append(cell_id("twin", arm, 2, order, 1, cycles, exposure_mode))
    if include_eco_rest:
        ids.append(cell_id("eco", arm, 2, "A_then_B", 0, cycles, exposure_mode))
        ids.append(cell_id("rest", arm, 2, "A_then_B", 0, cycles, exposure_mode))
    return ids


def expected_cell_ids() -> list[str]:
    ids: list[str] = []
    for arm, k in live_specs():
        ids.extend(_ids_for(arm, k, "live", include_eco_rest=True))
    for arm, k in replay_specs():
        ids.extend(_ids_for(arm, k, "replay", include_eco_rest=arm == "C4"))
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
                "cycles": c["cycles"],
                "exposure_mode": cell_exposure_mode(c),
                "passed": c["passed"],
                "retention_ok": c.get("retention_ok"),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_cell_coverage(cells: list[dict[str, Any]]) -> str:
    ids = [c["id"] for c in cells]
    expected = expected_cell_ids()
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    if set(ids) != set(expected):
        raise RuntimeError("cell IDs do not match frozen exposure_mode grid")
    kinds = Counter(c["kind"] for c in cells)
    if dict(kinds) != {"rank": EXPECTED_N_RANK, "twin": EXPECTED_N_TWIN, "eco": EXPECTED_N_ECO, "rest": EXPECTED_N_REST}:
        raise RuntimeError(f"kind counts {dict(kinds)}")
    modes = Counter(cell_exposure_mode(c) for c in cells)
    if dict(modes) != {"live": EXPECTED_N_LIVE, "replay": EXPECTED_N_REPLAY}:
        raise RuntimeError(f"exposure_mode counts {dict(modes)}")
    c1_live = [c for c in cells if c["arm"] == "C1" and int(c["cycles"]) == 16 and cell_exposure_mode(c) == "live"]
    c1_rep = [c for c in cells if c["arm"] == "C1" and int(c["cycles"]) == 16 and cell_exposure_mode(c) == "replay"]
    c3_live = [c for c in cells if c["arm"] == "C3" and int(c["cycles"]) == 16 and cell_exposure_mode(c) == "live"]
    c3_rep = [c for c in cells if c["arm"] == "C3" and int(c["cycles"]) == 16 and cell_exposure_mode(c) == "replay"]
    if len(c1_live) != EXPECTED_C1_K16_LIVE or len(c1_rep) != EXPECTED_C1_K16_REPLAY:
        raise RuntimeError("C1 k=16 live/replay split")
    if len(c3_live) != EXPECTED_C3_K16_LIVE or len(c3_rep) != EXPECTED_C3_K16_REPLAY:
        raise RuntimeError("C3 k=16 live/replay split")
    if set(c["id"] for c in c1_live) & set(c["id"] for c in c1_rep):
        raise RuntimeError("C1 k=16 live/replay IDs collided")
    if set(c["id"] for c in c3_live) & set(c["id"] for c in c3_rep):
        raise RuntimeError("C3 k=16 live/replay IDs collided")
    orders = Counter((c["kind"], c["order"]) for c in cells if c["kind"] in ("rank", "twin"))
    if orders[("rank", "A_then_B")] != orders[("rank", "B_then_A")]:
        raise RuntimeError("rank order counts unequal")
    if orders[("twin", "A_then_B")] != orders[("twin", "B_then_A")]:
        raise RuntimeError("twin order counts unequal")
    for c in cells:
        if c["arm"] == "C4" and not c.get("c4_ceiling_only", False):
            raise RuntimeError("C4 must remain ceiling-only")
        if c["arm"] == "C4" and cell_exposure_mode(c) != "replay":
            raise RuntimeError("C4 must be replay")
        if c.get("p1_source") not in ("live_regenerated", "frozen_first_pass", None):
            raise RuntimeError(f"bad p1_source {c.get('p1_source')}")
        if cell_exposure_mode(c) == "live" and c.get("p1_source") not in (None, "live_regenerated"):
            raise RuntimeError("live cell must regenerate P1")
        if cell_exposure_mode(c) == "replay" and c.get("p1_source") not in (None, "frozen_first_pass"):
            raise RuntimeError("replay cell must reuse frozen first-pass P1")
    return cell_manifest_hash(cells)


def _append_map(
    cells: list[dict[str, Any]],
    *,
    arm: str,
    cycles: int,
    exposure_mode: str,
    include_eco_rest: bool,
) -> None:
    p = load_prereg()
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
                raise RuntimeError("SCORE identifier appeared in DEV payload")
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                block = eval_learner_block(
                    arm=arm,
                    cycles=cycles,
                    exposure_mode=exposure_mode,
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"cvg_{arm}_{exposure_mode}_{wi}_{n_cues}_{order}_k{cycles}",
                )
                cells.append(
                    decorate(
                        block,
                        kind="rank",
                        arm=arm,
                        n_cues=n_cues,
                        order=order,
                        world=wi,
                        domain=world["domain"],
                        cycles=cycles,
                        exposure_mode=exposure_mode,
                    )
                )
    world_t = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
    world_t["purpose"] = "rename_twin"
    if SCORE_DOMAIN in world_t["domain"] or "SCORE." in world_t["domain"]:
        raise RuntimeError("SCORE identifier appeared in DEV payload")
    pairs_t = mapping_pairs(world_t, flip=False)
    for order in TEACH_ORDERS:
        block = eval_learner_block(
            arm=arm,
            cycles=cycles,
            exposure_mode=exposure_mode,
            world=world_t,
            pairs=pairs_t,
            order=order,
            tag=f"cvg_{arm}_{exposure_mode}_twin_{order}_k{cycles}",
        )
        cells.append(
            decorate(
                block,
                kind="twin",
                arm=arm,
                n_cues=2,
                order=order,
                world=1,
                domain=world_t["domain"],
                cycles=cycles,
                exposure_mode=exposure_mode,
            )
        )
    if not include_eco_rest:
        return
    world_c = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    pairs_c = mapping_pairs(world_c, flip=False)
    eco = eval_ecological(arm, cycles, exposure_mode, world_c, tag=f"cvg_{arm}_{exposure_mode}_eco_k{cycles}")
    cells.append(
        decorate(
            eco,
            kind="eco",
            arm=arm,
            n_cues=2,
            order="A_then_B",
            world=0,
            domain=world_c["domain"],
            cycles=cycles,
            exposure_mode=exposure_mode,
        )
    )
    rest = eval_learner_block(
        arm=arm,
        cycles=cycles,
        exposure_mode=exposure_mode,
        world=world_c,
        pairs=pairs_c,
        order="A_then_B",
        tag=f"cvg_{arm}_{exposure_mode}_rest_k{cycles}",
        rest=True,
    )
    cells.append(
        decorate(
            rest,
            kind="rest",
            arm=arm,
            n_cues=2,
            order="A_then_B",
            world=0,
            domain=world_c["domain"],
            cycles=cycles,
            exposure_mode=exposure_mode,
        )
    )


def run_dev() -> dict[str, Any]:
    refuse_v31()
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    refuse_pa_grid(p)
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    cells: list[dict[str, Any]] = []
    for arm, k in live_specs():
        _append_map(cells, arm=arm, cycles=k, exposure_mode="live", include_eco_rest=True)
    for arm, k in replay_specs():
        _append_map(cells, arm=arm, cycles=k, exposure_mode="replay", include_eco_rest=arm == "C4")
    for c in cells:
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        if c["kind"] in ("rank", "twin", "rest"):
            if int(c.get("n_probe") or 0) != int(c["n_cues"]):
                raise RuntimeError(f"empty probe {c['id']}")
            if cell_exposure_mode(c) == "live" and int(c.get("n_live_teaches") or 0) != int(c["n_cues"]) * int(c["cycles"]):
                raise RuntimeError(f"live teach count {c['id']}")
            if cell_exposure_mode(c) == "replay" and int(c.get("n_capture") or 0) != int(c["n_cues"]):
                raise RuntimeError(f"replay capture count {c['id']}")
            if cell_exposure_mode(c) == "replay" and int(c.get("n_replay_updates") or 0) != int(c["n_cues"]) * int(c["cycles"]):
                raise RuntimeError(f"replay update count {c['id']}")
        if not c.get("retention_ok", True) and c["passed"]:
            raise RuntimeError(f"passed without retention {c['id']}")
    manifest = assert_cell_coverage(cells)
    code, then, extra = _decision(cells, p)
    out = {
        "version": "TM.0.24.CONVERGENCEMAP.DEV",
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
        "c4_ceiling_only": True,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "live_robust": extra["live"],
        "replay_robust": extra["replay"],
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells),
        "manifest_sha": manifest,
        "cells": cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": convergencemap_shas(),
        "note": "CONVERGENCEMAP DEV only. Write-geometry closed. C4 ceiling-only. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def smoke() -> dict[str, Any]:
    p = load_prereg()
    refuse_pa_grid(p)
    world = capacity_world(0, "TM024.CONVERGENCEMAP.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    c0 = eval_learner_block(
        arm="C0", cycles=1, exposure_mode="live", world=world, pairs=pairs, order="A_then_B", tag="cvgsmk0"
    )
    c2 = eval_learner_block(
        arm="C2", cycles=1, exposure_mode="live", world=world, pairs=pairs, order="A_then_B", tag="cvgsmk2"
    )
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    z = np.zeros(64, dtype=np.float64)
    step0 = pa_step(z, z, x, 0.01)
    tau0 = min_tau_for_margin(z, z, x, 0.01)
    w_ch = z + 0.01 * x
    w_ot = z - 0.01 * x
    tau_done = min_tau_for_margin(w_ch, w_ot, x, 0.01)
    live_id = cell_id("rank", "C1", 8, "A_then_B", 0, 16, "live")
    replay_id = cell_id("rank", "C1", 8, "A_then_B", 0, 16, "replay")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "c0_n_live": c0["n_live_teaches"],
        "c0_retention_ok": c0["retention_ok"],
        "c0_exposure_mode": c0["exposure_mode"],
        "c0_p1_source": c0["p1_source"],
        "c2_n_live": c2["n_live_teaches"],
        "c2_n_updates": c2["n_updates"],
        "c2_exposure_mode": c2["exposure_mode"],
        "c2_pa_last_status": c2.get("pa_last_status"),
        "c2_pa_last_geometric": c2.get("pa_last_geometric"),
        "tau_zero_init": float(tau0),
        "tau_already_met": float(tau_done),
        "pa_zero_status": step0["status"],
        "pa_zero_geometric": float(step0["geometric"]),
        "c1_k16_ids_distinct": live_id != replay_id,
        "expected_n_cells": EXPECTED_N_CELLS,
        "expected_id_count": len(expected_cell_ids()),
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "act_score_mode": p["act_score_mode"],
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "c4_ceiling_only": True,
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("convergencemap runner.lock already exists")
    refuse_v31()
    prereg = load_prereg()
    refuse_pa_grid(prereg)
    lock = {
        "version": "TM.0.24.CONVERGENCEMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "shas": convergencemap_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "arms": list(ARMS),
        "c1_live_cycles": list(C1_LIVE_CYCLES),
        "c3_live_cycles": list(C3_LIVE_CYCLES),
        "replay_cycles": REPLAY_CYCLES,
        "pa_geometric_margin_target": 0.01,
        "pa_learning_rate_grid": False,
        "pa_status_is_normalized_geometric": True,
        "exposure_mode_field": "exposure_mode",
        "c4_ceiling_only": True,
        "trace_budget_unopened": 512,
        "declared_budget_remains_closed": 1536,
        "expected_n_cells": EXPECTED_N_CELLS,
        "expected_n_live": EXPECTED_N_LIVE,
        "expected_n_replay": EXPECTED_N_REPLAY,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen CONVERGENCEMAP runner. DEV lock only after this file is on origin/main. No neural edit.",
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
        raise RuntimeError("convergencemap decision lock already exists")
    out = {
        "version": "TM.0.24.CONVERGENCEMAP.DECISION",
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
        "c4_ceiling_only": True,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "live_robust": dev.get("live_robust"),
            "replay_robust": dev.get("replay_robust"),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "Runner-only one-shot vs replay on P1. Write-geometry closed. "
            "No v31. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.CONVERGENCEMAP DEV\n\n"
        f"Decision: **{out['decision']['code']}**.\n\n"
        f"Live robust: `{out['decision']['live_robust']}`.\n\n"
        f"Replay robust: `{out['decision']['replay_robust']}`.\n\n"
        "Write-geometry closed. SCORE unopened. No neural candidate. "
        "512/1536 budgets stay closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze installs a sufficient write rule")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("CONVERGENCEMAP DEV lock requires runner.lock on origin/main after this freeze")
    assert_runner_frozen()
    refuse_rerun()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
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
        assert p["arms"]["C2"]["learning_rate_grid"] is False
        assert p["arms"]["C2"]["geometric_margin_target"] == 0.01
        assert p["expected_n_cells"] == EXPECTED_N_CELLS
        print(json.dumps({"ok": True, "product": p["product"], "expected_n_cells": EXPECTED_N_CELLS}, indent=2))
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
