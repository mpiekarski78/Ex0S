"""TM.0.24.TRACEBRIDGE — runner-only P1 trace-sufficiency test.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Write-geometry closed. W1 not resurrected. Default v29 query scoring.
DEV on unused TM024.TRACEBRIDGE.DEV. after this freeze is on origin/main.
SCORE reserved and unopened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.run_tm023cortex import physics, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue, prep_eval
from experiments.run_tm024collisionmap import parse_stages
from experiments.run_tm024discrimmap_r2 import (
    ACCEPTED_STATUS,
    hard_margin_linear,
    min_geometric_margin,
    perturb_sign_stable,
    require_accepted,
)
from experiments.run_tm024eligmap import (
    _fresh,
    capacity_world,
    clipnorm,
    mapping_pairs,
    record_rest,
    unit_or_zero,
)
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024writegeom import NEG_DELTA, domain_seed, ranking_margin, set_handle_delta
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_tracebridge.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_tracebridge_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_tracebridge.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_tracebridge.runner.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_tracebridge.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_tracebridge.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024tracebridge_results.md"
PM_DEC = REPO_ROOT / "docs" / "lineage_phasemap.decision.lock"
PM_ADD = REPO_ROOT / "docs" / "lineage_phasemap.decision.addendum.lock"
R2_PREREG = REPO_ROOT / "docs" / "lineage_discrimmap.r2.prereg.lock"
R2_RUNNER = REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.TRACEBRIDGE.DEV."
TWIN_DOMAIN = "TM024.TRACEBRIDGE.TWIN."
SCORE_DOMAIN = "TM024.TRACEBRIDGE.SCORE."
SCORE_MARKERS = ("TM024.TRACEBRIDGE.SCORE.",)
EPS = 1e-12
ARMS = ("B0", "B1", "B2", "B3", "B4")
V29_ARMS = ("B0", "B1", "B2")
EXPECTED_N_RANK = 5 * 3 * 2 * 2
EXPECTED_N_TWIN = 5 * 2
EXPECTED_N_ECO = 5
EXPECTED_N_HOLD = 5
EXPECTED_N_REST = 5
EXPECTED_N_CELLS = EXPECTED_N_RANK + EXPECTED_N_TWIN + EXPECTED_N_ECO + EXPECTED_N_HOLD + EXPECTED_N_REST


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def d1_spec() -> dict[str, Any]:
    spec = json.loads(R2_PREREG.read_text(encoding="utf-8"))["arms"]["D1"]
    if spec.get("soft_margin") or spec.get("soft_margin_C") is not None:
        raise RuntimeError("D1 oracle acquired a soft-margin degree of freedom")
    return spec


def tracebridge_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "phasemap_decision": PM_DEC,
        "phasemap_addendum": PM_ADD,
        "r2_prereg": R2_PREREG,
        "r2_runner": R2_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, arm: str, n_cues: int, order: str, world: int) -> str:
    return f"{kind}|{arm}|c{n_cues}|{order}|w{world}"


def require_query(ag: NeuralCortex) -> None:
    if str(ag.genome.act_score_mode) != "query":
        raise RuntimeError("act_score_mode not query")


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no tracebridge runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if tracebridge_shas() != lock.get("shas"):
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


class EventBoundaryTrace:
    """Eight actuator-local rows. Ingest only at innate v_end. λ=0 is a register."""

    def __init__(self, n: int, handles: list[str], lam: float):
        self.n = int(n)
        self.lam = float(lam)
        self.handles = list(handles)
        z = np.zeros(self.n, dtype=np.float64)
        self.e = {h: z.copy() for h in self.handles}

    def copy(self) -> "EventBoundaryTrace":
        out = EventBoundaryTrace(self.n, self.handles, self.lam)
        out.e = {h: v.copy() for h, v in self.e.items()}
        return out

    def ingest_event_end(self, rho: np.ndarray) -> None:
        r = unit_or_zero(rho)
        om = 1.0 - self.lam
        for h in self.handles:
            self.e[h] = unit_or_zero(self.lam * self.e[h] + om * r)

    def address(self) -> np.ndarray:
        return next(iter(self.e.values())).copy()


class CompetitiveBank:
    """Runner-only D3 geometry, signed by delayed advantage. Not installed."""

    def __init__(self, handles: list[str], *, eta: float, c_max: float):
        self.handles = list(handles)
        self.eta = float(eta)
        self.c_max = float(c_max)
        self.rows = {h: np.zeros(64, dtype=np.float64) for h in self.handles}

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        if abs(float(adv)) <= EPS or chosen not in self.rows:
            return
        ehat = unit_or_zero(addr)
        sign = 1.0 if float(adv) > 0.0 else -1.0
        for h in self.handles:
            delta = sign * self.eta * ehat if h == chosen else -sign * self.eta * ehat
            self.rows[h] = clipnorm(self.rows[h] + delta, self.c_max)

    def scores(self, addr: np.ndarray) -> dict[str, float]:
        x = unit_or_zero(addr)
        out: dict[str, float] = {}
        for h, row in self.rows.items():
            rn = float(np.linalg.norm(row))
            out[h] = 0.0 if rn <= EPS else float(np.dot(row, x) / rn)
        return out


def event_end_rho(ag: NeuralCortex) -> np.ndarray:
    """Structural v_end tick: penultimate sensory state when a cue event is present."""
    sens = [np.asarray(x, dtype=np.float64) for x in ag.sensory_trajectory]
    if not sens:
        raise RuntimeError("empty sensory trajectory")
    if len(sens) >= 3:
        return sens[-2].copy()
    return sens[-1].copy()


def p5_rho(ag: NeuralCortex) -> np.ndarray:
    return np.asarray(parse_stages(ag)["rho_elig"], dtype=np.float64).copy()


def inject_pending(ag: NeuralCortex, addr: np.ndarray) -> None:
    vec = np.asarray(addr, dtype=np.float64).reshape(-1).copy()
    if ag._pending is None:
        raise RuntimeError("no pending action for address injection")
    ag._pending["rho_elig"] = vec.copy()
    ag._pending["rho_op"] = vec.copy()
    ag._pending["rho_motor"] = vec.copy()


def select_address(arm: str, ag: NeuralCortex, tracer: EventBoundaryTrace | None) -> np.ndarray:
    p1 = unit_or_zero(event_end_rho(ag))
    if arm == "B0":
        return unit_or_zero(p5_rho(ag))
    if arm == "B2":
        if tracer is None:
            raise RuntimeError("B2 missing event-boundary trace")
        tracer.ingest_event_end(event_end_rho(ag))
        return unit_or_zero(tracer.address())
    return p1


def probe_address(arm: str, ag: NeuralCortex, tracer: EventBoundaryTrace | None) -> np.ndarray:
    p1 = unit_or_zero(event_end_rho(ag))
    if arm == "B0":
        return unit_or_zero(p5_rho(ag))
    if arm == "B2":
        if tracer is None:
            raise RuntimeError("B2 missing event-boundary trace")
        tr = tracer.copy()
        tr.ingest_event_end(event_end_rho(ag))
        return unit_or_zero(tr.address())
    return p1


def teach_bridged(
    ag: NeuralCortex,
    world: dict[str, Any],
    cue: str,
    handle: str,
    *,
    arm: str,
    tracer: EventBoundaryTrace | None,
    bank: CompetitiveBank | None,
    tag: str,
) -> dict[str, Any]:
    require_query(ag)
    observe_cue(ag, world, tag=f"{tag}_sel", body=list(MID_BODY), symbols=[cue])
    addr = select_address(arm, ag, tracer)
    clamped = ag.clamp_action("ACT", handle)
    if not clamped.get("ok"):
        raise RuntimeError(f"clamp failed {clamped}")
    if arm in V29_ARMS:
        inject_pending(ag, addr)
    _, body2 = physics(list(MID_BODY), handle, world["latent"])
    cred = observe_cue(ag, world, tag=f"{tag}_obs", body=list(body2), symbols=[cue])
    prep_eval(ag)
    adv = float((cred.get("metrics") or {}).get("adv") or 0.0)
    if abs(adv) <= EPS:
        raise RuntimeError("zero advantage on teach select")
    if arm == "B3" and bank is not None:
        bank.update(addr, handle, adv)
    return {"cue": cue, "handle": handle, "adv": float(adv), "addr": addr}


def score_addr(
    arm: str, ag: NeuralCortex, addr: np.ndarray, bank: CompetitiveBank | None
) -> dict[str, float]:
    if arm == "B3":
        if bank is None:
            raise RuntimeError("B3 missing competitive bank")
        return bank.scores(addr)
    return ag.actuator_scores(addr)


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
        ranked = max(scores, key=lambda h: scores[h])
        if ranked == winner:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def winner_of(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores, key=lambda h: scores[h])


def eval_learner_block(
    *,
    arm: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool = False,
) -> dict[str, Any]:
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    spec = d1_spec()
    with tempfile.TemporaryDirectory(prefix="tb_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        tracer = EventBoundaryTrace(64, handles, float(p["arms"]["B2"]["lambda"])) if arm == "B2" else None
        bank = (
            CompetitiveBank(handles, eta=float(p["arms"]["B3"]["eta"]), c_max=float(p["arms"]["B3"]["c_max"]))
            if arm == "B3"
            else None
        )
        taught = [
            teach_bridged(ag, world, cue, handle, arm=arm, tracer=tracer, bank=bank, tag=f"{tag}_t{i}")
            for i, (cue, handle) in enumerate(seq)
        ]
        taught_by_cue = {t["cue"]: t for t in taught}
        ordered = [taught_by_cue[c] for c, _h in pairs]
        if rest:
            record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
        teach_x = [t["addr"] for t in ordered]
        teach_h = [t["handle"] for t in ordered]
        h0 = handles[0]
        train_scores = [score_addr(arm, ag, x, bank) for x in teach_x] if arm != "B4" else []
        train_rank = bool(train_scores) and all(
            winner_of(sc) == h for sc, h in zip(train_scores, teach_h)
        )
        probes: list[dict[str, Any]] = []
        probe_x: list[np.ndarray] = []
        ranking_ok = True
        all_ok = True
        for i, (cue, handle) in enumerate(pairs):
            probe = clone_frozen(ag)
            require_query(probe)
            observe_cue(probe, world, tag=f"{tag}_p{i}", body=list(MID_BODY), symbols=[cue])
            addr = probe_address(arm, probe, tracer)
            probe_x.append(addr)
            if arm == "B4":
                continue
            scores = score_addr(arm, ag, addr, bank)
            win = winner_of(scores)
            margin = ranking_margin(scores, win or "") if win else 0.0
            stab = perturb_rank(
                lambda u, a=arm, g=ag, b=bank: score_addr(a, g, u, b),
                addr,
                win or "",
                domain=world["domain"],
                key=f"{tag}_{cue}",
            )
            ok = bool(win == handle and margin >= gmin and stab["stable"])
            ranking_ok = ranking_ok and bool(win == handle)
            all_ok = all_ok and ok
            probes.append(
                {
                    "cue": cue,
                    "want": handle,
                    "winner": win,
                    "margin": float(margin),
                    "perturb_stable": bool(stab["stable"]),
                    "ok": ok,
                }
            )
        if arm == "B4":
            y = np.asarray([1.0 if h == h0 else -1.0 for h in teach_h])
            X_tr = np.stack([unit_or_zero(x) for x in teach_x])
            X_te = np.stack([unit_or_zero(x) for x in probe_x])
            if X_tr.shape[0] != len(pairs) or X_te.shape[0] != len(pairs):
                raise RuntimeError("probe rows passed to fit")
            fit = hard_margin_linear(X_tr, y, spec)
            require_accepted(fit["status"] if fit["status"] in ACCEPTED_STATUS else "error")
            w, b, status = fit["w"], fit["b"], fit["status"]
            probe_y = np.asarray([1.0 if h == h0 else -1.0 for _c, h in pairs])
            train_g = min_geometric_margin(w, b, X_tr, y)
            probe_g = min_geometric_margin(w, b, X_te, probe_y)
            train_rank = bool(len(y) and np.all((X_tr @ w + b) * y > 0.0))
            ranking_ok = bool(len(probe_y) and np.all((X_te @ w + b) * probe_y > 0.0))
            gmin_g = float(p["margin"]["geometric_margin_min"])
            stab = {"stable": False, "n_ok": 0}
            if status == "optimal":
                stab = perturb_sign_stable(w, b, X_te, probe_y, domain=world["domain"], key=f"{tag}_d1")
            all_ok = bool(
                status == "optimal"
                and train_rank
                and ranking_ok
                and train_g >= gmin_g
                and probe_g >= gmin_g
                and stab["stable"]
            )
            probes = [
                {
                    "cue": c,
                    "want": h,
                    "winner": h0 if (unit_or_zero(x) @ w + b) > 0 else handles[1],
                    "margin": float(probe_g),
                    "perturb_stable": bool(stab["stable"]),
                    "ok": all_ok,
                }
                for (c, h), x in zip(pairs, probe_x)
            ]
            return {
                "passed": all_ok,
                "ranking_ok": ranking_ok,
                "train_ranking_ok": train_rank,
                "train_geometric_margin": float(train_g),
                "probe_geometric_margin": float(probe_g),
                "solver_status": status,
                "n_sv": int(fit["n_sv"]),
                "perturb_stable": bool(stab["stable"]),
                "taught": [{"cue": t["cue"], "handle": t["handle"], "adv": t["adv"]} for t in ordered],
                "probes": probes,
                "n_train": len(teach_x),
                "n_probe": len(probe_x),
            }
    return {
        "passed": all_ok,
        "ranking_ok": ranking_ok,
        "train_ranking_ok": train_rank,
        "perturb_stable": bool(all(q["perturb_stable"] for q in probes)) if probes else False,
        "min_probe_margin": float(min(q["margin"] for q in probes)) if probes else 0.0,
        "taught": [{"cue": t["cue"], "handle": t["handle"], "adv": t["adv"]} for t in ordered],
        "probes": probes,
        "n_train": len(teach_x),
        "n_probe": len(pairs),
    }


def eval_ecological(arm: str, world: dict[str, Any], *, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1 = world["handles"][0]
    h2 = world["handles"][1]
    p = load_prereg()
    gmin = float(p["margin"]["native_ranking_min"])
    with tempfile.TemporaryDirectory(prefix="tb_eco_") as tmp:
        ag = _fresh(tmp, "s", world)
        tracer = EventBoundaryTrace(64, list(world["handles"]), float(p["arms"]["B2"]["lambda"])) if arm == "B2" else None
        bank = (
            CompetitiveBank(list(world["handles"]), eta=float(p["arms"]["B3"]["eta"]), c_max=float(p["arms"]["B3"]["c_max"]))
            if arm == "B3"
            else None
        )
        t1 = teach_bridged(ag, world, cue, h1, arm=arm, tracer=tracer, bank=bank, tag=f"{tag}_p")
        wneg = set_handle_delta(world, h1, NEG_DELTA)
        t2 = teach_bridged(ag, wneg, cue, h1, arm=arm, tracer=tracer, bank=bank, tag=f"{tag}_n")
        t3 = teach_bridged(ag, world, cue, h2, arm=arm, tracer=tracer, bank=bank, tag=f"{tag}_r")
        block_probe = clone_frozen(ag)
        observe_cue(block_probe, world, tag=f"{tag}_q", body=list(MID_BODY), symbols=[cue])
        addr = probe_address(arm, block_probe, tracer)
        if arm == "B4":
            X = np.stack([unit_or_zero(t["addr"]) for t in (t1, t2, t3)])
            y = np.asarray([1.0, 1.0, -1.0])
            fit = hard_margin_linear(X, y, d1_spec())
            w, b, status = fit["w"], fit["b"], fit["status"]
            pred = float(unit_or_zero(addr) @ w + b)
            win = h1 if pred > 0 else h2
            ok = bool(status == "optimal" and win == h2)
            return {
                "passed": ok,
                "required": False,
                "adv": [t1["adv"], t2["adv"], t3["adv"]],
                "winner": win,
                "want": h2,
                "solver_status": status,
            }
        scores = score_addr(arm, ag, addr, bank)
        win = winner_of(scores)
        margin = ranking_margin(scores, win or "") if win else 0.0
        stab = perturb_rank(
            lambda u, a=arm, g=ag, b=bank: score_addr(a, g, u, b),
            addr,
            win or "",
            domain=world["domain"],
            key=f"{tag}_eco",
        )
        ok = bool(
            t1["adv"] > 0.0
            and t2["adv"] < 0.0
            and t3["adv"] > 0.0
            and win == h2
            and margin >= gmin
            and stab["stable"]
        )
    return {
        "passed": ok,
        "required": True,
        "adv": [float(t1["adv"]), float(t2["adv"]), float(t3["adv"])],
        "winner": win,
        "want": h2,
        "margin": float(margin),
        "perturb_stable": bool(stab["stable"]),
    }


def eval_hold(arm: str, world: dict[str, Any], *, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="tb_hold_") as tmp:
        ag = _fresh(tmp, "s", world)
        tracer = EventBoundaryTrace(64, list(world["handles"]), float(p["arms"]["B2"]["lambda"])) if arm == "B2" else None
        require_query(ag)
        wq0 = ag.W_act_query.detach().clone()
        observe_cue(ag, world, tag=f"{tag}_sel", body=list(MID_BODY), symbols=[cue])
        addr = select_address(arm, ag, tracer)
        clamped = ag.clamp_action("HOLD", None)
        if not clamped.get("ok"):
            raise RuntimeError(f"HOLD clamp failed {clamped}")
        if arm in V29_ARMS:
            inject_pending(ag, addr)
        observe_cue(ag, world, tag=f"{tag}_obs", body=list(MID_BODY), symbols=[cue])
        prep_eval(ag)
        d_wq = float((ag.W_act_query - wq0).abs().max().item())
    return {"passed": bool(d_wq <= 1e-12), "required": arm in V29_ARMS or arm == "B3", "d_w_act_query": d_wq}


def decorate(
    out: dict[str, Any], *, kind: str, arm: str, n_cues: int, order: str, world: int, domain: str
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
        }
    )
    return out


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    def rows(arm: str, *, kind: str | None = None, n_cues: int | None = None) -> list[dict[str, Any]]:
        out = [c for c in cells if c["arm"] == arm]
        if kind is not None:
            out = [c for c in out if c["kind"] == kind]
        if n_cues is not None:
            out = [c for c in out if c["n_cues"] == n_cues]
        return out

    def robust(arm: str) -> bool:
        rank = rows(arm, kind="rank")
        twin = rows(arm, kind="twin")
        extra = rows(arm, kind="eco") + rows(arm, kind="hold") + rows(arm, kind="rest")
        need = rank + twin
        if arm != "B4":
            need = need + extra
        return bool(need) and all(bool(c["passed"]) for c in need)

    def train_only(arm: str) -> bool:
        rank8 = rows(arm, kind="rank", n_cues=8)
        return bool(rank8) and all(bool(c.get("train_ranking_ok")) for c in rank8) and not all(
            bool(c["passed"]) for c in rank8
        )

    flags = {
        arm: {
            "robust": robust(arm),
            "train_only_8cue": train_only(arm),
            "rank8_pass": all(bool(c["passed"]) for c in rows(arm, kind="rank", n_cues=8))
            if rows(arm, kind="rank", n_cues=8)
            else False,
        }
        for arm in ARMS
    }
    ladder = p["decision_ladder"]
    if not flags["B4"]["robust"]:
        return ladder[0]["id"], ladder[0]["then"], {"flags": flags}
    if flags["B1"]["train_only_8cue"] and not flags["B1"]["robust"]:
        return ladder[1]["id"], ladder[1]["then"], {"flags": flags}
    if flags["B1"]["robust"] and flags["B2"]["robust"]:
        return ladder[2]["id"], ladder[2]["then"], {"flags": flags}
    if flags["B1"]["robust"] and not flags["B2"]["robust"]:
        return ladder[3]["id"], ladder[3]["then"], {"flags": flags}
    if (not flags["B1"]["robust"]) and flags["B3"]["robust"]:
        return ladder[4]["id"], ladder[4]["then"], {"flags": flags}
    return ladder[5]["id"], ladder[5]["then"], {"flags": flags}


def run_dev() -> dict[str, Any]:
    refuse_v31()
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        for spec in p["capacity"]:
            n_cues = int(spec["n_cues"])
            for wi in range(2):
                world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
                if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
                    raise RuntimeError("SCORE identifier appeared in DEV payload")
                pairs = mapping_pairs(world, flip=False)
                for order in TEACH_ORDERS:
                    block = eval_learner_block(
                        arm=arm, world=world, pairs=pairs, order=order, tag=f"tb_{arm}_{wi}_{n_cues}_{order}"
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
                        )
                    )
        world_t = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
        world_t["purpose"] = "rename_twin"
        if SCORE_DOMAIN in world_t["domain"] or "SCORE." in world_t["domain"]:
            raise RuntimeError("SCORE identifier appeared in DEV payload")
        pairs_t = mapping_pairs(world_t, flip=False)
        for order in TEACH_ORDERS:
            block = eval_learner_block(
                arm=arm, world=world_t, pairs=pairs_t, order=order, tag=f"tb_{arm}_twin_{order}"
            )
            cells.append(
                decorate(block, kind="twin", arm=arm, n_cues=2, order=order, world=1, domain=world_t["domain"])
            )
        world_c = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
        pairs_c = mapping_pairs(world_c, flip=False)
        eco = eval_ecological(arm, world_c, tag=f"tb_{arm}_eco")
        eco["kind"] = "eco"
        cells.append(decorate(eco, kind="eco", arm=arm, n_cues=2, order="A_then_B", world=0, domain=world_c["domain"]))
        hold = eval_hold(arm, world_c, tag=f"tb_{arm}_hold")
        cells.append(decorate(hold, kind="hold", arm=arm, n_cues=2, order="A_then_B", world=0, domain=world_c["domain"]))
        rest = eval_learner_block(
            arm=arm, world=world_c, pairs=pairs_c, order="A_then_B", tag=f"tb_{arm}_rest", rest=True
        )
        cells.append(decorate(rest, kind="rest", arm=arm, n_cues=2, order="A_then_B", world=0, domain=world_c["domain"]))
    ids = [c["id"] for c in cells]
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    for c in cells:
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        if c["kind"] in ("rank", "twin", "rest") and (
            int(c.get("n_train") or 0) != int(c["n_cues"]) or int(c.get("n_probe") or 0) != int(c["n_cues"])
        ):
            raise RuntimeError(f"empty teach/probe {c['id']}")
    code, then, extra = _decision(cells, p)
    flags = extra["flags"]
    out = {
        "version": "TM.0.24.TRACEBRIDGE.DEV",
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
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "arm_robust": {arm: flags[arm]["robust"] for arm in ARMS},
        "arm_train_only_8cue": {arm: flags[arm]["train_only_8cue"] for arm in ARMS},
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells),
        "cells": cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": tracebridge_shas(),
        "note": "TRACEBRIDGE DEV only. Write-geometry closed. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def smoke() -> dict[str, Any]:
    p = load_prereg()
    world = capacity_world(0, "TM024.TRACEBRIDGE.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    b1 = eval_learner_block(arm="B1", world=world, pairs=pairs, order="A_then_B", tag="tbsmk")
    hold = eval_hold("B1", world, tag="tbsmkh")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "b1_n_train": b1["n_train"],
        "b1_train_ranking_ok": b1["train_ranking_ok"],
        "hold_d_wq": hold["d_w_act_query"],
        "expected_n_cells": EXPECTED_N_CELLS,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "act_score_mode": p["act_score_mode"],
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "env": torch_env(),
    }


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("tracebridge runner.lock already exists")
    refuse_v31()
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.TRACEBRIDGE.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "shas": tracebridge_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "arms": list(ARMS),
        "b2_lambda": prereg["arms"]["B2"]["lambda"],
        "b2_rows": prereg["arms"]["B2"]["rows"],
        "trace_budget_unopened": 512,
        "declared_budget_remains_closed": 1536,
        "expected_n_cells": EXPECTED_N_CELLS,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen TRACEBRIDGE runner. DEV lock only after this file is on origin/main. No neural edit.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    refuse_rerun()
    refuse_score_markers(json.dumps(out, default=str))
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("tracebridge decision lock already exists")
    out = {
        "version": "TM.0.24.TRACEBRIDGE.DECISION",
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
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "arm_robust": dev.get("arm_robust"),
            "arm_train_only_8cue": dev.get("arm_train_only_8cue"),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "Runner-only P1 trace sufficiency. Write-geometry closed. "
            "No v31. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.TRACEBRIDGE DEV\n\n"
        f"Decision: **{out['decision']['code']}**.\n\n"
        f"Arm robust: `{out['decision']['arm_robust']}`.\n\n"
        "Write-geometry closed. SCORE unopened. No neural candidate. "
        "512/1536 budgets stay closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze installs a sufficient P1 trace")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("TRACEBRIDGE DEV lock requires runner.lock on origin/main after this freeze")
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
        assert p["arms"]["B2"]["lambda"] == 0.0
        assert p["declared_budget_if_later_authorized"]["event_end_trace_rows"] == 512
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
