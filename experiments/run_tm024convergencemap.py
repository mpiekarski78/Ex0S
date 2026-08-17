"""TM.0.24.CONVERGENCEMAP — runner-only one-shot vs replay on exact P1.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Write-geometry closed. W1 not resurrected. Default v29 query scoring.
DEV on unused TM024.CONVERGENCEMAP.DEV. after this freeze is on origin/main.
SCORE reserved and unopened. No trace or neural candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.run_tm023cortex import torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024eligmap import _fresh, capacity_world, mapping_pairs, record_rest, unit_or_zero
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
PA_BISECT = 50


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


def cell_id(kind: str, arm: str, n_cues: int, order: str, world: int, cycles: int, exposure: str) -> str:
    return f"{kind}|{arm}|c{n_cues}|{order}|w{world}|k{cycles}|{exposure}"


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
    def __init__(self, handles: list[str], *, gamma: float):
        self.handles = list(handles)
        self.gamma = float(gamma)
        self.rows = {h: np.zeros(64, dtype=np.float64) for h in self.handles}
        self.n_updates = 0
        self.last_tau = 0.0

    def update(self, addr: np.ndarray, chosen: str, adv: float) -> None:
        if abs(float(adv)) <= EPS or chosen not in self.rows:
            self.last_tau = 0.0
            return
        x = unit_or_zero(addr)
        ch, ot = desired_pair(self.handles, chosen, adv)
        tau = min_tau_for_margin(self.rows[ch], self.rows[ot], x, self.gamma)
        self.last_tau = float(tau)
        if tau <= EPS:
            return
        self.rows[ch] = self.rows[ch] + tau * x
        self.rows[ot] = self.rows[ot] - tau * x
        self.n_updates += 1

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
    probe = clone_frozen(ag)
    require_query(probe)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    addr = probe_address("B3", probe, None)
    scores = learner.scores(addr)
    win = winner_of(scores)
    margin = ranking_margin(scores, win or "") if win else 0.0
    return {"cue": cue, "winner": win, "addr": addr, "scores": scores, "margin": float(margin)}


def eval_learner_block(
    *,
    arm: str,
    cycles: int,
    exposure: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool = False,
) -> dict[str, Any]:
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
    last_by_cue: dict[str, dict[str, Any]] = {}
    stored: list[dict[str, Any]] = []
    taught_cues: list[str] = []

    def mark_retention(ok: bool) -> None:
        nonlocal retention_ok, n_checkpoints
        n_checkpoints += 1
        retention_ok = retention_ok and bool(ok)

    with tempfile.TemporaryDirectory(prefix="cvg_") as tmp:
        ag = _fresh(tmp, "s", world)
        require_query(ag)
        if exposure == "live":
            for cy in range(int(cycles)):
                for i, (cue, handle) in enumerate(seq):
                    rec = teach_bridged(
                        ag, world, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_c{cy}_t{i}"
                    )
                    n_live += 1
                    learner.update(rec["addr"], rec["handle"], rec["adv"])
                    last_by_cue[cue] = rec
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
                stored.append(rec)
                last_by_cue[cue] = rec
            if rest:
                record_rest(ag, n_ticks=int(p["n_rest_ticks"]), tag=f"{tag}_rest")
                rest = False
            seen: list[str] = []
            for cy in range(int(cycles)):
                for rec in stored:
                    learner.update(rec["addr"], rec["handle"], rec["adv"])
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
    n_train = int(getattr(learner, "n_updates", n_live + int(cycles) * n_capture))
    return {
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
        "n_probe": len(pairs),
        "n_updates": int(getattr(learner, "n_updates", 0)),
        "cycles": int(cycles),
        "exposure": exposure,
    }


def eval_ecological(
    arm: str,
    cycles: int,
    exposure: str,
    world: dict[str, Any],
    *,
    tag: str,
) -> dict[str, Any]:
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
    with tempfile.TemporaryDirectory(prefix="cvg_eco_") as tmp:
        ag = _fresh(tmp, "s", world)

        def teach(w: dict[str, Any], handle: str, suffix: str) -> dict[str, Any]:
            return teach_bridged(ag, w, cue, handle, arm="B3", tracer=None, bank=None, tag=f"{tag}_{suffix}")

        def retain_live(want: str) -> None:
            nonlocal retention_ok, n_checkpoints
            n_checkpoints += 1
            pr = live_probe(ag, world, cue, learner, tag=f"{tag}_r{n_checkpoints}")
            retention_ok = retention_ok and pr["winner"] == want

        if exposure == "live":
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
            wants = [h1, h2, h2]
            seen: list[int] = []
            for cy in range(int(cycles)):
                for i, rec in enumerate(stored):
                    learner.update(rec["addr"], rec["handle"], rec["adv"])
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
    return {
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
        "exposure": exposure,
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
    cycles: int,
    exposure: str,
) -> dict[str, Any]:
    out.update(
        {
            "id": cell_id(kind, arm, n_cues, order, world, cycles, exposure),
            "kind": kind,
            "arm": arm,
            "n_cues": n_cues,
            "order": order,
            "world": world,
            "domain": domain,
            "cycles": int(cycles),
            "exposure": exposure,
            "required": True,
        }
    )
    return out


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    def rows(arm: str, cycles: int, exposure: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        out = [
            c
            for c in cells
            if c["arm"] == arm and int(c["cycles"]) == int(cycles) and c["exposure"] == exposure
        ]
        if kind is not None:
            out = [c for c in out if c["kind"] == kind]
        return out

    def robust(arm: str, cycles: int, exposure: str) -> bool:
        rank = rows(arm, cycles, exposure, kind="rank")
        twin = rows(arm, cycles, exposure, kind="twin")
        need = rank + twin
        if exposure == "live" or arm == "C4":
            need = need + rows(arm, cycles, exposure, kind="eco") + rows(arm, cycles, exposure, kind="rest")
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


def _append_map(
    cells: list[dict[str, Any]],
    *,
    arm: str,
    cycles: int,
    exposure: str,
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
                    exposure=exposure,
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"cvg_{arm}_{exposure}_{wi}_{n_cues}_{order}_k{cycles}",
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
                        exposure=exposure,
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
            exposure=exposure,
            world=world_t,
            pairs=pairs_t,
            order=order,
            tag=f"cvg_{arm}_{exposure}_twin_{order}_k{cycles}",
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
                exposure=exposure,
            )
        )
    if not include_eco_rest:
        return
    world_c = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    pairs_c = mapping_pairs(world_c, flip=False)
    eco = eval_ecological(arm, cycles, exposure, world_c, tag=f"cvg_{arm}_{exposure}_eco_k{cycles}")
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
            exposure=exposure,
        )
    )
    rest = eval_learner_block(
        arm=arm,
        cycles=cycles,
        exposure=exposure,
        world=world_c,
        pairs=pairs_c,
        order="A_then_B",
        tag=f"cvg_{arm}_{exposure}_rest_k{cycles}",
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
            exposure=exposure,
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
        _append_map(cells, arm=arm, cycles=k, exposure="live", include_eco_rest=True)
    for arm, k in replay_specs():
        _append_map(cells, arm=arm, cycles=k, exposure="replay", include_eco_rest=arm == "C4")
    ids = [c["id"] for c in cells]
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    for c in cells:
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        if c["kind"] in ("rank", "twin", "rest"):
            if int(c.get("n_probe") or 0) != int(c["n_cues"]):
                raise RuntimeError(f"empty probe {c['id']}")
            if c["exposure"] == "live" and int(c.get("n_live_teaches") or 0) != int(c["n_cues"]) * int(c["cycles"]):
                raise RuntimeError(f"live teach count {c['id']}")
            if c["exposure"] == "replay" and int(c.get("n_capture") or 0) != int(c["n_cues"]):
                raise RuntimeError(f"replay capture count {c['id']}")
        if not c.get("retention_ok", True) and c["passed"]:
            raise RuntimeError(f"passed without retention {c['id']}")
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
        "write_geometry_branch_closed": True,
        "w1_resurrected": False,
        "act_score_mode": "query",
        "live_robust": extra["live"],
        "replay_robust": extra["replay"],
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells),
        "cells": cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": convergencemap_shas(),
        "note": "CONVERGENCEMAP DEV only. Write-geometry closed. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def smoke() -> dict[str, Any]:
    p = load_prereg()
    refuse_pa_grid(p)
    world = capacity_world(0, "TM024.CONVERGENCEMAP.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    c0 = eval_learner_block(
        arm="C0", cycles=1, exposure="live", world=world, pairs=pairs, order="A_then_B", tag="cvgsmk0"
    )
    c2 = eval_learner_block(
        arm="C2", cycles=1, exposure="live", world=world, pairs=pairs, order="A_then_B", tag="cvgsmk2"
    )
    x = unit_or_zero(np.arange(64, dtype=np.float64) + 1.0)
    z = np.zeros(64, dtype=np.float64)
    tau0 = min_tau_for_margin(z, z, x, 0.01)
    w_ch = z + 0.01 * x
    w_ot = z - 0.01 * x
    tau_done = min_tau_for_margin(w_ch, w_ot, x, 0.01)
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "c0_n_live": c0["n_live_teaches"],
        "c0_retention_ok": c0["retention_ok"],
        "c2_n_live": c2["n_live_teaches"],
        "c2_n_updates": c2["n_updates"],
        "tau_zero_init": float(tau0),
        "tau_already_met": float(tau_done),
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
        "trace_budget_unopened": 512,
        "declared_budget_remains_closed": 1536,
        "expected_n_cells": EXPECTED_N_CELLS,
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
