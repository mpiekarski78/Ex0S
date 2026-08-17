"""TM.0.24.ELIGMAP — runner-only eligibility-address diagnostic.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
DEV on unused TM024.ELIGMAP.DEV. worlds. SCORE reserved and unopened.
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
from experiments.run_tm024actorcredit import MID_BODY, observe_cue, prep_eval
from experiments.run_tm024collisionmap import parse_stages
from experiments.run_tm024motorpersist import TEACH_ORDERS
from experiments.run_tm024writegeom import (
    NEG_DELTA,
    capacity_world,
    mapping_pairs,
    set_handle_delta,
)
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_eligmap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_eligmap_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_eligmap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_eligmap.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_eligmap.decision.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_eligmap.dev.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024eligmap_results.md"
ADDENDUM = REPO_ROOT / "docs" / "lineage_writegeom.decision.addendum.lock"
WG_DECISION = REPO_ROOT / "docs" / "lineage_writegeom.decision.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"
CANDIDATE_V32 = REPO_ROOT / "docs" / "cortex.candidate.v32.lock"

DEV_DOMAIN = "TM024.ELIGMAP.DEV."
TWIN_DOMAIN = "TM024.ELIGMAP.TWIN."
SCORE_DOMAIN = "TM024.ELIGMAP.SCORE."
EPS = 1e-12


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def eligmap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "writegeom_decision": WG_DECISION,
        "writegeom_addendum": ADDENDUM,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def unit_or_zero(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        return np.zeros_like(v)
    return v / n


def clipnorm(x: np.ndarray, c_max: float) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        return np.zeros_like(v)
    if n > float(c_max):
        return v * (float(c_max) / n)
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def ranking_margin(scores: dict[str, float], winner: str) -> float:
    if winner not in scores:
        return 0.0
    others = [float(v) for k, v in scores.items() if k != winner]
    if not others:
        return float(scores[winner])
    return float(scores[winner] - max(others))


def distinct_pair(a: np.ndarray, b: np.ndarray, sep: dict[str, Any]) -> bool:
    return bool(cosine(a, b) < float(sep["cos_distinct_max"]) or l2(a, b) > float(sep["l2_distinct_min"]))


def copy_traj(ag: NeuralCortex) -> list[np.ndarray]:
    return [np.asarray(x, dtype=np.float64).copy() for x in ag.last_trajectory]


def snapshot_tick(ag: NeuralCortex, *, kind: str, cue: str | None, handle: str | None, adv: float) -> dict[str, Any]:
    stages = parse_stages(ag)
    return {
        "kind": kind,
        "cue": cue,
        "handle": handle,
        "adv": float(adv),
        "rho_elig": np.asarray(stages["rho_elig"], dtype=np.float64).copy(),
        "observable": np.asarray(stages["observable"], dtype=np.float64).copy(),
        "trajectory": copy_traj(ag),
    }


class EligTrace:
    """Ungated leaky connection-local presynaptic state. Rows may be identical."""

    def __init__(self, n: int, handles: list[str], lam: float):
        self.lam = float(lam)
        self.handles = list(handles)
        z = np.zeros(int(n), dtype=np.float64)
        self.e = {h: z.copy() for h in self.handles}

    def copy(self) -> "EligTrace":
        out = EligTrace(len(next(iter(self.e.values()))), self.handles, self.lam)
        out.e = {h: v.copy() for h, v in self.e.items()}
        return out

    def ingest(self, trajectory: list[np.ndarray]) -> None:
        lam = self.lam
        om = 1.0 - lam
        for rho in trajectory:
            r = np.asarray(rho, dtype=np.float64).reshape(-1)
            for h in self.handles:
                self.e[h] = lam * self.e[h] + om * r

    def address(self) -> np.ndarray:
        return next(iter(self.e.values())).copy()


class ProtoBank:
    def __init__(self, handles: list[str], law: str, *, eta: float, c_max: float):
        self.handles = list(handles)
        self.law = str(law)
        self.eta = float(eta)
        self.c_max = float(c_max)
        z = np.zeros(64, dtype=np.float64)
        self.w = {h: z.copy() for h in self.handles}
        self.c = {h: 0.0 for h in self.handles}

    def update(self, handle: str, adv: float, e: np.ndarray) -> None:
        if handle not in self.w or float(adv) == 0.0:
            return
        ehat = unit_or_zero(e)
        if float(np.linalg.norm(ehat)) <= EPS:
            return
        eta = self.eta
        if self.law == "N0":
            z = self.w[handle] + eta * float(adv) * ehat
            self.w[handle] = unit_or_zero(z)
        elif self.law == "N1":
            z = self.w[handle] + eta * float(adv) * ehat
            self.w[handle] = clipnorm(z, self.c_max)
        elif self.law == "N2":
            if float(np.linalg.norm(self.w[handle])) <= EPS:
                if float(adv) <= 0.0:
                    return
                self.w[handle] = ehat.copy()
                self.c[handle] = min(self.c_max, eta * float(adv))
                return
            conf = max(0.0, min(self.c_max, self.c[handle] + eta * float(adv)))
            if conf <= EPS:
                self.w[handle] = np.zeros(64, dtype=np.float64)
                self.c[handle] = 0.0
            else:
                if float(adv) > 0.0:
                    self.w[handle] = ehat.copy()
                self.c[handle] = conf
        else:
            raise RuntimeError(f"unknown write law {self.law}")

    def scores(self, e: np.ndarray) -> dict[str, float]:
        ehat = unit_or_zero(e)
        out: dict[str, float] = {}
        for h in self.handles:
            w = self.w[h]
            wn = float(np.linalg.norm(w))
            if self.law == "N0":
                out[h] = 0.0 if wn <= EPS else cosine(w, ehat)
            elif self.law == "N1":
                out[h] = 0.0 if wn <= EPS else float(np.dot(w, ehat))
            else:
                out[h] = 0.0 if wn <= EPS else float(self.c[h]) * cosine(w, ehat)
        return out


def address_ids(prereg: dict[str, Any] | None = None) -> list[str]:
    p = prereg or load_prereg()
    ids = ["E0", "E1", "Edelta"]
    for lam in p["elam"]["lambda_grid"]:
        ids.append(f"Elam_{lam}")
    return ids


def parse_lam(aid: str) -> float | None:
    if not aid.startswith("Elam_"):
        return None
    return float(aid.split("Elam_", 1)[1])


def tick_static(tick: dict[str, Any], aid: str) -> np.ndarray:
    e0 = np.asarray(tick["rho_elig"], dtype=np.float64)
    e1 = np.asarray(tick["observable"], dtype=np.float64)
    if aid == "E0":
        return e0
    if aid == "E1":
        return e1
    if aid == "Edelta":
        return e1 - e0
    raise RuntimeError(f"not a static address {aid}")


def unit_norm_negative_inert(alpha: float = 0.4) -> bool:
    rho = np.arange(64, dtype=np.float64) + 1.0
    rho = unit_or_zero(rho)
    proto = rho.copy()
    z = proto - alpha * rho
    out = unit_or_zero(z)
    return bool(np.allclose(out, rho, atol=1e-12))


def n1_negative_shrinks(alpha: float = 0.4) -> bool:
    rho = np.arange(64, dtype=np.float64) + 1.0
    rho = unit_or_zero(rho)
    proto = rho.copy()
    z = clipnorm(proto - alpha * rho, 1.0)
    return bool(float(np.linalg.norm(z)) < 1.0 - 1e-9 and cosine(z, rho) > 0.99)


def record_observe(
    ag: NeuralCortex,
    world: dict[str, Any],
    *,
    kind: str,
    tag: str,
    cue: str | None,
    handle: str | None = None,
    body: list[float] | None = None,
) -> dict[str, Any]:
    symbols = [cue] if cue else []
    observe_cue(ag, world, tag=tag, body=list(body if body is not None else MID_BODY), symbols=symbols)
    return snapshot_tick(ag, kind=kind, cue=cue, handle=handle, adv=0.0)


def teach_with_adv(
    ag: NeuralCortex,
    world: dict[str, Any],
    cue: str,
    handle: str,
    *,
    tag: str,
) -> tuple[list[dict[str, Any]], float]:
    sel = record_observe(ag, world, kind="select", tag=f"{tag}_sel", cue=cue, handle=handle)
    ag.clamp_action("ACT", handle)
    _, body2 = physics(list(MID_BODY), handle, world["latent"])
    cred_out = observe_cue(
        ag, world, tag=f"{tag}_obs", body=list(body2), symbols=[cue]
    )
    cred = snapshot_tick(ag, kind="credit", cue=cue, handle=handle, adv=0.0)
    prep_eval(ag)
    adv = float((cred_out.get("metrics") or {}).get("adv") or 0.0)
    sel = dict(sel)
    sel["adv"] = adv
    return [sel, cred], adv


def record_rest(ag: NeuralCortex, *, n_ticks: int, tag: str) -> list[dict[str, Any]]:
    fatigued = list(MID_BODY)
    if len(fatigued) >= 4:
        fatigued[3] = min(1.0, float(fatigued[3]) + 0.3)
    ticks: list[dict[str, Any]] = []
    ag._resting = True
    try:
        for i in range(int(n_ticks)):
            ag.observe(
                {
                    "interaction_token": f"{tag}_{ag._t}_{i}",
                    "source_token": "src_rest",
                    "ordered_symbols": [],
                    "observable_state": ["st_idle"],
                    "body_state": fatigued,
                }
            )
            ticks.append(snapshot_tick(ag, kind="rest", cue=None, handle=None, adv=0.0))
    finally:
        ag._resting = False
    ag.reset_rho()
    ag.dev_epoch += 1
    return ticks


def unused_cues(world: dict[str, Any]) -> list[str]:
    used = {str(m["cue"]) for m in world["cue_handle"]}
    return [s for s in world["symbols"] if s not in used]


def perturb_stable(
    bank: ProtoBank,
    e: np.ndarray,
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
    r_hat = unit_or_zero(e)
    n_ok = 0
    for _i in range(n):
        noise = rng.normal(0.0, sigma, size=r_hat.shape)
        unit = unit_or_zero(r_hat + noise)
        scores = bank.scores(unit)
        if not scores:
            continue
        ranked = max(scores, key=lambda h: scores[h])
        if ranked == winner:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def pass_margin(margin: float, stable: bool) -> bool:
    m = load_prereg()["margin"]
    return bool(margin >= float(m["native_score_margin_min"]) and stable)


def pairwise_sep(vecs: dict[str, np.ndarray]) -> dict[str, Any]:
    sep = load_prereg()["separation"]
    keys = list(vecs)
    if len(keys) < 2:
        return {"n_pairs": 0, "all_distinct": False, "min_cosine": 1.0, "min_l2": 0.0}
    min_cos = 1.0
    min_l = float("inf")
    n_dist = 0
    n_pairs = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            n_pairs += 1
            c = cosine(vecs[a], vecs[b])
            d = l2(vecs[a], vecs[b])
            min_cos = min(min_cos, c)
            min_l = min(min_l, d)
            if distinct_pair(vecs[a], vecs[b], sep):
                n_dist += 1
    return {
        "n_pairs": n_pairs,
        "n_distinct": n_dist,
        "all_distinct": n_dist == n_pairs,
        "min_cosine": float(min_cos),
        "min_l2": float(min_l if min_l < float("inf") else 0.0),
    }


def replay_address(
    ticks: list[dict[str, Any]],
    probe_ticks: list[dict[str, Any]],
    *,
    aid: str,
    law: str,
    handles: list[str],
    domain: str,
    tag: str,
    want: list[tuple[str, str]],
) -> dict[str, Any]:
    p = load_prereg()
    eta = float(p["negative_write"]["eta"])
    c_max = float(p["negative_write"]["c_max"])
    lam = parse_lam(aid)
    tracer = EligTrace(64, handles, lam if lam is not None else 0.0)
    bank = ProtoBank(handles, law, eta=eta, c_max=c_max)

    def ingest_and_addr(tick: dict[str, Any], tr: EligTrace) -> np.ndarray:
        tr.ingest(tick["trajectory"])
        if lam is not None:
            return tr.address()
        return tick_static(tick, aid)

    for tick in ticks:
        e = ingest_and_addr(tick, tracer)
        if tick["kind"] == "select" and tick.get("handle") and float(tick.get("adv") or 0.0) != 0.0:
            bank.update(str(tick["handle"]), float(tick["adv"]), e)

    probes = []
    cue_addr: dict[str, np.ndarray] = {}
    all_ok = True
    ranking_ok = True
    for i, (cue, handle) in enumerate(want):
        pt = probe_ticks[i]
        tr = tracer.copy()
        e = ingest_and_addr(pt, tr)
        cue_addr[cue] = e.copy()
        scores = bank.scores(e)
        winner = max(scores, key=lambda h: scores[h]) if scores else None
        margin = ranking_margin(scores, winner) if winner else 0.0
        stab = perturb_stable(bank, e, winner or "", domain=domain, key=f"{tag}_{aid}_{cue}")
        ok = bool(winner == handle and pass_margin(margin, bool(stab["stable"])))
        all_ok = all_ok and ok
        ranking_ok = ranking_ok and bool(winner == handle)
        probes.append(
            {
                "cue": cue,
                "want": handle,
                "winner": winner,
                "margin": float(margin),
                "perturb_stable": bool(stab["stable"]),
                "ok": ok,
                "scores": {k: float(v) for k, v in scores.items()},
            }
        )
    sep = pairwise_sep(cue_addr)
    return {
        "address": aid,
        "law": law,
        "passed": all_ok,
        "ranking_ok": ranking_ok,
        "probes": probes,
        "separation": sep,
        "trace_separates": bool(sep["all_distinct"]),
    }


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    return ag


def collect_stream(
    world: dict[str, Any],
    teach_pairs: list[tuple[str, str]],
    *,
    tag: str,
    probe_pairs: list[tuple[str, str]] | None = None,
    rest: bool = False,
    distractor: bool = False,
    event_probe: bool = False,
) -> dict[str, Any]:
    probe_pairs = list(probe_pairs if probe_pairs is not None else teach_pairs)
    with tempfile.TemporaryDirectory(prefix="em_rec_") as tmp:
        ag = _fresh(tmp, "s", world)
        ticks: list[dict[str, Any]] = []
        taught = []
        for i, (cue, handle) in enumerate(teach_pairs):
            chunk, adv = teach_with_adv(ag, world, cue, handle, tag=f"{tag}_t{i}")
            ticks.extend(chunk)
            taught.append({"cue": cue, "handle": handle, "adv": adv})
        extra: list[dict[str, Any]] = []
        if rest:
            extra.extend(record_rest(ag, n_ticks=4, tag=f"{tag}_rest"))
        if distractor:
            extras = unused_cues(world)[:2]
            for j, d in enumerate(extras):
                extra.append(
                    record_observe(ag, world, kind="distractor", tag=f"{tag}_d{j}", cue=d)
                )
        probes = []
        for i, (cue, _h) in enumerate(probe_pairs):
            ptag = f"{tag}_evt{i}" if event_probe else f"{tag}_p{i}"
            probes.append(record_observe(ag, world, kind="probe", tag=ptag, cue=cue))
    return {"ticks": ticks, "extra": extra, "probes": probes, "taught": taught}


def eval_capacity_cell(
    *,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    aid: str,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    rec = collect_stream(world, seq, tag=tag, probe_pairs=pairs)
    out = replay_address(
        rec["ticks"],
        rec["probes"],
        aid=aid,
        law="N0",
        handles=list(world["handles"]),
        domain=world["domain"],
        tag=tag,
        want=list(pairs),
    )
    out.update(
        {
            "order": order,
            "n_cues": world["capacity"]["n_cues"],
            "n_handles": world["capacity"]["n_handles"],
            "taught_adv": [float(t["adv"]) for t in rec["taught"]],
            "purpose": world.get("purpose"),
        }
    )
    return out


def eval_survival(world: dict[str, Any], aid: str, *, tag: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    rest = collect_stream(world, pairs, tag=f"{tag}_r", rest=True)
    dist = collect_stream(world, pairs, tag=f"{tag}_d", distractor=True)
    evt = collect_stream(world, pairs, tag=f"{tag}_e", event_probe=True)
    handles = list(world["handles"])

    def run(rec: dict[str, Any], name: str) -> dict[str, Any]:
        ticks = list(rec["ticks"]) + list(rec["extra"])
        return replay_address(
            ticks,
            rec["probes"],
            aid=aid,
            law="N0",
            handles=handles,
            domain=world["domain"],
            tag=f"{tag}_{name}",
            want=pairs,
        )

    r_rest, r_dist, r_evt = run(rest, "rest"), run(dist, "dist"), run(evt, "evt")
    # permutation: keyed by opaque id — replay with reversed handle list in bank construction
    perm = replay_address(
        rest["ticks"],
        rest["probes"],
        aid=aid,
        law="N0",
        handles=list(reversed(handles)),
        domain=world["domain"],
        tag=f"{tag}_perm",
        want=pairs,
    )
    twin = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
    twin["purpose"] = "rename_twin"
    twin_pairs = mapping_pairs(twin, flip=False)
    twin_rec = collect_stream(twin, twin_pairs, tag=f"{tag}_twin")
    r_twin = replay_address(
        twin_rec["ticks"],
        twin_rec["probes"],
        aid=aid,
        law="N0",
        handles=list(twin["handles"]),
        domain=twin["domain"],
        tag=f"{tag}_twin",
        want=twin_pairs,
    )
    passed = bool(
        r_rest["passed"] and r_dist["passed"] and r_evt["passed"] and perm["passed"] and r_twin["passed"]
    )
    return {
        "id": "survival",
        "address": aid,
        "passed": passed,
        "rest": r_rest["passed"],
        "distractor": r_dist["passed"],
        "event": r_evt["passed"],
        "permutation": perm["passed"],
        "rename_twin": r_twin["passed"],
        "rest_ranking": r_rest["ranking_ok"],
        "twin_ranking": r_twin["ranking_ok"],
    }


def eval_negative(world: dict[str, Any], aid: str, law: str, *, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1 = world["handles"][0]
    h2 = world["handles"][1]
    wneg = set_handle_delta(world, h1, NEG_DELTA)
    with tempfile.TemporaryDirectory(prefix="em_neg_") as tmp:
        ag = _fresh(tmp, "s", world)
        t1, a1 = teach_with_adv(ag, world, cue, h1, tag=f"{tag}_p")
        t2, a2 = teach_with_adv(ag, wneg, cue, h1, tag=f"{tag}_n")
        t3, a3 = teach_with_adv(ag, world, cue, h2, tag=f"{tag}_r")
        probe = record_observe(ag, world, kind="probe", tag=f"{tag}_q", cue=cue)
    ticks = t1 + t2 + t3
    out = replay_address(
        ticks,
        [probe],
        aid=aid,
        law=law,
        handles=list(world["handles"]),
        domain=world["domain"],
        tag=tag,
        want=[(cue, h2)],
    )
    out.update(
        {
            "id": "ecological_reversal",
            "adv": [a1, a2, a3],
            "want": h2,
            "required": False,
        }
    )
    return out


def run_dev() -> dict[str, Any]:
    prereg = load_prereg()
    cells: list[dict[str, Any]] = []
    aids = address_ids(prereg)
    for spec in prereg["capacity"]:
        n_cues, n_handles = int(spec["n_cues"]), int(spec["n_handles"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            world["purpose"] = spec["purpose"]
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                rec = collect_stream(
                    world,
                    list(reversed(pairs)) if order == "B_then_A" else list(pairs),
                    tag=f"cap{wi}_{n_cues}_{order}",
                    probe_pairs=pairs,
                )
                for aid in aids:
                    want = list(pairs)
                    seq_ticks = rec["ticks"]
                    out = replay_address(
                        seq_ticks,
                        rec["probes"],
                        aid=aid,
                        law="N0",
                        handles=list(world["handles"]),
                        domain=world["domain"],
                        tag=f"cap{wi}_{n_cues}_{order}_{aid}",
                        want=want,
                    )
                    out.update(
                        {
                            "order": order,
                            "n_cues": n_cues,
                            "n_handles": n_handles,
                            "world": wi,
                            "required": bool(spec["required"]),
                            "purpose": spec["purpose"],
                            "taught_adv": [float(t["adv"]) for t in rec["taught"]],
                        }
                    )
                    cells.append(out)
    w0 = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    survival = []
    for aid in aids:
        survival.append(eval_survival(w0, aid, tag=f"surv_{aid}"))
    negative = []
    for aid in prereg["negative_write"]["reversal_battery_addresses"]:
        for law in ("N0", "N1", "N2"):
            negative.append(eval_negative(w0, aid, law, tag=f"neg_{law}_{aid}"))

    def _cap(aid: str, n_cues: int) -> list[dict[str, Any]]:
        return [
            c
            for c in cells
            if c["address"] == aid and c["n_cues"] == n_cues and c["n_handles"] == 2
        ]

    def _all_pass(rows: list[dict[str, Any]]) -> bool:
        return bool(rows) and all(bool(r["passed"]) for r in rows)

    def _any_rank(rows: list[dict[str, Any]]) -> bool:
        return any(bool(r.get("ranking_ok")) for r in rows)

    def _sep8(aid: str) -> bool:
        rows = _cap(aid, 8)
        return bool(rows) and all(bool(r.get("trace_separates")) for r in rows)

    surv_by = {s["address"]: s for s in survival}
    elam_ids = [a for a in aids if a.startswith("Elam_")]
    elam_robust = False
    elam_robust_id = None
    for aid in elam_ids:
        cap8 = _all_pass(_cap(aid, 8))
        surv = bool(surv_by.get(aid, {}).get("passed"))
        if cap8 and surv:
            elam_robust = True
            elam_robust_id = aid
            break
    e1_all = all(_all_pass(_cap("E1", n)) for n in (2, 4, 8))
    e0_all = all(_all_pass(_cap("E0", n)) for n in (2, 4, 8))
    ed_all = all(_all_pass(_cap("Edelta", n)) for n in (2, 4, 8))
    elam_any_rank8 = any(_all_pass(_cap(a, 8)) for a in elam_ids)
    elam_sep8 = any(_sep8(a) for a in elam_ids)
    e1_rank_only = e1_all and (not e0_all) and (not elam_any_rank8)
    n1_fix = any(bool(r["passed"]) and r["law"] in ("N1", "N2") for r in negative)
    n0_eco = any(bool(r["passed"]) and r["law"] == "N0" for r in negative)
    ladder = prereg["decision_ladder"]
    code = "no_address_passes"
    then = ladder[3]["then"]
    if elam_robust:
        code, then = ladder[0]["id"], ladder[0]["then"]
    elif elam_sep8 and not elam_any_rank8:
        code, then = ladder[1]["id"], ladder[1]["then"]
    elif e1_rank_only:
        code, then = ladder[2]["id"], ladder[2]["then"]
    elif not (e0_all or e1_all or ed_all or elam_any_rank8):
        code, then = ladder[3]["id"], ladder[3]["then"]
    elif n1_fix and not elam_robust:
        code, then = ladder[4]["id"], ladder[4]["then"]

    authorize_elig = bool(code == "elam_robust_8cue")
    return {
        "version": "TM.0.24.ELIGMAP.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "domain": DEV_DOMAIN,
        "score_domain_opened": False,
        "neural_edit": False,
        "implementation_authorized": authorize_elig,
        "elam_robust_8cue": elam_robust,
        "elam_robust_id": elam_robust_id,
        "e0_required_pass": e0_all,
        "e1_required_pass": e1_all,
        "edelta_required_pass": ed_all,
        "elam_any_8cue_pass": elam_any_rank8,
        "elam_sep_8cue": elam_sep8,
        "n1n2_reversal_any": n1_fix,
        "n0_reversal_any": n0_eco,
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(cells) + len(survival) + len(negative),
        "cells": cells,
        "survival": survival,
        "negative": negative,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": eligmap_shas(),
        "note": "DEV only. SCORE unopened. No neural edit. Product remains 0.0.004.",
    }


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    if DEV_LOCK.exists():
        raise RuntimeError("eligmap DEV lock already exists")
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("eligmap decision lock already exists")
    out = {
        "version": "TM.0.24.ELIGMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "scored_worlds": False,
        "neural_edit": False,
        "implementation_authorized": bool(dev.get("implementation_authorized")),
        "candidate_v31": False,
        "candidate_v32": False,
        "lineage_reopened": False,
        "q3": False,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "elam_robust_8cue": bool(dev.get("elam_robust_8cue")),
        },
        "writegeom_addendum_refined_code": "w1_query_margin_insufficient__unit_norm_negative_inert",
        "declared_budget_if_authorized": 1536,
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "Runner-only eligibility-address diagnostic. SCORE unopened. "
            "No v31/v32 candidate. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    brief = {
        "decision": out["decision"]["code"],
        "elam_robust": bool(dev.get("elam_robust_8cue")),
        "e1": bool(dev.get("e1_required_pass")),
        "e0": bool(dev.get("e0_required_pass")),
        "n1n2": bool(dev.get("n1n2_reversal_any")),
        "then": out["decision"]["then"],
    }
    RESULT_MD.write_text(
        "# TM.0.24.ELIGMAP DEV\n\n"
        f"Decision: **{brief['decision']}**. "
        f"Eλ robust 8-cue: **{brief['elam_robust']}**. "
        f"E1 required: **{brief['e1']}**. E0 required: **{brief['e0']}**. "
        f"N1/N2 reversal any: **{brief['n1n2']}**.\n\n"
        f"Next: `{out['decision']['then']}`. SCORE unopened. No neural candidate. "
        "Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    inert = unit_norm_negative_inert()
    shrink = n1_negative_shrinks()
    w = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory(prefix="em_smk_") as tmp:
        ag = _fresh(tmp, "s", w)
        a = record_observe(ag, w, kind="probe", tag="smk_a", cue=w["cue_handle"][0]["cue"])
        b = record_observe(ag, w, kind="probe", tag="smk_b", cue=w["cue_handle"][1]["cue"])
    e1_sep = distinct_pair(a["observable"], b["observable"], prereg["separation"])
    e0_sep = distinct_pair(a["rho_elig"], b["rho_elig"], prereg["separation"])
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "H_max": prereg["H_max"],
        "declared_budget_if_later_authorized": prereg["declared_budget_if_later_authorized"]["total"],
        "unit_norm_negative_inert": inert,
        "n1_negative_shrinks": shrink,
        "e1_birth_distinct": e1_sep,
        "e0_birth_distinct": e0_sep,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "v32_exists": CANDIDATE_V32.exists(),
        "env": torch_env(),
    }


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after a later freeze authorizes an eligibility law on origin/main")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
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
        assert p["implementation_authorized"] is False
        assert p["elam"]["lambda_grid"][-1] == 0.99
        assert p["declared_budget_if_later_authorized"]["total"] == 1536
        assert p["margin"]["cosine_margin_min"] == 0.01
        print(json.dumps({"ok": True, "product": p["product"]}, indent=2))
    elif args.dev:
        out = run_dev()
        brief = {k: v for k, v in out.items() if k not in ("cells", "survival", "negative")}
        print(json.dumps(brief, indent=2, default=str))
    elif args.write_dev_lock:
        out = run_dev()
        write_dev_lock(out)
        print(json.dumps({k: v for k, v in out.items() if k not in ("cells", "survival", "negative")}, indent=2, default=str))
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
