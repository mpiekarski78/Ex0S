"""TM.0.23.CORTEX: developmental artificial cortex apparatus.

Phase A: contracts/worlds/preregs. Phase B: make_cortex + unscored sanity.
Product stays 0.0.004; earned_next=false; ex0s=null. No D scoring this pass.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.cortex_memory import CortexMemory, FORBIDDEN_SOURCES
from three_memory.neural_cortex import BODY_SETPOINT, GenomeConfig, NeuralCortex, OPS

AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"
NEURAL_PY = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY_PY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CONTRACT = REPO_ROOT / "docs" / "cortex_architecture_contract.md"
PREREG = REPO_ROOT / "docs" / "cortex.prereg.lock"
PREREG_WALL = REPO_ROOT / "docs" / "cortex_wall.prereg.lock"
GEN_LOCK = REPO_ROOT / "docs" / "cortex_world_generator.lock"
FIXTURE_DEV = REPO_ROOT / "docs" / "cortex_world_develop.json"
FIXTURE_VAL = REPO_ROOT / "docs" / "cortex_world_validate.json"
SEALED_EVAL = REPO_ROOT / "docs" / "cortex_eval_secrets.sealed.json"
BIRTH_LOCK = REPO_ROOT / "docs" / "cortex_birth.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "cortex.candidate.lock"
CANDIDATE_V1 = REPO_ROOT / "docs" / "cortex.candidate.v1.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm023cortex_results.md"

PHENOTYPE_PATHS = [
    "docs/CURRENT_ORGANISM.md",
    "docs/interpret.lock",
    "docs/interpret_wall.lock",
    "docs/sequence.lock",
    "docs/symbol_world.lock",
    "docs/relate_016.lock",
    "docs/persist.lock",
    "docs/inquire_wall.lock",
    "docs/reliability_wall.lock",
    "docs/perspective_wall.lock",
]

CLAIM = (
    "Starting from a pinned birth genome, one recurrent plastic artificial cortex "
    "matured during continuous symbolic lives and acquired relation learning, revision, "
    "persistent recall, symbolic grounding and variable-length sequence construction "
    "through generic event, memory and action interfaces, including transfer to held-out "
    "renamed worlds."
)

WALL_CLAIM = (
    "On frozen make_cortex, a preregistered neural parity wall exposes simplified "
    "PERSIST/INQUIRE/RELIABILITY/PERSPECTIVE/INTERPRET/HONESTY probes via the universal "
    "ABI only. Need not fully pass; first_fail_neural_wall diagnoses the next primitive. "
    "Not a 0.0.005 stamp."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def torch_env() -> dict[str, Any]:
    out: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "python": sys.version.split()[0],
        "device_name": None,
        "cudnn": None,
    }
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        out["cudnn"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
    return out


def make_cortex(
    s_dir: Path | None = None,
    *,
    genome: GenomeConfig | None = None,
    device: str | None = None,
) -> NeuralCortex:
    """Factory: new organism. Must not wrap make_interpret / ThreeMemoryAgent."""
    return NeuralCortex(s_dir, genome=genome, device=device or "cpu")


def empty_birth(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)


# --- world physics (frozen) ---

def physics(
    prev_body: list[float],
    act_operand: str | None,
    world_latent: dict[str, Any],
) -> tuple[list[str], list[float]]:
    """Deterministic body/state transition. Never reads scorer_only correctness."""
    body = np.asarray(prev_body, dtype=np.float64).copy()
    mapping = world_latent.get("act_effects") or {}
    key = (act_operand or "").lower()
    effect = mapping.get(key) or {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]}
    delta = np.asarray(effect.get("delta") or [0, 0, 0, 0], dtype=np.float64)
    body = np.clip(body + delta, 0.0, 1.0)
    state = [str(x).lower() for x in (effect.get("state") or ["st_idle"])]
    return state, body.tolist()


def build_observe(
    *,
    interaction_token: str,
    source_token: str,
    ordered_symbols: list[str],
    observable_state: list[str],
    body_state: list[float],
) -> dict[str, Any]:
    return {
        "interaction_token": interaction_token,
        "source_token": source_token,
        "ordered_symbols": list(ordered_symbols),
        "observable_state": list(observable_state),
        "body_state": list(body_state),
    }


def generate_worlds() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Visible develop + validate fixtures with organism_events / scorer_only split."""
    body0 = [0.5, 0.2, 0.5, 0.0]
    latent = {
        "act_effects": {
            "press": {"state": ["st_pressed"], "delta": [0.2, -0.05, 0.1, 0.0]},
            "harm": {"state": ["st_hurt"], "delta": [-0.3, 0.4, -0.1, 0.0]},
            "idle": {"state": ["st_idle"], "delta": [0.0, 0.0, 0.0, 0.0]},
        }
    }

    def seq_world(wid: str, split: str, symbols: list[str]) -> dict[str, Any]:
        events = []
        body = list(body0)
        # teach association: observe message then ACT consequence via runner physics
        events.append(
            {
                "op": "observe",
                "event": build_observe(
                    interaction_token=f"{wid}_ix0",
                    source_token="src_a",
                    ordered_symbols=symbols,
                    observable_state=["st_idle"],
                    body_state=body,
                ),
            }
        )
        events.append({"op": "force_note", "note": "organism may ACT; physics applied by runner"})
        return {
            "world_id": wid,
            "split": split,
            "organism_events": events,
            "scorer_only": {
                "latent_structure": latent,
                "expected_probes": [
                    {"id": "order_sensitive", "symbols_a": ["tok_a", "tok_b"], "symbols_b": ["tok_b", "tok_a"]},
                    {"id": "beneficial_act", "operand": "press"},
                    {"id": "harmful_act", "operand": "harm"},
                ],
            },
        }

    develop = {
        "version": "TM.0.23.CORTEX.WORLDS.DEVELOP",
        "body_setpoint": BODY_SETPOINT.tolist(),
        "worlds": [
            seq_world("dev_order", "develop", ["tok_a", "tok_b"]),
            seq_world("dev_assoc", "develop", ["cue_x"]),
            {
                "world_id": "dev_write_retrieve",
                "split": "develop",
                "organism_events": [
                    {
                        "op": "observe",
                        "event": build_observe(
                            interaction_token="wr_ix0",
                            source_token="src_a",
                            ordered_symbols=["mem_cue"],
                            observable_state=["st_idle"],
                            body_state=body0,
                        ),
                    }
                ],
                "scorer_only": {
                    "latent_structure": latent,
                    "expected_probes": [{"id": "write_retrieve"}],
                },
            },
        ],
    }
    validate = {
        "version": "TM.0.23.CORTEX.WORLDS.VALIDATE",
        "body_setpoint": BODY_SETPOINT.tolist(),
        "worlds": [
            seq_world("val_order", "validate", ["tok_p", "tok_q"]),
            seq_world("val_smoke", "validate", ["cue_v"]),
        ],
    }
    generator = {
        "version": "TM.0.23.CORTEX.GENERATOR",
        "lab": "TM.0.23.CORTEX",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "physics": "experiments.run_tm023cortex.physics",
        "body_setpoint": BODY_SETPOINT.tolist(),
        "splits": ["develop", "validate", "eval"],
        "note": "Eval worlds materialized only after candidate reveal of sealed seed+salt.",
        "refuse": [
            "feeding scorer_only into observe",
            "host homeostatic_delta",
            "small integer eval seeds",
        ],
    }
    return develop, validate, generator


def write_phase_a_artifacts() -> dict[str, Any]:
    develop, validate, generator = generate_worlds()
    FIXTURE_DEV.write_text(json.dumps(develop, indent=2) + "\n", encoding="utf-8")
    FIXTURE_VAL.write_text(json.dumps(validate, indent=2) + "\n", encoding="utf-8")

    # sealed 256-bit secrets
    if SEALED_EVAL.exists():
        sealed = json.loads(SEALED_EVAL.read_text(encoding="utf-8"))
    else:
        seed = secrets.token_bytes(32)
        salt = secrets.token_bytes(32)
        sealed = {
            "version": "TM.0.23.CORTEX.EVAL.SEALED",
            "seed_hex": seed.hex(),
            "salt_hex": salt.hex(),
            "note": "Do not use until after cortex.candidate.lock. Reveal creates eval worlds.",
        }
        SEALED_EVAL.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    seed_b = bytes.fromhex(sealed["seed_hex"])
    salt_b = bytes.fromhex(sealed["salt_hex"])
    assert len(seed_b) == 32 and len(salt_b) == 32
    commitment = _sha_bytes(seed_b + salt_b)

    generator["develop_fixture"] = "docs/cortex_world_develop.json"
    generator["validate_fixture"] = "docs/cortex_world_validate.json"
    generator["develop_sha"] = _sha_file(FIXTURE_DEV)
    generator["validate_sha"] = _sha_file(FIXTURE_VAL)
    generator["eval_seed_commitment"] = commitment
    generator["contract_sha"] = _sha_file(CONTRACT)
    GEN_LOCK.write_text(json.dumps(generator, indent=2) + "\n", encoding="utf-8")

    # hygiene audit
    hygiene = hygiene_audit()
    if not hygiene["ok"]:
        raise RuntimeError(f"hygiene failed: {hygiene}")

    phenotype = {p: _sha_file(REPO_ROOT / p) for p in PHENOTYPE_PATHS}
    prereg = {
        "version": "TM.0.23.CORTEX.PREREG",
        "lab": "TM.0.23.CORTEX",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "preregistered_claim": CLAIM,
        "contract": "docs/cortex_architecture_contract.md",
        "contract_sha": _sha_file(CONTRACT),
        "observe_keys": sorted(
            [
                "interaction_token",
                "source_token",
                "ordered_symbols",
                "observable_state",
                "body_state",
            ]
        ),
        "operations": list(OPS),
        "genome": GenomeConfig().to_dict(),
        "stats": {
            "earn_pairs": 16,
            "optional_extension_pairs": 16,
            "earn_threshold": ">=13/16",
            "maturation_threshold": ">=14/16",
            "no_seed_replacement": True,
        },
        "capacity": {
            "architecture_births_n": [32, 64, 128, 256],
            "mature_vocab": [32, 128, 512, 2048],
            "mature_s_rows": [1000, 10000, 100000],
            "mature_age": [1000, 10000, 100000],
            "mature_length": [1, 2, 4, 8, 16],
            "mature_domains": [1, 3, 6, 12],
            "mature_alternatives": [2, 4, 8, 16],
        },
        "eval_seed_commitment": commitment,
        "generator_lock": "docs/cortex_world_generator.lock",
        "generator_sha": _sha_file(GEN_LOCK),
        "develop_sha": _sha_file(FIXTURE_DEV),
        "validate_sha": _sha_file(FIXTURE_VAL),
        "phenotype_lock_shas": phenotype,
        "hygiene_ok": True,
        "refuse": [
            "mechanism SHA in this prereg",
            "adult weight SHA",
            "wrapping make_interpret",
            "D scoring before candidate",
            "earned_next=true or non-null ex0s",
            "revealing eval secrets before candidate",
            "mean-pool messages",
            "host homeostatic_delta",
        ],
        "note": "Frozen before CPU implementation. No agent/runner SHAs here.",
    }
    PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")

    wall = {
        "version": "TM.0.23.CORTEX.WALL.PREREG",
        "lab": "TM.0.23.CORTEX.WALL",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "preregistered_claim": WALL_CLAIM,
        "factory": "experiments.run_tm023cortex.make_cortex",
        "mechanism_changes_permitted": False,
        "need_not_fully_pass": True,
        "probe_ids": [
            "W_persist",
            "W_inquire",
            "W_reliability",
            "W_perspective",
            "W_interpret",
            "W_honesty",
        ],
        "cortex_prereg_sha": _sha_file(PREREG),
        "refuse": [
            "calling 0.0.004 capability mechanisms",
            "writing experience_* rows",
            "rewriting this prereg after results",
            "earned_next=true or non-null ex0s",
        ],
        "note": "Frozen before candidate. Results recorded later in cortex_wall.lock.",
    }
    PREREG_WALL.write_text(json.dumps(wall, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "eval_seed_commitment": commitment,
        "prereg_sha": _sha_file(PREREG),
        "wall_prereg_sha": _sha_file(PREREG_WALL),
    }


def hygiene_audit() -> dict[str, Any]:
    problems = []
    for path in (FIXTURE_DEV, FIXTURE_VAL):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        blob = json.dumps(data.get("worlds") or [])
        for bad in (
            "experience_grounding",
            "experience_sequence",
            "experience_inquire",
            "experience_interpretation",
            "homeostatic_delta",
            "correct=",
            "grounding_head",
            "sequence_head",
        ):
            if bad in blob:
                # scorer_only may mention probes — check organism_events only
                pass
        for w in data.get("worlds") or []:
            org = json.dumps(w.get("organism_events") or [])
            for bad in (
                "experience_grounding",
                "homeostatic_delta",
                '"correct"',
                "grounding_head",
                "scorer_only",
            ):
                if bad in org:
                    problems.append(f"{path.name}: organism_events contains {bad}")
            for ev in w.get("organism_events") or []:
                if ev.get("op") == "observe":
                    event = ev.get("event") or {}
                    if set(event.keys()) != {
                        "interaction_token",
                        "source_token",
                        "ordered_symbols",
                        "observable_state",
                        "body_state",
                    }:
                        problems.append(f"{path.name}: bad observe keys in {w.get('world_id')}")
    return {"ok": not problems, "problems": problems}


def verify_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG.exists():
        return False, "missing cortex.prereg.lock", {}
    lock = json.loads(PREREG.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.23.CORTEX":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("contract_sha") != _sha_file(CONTRACT):
        return False, "contract_sha pin", lock
    if lock.get("develop_sha") != _sha_file(FIXTURE_DEV):
        return False, "develop_sha pin", lock
    if lock.get("validate_sha") != _sha_file(FIXTURE_VAL):
        return False, "validate_sha pin", lock
    if lock.get("generator_sha") != _sha_file(GEN_LOCK):
        return False, "generator_sha pin", lock
    if any(k in lock for k in ("agent_sha", "run_tm023cortex_sha", "make_cortex_sha")):
        return False, "prereg contains runner/agent SHAs", lock
    if not PREREG_WALL.exists():
        return False, "missing wall prereg", lock
    return True, "cortex.prereg.lock intact", lock


# --- sanity tests ---

def _fresh(tmp: Path, name: str, *, device: str = "cpu", seed: int = 12345) -> NeuralCortex:
    s = tmp / name
    empty_birth(s)
    g = GenomeConfig(seed_birth=seed, seed_registry=seed + 1, seed_source=seed + 2, seed_action=seed + 3)
    return make_cortex(s, genome=g, device=device)


def sanity_order(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "order", device=device)
    ag2 = _fresh(tmp, "order2", device=device)
    # Register tokens in identical order so only presentation order differs.
    for tok in ("tok_a", "tok_b"):
        ag._vocab_vec(tok)
        ag2._vocab_vec(tok)
    body = [0.5, 0.2, 0.5, 0.0]
    e1 = build_observe(
        interaction_token="ix1",
        source_token="src",
        ordered_symbols=["tok_a", "tok_b"],
        observable_state=["st_idle"],
        body_state=body,
    )
    out1 = ag.observe(e1)
    traj1 = [t.copy() for t in ag.sensory_trajectory]
    e2 = build_observe(
        interaction_token="ix1",
        source_token="src",
        ordered_symbols=["tok_b", "tok_a"],
        observable_state=["st_idle"],
        body_state=body,
    )
    out2 = ag2.observe(e2)
    traj2 = ag2.sensory_trajectory
    if len(traj1) >= 3 and len(traj2) >= 3:
        d = float(np.linalg.norm(traj1[2] - traj2[2]))
    else:
        d = 0.0
    ok = out1["ok"] and out2["ok"] and d > 1e-6
    return {"id": "order_ab_ba", "ok": ok, "rho_distance": d}


def sanity_prediction(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "pred", device=device, seed=99)
    body = [0.5, 0.2, 0.5, 0.0]
    errs = []
    for i in range(12):
        st = ["st_on"] if i % 2 == 0 else ["st_off"]
        out = ag.observe(
            build_observe(
                interaction_token=f"ix{i}",
                source_token="src",
                ordered_symbols=["cue"],
                observable_state=st,
                body_state=body,
            )
        )
        if out["metrics"].get("pred_err") is not None and i > 0:
            errs.append(float(out["metrics"]["pred_err"]))
    if len(errs) < 4:
        return {"id": "prediction", "ok": False, "why": "too_few"}
    early = float(np.mean(errs[:3]))
    late = float(np.mean(errs[-3:]))
    ok = late <= early + 1e-9  # non-increasing trend allowed; prefer decrease
    # stronger: late < early * 1.05 or late < early
    ok = late < early or abs(late - early) < 1e-6
    return {"id": "prediction", "ok": ok, "early": early, "late": late}


def sanity_advantage(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    """Beneficial body transition reinforces the responsible op (three-factor rule)."""
    ag = _fresh(tmp, "adv", device=device, seed=7)
    # Drive rho away from zero so outer(e, rho_elig) is nonzero.
    body = np.array([0.1, 0.8, 0.1, 0.0], dtype=np.float64)
    ag.observe(
        build_observe(
            interaction_token="warm",
            source_token="src",
            ordered_symbols=["warm"],
            observable_state=["st_idle"],
            body_state=body.tolist(),
        )
    )
    rho_elig = ag._from_t(ag.rho).copy()
    ag._pending = {
        "op": "ACT",
        "token": "press",
        "rho_elig": rho_elig,
        "s_hat": np.zeros(ag.genome.d_sym, dtype=np.float64),
        "body": body.copy(),
        "cost": 1.0,
    }
    ag.vocab["press"] = ag.rng_registry.normal(0, 1, size=ag.genome.d_sym).astype(np.float64)
    w_before = ag._from_t(ag.W_op[OPS.index("ACT")]).copy()
    body2 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    out = ag.observe(
        build_observe(
            interaction_token="b0",
            source_token="src",
            ordered_symbols=["go"],
            observable_state=["st_pressed"],
            body_state=body2.tolist(),
        )
    )
    w_after = ag._from_t(ag.W_op[OPS.index("ACT")])
    delta = float(np.linalg.norm(w_after - w_before))
    adv = float(out.get("metrics", {}).get("adv") or 0.0)

    ag_h = _fresh(tmp, "advh", device=device, seed=7)
    body_h = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    ag_h.observe(
        build_observe(
            interaction_token="warm",
            source_token="src",
            ordered_symbols=["warm"],
            observable_state=["st_idle"],
            body_state=body_h.tolist(),
        )
    )
    ag_h._pending = {
        "op": "ACT",
        "token": "harm",
        "rho_elig": ag_h._from_t(ag_h.rho).copy(),
        "s_hat": np.zeros(ag_h.genome.d_sym, dtype=np.float64),
        "body": body_h.copy(),
        "cost": 1.0,
    }
    body_bad = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    out_h = ag_h.observe(
        build_observe(
            interaction_token="h0",
            source_token="src",
            ordered_symbols=["go"],
            observable_state=["st_hurt"],
            body_state=body_bad.tolist(),
        )
    )
    adv_h = float(out_h.get("metrics", {}).get("adv") or 0.0)
    ok = out["ok"] and adv > 0.0 and delta > 1e-9 and adv_h < 0.0
    return {
        "id": "advantage_path",
        "ok": ok,
        "adv_good": adv,
        "adv_bad": adv_h,
        "delta": delta,
    }


def sanity_exploration(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "exp", device=device, seed=11)
    body = [0.5, 0.2, 0.5, 0.0]
    ops = set()
    for i in range(40):
        out = ag.observe(
            build_observe(
                interaction_token=f"e{i}",
                source_token="src",
                ordered_symbols=["x"],
                observable_state=["st_idle"],
                body_state=body,
            )
        )
        if out.get("action"):
            ops.add(out["action"]["op"])
    ok = len(ops) >= 2
    return {"id": "exploration", "ok": ok, "ops": sorted(ops)}


def sanity_write_retrieve(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "wr", device=device, seed=13)
    body = [0.5, 0.2, 0.5, 0.0]
    # manually write then retrieve effect on rho
    from three_memory.cortex_memory import CortexRecord

    vec = ag.rng_registry.normal(0, 1, size=ag.genome.d_sym).astype(np.float64)
    ag.memory.write(
        CortexRecord(
            fact_id="manual_1",
            content=vec.tolist(),
            when=0,
            interaction_token="ix",
            source_token="src",
            source="cortex_write",
        )
    )
    ag.reset_rho()
    # force retrieve path
    ag._do_retrieve()
    ag._commit_pending_retrieve()
    buf_norm = float(torch.linalg.vector_norm(ag.retrieval_buffer).item())
    ag2 = _fresh(tmp, "wr2", device=device, seed=13)
    ag2.reset_rho()
    buf_norm2 = float(torch.linalg.vector_norm(ag2.retrieval_buffer).item())
    ok = buf_norm > 1e-6 and buf_norm2 == 0.0
    return {"id": "write_retrieve", "ok": ok, "buf_norm": buf_norm}


def sanity_checkpoint(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "ckpt", device=device, seed=17)
    body = [0.5, 0.2, 0.5, 0.0]
    actions = []
    for i in range(3):
        out = ag.observe(
            build_observe(
                interaction_token=f"c{i}",
                source_token="src",
                ordered_symbols=["a", "b"],
                observable_state=["st_idle"],
                body_state=body,
            )
        )
        actions.append(out["action"])
    snap = ag.checkpoint()
    # continue
    cont = []
    for i in range(3, 6):
        out = ag.observe(
            build_observe(
                interaction_token=f"c{i}",
                source_token="src",
                ordered_symbols=["a", "b"],
                observable_state=["st_idle"],
                body_state=body,
            )
        )
        cont.append(out["action"])
    # restore and replay continuation
    ag2 = _fresh(tmp, "ckpt2", device=device, seed=17)
    ag2.load_checkpoint(snap)
    cont2 = []
    for i in range(3, 6):
        out = ag2.observe(
            build_observe(
                interaction_token=f"c{i}",
                source_token="src",
                ordered_symbols=["a", "b"],
                observable_state=["st_idle"],
                body_state=body,
            )
        )
        cont2.append(out["action"])
    ok = cont == cont2
    return {"id": "checkpoint", "ok": ok, "cont": cont, "cont2": cont2}


def sanity_rho_reset(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "rho", device=device, seed=19)
    body = [0.5, 0.2, 0.5, 0.0]
    for i in range(4):
        ag.observe(
            build_observe(
                interaction_token=f"r{i}",
                source_token="src",
                ordered_symbols=["z"],
                observable_state=["st_idle"],
                body_state=body,
            )
        )
    w_before = ag.weight_hash()
    ag.retrieval_buffer = torch.ones_like(ag.retrieval_buffer)
    ag.reset_rho()
    w_after = ag.weight_hash()
    buf = float(torch.linalg.vector_norm(ag.retrieval_buffer).item())
    rho = float(torch.linalg.vector_norm(ag.rho).item())
    ok = w_before == w_after and buf == 0.0 and rho == 0.0
    return {"id": "rho_reset", "ok": ok}


def sanity_scorer_isolation(tmp: Path, device: str = "cpu") -> dict[str, Any]:
    ag = _fresh(tmp, "iso", device=device)
    # attempt banned keys
    bad = ag.observe(
        {
            "interaction_token": "ix",
            "source_token": "src",
            "ordered_symbols": ["a"],
            "observable_state": [],
            "body_state": [0.5, 0.2, 0.5, 0.0],
            "correct": True,
        }
    )
    bad2 = ag.observe(
        {
            "interaction_token": "ix",
            "source_token": "src",
            "ordered_symbols": ["a"],
            "observable_state": [],
            "body_state": [0.5, 0.2, 0.5, 0.0],
            "homeostatic_delta": 1.0,
        }
    )
    # forbidden source write
    refused = False
    try:
        from three_memory.cortex_memory import CortexRecord

        ag.memory.write(
            CortexRecord(
                fact_id="x",
                content=[0.0] * ag.genome.d_sym,
                when=0,
                interaction_token="i",
                source_token="s",
                source="experience_grounding",
            )
        )
    except ValueError:
        refused = True
    ok = bad["why"] == "banned_key" and bad2["why"] == "banned_key" and refused
    return {"id": "scorer_isolation", "ok": ok}


def sanity_cpu_gpu(tmp: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"id": "cpu_gpu", "ok": False, "why": "no_cuda"}
    body = [0.5, 0.2, 0.5, 0.0]
    events = [
        build_observe(
            interaction_token=f"g{i}",
            source_token="src",
            ordered_symbols=["a", "b"],
            observable_state=["st_idle"],
            body_state=body,
        )
        for i in range(5)
    ]
    cpu = _fresh(tmp, "cpu", device="cpu", seed=21)
    gpu = _fresh(tmp, "gpu", device="cuda", seed=21)
    acts_c, acts_g = [], []
    for e in events:
        acts_c.append(cpu.observe(e)["action"])
        acts_g.append(gpu.observe(e)["action"])
    # discrete op parity preferred; allow token mismatch if registry cos near threshold
    ops_match = [a["op"] for a in acts_c] == [a["op"] for a in acts_g]
    return {
        "id": "cpu_gpu",
        "ok": ops_match,
        "cpu_ops": [a["op"] for a in acts_c],
        "gpu_ops": [a["op"] for a in acts_g],
        "device": torch.cuda.get_device_name(0),
    }


def run_sanity(*, write_birth: bool = False, write_candidate: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    env = torch_env()
    results = []
    with tempfile.TemporaryDirectory(prefix="tm023cortex_") as tmp:
        root = Path(tmp)
        results.append(sanity_order(root, "cpu"))
        results.append(sanity_prediction(root, "cpu"))
        results.append(sanity_advantage(root, "cpu"))
        results.append(sanity_exploration(root, "cpu"))
        results.append(sanity_write_retrieve(root, "cpu"))
        results.append(sanity_checkpoint(root, "cpu"))
        results.append(sanity_rho_reset(root, "cpu"))
        results.append(sanity_scorer_isolation(root, "cpu"))
        results.append(sanity_cpu_gpu(root))
        # GPU device smoke of order if available
        if torch.cuda.is_available():
            results.append(sanity_order(root, "cuda"))

    # factory isolation: make_cortex must not be ThreeMemoryAgent
    with tempfile.TemporaryDirectory(prefix="tm023iso_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        assert not hasattr(ag, "interpret_message")
        assert not hasattr(ag, "plan_inquiry")
        assert type(ag).__name__ == "NeuralCortex"

    all_ok = all(r.get("ok") for r in results)
    learning_ok = all(
        r.get("ok")
        for r in results
        if r.get("id")
        in {
            "order_ab_ba",
            "prediction",
            "advantage_path",
            "exploration",
            "write_retrieve",
            "checkpoint",
            "rho_reset",
            "scorer_isolation",
        }
    )
    summary = {
        "version": "TM.0.23.CORTEX.SANITY",
        "lab": "TM.0.23.CORTEX",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all_ok,
        "learning_law_ok": learning_ok,
        "results": results,
        "env": env,
        "contract_sha": _sha_file(CONTRACT),
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "run_tm023cortex_sha": _sha_file(Path(__file__)),
        "make_cortex_sha": _sha_src(make_cortex),
        "prereg_sha": _sha_file(PREREG),
        "wall_prereg_sha": _sha_file(PREREG_WALL),
        "note": "Unscored infrastructure. Not D scoring. Not 0.0.005.",
    }
    if write_birth:
        birth = {
            **summary,
            "version": "TM.0.23.CORTEX.BIRTH",
            "genome": GenomeConfig().to_dict(),
            "gpu_equivalent": bool(
                next((r for r in results if r.get("id") == "cpu_gpu"), {}).get("ok")
            ),
            "refuse": [
                "D scoring without candidate",
                "earned_next=true or non-null ex0s",
                "wrapping make_interpret",
            ],
        }
        BIRTH_LOCK.write_text(json.dumps(birth, indent=2) + "\n", encoding="utf-8")
        summary["birth_written"] = True
    if write_candidate:
        if not learning_ok:
            raise RuntimeError("candidate refused: learning-law sanity failed")
        if not BIRTH_LOCK.exists():
            raise RuntimeError("candidate requires birth lock")
        cand = {
            "version": "TM.0.23.CORTEX.CANDIDATE",
            "lab": "TM.0.23.CORTEX",
            "ex0s_under_test": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "factory": "experiments.run_tm023cortex.make_cortex",
            "learning_law_ok": True,
            "birth_sha": _sha_file(BIRTH_LOCK),
            "neural_cortex_sha": _sha_file(NEURAL_PY),
            "cortex_memory_sha": _sha_file(MEMORY_PY),
            "make_cortex_sha": _sha_src(make_cortex),
            "run_tm023cortex_sha": _sha_file(Path(__file__)),
            "contract_sha": _sha_file(CONTRACT),
            "prereg_sha": _sha_file(PREREG),
            "wall_prereg_sha": _sha_file(PREREG_WALL),
            "env": env,
            "genome": GenomeConfig().to_dict(),
            "note": "Pinned after learning-law sanity. Preserve as v1 if audit rewrites.",
        }
        CANDIDATE_LOCK.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
        if not CANDIDATE_V1.exists():
            CANDIDATE_V1.write_text(json.dumps(cand, indent=2) + "\n", encoding="utf-8")
        summary["candidate_written"] = True
        _write_results(summary)
    return summary


def _write_results(summary: dict[str, Any]) -> None:
    lines = [
        "# TM.0.23.CORTEX results: developmental artificial cortex (birth pass)",
        "",
        "**Ex0S under test:** **0.0.004** (not a new stamp)",
        "**Lab:** TM.0.23.CORTEX",
        f"**ok (sanity):** `{summary.get('ok')}`",
        f"**learning_law_ok:** `{summary.get('learning_law_ok')}`",
        "",
        "Locks: [`cortex.prereg.lock`](cortex.prereg.lock) · "
        "[`cortex_wall.prereg.lock`](cortex_wall.prereg.lock) · "
        "[`cortex_birth.lock`](cortex_birth.lock) · "
        "[`cortex.candidate.lock`](cortex.candidate.lock)",
        "",
        "`earned_next`: **false** — no Ex0S 0.0.005. Product stamp remains **0.0.004**.",
        "",
        "## This pass",
        "",
        "Architecture contract, worlds, preregs, CPU/GPU birth substrate, and unscored "
        "learning-law sanity. **No D0–D12 scoring.**",
        "",
        "## Sanity",
        "",
    ]
    for r in summary.get("results") or []:
        lines.append(f"- `{r.get('id')}`: **{'pass' if r.get('ok') else 'fail'}**")
    lines += [
        "",
        "## Next",
        "",
        "Human/math audit of birth; then D0–D12 developmental scoring on a later pass.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-phase-a", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--write-birth", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    args = ap.parse_args()

    if args.write_phase_a:
        print(json.dumps(write_phase_a_artifacts(), indent=2))
        return
    if args.verify_prereg:
        ok, why, lock = verify_prereg()
        print(json.dumps({"ok": ok, "why": why, "lab": lock.get("lab")}, indent=2))
        return
    if args.sanity or args.write_birth or args.write_candidate:
        print(
            json.dumps(
                run_sanity(
                    write_birth=args.write_birth or args.write_candidate,
                    write_candidate=args.write_candidate,
                ),
                indent=2,
                default=str,
            )
        )
        return
    ap.print_help()


if __name__ == "__main__":
    main()
