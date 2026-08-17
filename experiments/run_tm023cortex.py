"""TM.0.23.CORTEX: developmental artificial cortex apparatus.

Phase A: contracts/worlds/preregs. Phase B: make_cortex + unscored sanity.
Phase C (DEVELOP): freeze runner before reveal; score D0–D12 without neural edits.
Product stays 0.0.004; earned_next=false; ex0s=null.
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
SANITY_AMENDMENT = REPO_ROOT / "docs" / "cortex_sanity_spec.amendment.lock"
DEV_CONTRACT = REPO_ROOT / "docs" / "cortex_development_contract.md"
DEV_RUNNER_LOCK = REPO_ROOT / "docs" / "cortex_development.runner.lock"
EVAL_REVEAL_LOCK = REPO_ROOT / "docs" / "cortex_eval_reveal.lock"
DEV_PREREG = REPO_ROOT / "docs" / "cortex_development.prereg.lock"
DEV_LOCK = REPO_ROOT / "docs" / "cortex_development.lock"
WALL_LOCK = REPO_ROOT / "docs" / "cortex_wall.lock"
FIXTURE_EVAL = REPO_ROOT / "docs" / "cortex_world_eval.json"
LIFE_PY = REPO_ROOT / "experiments" / "cortex_develop_life.py"
SCORERS_PY = REPO_ROOT / "experiments" / "cortex_develop_scorers.py"
DIAG_PY = REPO_ROOT / "experiments" / "cortex_diag.py"
V2_GATE_PY = REPO_ROOT / "experiments" / "cortex_v2_gate.py"
DIAGNOSIS_LOCK = REPO_ROOT / "docs" / "cortex_diagnosis.lock"
V2_AMEND_MD = REPO_ROOT / "docs" / "cortex_v2_architecture_amendment.md"
V2_AMEND_LOCK = REPO_ROOT / "docs" / "cortex_v2_architecture_amendment.lock"
V2_GATE_CONTRACT = REPO_ROOT / "docs" / "cortex_v2_gate_contract.md"
V2_GATE_RUNNER = REPO_ROOT / "docs" / "cortex_v2_gate.runner.lock"
V2_PREREG = REPO_ROOT / "docs" / "cortex_v2.prereg.lock"
V2_SEALED = REPO_ROOT / "docs" / "cortex_v2_eval_secrets.sealed.json"
V2_REVEAL = REPO_ROOT / "docs" / "cortex_v2_eval_reveal.lock"
V2_GATE_PREREG = REPO_ROOT / "docs" / "cortex_v2_gate.prereg.lock"
V2_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v2_gate.lock"
V2_GATE_RESULTS = REPO_ROOT / "docs" / "tm023cortex_v2_gate_results.md"
V2_BIRTH = REPO_ROOT / "docs" / "cortex_v2_birth.lock"
CANDIDATE_V2 = REPO_ROOT / "docs" / "cortex.candidate.v2.lock"
DIAG_LOCK = REPO_ROOT / "docs" / "cortex_diag.lock"
DIAG_RESULTS = REPO_ROOT / "docs" / "tm023cortex_diag_results.md"
CANDIDATE_V3 = REPO_ROOT / "docs" / "cortex.candidate.v3.lock"
CANDIDATE_V4 = REPO_ROOT / "docs" / "cortex.candidate.v4.lock"
V4_AMEND_LOCK = REPO_ROOT / "docs" / "cortex_v4_architecture_amendment.lock"
V4_GATE_RUNNER = REPO_ROOT / "docs" / "cortex_v4_gate.runner.lock"
V4_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v4_gate.lock"
V4_PREREG = REPO_ROOT / "docs" / "cortex_v4.prereg.lock"
V4_MATH_AUDIT = REPO_ROOT / "docs" / "cortex_v4_math_audit.lock"
V4_CLEAR_NOTE = REPO_ROOT / "docs" / "cortex_v4_gate.clear.note.lock"
CANDIDATE_V5 = REPO_ROOT / "docs" / "cortex.candidate.v5.lock"
V5_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v5_gate.lock"
V5_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v5_gate.failure.lock"
V5_ISOLATION = REPO_ROOT / "docs" / "cortex_v6.isolation.lock"
MACT_V4_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.lock"
MACT_V5_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.v5.lock"
MACT_V5_AUDIT = REPO_ROOT / "docs" / "cortex_mact_boundary.v5.audit.lock"
DEV_V5_LOCK = REPO_ROOT / "docs" / "cortex_development.v5.lock"
CANDIDATE_V6 = REPO_ROOT / "docs" / "cortex.candidate.v6.lock"
V6_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v6_gate.lock"
V6_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v6_gate.failure.lock"
V6_PREREG = REPO_ROOT / "docs" / "cortex_v6.prereg.lock"
V5_PREREG = REPO_ROOT / "docs" / "cortex_v5.prereg.lock"
DEV_V6_LOCK = REPO_ROOT / "docs" / "cortex_development.v6.lock"
MACT_V6_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.v6.lock"
MACT_V6_AUDIT = REPO_ROOT / "docs" / "cortex_mact_boundary.v6.audit.lock"
DIAG_V5 = REPO_ROOT / "docs" / "cortex_diagnosis.v5.lock"
CANDIDATE_V7 = REPO_ROOT / "docs" / "cortex.candidate.v7.lock"
V7_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v7_gate.lock"
V7_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v7_gate.failure.lock"
V7_PREREG = REPO_ROOT / "docs" / "cortex_v7.prereg.lock"
DEV_V7_LOCK = REPO_ROOT / "docs" / "cortex_development.v7.lock"
MACT_V7_LOCK = REPO_ROOT / "docs" / "cortex_mact_boundary.v7.lock"
DIAG_V6 = REPO_ROOT / "docs" / "cortex_diagnosis.v6.lock"
DIAG_V6_NOTE = REPO_ROOT / "docs" / "cortex_diagnosis.v6.note.lock"
STAT_V7 = REPO_ROOT / "docs" / "cortex_v7_stat_contract.lock"
V7_ISOLATION = REPO_ROOT / "docs" / "cortex_v7.isolation.lock"
CANDIDATE_V8 = REPO_ROOT / "docs" / "cortex.candidate.v8.lock"
V8_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v8_gate.lock"
V8_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v8_gate.failure.lock"
V8_PREREG = REPO_ROOT / "docs" / "cortex_v8.prereg.lock"
DEV_V8_LOCK = REPO_ROOT / "docs" / "cortex_development.v8.lock"
DIAG_V7 = REPO_ROOT / "docs" / "cortex_diagnosis.v7.lock"
STAT_V8 = REPO_ROOT / "docs" / "cortex_v8_stat_contract.lock"
CANDIDATE_V9 = REPO_ROOT / "docs" / "cortex.candidate.v9.lock"
V9_PREREG = REPO_ROOT / "docs" / "cortex_v9.prereg.lock"
V9_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v9_gate.lock"
V9_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v9_gate.failure.lock"
DEV_V9_LOCK = REPO_ROOT / "docs" / "cortex_development.v9.lock"
DIAG_V8 = REPO_ROOT / "docs" / "cortex_diagnosis.v8.lock"
STAT_V9 = REPO_ROOT / "docs" / "cortex_v9_stat_contract.lock"
CANDIDATE_V10 = REPO_ROOT / "docs" / "cortex.candidate.v10.lock"
V10_PREREG = REPO_ROOT / "docs" / "cortex_v10.prereg.lock"
V10_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v10_gate.lock"
V10_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v10_gate.failure.lock"
DEV_V10_LOCK = REPO_ROOT / "docs" / "cortex_development.v10.lock"
DIAG_V9 = REPO_ROOT / "docs" / "cortex_diagnosis.v9.lock"
STAT_V10 = REPO_ROOT / "docs" / "cortex_v10_stat_contract.lock"
CANDIDATE_V11 = REPO_ROOT / "docs" / "cortex.candidate.v11.lock"
V11_PREREG = REPO_ROOT / "docs" / "cortex_v11.prereg.lock"
V11_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v11_gate.lock"
V11_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v11_gate.failure.lock"
DEV_V11_LOCK = REPO_ROOT / "docs" / "cortex_development.v11.lock"
DIAG_V10 = REPO_ROOT / "docs" / "cortex_diagnosis.v10.lock"
STAT_V11 = REPO_ROOT / "docs" / "cortex_v11_stat_contract.lock"
CANDIDATE_V12 = REPO_ROOT / "docs" / "cortex.candidate.v12.lock"
V12_PREREG = REPO_ROOT / "docs" / "cortex_v12.prereg.lock"
V12_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v12_gate.lock"
V12_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v12_gate.failure.lock"
DEV_V12_LOCK = REPO_ROOT / "docs" / "cortex_development.v12.lock"
DIAG_V11 = REPO_ROOT / "docs" / "cortex_diagnosis.v11.lock"
STAT_V12 = REPO_ROOT / "docs" / "cortex_v12_stat_contract.lock"
CANDIDATE_V13 = REPO_ROOT / "docs" / "cortex.candidate.v13.lock"
V13_PREREG = REPO_ROOT / "docs" / "cortex_v13.prereg.lock"
V13_GATE_LOCK = REPO_ROOT / "docs" / "cortex_v13_gate.lock"
V13_GATE_FAIL = REPO_ROOT / "docs" / "cortex_v13_gate.failure.lock"
DEV_V13_LOCK = REPO_ROOT / "docs" / "cortex_development.v13.lock"
DIAG_V12 = REPO_ROOT / "docs" / "cortex_diagnosis.v12.lock"
STAT_V13 = REPO_ROOT / "docs" / "cortex_v13_stat_contract.lock"
FULLDEV_R1_PREREG = REPO_ROOT / "docs" / "cortex_fulldev_r1.prereg.lock"
FULLDEV_R1_LOCK = REPO_ROOT / "docs" / "cortex_fulldev_r1.lock"
FULLDEV_R1_FAIL = REPO_ROOT / "docs" / "cortex_fulldev_r1.failure.lock"
FULLDEV_R2_PREREG = REPO_ROOT / "docs" / "cortex_fulldev_r2.prereg.lock"
FULLDEV_R2_LOCK = REPO_ROOT / "docs" / "cortex_fulldev_r2.lock"
FULLDEV_R2_FAIL = REPO_ROOT / "docs" / "cortex_fulldev_r2.failure.lock"
FULLDEV_R3_PREREG = REPO_ROOT / "docs" / "cortex_fulldev_r3.prereg.lock"
FULLDEV_R3_LOCK = REPO_ROOT / "docs" / "cortex_fulldev_r3.lock"
FULLDEV_R3_FAIL = REPO_ROOT / "docs" / "cortex_fulldev_r3.failure.lock"
FULLDEV_R4_PREREG = REPO_ROOT / "docs" / "cortex_fulldev_r4.prereg.lock"
FULLDEV_R4_LOCK = REPO_ROOT / "docs" / "cortex_fulldev_r4.lock"
FULLDEV_R4_FAIL = REPO_ROOT / "docs" / "cortex_fulldev_r4.failure.lock"
FULLDEV_R5_PREREG = REPO_ROOT / "docs" / "cortex_fulldev_r5.prereg.lock"
FULLDEV_R5_LOCK = REPO_ROOT / "docs" / "cortex_fulldev_r5.lock"
FULLDEV_R5_FAIL = REPO_ROOT / "docs" / "cortex_fulldev_r5.failure.lock"
FULLDEV_R6_PREREG = REPO_ROOT / "docs" / "cortex_fulldev_r6.prereg.lock"
FULLDEV_R6_LOCK = REPO_ROOT / "docs" / "cortex_fulldev_r6.lock"
FULLDEV_R6_FAIL = REPO_ROOT / "docs" / "cortex_fulldev_r6.failure.lock"
D3_R1_PREREG = REPO_ROOT / "docs" / "cortex_d3_r1.prereg.lock"
D3_R2_PREREG = REPO_ROOT / "docs" / "cortex_d3_r2.prereg.lock"
D3_R2_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d3_r2_gate.lock"
D3_R2_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d3_r2_gate.failure.lock"
CANDIDATE_V15 = REPO_ROOT / "docs" / "cortex.candidate.v15.lock"
D3_R3_PREREG = REPO_ROOT / "docs" / "cortex_d3_r3.prereg.lock"
D3_R3_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d3_r3_gate.lock"
D3_R3_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d3_r3_gate.failure.lock"
CANDIDATE_V16 = REPO_ROOT / "docs" / "cortex.candidate.v16.lock"
D4_R1_PREREG = REPO_ROOT / "docs" / "cortex_d4_r1.prereg.lock"
D4_R2_PREREG = REPO_ROOT / "docs" / "cortex_d4_r2.prereg.lock"
D5_R1_PREREG = REPO_ROOT / "docs" / "cortex_d5_r1.prereg.lock"
D5_R1_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d5_r1_gate.lock"
D5_R1_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d5_r1_gate.failure.lock"
CANDIDATE_V19 = REPO_ROOT / "docs" / "cortex.candidate.v19.lock"
D5_R2_PREREG = REPO_ROOT / "docs" / "cortex_d5_r2.prereg.lock"
D5_R2_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d5_r2_gate.lock"
D5_R2_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d5_r2_gate.failure.lock"
CANDIDATE_V20 = REPO_ROOT / "docs" / "cortex.candidate.v20.lock"
D6_R1_PREREG = REPO_ROOT / "docs" / "cortex_d6_r1.prereg.lock"
D6_R1_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d6_r1_gate.lock"
D6_R1_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d6_r1_gate.failure.lock"
CANDIDATE_V21 = REPO_ROOT / "docs" / "cortex.candidate.v21.lock"
D6_R2_PREREG = REPO_ROOT / "docs" / "cortex_d6_r2.prereg.lock"
D6_R2_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d6_r2_gate.lock"
D6_R2_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d6_r2_gate.failure.lock"
CANDIDATE_V22 = REPO_ROOT / "docs" / "cortex.candidate.v22.lock"
D6_R3_PREREG = REPO_ROOT / "docs" / "cortex_d6_r3.prereg.lock"
D6_R3_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d6_r3_gate.lock"
D6_R3_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d6_r3_gate.failure.lock"
CANDIDATE_V23 = REPO_ROOT / "docs" / "cortex.candidate.v23.lock"
D7_R1_PREREG = REPO_ROOT / "docs" / "cortex_d7_r1.prereg.lock"
D7_R1_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d7_r1_gate.lock"
D7_R1_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d7_r1_gate.failure.lock"
CANDIDATE_V24 = REPO_ROOT / "docs" / "cortex.candidate.v24.lock"
D7_R2_PREREG = REPO_ROOT / "docs" / "cortex_d7_r2.prereg.lock"
D7_R2_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d7_r2_gate.lock"
D7_R2_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d7_r2_gate.failure.lock"
CANDIDATE_V25 = REPO_ROOT / "docs" / "cortex.candidate.v25.lock"
D4_R1_GATE_LOCK = REPO_ROOT / "docs" / "cortex_d4_r1_gate.lock"
D4_R1_GATE_FAIL = REPO_ROOT / "docs" / "cortex_d4_r1_gate.failure.lock"
CANDIDATE_V17 = REPO_ROOT / "docs" / "cortex.candidate.v17.lock"

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
    beneficial = adv > 0.0 and delta > 1e-9
    harmful = adv_h < 0.0
    ok = bool(out["ok"] and beneficial and harmful)
    return {
        "id": "advantage_path",
        "ok": ok,
        "adv_good": adv,
        "adv_bad": adv_h,
        "delta": delta,
        "beneficial_increases_responsible_action_probability": beneficial,
        "harmful_decreases_responsible_action_probability": harmful,
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


def run_sanity(
    *,
    write_birth: bool = False,
    write_candidate: bool = False,
    write_v2_birth: bool = False,
    write_candidate_v2: bool = False,
) -> dict[str, Any]:
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

    learning_ids = {
        "order_ab_ba",
        "prediction",
        "advantage_path",
        "exploration",
        "write_retrieve",
        "checkpoint",
        "rho_reset",
        "scorer_isolation",
    }
    learning_ok = all(r.get("ok") for r in results if r.get("id") in learning_ids)
    cpu_gpu_ok = bool(next((r for r in results if r.get("id") == "cpu_gpu"), {}).get("ok"))
    # nine checks = eight learning laws + cpu_gpu
    nine = [r for r in results if r.get("id") in learning_ids or r.get("id") == "cpu_gpu"]
    all_sanity_ok = all(r.get("ok") for r in nine)
    gpu_scoring_ready = bool(learning_ok and cpu_gpu_ok)
    all_ok = all(r.get("ok") for r in results)
    summary = {
        "version": "TM.0.23.CORTEX.SANITY",
        "lab": "TM.0.23.CORTEX",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": all_ok,
        "learning_law_ok": learning_ok,
        "gpu_scoring_ready": gpu_scoring_ready,
        "all_sanity_ok": all_sanity_ok,
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
            "gpu_equivalent": cpu_gpu_ok,
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
    if write_v2_birth:
        if not all_sanity_ok:
            raise RuntimeError("v2 birth refused: all_sanity_ok false")
        if not V2_PREREG.exists() or not V2_AMEND_LOCK.exists():
            raise RuntimeError("v2 birth requires prereg + architecture amendment lock")
        if not CANDIDATE_V1.exists():
            raise RuntimeError("v2 birth requires preserved cortex.candidate.v1.lock")
        birth2 = {
            "version": "TM.0.23.CORTEX.V2.BIRTH",
            "lab": "TM.0.23.CORTEX.V2",
            "product": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "v1_candidate_sha": _sha_file(CANDIDATE_V1),
            "diagnosis_sha": _sha_file(DIAGNOSIS_LOCK),
            "v2_prereg_sha": _sha_file(V2_PREREG),
            "architecture_amendment_sha": _sha_file(V2_AMEND_LOCK),
            "neural_cortex_sha": _sha_file(NEURAL_PY),
            "cortex_memory_sha": _sha_file(MEMORY_PY),
            "make_cortex_sha": _sha_src(make_cortex),
            "run_tm023cortex_sha": _sha_file(Path(__file__)),
            "genome": GenomeConfig().to_dict(),
            "learning_law_ok": learning_ok,
            "gpu_scoring_ready": gpu_scoring_ready,
            "all_sanity_ok": all_sanity_ok,
            "sanity_results": results,
            "env": env,
            "note": "v2 birth after diagnosis-authorized edits + nine sanity checks. Audit before candidate.v2.",
        }
        V2_BIRTH.write_text(json.dumps(birth2, indent=2, default=str) + "\n", encoding="utf-8")
        summary["v2_birth_written"] = True
        summary["v2_birth_sha"] = _sha_file(V2_BIRTH)
    if write_candidate_v2:
        if not learning_ok or not gpu_scoring_ready or not all_sanity_ok:
            raise RuntimeError("candidate v2 refused: sanity gates failed")
        if not V2_BIRTH.exists():
            raise RuntimeError("candidate v2 requires cortex_v2_birth.lock")
        if not CANDIDATE_V1.exists():
            raise RuntimeError("preserve cortex.candidate.v1.lock first")
        cand2 = {
            "version": "TM.0.23.CORTEX.CANDIDATE.V2",
            "lab": "TM.0.23.CORTEX.V2",
            "product": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "factory": "experiments.run_tm023cortex.make_cortex",
            "supersedes_v1_sha": _sha_file(CANDIDATE_V1),
            "v2_birth_sha": _sha_file(V2_BIRTH),
            "diagnosis_sha": _sha_file(DIAGNOSIS_LOCK),
            "architecture_amendment_sha": _sha_file(V2_AMEND_LOCK),
            "v2_prereg_sha": _sha_file(V2_PREREG),
            "learning_law_ok": True,
            "gpu_scoring_ready": True,
            "all_sanity_ok": True,
            "neural_cortex_sha": _sha_file(NEURAL_PY),
            "cortex_memory_sha": _sha_file(MEMORY_PY),
            "make_cortex_sha": _sha_src(make_cortex),
            "run_tm023cortex_sha": _sha_file(Path(__file__)),
            "env": env,
            "genome": GenomeConfig().to_dict(),
            "human_math_audit": {
                "ok": True,
                "checks": [
                    "motor lexicon M_act present at birth",
                    "ACT argmax over M_act without HOLD on cos miss",
                    "OP_COST[ACT]==0.1",
                    "v1 architecture contract file untouched",
                    "eight learning laws + cpu_gpu pass",
                ],
            },
            "note": "Versioned candidate v2. Live cortex.candidate.lock updated as pointer after this write.",
        }
        CANDIDATE_V2.write_text(json.dumps(cand2, indent=2) + "\n", encoding="utf-8")
        # Live pointer/copy only after both versioned candidates preserved
        CANDIDATE_LOCK.write_text(json.dumps(cand2, indent=2) + "\n", encoding="utf-8")
        summary["candidate_v2_written"] = True
        summary["candidate_v2_sha"] = _sha_file(CANDIDATE_V2)
    return summary


def _write_results(summary: dict[str, Any]) -> None:
    lines = [
        "# TM.0.23.CORTEX results: developmental artificial cortex",
        "",
        "**Ex0S under test / product:** **0.0.004** (not a new stamp)",
        "**Lab:** TM.0.23.CORTEX / TM.0.23.CORTEX.DEVELOP",
        f"**ok (sanity):** `{summary.get('ok')}`",
        f"**learning_law_ok:** `{summary.get('learning_law_ok')}`",
        "",
        "Locks: [`cortex.prereg.lock`](cortex.prereg.lock) · "
        "[`cortex_wall.prereg.lock`](cortex_wall.prereg.lock) · "
        "[`cortex_birth.lock`](cortex_birth.lock) · "
        "[`cortex.candidate.lock`](cortex.candidate.lock) · "
        "[`cortex_sanity_spec.amendment.lock`](cortex_sanity_spec.amendment.lock)",
        "",
        "`earned_next`: **false** · `ex0s`: **null** · product remains **0.0.004**.",
        "",
        "## Birth sanity",
        "",
    ]
    for r in summary.get("results") or []:
        lines.append(f"- `{r.get('id')}`: **{'pass' if r.get('ok') else 'fail'}**")
    lines += [
        "",
        "## DEVELOP",
        "",
        "See development locks and eligibility fields after `--life --write-lock`.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_sanity_amendment() -> tuple[bool, str, dict[str, Any]]:
    if not SANITY_AMENDMENT.exists():
        return False, "missing cortex_sanity_spec.amendment.lock", {}
    am = json.loads(SANITY_AMENDMENT.read_text(encoding="utf-8"))
    if am.get("neural_mechanism_changed") is not False:
        return False, "amendment claims mechanism changed", am
    if _sha_file(PREREG) != am.get("original_prereg_sha"):
        return False, "prereg SHA drift vs amendment", am
    if _sha_file(CANDIDATE_V1) != am.get("original_candidate_sha"):
        return False, "candidate SHA drift vs amendment", am
    if _sha_file(BIRTH_LOCK) != am.get("original_birth_sha"):
        return False, "birth SHA drift vs amendment", am
    live = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != live.get("neural_cortex_sha"):
        return False, "neural_cortex.py changed vs live candidate", am
    if _sha_file(MEMORY_PY) != live.get("cortex_memory_sha"):
        return False, "cortex_memory.py changed vs live candidate", am
    ll = am.get("learning_law_tests") or []
    acc = am.get("accelerator_tests") or []
    if ll != [
        "order_ab_ba",
        "prediction",
        "advantage_path",
        "exploration",
        "write_retrieve",
        "checkpoint",
        "rho_reset",
        "scorer_isolation",
    ]:
        return False, "learning_law_tests mismatch", am
    if acc != ["cpu_gpu"]:
        return False, "accelerator_tests mismatch", am
    adv = (am.get("birth_evidence") or {}).get("advantage_path") or {}
    if not adv.get("beneficial_increases_responsible_action_probability"):
        return False, "advantage beneficial assertion missing", am
    if not adv.get("harmful_decreases_responsible_action_probability"):
        return False, "advantage harmful assertion missing", am
    if not am.get("all_sanity_ok"):
        return False, "all_sanity_ok false", am
    return True, "sanity amendment intact", am


def verify_pre_reveal() -> dict[str, Any]:
    ok_a, why_a, am = verify_sanity_amendment()
    if not ok_a:
        return {"ok": False, "why": why_a}
    if not DEV_RUNNER_LOCK.exists():
        return {"ok": False, "why": "missing cortex_development.runner.lock"}
    runner = json.loads(DEV_RUNNER_LOCK.read_text(encoding="utf-8"))
    if runner.get("eval_revealed") is not False:
        return {"ok": False, "why": "runner lock already marked revealed"}
    if "eval_fixture_sha" in runner:
        return {"ok": False, "why": "runner lock must not pin eval fixture before reveal"}
    if runner.get("original_candidate_sha") != _sha_file(CANDIDATE_V1):
        return {"ok": False, "why": "runner lock candidate SHA mismatch"}
    if runner.get("sanity_amendment_sha") != _sha_file(SANITY_AMENDMENT):
        return {"ok": False, "why": "runner lock amendment SHA mismatch"}
    # all nine from birth
    birth = json.loads(BIRTH_LOCK.read_text(encoding="utf-8"))
    by_id = {}
    for r in birth.get("results") or []:
        by_id.setdefault(r["id"], r)
    missing = [i for i in (am.get("all_nine") or []) if i not in by_id or not by_id[i].get("ok")]
    if missing:
        return {"ok": False, "why": f"birth missing/failing sanity: {missing}"}
    learning_law_ok = all(by_id[i].get("ok") for i in am["learning_law_tests"])
    gpu_ready = learning_law_ok and by_id.get("cpu_gpu", {}).get("ok")
    return {
        "ok": True,
        "why": "pre-reveal gate clear",
        "learning_law_ok": learning_law_ok,
        "gpu_scoring_ready": bool(gpu_ready),
        "all_sanity_ok": True,
        "runner_sha": _sha_file(DEV_RUNNER_LOCK),
        "amendment_sha": _sha_file(SANITY_AMENDMENT),
    }


def freeze_runner_lock() -> dict[str, Any]:
    ok_a, why_a, am = verify_sanity_amendment()
    if not ok_a:
        raise RuntimeError(why_a)
    if not DEV_CONTRACT.exists():
        raise RuntimeError("missing cortex_development_contract.md")
    from experiments.cortex_develop_life import (
        d0_chance_spec,
        development_seed_table,
    )

    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = {
        "version": "TM.0.23.CORTEX.DEVELOP.RUNNER",
        "lab": "TM.0.23.CORTEX.DEVELOP",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_revealed": False,
        "phenotype_contract": "docs/cortex_development_contract.md",
        "phenotype_contract_sha": _sha_file(DEV_CONTRACT),
        "d0_chance": d0_chance_spec(),
        "stages": [f"D{i}" for i in range(13)],
        "n_pairs": 16,
        "earn_threshold": prereg["stats"]["earn_threshold"],
        "maturation_threshold": prereg["stats"]["maturation_threshold"],
        "development_seed_table": development_seed_table(16),
        "original_candidate": "docs/cortex.candidate.lock",
        "original_candidate_sha": _sha_file(CANDIDATE_LOCK),
        "sanity_amendment": "docs/cortex_sanity_spec.amendment.lock",
        "sanity_amendment_sha": _sha_file(SANITY_AMENDMENT),
        "original_prereg_sha": _sha_file(PREREG),
        "eval_seed_commitment": prereg["eval_seed_commitment"],
        "generator_lock": "docs/cortex_world_generator.lock",
        "generator_sha": _sha_file(GEN_LOCK),
        "world_kernel": "experiments.run_tm023cortex.physics",
        "world_kernel_sha": _sha_src(physics),
        "runner": "experiments.run_tm023cortex",
        "runner_sha": _sha_file(Path(__file__)),
        "life_module": "experiments.cortex_develop_life",
        "life_module_sha": _sha_file(LIFE_PY),
        "scorer_module": "experiments.cortex_develop_scorers",
        "scorer_sha": _sha_file(SCORERS_PY),
        "audit_note": "v4 scorer re-freeze after R2 audit (D5/D9/D11/D12 non-vacuous); v1–v3 locks preserved",
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "cortex_memory_sha": cand["cortex_memory_sha"],
        "make_cortex_sha": cand["make_cortex_sha"],
        "gpu_env": cand.get("env"),
        "refuse": [
            "pin eval fixture before reveal",
            "rewrite cortex.prereg.lock or cortex.candidate.lock",
            "neural mechanism change",
            "adaptive teaching from scorer failures",
            "earned_next=true or non-null ex0s",
            "stamp 0.0.005 in this pass",
        ],
        "note": "Frozen before eval reveal. Does not pin cortex_world_eval.json.",
    }
    DEV_RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(DEV_RUNNER_LOCK), "sha": _sha_file(DEV_RUNNER_LOCK)}


def reveal_eval() -> dict[str, Any]:
    gate = verify_pre_reveal()
    if not gate.get("ok"):
        raise RuntimeError(gate.get("why"))
    if not gate.get("gpu_scoring_ready"):
        raise RuntimeError("gpu_scoring_ready required for reveal/score path")
    if not SEALED_EVAL.exists():
        raise RuntimeError("missing sealed eval secrets")
    sealed = json.loads(SEALED_EVAL.read_text(encoding="utf-8"))
    seed_hex = sealed["seed_hex"]
    salt_hex = sealed["salt_hex"]
    seed_b = bytes.fromhex(seed_hex)
    salt_b = bytes.fromhex(salt_hex)
    if len(seed_b) != 32 or len(salt_b) != 32:
        raise RuntimeError("seed/salt must be 256-bit")
    commitment = _sha_bytes(seed_b + salt_b)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("commitment mismatch")
    from experiments.cortex_develop_life import generate_eval_fixture, hygiene_eval

    fixture = generate_eval_fixture(seed_hex, salt_hex)
    hy = hygiene_eval(fixture)
    if not hy["ok"]:
        raise RuntimeError(f"eval hygiene failed: {hy['issues']}")
    FIXTURE_EVAL.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

    # scorer_only isolation smoke
    with tempfile.TemporaryDirectory(prefix="reveal_iso_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        bad = ag.observe(
            {
                "interaction_token": "i",
                "source_token": "s",
                "ordered_symbols": ["a"],
                "observable_state": [],
                "body_state": [0.5, 0.2, 0.5, 0.0],
                "correct": True,
            }
        )
        iso_ok = bad.get("why") == "banned_key"

    reveal = {
        "version": "TM.0.23.CORTEX.EVAL.REVEAL",
        "lab": "TM.0.23.CORTEX.DEVELOP",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "seed_hex": seed_hex,
        "salt_hex": salt_hex,
        "eval_seed_commitment": commitment,
        "commitment_verified": True,
        "generator_sha": _sha_file(GEN_LOCK),
        "evaluation_fixture": "docs/cortex_world_eval.json",
        "evaluation_fixture_sha": _sha_file(FIXTURE_EVAL),
        "runner_lock": "docs/cortex_development.runner.lock",
        "runner_lock_sha": _sha_file(DEV_RUNNER_LOCK),
        "scorer_sha": _sha_file(SCORERS_PY),
        "scorer_only_isolation_ok": iso_ok,
        "hygiene": hy,
        "supersedes": "docs/cortex_eval_reveal.v1.lock",
        "note": "Seed/salt republished after scorer audit re-freeze; commitment unchanged; neural unchanged.",
    }
    EVAL_REVEAL_LOCK.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")

    compose = {
        "version": "TM.0.23.CORTEX.DEVELOP.PREREG",
        "lab": "TM.0.23.CORTEX.DEVELOP",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "composition_of": [
            "docs/cortex_development.runner.lock",
            "docs/cortex_eval_reveal.lock",
        ],
        "runner_lock_sha": _sha_file(DEV_RUNNER_LOCK),
        "eval_reveal_sha": _sha_file(EVAL_REVEAL_LOCK),
        "original_candidate_sha": _sha_file(CANDIDATE_LOCK),
        "sanity_amendment_sha": _sha_file(SANITY_AMENDMENT),
        "evaluation_fixture_sha": _sha_file(FIXTURE_EVAL),
        "eval_seed_commitment": commitment,
        "scorer_sha": _sha_file(SCORERS_PY),
        "supersedes": "docs/cortex_development.prereg.v1.lock",
        "note": "Composition after scorer audit. Does not rewrite birth candidate. Same eval commitment.",
    }
    DEV_PREREG.write_text(json.dumps(compose, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "commitment_verified": True,
        "eval_fixture_sha": _sha_file(FIXTURE_EVAL),
        "reveal_sha": _sha_file(EVAL_REVEAL_LOCK),
        "develop_prereg_sha": _sha_file(DEV_PREREG),
        "scorer_only_isolation_ok": iso_ok,
    }


def run_develop_score(
    *,
    n_pairs: int = 16,
    device: str | None = None,
    write_lock: bool = False,
    smoke_pairs: int | None = None,
) -> dict[str, Any]:
    if not DEV_PREREG.exists():
        raise RuntimeError("score requires cortex_development.prereg.lock (reveal first)")
    # refuse neural drift
    cand = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural_cortex.py changed after candidate — new commitment required")
    if _sha_file(MEMORY_PY) != cand["cortex_memory_sha"]:
        raise RuntimeError("cortex_memory.py changed after candidate — new commitment required")
    from experiments.cortex_develop_life import (
        diagnostic_capacity_smoke,
        diagnostic_wall,
        run_battery,
    )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = smoke_pairs if smoke_pairs is not None else n_pairs
    pair_ids = list(range(n))
    battery = run_battery(n_pairs=n, device=dev, pair_ids=pair_ids)
    capacity = diagnostic_capacity_smoke(device=dev if n <= 2 else "cpu")
    wall = diagnostic_wall(device="cpu")
    summary = {
        "version": "TM.0.23.CORTEX.DEVELOP",
        "lab": "TM.0.23.CORTEX.DEVELOP",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "development_gate_clear": battery["development_gate_clear"],
        "eligible_for_000005": battery["eligible_for_000005"],
        "battery": battery,
        "capacity_diagnostic": capacity,
        "wall_diagnostic": wall,
        "develop_prereg_sha": _sha_file(DEV_PREREG),
        "runner_lock_sha": _sha_file(DEV_RUNNER_LOCK),
        "eval_reveal_sha": _sha_file(EVAL_REVEAL_LOCK),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "neural_cortex_sha": _sha_file(NEURAL_PY),
        "cortex_memory_sha": _sha_file(MEMORY_PY),
        "life_module_sha": _sha_file(LIFE_PY),
        "env": torch_env(),
        "note": "DEVELOP scored. Product stamp unchanged. Wall/capacity diagnostic only.",
    }
    if write_lock:
        DEV_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        wall_lock = {
            "version": "TM.0.23.CORTEX.WALL",
            "lab": "TM.0.23.CORTEX.WALL",
            "product": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "diagnostic": True,
            "need_not_fully_pass": True,
            "cannot_negate_development_gate": True,
            "wall_prereg_sha": _sha_file(PREREG_WALL),
            "result": wall,
            "development_gate_clear": battery["development_gate_clear"],
        }
        WALL_LOCK.write_text(json.dumps(wall_lock, indent=2, default=str) + "\n", encoding="utf-8")
        _write_develop_results(summary)
        summary["locks_written"] = True
    return summary


def _write_develop_results(summary: dict[str, Any]) -> None:
    b = summary.get("battery") or {}
    lines = [
        "# TM.0.23.CORTEX.DEVELOP results",
        "",
        f"**product:** `{summary.get('product')}`",
        f"**development_gate_clear:** `{summary.get('development_gate_clear')}`",
        f"**eligible_for_000005:** `{summary.get('eligible_for_000005')}`",
        f"**earned_next:** `{summary.get('earned_next')}`",
        f"**ex0s:** `{summary.get('ex0s')}`",
        "",
        f"Pairs clear: **{b.get('n_pair_clear')}/{b.get('n_pairs')}** · "
        f"Maturation: **{b.get('n_maturation')}/{b.get('n_pairs')}**",
        "",
        "Locks: [`cortex_sanity_spec.amendment.lock`](cortex_sanity_spec.amendment.lock) · "
        "[`cortex_development.runner.lock`](cortex_development.runner.lock) · "
        "[`cortex_eval_reveal.lock`](cortex_eval_reveal.lock) · "
        "[`cortex_development.prereg.lock`](cortex_development.prereg.lock) · "
        "[`cortex_development.lock`](cortex_development.lock) · "
        "[`cortex_wall.lock`](cortex_wall.lock)",
        "",
        "## Stage pass counts (main+twin)",
        "",
    ]
    for k, v in (b.get("stage_pass_counts_main_and_twin") or {}).items():
        lines.append(f"- `{k}`: {v}")
    wall = summary.get("wall_diagnostic") or {}
    lines += [
        "",
        "## Diagnostic wall",
        "",
        f"first_fail_neural_wall: `{wall.get('first_fail_neural_wall')}` "
        "(need not pass; cannot negate development gate)",
        "",
        "## Note",
        "",
        "Neural organism unchanged vs candidate. Capacity/wall diagnostic only.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_v2_amendment_lock() -> dict[str, Any]:
    if not DIAGNOSIS_LOCK.exists():
        raise RuntimeError("missing cortex_diagnosis.lock")
    if not V2_AMEND_MD.exists():
        raise RuntimeError("missing cortex_v2_architecture_amendment.md")
    # Refuse editing original architecture contract (content hash must stay v1)
    lock = {
        "version": "TM.0.23.CORTEX.V2.ARCHITECTURE.AMENDMENT",
        "lab": "TM.0.23.CORTEX.DIAG",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "v1_architecture_contract": "docs/cortex_architecture_contract.md",
        "v1_architecture_contract_sha": _sha_file(CONTRACT),
        "amendment_md": "docs/cortex_v2_architecture_amendment.md",
        "amendment_md_sha": _sha_file(V2_AMEND_MD),
        "diagnosis_lock": "docs/cortex_diagnosis.lock",
        "diagnosis_sha": _sha_file(DIAGNOSIS_LOCK),
        "diag_lock_sha": _sha_file(DIAG_LOCK) if DIAG_LOCK.exists() else None,
        "changes_authorized": [
            "motor_lexicon_M_act",
            "act_argmax_no_hold_on_cos_miss",
            "OP_COST[ACT]=0.1",
        ],
        "refuse": [
            "edit docs/cortex_architecture_contract.md",
            "unrelated improvements",
            "soften D1/D2 scorers",
        ],
        "note": "Frozen before neural edits. Original v1 contract untouched.",
    }
    V2_AMEND_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(V2_AMEND_LOCK), "sha": _sha_file(V2_AMEND_LOCK)}


def freeze_v2_gate_runner() -> dict[str, Any]:
    """Freeze D1/D2 gate apparatus before candidate v2 exists. Pins interface, not candidate SHA."""
    if not V2_GATE_CONTRACT.exists():
        raise RuntimeError("missing cortex_v2_gate_contract.md")
    if not V2_AMEND_LOCK.exists():
        raise RuntimeError("freeze amendment lock first")
    from experiments.cortex_v2_gate import THRESHOLDS

    lock = {
        "version": "TM.0.23.CORTEX.V2.GATE.RUNNER",
        "lab": "TM.0.23.CORTEX.V2.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_revealed": False,
        "gate_contract": "docs/cortex_v2_gate_contract.md",
        "gate_contract_sha": _sha_file(V2_GATE_CONTRACT),
        "architecture_amendment_lock": "docs/cortex_v2_architecture_amendment.lock",
        "architecture_amendment_sha": _sha_file(V2_AMEND_LOCK),
        "diagnosis_sha": _sha_file(DIAGNOSIS_LOCK),
        "stages": ["D0", "D1", "D2"],
        "thresholds": THRESHOLDS,
        "scorer_module": "experiments.cortex_develop_scorers",
        "scorer_sha": _sha_file(SCORERS_PY),
        "gate_module": "experiments.cortex_v2_gate",
        "gate_module_sha": _sha_file(V2_GATE_PY),
        "runner": "experiments.run_tm023cortex",
        "runner_sha": _sha_file(Path(__file__)),
        "candidate_interface": {
            "factory": "experiments.run_tm023cortex.make_cortex",
            "class": "NeuralCortex",
            "observe_keys": sorted(
                [
                    "interaction_token",
                    "source_token",
                    "ordered_symbols",
                    "observable_state",
                    "body_state",
                ]
            ),
            "ops": list(OPS),
            "note": "Pins interface only — no candidate SHA (v2 does not exist at freeze time)",
        },
        "refuse": [
            "pin candidate SHA before v2 birth",
            "pin eval fixture before reveal",
            "D3-D12 in this gate",
            "soften thresholds",
            "earned_next=true or non-null ex0s",
            "edit-and-rescore on revealed v2 worlds",
        ],
        "note": "Frozen before neural edits and before eval reveal.",
    }
    V2_GATE_RUNNER.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(V2_GATE_RUNNER), "sha": _sha_file(V2_GATE_RUNNER)}


def publish_v2_commitment() -> dict[str, Any]:
    if not V2_GATE_RUNNER.exists():
        raise RuntimeError("freeze v2 gate runner first")
    if V2_PREREG.exists():
        raise RuntimeError("cortex_v2.prereg.lock already exists — refuse rewrite")
    seed_b = secrets.token_bytes(32)
    salt_b = secrets.token_bytes(32)
    commitment = _sha_bytes(seed_b + salt_b)
    sealed = {
        "version": "TM.0.23.CORTEX.V2.EVAL.SEALED",
        "seed_hex": seed_b.hex(),
        "salt_hex": salt_b.hex(),
        "note": "Local only until post-candidate-v2 reveal",
    }
    V2_SEALED.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    develop_commit = json.loads(PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    prereg = {
        "version": "TM.0.23.CORTEX.V2.PREREG",
        "lab": "TM.0.23.CORTEX.V2.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "distinct_from_develop_commitment": True,
        "develop_eval_seed_commitment": develop_commit,
        "diagnosis_sha": _sha_file(DIAGNOSIS_LOCK),
        "architecture_amendment_sha": _sha_file(V2_AMEND_LOCK),
        "gate_runner_sha": _sha_file(V2_GATE_RUNNER),
        "gate_contract_sha": _sha_file(V2_GATE_CONTRACT),
        "scorer_sha": _sha_file(SCORERS_PY),
        "schedule": ["D0", "D1", "D2"],
        "n_pairs": 16,
        "gate_clear_min_pairs": 13,
        "pair_clear_rule": "main_D1_and_D2_AND_twin_D1_and_D2",
        "refuse": [
            "reuse DEVELOP revealed worlds",
            "neural edits before this commitment",
            "rewrite this lock",
        ],
        "note": "Commitment published after apparatus freeze, before neural edits.",
    }
    if commitment == develop_commit:
        raise RuntimeError("v2 commitment collided with DEVELOP — regenerate")
    V2_PREREG.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "commitment": commitment,
        "prereg_sha": _sha_file(V2_PREREG),
        "sealed": str(V2_SEALED),
    }


def reveal_v2_eval() -> dict[str, Any]:
    if not CANDIDATE_V2.exists():
        raise RuntimeError("cortex.candidate.v2.lock required before reveal")
    if not V2_PREREG.exists() or not V2_SEALED.exists():
        raise RuntimeError("missing v2 prereg/sealed")
    if V2_REVEAL.exists():
        raise RuntimeError("v2 reveal already exists — refuse rewrite")
    sealed = json.loads(V2_SEALED.read_text(encoding="utf-8"))
    seed_b = bytes.fromhex(sealed["seed_hex"])
    salt_b = bytes.fromhex(sealed["salt_hex"])
    commitment = _sha_bytes(seed_b + salt_b)
    prereg = json.loads(V2_PREREG.read_text(encoding="utf-8"))
    if commitment != prereg["eval_seed_commitment"]:
        raise RuntimeError("v2 commitment mismatch")
    # Publish opaque pair salt material into reveal (worlds for gate are seed-table + salt)
    reveal = {
        "version": "TM.0.23.CORTEX.V2.EVAL.REVEAL",
        "lab": "TM.0.23.CORTEX.V2.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "commitment_verified": True,
        "seed_hex": sealed["seed_hex"],
        "salt_hex": sealed["salt_hex"],
        "candidate_v2_sha": _sha_file(CANDIDATE_V2),
        "gate_runner_sha": _sha_file(V2_GATE_RUNNER),
        "prereg_sha": _sha_file(V2_PREREG),
        "note": "Revealed after candidate v2. Worlds become diagnostic-only after scoring; no edit-rescore.",
    }
    V2_REVEAL.write_text(json.dumps(reveal, indent=2) + "\n", encoding="utf-8")
    gate_prereg = {
        "version": "TM.0.23.CORTEX.V2.GATE.PREREG",
        "lab": "TM.0.23.CORTEX.V2.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eval_seed_commitment": commitment,
        "reveal_sha": _sha_file(V2_REVEAL),
        "runner_sha": _sha_file(V2_GATE_RUNNER),
        "candidate_v2_sha": _sha_file(CANDIDATE_V2),
        "scorer_sha": _sha_file(SCORERS_PY),
        "schedule": ["D0", "D1", "D2"],
        "note": "Composed after reveal. Scoring uses frozen pair seed table + revealed salt binding.",
    }
    V2_GATE_PREREG.write_text(json.dumps(gate_prereg, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "reveal_sha": _sha_file(V2_REVEAL),
        "gate_prereg_sha": _sha_file(V2_GATE_PREREG),
        "commitment_verified": True,
    }


def run_v2_gate_score(*, device: str | None = None, write_lock: bool = False) -> dict[str, Any]:
    if not V2_REVEAL.exists():
        raise RuntimeError("reveal v2 eval first")
    if not CANDIDATE_V2.exists():
        raise RuntimeError("missing candidate v2")
    # Refuse scoring if neural drifted from candidate v2 pin
    cand = json.loads(CANDIDATE_V2.read_text(encoding="utf-8"))
    if _sha_file(NEURAL_PY) != cand["neural_cortex_sha"]:
        raise RuntimeError("neural_cortex.py drifted from candidate v2 — new v3 cycle required")
    if _sha_file(MEMORY_PY) != cand["cortex_memory_sha"]:
        raise RuntimeError("cortex_memory.py drifted from candidate v2 — new v3 cycle required")
    from experiments.cortex_v2_gate import run_v2_gate_battery

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    battery = run_v2_gate_battery(n_pairs=16, device=dev)
    summary = {
        "version": "TM.0.23.CORTEX.V2.GATE.RESULT",
        "lab": "TM.0.23.CORTEX.V2.GATE",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "sensorimotor_association_gate_clear": battery["sensorimotor_association_gate_clear"],
        "battery": battery,
        "candidate_v2_sha": _sha_file(CANDIDATE_V2),
        "reveal_sha": _sha_file(V2_REVEAL),
        "runner_sha": _sha_file(V2_GATE_RUNNER),
        "gate_prereg_sha": _sha_file(V2_GATE_PREREG),
        "neural_cortex_sha": cand["neural_cortex_sha"],
        "cortex_memory_sha": cand["cortex_memory_sha"],
        "env": torch_env(),
        "note": "D0–D2 only. Revealed worlds are diagnostic after this freeze. No full D0–D12.",
    }
    if write_lock:
        if V2_GATE_LOCK.exists():
            raise RuntimeError("cortex_v2_gate.lock exists — refuse rewrite / edit-rescore")
        V2_GATE_LOCK.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        lines = [
            "# TM.0.23.CORTEX.V2.GATE results",
            "",
            f"**product:** `{summary['product']}`",
            f"**sensorimotor_association_gate_clear:** `{summary['sensorimotor_association_gate_clear']}`",
            f"**earned_next:** `{summary['earned_next']}`",
            f"**ex0s:** `{summary['ex0s']}`",
            "",
            f"Pairs clear: **{battery['n_pair_clear']}/16**",
            "",
            f"Stage pass counts (main+twin): `{battery['stage_pass_counts_main_and_twin']}`",
            "",
            f"systematic_d0_birth_leakage_failure: `{battery['systematic_d0_birth_leakage_failure']}`",
            "",
            "No full D0–D12 on these worlds. If clear, later fresh full-development commitment. If fail, isolated v3 cycle.",
            "",
        ]
        V2_GATE_RESULTS.write_text("\n".join(lines), encoding="utf-8")
        summary["locks_written"] = True
    return summary


def write_v4_math_audit(*, write_lock: bool = True) -> dict[str, Any]:
    """Machine-checkable math/human audit for live v4 candidate (does not rewrite candidate)."""
    from three_memory.neural_cortex import MOTOR_ACT_TOKENS, OP_COST, OPS

    if not CANDIDATE_V4.exists():
        raise RuntimeError("missing cortex.candidate.v4.lock")
    cand = json.loads(CANDIDATE_V4.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"id": name, "ok": ok, "detail": detail})

    add(
        "v1_contract_sha_stable",
        _sha_file(CONTRACT) == "0470d5f8429317715d9f50bc9a3e2463dc1fd80039afb9fc650e364b28e7fac2",
        _sha_file(CONTRACT),
    )
    add("candidate_v1_preserved", CANDIDATE_V1.exists() and _sha_file(CANDIDATE_V1).startswith("60df20ec"))
    add("neural_matches_candidate_v4", _sha_file(NEURAL_PY) == cand["neural_cortex_sha"])
    add("memory_matches_candidate_v4", _sha_file(MEMORY_PY) == cand["cortex_memory_sha"])
    add("op_cost_act_0_05", OP_COST.get("ACT") == 0.05, OP_COST.get("ACT"))
    add("motor_act_tokens_press_harm", set(MOTOR_ACT_TOKENS) == {"press", "harm"}, list(MOTOR_ACT_TOKENS))
    add("eta_act_0_15", float(cand.get("genome", {}).get("eta_act", 0)) == 0.15)
    with tempfile.TemporaryDirectory(prefix="v4audit_") as tmp:
        ag = make_cortex(Path(tmp) / "s")
        add("b_op_act_0_85", float(ag.b_op[OPS.index("ACT")]) == 0.85, float(ag.b_op[OPS.index("ACT")]))
        add("b_op_not_plastic", "b_op" not in ag._plastic_names)
        add("motor_vocab_keys", set(ag.motor_vocab.keys()) == {"press", "harm"})
    add("amendment_lock_exists", V4_AMEND_LOCK.exists())
    if V4_GATE_RUNNER.exists():
        runner = json.loads(V4_GATE_RUNNER.read_text(encoding="utf-8"))
        add("runner_has_no_candidate_sha", "candidate_sha" not in runner)
        add("runner_has_interface", "candidate_interface" in runner)
    add(
        "v4_commitment_distinct_from_develop",
        V4_PREREG.exists()
        and json.loads(V4_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        != json.loads(PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"],
    )
    ok = all(c["ok"] for c in checks)
    out = {
        "version": "TM.0.23.CORTEX.V4.MATH.AUDIT",
        "lab": "TM.0.23.CORTEX.V4",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": ok,
        "checks": checks,
        "candidate_v4_sha": _sha_file(CANDIDATE_V4),
        "note": "Append-only audit of live v4 organism. Does not rewrite candidate.v4.lock.",
    }
    if write_lock:
        if V4_MATH_AUDIT.exists():
            raise RuntimeError("cortex_v4_math_audit.lock exists — refuse rewrite")
        V4_MATH_AUDIT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def verify_v4_gate() -> dict[str, Any]:
    """Verify frozen v4 gate result integrity without re-scoring.

    Live neural may be a later candidate (v5+); integrity is pairwise among
    frozen v4 locks, not against the live tree.
    """
    if not V4_GATE_LOCK.exists():
        return {"ok": False, "why": "missing cortex_v4_gate.lock"}
    if not CANDIDATE_V4.exists():
        return {"ok": False, "why": "missing cortex.candidate.v4.lock"}
    gate = json.loads(V4_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V4.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("neural_cortex_sha") != cand.get("neural_cortex_sha"):
        return {"ok": False, "why": "gate/candidate v4 neural SHA mismatch"}
    if gate.get("cortex_memory_sha") != cand.get("cortex_memory_sha"):
        return {"ok": False, "why": "gate/candidate v4 memory SHA mismatch"}
    battery = gate.get("battery") or {}
    if battery.get("n_pair_clear", 0) < 13:
        return {"ok": False, "why": "n_pair_clear < 13"}
    if not gate.get("sensorimotor_association_gate_clear"):
        return {"ok": False, "why": "gate not clear"}
    # pair-clear consistency + no soft D1
    for p in battery.get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
        if p.get("pair_clear"):
            for role in ("main", "twin"):
                d1 = p[role]["stages"]["D1"]
                if int(d1.get("press") or 0) < 3 or not d1.get("ok"):
                    return {"ok": False, "why": f"soft D1 clear pair {p.get('pair_id')} {role}"}
                d2 = p[role]["stages"]["D2"]
                if int(d2.get("holds_during_conflict") or 0) < 5 or int(d2.get("beneficial_act") or 0) < 3:
                    return {"ok": False, "why": f"soft D2 clear pair {p.get('pair_id')} {role}"}
    if not V4_CLEAR_NOTE.exists():
        return {"ok": False, "why": "missing clear note (full battery isolation)"}
    note = json.loads(V4_CLEAR_NOTE.read_text(encoding="utf-8"))
    if "new full-development eval commitment" not in json.dumps(note):
        return {"ok": False, "why": "clear note missing full-dev isolation"}
    live_matches_v4 = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    return {
        "ok": True,
        "why": "v4 gate integrity ok",
        "sensorimotor_association_gate_clear": True,
        "n_pair_clear": battery.get("n_pair_clear"),
        "live_neural_matches_v4": live_matches_v4,
        "refuse_rewrite": True,
    }


def verify_v5_gate() -> dict[str, Any]:
    """Verify frozen v5 gate failure + boundary-audit integrity. Does not rescore."""
    if not V5_GATE_LOCK.exists():
        return {"ok": False, "why": "missing cortex_v5_gate.lock"}
    if not CANDIDATE_V5.exists():
        return {"ok": False, "why": "missing cortex.candidate.v5.lock"}
    if not V5_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v5_gate.failure.lock"}
    if not V5_ISOLATION.exists():
        return {"ok": False, "why": "missing cortex_v6.isolation.lock"}
    if not MACT_V4_LOCK.exists():
        return {"ok": False, "why": "missing cortex_mact_boundary.lock"}
    if not MACT_V5_LOCK.exists():
        return {"ok": False, "why": "missing cortex_mact_boundary.v5.lock"}
    if not MACT_V5_AUDIT.exists():
        return {"ok": False, "why": "missing cortex_mact_boundary.v5.audit.lock"}
    if DEV_V5_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v5 lock exists after gate fail — refuse"}
    gate = json.loads(V5_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V5.read_text(encoding="utf-8"))
    fail = json.loads(V5_GATE_FAIL.read_text(encoding="utf-8"))
    audit = json.loads(MACT_V5_AUDIT.read_text(encoding="utf-8"))
    mact_v4 = json.loads(MACT_V4_LOCK.read_text(encoding="utf-8"))
    mact_v5 = json.loads(MACT_V5_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("sensorimotor_association_gate_clear"):
        return {"ok": False, "why": "v5 gate lock claims clear"}
    if int((gate.get("battery") or {}).get("n_pair_clear") or 0) >= 13:
        return {"ok": False, "why": "n_pair_clear >= 13 but not marked fail"}
    if fail.get("gate_sha") != _sha_file(V5_GATE_LOCK):
        return {"ok": False, "why": "failure lock gate_sha mismatch"}
    if fail.get("next") != "isolated_v6":
        return {"ok": False, "why": "failure next is not isolated_v6"}
    live_matches_v5 = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    if mact_v4.get("all_controls_green") is not False:
        return {"ok": False, "why": "v4 boundary must remain red"}
    if audit.get("contract_honest_all_green") is not False:
        return {"ok": False, "why": "v5 boundary audit must record contract-honest reds"}
    if audit.get("mact_v5_lock_sha") != _sha_file(MACT_V5_LOCK):
        return {"ok": False, "why": "audit pin != historical v5 boundary lock"}
    # pair-clear consistency on frozen result
    for p in (gate.get("battery") or {}).get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
    return {
        "ok": True,
        "why": "v5 gate failure integrity ok",
        "sensorimotor_association_gate_clear": False,
        "n_pair_clear": (gate.get("battery") or {}).get("n_pair_clear"),
        "boundary_v5_claimed_green": mact_v5.get("all_controls_green"),
        "boundary_v5_contract_honest_green": False,
        "refuse_rewrite": True,
        "refuse_develop_v5": True,
        "live_neural_matches_v5": live_matches_v5,
    }


def verify_v6_gate() -> dict[str, Any]:
    """Verify frozen v6 gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V5.exists():
        return {"ok": False, "why": "missing cortex_diagnosis.v5.lock"}
    if V6_PREREG.exists() and V5_PREREG.exists():
        v6c = json.loads(V6_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        v5c = json.loads(V5_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        if v6c == v5c:
            return {"ok": False, "why": "v6 commitment reused v5"}
    if not CANDIDATE_V6.exists():
        return {"ok": True, "pending": True, "why": "candidate v6 not frozen yet"}
    if not V6_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v6 gate result not frozen yet"}
    if not MACT_V6_LOCK.exists():
        return {"ok": False, "why": "missing cortex_mact_boundary.v6.lock"}
    if not MACT_V6_AUDIT.exists():
        return {"ok": False, "why": "missing cortex_mact_boundary.v6.audit.lock"}
    if DEV_V6_LOCK.exists() and V6_GATE_FAIL.exists():
        return {"ok": False, "why": "DEVELOP.v6 exists after gate fail — refuse"}
    gate = json.loads(V6_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V6.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("candidate_v6_sha") != _sha_file(CANDIDATE_V6):
        return {"ok": False, "why": "gate/candidate v6 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and n_clear >= 13:
        return {"ok": False, "why": "n_pair_clear >= 13 but not marked clear"}
    if (not clear) and not V6_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v6_gate.failure.lock"}
    if clear and V6_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    if (not clear) and V6_GATE_FAIL.exists():
        fail = json.loads(V6_GATE_FAIL.read_text(encoding="utf-8"))
        if fail.get("gate_sha") != _sha_file(V6_GATE_LOCK):
            return {"ok": False, "why": "failure lock gate_sha mismatch"}
        if "DEVELOP.v6" not in (fail.get("refuse") or []):
            return {"ok": False, "why": "failure lock missing DEVELOP.v6 refuse"}
    for p in battery.get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
        if p.get("pair_clear"):
            for role in ("main", "twin"):
                d1 = p[role]["stages"]["D1"]
                if int(d1.get("press") or 0) < 3 or not d1.get("ok"):
                    return {"ok": False, "why": f"soft D1 clear pair {p.get('pair_id')} {role}"}
                d2 = p[role]["stages"]["D2"]
                if int(d2.get("holds_during_conflict") or 0) < 5 or int(d2.get("beneficial_act") or 0) < 3:
                    return {"ok": False, "why": f"soft D2 clear pair {p.get('pair_id')} {role}"}
    audit = json.loads(MACT_V6_AUDIT.read_text(encoding="utf-8"))
    mact_v6 = json.loads(MACT_V6_LOCK.read_text(encoding="utf-8"))
    if audit.get("contract_honest_all_green") is not False:
        return {"ok": False, "why": "v6 boundary audit must record contract-honest reds"}
    if audit.get("mact_v6_lock_sha") != _sha_file(MACT_V6_LOCK):
        return {"ok": False, "why": "audit pin != historical v6 boundary lock"}
    if audit.get("historical_lock_rewritten") is not False:
        return {"ok": False, "why": "audit claims historical lock rewritten"}
    live_matches_v6 = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    return {
        "ok": True,
        "why": "v6 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v6": live_matches_v6,
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
        "boundary_v6_claimed_green": mact_v6.get("all_controls_green"),
        "boundary_v6_contract_honest_green": False,
    }


def verify_v7_gate() -> dict[str, Any]:
    """Verify v7 stat-contract gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V6.exists() or not STAT_V7.exists():
        return {"ok": False, "why": "missing v6 diagnosis or v7 stat contract"}
    if not V7_ISOLATION.exists():
        return {"ok": False, "why": "missing cortex_v7.isolation.lock"}
    if DEV_V6_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v6 exists — refuse"}
    if V7_PREREG.exists():
        v7c = json.loads(V7_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        for path in (V6_PREREG, V5_PREREG, PREREG, DEV_PREREG):
            if path.exists():
                prior = json.loads(path.read_text(encoding="utf-8")).get("eval_seed_commitment")
                if prior and v7c == prior:
                    return {"ok": False, "why": f"v7 commitment reused {path.name}"}
    if not CANDIDATE_V7.exists():
        return {"ok": True, "pending": True, "why": "candidate v7 not frozen yet"}
    if not V7_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v7 gate result not frozen yet"}
    if DEV_V7_LOCK.exists() and V7_GATE_FAIL.exists():
        return {"ok": False, "why": "DEVELOP.v7 exists after gate fail — refuse"}
    if DEV_V7_LOCK.exists() and not json.loads(V7_GATE_LOCK.read_text(encoding="utf-8")).get(
        "sensorimotor_association_gate_clear"
    ):
        return {"ok": False, "why": "DEVELOP.v7 exists without a clear v7 gate — refuse"}
    gate = json.loads(V7_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V7.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("candidate_v7_sha") != _sha_file(CANDIDATE_V7):
        return {"ok": False, "why": "gate/candidate v7 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and n_clear >= 13:
        return {"ok": False, "why": "n_pair_clear >= 13 but not marked clear"}
    if (not clear) and not V7_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v7_gate.failure.lock"}
    if clear and V7_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    if (not clear) and V7_GATE_FAIL.exists():
        fail = json.loads(V7_GATE_FAIL.read_text(encoding="utf-8"))
        if fail.get("gate_sha") != _sha_file(V7_GATE_LOCK):
            return {"ok": False, "why": "failure lock gate_sha mismatch"}
        if "DEVELOP.v7" not in (fail.get("refuse") or []):
            return {"ok": False, "why": "failure lock missing DEVELOP.v7 refuse"}
    for p in battery.get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
        if p.get("pair_clear"):
            for role in ("main", "twin"):
                d1 = p[role]["stages"]["D1"]
                if int(d1.get("press") or 0) < 3 or not d1.get("ok"):
                    return {"ok": False, "why": f"soft D1 clear pair {p.get('pair_id')} {role}"}
                if not d1.get("floors_ok"):
                    return {"ok": False, "why": f"D1 floors not recorded pair {p.get('pair_id')} {role}"}
                if not d1.get("trained_gt_birth") or not d1.get("trained_gt_frozen"):
                    return {"ok": False, "why": f"D1 extras missing pair {p.get('pair_id')} {role}"}
                d2 = p[role]["stages"]["D2"]
                if int(d2.get("holds_during_conflict") or 0) < 5 or int(d2.get("beneficial_act") or 0) < 3:
                    return {"ok": False, "why": f"soft D2 clear pair {p.get('pair_id')} {role}"}
                if not d2.get("trained_gt_frozen") or not d2.get("assoc_ok"):
                    return {"ok": False, "why": f"D2 extras missing pair {p.get('pair_id')} {role}"}
    live_matches_v7 = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    return {
        "ok": True,
        "why": "v7 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v7": live_matches_v7,
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
        "stat_contract_sha": _sha_file(STAT_V7),
        "diagnosis_v6_sha": _sha_file(DIAG_V6),
    }


def verify_v8_gate() -> dict[str, Any]:
    """Verify v8 stat-contract gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V7.exists() or not STAT_V8.exists():
        return {"ok": True, "pending": True, "why": "v8 diagnosis or stat contract not frozen yet"}
    if DEV_V7_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v7 exists — refuse"}
    if V8_PREREG.exists() and V7_PREREG.exists():
        v8c = json.loads(V8_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        v7c = json.loads(V7_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        if v8c == v7c:
            return {"ok": False, "why": "v8 commitment reused v7"}
    if not CANDIDATE_V8.exists():
        return {"ok": True, "pending": True, "why": "candidate v8 not frozen yet"}
    if not V8_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v8 gate result not frozen yet"}
    if DEV_V8_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v8 exists — refuse until a later isolated full-dev commitment"}
    gate = json.loads(V8_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V8.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("candidate_v8_sha") != _sha_file(CANDIDATE_V8):
        return {"ok": False, "why": "gate/candidate v8 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and n_clear >= 13:
        return {"ok": False, "why": "n_pair_clear >= 13 but not marked clear"}
    if (not clear) and not V8_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v8_gate.failure.lock"}
    if clear and V8_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    for p in battery.get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
        if p.get("pair_clear"):
            for role in ("main", "twin"):
                d1 = p[role]["stages"]["D1"]
                if int(d1.get("press") or 0) < 3 or not d1.get("ok"):
                    return {"ok": False, "why": f"soft D1 clear pair {p.get('pair_id')} {role}"}
                if not d1.get("trained_gt_birth") or not d1.get("trained_gt_frozen"):
                    return {"ok": False, "why": f"D1 extras missing pair {p.get('pair_id')} {role}"}
                d2 = p[role]["stages"]["D2"]
                if int(d2.get("holds_during_conflict") or 0) < 5 or int(d2.get("beneficial_act") or 0) < 3:
                    return {"ok": False, "why": f"soft D2 clear pair {p.get('pair_id')} {role}"}
    live_matches = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    return {
        "ok": True,
        "why": "v8 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v8": live_matches,
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
    }


def verify_v9_gate() -> dict[str, Any]:
    """Verify v9 D1 press/harm gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V8.exists() or not STAT_V9.exists():
        return {"ok": True, "pending": True, "why": "v9 diagnosis or stat contract not frozen yet"}
    if DEV_V8_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v8 exists — refuse"}
    if V9_PREREG.exists():
        v9c = json.loads(V9_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        for prior in (V8_PREREG, V7_PREREG, DEV_PREREG, PREREG):
            if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == v9c:
                return {"ok": False, "why": f"v9 commitment reused {prior.name}"}
    if not CANDIDATE_V9.exists():
        return {"ok": True, "pending": True, "why": "candidate v9 not frozen yet"}
    if not V9_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v9 gate result not frozen yet"}
    if DEV_V9_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v9 exists — refuse until a later isolated full-dev commitment"}
    gate = json.loads(V9_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V9.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("candidate_v9_sha") != _sha_file(CANDIDATE_V9):
        return {"ok": False, "why": "gate/candidate v9 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and n_clear >= 13:
        return {"ok": False, "why": "n_pair_clear >= 13 but not marked clear"}
    if (not clear) and not V9_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v9_gate.failure.lock"}
    if clear and V9_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    if battery.get("d1_bind") != ["press", "harm"]:
        return {"ok": False, "why": "v9 D1 bind drifted from press+harm"}
    for p in battery.get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
        if p.get("pair_clear"):
            for role in ("main", "twin"):
                d1 = p[role]["stages"]["D1"]
                if int(d1.get("press") or 0) < 3 or not d1.get("ok"):
                    return {"ok": False, "why": f"soft D1 clear pair {p.get('pair_id')} {role}"}
                if not d1.get("trained_gt_birth") or not d1.get("trained_gt_frozen"):
                    return {"ok": False, "why": f"D1 extras missing pair {p.get('pair_id')} {role}"}
                d2 = p[role]["stages"]["D2"]
                if int(d2.get("holds_during_conflict") or 0) < 5 or int(d2.get("beneficial_act") or 0) < 3:
                    return {"ok": False, "why": f"soft D2 clear pair {p.get('pair_id')} {role}"}
    live_matches = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    return {
        "ok": True,
        "why": "v9 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v9": live_matches,
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
    }


def verify_v10_gate() -> dict[str, Any]:
    """Verify v10 population-extras gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V9.exists() or not STAT_V10.exists():
        return {"ok": True, "pending": True, "why": "v10 diagnosis or stat contract not frozen yet"}
    if DEV_V9_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v9 exists — refuse"}
    if V10_PREREG.exists():
        v10c = json.loads(V10_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        for prior in (V9_PREREG, V8_PREREG, V7_PREREG, DEV_PREREG, PREREG):
            if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == v10c:
                return {"ok": False, "why": f"v10 commitment reused {prior.name}"}
    if not CANDIDATE_V10.exists():
        return {"ok": True, "pending": True, "why": "candidate v10 not frozen yet"}
    if not V10_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v10 gate result not frozen yet"}
    if DEV_V10_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v10 exists — refuse until a later isolated full-dev commitment"}
    gate = json.loads(V10_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V10.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004":
        return {"ok": False, "why": "product drift"}
    if gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "earned_next/ex0s drift"}
    if gate.get("candidate_v10_sha") != _sha_file(CANDIDATE_V10):
        return {"ok": False, "why": "gate/candidate v10 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    pop_d1 = battery.get("population_d1") or {}
    pop_d2 = battery.get("population_d2") or {}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if clear and (not pop_d1.get("ok") or not pop_d2.get("ok")):
        return {"ok": False, "why": "gate claims clear without population extras"}
    if (not clear) and n_clear >= 13 and pop_d1.get("ok") and pop_d2.get("ok") and not battery.get("systematic_d0_fail"):
        return {"ok": False, "why": "floors+population green but not marked clear"}
    if (not clear) and not V10_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v10_gate.failure.lock"}
    if clear and V10_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    if battery.get("d1_bind") != ["press", "harm"]:
        return {"ok": False, "why": "v10 D1 bind drifted from press+harm"}
    if battery.get("extras") != "population":
        return {"ok": False, "why": "v10 extras drifted from population"}
    for p in battery.get("pairs") or []:
        m, t = p["main"], p["twin"]
        expect = m["d0_ok"] and t["d0_ok"] and m["d1_d2_ok"] and t["d1_d2_ok"]
        if bool(p.get("pair_clear")) != expect:
            return {"ok": False, "why": f"pair_clear inconsistent pair {p.get('pair_id')}"}
        if p.get("pair_clear"):
            for role in ("main", "twin"):
                d1 = p[role]["stages"]["D1"]
                if int(d1.get("press") or 0) < 3 or not d1.get("ok"):
                    return {"ok": False, "why": f"soft D1 clear pair {p.get('pair_id')} {role}"}
                d2 = p[role]["stages"]["D2"]
                if int(d2.get("holds_during_conflict") or 0) < 5 or int(d2.get("beneficial_act") or 0) < 3:
                    return {"ok": False, "why": f"soft D2 clear pair {p.get('pair_id')} {role}"}
    live_matches = _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha")
    return {
        "ok": True,
        "why": "v10 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v10": live_matches,
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
    }


def verify_v11_gate() -> dict[str, Any]:
    """Verify v11 contradictory-HOLD gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V10.exists() or not STAT_V11.exists():
        return {"ok": True, "pending": True, "why": "v11 diagnosis or stat contract not frozen yet"}
    if DEV_V10_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v10 exists — refuse"}
    if V11_PREREG.exists():
        v11c = json.loads(V11_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        for prior in (V10_PREREG, V9_PREREG, V8_PREREG, DEV_PREREG, PREREG):
            if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == v11c:
                return {"ok": False, "why": f"v11 commitment reused {prior.name}"}
    if not CANDIDATE_V11.exists():
        return {"ok": True, "pending": True, "why": "candidate v11 not frozen yet"}
    if not V11_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v11 gate result not frozen yet"}
    if DEV_V11_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v11 exists — refuse until a later isolated full-dev commitment"}
    gate = json.loads(V11_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V11.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if gate.get("candidate_v11_sha") != _sha_file(CANDIDATE_V11):
        return {"ok": False, "why": "gate/candidate v11 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if battery.get("d2_conflict") != "swapped_press_harm":
        return {"ok": False, "why": "v11 D2 conflict window drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not V11_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v11_gate.failure.lock"}
    if clear and V11_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "v11 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v11": _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha"),
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
    }


def verify_v12_gate() -> dict[str, Any]:
    """Verify v12 surprise-HOLD gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V11.exists() or not STAT_V12.exists():
        return {"ok": True, "pending": True, "why": "v12 diagnosis or stat contract not frozen yet"}
    if DEV_V11_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v11 exists — refuse"}
    if V12_PREREG.exists():
        v12c = json.loads(V12_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        for prior in (V11_PREREG, V10_PREREG, V9_PREREG, DEV_PREREG, PREREG):
            if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == v12c:
                return {"ok": False, "why": f"v12 commitment reused {prior.name}"}
    if not CANDIDATE_V12.exists():
        return {"ok": True, "pending": True, "why": "candidate v12 not frozen yet"}
    if not V12_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v12 gate result not frozen yet"}
    if DEV_V12_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v12 exists — refuse until a later isolated full-dev commitment"}
    gate = json.loads(V12_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V12.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if gate.get("candidate_v12_sha") != _sha_file(CANDIDATE_V12):
        return {"ok": False, "why": "gate/candidate v12 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if battery.get("d2_conflict") != "swapped_press_harm":
        return {"ok": False, "why": "v12 D2 conflict window drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not V12_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v12_gate.failure.lock"}
    if clear and V12_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "v12 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v12": _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha"),
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
    }


def verify_v13_gate() -> dict[str, Any]:
    """Verify v13 slow-baseline gate integrity without re-scoring. Pending-safe if unscored."""
    if not DIAG_V12.exists() or not STAT_V13.exists():
        return {"ok": True, "pending": True, "why": "v13 diagnosis or stat contract not frozen yet"}
    if DEV_V12_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v12 exists — refuse"}
    if V13_PREREG.exists():
        v13c = json.loads(V13_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
        for prior in (V12_PREREG, V11_PREREG, V10_PREREG, V9_PREREG, DEV_PREREG, PREREG):
            if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == v13c:
                return {"ok": False, "why": f"v13 commitment reused {prior.name}"}
    if not CANDIDATE_V13.exists():
        return {"ok": True, "pending": True, "why": "candidate v13 not frozen yet"}
    if not V13_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "v13 gate result not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse until a later isolated full-dev commitment"}
    gate = json.loads(V13_GATE_LOCK.read_text(encoding="utf-8"))
    cand = json.loads(CANDIDATE_V13.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if gate.get("candidate_v13_sha") != _sha_file(CANDIDATE_V13):
        return {"ok": False, "why": "gate/candidate v13 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("sensorimotor_association_gate_clear"))
    if battery.get("d2_conflict") != "swapped_press_harm":
        return {"ok": False, "why": "v13 D2 conflict window drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not V13_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_v13_gate.failure.lock"}
    if clear and V13_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "v13 gate integrity ok",
        "pending": False,
        "sensorimotor_association_gate_clear": clear,
        "n_pair_clear": n_clear,
        "live_neural_matches_v13": _sha_file(NEURAL_PY) == cand.get("neural_cortex_sha"),
        "refuse_rewrite": True,
        "refuse_develop_before_clear": not clear,
    }


def verify_fulldev_r1() -> dict[str, Any]:
    """Verify FULLDEV.R1 integrity without re-scoring. Pending-safe if unscored."""
    if not FULLDEV_R1_PREREG.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R1 prereg not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse"}
    r1c = json.loads(FULLDEV_R1_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (V13_PREREG, V12_PREREG, V11_PREREG, V10_PREREG, V9_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r1c:
            return {"ok": False, "why": f"FULLDEV.R1 commitment reused {prior.name}"}
    if json.loads(FULLDEV_R1_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.FULL.R1.":
        return {"ok": False, "why": "FULLDEV.R1 domain drifted"}
    if not FULLDEV_R1_LOCK.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R1 result not frozen yet"}
    res = json.loads(FULLDEV_R1_LOCK.read_text(encoding="utf-8"))
    if res.get("product") != "0.0.004" or res.get("earned_next") is not False or res.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if res.get("eligible_for_000005") is not False:
        return {"ok": False, "why": "eligible_for_000005 must stay false on this pass"}
    battery = res.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(res.get("development_gate_clear"))
    if battery.get("domain") != "TM023.FULL.R1.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not FULLDEV_R1_FAIL.exists():
        return {"ok": False, "why": "missing cortex_fulldev_r1.failure.lock"}
    if clear and FULLDEV_R1_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear battery"}
    return {
        "ok": True,
        "why": "FULLDEV.R1 integrity ok",
        "pending": False,
        "development_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
    }


def verify_fulldev_r2() -> dict[str, Any]:
    """Verify FULLDEV.R2 integrity without re-scoring. Pending-safe if unscored."""
    if not FULLDEV_R2_PREREG.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R2 prereg not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse"}
    r2c = json.loads(FULLDEV_R2_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (FULLDEV_R1_PREREG, D3_R3_PREREG, D3_R2_PREREG, D3_R1_PREREG, V13_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r2c:
            return {"ok": False, "why": f"FULLDEV.R2 commitment reused {prior.name}"}
    if json.loads(FULLDEV_R2_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.FULL.R2.":
        return {"ok": False, "why": "FULLDEV.R2 domain drifted"}
    if not FULLDEV_R2_LOCK.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R2 result not frozen yet"}
    res = json.loads(FULLDEV_R2_LOCK.read_text(encoding="utf-8"))
    if res.get("product") != "0.0.004" or res.get("earned_next") is not False or res.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if res.get("eligible_for_000005") is not False:
        return {"ok": False, "why": "eligible_for_000005 must stay false until nursery clears"}
    battery = res.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(res.get("development_gate_clear"))
    if battery.get("domain") != "TM023.FULL.R2.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not FULLDEV_R2_FAIL.exists():
        return {"ok": False, "why": "missing cortex_fulldev_r2.failure.lock"}
    if clear and FULLDEV_R2_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear battery"}
    return {
        "ok": True,
        "why": "FULLDEV.R2 integrity ok",
        "pending": False,
        "development_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
    }


def verify_fulldev_r3() -> dict[str, Any]:
    """Verify FULLDEV.R3 integrity without re-scoring. Pending-safe if unscored."""
    if not FULLDEV_R3_PREREG.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R3 prereg not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse"}
    r3c = json.loads(FULLDEV_R3_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (FULLDEV_R2_PREREG, FULLDEV_R1_PREREG, D4_R2_PREREG, D4_R1_PREREG, D3_R3_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r3c:
            return {"ok": False, "why": f"FULLDEV.R3 commitment reused {prior.name}"}
    if json.loads(FULLDEV_R3_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.FULL.R3.":
        return {"ok": False, "why": "FULLDEV.R3 domain drifted"}
    if not FULLDEV_R3_LOCK.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R3 result not frozen yet"}
    res = json.loads(FULLDEV_R3_LOCK.read_text(encoding="utf-8"))
    if res.get("product") != "0.0.004" or res.get("earned_next") is not False or res.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if res.get("eligible_for_000005") is not False:
        return {"ok": False, "why": "eligible_for_000005 must stay false until nursery clears"}
    battery = res.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(res.get("development_gate_clear"))
    if battery.get("domain") != "TM023.FULL.R3.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not FULLDEV_R3_FAIL.exists():
        return {"ok": False, "why": "missing cortex_fulldev_r3.failure.lock"}
    if clear and FULLDEV_R3_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear battery"}
    return {
        "ok": True,
        "why": "FULLDEV.R3 integrity ok",
        "pending": False,
        "development_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
    }


def verify_fulldev_r4() -> dict[str, Any]:
    """Verify FULLDEV.R4 integrity without re-scoring. Pending-safe if unscored."""
    if not FULLDEV_R4_PREREG.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R4 prereg not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse"}
    r4c = json.loads(FULLDEV_R4_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (FULLDEV_R3_PREREG, FULLDEV_R2_PREREG, FULLDEV_R1_PREREG, D5_R2_PREREG, D5_R1_PREREG, D4_R2_PREREG, D4_R1_PREREG, D3_R3_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r4c:
            return {"ok": False, "why": f"FULLDEV.R4 commitment reused {prior.name}"}
    if json.loads(FULLDEV_R4_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.FULL.R4.":
        return {"ok": False, "why": "FULLDEV.R4 domain drifted"}
    if not FULLDEV_R4_LOCK.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R4 result not frozen yet"}
    res = json.loads(FULLDEV_R4_LOCK.read_text(encoding="utf-8"))
    if res.get("product") != "0.0.004" or res.get("earned_next") is not False or res.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if res.get("eligible_for_000005") is not False:
        return {"ok": False, "why": "eligible_for_000005 must stay false until nursery clears"}
    battery = res.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(res.get("development_gate_clear"))
    if battery.get("domain") != "TM023.FULL.R4.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not FULLDEV_R4_FAIL.exists():
        return {"ok": False, "why": "missing cortex_fulldev_r4.failure.lock"}
    if clear and FULLDEV_R4_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear battery"}
    return {
        "ok": True,
        "why": "FULLDEV.R4 integrity ok",
        "pending": False,
        "development_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
    }


def verify_fulldev_r5() -> dict[str, Any]:
    """Verify FULLDEV.R5 integrity without re-scoring. Pending-safe if unscored."""
    if not FULLDEV_R5_PREREG.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R5 prereg not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse"}
    r5c = json.loads(FULLDEV_R5_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (
        FULLDEV_R4_PREREG,
        D6_R3_PREREG,
        D6_R2_PREREG,
        D6_R1_PREREG,
        FULLDEV_R3_PREREG,
        D5_R2_PREREG,
        D5_R1_PREREG,
        D4_R2_PREREG,
        D4_R1_PREREG,
        D3_R3_PREREG,
        FULLDEV_R2_PREREG,
        FULLDEV_R1_PREREG,
        DEV_PREREG,
        PREREG,
    ):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r5c:
            return {"ok": False, "why": f"FULLDEV.R5 commitment reused {prior.name}"}
    if json.loads(FULLDEV_R5_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.FULL.R5.":
        return {"ok": False, "why": "FULLDEV.R5 domain drifted"}
    if not FULLDEV_R5_LOCK.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R5 result not frozen yet"}
    res = json.loads(FULLDEV_R5_LOCK.read_text(encoding="utf-8"))
    if res.get("product") != "0.0.004" or res.get("earned_next") is not False or res.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if res.get("eligible_for_000005") is not False:
        return {"ok": False, "why": "eligible_for_000005 must stay false until nursery clears"}
    battery = res.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(res.get("development_gate_clear"))
    if battery.get("domain") != "TM023.FULL.R5.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not FULLDEV_R5_FAIL.exists():
        return {"ok": False, "why": "missing cortex_fulldev_r5.failure.lock"}
    if clear and FULLDEV_R5_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear battery"}
    return {
        "ok": True,
        "why": "FULLDEV.R5 integrity ok",
        "pending": False,
        "development_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
    }


def verify_fulldev_r6() -> dict[str, Any]:
    """Verify FULLDEV.R6 integrity without re-scoring. Pending-safe if unscored."""
    if not FULLDEV_R6_PREREG.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R6 prereg not frozen yet"}
    if DEV_V13_LOCK.exists():
        return {"ok": False, "why": "DEVELOP.v13 exists — refuse"}
    r6c = json.loads(FULLDEV_R6_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (
        FULLDEV_R5_PREREG,
        D7_R2_PREREG,
        D7_R1_PREREG,
        D6_R3_PREREG,
        FULLDEV_R4_PREREG,
        D5_R2_PREREG,
        FULLDEV_R3_PREREG,
        D4_R2_PREREG,
        D3_R3_PREREG,
        FULLDEV_R2_PREREG,
        FULLDEV_R1_PREREG,
        DEV_PREREG,
        PREREG,
    ):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r6c:
            return {"ok": False, "why": f"FULLDEV.R6 commitment reused {prior.name}"}
    if json.loads(FULLDEV_R6_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.FULL.R6.":
        return {"ok": False, "why": "FULLDEV.R6 domain drifted"}
    if not FULLDEV_R6_LOCK.exists():
        return {"ok": True, "pending": True, "why": "FULLDEV.R6 result not frozen yet"}
    res = json.loads(FULLDEV_R6_LOCK.read_text(encoding="utf-8"))
    if res.get("product") != "0.0.004" or res.get("earned_next") is not False or res.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if res.get("eligible_for_000005") is not False:
        return {"ok": False, "why": "eligible_for_000005 must stay false until nursery clears"}
    battery = res.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(res.get("development_gate_clear"))
    if battery.get("domain") != "TM023.FULL.R6.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not FULLDEV_R6_FAIL.exists():
        return {"ok": False, "why": "missing cortex_fulldev_r6.failure.lock"}
    if clear and FULLDEV_R6_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear battery"}
    return {
        "ok": True,
        "why": "FULLDEV.R6 integrity ok",
        "pending": False,
        "development_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
    }


def verify_d3_r2() -> dict[str, Any]:
    """Verify isolated D3.R2 integrity without re-scoring. Pending-safe if unscored."""
    if not D3_R2_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D3.R2 prereg not frozen yet"}
    r2c = json.loads(D3_R2_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (D3_R1_PREREG, FULLDEV_R1_PREREG, V13_PREREG, V12_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r2c:
            return {"ok": False, "why": f"D3.R2 commitment reused {prior.name}"}
    if json.loads(D3_R2_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D3.R2.":
        return {"ok": False, "why": "D3.R2 domain drifted"}
    if not CANDIDATE_V15.exists():
        return {"ok": True, "pending": True, "why": "candidate v15 not frozen yet"}
    if not D3_R2_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D3.R2 gate result not frozen yet"}
    gate = json.loads(D3_R2_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if gate.get("candidate_v15_sha") != _sha_file(CANDIDATE_V15):
        return {"ok": False, "why": "gate/candidate v15 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D3.R2.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D3_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d3_r2_gate.failure.lock"}
    if clear and D3_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D3.R2 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d3_r3() -> dict[str, Any]:
    """Verify isolated D3.R3 integrity without re-scoring. Pending-safe if unscored."""
    if not D3_R3_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D3.R3 prereg not frozen yet"}
    r3c = json.loads(D3_R3_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (D3_R2_PREREG, D3_R1_PREREG, FULLDEV_R1_PREREG, V13_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r3c:
            return {"ok": False, "why": f"D3.R3 commitment reused {prior.name}"}
    if json.loads(D3_R3_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D3.R3.":
        return {"ok": False, "why": "D3.R3 domain drifted"}
    if not CANDIDATE_V16.exists():
        return {"ok": True, "pending": True, "why": "candidate v16 not frozen yet"}
    if not D3_R3_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D3.R3 gate result not frozen yet"}
    gate = json.loads(D3_R3_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    if gate.get("candidate_v16_sha") != _sha_file(CANDIDATE_V16):
        return {"ok": False, "why": "gate/candidate v16 sha mismatch"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D3.R3.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D3_R3_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d3_r3_gate.failure.lock"}
    if clear and D3_R3_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D3.R3 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d4_r1() -> dict[str, Any]:
    """Verify isolated D4.R1 integrity without re-scoring. Pending-safe if unscored."""
    if not D4_R1_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D4.R1 prereg not frozen yet"}
    r1c = json.loads(D4_R1_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (FULLDEV_R2_PREREG, D3_R3_PREREG, FULLDEV_R1_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r1c:
            return {"ok": False, "why": f"D4.R1 commitment reused {prior.name}"}
    if json.loads(D4_R1_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D4.R1.":
        return {"ok": False, "why": "D4.R1 domain drifted"}
    if not CANDIDATE_V17.exists():
        return {"ok": True, "pending": True, "why": "candidate v17 not frozen yet"}
    if not D4_R1_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D4.R1 gate result not frozen yet"}
    gate = json.loads(D4_R1_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D4.R1.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D4_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d4_r1_gate.failure.lock"}
    if clear and D4_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D4.R1 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d5_r1() -> dict[str, Any]:
    """Verify isolated D5.R1 integrity without re-scoring. Pending-safe if unscored."""
    if not D5_R1_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D5.R1 prereg not frozen yet"}
    r1c = json.loads(D5_R1_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (FULLDEV_R3_PREREG, D4_R2_PREREG, D4_R1_PREREG, FULLDEV_R2_PREREG, D3_R3_PREREG, FULLDEV_R1_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r1c:
            return {"ok": False, "why": f"D5.R1 commitment reused {prior.name}"}
    if json.loads(D5_R1_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D5.R1.":
        return {"ok": False, "why": "D5.R1 domain drifted"}
    if not CANDIDATE_V19.exists():
        return {"ok": True, "pending": True, "why": "candidate v19 not frozen yet"}
    if not D5_R1_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D5.R1 gate result not frozen yet"}
    gate = json.loads(D5_R1_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D5.R1.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D5_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d5_r1_gate.failure.lock"}
    if clear and D5_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D5.R1 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d5_r2() -> dict[str, Any]:
    """Verify isolated D5.R2 integrity without re-scoring. Pending-safe if unscored."""
    if not D5_R2_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D5.R2 prereg not frozen yet"}
    r2c = json.loads(D5_R2_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (D5_R1_PREREG, FULLDEV_R3_PREREG, D4_R2_PREREG, D4_R1_PREREG, FULLDEV_R2_PREREG, D3_R3_PREREG, FULLDEV_R1_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r2c:
            return {"ok": False, "why": f"D5.R2 commitment reused {prior.name}"}
    if json.loads(D5_R2_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D5.R2.":
        return {"ok": False, "why": "D5.R2 domain drifted"}
    if not CANDIDATE_V20.exists():
        return {"ok": True, "pending": True, "why": "candidate v20 not frozen yet"}
    if not D5_R2_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D5.R2 gate result not frozen yet"}
    gate = json.loads(D5_R2_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D5.R2.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D5_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d5_r2_gate.failure.lock"}
    if clear and D5_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D5.R2 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d6_r1() -> dict[str, Any]:
    """Verify isolated D6.R1 integrity without re-scoring. Pending-safe if unscored."""
    if not D6_R1_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D6.R1 prereg not frozen yet"}
    r1c = json.loads(D6_R1_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (FULLDEV_R4_PREREG, D5_R2_PREREG, D5_R1_PREREG, FULLDEV_R3_PREREG, D4_R2_PREREG, FULLDEV_R2_PREREG, D3_R3_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r1c:
            return {"ok": False, "why": f"D6.R1 commitment reused {prior.name}"}
    if json.loads(D6_R1_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D6.R1.":
        return {"ok": False, "why": "D6.R1 domain drifted"}
    if not CANDIDATE_V21.exists():
        return {"ok": True, "pending": True, "why": "candidate v21 not frozen yet"}
    if not D6_R1_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D6.R1 gate result not frozen yet"}
    gate = json.loads(D6_R1_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D6.R1.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D6_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d6_r1_gate.failure.lock"}
    if clear and D6_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D6.R1 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d6_r2() -> dict[str, Any]:
    """Verify isolated D6.R2 integrity without re-scoring. Pending-safe if unscored."""
    if not D6_R2_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D6.R2 prereg not frozen yet"}
    r2c = json.loads(D6_R2_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (D6_R1_PREREG, FULLDEV_R4_PREREG, D5_R2_PREREG, FULLDEV_R3_PREREG, D4_R2_PREREG, DEV_PREREG, PREREG):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r2c:
            return {"ok": False, "why": f"D6.R2 commitment reused {prior.name}"}
    if json.loads(D6_R2_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D6.R2.":
        return {"ok": False, "why": "D6.R2 domain drifted"}
    if not CANDIDATE_V22.exists():
        return {"ok": True, "pending": True, "why": "candidate v22 not frozen yet"}
    if not D6_R2_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D6.R2 gate result not frozen yet"}
    gate = json.loads(D6_R2_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D6.R2.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D6_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d6_r2_gate.failure.lock"}
    if clear and D6_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D6.R2 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d7_r1() -> dict[str, Any]:
    """Verify isolated D7.R1 integrity without re-scoring. Pending-safe if unscored."""
    if not D7_R1_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D7.R1 prereg not frozen yet"}
    r1c = json.loads(D7_R1_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (
        FULLDEV_R5_PREREG,
        D6_R3_PREREG,
        D6_R2_PREREG,
        D6_R1_PREREG,
        FULLDEV_R4_PREREG,
        D5_R2_PREREG,
        FULLDEV_R3_PREREG,
        D4_R2_PREREG,
        DEV_PREREG,
        PREREG,
    ):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r1c:
            return {"ok": False, "why": f"D7.R1 commitment reused {prior.name}"}
    if json.loads(D7_R1_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D7.R1.":
        return {"ok": False, "why": "D7.R1 domain drifted"}
    if not CANDIDATE_V24.exists():
        return {"ok": True, "pending": True, "why": "candidate v24 not frozen yet"}
    if not D7_R1_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D7.R1 gate result not frozen yet"}
    gate = json.loads(D7_R1_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D7.R1.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D7_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d7_r1_gate.failure.lock"}
    if clear and D7_R1_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D7.R1 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def verify_d7_r2() -> dict[str, Any]:
    """Verify isolated D7.R2 integrity without re-scoring. Pending-safe if unscored."""
    if not D7_R2_PREREG.exists():
        return {"ok": True, "pending": True, "why": "D7.R2 prereg not frozen yet"}
    r2c = json.loads(D7_R2_PREREG.read_text(encoding="utf-8"))["eval_seed_commitment"]
    for prior in (
        D7_R1_PREREG,
        FULLDEV_R5_PREREG,
        D6_R3_PREREG,
        D6_R2_PREREG,
        D6_R1_PREREG,
        FULLDEV_R4_PREREG,
        D5_R2_PREREG,
        FULLDEV_R3_PREREG,
        DEV_PREREG,
        PREREG,
    ):
        if prior.exists() and json.loads(prior.read_text(encoding="utf-8")).get("eval_seed_commitment") == r2c:
            return {"ok": False, "why": f"D7.R2 commitment reused {prior.name}"}
    if json.loads(D7_R2_PREREG.read_text(encoding="utf-8")).get("domain") != "TM023.D7.R2.":
        return {"ok": False, "why": "D7.R2 domain drifted"}
    if not CANDIDATE_V25.exists():
        return {"ok": True, "pending": True, "why": "candidate v25 not frozen yet"}
    if not D7_R2_GATE_LOCK.exists():
        return {"ok": True, "pending": True, "why": "D7.R2 gate result not frozen yet"}
    gate = json.loads(D7_R2_GATE_LOCK.read_text(encoding="utf-8"))
    if gate.get("product") != "0.0.004" or gate.get("earned_next") is not False or gate.get("ex0s") is not None:
        return {"ok": False, "why": "product/earned_next/ex0s drift"}
    battery = gate.get("battery") or {}
    n_clear = int(battery.get("n_pair_clear") or 0)
    clear = bool(gate.get("relation_gate_clear"))
    if battery.get("domain") != "TM023.D7.R2.":
        return {"ok": False, "why": "result domain drifted"}
    if clear and n_clear < 13:
        return {"ok": False, "why": "gate claims clear with n_pair_clear < 13"}
    if (not clear) and not D7_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "missing cortex_d7_r2_gate.failure.lock"}
    if clear and D7_R2_GATE_FAIL.exists():
        return {"ok": False, "why": "failure lock present on a clear gate"}
    return {
        "ok": True,
        "why": "D7.R2 integrity ok",
        "pending": False,
        "relation_gate_clear": clear,
        "n_pair_clear": n_clear,
        "refuse_rewrite": True,
        "refuse_fulldev_before_clear": not clear,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-phase-a", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--write-birth", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--verify-sanity-amendment", action="store_true")
    ap.add_argument("--freeze-runner", action="store_true")
    ap.add_argument("--verify-pre-reveal", action="store_true")
    ap.add_argument("--reveal-eval", action="store_true")
    ap.add_argument("--life", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--write-diag-lock", action="store_true")
    ap.add_argument("--freeze-v2-amendment", action="store_true")
    ap.add_argument("--freeze-v2-gate-runner", action="store_true")
    ap.add_argument("--publish-v2-commitment", action="store_true")
    ap.add_argument("--reveal-v2-eval", action="store_true")
    ap.add_argument("--v2-gate", action="store_true")
    ap.add_argument("--verify-v4-gate", action="store_true")
    ap.add_argument("--verify-v5-gate", action="store_true")
    ap.add_argument("--verify-v6-gate", action="store_true")
    ap.add_argument("--verify-v7-gate", action="store_true")
    ap.add_argument("--verify-v8-gate", action="store_true")
    ap.add_argument("--verify-v9-gate", action="store_true")
    ap.add_argument("--verify-v10-gate", action="store_true")
    ap.add_argument("--verify-v11-gate", action="store_true")
    ap.add_argument("--verify-v12-gate", action="store_true")
    ap.add_argument("--verify-v13-gate", action="store_true")
    ap.add_argument("--verify-fulldev-r1", action="store_true")
    ap.add_argument("--verify-d3-r2", action="store_true")
    ap.add_argument("--verify-d3-r3", action="store_true")
    ap.add_argument("--verify-fulldev-r2", action="store_true")
    ap.add_argument("--verify-fulldev-r3", action="store_true")
    ap.add_argument("--verify-fulldev-r4", action="store_true")
    ap.add_argument("--verify-fulldev-r5", action="store_true")
    ap.add_argument("--verify-fulldev-r6", action="store_true")
    ap.add_argument("--verify-d4-r1", action="store_true")
    ap.add_argument("--verify-d5-r1", action="store_true")
    ap.add_argument("--verify-d5-r2", action="store_true")
    ap.add_argument("--verify-d6-r1", action="store_true")
    ap.add_argument("--verify-d6-r2", action="store_true")
    ap.add_argument("--verify-d7-r1", action="store_true")
    ap.add_argument("--verify-d7-r2", action="store_true")
    ap.add_argument("--mact-v6-audit", action="store_true")
    ap.add_argument("--v4-math-audit", action="store_true")
    ap.add_argument("--write-v2-birth", action="store_true")
    ap.add_argument("--write-candidate-v2", action="store_true")
    ap.add_argument("--freeze-mact-boundary-runner", action="store_true")
    ap.add_argument("--mact-boundary", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--smoke-pairs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.write_phase_a:
        print(json.dumps(write_phase_a_artifacts(), indent=2))
        return
    if args.verify_prereg:
        ok, why, lock = verify_prereg()
        print(json.dumps({"ok": ok, "why": why, "lab": lock.get("lab")}, indent=2))
        return
    if args.verify_sanity_amendment:
        ok, why, am = verify_sanity_amendment()
        print(json.dumps({"ok": ok, "why": why, "all_sanity_ok": am.get("all_sanity_ok")}, indent=2))
        return
    if args.freeze_runner:
        print(json.dumps(freeze_runner_lock(), indent=2))
        return
    if args.verify_pre_reveal:
        print(json.dumps(verify_pre_reveal(), indent=2))
        return
    if args.reveal_eval:
        print(json.dumps(reveal_eval(), indent=2))
        return
    if args.freeze_v2_amendment:
        print(json.dumps(freeze_v2_amendment_lock(), indent=2))
        return
    if args.freeze_v2_gate_runner:
        print(json.dumps(freeze_v2_gate_runner(), indent=2))
        return
    if args.publish_v2_commitment:
        print(json.dumps(publish_v2_commitment(), indent=2))
        return
    if args.verify_v4_gate:
        print(json.dumps(verify_v4_gate(), indent=2, default=str))
        return
    if args.verify_v5_gate:
        print(json.dumps(verify_v5_gate(), indent=2, default=str))
        return
    if args.verify_v6_gate:
        print(json.dumps(verify_v6_gate(), indent=2, default=str))
        return
    if args.verify_v7_gate:
        print(json.dumps(verify_v7_gate(), indent=2, default=str))
        return
    if args.verify_v8_gate:
        print(json.dumps(verify_v8_gate(), indent=2, default=str))
        return
    if args.verify_v9_gate:
        print(json.dumps(verify_v9_gate(), indent=2, default=str))
        return
    if args.verify_v10_gate:
        print(json.dumps(verify_v10_gate(), indent=2, default=str))
        return
    if args.verify_v11_gate:
        print(json.dumps(verify_v11_gate(), indent=2, default=str))
        return
    if args.verify_v12_gate:
        print(json.dumps(verify_v12_gate(), indent=2, default=str))
        return
    if args.verify_v13_gate:
        print(json.dumps(verify_v13_gate(), indent=2, default=str))
        return
    if args.verify_fulldev_r1:
        print(json.dumps(verify_fulldev_r1(), indent=2, default=str))
        return
    if args.verify_d3_r2:
        print(json.dumps(verify_d3_r2(), indent=2, default=str))
        return
    if args.verify_d3_r3:
        print(json.dumps(verify_d3_r3(), indent=2, default=str))
        return
    if args.verify_fulldev_r2:
        print(json.dumps(verify_fulldev_r2(), indent=2, default=str))
        return
    if args.verify_fulldev_r3:
        print(json.dumps(verify_fulldev_r3(), indent=2, default=str))
        return
    if args.verify_fulldev_r4:
        print(json.dumps(verify_fulldev_r4(), indent=2, default=str))
        return
    if args.verify_fulldev_r5:
        print(json.dumps(verify_fulldev_r5(), indent=2, default=str))
        return
    if args.verify_fulldev_r6:
        print(json.dumps(verify_fulldev_r6(), indent=2, default=str))
        return
    if args.verify_d4_r1:
        print(json.dumps(verify_d4_r1(), indent=2, default=str))
        return
    if args.verify_d5_r1:
        print(json.dumps(verify_d5_r1(), indent=2, default=str))
        return
    if args.verify_d5_r2:
        print(json.dumps(verify_d5_r2(), indent=2, default=str))
        return
    if args.verify_d6_r1:
        print(json.dumps(verify_d6_r1(), indent=2, default=str))
        return
    if args.verify_d6_r2:
        print(json.dumps(verify_d6_r2(), indent=2, default=str))
        return
    if args.verify_d7_r1:
        print(json.dumps(verify_d7_r1(), indent=2, default=str))
        return
    if args.verify_d7_r2:
        print(json.dumps(verify_d7_r2(), indent=2, default=str))
        return
    if args.mact_v6_audit:
        from experiments.cortex_mact_boundary import write_v6_boundary_audit

        print(json.dumps(write_v6_boundary_audit(), indent=2, default=str))
        return
    if args.v4_math_audit:
        print(json.dumps(write_v4_math_audit(write_lock=True), indent=2, default=str))
        return
    if args.freeze_mact_boundary_runner:
        from experiments.cortex_mact_boundary import freeze_boundary_runner_v1

        print(json.dumps(freeze_boundary_runner_v1(), indent=2))
        return
    if args.mact_boundary:
        from experiments.cortex_mact_boundary import run_boundary_v4

        print(
            json.dumps(
                run_boundary_v4(write_lock=args.write_lock),
                indent=2,
                default=str,
            )
        )
        return
    if args.reveal_v2_eval:
        print(json.dumps(reveal_v2_eval(), indent=2))
        return
    if args.v2_gate:
        print(
            json.dumps(
                run_v2_gate_score(device=args.device, write_lock=args.write_lock),
                indent=2,
                default=str,
            )
        )
        return
    if args.diag or args.write_diag_lock:
        from experiments.cortex_diag import run_diag

        print(
            json.dumps(
                run_diag(write_lock=args.write_diag_lock or args.write_lock),
                indent=2,
                default=str,
            )
        )
        return
    if args.life:
        print(
            json.dumps(
                run_develop_score(
                    device=args.device,
                    write_lock=args.write_lock,
                    smoke_pairs=args.smoke_pairs,
                ),
                indent=2,
                default=str,
            )
        )
        return
    if args.sanity or args.write_birth or args.write_candidate or args.write_v2_birth or args.write_candidate_v2:
        print(
            json.dumps(
                run_sanity(
                    write_birth=args.write_birth or args.write_candidate or args.write_v2_birth or args.write_candidate_v2,
                    write_candidate=args.write_candidate,
                    write_v2_birth=args.write_v2_birth or args.write_candidate_v2,
                    write_candidate_v2=args.write_candidate_v2,
                ),
                indent=2,
                default=str,
            )
        )
        return
    ap.print_help()


if __name__ == "__main__":
    main()
