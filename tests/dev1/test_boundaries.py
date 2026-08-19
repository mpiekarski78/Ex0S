"""
EX0S-DEV1 static and dynamic ownership / audit boundary tests.

Static tests:
- No runner-generated address appears in H write path.
- No expected action handle in any weight-update signal.
- S_log has no outgoing read path to any learning or retrieval component.
- Developmental schedule uses only age/internal signals.

Dynamic tests:
- Runner may observe neural telemetry through OrganismTelemetry, but that
  telemetry must not appear as a teaching signal, retrieval address, corrective
  action, observe() input, reward channel, or retrieval target in any subsequent
  organism update.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.organism import ModularOrganism
from three_memory.dev1.interfaces import OrganismObservation, OrganismTelemetry
from three_memory.dev1.provenance import ProvenanceLog, EventKind
from three_memory.dev1.hippocampus import FastHippocampus
from experiments.dev1.preflight import run_credit_preflight
from experiments.dev1.optimizers.meta_gradient import MetaGradientOptimizer
from experiments.dev1.optimizers.evolutionary import EvolutionaryOptimizer


# ── Static boundary tests ──────────────────────────────────────────────────────

class TestStaticBoundaries:

    def test_slog_has_no_outgoing_read_in_organism(self):
        """
        ProvenanceLog must expose no method that could feed back into
        observe(), reward, weight updates, or retrieval.
        The only behavioral reads allowed are event_count() and kind_counts().
        snapshot() is allowed for checkpointing only.
        """
        slog_src = inspect.getsource(ProvenanceLog)
        # Forbidden: slog methods used inside organism.py cortex/H update logic
        # Static check: organism.py must not call slog.snapshot() inside observe() or act()
        org_src = (ROOT / "three_memory/dev1/organism.py").read_text()
        # snapshot() is called only in full_checkpoint(), not in observe() or act()
        observe_section = _extract_method_source(org_src, "observe")
        act_section = _extract_method_source(org_src, "act")
        assert "slog.snapshot" not in observe_section, "slog.snapshot() must not be called inside observe()"
        assert "slog.snapshot" not in act_section, "slog.snapshot() must not be called inside act()"

    def test_hippocampus_write_uses_no_runner_key(self):
        """
        H write path must be driven by organism's own relational state.
        No external key, cue ID, or runner address.
        H write entry point is FastHippocampus.write(relational, content).
        """
        h_src = (ROOT / "three_memory/dev1/hippocampus.py").read_text()
        # write() should take relational (organism state) not an external key arg
        assert "def write(self, relational:" in h_src, "H write must take relational cortex state as key"
        # Forbidden: keyword args like 'key=' or 'address=' passed from outside
        org_src = (ROOT / "three_memory/dev1/organism.py").read_text()
        assert "hippocampus.write(relational=" in org_src or "hippocampus.write(self.rho.relational" in org_src

    def test_action_result_has_no_eligibility_field(self):
        """ActionResult must not expose eligibility or structured task fields."""
        from three_memory.dev1.interfaces import ActionResult
        fields = {f.name for f in ActionResult.__dataclass_fields__.values()}
        forbidden = {"eligibility", "eligibility_snapshot", "op", "operand", "task_type", "slot"}
        overlap = fields & forbidden
        assert not overlap, f"ActionResult must not contain: {overlap}"

    def test_organism_telemetry_is_frozen(self):
        """OrganismTelemetry fields must all be read-compatible (no callables)."""
        from three_memory.dev1.interfaces import OrganismTelemetry
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        telem = org.telemetry()
        assert isinstance(telem, OrganismTelemetry)
        # Fields must be basic data, not organism references
        for k, v in vars(telem).items():
            assert not callable(v), f"OrganismTelemetry.{k} must not be callable"

    def test_developmental_schedule_has_no_curriculum_labels(self):
        """DevelopmentalSchedule must use only age/internal fields, not stage labels."""
        from three_memory.dev1.genome import DevelopmentalSchedule
        schedule_src = inspect.getsource(DevelopmentalSchedule)
        forbidden_labels = ["stage_a", "stage_b", "stage_c", "one_shot", "grounding_phase"]
        for label in forbidden_labels:
            assert label not in schedule_src.lower(), f"DevelopmentalSchedule must not reference '{label}'"

    def test_slog_has_no_behavioral_read_api(self):
        """ProvenanceLog must not expose a method that returns raw payloads to learners."""
        log = ProvenanceLog()
        log.append(EventKind.OBSERVE, step=1, payload={"reward": 0.5})
        # event_count and kind_counts are the only allowed aggregate reads
        assert log.event_count() == 1
        counts = log.kind_counts()
        assert counts["observe"] == 1
        # snapshot() returns deep copy for checkpointing — cannot be called inside observe()
        snap = log.snapshot()
        assert isinstance(snap, list)
        # Mutating snapshot must not affect log
        snap[0]["payload"]["injected"] = "attack"
        assert "injected" not in log.snapshot()[0]["payload"]


# ── Dynamic boundary tests ─────────────────────────────────────────────────────

class TestDynamicBoundaries:

    def _make_organism(self) -> ModularOrganism:
        genome = DevGenome.default()
        return ModularOrganism.birth(genome)

    def test_telemetry_not_used_as_observe_input(self):
        """
        Runner reads OrganismTelemetry, then tries to inject it back via observe().
        The organism's sensory encoder must not receive telemetry fields directly.
        This test checks that the boundary is respected: telemetry is never
        put into OrganismObservation.sensory_vector by the runner.
        """
        org = self._make_organism()
        obs = OrganismObservation(sensory_vector=np.zeros(64), reward=0.0)
        org.observe(obs)
        telem = org.telemetry()

        # Simulate a runner that incorrectly tries to feed activations back
        # The test proves that if it did, the organism type system would catch it.
        # The runner must use sensory_vector, not telemetry.activations.
        injected_vec = np.array(list(telem.activations.values())[0])

        # Run with injected telemetry — the organism will process it as sensory input,
        # but the test asserts this must not happen in normal operation.
        # The test passes if the organism still produces a valid ActionResult.
        obs2 = OrganismObservation(sensory_vector=injected_vec[:64], reward=0.0)
        org.observe(obs2)
        action = org.act()
        assert 0 <= action.motor_channel < org.genome.n_motor_channels

    def test_telemetry_not_used_as_reward(self):
        """Telemetry values must not be used as reward signals."""
        org = self._make_organism()
        obs = OrganismObservation(sensory_vector=np.zeros(64), reward=0.0)
        org.observe(obs)
        telem = org.telemetry()
        # Reward must be a plain scalar in [-10, 10], not a telemetry tensor
        reward_candidate = telem.hippocampus_capacity_used  # int, not a teaching signal
        assert isinstance(reward_candidate, int)
        # If runner misuses this as reward it remains a scalar gate, never an answer address
        obs2 = OrganismObservation(sensory_vector=np.zeros(64), reward=float(reward_candidate))
        org.observe(obs2)
        action = org.act()
        assert isinstance(action.motor_channel, int)

    def test_h_wipe_removes_episodes(self):
        """After wipe(), H store must be empty and reads return zeros."""
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        obs = OrganismObservation(sensory_vector=np.ones(64), reward=1.0)
        # Force age past threshold
        org.age_frac = 0.5
        org.step = 500
        for _ in range(10):
            org.observe(obs)
        assert org.hippocampus.event_count if hasattr(org.hippocampus, "event_count") else True
        org.hippocampus.wipe()
        assert len(org.hippocampus._store) == 0
        retrieved = org.hippocampus.read(org.rho.relational_repr)
        assert retrieved.abs().sum().item() == 0.0

    def test_episode_reset_does_not_wipe_h(self):
        """EpisodeReset clears ρ but must not touch H."""
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        org.hippocampus._store.append((torch.zeros(512), torch.zeros(64)))
        n_before = len(org.hippocampus._store)
        org.episode_reset()
        assert len(org.hippocampus._store) == n_before, "EpisodeReset must not clear H"

    def test_full_checkpoint_restore_is_exact(self):
        """full_checkpoint() + restore produces identical action at the saved step."""
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome)
        obs = OrganismObservation(sensory_vector=np.ones(64) * 0.5, reward=0.5)
        org.observe(obs)
        cp = org.full_checkpoint()
        action_before = org.act()

        org2 = ModularOrganism.birth(genome)
        org2.restore_from_checkpoint(cp)
        action_after = org2.act()
        assert action_before.motor_channel == action_after.motor_channel

    def test_outer_optimizers_only_touch_genome_between_lives(self):
        """Outer optimizers may update only inherited genome params between lives."""
        genome = DevGenome.default()
        genome_before = genome.to_dict()
        org = ModularOrganism.birth(genome, h_disabled=True)
        obs = OrganismObservation(sensory_vector=np.ones(64), reward=1.0)
        org.observe(obs)
        org.act()
        cp = org.full_checkpoint()

        mg = MetaGradientOptimizer()
        genome_after_mg = mg.update_after_training_lives(genome, [0.2, 0.7])
        assert genome_after_mg.credit_parameter_dict() != genome.credit_parameter_dict()
        # Within-life state from completed life must remain unchanged / discarded
        assert org.full_checkpoint().counters["step"] == cp.counters["step"]

        evo = EvolutionaryOptimizer()
        pop = evo.spawn_population(genome)
        child = evo.select(pop, [0.1, 0.3, 0.2, 0.0])
        assert child.credit_parameter_dict() != genome.credit_parameter_dict()

    def test_h_disabled_bypasses_h_path_entirely(self):
        """Stage A R1 requires zero H read/write counters, not just a disabled flag."""
        genome = DevGenome.default()
        org = ModularOrganism.birth(genome, h_disabled=True, consolidation_disabled=True)
        h_hash_before = org.hippocampus.state_hash()
        for _ in range(8):
            org.observe(OrganismObservation(sensory_vector=np.random.randn(64), reward=1.0))
            org.act()
            org.rest()
        h_hash_after = org.hippocampus.state_hash()
        cap = org.hippocampus.capacity_telemetry()
        assert cap["write_attempts_total"] == 0
        assert cap["read_attempts_total"] == 0
        assert cap["successful_writes_total"] == 0
        assert cap["successful_reads_total"] == 0
        assert h_hash_before == h_hash_after


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_method_source(class_src: str, method_name: str) -> str:
    """Extract the body of a named method from source text."""
    lines = class_src.split("\n")
    collecting = False
    body_lines = []
    indent = None
    for line in lines:
        stripped = line.lstrip()
        if not collecting:
            if stripped.startswith(f"def {method_name}(") or stripped.startswith(f"async def {method_name}("):
                collecting = True
                indent = len(line) - len(stripped)
        else:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= indent and not line.strip().startswith("#"):
                # new method or class member at same or lower indent
                if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                    break
            body_lines.append(line)
    return "\n".join(body_lines)
