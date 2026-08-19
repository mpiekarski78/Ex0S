"""
EX0S-DEV1 eight-phase developmental curriculum.

The curriculum supplies EXPERIENCES, never internal representations
or answer addresses.

Phase definitions
─────────────────
1. grounding      — motor-role learning through sensory consequences
2. renamed_cues   — same roles, new symbol vectors
3. one_shot_facts  — fact presentation + state reset
4. revision        — fact update without erasing unrelated knowledge
5. composition     — unseen two-step inference
6. ambiguity       — equal/conflicting evidence → ASK motor channel
7. clarification   — clarification event → correct answer
8. continued       — further learning after clarification

Curriculum invariants
─────────────────────
- The curriculum supplies experiences (events + rewards) only.
- It never injects internal representations, logical slot labels,
  expected action handles, cue IDs, or structured operands.
- Phases advance based on experience count / episode count; the organism's
  developmental schedule uses age/internal signals independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator

import numpy as np

from experiments.dev1.worlds import InteractionWorld, WorldConfig, WorldEvent, _symbol_vector


class Phase(IntEnum):
    GROUNDING = 1
    RENAMED_CUES = 2
    ONE_SHOT_FACTS = 3
    REVISION = 4
    COMPOSITION = 5
    AMBIGUITY = 6
    CLARIFICATION = 7
    CONTINUED = 8


@dataclass
class CurriculumConfig:
    world_cfg: WorldConfig = field(default_factory=WorldConfig)
    grounding_episodes: int = 32
    renamed_episodes: int = 8
    one_shot_presentations: int = 4
    revision_presentations: int = 2
    composition_trials: int = 8
    ambiguity_trials: int = 4
    clarification_trials: int = 4
    continued_episodes: int = 8


@dataclass
class CurriculumEvent:
    phase: Phase
    world_event: WorldEvent
    episode_idx: int
    step_in_episode: int
    reset_after: bool = False          # True → runner calls episode_reset() after this event
    is_fact_presentation: bool = False


class DevelopmentalCurriculum:
    """
    Generates the eight-phase experience sequence.

    Usage:
        for ce in curriculum.events():
            obs = organism.observe(OrganismObservation(...))
            action = organism.act()
            reward = world.reward_for_action(ce.world_event, action.motor_channel)
            # feed reward back via next observe()
            if ce.reset_after:
                organism.episode_reset()
    """

    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.world = InteractionWorld(cfg.world_cfg)
        self.renamed_world = InteractionWorld.renamed(cfg.world_cfg)

    def events(self) -> Iterator[CurriculumEvent]:
        yield from self._phase_grounding()
        yield from self._phase_renamed()
        yield from self._phase_one_shot()
        yield from self._phase_revision()
        yield from self._phase_composition()
        yield from self._phase_ambiguity()
        yield from self._phase_clarification()
        yield from self._phase_continued()

    # ── Phase generators ───────────────────────────────────────────────────────

    def _phase_grounding(self) -> Iterator[CurriculumEvent]:
        for ep in range(self.cfg.grounding_episodes):
            events = self.world.generate_episode()
            for s, we in enumerate(events):
                yield CurriculumEvent(
                    phase=Phase.GROUNDING,
                    world_event=we,
                    episode_idx=ep,
                    step_in_episode=s,
                    reset_after=(s == len(events) - 1),
                )

    def _phase_renamed(self) -> Iterator[CurriculumEvent]:
        for ep in range(self.cfg.renamed_episodes):
            events = self.renamed_world.generate_episode()
            for s, we in enumerate(events):
                yield CurriculumEvent(
                    phase=Phase.RENAMED_CUES,
                    world_event=we,
                    episode_idx=ep,
                    step_in_episode=s,
                    reset_after=(s == len(events) - 1),
                )

    def _phase_one_shot(self) -> Iterator[CurriculumEvent]:
        """Present a fact, then reset. Probes occur before cortical transfer."""
        dim = self.cfg.world_cfg.sensory_dim
        for i in range(self.cfg.one_shot_presentations):
            # A "fact" is a novel sensory event with a high-reward signal
            fact_vec = _symbol_vector(1000 + i, dim, None)  # type: ignore
            we = WorldEvent(sensory_vector=fact_vec, reward=1.0, _symbol_id=1000 + i, _correct_channel=0)
            yield CurriculumEvent(
                phase=Phase.ONE_SHOT_FACTS,
                world_event=we,
                episode_idx=i,
                step_in_episode=0,
                reset_after=True,
                is_fact_presentation=True,
            )

    def _phase_revision(self) -> Iterator[CurriculumEvent]:
        """Present a contradicting fact (revision). Previous fact must survive."""
        dim = self.cfg.world_cfg.sensory_dim
        for i in range(self.cfg.revision_presentations):
            revised_vec = _symbol_vector(2000 + i, dim, None)  # type: ignore
            we = WorldEvent(sensory_vector=revised_vec, reward=1.0, _symbol_id=2000 + i, _correct_channel=1)
            yield CurriculumEvent(
                phase=Phase.REVISION,
                world_event=we,
                episode_idx=i,
                step_in_episode=0,
                reset_after=True,
                is_fact_presentation=True,
            )

    def _phase_composition(self) -> Iterator[CurriculumEvent]:
        """Two-step inference trials (unseen symbol combinations)."""
        dim = self.cfg.world_cfg.sensory_dim
        for i in range(self.cfg.composition_trials):
            vec1 = _symbol_vector(3000 + i, dim, None)  # type: ignore
            vec2 = _symbol_vector(3100 + i, dim, None)  # type: ignore
            combined = (vec1 + vec2) / 2.0
            we = WorldEvent(sensory_vector=combined.astype(np.float32), _symbol_id=-1, _correct_channel=i % self.cfg.world_cfg.n_roles)
            yield CurriculumEvent(phase=Phase.COMPOSITION, world_event=we, episode_idx=i, step_in_episode=0, reset_after=(i % 4 == 3))

    def _phase_ambiguity(self) -> Iterator[CurriculumEvent]:
        """Equal-evidence trials where ASK is the correct motor response."""
        dim = self.cfg.world_cfg.sensory_dim
        # ASK channel is by convention the last motor channel
        ask_channel = self.cfg.world_cfg.n_roles
        for i in range(self.cfg.ambiguity_trials):
            vec = _symbol_vector(4000 + i, dim, None)  # type: ignore
            we = WorldEvent(sensory_vector=vec, reward=1.0, _symbol_id=4000 + i, _correct_channel=ask_channel)
            yield CurriculumEvent(phase=Phase.AMBIGUITY, world_event=we, episode_idx=i, step_in_episode=0, reset_after=False)

    def _phase_clarification(self) -> Iterator[CurriculumEvent]:
        """Clarification event followed by correct answer required."""
        dim = self.cfg.world_cfg.sensory_dim
        for i in range(self.cfg.clarification_trials):
            clar_vec = _symbol_vector(5000 + i, dim, None)  # type: ignore
            we = WorldEvent(sensory_vector=clar_vec, reward=1.0, _symbol_id=5000 + i, _correct_channel=i % self.cfg.world_cfg.n_roles)
            yield CurriculumEvent(phase=Phase.CLARIFICATION, world_event=we, episode_idx=i, step_in_episode=0, reset_after=False)

    def _phase_continued(self) -> Iterator[CurriculumEvent]:
        """Continued normal learning after revision and clarification."""
        for ep in range(self.cfg.continued_episodes):
            events = self.world.generate_episode()
            for s, we in enumerate(events):
                yield CurriculumEvent(phase=Phase.CONTINUED, world_event=we, episode_idx=ep, step_in_episode=s, reset_after=(s == len(events) - 1))
