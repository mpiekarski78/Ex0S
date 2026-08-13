"""Tiny key/door world: one hidden fact — red door opens only with key."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class Action(IntEnum):
    WAIT = 0
    OPEN = 1
    USE_KEY = 2
    PICK_KEY = 3


ACTION_NAMES = {a: a.name.lower() for a in Action}


@dataclass
class Obs:
    """Symbolic observation; also packed into a fixed bit vector for the cortex."""

    at_red_door: bool = False
    at_blue_door: bool = False
    has_key: bool = False
    key_visible: bool = False
    last_failed: bool = False
    last_succeeded: bool = False
    event_open_failed: bool = False
    event_key_worked: bool = False

    def vector(self, dim: int = 16) -> np.ndarray:
        bits = [
            self.at_red_door,
            self.at_blue_door,
            self.has_key,
            self.key_visible,
            self.last_failed,
            self.last_succeeded,
            self.event_open_failed,
            self.event_key_worked,
        ]
        v = np.zeros(dim, dtype=np.float64)
        for i, b in enumerate(bits):
            v[i] = 1.0 if b else 0.0
        return v

    def tags(self) -> dict[str, Any]:
        return {
            "at_red_door": self.at_red_door,
            "at_blue_door": self.at_blue_door,
            "has_key": self.has_key,
        }


@dataclass
class StepResult:
    obs: Obs
    reward: float
    success: bool | None
    done: bool
    info: dict[str, Any]


class KeyDoorWorld:
    """
    Rooms-with-objects toy.
    Fact: red door opens only with USE_KEY while holding key.
    Blue door opens with OPEN (foil / distractor).
    """

    FACT_ID = "red_door_needs_key"
    FACT_TEXT = "red door opens only with key"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.has_key = False
        self.phase = "start"
        self.t = 0

    def reset(self, scenario: str = "probe_red_with_key") -> Obs:
        self.has_key = False
        self.t = 0
        self.phase = scenario
        return self._obs_for_scenario(scenario, event=None)

    def _obs_for_scenario(self, scenario: str, event: str | None) -> Obs:
        o = Obs()
        if scenario in ("experience_teach", "probe_red_with_key", "probe_red_no_key"):
            o.at_red_door = True
        elif scenario in ("experience_foil", "probe_blue"):
            o.at_blue_door = True
        if scenario == "probe_red_with_key":
            self.has_key = True
        if scenario == "probe_red_no_key":
            self.has_key = False
            o.key_visible = True
        if scenario == "experience_teach":
            # Teaching: key available; agent must learn contingency.
            o.key_visible = not self.has_key
        o.has_key = self.has_key
        if event == "open_failed":
            o.event_open_failed = True
            o.last_failed = True
        elif event == "key_worked":
            o.event_key_worked = True
            o.last_succeeded = True
        elif event == "opened":
            o.last_succeeded = True
        elif event == "picked":
            o.last_succeeded = True
        return o

    def step(self, action: int, scenario: str | None = None) -> StepResult:
        if scenario is not None:
            self.phase = scenario
        self.t += 1
        action = int(action)
        event = None
        success: bool | None = None
        reward = 0.0
        done = False
        info: dict[str, Any] = {"action": ACTION_NAMES.get(Action(action), str(action))}

        if self.phase in ("experience_teach", "probe_red_with_key", "probe_red_no_key"):
            if action == Action.PICK_KEY and not self.has_key:
                self.has_key = True
                success = True
                reward = 0.1
                event = "picked"
            elif action == Action.OPEN:
                # Opening red door bare-handed always fails.
                success = False
                reward = -1.0
                event = "open_failed"
            elif action == Action.USE_KEY:
                if self.has_key:
                    success = True
                    reward = 1.0
                    event = "key_worked"
                    done = True
                    info["opened"] = True
                else:
                    success = False
                    reward = -0.5
                    event = "open_failed"
            else:
                success = None
                reward = -0.01

        elif self.phase in ("experience_foil", "probe_blue"):
            if action == Action.OPEN:
                success = True
                reward = 1.0
                event = "opened"
                done = True
                info["opened"] = True
            elif action == Action.USE_KEY:
                success = False
                reward = -0.5
                event = "open_failed"
            else:
                success = None
                reward = -0.01

        obs = self._obs_for_scenario(self.phase, event)
        return StepResult(obs=obs, reward=reward, success=success, done=done, info=info)
