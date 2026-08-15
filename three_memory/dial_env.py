"""Channel dial world for TM.0.4.0 — not a key/door toy.

Three channels. Correct motor ≠ place code on the held-out channel.
Species prior (in the agent) prefers HOLD — wrong on A and C — so empty S fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class DialAction(IntEnum):
    IDLE = 0
    PRESS = 1
    HOLD = 2
    TUNE = 3
    FLIP = 4


DIAL_ACTION_NAMES = {a: a.name.lower() for a in DialAction}

# Place codes deliberately ≠ correct motor on held-out C (C code == PRESS).
CH_A = 0
CH_B = 4
CH_C = 1  # equals PRESS — copying place fails held-out

# Innate station names (body vocabulary). Not English. Not place ints.
STATION_NAMES = {CH_A: "cha", CH_B: "chb", CH_C: "chc"}

CORRECT = {
    CH_A: DialAction.PRESS,
    CH_B: DialAction.HOLD,
    CH_C: DialAction.TUNE,
}


@dataclass
class DialObs:
    at_a: bool = False
    at_b: bool = False
    at_c: bool = False
    last_failed: bool = False
    last_ok: bool = False
    # Not a cortex input. MATCH reads this; vector() must stay bit-identical.
    tokens: frozenset[str] = field(default_factory=frozenset)

    def vector(self, dim: int = 16) -> np.ndarray:
        bits = [self.at_a, self.at_b, self.at_c, self.last_failed, self.last_ok]
        v = np.zeros(dim, dtype=np.float64)
        for i, b in enumerate(bits):
            v[i] = 1.0 if b else 0.0
        return v


@dataclass
class DialStepResult:
    obs: DialObs
    reward: float
    success: bool | None
    done: bool
    info: dict[str, Any]


class ChannelDialWorld:
    """Bench with three channels. Open the channel with the right motor act."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.phase = "probe_channel_a"
        self.t = 0
        self.channel = CH_A

    def reset(self, scenario: str = "probe_channel_a") -> DialObs:
        self.t = 0
        self.phase = scenario
        if scenario in ("experience_channel_a", "probe_channel_a"):
            self.channel = CH_A
        elif scenario in ("experience_channel_b", "probe_channel_b"):
            self.channel = CH_B
        elif scenario in ("experience_channel_c", "probe_channel_c"):
            self.channel = CH_C
        else:
            raise ValueError(scenario)
        return self._obs(event=None)

    def _obs(self, event: str | None) -> DialObs:
        o = DialObs(
            at_a=self.channel == CH_A,
            at_b=self.channel == CH_B,
            at_c=self.channel == CH_C,
        )
        if event == "failed":
            o.last_failed = True
        elif event == "ok":
            o.last_ok = True
        return o

    def place_code(self) -> int:
        return int(self.channel)

    def step(self, action: int, scenario: str | None = None) -> DialStepResult:
        if scenario is not None:
            self.phase = scenario
            if "channel_a" in scenario:
                self.channel = CH_A
            elif "channel_b" in scenario:
                self.channel = CH_B
            elif "channel_c" in scenario:
                self.channel = CH_C
        self.t += 1
        action = int(action)
        need = CORRECT[self.channel]
        info: dict[str, Any] = {
            "action": DIAL_ACTION_NAMES.get(DialAction(action), str(action)),
            "station": STATION_NAMES[self.channel],
            "channel": self.channel,
            "need": int(need),
        }
        if action == DialAction.IDLE or action == DialAction.FLIP:
            return DialStepResult(self._obs(None), 0.0, None, False, info)
        if action == int(need):
            info["opened"] = True
            return DialStepResult(self._obs("ok"), 1.0, True, True, info)
        info["opened"] = False
        return DialStepResult(self._obs("failed"), -1.0, False, False, info)
