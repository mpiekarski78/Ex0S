"""Intervention flags for Reference Birth causal ablations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpropIntervention:
    name: str = "none"
    reward_off: bool = False
    eligibility_zero: bool = False
    eligibility_permuted: bool = False
    motor_feedback_permuted: bool = False
    signal_generator_off: bool = False
    signal_generator_permuted: bool = False

    @classmethod
    def none(cls) -> "EpropIntervention":
        return cls()

    @classmethod
    def with_reward_off(cls) -> "EpropIntervention":
        return cls(name="reward_off", reward_off=True)

    @classmethod
    def with_eligibility_zero(cls) -> "EpropIntervention":
        return cls(name="eligibility_zero", eligibility_zero=True)

    @classmethod
    def with_eligibility_permuted(cls) -> "EpropIntervention":
        return cls(name="eligibility_permuted", eligibility_permuted=True)

    @classmethod
    def with_motor_feedback_permuted(cls) -> "EpropIntervention":
        return cls(name="motor_feedback_permuted", motor_feedback_permuted=True)

    @classmethod
    def with_signal_generator_off(cls) -> "EpropIntervention":
        return cls(name="signal_generator_off", signal_generator_off=True)

    @classmethod
    def with_signal_generator_permuted(cls) -> "EpropIntervention":
        return cls(name="signal_generator_permuted", signal_generator_permuted=True)
