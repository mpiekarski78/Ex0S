"""Reward-based e-prop rate adaptation (EX0S-DEV1 Reference Birth)."""

from three_memory.dev1.plasticity.eprop.reward_eprop import RewardEpropRateAdaptation
from three_memory.dev1.plasticity.eprop.learned_eprop import InheritedSignalGeneratorEprop
from three_memory.dev1.plasticity.eprop.signal_generator import InheritedLearningSignalGenerator

__all__ = [
    "RewardEpropRateAdaptation",
    "InheritedSignalGeneratorEprop",
    "InheritedLearningSignalGenerator",
]
