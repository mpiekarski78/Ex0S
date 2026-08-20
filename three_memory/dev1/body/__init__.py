"""
Developmental Birth R4 body package.

Import submodules directly when possible to avoid circular imports.
"""

__all__ = [
    "BodyConfig",
    "BodyState",
    "BodyStepResult",
    "GenericBody",
    "ClosedLoopGroundingWorld",
]


def __getattr__(name: str):
    if name in ("BodyConfig", "BodyState", "BodyStepResult", "GenericBody"):
        from three_memory.dev1.body import physics as p
        return getattr(p, name)
    if name == "ClosedLoopGroundingWorld":
        from three_memory.dev1.body.world import ClosedLoopGroundingWorld
        return ClosedLoopGroundingWorld
    raise AttributeError(name)
