"""
Developmental Birth R4 — generative construction and gestation.

Import submodules directly to avoid circular imports with body/.
"""

__all__ = [
    "GenerativeGenome",
    "ConstructionReceipt",
    "construct_post_growth_organism",
    "GestationMode",
    "GestationReceipt",
    "run_gestation",
    "OrganismValenceCircuit",
]


def __getattr__(name: str):
    if name == "GenerativeGenome":
        from three_memory.dev1.development.generative_genome import GenerativeGenome
        return GenerativeGenome
    if name in ("ConstructionReceipt", "construct_post_growth_organism"):
        from three_memory.dev1.development import construction as c
        return getattr(c, name)
    if name in ("GestationMode", "GestationReceipt", "run_gestation"):
        from three_memory.dev1.development import gestation as g
        return getattr(g, name)
    if name == "OrganismValenceCircuit":
        from three_memory.dev1.development.valence import OrganismValenceCircuit
        return OrganismValenceCircuit
    raise AttributeError(name)
