"""Hash frozen slow weights (species prior). Must not change during a life."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def hash_arrays(arrays: Iterable[np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(a.dtype.str.encode("ascii"))
        h.update(np.array(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()
