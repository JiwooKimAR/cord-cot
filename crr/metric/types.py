"""Shared type aliases for the metric layer (self-documenting signatures)."""

from __future__ import annotations

import numpy as np

# a-matrix a(p, r): shape (|P|, |R|), binary entries 1[Y_A(p,r) == Y_0].
AMatrix = np.ndarray
# b-curve b(p): shape (|P|,), binary entries 1[Y_B(p) == Y_0].
BCurve = np.ndarray
# Position grid P: percentages of M, e.g. [0, 5, ..., 100].
PGrid = list[int]

__all__ = ["AMatrix", "BCurve", "PGrid"]
