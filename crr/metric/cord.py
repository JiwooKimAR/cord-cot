"""CORD - Causal Redundancy of Reasoning via Disruption (canonical metric).

The shortcut-tendency metric reported throughout the paper::

    CORD(X) = H( a_bar(X), q*_B(X) )                         (paper Eq. 9)

    a_bar(X) = simple mean of binary a(p,r) over the full P x R grid  (Eq. 7)
    q*_B(X) = 1 - p*_B(X) / 100,  p*_B = min{p : b(p) = 1} else max(P) (Eq. 8)
              (b is binary; the code tests b(p) >= 0.95)
    H(x,y)  = 2xy/(x+y), H(0,0) = 0

Typical use::

    from crr import cord, score
    cord(a_mean=0.70, p_star=40)        # -> 0.646   (the common scalar case)
    r = score(a_matrix, b_curve, P)     # -> CordResult(cord=.., a_mean=.., p_star=40)
    r.cord, r.p_star, float(r)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from crr.metric.grid import compute_p_star
from crr.metric.types import AMatrix, BCurve, PGrid

__all__ = [
    "cord",
    "score",
    "CordResult",
]


# ---------------------------------------------------------------------------
# components (module-internal mechanism)
# ---------------------------------------------------------------------------
def harmonic_mean(x: float, y: float) -> float:
    """H(x, y) = 2xy/(x+y); 0 when x + y <= 0 (paper Eq. 9)."""
    s = x + y
    return 2.0 * x * y / s if s > 0 else 0.0


def a_component(a_matrix: AMatrix) -> float:
    """a_bar(X): simple mean of the binary a-matrix over the full P x R grid (Eq. 7)."""
    return float(np.asarray(a_matrix, dtype=float).mean())


def b_component(p_star: int | float) -> float:
    """q*_B = max(0, 1 - p_star/100), the B-axis component (Sec. 3.4)."""
    return max(0.0, 1.0 - p_star / 100.0)


# ---------------------------------------------------------------------------
# headline scalar metric
# ---------------------------------------------------------------------------
def cord(a_mean: float, p_star: int | float) -> float:
    """CORD from its two components: H(a_bar, 1 - p_star/100).

    The common case - call directly with the per-instance ``a_mean`` and
    ``p_star``.
    """
    return harmonic_mean(a_mean, b_component(p_star))


# ---------------------------------------------------------------------------
# structured result from the raw grid
# ---------------------------------------------------------------------------
class CordResult(NamedTuple):
    """Instance-level CORD with its components. ``float(result)`` -> ``cord``."""

    cord: float
    a_mean: float
    p_star: int
    b_comp: float

    def __float__(self) -> float:
        return self.cord


def score(
    a_matrix: AMatrix,
    b_curve: BCurve,
    p_values: PGrid,
    baseline_cot: str = "",
    baseline_answer: str = "",
    saturation_threshold: float = 0.95,
    cutoff_mode: str = "A1",
) -> CordResult:
    """Compute CORD straight from the raw a-matrix and b-curve.

    ``p_star`` is the B-axis saturation position (Eq. 8); under the default
    ``cutoff_mode="A1"`` it depends only on ``b_curve`` and the threshold
    (baseline_cot / baseline_answer feed only the A2/A3 leakage modes).
    """
    a_mean = a_component(a_matrix)
    p_star = compute_p_star(
        baseline_cot, baseline_answer, b_curve, p_values,
        mode=cutoff_mode, saturation_threshold=saturation_threshold,
    )
    b_comp = b_component(p_star)
    return CordResult(cord=harmonic_mean(a_mean, b_comp),
                      a_mean=a_mean, p_star=p_star, b_comp=b_comp)
