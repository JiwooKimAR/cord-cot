"""crr - Causally Redundant Reasoning.

CORD: a dual-disruption shortcut-tendency metric for chain-of-thought.

    from crr import cord, score
    cord(a_mean=0.70, p_star=40)        # 0.646   (scalar, the common case)
    r = score(a_matrix, b_curve, P)     # CordResult(cord=.., a_mean=.., p_star=..)
    r.cord, r.p_star, float(r)
"""

from crr.metric.cord import (
    cord,
    score,
    CordResult,
)

__version__ = "1.0.0"

__all__ = [
    "cord",
    "score",
    "CordResult",
    "__version__",
]
