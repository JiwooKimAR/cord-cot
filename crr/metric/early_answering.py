"""Early Answering score (Lanham et al. 2023), a comparison baseline for CORD (Sec. 5.2).

The CoT is truncated at p% and the answer is re-elicited; ``b_EA(p)`` is 1 when
that answer matches the full-CoT answer. The score integrates ``b_EA`` over p.
"""

from __future__ import annotations

import numpy as np

from crr.metric.grid import trapz_normalized
from crr.metric.types import PGrid

__all__ = ["ea_shortcut_score_integral"]


def ea_shortcut_score_integral(m_match_by_p: dict[int, int | float],
                               p_values: PGrid | None = None) -> float:
    """Trapezoidal integral of b_EA(p) dp over normalized [0,1] (paper-faithful EA score)."""
    ps = p_values if p_values is not None else sorted(m_match_by_p)
    vals = np.array([float(m_match_by_p[p]) for p in ps], dtype=float)
    return trapz_normalized(vals, ps)
