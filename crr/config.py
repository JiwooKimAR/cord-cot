"""Central constants for CORD measurement and analysis.

The disruption measurement grid (P x R) shared by the two axes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Disruption measurement grid  (paper Sec. 4.1 / Appendix A.1)
# ---------------------------------------------------------------------------
# P: intervention/visibility positions as % of M (21 points). Shared by the
# A-axis (inverse decoding) and B-axis (X-mask recovery).
POSITION_GRID: list[int] = list(range(0, 101, 5))

# R: token-rank bands for A-axis inverse decoding (5 ranks). Primary = bottom-1.
RANK_GRID: list[str] = ["bottom-1", "bottom-5", "bottom-10", "top-5", "top-10"]
