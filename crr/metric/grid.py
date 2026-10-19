"""Disruption-grid construction and cutoff utilities.

Builds the per-instance a-matrix a(p,r) (Experiment A / necessity axis) and
b-curve b(p) (Experiment B / sufficiency axis) from raw measurement rows, and
the saturation cutoff p*_B. These are pure array utilities used by the
canonical metric (``crr.metric.cord``).
"""

from __future__ import annotations

import numpy as np

from crr.answers.leak import first_answer_pos_frac as _detector_pos_frac


# ---------------------------------------------------------------------------
# element-wise helpers
# ---------------------------------------------------------------------------


def trapz_normalized(values: np.ndarray, axis_points: list[int]) -> float:
    """Trapezoidal integral over axis normalized to [0, 1] (axis_points are %)."""
    x = np.asarray(axis_points, dtype=float) / 100.0
    return float(np.trapezoid(values, x))


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------
def build_a_matrix(
    rows: list[dict],
    query_id: int,
    intervention_points: list[int],
    rank_positions: list[str],
) -> np.ndarray | None:
    """Build a[p, r] for one instance from Experiment A rows. None if no rows."""
    a = np.zeros((len(intervention_points), len(rank_positions)), dtype=float)
    p_idx = {p: i for i, p in enumerate(intervention_points)}
    r_idx = {r: i for i, r in enumerate(rank_positions)}

    found = False
    for row in rows:
        if row.get("query_id") != query_id:
            continue
        if row.get("condition_type") != "main":
            continue
        if row.get("skipped") or row.get("is_shortcut") is None:
            continue
        p = row.get("intervention_point")
        r = row.get("rank_position")
        if p not in p_idx or r not in r_idx:
            continue
        a[p_idx[p], r_idx[r]] = 1.0 if row["is_shortcut"] else 0.0
        found = True

    return a if found else None


def build_b_curve(
    rows: list[dict],
    query_id: int,
    p_values: list[int],
    mask_unit: str,
) -> np.ndarray | None:
    """Build b[p] for one instance from Experiment B rows. None if no rows.

    ``is_shortcut=True`` in Experiment B means Y_B == Y_original.
    """
    b = np.zeros(len(p_values), dtype=float)
    p_idx = {p: i for i, p in enumerate(p_values)}

    found = False
    for row in rows:
        if row.get("query_id") != query_id:
            continue
        if row.get("condition_type") != "main":
            continue
        if row.get("mask_unit") != mask_unit:
            continue
        if row.get("skipped") or row.get("is_shortcut") is None:
            continue
        p = row.get("p_visible")
        if p not in p_idx:
            continue
        b[p_idx[p]] = 1.0 if row["is_shortcut"] else 0.0
        found = True

    return b if found else None


# ---------------------------------------------------------------------------
# cutoff p*_B
# ---------------------------------------------------------------------------
def _first_answer_pos_frac(cot: str, answer: str) -> float | None:
    if not cot or not answer:
        return None
    return _detector_pos_frac(cot, str(answer))


def compute_p_star(
    baseline_cot: str,
    baseline_answer: str,
    b_curve: np.ndarray,
    p_values: list[int],
    mode: str = "A1",
    saturation_threshold: float = 0.95,
) -> int:
    """Adaptive cutoff p*_B (Eq. 8).

    mode "A1" - first p where b_curve >= threshold (else max).
         "A2" - answer first-occurrence frac rounded down the grid (leak-free).
         "A3" - min(A1, A2).
    """
    p_max = max(p_values)
    a1 = p_max
    for i, p in enumerate(p_values):
        if b_curve[i] >= saturation_threshold:
            a1 = p
            break
    frac = _first_answer_pos_frac(baseline_cot, baseline_answer)
    if frac is None:
        a2 = p_max
    else:
        ans_p = frac * 100
        a2 = max([p for p in p_values if p < ans_p], default=0)
    if mode == "A1":
        return a1
    if mode == "A2":
        return a2
    if mode == "A3":
        return min(a1, a2)
    raise ValueError(f"Unknown cutoff mode: {mode!r} (expected 'A1', 'A2', or 'A3')")
