"""Unit tests for grid construction and the p*_B cutoff (crr.metric.grid)."""

import numpy as np
import pytest

import crr.config as cfg
from crr.metric.grid import build_a_matrix, build_b_curve, compute_p_star
from crr.prompts import ANSWER_CUE, SYSTEM_PROMPT

P = cfg.POSITION_GRID
R = cfg.RANK_GRID


def _a_row(qid, p, r, is_shortcut, condition="main"):
    return {"query_id": qid, "condition_type": condition, "intervention_point": p,
            "rank_position": r, "is_shortcut": is_shortcut}


def _b_row(qid, p, is_shortcut, unit="token", condition="main"):
    return {"query_id": qid, "condition_type": condition, "p_visible": p,
            "mask_unit": unit, "is_shortcut": is_shortcut}


# ---------------------------------------------------------------------------
# build_a_matrix / build_b_curve
# ---------------------------------------------------------------------------
def test_a_matrix_fills_cells():
    rows = [_a_row(0, p, r, True) for p in P for r in R]
    rows[3]["is_shortcut"] = False
    a = build_a_matrix(rows, 0, P, R)
    assert a.shape == (len(P), len(R))
    assert a.sum() == len(P) * len(R) - 1


def test_no_answer_cells_count_as_changed():
    """is_shortcut=None (no extractable answer) contributes 0, like a changed answer."""
    rows = [_a_row(0, p, r, True) for p in P for r in R]
    rows[0]["is_shortcut"] = None
    a = build_a_matrix(rows, 0, P, R)
    assert a[0, 0] == 0.0
    assert a.mean() == pytest.approx((len(P) * len(R) - 1) / (len(P) * len(R)))


def test_a_matrix_none_when_no_usable_rows():
    rows = [_a_row(0, 0, "bottom-1", None), _a_row(1, 0, "bottom-1", True),
            _a_row(0, 5, "bottom-1", True, condition="no_cot")]
    assert build_a_matrix(rows, 0, P, R) is None


def test_b_curve_filters_mask_unit_and_query():
    rows = [_b_row(0, p, p >= 50) for p in P]
    rows += [_b_row(0, p, True, unit="sentence") for p in P]
    rows += [_b_row(1, p, True) for p in P]
    b = build_b_curve(rows, 0, P, mask_unit="token")
    assert b.tolist() == [1.0 if p >= 50 else 0.0 for p in P]


# ---------------------------------------------------------------------------
# compute_p_star cutoff modes
# ---------------------------------------------------------------------------
# 100-char CoT whose answer "18" starts at char 41 -> first-occurrence frac 0.41.
COT = "a" * 40 + " 18 " + "b" * 56
B_SAT_60 = np.array([1.0 if p >= 60 else 0.0 for p in P])


def test_p_star_a1_first_saturated_point():
    assert compute_p_star(COT, "18", B_SAT_60, P, mode="A1") == 60


def test_p_star_a1_never_saturates():
    assert compute_p_star("", "", np.zeros(len(P)), P, mode="A1") == 100


def test_p_star_a1_threshold():
    b = np.array([0.9 if p >= 30 else 0.0 for p in P])
    assert compute_p_star("", "", b, P, mode="A1", saturation_threshold=0.95) == 100
    assert compute_p_star("", "", b, P, mode="A1", saturation_threshold=0.9) == 30


def test_p_star_a2_answer_first_occurrence():
    assert compute_p_star(COT, "18", B_SAT_60, P, mode="A2") == 40


def test_p_star_a2_answer_absent():
    assert compute_p_star(COT, "7", B_SAT_60, P, mode="A2") == 100


def test_p_star_a3_is_min():
    assert compute_p_star(COT, "18", B_SAT_60, P, mode="A3") == 40


def test_p_star_unknown_mode():
    with pytest.raises(ValueError):
        compute_p_star(COT, "18", B_SAT_60, P, mode="A4")


# ---------------------------------------------------------------------------
# shared prompt strings (pinned: changing them changes every measurement)
# ---------------------------------------------------------------------------
def test_prompt_strings_pinned():
    assert SYSTEM_PROMPT == (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert ANSWER_CUE == "\n\nTherefore, the final answer is \\boxed{"
