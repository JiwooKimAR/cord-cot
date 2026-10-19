"""Unit tests for the canonical CORD metric (crr.metric.cord).

Locks the metric definition CORD = H(a_bar, q*_B) and its four pre-registered
axioms (paper Appendix B.2).
"""

import numpy as np
import pytest

from crr.metric.cord import (
    harmonic_mean,
    a_component,
    b_component,
    cord,
    score,
    CordResult,
)
import crr.config as cfg


P_GRID = cfg.POSITION_GRID
RANKS = cfg.RANK_GRID


# ---------------------------------------------------------------------------
# Harmonic mean + components
# ---------------------------------------------------------------------------
def test_harmonic_boundaries():
    assert harmonic_mean(0.0, 0.0) == 0.0
    assert harmonic_mean(0.0, 1.0) == 0.0      # A3: one axis zero -> 0
    assert harmonic_mean(1.0, 0.0) == 0.0
    assert harmonic_mean(1.0, 1.0) == pytest.approx(1.0)   # A4
    assert harmonic_mean(0.5, 0.5) == pytest.approx(0.5)


def test_b_component():
    assert b_component(0) == pytest.approx(1.0)
    assert b_component(40) == pytest.approx(0.6)
    assert b_component(100) == pytest.approx(0.0)
    assert b_component(150) == 0.0   # clamped at 0


def test_a_component_is_full_grid_mean():
    a = np.zeros((len(P_GRID), len(RANKS)))
    a[:10, :] = 1.0
    assert a_component(a) == pytest.approx(a.mean())


# ---------------------------------------------------------------------------
# Axioms A1-A4 (paper Appendix B.2)
# ---------------------------------------------------------------------------
def test_axiom_A1_monotone_in_a():
    for b in (0.2, 0.5, 0.9):
        vals = [cord(a, int((1 - b) * 100)) for a in (0.1, 0.3, 0.6, 0.9)]
        assert all(x < y for x, y in zip(vals, vals[1:]))


def test_axiom_A2_monotone_in_b():
    # b increases as p_star decreases
    for a in (0.2, 0.5, 0.9):
        vals = [cord(a, p) for p in (90, 60, 30, 0)]
        assert all(x < y for x, y in zip(vals, vals[1:]))


def test_axiom_A3_zero_at_boundary():
    assert cord(0.0, 50) == 0.0     # a_bar = 0
    assert cord(0.7, 100) == 0.0    # q*_B = 0


def test_axiom_A4_one_at_corner():
    assert cord(1.0, 0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score() end-to-end on a synthetic instance
# ---------------------------------------------------------------------------
def test_score_full_from_matrix():
    # a-matrix all ones -> a_bar = 1; b-curve saturates at p=20 -> p*_B = 20
    a = np.ones((len(P_GRID), len(RANKS)))
    b = np.array([1.0 if p >= 20 else 0.0 for p in P_GRID])
    r = score(a, b, P_GRID)
    assert isinstance(r, CordResult)
    assert r.a_mean == pytest.approx(1.0)
    assert r.p_star == 20
    assert r.b_comp == pytest.approx(0.8)
    assert r.cord == pytest.approx(harmonic_mean(1.0, 0.8))
    assert float(r) == pytest.approx(r.cord)   # CordResult behaves like its scalar


def test_cord_scalar_headline():
    assert cord(a_mean=0.70, p_star=40) == pytest.approx(harmonic_mean(0.7, 0.6))


def test_cord_matches_inline_formula():
    """CORD equals the closed-form harmonic mean of a_bar and 1 - p*_B/100."""
    def inline_cord(a, p_star):
        b = max(0.0, 1.0 - p_star / 100.0)
        return 0.0 if a + b <= 0 else 2 * a * b / (a + b)

    for a in (0.0, 0.3, 0.686, 1.0):
        for p in (0, 40, 95, 100):
            assert cord(a, p) == pytest.approx(inline_cord(a, p))
