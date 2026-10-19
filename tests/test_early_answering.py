"""Unit tests for the Early Answering score (crr.metric.early_answering)."""

import pytest

import crr.config as cfg
from crr.metric.early_answering import ea_shortcut_score_integral

P = cfg.POSITION_GRID


def test_constant_curves():
    assert ea_shortcut_score_integral({p: 1 for p in P}, P) == pytest.approx(1.0)
    assert ea_shortcut_score_integral({p: 0 for p in P}, P) == pytest.approx(0.0)


def test_trapezoid_on_default_grid():
    # b_EA = 0, 1, 1 at p = 0, 50, 100 -> 0.25 + 0.5
    assert ea_shortcut_score_integral({0: 0, 50: 1, 100: 1}) == pytest.approx(0.75)


def test_explicit_grid_overrides_keys():
    m = {0: 0, 50: 1, 100: 1, 25: 0}
    assert ea_shortcut_score_integral(m, [0, 50, 100]) == pytest.approx(0.75)
