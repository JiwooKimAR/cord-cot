"""Unit tests for crr/answers/leak.py.

Spec cases + edge cases for ``normalize`` and the boundary-aware answer
matching behind ``first_answer_pos_frac``.

Run: ``pytest tests/test_leak_detector.py`` or ``python tests/test_leak_detector.py``.
"""

from __future__ import annotations

import os
import sys

# Add repo root to sys.path - importable even without pytest installed.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pytest
    HAS_PYTEST = True
except ImportError:  # pragma: no cover
    HAS_PYTEST = False

    class _PytestStub:  # minimal stub so module imports without pytest.
        @staticmethod
        def skip(reason: str) -> None:
            raise RuntimeError(f"SKIP: {reason}")

        class mark:  # type: ignore[misc]
            @staticmethod
            def parametrize(*args, **kwargs):
                def decorator(fn):
                    fn._parametrized = True
                    return fn
                return decorator

    pytest = _PytestStub()  # type: ignore[assignment]

from crr.answers.leak import (
    first_answer_pos_frac,
    normalize,
)


# ---------------------------------------------------------------------------
# Boundary-aware answer matching - spec cases
# ---------------------------------------------------------------------------
#
# Each case: (visible_text, y0_raw, label). ``first_answer_pos_frac`` locates the
# answer with the same boundary-aware pattern, so "found" means not None.
# ---------------------------------------------------------------------------

FOUND_CASES = [
    ("answer is 18",       "18",   "plain"),
    (r"\boxed{18}",        "18",   "latex"),
    ("**18**",             "18",   "markdown"),
    ("the answer is 1,800", "1800", "thousands"),
    ("x = 18",             "18",   "equation"),
    (r"answer is $18$",    "18",   "latex math"),
    ("page 18 of paper",   "18",   "isolated occurrence (word boundaries on both sides)"),
]

NOT_FOUND_CASES = [
    ("answer is 180",        "18", "substring (longer number)"),
    ("answer is 2018",       "18", "substring (longer prefix)"),
    ("18th step",            "18", "ordinal"),
]


# ---------------------------------------------------------------------------
# normalize() - normalization-only cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("18", "18"),
        (r"\boxed{18}", "18"),
        (r"\boxed{ 18 }", "18"),
        ("**18**", "18"),
        ("__18__", "18"),
        ("*18*", "18"),
        ("1,800", "1800"),
        ("1,234,567", "1234567"),
        ("3.14", "3.14"),                # decimal point preserved
        ("3,14", "3,14"),                # fewer than 4 digits + exactly 2 after: not a thousands comma - preserved (safe)
        (r"\frac{1}{2}", "1/2"),
        (r"\dfrac{1}{2}", "1/2"),
        (r"\tfrac{1}{2}", "1/2"),
        ("Answer", "answer"),            # case fold
        ("18.", "18"),
        ("18!", "18"),
        ("18,", "18"),
        ("  18  \n", "18"),              # whitespace
        (r"$18$", "18"),                 # inline math
        ("", ""),
        (None, ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


# ---------------------------------------------------------------------------
# Boundary-aware matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("visible, y0_raw, label", FOUND_CASES)
def test_answer_found(visible, y0_raw, label):
    assert first_answer_pos_frac(visible, y0_raw) is not None, label


@pytest.mark.parametrize("visible, y0_raw, label", NOT_FOUND_CASES)
def test_answer_not_found(visible, y0_raw, label):
    assert first_answer_pos_frac(visible, y0_raw) is None, label


# ---------------------------------------------------------------------------
# Edge cases - empty, fraction normalize, numerical-equivalence rejection
# ---------------------------------------------------------------------------

def test_edge_empty_visible():
    assert first_answer_pos_frac("", "18") is None


def test_edge_empty_y0():
    assert first_answer_pos_frac("answer is 18", "") is None


def test_edge_fraction_normalize():
    """``\\frac{1}{2}`` in visible matches ``1/2`` after normalize."""
    assert first_answer_pos_frac(r"\frac{1}{2}", "1/2") is not None


def test_edge_dfrac_tfrac_normalize():
    assert first_answer_pos_frac(r"\dfrac{1}{2}", "1/2") is not None
    assert first_answer_pos_frac(r"\tfrac{1}{2}", "1/2") is not None


def test_edge_numerical_equivalence_rejected():
    """Answer ``1/2`` with visible ``0.5`` - numerical equivalence not applied."""
    assert first_answer_pos_frac("0.5", "1/2") is None


def test_edge_thousands_comma():
    """Answer ``1800`` with visible ``1,800`` - match after normalize."""
    assert first_answer_pos_frac("the answer is 1,800", "1800") is not None


def test_edge_decimal_point_preserved():
    """Decimal point differs from comma. Answer ``3.14`` with visible ``3.14`` - match."""
    assert first_answer_pos_frac("answer is 3.14", "3.14") is not None


# ---------------------------------------------------------------------------
# first_answer_pos_frac - regression for crr.metric.grid.compute_p_star
# ---------------------------------------------------------------------------

def test_first_answer_pos_frac_returns_position():
    cot = "Step 1: x = 5. Step 2: y = 18. Final answer: 18."
    frac = first_answer_pos_frac(cot, "18")
    assert frac is not None
    # Position of the first real 18 - middle of the cot (inside Step 2). normalize
    # slightly changes the length so the exact frac is a bit off, but 0.4-0.7 is
    # clearly the middle.
    assert 0.4 < frac < 0.7


def test_first_answer_pos_frac_none_when_absent():
    cot = "Step 1: x = 5. The answer is 7."
    assert first_answer_pos_frac(cot, "18") is None


def test_first_answer_pos_frac_boundary_aware():
    """``18`` inside ``180`` must not be caught as the first occurrence."""
    cot = "We have 180 apples. After dividing by 10 we get 18."
    frac = first_answer_pos_frac(cot, "18")
    assert frac is not None
    # Position of the first real 18 (not 180) - latter half of the cot.
    assert frac > 0.5


# ---------------------------------------------------------------------------
# Direct execution - works without pytest installed (fallback)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Without pytest, call only the non-parametrized functions directly.
    import inspect
    failures: list[tuple[str, BaseException]] = []
    passed = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        if sig.parameters:
            # parametrized test - not called directly, needs pytest.
            continue
        try:
            fn()
        except BaseException as e:
            failures.append((name, e))
        else:
            passed += 1
    print(f"{passed} passed, {len(failures)} failed (parametrized cases need pytest).")
    for name, e in failures:
        print(f"  FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
