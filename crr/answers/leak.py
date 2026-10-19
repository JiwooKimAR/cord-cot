"""Answer-string occurrence in the baseline CoT.

Locates where the baseline answer Y_0 first appears in the CoT text, used by the
answer-first-occurrence cutoff modes of p*_B (``crr.metric.grid.compute_p_star``).

- ``normalize(answer_str)`` - LaTeX/markdown cleanup, whitespace, trailing
  punctuation, thousands separators, case fold. No numeric equivalence
  (1/2 vs 0.5 are different answers).
- ``first_answer_pos_frac(cot, answer)`` - boundary-aware position of the first
  occurrence of the answer in the CoT, as a fraction of the CoT length.
"""

from __future__ import annotations

import re


# -----------------------------------------------------------------------------
# normalize
# -----------------------------------------------------------------------------

# Thousands separator: pattern (?<=\d),(?=\d{3}(\D|$)) on numbers with 4+ digits.
# Decimal commas preserved: removed only when followed by exactly 3 digits and then a non-digit.
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# \boxed{X} -> X (up to one level of nested braces).
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")

# \frac{a}{b} -> a/b. No nesting support (sufficient for simple MATH-500 cases).
_FRAC = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

# \dfrac and \tfrac handled the same way: normalized to \frac.
_DTFRAC = re.compile(r"\\[dt]frac\b")

# Markdown emphasis: **X**, *X*, __X__, _X_ -> X. Non-greedy *? to avoid over-matching.
_MD_BOLD = re.compile(r"\*\*([^*]+?)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
_MD_UNDERLINE_BOLD = re.compile(r"__([^_]+?)__")
_MD_UNDERLINE_ITALIC = re.compile(r"(?<!_)_([^_]+?)_(?!_)")

# Inline math $...$ - simple strip (keep only the contents).
_MATH_INLINE = re.compile(r"\$([^$]+)\$")

# Trailing punctuation (period/comma/exclam) - remove one or more.
_TRAILING_PUNCT = re.compile(r"[.,!]+$")

# Whitespace collapse.
_WS = re.compile(r"\s+")


def normalize(answer_str: str) -> str:
    """Normalize an answer string for leak matching.

    Order of operations matters - LaTeX -> markdown -> math delim -> whitespace
    -> trailing punct -> thousands -> case fold.

    No numeric equivalence: ``1/2`` and ``0.5`` are different answers.
    """
    if answer_str is None:
        return ""
    s = str(answer_str)

    # LaTeX
    s = _DTFRAC.sub(r"\\frac", s)
    # Nested \boxed/\frac may occur, so use a fixed-point loop (enough for simple structures).
    for _ in range(3):
        new = _BOXED.sub(r"\1", s)
        new = _FRAC.sub(r"\1/\2", new)
        if new == s:
            break
        s = new
    s = _MATH_INLINE.sub(r"\1", s)

    # Markdown - bold first (** before *), same for underline.
    s = _MD_BOLD.sub(r"\1", s)
    s = _MD_UNDERLINE_BOLD.sub(r"\1", s)
    s = _MD_ITALIC.sub(r"\1", s)
    s = _MD_UNDERLINE_ITALIC.sub(r"\1", s)

    # Whitespace
    s = _WS.sub(" ", s).strip()

    # Trailing punct
    s = _TRAILING_PUNCT.sub("", s)

    # Thousands separators (decimal commas preserved)
    s = _THOUSANDS_COMMA.sub("", s)

    # Case fold (Latin). No effect on Korean/digits/LaTeX.
    s = s.casefold()

    return s


# -----------------------------------------------------------------------------
# boundary-aware answer pattern
# -----------------------------------------------------------------------------

def _build_direct_pattern(normalized_y0: str) -> re.Pattern:
    """Boundary-aware pattern matching ``normalized_y0`` as a standalone token.

    Safe for expressions (e.g. ``1/2``, ``a^2 + b^2``) as well as numeric answers.

    Word boundary definition:
    - Neither side may be ``\\w`` (ASCII) (Hangul is not a word char, so OK).
    - ``.`` and ``/`` break the boundary *only when followed by a digit*. I.e. matching
      ``3`` in ``3.14`` is rejected, but matching ``18`` in ``18.`` (sentence-final
      period) is allowed. Matching ``18`` in ``18/2`` is rejected.

    ``re.ASCII`` flag - so that Hangul is not a word char.
    """
    escaped = re.escape(normalized_y0)
    # Lookbehind: preceding char is a word char, or a digit followed by . or /
    # (lookbehind only allows fixed width, so expressed as alternatives).
    # Must be neither a word char, nor (digit two back AND ./ just before).
    # Simplification: if ``escaped`` starts with a digit, (?<![\w]) suffices; the
    # ``.``/``/`` break is expressed separately via lookahead/lookbehind.
    pat = (
        rf"(?<![\w])"                     # preceded by non-word char (Hangul OK)
        rf"(?<!\d[./])"                   # not preceded by (digit + . or /)
        rf"{escaped}"
        rf"(?![\w])"                      # followed by non-word char
        rf"(?![./]\d)"                    # not followed by (./ + digit)
    )
    return re.compile(pat, flags=re.ASCII)


# -----------------------------------------------------------------------------
# Convenience - relative position of the first answer occurrence in the visible
# region (boundary-aware).
# -----------------------------------------------------------------------------


def first_answer_pos_frac(cot: str, answer: str) -> float | None:
    """Position of the first occurrence of the answer string in baseline_cot
    (as a fraction of cot length).

    Boundary-aware (see ``_build_direct_pattern``). None if the answer is not in cot.
    """
    if not cot or not answer:
        return None
    norm_cot = normalize(cot)
    norm_ans = normalize(answer)
    if not norm_cot or not norm_ans:
        return None
    pat = _build_direct_pattern(norm_ans)
    m = pat.search(norm_cot)
    if m is None:
        return None
    # frac is relative to the *normalized* cot length. normalize slightly changes
    # the length (markdown removal, whitespace collapse), so frac may differ
    # marginally from a raw-length fraction.
    return m.start() / len(norm_cot)
