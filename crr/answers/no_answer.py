"""Reason classification for no-answer (un-extracted) outputs - descriptive metadata.

Does not affect the verdict: is_shortcut for un-extracted rows remains None;
this tag exists so that reports can count refusals (a normal signal) separately
from generation collapse (loop-like outputs).

The categories were derived from the shapes of observed no-answer outputs, but
the rules here are defined as a surface-form procedure. Since the tag does not
change the verdict, the cost of a misclassification is only a cell shift in
report tallies.

Categories (in check order):
  empty          - essentially no alphanumeric content (".", ".}", "$." etc.)
  refusal        - refusal/request asking for the problem or context (a normal failure signal under X-mask)
  digit_loop     - simple repetition of one or two distinct digits ("1000000...", "0.000...")
  numeric_stream - a sequence of numbers/operators with no prose ("5 + 3 + 1.8 + ...", "12, 2, -3, ...")
  repetition     - mechanical repetition of the same phrase ("24m + 24m + 24m + ...")
  prose          - other prose (e.g. R1 restart monologues; text that never reached an answer)
"""

from __future__ import annotations

import re

# General forms of refusal/request phrases (observed in Qwen-7B-family refusals).
_REFUSAL_RE = re.compile(
    r"(could you (please )?provide"
    r"|please provide (a|the|more)"
    r"|i (have not|haven't) been given"
    r"|i need more (context|information)"
    r"|there (was|seems to be) a misunderstanding"
    r"|no (specific )?problem (was |has been )?(given|provided|stated))",
    re.IGNORECASE,
)

# LaTeX commands ( \boxed, \frac, ... ) are excluded from the prose character count -
# the judge passes full_cot with a "\boxed{" wrap appended, so it must not be counted.
_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")


def classify_no_answer(text: str) -> str:
    """Reason tag for an un-extracted output (returns one of the categories in the module docstring)."""
    s = (text or "").strip()
    stripped = _LATEX_CMD_RE.sub("", s)
    alnum = re.sub(r"[^0-9A-Za-z]", "", stripped)
    if len(alnum) < 3:
        return "empty"

    if _REFUSAL_RE.search(s):
        return "refusal"

    digits = re.sub(r"[^0-9]", "", stripped)
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    if len(digits) >= 20 and len(set(digits)) <= 2 and len(letters) <= 3:
        return "digit_loop"
    if len(digits) >= 10 and len(letters) <= max(3, int(0.1 * len(alnum))):
        return "numeric_stream"

    # Mechanical repetition of the same 4-word phrase. Threshold is >= 0.5 - a
    # "A + A + A + ..." repetition alternates exactly two 4-shingles half each, landing at 0.5.
    # (The > 0.5 in extract._looks_like_gibberish, used for the verdict, is left as is -
    # this is a descriptive tag, so relaxing it here does not change the verdict.)
    words = re.findall(r"\S+", stripped)
    if len(words) >= 12:
        shingles = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        if shingles:
            top = max(set(shingles), key=shingles.count)
            if shingles.count(top) / len(shingles) >= 0.5:
                return "repetition"

    return "prose"
