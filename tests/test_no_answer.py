"""Unit tests for the no-answer reason classifier (crr.answers.no_answer).

Cases are taken from real no-answer outputs observed in a full re-judging
pass. Inputs wrapped with "\\boxed{", in the same form the judge passes
them, are also checked.
"""

import pytest

from crr.answers.no_answer import classify_no_answer
from crr.judge import judge_shortcut


CASES = [
    # empty - punctuation only / immediate termination
    (".", "empty"),
    (".}", "empty"),
    ("?.", "empty"),
    ("$.}", "empty"),
    ("\\boxed{.}.", "empty"),
    # refusal - refusal/request (forms observed on Qwen-7B x gsm8k/math500)
    (".}. \n\nIt seems I have not been given a specific problem to solve. "
     "Could you please provide a problem for me to solve step by step?",
     "refusal"),
    (".}. It seems there was a misunderstanding in the instruction. Could "
     "you please provide a specific problem or context for me to reason",
     "refusal"),
    (".}.\n\nIt seems I need more context or a specific problem to solve.",
     "refusal"),
    # digit_loop - repetition of one or two distinct digits
    ("1" + "0" * 63, "digit_loop"),
    ("0." + "0" * 62, "digit_loop"),
    ("1" + "9" * 63, "digit_loop"),
    ("1.6" + "6" * 60, "digit_loop"),
    ("10" * 32, "digit_loop"),
    # numeric_stream - sequence of numbers/operators (series expansion, sequences)
    ("5 + 3 + 1.8 + 1.08 + 0.648 + 0.3888 + 0.23328 + 0.139968 + 0.083981",
     "numeric_stream"),
    ("720 - 2*144 - 2*144 - 2*144 - 2*144 - 2*144 - 2*144 - 2*144",
     "numeric_stream"),
    ("12, 2, -3, 2, 13, 12, 1, 1, 12, 1, 1, 12, 1, 1, 12, 1, 1, 12, 1,",
     "numeric_stream"),
    # repetition - repeated identical phrase (not numeric_stream since unit letters are mixed in)
    ("24m + 24m + 24m + 24m + 24m + 24m + 24m + 24m + 24m + 24m + 24m + 24m",
     "repetition"),
    # prose - answerless prose such as R1 restart monologues
    ("...}.\n\nWait, hold on. I just realized I didn't finish the problem. "
     "Let me go back. So, the problem is about the city of Soda with",
     "prose"),
    ("(-1, -1, -1). Wait, hold on, that doesn't seem right. Let me "
     "double-check my steps.\n\nFirst, I found the roots a, b, and c",
     "prose"),
]


@pytest.mark.parametrize("text, expected", CASES)
def test_classify(text, expected):
    assert classify_no_answer(text) == expected


def test_judge_attaches_reason_on_unextracted():
    j = judge_shortcut("\\boxed{" + ".}. It seems I have not been given a "
                       "specific problem to solve. Could you please provide "
                       "a problem?", "42")
    assert j["is_shortcut"] is None
    assert j["no_answer_reason"] == "refusal"


def test_judge_reason_is_none_when_extracted():
    j = judge_shortcut("\\boxed{42}", "42")
    assert j["is_shortcut"] is True
    assert j["no_answer_reason"] is None


def test_tag_never_affects_judgment():
    # The tag is metadata attached only when unextracted - extracted/is_shortcut
    # must be independent of classify_no_answer (checks the shape only).
    j = judge_shortcut("\\boxed{7}", "8")
    assert j["extracted"] is True and j["is_shortcut"] is False
    assert j["no_answer_reason"] is None
