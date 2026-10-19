"""Answer handling: extraction / checking / correctness, and answer occurrence in the CoT.

    from crr.answers import extract_answer, check_answer, evaluate_correctness
    from crr.answers.leak import first_answer_pos_frac
"""

from crr.answers.extract import (
    extract_answer,
    check_answer,
    normalize_number,
    normalize_answer_string,
    normalize_latex,
    latex_frac_to_float,
    extract_gsm8k_answer,
    extract_gold_answer,
    evaluate_correctness,
)
from crr.answers.no_answer import classify_no_answer
from crr.answers import leak

__all__ = [
    "classify_no_answer",
    "extract_answer",
    "check_answer",
    "normalize_number",
    "normalize_answer_string",
    "normalize_latex",
    "latex_frac_to_float",
    "extract_gsm8k_answer",
    "extract_gold_answer",
    "evaluate_correctness",
    "leak",
]
