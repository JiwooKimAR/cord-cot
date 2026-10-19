"""Shortcut judgment: compare an intervention's answer with the baseline answer.

Common setting:
  - Shortcut if the post-intervention answer equals the baseline answer.
  - Not a shortcut if the answer changes.
  - Comparison is done on the extracted final numeric answer (categorical).

Extraction failure handling:
  If the model's output does not contain an extractable answer, we return
  is_shortcut=None (rather than False), tagged with a ``no_answer_reason``.
  When CORD is computed, ``crr.metric.grid.build_a_matrix`` /
  ``build_b_curve`` count such cells as 0 (answer changed), the same as a
  different answer.
"""

from crr.answers import extract_answer, check_answer, classify_no_answer


def judge_shortcut(intervened_text, baseline_answer, strict="semi"):
    """
    Decide whether an intervention left the baseline answer unchanged.

    By default uses SEMI-strict extraction: structured markers first, then a
    small set of permissive fallbacks (display math result, trailing
    answer-line, bare-number last line) guarded by a gibberish detector
    that rejects degenerate repetitive/symbolic outputs even when they
    contain a stray number.

    Pass strict=True to restrict extraction to structured markers only.

    Args:
        intervened_text: Full model output after intervention (CoT + answer).
        baseline_answer: The already-extracted baseline predicted answer.
        strict: True | "semi" | False (see extract_answer for semantics).

    Returns:
        dict with:
          predicted_answer: extracted answer from intervened_text (may be empty)
          extracted: True if an answer could be extracted
          answer_unchanged: True if extracted answer matches baseline
          is_shortcut: True (unchanged), False (changed), or None (no answer)
    """
    predicted = extract_answer(intervened_text, strict=strict)
    extracted = bool(predicted) and predicted.strip() != ""
    if not extracted:
        return {
            "predicted_answer": predicted,
            "extracted": False,
            "answer_unchanged": None,
            "is_shortcut": None,
            # No-answer reason tag (technical metadata - does not affect the
            # judgment). Used to report refusals (expected failure under
            # X-mask) separately from generation collapse (digit_loop, etc.).
            "no_answer_reason": classify_no_answer(intervened_text),
        }
    unchanged = check_answer(predicted, baseline_answer)
    return {
        "predicted_answer": predicted,
        "extracted": True,
        "answer_unchanged": unchanged,
        "is_shortcut": unchanged,
        "no_answer_reason": None,
    }
