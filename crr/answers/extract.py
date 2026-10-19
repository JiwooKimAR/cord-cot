"""Math answer extraction and normalization."""

import re


def extract_answer(text, strict=False):
    """
    Extract the final numerical answer from model output.

    Supports three modes for judging intervention outputs:

    - strict=True  : only structured markers count (\\boxed{}, ####,
                     "the answer is"). Everything else -> "".
    - strict="semi": structured markers first; if none, a small set of
                     permissive fallbacks (display math result, trailing
                     "answer:" line, bare-number last line). Gibberish
                     guard rejects outputs dominated by repeated tokens
                     or non-linguistic symbols even if a number appears.
    - strict=False : structured markers + last-number fallback, no guard.
                     Matches the legacy lenient behavior used for gold.
    """
    if not text:
        return ""

    # Structured markers (common to all modes).
    # boxed is scanned from back to front, taking the first occurrence whose
    # (a) braces are balanced and (b) content is non-empty. Trying only the
    # last occurrence would miss a complete earlier \boxed{answer} whenever the
    # answer-generation window cuts off mid repetition loop (last boxed
    # truncated/unclosed). (b) prevents the scan from stopping at an empty
    # "\boxed{}" quoted from the instruction (e.g. 'put your final answer
    # within \boxed{}'). If the last occurrence is complete and non-empty,
    # the result equals last-occurrence extraction.
    idx = text.rfind('\\boxed{')
    while idx != -1:
        inner = _extract_braced(text, idx + 6)
        if inner is not None:
            norm = normalize_answer_string(inner)
            if norm.strip():
                return norm
        idx = text.rfind('\\boxed{', 0, idx)

    hash_match = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hash_match:
        return normalize_number(hash_match[-1].strip())

    answer_match = re.findall(
        r'(?:the answer is|the final answer is|therefore,? the answer is)\s*[:\s]*([^\n.]+)',
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return normalize_number(answer_match[-1].strip())

    if strict is True:
        return ""

    if strict == "semi":
        if _looks_like_gibberish(text):
            return ""
        semi = _extract_semi_structured(text)
        if semi:
            return semi
        return ""

    # Legacy lenient: last number anywhere.
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return normalize_number(numbers[-1])

    return text.strip()


_ANSWER_LINE_RE = re.compile(
    r'(?:^|\n)\s*(?:answer|final answer|final|ans|result|so|thus|therefore|hence)\s*[:=]?\s*'
    r'([^\n]{1,80}?)\s*$',
    re.IGNORECASE,
)
_DISPLAY_MATH_RE = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
_BARE_NUM_RE = re.compile(r'^\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$')
_TRAILING_EQ_RE = re.compile(r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*[.\s]*$')


def _extract_semi_structured(text):
    """Permissive fallback after structured markers miss.

    Accepts the last display-math block, a trailing 'answer:' style line,
    a bare-number final line, or a '= 42' at the very end of the text.
    Never scans for an arbitrary number mid-stream.
    """
    disp = _DISPLAY_MATH_RE.findall(text)
    if disp:
        inside = disp[-1].strip()
        num = normalize_number(inside)
        if num and num != inside:
            return num
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', inside)
        if nums:
            return normalize_number(nums[-1])

    m = _TRAILING_EQ_RE.search(text.rstrip())
    if m:
        return normalize_number(m.group(1))

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        bare = _BARE_NUM_RE.match(last)
        if bare:
            return normalize_number(bare.group(1))
        for m in _ANSWER_LINE_RE.finditer(text):
            cand = m.group(1).strip().rstrip('.').rstrip(',')
            nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', cand)
            if nums:
                return normalize_number(nums[-1])

    return ""


def _looks_like_gibberish(text):
    """Return True if the output is dominated by repeated tokens or symbols.

    Triggers when an intervention has collapsed generation into repetitive
    noise (e.g. ' Leone$ Leone$ Leone$ ...'). Even if a stray number still
    appears, we do not want to credit the model with having 'produced an
    answer'.
    """
    s = text.strip()
    if len(s) < 40:
        return False

    # Repetition: count frequency of 4-word shingles. If the top shingle
    # covers more than half of all shingles, the text is degenerate.
    words = re.findall(r'\S+', s)
    if len(words) >= 20:
        shingles = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if shingles:
            top = max(set(shingles), key=shingles.count)
            if shingles.count(top) / len(shingles) > 0.5:
                return True

    # Letter share: if less than 30% of non-space characters are ASCII
    # letters, the output is mostly symbols/punctuation, not prose.
    non_space = [c for c in s if not c.isspace()]
    if non_space:
        letters = sum(1 for c in non_space if c.isalpha())
        if letters / len(non_space) < 0.30:
            return True

    return False


def normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    s = s.strip().rstrip('.')
    # Remove commas
    s = s.replace(',', '')
    # Remove dollar signs, percent signs
    s = s.replace('$', '').replace('%', '')
    # Try to convert to float then back to clean string
    try:
        val = float(s)
        if not (val == float('inf') or val == float('-inf') or val != val):
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s.strip()


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract answer from GSM8K ground truth format (#### number)."""
    match = re.search(r'####\s*(.+?)$', answer_text, re.MULTILINE)
    if match:
        return normalize_number(match.group(1).strip())
    return normalize_number(answer_text.strip())


def extract_gold_answer(answer_text: str, dataset_type: str = "gsm8k") -> str:
    """Extract gold answer based on dataset type."""
    if dataset_type == "gsm8k":
        return extract_gsm8k_answer(answer_text)
    elif dataset_type == "math500":
        # MATH-500 answer field is already the clean answer (may contain LaTeX).
        return normalize_answer_string(answer_text)
    else:
        return normalize_number(answer_text.strip())


def _extract_braced(s: str, start: int) -> str | None:
    """Extract content inside balanced braces starting at s[start] == '{'."""
    if start >= len(s) or s[start] != '{':
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start + 1:i]
    return None


def normalize_latex(s: str) -> str:
    """Normalize LaTeX expressions for robust comparison."""
    s = s.strip()
    # Collapse double (or more) backslashes to single FIRST: \\text -> \text
    while '\\\\' in s:
        s = s.replace('\\\\', '\\')
    # Remove \boxed{...} wrapper (supports nested braces)
    idx = s.rfind('\\boxed{')
    if idx != -1:
        inner = _extract_braced(s, idx + 6)
        if inner is not None:
            s = inner
    # Remove \left and \right
    s = re.sub(r'\\left\s*', '', s)
    s = re.sub(r'\\right\s*', '', s)
    # \dfrac -> \frac, \tfrac -> \frac
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Remove \text{...} wrappers (keep inner text for unit stripping later)
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # Remove LaTeX thin/negative spaces: \! \, \; \: \quad \qquad
    s = re.sub(r'\\[!,;:]', '', s)
    s = re.sub(r'\\q?quad', '', s)
    # Remove stray backslash before digits: \12 -> 12, \66 -> 66
    s = re.sub(r'\\(\d)', r'\1', s)
    # Unicode square-root sign (U+221A) -> \sqrt for uniform handling
    s = re.sub(r'√(\d+)', r'\\sqrt{\1}', s)
    s = s.replace('√', '\\sqrt')
    # Remove degree: ^\circ, ^{\circ}, degree sign (U+00B0)
    s = re.sub(r'\^?\{?\\circ\}?', '', s)
    s = re.sub(r'\\?°', '', s)
    # Remove variable prefix: "x = 5" -> "5", "x=" -> ""
    s = re.sub(r'^[a-zA-Z]\s*[=∈]\s*', '', s)
    # Remove common unit words (after \text removal)
    s = re.sub(r'(hours?|minutes?|seconds?|liters?|meters?|cm|kg|km)\b', '', s, flags=re.IGNORECASE)
    # Normalize whitespace
    s = re.sub(r'\s+', '', s).strip()
    return s


def latex_frac_to_float(s: str) -> float | None:
    """Try to evaluate \\frac{a}{b} (possibly with sign/sqrt) to a float."""
    s = normalize_latex(s)
    # \frac with nested brace support
    sign = 1
    rest = s
    if rest.startswith('-'):
        sign = -1
        rest = rest[1:]
    if rest.startswith('\\frac{'):
        num_s = _extract_braced(rest, 5)
        if num_s is not None:
            den_start = 5 + len(num_s) + 2  # skip {num_s}}
            if den_start < len(rest) and rest[den_start] == '{':
                den_s = _extract_braced(rest, den_start)
                if den_s is not None:
                    try:
                        num_val = _eval_simple_expr(num_s)
                        den_val = _eval_simple_expr(den_s)
                        if den_val and den_val != 0:
                            return sign * num_val / den_val
                    except (ValueError, ZeroDivisionError):
                        pass
    # Plain text fraction: "a/b" (a and b may be simple expressions like 3\sqrt{3})
    slash_idx = s.find('/')
    if slash_idx > 0 and slash_idx < len(s) - 1:
        num_part = s[:slash_idx].strip()
        den_part = s[slash_idx + 1:].strip()
        try:
            num_val = _eval_simple_expr(num_part)
            den_val = _eval_simple_expr(den_part)
            if den_val and den_val != 0:
                return num_val / den_val
        except (ValueError, ZeroDivisionError):
            pass
    return None


def _eval_simple_expr(s: str) -> float:
    """Evaluate simple math: integers, sqrt, products like 3\\sqrt{3}."""
    s = s.strip()
    # Pure number
    try:
        return float(s)
    except ValueError:
        pass
    # \sqrt{n}
    m = re.fullmatch(r'\\sqrt\{(\d+(?:\.\d+)?)\}', s)
    if m:
        import math
        return math.sqrt(float(m.group(1)))
    # coefficient * \sqrt{n}: "3\\sqrt{3}" or "3\sqrt{3}"
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\\?sqrt\{(\d+(?:\.\d+)?)\}', s)
    if m:
        import math
        return float(m.group(1)) * math.sqrt(float(m.group(2)))
    raise ValueError(f"Cannot evaluate: {s}")


def normalize_answer_string(s: str) -> str:
    """Normalize a math answer string that may contain LaTeX."""
    s = normalize_latex(s)
    # Try to extract a simple number first
    num = normalize_number(s)
    if num != s:
        return num
    # Try to evaluate fraction to a float
    val = latex_frac_to_float(s)
    if val is not None:
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.10g}"
    return s


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    # 1. Direct number comparison
    pred = normalize_number(predicted)
    gold_norm = normalize_number(gold)
    if pred == gold_norm:
        return True

    # 2. Full LaTeX normalization comparison
    pred_latex = normalize_latex(predicted)
    gold_latex = normalize_latex(gold)
    if pred_latex == gold_latex:
        return True

    # 3. Normalize both as answer strings (evaluates fractions etc.)
    pred_ans = normalize_answer_string(predicted)
    gold_ans = normalize_answer_string(gold)
    if pred_ans == gold_ans:
        return True

    # 4. Numeric comparison via fraction evaluation
    pred_val = latex_frac_to_float(predicted)
    gold_val = latex_frac_to_float(gold)
    if pred_val is not None and gold_val is not None:
        if abs(pred_val - gold_val) < 1e-6:
            return True

    # 5. Try normalizing predicted as number, gold as fraction (or vice versa)
    try:
        pf = float(pred)
        if gold_val is not None and abs(pf - gold_val) < 1e-6:
            return True
    except ValueError:
        pass
    try:
        gf = float(gold_norm)
        if pred_val is not None and abs(pred_val - gf) < 1e-6:
            return True
    except ValueError:
        pass

    return False


# --- correctness ---
def evaluate_correctness(model_output: str, gold_answer: str) -> dict:
    """
    Evaluate whether the model's output contains the correct answer.

    Args:
        model_output: Full model-generated text (CoT + answer)
        gold_answer: Ground truth answer (already extracted/normalized by baseline)

    Returns:
        dict with predicted answer, gold answer, and correctness
    """
    predicted = extract_answer(model_output)
    gold = normalize_answer_string(gold_answer) if gold_answer else gold_answer
    correct = check_answer(predicted, gold)

    return {
        "predicted_answer": predicted,
        "gold_answer": gold,
        "correct": correct,
    }
