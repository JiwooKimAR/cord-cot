"""Self-Refine (Madaan 2023) - 1 round refinement on math reasoning.

Used to test whether CORD predicts the self-refine change rate (Sec. 4.3).

**Madaan 2023 spec (NeurIPS 2023, arXiv:2303.17651)**:

1. Initial answer ``Y_0`` and baseline CoT ``M_0`` (= our baseline.jsonl).
2. **Critique**: the model itself reviews ``M_0`` + ``Y_0`` step by step.
3. **Refine**: given the critique, the answer is generated *again* -> ``Y_refined``.
4. Output: ``Y_0``, ``Y_refined`` and the change flag (``Y_refined != Y_0``).

**Reference impl**: [madaan/self-refine](https://github.com/madaan/self-refine)
(Apache-2.0). Their GSM prompt is PAL-style (Python code generation), which does
not match our natural-language CoT setup - *the prompts themselves are our own
natural-language templates*; only the structure (the init -> feedback -> refine
2-stage chain) and license attribution are borrowed.

**Backend-agnostic**: the caller injects ``llm_call(prompt) -> str`` (e.g. a vLLM
wrapper for R1-Distill / Qwen-Math self-refine). No auxiliary LLM is needed -
Madaan's spec is *self*-refine (critique + refine by the same model).
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------------

CRITIQUE_PROMPT = """Below is a math problem and a proposed solution.

Problem: {query}
Solution: {baseline_cot}
Answer: {y_0}

Review the solution step by step. Identify any errors in reasoning or calculation.
Critique:"""


# -----------------------------------------------------------------------------
# Core: 1-round refine
# -----------------------------------------------------------------------------


def make_critique_prompt(query: str, baseline_cot: str, y_0: str) -> str:
    """Fill the CRITIQUE_PROMPT placeholders."""
    return CRITIQUE_PROMPT.format(query=query, baseline_cot=baseline_cot, y_0=y_0)


# -----------------------------------------------------------------------------
# Confound experiments - trigger swap (A), M-ablation (D''), y_0 ablation (E)
# Length matching: 10-12 tokens measured with the Qwen tokenizer (real=11, +/-1).
# -----------------------------------------------------------------------------

REFINE_TRIGGER_VARIANTS = {
    "real": "Based on the critique below, provide a refined solution.",
    "strong_critique": "The solution has clear errors that must be fixed.",
    "neutral_review": "Please review the solution above and refine it if needed.",
    "weak_suggestion": "You may revise if needed; otherwise restate the answer.",
    "anti_trigger": "The solution is likely correct; restate unless certain.",
}

# Includes the y_0 line (default)
_REFINE_BODY = """\
Problem: {query}
Original solution: {baseline_cot}
Original answer: {y_0}
Critique: {critique}

Refined solution. After your reasoning, you MUST end with the final answer in this exact format on a separate line:
Final Answer: \\boxed{{ANSWER}}

where ANSWER is your final answer (number, expression, or multi-character text).

Refined solution:"""

# y_0 line removed (experiment E)
_REFINE_BODY_NO_Y0 = """\
Problem: {query}
Original solution: {baseline_cot}
Critique: {critique}

Refined solution. After your reasoning, you MUST end with the final answer in this exact format on a separate line:
Final Answer: \\boxed{{ANSWER}}

where ANSWER is your final answer (number, expression, or multi-character text).

Refined solution:"""


def make_refine_prompt(
    query: str,
    baseline_cot: str,
    y_0: str,
    critique: str,
    *,
    trigger_variant: str = "real",
    m_ablate: bool = False,
    y0_ablate: bool = False,
    critique_ablate: bool = False,
) -> str:
    """Build the refine prompt, plus ablations for the confound experiments.

    Args:
        trigger_variant: 'real' (original) | 'strong_critique' | 'neutral_review'
            | 'weak_suggestion' | 'anti_trigger'. For experiment A.
        m_ablate: if True, replace baseline_cot with "[omitted]" (experiment D'').
            The critique stage is unaffected - the caller uses the normal critique.
        y0_ablate: if True, remove the "Original answer: {y_0}" line (experiment E).
        critique_ablate: if True, replace the critique with "[omitted]" (experiment
            C-ablate). Symmetric to D'' - structure preserved, only critique content removed.

    With all keyword arguments at their defaults, this is the unablated refine
    prompt used for the main self-refine run.
    """
    if trigger_variant not in REFINE_TRIGGER_VARIANTS:
        raise ValueError(
            f"unknown trigger_variant: {trigger_variant}. "
            f"choices: {list(REFINE_TRIGGER_VARIANTS.keys())}"
        )
    trigger = REFINE_TRIGGER_VARIANTS[trigger_variant]
    cot_value = "[omitted]" if m_ablate else baseline_cot
    critique_value = "[omitted]" if critique_ablate else critique
    body_tmpl = _REFINE_BODY_NO_Y0 if y0_ablate else _REFINE_BODY

    return trigger + "\n\n" + body_tmpl.format(
        query=query, baseline_cot=cot_value, y_0=y_0, critique=critique_value,
    )


