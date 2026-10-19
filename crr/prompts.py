"""Prompt strings shared by baseline generation and both disruption axes.

Every measurement renders the question with ``SYSTEM_PROMPT`` (as a system turn,
or appended to the user turn for user-only chat templates; see
``crr.models.chat_template``) and elicits the answer after ``ANSWER_CUE``.
"""

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

ANSWER_CUE = "\n\nTherefore, the final answer is \\boxed{"
