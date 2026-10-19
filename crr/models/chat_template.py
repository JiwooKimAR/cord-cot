"""Chat-template policy helpers - single source of truth for model-family
quirks that affect how we render prompts.

Three policies are exposed:

  - `is_r1_distill(name)`: DeepSeek-R1-Distill family. Their chat_template
    auto-prepends `<think>\\n` after the assistant turn marker; adding a
    system role overrides the trained template and degrades reasoning.

  - `uses_appended_instruction(name)`: DeepSeek-Math family. Per the HF
    model card, the reasoning instruction goes inside the user turn
    rather than as a separate system role. Empirical A/B (20 MATH-500
    Level 4-5 problems) showed 3/20 vs 7/20 in favor of appended.

  - `uses_user_only_chat(name)`: union of the two above - the prompt
    should be rendered with no system role at all. Used by the baseline
    and B-axis (X/M-mask) prompts.
"""


def render_chat(tokenizer, messages, add_generation_prompt: bool = True) -> str:
    """Render with apply_chat_template(tokenize=False), then strip the leading BOS string.

    Fixes a double-BOS artifact: the rendered template string contains the
    BOS token text, and every downstream tokenization (HF default
    add_special_tokens=True, vLLM internal tokenization) prepends BOS once
    more, giving BOSx2 on 4/6 models. Stripping the BOS from the string here
    yields exactly one after tokenization. No-op for the Qwen family
    (bos_token=None).

    All prompt construction/measurement must go through this function.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    bos = getattr(tokenizer, "bos_token", None)
    if bos:
        while rendered.startswith(bos):
            rendered = rendered[len(bos):]
    return rendered


def is_r1_distill(model_name: str) -> bool:
    n = model_name.lower()
    return "r1-distill" in n or "deepseek-r1" in n


def uses_appended_instruction(model_name: str) -> bool:
    n = model_name.lower()
    return "deepseek" in n and "math" in n


def uses_user_only_chat(model_name: str) -> bool:
    """Union policy: model expects no `system` role in the chat template."""
    return is_r1_distill(model_name) or uses_appended_instruction(model_name)
