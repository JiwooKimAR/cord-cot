"""Greedy decoding.

Three paths:
  - greedy_generate(): token-by-token HF transformers loop; accepts an arbitrary
    attention mask (used by the B-axis X/M-mask).
  - greedy_generate_batched(): the same loop over a batch of rows.
  - vllm_batch_generate(): vLLM offline batch inference (baseline CoT and
    A-axis answers).
"""

import torch
from typing import Optional, List


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    max_new_tokens: int = 512,
    eos_token_id: Optional[int] = None,
) -> dict:
    """Greedy decoding for one sequence (batch size 1) under an arbitrary
    attention mask.

    Returns:
        dict with generated_ids (list[int], EOS excluded) and n_tokens.
    """
    device = input_ids.device
    generated_ids = []

    current_ids = input_ids
    current_mask = attention_mask
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=current_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]  # (batch=1, vocab)
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(logits, dim=-1)  # (batch=1,)

        # EOS is not included in generated_ids - consistent with the batched
        # path (greedy_generate_batched). The answer text is decoded with
        # skip_special_tokens either way.
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        generated_ids.append(next_token.item())

        current_ids = next_token.unsqueeze(0)
        current_mask = torch.cat(
            [current_mask, torch.ones((1, 1), device=device, dtype=current_mask.dtype)],
            dim=1,
        )

    return {
        "generated_ids": generated_ids,
        "n_tokens": len(generated_ids),
    }


@torch.no_grad()
def greedy_generate_batched(
    model,
    input_ids: torch.LongTensor,        # (B, T_in) - last input tokens per row
    attention_mask: torch.LongTensor,   # (B, T)    - full running mask
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> list:
    """Batched greedy decoding with per-row EOS handling.

    Each row continues until it emits eos_token_id (or max_new_tokens is
    reached). Finished rows are "parked" - their next-step input becomes
    pad_token_id and their emitted tokens are dropped from the return list.

    Returns: list of length B; each element is a list[int] of generated ids
    (excluding the final eos_token_id, matching greedy_generate semantics).
    """
    assert input_ids.dim() == 2
    B = input_ids.shape[0]
    device = input_ids.device
    pad = pad_token_id if pad_token_id is not None else (eos_token_id if eos_token_id is not None else 0)

    generated = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    current_ids = input_ids
    current_mask = attention_mask
    past_key_values = None

    # logits_to_keep=1: compute lm_head only for the last position. Without
    # it, a batched long-prompt prefill materialises (B, T, vocab) logits
    # (tens of GB); the kept logits are bit-identical (row-wise matmul).
    _ltk_kwargs = {"logits_to_keep": 1}
    for _ in range(max_new_tokens):
        try:
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **_ltk_kwargs,
            )
        except TypeError:
            _ltk_kwargs = {}
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)  # (B,)

        # For finished rows, override with pad so KV growth is cheap and
        # the parked row stays quiescent.
        if finished.any():
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad),
                next_tokens,
            )

        tokens_cpu = next_tokens.tolist()
        for i, tok in enumerate(tokens_cpu):
            if finished[i]:
                continue
            if eos_token_id is not None and tok == eos_token_id:
                finished[i] = True
                continue
            generated[i].append(tok)

        if finished.all():
            break

        current_ids = next_tokens.unsqueeze(1)  # (B, 1)
        current_mask = torch.cat(
            [current_mask, torch.ones((B, 1), device=device, dtype=current_mask.dtype)],
            dim=1,
        )

    return generated


def vllm_batch_generate(
    engine,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> List[dict]:
    """
    Batch greedy generation using vLLM.

    Args:
        engine: vllm.LLM instance
        tokenizer: HF tokenizer (kept for call-site symmetry; unused)
        prompts: list of formatted prompt strings
        max_new_tokens: max tokens to generate per prompt
        temperature: 0.0 for greedy

    Returns:
        list of dicts, each with: generated_ids (as returned by vLLM, including
        the stop token when generation stops on EOS), n_tokens, generated_text.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    outputs = engine.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        completion = output.outputs[0]
        generated_ids = list(completion.token_ids)
        results.append({
            "generated_ids": generated_ids,
            "n_tokens": len(generated_ids),
            "generated_text": completion.text,
        })

    return results
