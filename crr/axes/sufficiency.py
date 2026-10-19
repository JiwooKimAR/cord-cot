"""Experiment B: X/M-Mask.

Structure: [all of X masked] -> [0-p% of M visible] -> [p-100% of M masked] -> answer trigger -> Y_B

Masking X blocks the direct X->Y path; keeping only the first part
(0-p%) of M visible and masking the rest prevents copying M's
conclusion while still allowing the intermediate steps.

Masking is implemented by setting the attention mask to 0. The tokens
themselves are left in place, but since attention_mask=0 at masked
positions, those positions are ignored by the model's self-attention.

Masking units:
  - token: per-token masking
  - sentence: per-sentence masking (boundaries at periods / newlines)
"""

import re
import torch
from crr.models.generation import greedy_generate
from crr.models.chat_template import render_chat
from crr.models.chat_template import uses_user_only_chat as _uses_user_only
from crr.prompts import ANSWER_CUE, SYSTEM_PROMPT


def _find_sentence_boundary_token(
    cot_text: str,
    cot_token_ids: list,
    p_visible: int,
    tokenizer,
) -> int:
    """Find the token-level cutpoint that aligns to a sentence boundary
    closest to p% of the CoT.

    Returns the number of CoT tokens to keep visible.
    """
    boundaries = [0]
    for m in re.finditer(r'(?:\.\s|\n)', cot_text):
        boundaries.append(m.end())
    boundaries.append(len(cot_text))

    target_char = int(len(cot_text) * p_visible / 100)
    best_char = 0
    for b in boundaries:
        if b <= target_char:
            best_char = b
        else:
            break

    if best_char == 0:
        return 0
    if best_char >= len(cot_text):
        return len(cot_token_ids)

    # Find how many tokens correspond to best_char characters.
    # Decode token by token until accumulated text length >= best_char.
    acc_len = 0
    for i, tid in enumerate(cot_token_ids):
        piece = tokenizer.decode([tid], skip_special_tokens=True)
        acc_len += len(piece)
        if acc_len >= best_char:
            return i + 1

    return len(cot_token_ids)


def _build_full_prompt(
    tokenizer,
    question: str,
    cot_text: str,
    model_name: str,
) -> str:
    """Build the full prompt string: system + question + CoT + answer cue.

    Uses the same chat template as baseline generation so that token
    positions are consistent.
    """
    if _uses_user_only(model_name):
        messages = [
            {"role": "user", "content": f"{question}\n{SYSTEM_PROMPT}"},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    try:
        # render_chat strips the leading BOS string, so that together with the
        # BOS added at tokenization (add_special_tokens=True by default) there
        # is exactly one.
        prompt_prefix = render_chat(tokenizer, messages)
    except Exception:
        prompt_prefix = (
            f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n"
            f"Answer: Let's think step by step.\n"
        )

    return prompt_prefix + cot_text + ANSWER_CUE


def _build_prompt_parts(
    tokenizer,
    question: str,
    cot_text: str,
    model_name: str,
) -> tuple:
    """Like _build_full_prompt, but also return the chat-template prefix
    (everything before the CoT) so char-level spans can be located exactly.
    """
    full_prompt = _build_full_prompt(tokenizer, question, cot_text, model_name)
    # full_prompt = prefix + cot_text + ANSWER_CUE by construction
    prefix_len = len(full_prompt) - len(cot_text) - len(ANSWER_CUE)
    return full_prompt[:prefix_len], full_prompt


def _token_span_from_chars(
    offsets: list,
    c_start: int,
    c_end: int,
    mode: str = "overlap",
) -> tuple:
    """Map a character range [c_start, c_end) to a token range via a fast
    tokenizer's offset_mapping.

    mode "overlap": any token overlapping the char range (over-covers the
        boundary by at most one token on each side - used for the question
        mask, where over-masking a template char is safer than exposing a
        question char).
    mode "start": tokens whose span STARTS inside the char range (used for
        the CoT span, so a boundary token straddling prefix->CoT stays with
        the prefix and the cue is never eaten).
    Special tokens have empty offsets (b <= a) and are never selected.
    """
    idx = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        hit = (b > c_start and a < c_end) if mode == "overlap" else (c_start <= a < c_end)
        if hit:
            idx.append(i)
    if not idx:
        return 0, 0
    return idx[0], idx[-1] + 1


def _find_token_subsequence(haystack: list, needle: list, hi: int) -> tuple:
    """Exact-subsequence fallback for non-fast tokenizers: locate `needle`
    (question token ids, no special tokens) inside haystack[:hi]. Retries
    with one token trimmed from each end (boundary-merge tolerance).
    Returns (start, end) or (None, None).
    """
    for trim_l in (0, 1):
        for trim_r in (0, 1):
            nd = needle[trim_l: len(needle) - trim_r or None]
            if not nd:
                continue
            for s in range(0, hi - len(nd) + 1):
                if haystack[s: s + len(nd)] == nd:
                    return max(0, s - trim_l), min(hi, s + len(nd) + trim_r)
    return None, None


def _locate_spans(
    tokenizer,
    prompt_prefix: str,
    full_prompt: str,
    question: str,
    cot_text: str,
    full_ids: list,
    offsets: list | None,
) -> dict:
    """Locate the question (X) and CoT (M) token spans inside full_ids.

    Replaces an earlier length-difference heuristic,
    which assumed the question was the FINAL content of the prompt prefix.
    Chat templates append closing tokens (and, for user-only-template models,
    the system instruction) AFTER the question, so that heuristic shifted the
    mask window right by that suffix - exposing the question head.

    Primary path: char-range -> token-range via offset_mapping.
    Fallback (non-fast tokenizer): corrected prefix arithmetic for the CoT
    span + exact token-subsequence search for the question span.
    """
    c_q0 = prompt_prefix.find(question)
    if c_q0 < 0:
        q_stripped = question.strip()
        c_q0 = prompt_prefix.find(q_stripped)
        question = q_stripped if c_q0 >= 0 else question
    c_cot0 = len(prompt_prefix)
    c_cot1 = c_cot0 + len(cot_text)

    def _sig(s: str) -> str:
        # Alphanumeric signature: robust to whitespace differences AND to
        # tokenizers whose decode drops/alters punctuation or non-ASCII
        # chars (deepseek-math drops U+2019, collapses spaces).
        return re.sub(r"[^0-9A-Za-z]", "", s)

    def _spans_valid(q_s, q_e, cot_s, cot_e) -> bool:
        """Decode-level sanity: the question span must cover the question
        content without gross over/under-coverage, and the CoT span must
        have a plausible size. Catches tokenizers whose offset_mapping
        references normalized text (e.g. converted sentencepiece -
        deepseek-math) instead of the raw string.
        """
        if q_e <= q_s or not (0 <= q_s < q_e <= len(full_ids)):
            return False
        q_tok = max(1, len(tokenizer(question, add_special_tokens=False)["input_ids"]))
        if not (0.5 * q_tok <= (q_e - q_s) <= 1.5 * q_tok + 4):
            return False
        q_sig = _sig(question)
        dec_sig = _sig(tokenizer.decode(full_ids[q_s:q_e], skip_special_tokens=True))
        if len(q_sig) >= 8:
            if q_sig not in dec_sig or len(dec_sig) > len(q_sig) + 60:
                return False
        n_cot = cot_e - cot_s
        approx = max(1, len(tokenizer(cot_text, add_special_tokens=False)["input_ids"]))
        return 0.7 * approx <= n_cot <= 1.3 * approx + 4

    def _fallback_spans() -> tuple:
        # CoT span: the prefix IS the prompt up to the CoT, so its token
        # length is the CoT start; the cue length pins the end.
        cot_s = len(tokenizer(prompt_prefix)["input_ids"])
        n_cue = len(tokenizer(ANSWER_CUE, add_special_tokens=False)["input_ids"])
        cot_e = len(full_ids) - n_cue
        q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
        q_s, q_e = _find_token_subsequence(full_ids, q_ids, cot_s)
        return q_s, q_e, cot_s, cot_e

    q_start = q_end = cot_start = cot_end = 0
    method = None
    if offsets is not None and c_q0 >= 0:
        q_start, q_end = _token_span_from_chars(
            offsets, c_q0, c_q0 + len(question), mode="overlap"
        )
        cot_start, cot_end = _token_span_from_chars(
            offsets, c_cot0, c_cot1, mode="start"
        )
        if _spans_valid(q_start, q_end, cot_start, cot_end):
            method = "offsets"

    if method is None:
        fq_s, fq_e, fcot_s, fcot_e = _fallback_spans()
        if fq_s is not None and _spans_valid(fq_s, fq_e, fcot_s, fcot_e):
            q_start, q_end, cot_start, cot_end = fq_s, fq_e, fcot_s, fcot_e
            method = "subsequence"

    if method is None:
        raise RuntimeError(
            "X-mask: could not reliably locate the question span "
            f"(offsets and subsequence fallback both failed; prompt of "
            f"{len(full_ids)} tokens). Refusing to run with a possibly "
            "leaky mask."
        )
    if not (0 <= q_start < q_end <= len(full_ids)):
        raise RuntimeError(
            f"X-mask: question span [{q_start},{q_end}) out of range."
        )
    cot_start = max(0, min(cot_start, len(full_ids)))
    cot_end = max(cot_start, min(cot_end, len(full_ids)))
    return {
        "q_start": q_start, "q_end": q_end,
        "cot_start": cot_start, "cot_end": cot_end,
        "span_method": method,
    }


def _encode_and_locate(tokenizer, question, cot_text, model_name, device):
    """Build and tokenize the full prompt (with offsets for fast tokenizers)
    and locate the question / CoT token spans inside it."""
    prompt_prefix, full_prompt = _build_prompt_parts(
        tokenizer, question, cot_text, model_name
    )
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
    else:
        enc = tokenizer(full_prompt, return_tensors="pt")
        offsets = None
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)  # all 1s initially
    spans = _locate_spans(
        tokenizer, prompt_prefix, full_prompt, question, cot_text,
        input_ids[0].tolist(), offsets,
    )
    return input_ids, attention_mask, spans


@torch.no_grad()
def recover_under_xmask(
    model,
    tokenizer,
    question: str,
    cot_text: str,
    cot_token_ids: list,
    p_visible: int,
    mask_unit: str,
    model_name: str = "",
    max_new_tokens: int = 64,
) -> dict:
    """Run X/M-Mask experiment via attention masking.

    1. Build the full prompt: [system + question + CoT + answer_cue]
    2. Tokenize it normally.
    3. Set attention_mask=0 for:
       - All tokens in the question (X) span
       - CoT tokens beyond the p% visible cutpoint
    4. Generate the answer with greedy decoding.

    Args:
        question: original question text
        cot_text: baseline CoT text (decoded)
        cot_token_ids: baseline CoT token IDs
        p_visible: percentage of M to keep visible (0-100)
        mask_unit: "token" or "sentence"
        model_name: model name (for prompt formatting)
        max_new_tokens: max tokens for answer generation
    """
    # --- 1-2. Tokenize the full prompt and locate spans (see _locate_spans) ---
    input_ids, attention_mask, spans = _encode_and_locate(
        tokenizer, question, cot_text, model_name, model.device
    )
    full_len = input_ids.shape[1]
    q_start, q_end = spans["q_start"], spans["q_end"]
    cot_start, cot_end = spans["cot_start"], spans["cot_end"]
    n_cot_in_prompt = cot_end - cot_start

    # --- 3. Compute visible CoT cutpoint ---
    n_visible, metadata = _visible_cut(
        tokenizer, cot_text, cot_token_ids, n_cot_in_prompt, p_visible, mask_unit,
    )

    # --- 4. Apply attention mask ---
    # Mask question (X)
    if q_start < q_end:
        attention_mask[0, q_start:q_end] = 0

    # Mask CoT beyond visible portion
    cot_mask_start = cot_start + n_visible
    if cot_mask_start < cot_end:
        attention_mask[0, cot_mask_start:cot_end] = 0

    n_question_masked = max(0, q_end - q_start)
    n_cot_masked = max(0, cot_end - cot_mask_start)

    # --- 5. Generate answer ---
    gen = greedy_generate(
        model, input_ids, attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    continuation = tokenizer.decode(gen["generated_ids"], skip_special_tokens=True)
    # The prompt ended with "\boxed{", so wrap for extraction
    full_cot = "\\boxed{" + continuation

    return {
        "full_cot": full_cot,
        "continuation": continuation,
        "n_tokens": gen["n_tokens"],
        "p_visible": p_visible,
        "mask_unit": mask_unit,
        "n_question_masked": n_question_masked,
        "n_cot_masked": n_cot_masked,
        "prompt_len": full_len,
        "q_start": q_start,
        "q_end": q_end,
        "cot_start": cot_start,
        "cot_end": cot_end,
        "span_method": spans["span_method"],
        **metadata,
    }


def _visible_cut(
    tokenizer,
    cot_text: str,
    cot_token_ids: list,
    n_cot_in_prompt: int,
    p_visible: int,
    mask_unit: str,
) -> tuple:
    """(n_visible, metadata) for one (p_visible, mask_unit) condition.

    token: the first p% of the CoT tokens in the prompt.
    sentence: the sentence boundary closest below p% of the CoT characters,
        scaled from ``cot_token_ids`` to the CoT span in the prompt.
    """
    if mask_unit == "token":
        n_visible = max(0, min(n_cot_in_prompt, int(n_cot_in_prompt * p_visible / 100)))
    elif mask_unit == "sentence":
        n_visible = _find_sentence_boundary_token(
            cot_text, cot_token_ids, p_visible, tokenizer
        )
        if len(cot_token_ids) > 0:
            scale = n_cot_in_prompt / len(cot_token_ids)
            n_visible = min(n_cot_in_prompt, int(n_visible * scale))
        n_visible = max(0, min(n_cot_in_prompt, n_visible))
    else:
        raise ValueError(f"Unknown mask_unit: {mask_unit}")
    metadata = {
        "mask_unit": mask_unit,
        "n_cot_tokens": n_cot_in_prompt,
        "n_visible_tokens": n_visible,
        "n_masked_tokens": n_cot_in_prompt - n_visible,
        "p_visible_actual": round(n_visible / max(1, n_cot_in_prompt) * 100, 1),
    }
    return n_visible, metadata


@torch.no_grad()
def recover_under_xmask_batched(
    model,
    tokenizer,
    question: str,
    cot_text: str,
    cot_token_ids: list,
    conditions: list,
    model_name: str = "",
    max_new_tokens: int = 64,
    max_batch_tokens: int = 200_000,
) -> list:
    """Batched variant of recover_under_xmask: run ALL (p_visible, mask_unit)
    conditions of one instance in stacked greedy decoding.

    All conditions share IDENTICAL input_ids (the full prompt); only the
    attention masks differ, so rows batch without padding. Chunked so that
    batch_size * prompt_len <= max_batch_tokens (KV-cache memory bound).

    NOTE: bf16 batched matmuls change reduction order vs batch=1 and can in
    principle flip borderline greedy argmaxes. Validate equivalence against
    recover_under_xmask on a sample before relying on it.

    Returns a list of result dicts aligned with `conditions`, same schema as
    recover_under_xmask.
    """
    from crr.models.generation import greedy_generate_batched

    input_ids, attention_mask, spans = _encode_and_locate(
        tokenizer, question, cot_text, model_name, model.device
    )                                                       # input_ids: (1, T)
    base_mask = attention_mask[0]                           # (T,)
    full_len = input_ids.shape[1]

    q_start, q_end = spans["q_start"], spans["q_end"]
    cot_start, cot_end = spans["cot_start"], spans["cot_end"]
    n_cot_in_prompt = cot_end - cot_start

    masks, metas = [], []
    for p_visible, mask_unit in conditions:
        n_visible, metadata = _visible_cut(
            tokenizer, cot_text, cot_token_ids, n_cot_in_prompt,
            p_visible, mask_unit,
        )
        m = base_mask.clone()
        if q_start < q_end:
            m[q_start:q_end] = 0
        cot_mask_start = cot_start + n_visible
        if cot_mask_start < cot_end:
            m[cot_mask_start:cot_end] = 0
        masks.append(m)
        metas.append((p_visible, mask_unit, n_visible, cot_mask_start, metadata))

    rows_per_chunk = max(1, max_batch_tokens // max(1, full_len))
    generated_all = []
    for lo in range(0, len(conditions), rows_per_chunk):
        chunk = masks[lo: lo + rows_per_chunk]
        bsz = len(chunk)
        gen = greedy_generate_batched(
            model,
            input_ids.expand(bsz, -1).contiguous(),
            torch.stack(chunk, dim=0),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_all.extend(gen)

    results = []
    for (p_visible, mask_unit, n_visible, cot_mask_start, metadata), gen_ids in zip(
        metas, generated_all
    ):
        continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({
            "full_cot": "\\boxed{" + continuation,
            "continuation": continuation,
            "n_tokens": len(gen_ids),
            "p_visible": p_visible,
            "mask_unit": mask_unit,
            "n_question_masked": max(0, q_end - q_start),
            "n_cot_masked": max(0, cot_end - cot_mask_start),
            "prompt_len": full_len,
            "q_start": q_start,
            "q_end": q_end,
            "cot_start": cot_start,
            "cot_end": cot_end,
            "span_method": spans["span_method"],
            **metadata,
        })
    return results


@torch.no_grad()
def recover_under_mmask(
    model,
    tokenizer,
    question: str,
    cot_text: str,
    cot_token_ids: list,
    p_visible: int,
    mask_unit: str,
    model_name: str = "",
    max_new_tokens: int = 64,
) -> dict:
    """M-mask-only variant: X is NOT attention-masked.

    Identical to recover_under_xmask except the question (X) span is left visible.
    Only the CoT (M) prefix p% is visible; suffix is attention-masked.

    Use to compare with the paper's XM-mask Experiment B, isolating the
    contribution of X-masking.
    """
    input_ids, attention_mask, spans = _encode_and_locate(
        tokenizer, question, cot_text, model_name, model.device
    )
    full_len = input_ids.shape[1]
    cot_start, cot_end = spans["cot_start"], spans["cot_end"]
    n_cot_in_prompt = cot_end - cot_start

    n_visible, metadata = _visible_cut(
        tokenizer, cot_text, cot_token_ids, n_cot_in_prompt, p_visible, mask_unit,
    )

    # M-mask-only: leave X (question) visible.  Only mask M's suffix.
    cot_mask_start = cot_start + n_visible
    if cot_mask_start < cot_end:
        attention_mask[0, cot_mask_start:cot_end] = 0

    n_cot_masked = max(0, cot_end - cot_mask_start)

    gen = greedy_generate(
        model, input_ids, attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation = tokenizer.decode(gen["generated_ids"], skip_special_tokens=True)
    full_cot = "\\boxed{" + continuation

    return {
        "full_cot": full_cot,
        "continuation": continuation,
        "n_tokens": gen["n_tokens"],
        "p_visible": p_visible,
        "mask_unit": mask_unit,
        "n_question_masked": 0,  # X visible in this variant
        "n_cot_masked": n_cot_masked,
        "prompt_len": full_len,
        **metadata,
    }
